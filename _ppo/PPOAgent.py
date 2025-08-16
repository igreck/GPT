import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import List, Optional, Dict, Tuple
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
from collections import deque
from transformers import StoppingCriteria, StoppingCriteriaList

from .reward_fn_math_reasoning import RewardFunctions
from .StopingConditions import _StopOnSubsequence



class PPOAgent:
    """PPO token-level pentru LLM cu:
       - Forward pe [prompt + completare]
       - labels = -100 pe PAD și pe tokenii de prompt (pierdere doar pe completare)
       - Entropie/KL mediate DOAR pe completare via comp_mask
       - AMP, micro-batching, gradient accumulation
       - Off-policy replay (buffer) cu păstrarea logprob-urilor vechi pe token
    """

    def __init__(
        self,
        policy_model,
        policy_ref,
        reward_model,
        reward_tokenizer,
        optimizer,
        tokenizer,
        config,
        scheduler
    ):
        self.cfg = config
        self.device = config.device
        self.scheduler = scheduler

        # Clip epsilon schedule (linear)
        self.initial_clip_epsilon = getattr(self.cfg, "clip_epsilon", 0.2)
        self.final_clip_epsilon   = getattr(self.cfg, "clip_epsilon_final", self.initial_clip_epsilon)

        # Reproducibility
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        os.environ["PYTHONHASHSEED"] = str(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.cfg.seed)

        # CuDNN perf
        torch.backends.cudnn.deterministic = False
        cudnn.benchmark = True

        # AMP
        self.amp_enabled = torch.cuda.is_available()
        self.autocast_dtype = torch.bfloat16 if (self.amp_enabled and torch.cuda.is_bf16_supported()) else None
        if self.amp_enabled:
            torch.set_float32_matmul_precision("high")

        # Replay buffer pentru off-policy
        self.replay_buffer = deque(maxlen=self.cfg.buffer_size)

        # Modele
        self.policy_model = policy_model.to(self.device).train()
        self.policy_ref   = policy_ref.to(self.device).eval()
        self.reward_model = reward_model.to(self.device).eval()

        # Reward-model positive/negative indices
        labels_map = getattr(self.reward_model.config, "id2label", {0: "NEGATIVE", 1: "POSITIVE"})
        inv = {v.lower(): k for k, v in labels_map.items()}
        self.pos_idx = inv.get("positive", 1)
        self.neg_idx = inv.get("negative", 0 if self.pos_idx != 0 else 1)

        # AMP scaler & TB
        self.scaler = GradScaler(enabled=self.amp_enabled)
        self.writer = SummaryWriter(log_dir=self.cfg.log_dir)
        self.best_reward = -float("inf")
        self.no_improve_epochs = 0
        self._kl_running = []  # pentru adaptive KL (forward KL token-wise)

        # Gradient checkpointing dacă e disponibil
        if hasattr(self.policy_model, "model") and hasattr(self.policy_model.model, "gradient_checkpointing_enable"):
            self.policy_model.model.gradient_checkpointing_enable()

        # Tokenizer: pad + left padding
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.reward_tokenizer = reward_tokenizer

        # Reward functions custom (keyword/ROUGE/regex etc.)
        self.rfn = RewardFunctions(self.tokenizer, self.device)

    # ===================== Utils =====================

    @staticmethod
    def _active_mask_from_labels(labels: torch.Tensor) -> torch.Tensor:
        """1 pe tokenii activi (labels != -100), 0 altfel."""
        return (labels != -100).float()

    # === Înlocuiește complet compute_logprobs_tokenwise cu asta ===
    def compute_logprobs_tokenwise(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Log-prob pe token (shape [B, T-1]) cu -inf pe pozițiile inactive (labels==-100).
        Calculat din log_softmax + gather (evită cross_entropy -> dtype float32).
        Rulează în float32 pt. stabilitate sub AMP/bfloat16.
        """
        # Shift pentru LM cauzal: logits[:, :-1] -> labels[:, 1:]
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]

        # Forțează float32 pentru numerica stabilă; NU schimbăm gradul (merge înainte)
        logp = F.log_softmax(shift_logits.float(), dim=-1)         # [B, T-1, V]

        # Pregătim out cu -inf (float32)
        B, Tm1, _ = logp.shape
        out = torch.full((B, Tm1), float("-inf"), device=logp.device, dtype=logp.dtype)

        # Poziții active sunt cele unde avem target (nu -100)
        active = shift_labels.ne(-100)
        if active.any():
            # Pentru pozițiile inactive punem o etichetă dummy (0) ca să nu stricăm gather
            safe_labels = shift_labels.masked_fill(~active, 0).long()
            gathered = logp.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
            out[active] = gathered[active]  # dtypes identice (float32)
        return out  # float32

    def compute_entropy(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Entropie mediată PE COMP_MASK (1 = completare, 0 = prompt/PAD)."""
        log_probs = F.log_softmax(logits, dim=-1)   # [B, T, V]
        probs = log_probs.exp()
        ent = -(probs * log_probs).sum(dim=-1)      # [B, T]
        denom = mask.sum(dim=1).clamp_min(1)
        return (ent * mask).sum(dim=1) / denom      # [B]

    def forward_token_kl(self, logits_new: torch.Tensor, logits_ref: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward KL(π_new || π_ref) pe token, mediat pe comp_mask. Returnează [B]."""
        logp_new = F.log_softmax(logits_new, dim=-1)     # [B, T, V]
        logp_ref = F.log_softmax(logits_ref, dim=-1)     # [B, T, V]
        p_new    = logp_new.exp()
        kl_tok   = (p_new * (logp_new - logp_ref)).sum(dim=-1)  # [B, T]
        denom    = mask.sum(dim=1).clamp_min(1)
        return (kl_tok * mask).sum(dim=1) / denom

    # INFO: Neutilizat (un GAE - Artificial pe batch) — păstrat pentru referință
    def compute_advantages(self, rewards, values, next_values, dones):
        advantages = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.cfg.gamma * next_values[t] * mask - values[t]
            gae = delta + self.cfg.gamma * self.cfg.lambda_gae * mask * gae
            advantages[t] = gae
        return advantages

    def value_loss_elementwise_clipped(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        old_values: torch.Tensor,
    ) -> torch.Tensor:
        """Implementare PPO corectă: max element-wise, apoi media."""
        clip = self.cfg.value_clip_range
        diff = (values - rewards)
        diff_clip = (old_values + (values - old_values).clamp(-clip, clip)) - rewards
        per_elem = torch.max(diff.pow(2), diff_clip.pow(2))
        return per_elem.mean()

    def compute_sentiment_reward(
        self,
        texts: List[str],
        use_margin: bool = False,
        add_length_bonus: bool = True,
        min_words: int = 6,
        bonus: float = 0.05
    ):
        """Reward de sentiment + bonus de lungime + bonus de diversitate (corectat)."""
        with torch.inference_mode():
            inputs = self.reward_tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.cfg.reward_max_length,
            )
            inputs.pop("token_type_ids", None)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            logits = self.reward_model(**inputs).logits  # [B, C]
            if use_margin and logits.size(-1) >= 2:
                margin = logits[:, self.pos_idx] - logits[:, self.neg_idx]
                raw = torch.sigmoid(margin)            # (0,1)
            else:
                probs = F.softmax(logits, dim=-1)
                raw = probs[:, self.pos_idx]           # p(positive)

            if add_length_bonus:
                lens = torch.tensor([len(t.split()) for t in texts], device=self.device, dtype=torch.float32)
                len_bonus = (lens >= min_words).float() * bonus
                raw = (raw + len_bonus).clamp(0.0, 1.0)

            # Bonus de diversitate (proporția de bigrame unice)
            def diversity_score(text: str) -> float:
                toks = text.split()
                if len(toks) < 2:
                    return 1.0  # nimic de repetat
                bigrams = [(toks[i], toks[i+1]) for i in range(len(toks)-1)]
                uniq = len(set(bigrams))
                total = max(1, len(bigrams))
                return float(uniq) / float(total)

            divers = torch.tensor([diversity_score(t) for t in texts], device=self.device)
            raw = (raw + divers * self.cfg.diversity_coef).clamp(0.0, 1.0)

            # scale la [-1,1]
            raw = raw * 2.0 - 1.0
            raw = raw.clamp(-1.0, 1.0)
        return raw  # [B]

    # ============== Prompt + Completion concat ==============

    def _concat_prompt_and_completion(self, batch_inputs: Dict[str, torch.Tensor], completions: List[str]) -> Dict[str, torch.Tensor]:
        B = batch_inputs["input_ids"].size(0)
        max_len = self.cfg.policy_max_length
        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id

        comp_tok = self.tokenizer(completions, add_special_tokens=False, return_tensors=None)["input_ids"]

        out_input_ids, out_attn, out_labels, out_comp_mask = [], [], [], []
        for i in range(B):
            am = batch_inputs["attention_mask"][i]
            plen = int(am.sum().item())
            prompt_ids = batch_inputs["input_ids"][i, -plen:].to(self.device)

            cids = comp_tok[i]
            if eos_id is not None and (len(cids) == 0 or cids[-1] != eos_id):
                cids = cids + [eos_id]
            cids_t = torch.tensor(cids, dtype=torch.long, device=self.device)

            merged = torch.cat([prompt_ids, cids_t], dim=0)

            labels_i = merged.clone()
            labels_i[:plen] = -100

            comp_mask_i = torch.zeros_like(merged, dtype=torch.long, device=self.device)
            comp_mask_i[plen:] = 1

            if merged.size(0) > max_len:
                start = merged.size(0) - max_len
                merged = merged[start:]
                labels_i = labels_i[start:]
                comp_mask_i = comp_mask_i[start:]

            pad_needed = max_len - merged.size(0)
            if pad_needed > 0:
                pad_ids = torch.full((pad_needed,), pad_id, device=self.device, dtype=torch.long)
                merged = torch.cat([pad_ids, merged], dim=0)
                labels_i = torch.cat([torch.full((pad_needed,), -100, device=self.device, dtype=torch.long), labels_i], dim=0)
                comp_mask_i = torch.cat([torch.zeros((pad_needed,), device=self.device, dtype=torch.long), comp_mask_i], dim=0)

            # attention_mask by position
            attn_i = torch.zeros_like(merged, dtype=torch.long, device=self.device)
            attn_i[pad_needed:] = 1

            out_input_ids.append(merged)
            out_labels.append(labels_i)
            out_attn.append(attn_i)
            out_comp_mask.append(comp_mask_i)

        out = {
            "input_ids": torch.stack(out_input_ids, dim=0),
            "attention_mask": torch.stack(out_attn, dim=0),
            "labels": torch.stack(out_labels, dim=0),
            "comp_mask": torch.stack(out_comp_mask, dim=0).float(),
        }

        # Sanity checks
        assert (out["comp_mask"] <= out["attention_mask"]).all(), "comp_mask trebuie să fie subset din attention_mask."
        assert (out["labels"] == -100).sum() >= (out["attention_mask"].sum() - out["comp_mask"].sum()), \
            "Promptul nu este complet mascat în labels."

        return out


    # ===================== Generare =====================

    def generate_batch_with_stopping(self, batch_inputs: Dict[str, torch.Tensor], max_new_tokens: Optional[int] = None, stop_text: str = "</SOLUTION>") -> List[str]:
        """Batch generation with a stopping criterion that only triggers on the completion part
        (ignores occurrences inside the prompt). Generation stops when **all** samples in the batch
        have ended with the stop subsequence; afterwards we also post-trim per sample.
        """
        with torch.inference_mode():
            max_new_tokens = max_new_tokens or self.cfg.max_new_tokens
            rep_pen = getattr(self.cfg, "repetition_penalty", 1.0)

            prompt_lens = batch_inputs["attention_mask"].sum(dim=1).tolist()
            stop_ids = self.tokenizer.encode(stop_text, add_special_tokens=False)
            stopping = None
            if len(stop_ids) > 0:
                stopping = StoppingCriteriaList([_StopOnSubsequence(stop_ids, prompt_lens)])

            amp = self.amp_enabled
            with autocast(
                device_type="cuda" if amp else "cpu",
                dtype=self.autocast_dtype,
                enabled=amp
            ):
                gen = self.policy_model.generate(
                    input_ids=batch_inputs["input_ids"],
                    attention_mask=batch_inputs["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    top_p=self.cfg.top_p,
                    temperature=self.cfg.temperature,
                    repetition_penalty=rep_pen,
                    pad_token_id=self.tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    stopping_criteria=stopping,
                )

            input_len = batch_inputs["input_ids"].size(1)
            gen_only_ids = gen.sequences[:, input_len:]
            texts = self.tokenizer.batch_decode(gen_only_ids, skip_special_tokens=True)
            # post-trim per sample at first occurrence of stop_text
            cut = []
            for t in texts:
                pos = t.find(stop_text)
                cut.append(t if pos == -1 else t[:pos + len(stop_text)])
            return cut

    def generate_loop_per_sample(self, batch_inputs: Dict[str, torch.Tensor], max_new_tokens: Optional[int] = None, stop_text: str = "</SOLUTION>") -> List[str]:
        """Per-sample token-by-token generation (loop). Independent stopping for each sample.
        Less efficient (calls generate with max_new_tokens=1 repeatedly), but robust and simple.
        """
        with torch.inference_mode():
            max_new_tokens = max_new_tokens or self.cfg.max_new_tokens
            rep_pen = getattr(self.cfg, "repetition_penalty", 1.0)
            inputs_ids = batch_inputs["input_ids"]
            inputs_attn = batch_inputs["attention_mask"]
            B = inputs_ids.size(0)
            stop_ids = self.tokenizer.encode(stop_text, add_special_tokens=False)
            stop_len = len(stop_ids)

            completions: List[str] = []
            amp = self.amp_enabled

            for i in range(B):
                cur_ids = inputs_ids[i:i+1].clone()
                cur_attn = inputs_attn[i:i+1].clone()
                prompt_len = int(cur_attn.sum().item())
                gen_tokens: List[int] = []

                for _ in range(max_new_tokens):
                    with autocast(
                        device_type="cuda" if amp else "cpu",
                        dtype=self.autocast_dtype,
                        enabled=amp,
                    ):
                        out = self.policy_model.generate(
                            input_ids=cur_ids,
                            attention_mask=cur_attn,
                            max_new_tokens=1,
                            do_sample=True,
                            top_p=self.cfg.top_p,
                            temperature=self.cfg.temperature,
                            repetition_penalty=rep_pen,
                            pad_token_id=self.tokenizer.pad_token_id,
                            return_dict_in_generate=True,
                        )
                    # append the newly generated token(s)
                    new_ids = out.sequences[:, cur_ids.size(1):]
                    if new_ids.numel() == 0:
                        break
                    nid = new_ids[0].tolist()
                    gen_tokens.extend(nid)
                    # update current ids/attn for next step
                    cur_ids = out.sequences
                    cur_attn = torch.ones_like(cur_ids, dtype=inputs_attn.dtype, device=cur_ids.device)

                    # check stop on the completion tail only
                    if stop_len > 0 and len(gen_tokens) >= stop_len:
                        if gen_tokens[-stop_len:] == stop_ids:
                            break

                text = self.tokenizer.decode(torch.tensor(gen_tokens, device=cur_ids.device), skip_special_tokens=True)
                # extra safety trim on text
                pos = text.find(stop_text)
                if pos != -1:
                    text = text[:pos + len(stop_text)]
                completions.append(text)

            return completions

    def generate(self, batch_inputs: Dict[str, torch.Tensor], max_new_tokens: Optional[int] = None) -> List[str]:
        with torch.inference_mode():
            max_new_tokens = max_new_tokens or self.cfg.max_new_tokens
            rep_pen = getattr(self.cfg, "repetition_penalty", 1.0)
            with autocast(
                device_type="cuda" if self.amp_enabled else "cpu",
                dtype=self.autocast_dtype,
                enabled=self.amp_enabled
            ):
                gen = self.policy_model.generate(
                    input_ids=batch_inputs["input_ids"],
                    attention_mask=batch_inputs["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    top_p=self.cfg.top_p,
                    temperature=self.cfg.temperature,
                    repetition_penalty=rep_pen,
                    pad_token_id=self.tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                )
            input_len = batch_inputs["input_ids"].size(1)
            gen_only_ids = gen.sequences[:, input_len:]
            completions = self.tokenizer.batch_decode(gen_only_ids, skip_special_tokens=True)
        return completions

    # ===================== PPO Loss (token-level) =====================

    def ppo_loss(
        self,
        old_logprobs_tok: torch.Tensor,  # [B, T-1] logπ_old pe token (−inf pe poziții inactive)
        logits_new: torch.Tensor,         # [B, T, V]
        ref_logits: torch.Tensor,         # [B, T, V] (no-grad)
        rewards: torch.Tensor,            # [B]
        values: torch.Tensor,             # [B]
        old_values: torch.Tensor,         # [B]
        labels: torch.Tensor,             # [B, T]
        comp_mask: torch.Tensor           # [B, T] (1 pe completare)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Avantaj pe secvență + whitening (numerically safe)
        rewards = torch.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)
        old_values = torch.nan_to_num(old_values, nan=0.0, posinf=0.0, neginf=0.0)
        advantages = (rewards - old_values.detach())
        advantages = torch.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)
        adv_mean = advantages.mean()
        adv_std = advantages.std(unbiased=False)
        advantages = (advantages - adv_mean) / (adv_std + 1e-8)
        advantages = torch.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)

        # Împărțim avantajul pe tokenii completării folosind labels mask shift
        active_tok = (labels[:, 1:] != -100).float()  # [B, T-1]
        tok_count = active_tok.sum(dim=1, keepdim=True).clamp_min(1.0)  # [B,1]
        adv_tok = (advantages[:, None] * active_tok) / tok_count        # [B, T-1]

        # Log-prob nou pe token (doar pe pozițiile active)
        new_logprobs_tok = self.compute_logprobs_tokenwise(logits_new, labels)  # [B, T-1] (float32)

        # Aliniază dtype și sanitizează pozițiile inactive ca să eviți NaN din (-inf) - (-inf)
        new_logprobs_tok = new_logprobs_tok.to(dtype=torch.float32)
        old_logprobs_tok = old_logprobs_tok.to(dtype=torch.float32)
        active_bool = active_tok.bool()
        new_logprobs_tok = torch.where(active_bool, new_logprobs_tok, torch.zeros_like(new_logprobs_tok))
        old_logprobs_tok = torch.where(active_bool, old_logprobs_tok, torch.zeros_like(old_logprobs_tok))

        # Raport token-level + clip (mascat pe tokenii activi)
        delta_tok = new_logprobs_tok - old_logprobs_tok
        delta_tok = torch.where(active_bool, delta_tok, torch.zeros_like(delta_tok))
        delta_tok = delta_tok.clamp(-20.0, 20.0)
        ratio_tok = torch.exp(delta_tok)  # [B, T-1]

        surr1 = ratio_tok * adv_tok
        surr2 = torch.clamp(ratio_tok, 1 - self.cfg.clip_epsilon, 1 + self.cfg.clip_epsilon) * adv_tok

        # Policy loss: medie pe tokenii completării (apoi medie pe batch)
        policy_term = torch.min(surr1, surr2)
        policy_loss = - (policy_term * active_tok).sum(dim=1) / tok_count.squeeze(1)
        policy_loss = policy_loss.mean()
        policy_loss = torch.nan_to_num(policy_loss, nan=0.0, posinf=0.0, neginf=0.0)

        # Value loss cu clipping corect (element-wise)
        value_loss = self.value_loss_elementwise_clipped(values, rewards, old_values)

        # Entropie pe completare
        entropy = self.compute_entropy(logits_new, comp_mask.float()).mean()

        # KL(π_new || π_ref) token-wise pe completare
        with torch.no_grad():
            # ref_logits nu are grad
            pass
        kl_forward = self.forward_token_kl(logits_new, ref_logits.detach(), comp_mask.float()).mean()

        total_loss = (
            policy_loss
            + self.cfg.value_loss_weight * value_loss
            + self.cfg.kl_coef * kl_forward
            - self.cfg.entropy_weight * entropy
        )
        return total_loss, policy_loss, value_loss, kl_forward, entropy

    # ===================== Train Loop =====================

    def train(self, dataloader):
        step = 0
        total_updates = max(1, self.cfg.n_epochs * len(dataloader))
        initial_clip = self.initial_clip_epsilon
        final_clip   = self.final_clip_epsilon

        for epoch in range(self.cfg.n_epochs):
            self.policy_model.train()
            self.policy_ref.eval()
            self.reward_model.eval()
            epoch_loss = epoch_policy = epoch_value = epoch_kl = epoch_entropy = epoch_reward = 0.0
            n_batches = 0

            for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
                current_update = epoch * len(dataloader) + n_batches + 1
                frac = current_update / total_updates
                self.cfg.clip_epsilon = initial_clip * (1 - frac) + final_clip * frac
                step += 1

                batch_inputs = {
                    "input_ids": batch["input_ids"].to(self.device),
                    "attention_mask": batch["attention_mask"].to(self.device),
                }

                # // SCHED: Sampling schedule (temperature/top_p) pe baza KL rulant
                if len(self._kl_running) > 0:
                    kl_recent = float(np.mean(self._kl_running))
                    if kl_recent < 0.5 * self.cfg.target_kl:
                        # crește ușor explorarea când KL e mic
                        self.cfg.temperature = float(min(self.cfg.temperature + 0.02, 0.5))
                        self.cfg.top_p = float(min(self.cfg.top_p + 0.02, 0.9))
                    elif kl_recent > 1.5 * self.cfg.target_kl:
                        # reduce explorarea când KL e mare
                        self.cfg.temperature = float(max(self.cfg.temperature * 0.98, 0.25))
                        self.cfg.top_p = float(max(self.cfg.top_p * 0.99, 0.75))

                # 1) Generează completările
                completions = self.generate_batch_with_stopping(batch_inputs)

                # 2) Concat [prompt || completion] + comp_mask
                cat = self._concat_prompt_and_completion(batch_inputs, completions)
                input_ids, attention_mask, labels, comp_mask = (
                    cat["input_ids"], cat["attention_mask"], cat["labels"], cat["comp_mask"]
                )

                clean_completions = completions

                # 3) Old policy/ref (no_grad) pe secvența completă
                with torch.inference_mode():
                    with autocast(
                        device_type="cuda" if self.amp_enabled else "cpu",
                        dtype=self.autocast_dtype,
                        enabled=self.amp_enabled
                    ):
                        old_logits, old_values_ = self.policy_model(input_ids, attention_mask, labels)
                        ref_logits = self.policy_ref(input_ids=input_ids, attention_mask=attention_mask).logits

                    if old_values_ is None:
                        old_values_ = torch.zeros(input_ids.size(0), device=self.device)

                    old_logprobs_tok = self.compute_logprobs_tokenwise(old_logits, labels)  # [B, T-1]

                # 4) Rewards (sentiment + funcții custom)
                raw_sentiment_rewards = self.compute_sentiment_reward(
                    clean_completions,
                    use_margin=getattr(self.cfg, "use_margin_reward", False),
                    add_length_bonus=getattr(self.cfg, "add_length_bonus", True),
                    min_words=getattr(self.cfg, "min_words_bonus", 6),
                    bonus=getattr(self.cfg, "length_bonus", 0.05),
                )
                # --- schedule de ponderi pentru RewardsFunctions (format-first warmup) ---
                warmup = int(getattr(self.cfg, "format_warmup_steps", 300))
                if step <= warmup:
                    weights = {
                        "match_format_exactly": 3.0,
                        "match_format_approximately": 1.0,
                        "check_answer": 1.0,
                        "check_numbers": 0.5,
                    }
                else:
                    weights = {
                        "match_format_exactly": 1.0,
                        "match_format_approximately": 0.5,
                        "check_answer": 3.0,
                        "check_numbers": 1.0,
                    }

                raw_fn_rewards = self.rfn.compute_all_rewards(
                    prompts=batch["prompt"],
                    completions=clean_completions,
                    answers=batch["solution"],
                    weights=weights,
                )
                raw_rewards = (raw_sentiment_rewards + raw_fn_rewards).to(self.device)

                # Normalizare stabilă (batch-wise) cu fallback când varianța este foarte mică
                mean = raw_rewards.mean()
                std  = raw_rewards.std(unbiased=False)
                rewards = (raw_rewards - mean) / (std + 1e-8)
                # scalare opțională din config pentru a regla mărimea avantajului
                reward_scale = getattr(self.cfg, "reward_scale", 1.0)
                rewards = rewards * reward_scale
                # fallback: dacă std ~ 0, adaugă un mic jitter ca să nu obții avantaje nule
                if float(std.item()) < 1e-6:
                    rewards = rewards + 1e-3 * torch.randn_like(rewards)

                # format_rate: proporția completărilor care conțin corect blocul <SOLUTION>…</SOLUTION>
                try:
                    fmt_hits = [1 if self.rfn.solution_block.search(t) is not None else 0 for t in clean_completions]
                    format_rate = float(sum(fmt_hits)) / max(1, len(fmt_hits))
                except Exception:
                    format_rate = 0.0
                self.writer.add_scalar("Reward/format_rate", format_rate, step)

                # logging per-exemplu
                if step % getattr(self.cfg, "log_every", 50) == 0:
                    os.makedirs("./logs", exist_ok=True)
                    with open("./logs/rewards_log.txt", "a") as f:
                        for i, (text, reward) in enumerate(zip(clean_completions, raw_rewards)):
                            am = batch_inputs["attention_mask"][i]
                            pl = int(am.sum().item())
                            prompt_ids = batch_inputs["input_ids"][i, -pl:]
                            prompt_str = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
                            space_star = "-" * 20
                            has_sol = "YES" if self.rfn.solution_block.search(text) is not None else "NO"
                            f.write(
                                f"Step {step} | Example {i} | Reward: {float(reward):.4f} | HAS_SOLUTION: {has_sol}\n"
                                f"Prompt: {prompt_str} \n"
                                f"Completion: {text}\n {space_star}\n"
                            )
                    self.writer.add_scalar("Reward/raw_mean", rewards.mean().item(), step)
                    self.writer.add_scalar("Reward/raw_std", rewards.std().item(), step)

                # 5) Salvează tranziția în buffer (CPU)
                self.replay_buffer.append({
                    "input_ids": input_ids.detach().cpu(),
                    "attention_mask": attention_mask.detach().cpu(),
                    "labels": labels.detach().cpu(),
                    "comp_mask": comp_mask.detach().cpu(),
                    "old_logprobs_tok": old_logprobs_tok.detach().cpu(),
                    "old_values": old_values_.detach().cpu(),
                    "rewards": rewards.detach().cpu()
                })

                # 6) Setup pentru update-uri
                ppo_epochs = max(1, getattr(self.cfg, "ppo_epochs", 4))

                # Dimensiune micro-batch adaptivă în funcție de lungimea medie
                seq_lens = attention_mask.sum(dim=1).float()
                avg_len = max(1.0, seq_lens.mean().item())
                token_budget = max(1, int(getattr(self.cfg, "token_microbatch_size", 2048)))
                cfg_micro = getattr(self.cfg, "microbatch_size", input_ids.size(0))  # fallback: tot batch-ul
                microbatch_size = min(
                    cfg_micro,
                    max(1, int(token_budget / avg_len))
                )
                accum_steps = getattr(self.cfg, "accum_steps", 1)

                B = input_ids.size(0)
                if microbatch_size >= B:
                    microbatch_size = B

                def ppo_update_pass(
                    input_ids, attention_mask, labels,
                    old_logprobs_tok,
                    rewards, old_values,
                    comp_mask
                ):
                    mb_B = input_ids.size(0)
                    mb_losses = []; mb_pols = []; mb_vals = []; mb_kls = []; mb_ents = []
                    mb_count = 0
                    self.optimizer.zero_grad(set_to_none=True)

                    for start in range(0, mb_B, microbatch_size):
                        end = min(start + microbatch_size, mb_B)
                        mb_input_ids      = input_ids[start:end]
                        mb_attention_mask = attention_mask[start:end]
                        mb_labels         = labels[start:end]
                        mb_comp_mask      = comp_mask[start:end]
                        mb_old_logp_tok   = old_logprobs_tok[start:end]
                        mb_rewards        = rewards[start:end]
                        mb_old_values     = old_values[start:end]

                        with autocast(
                            device_type="cuda" if self.amp_enabled else "cpu",
                            dtype=self.autocast_dtype,
                            enabled=self.amp_enabled
                        ):
                            logits_new, values = self.policy_model(mb_input_ids, mb_attention_mask, mb_labels)
                            # Recalculează ref_logits pentru minibatch (nu le stocăm pentru a nu umple RAM-ul)
                            with torch.inference_mode():
                                ref_logits = self.policy_ref(input_ids=mb_input_ids, attention_mask=mb_attention_mask).logits

                            loss, pol_loss, val_loss, kl_fwd, entropy = self.ppo_loss(
                                mb_old_logp_tok, logits_new, ref_logits,
                                mb_rewards, values, mb_old_values,
                                mb_labels, mb_comp_mask
                            )

                            loss = loss / accum_steps

                        self.scaler.scale(loss).backward()
                        mb_count += 1

                        mb_losses.append(loss.item() * accum_steps)
                        mb_pols.append(pol_loss.item()); mb_vals.append(val_loss.item())
                        mb_kls.append(kl_fwd.item()); mb_ents.append(entropy.item())

                        if (mb_count % accum_steps == 0) or (end == mb_B):
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            if self.scheduler is not None:
                                self.scheduler.step()
                            self.optimizer.zero_grad(set_to_none=True)

                    return (
                        torch.tensor(mb_losses, device=self.device).mean().detach().cpu(),
                        torch.tensor(mb_pols,   device=self.device).mean().detach().cpu(),
                        torch.tensor(mb_vals,   device=self.device).mean().detach().cpu(),
                        torch.tensor(mb_kls,    device=self.device).mean().detach().cpu(),
                        torch.tensor(mb_ents,   device=self.device).mean().detach().cpu(),
                    )

                # 7) Off-policy replay (dacă buffer plin)
                if len(self.replay_buffer) >= self.cfg.buffer_size:
                    batch_samples = random.sample(self.replay_buffer, self.cfg.replay_batch_size)

                    # concat și mutare pe device
                    buffer_input_ids      = torch.cat([b["input_ids"] for b in batch_samples], dim=0).to(self.device)
                    buffer_attention_mask = torch.cat([b["attention_mask"] for b in batch_samples], dim=0).to(self.device)
                    buffer_labels         = torch.cat([b["labels"] for b in batch_samples], dim=0).to(self.device)
                    buffer_comp_mask      = torch.cat([b["comp_mask"] for b in batch_samples], dim=0).to(self.device)
                    buffer_old_logp_tok   = torch.cat([b["old_logprobs_tok"] for b in batch_samples], dim=0).to(self.device)
                    buffer_old_values     = torch.cat([b["old_values"] for b in batch_samples], dim=0).to(self.device)
                    buffer_rewards        = torch.cat([b["rewards"] for b in batch_samples], dim=0).to(self.device)

                    _ = ppo_update_pass(
                        buffer_input_ids, buffer_attention_mask, buffer_labels,
                        buffer_old_logp_tok,
                        buffer_rewards, buffer_old_values,
                        comp_mask=buffer_comp_mask
                    )

                # Inițializează last_* (evită UnboundLocalError)
                last_loss = torch.tensor(0.0)
                last_pol  = torch.tensor(0.0)
                last_val  = torch.tensor(0.0)
                last_kl   = torch.tensor(0.0)
                last_ent  = torch.tensor(0.0)

                # 8) PPO epochs pe batch-ul curent
                for _ in range(ppo_epochs):
                    losses, pols, vals, kls, ents = ppo_update_pass(
                        input_ids, attention_mask, labels,
                        old_logprobs_tok,
                        rewards, old_values_,
                        comp_mask=comp_mask
                    )

                    # salvează imediat (în caz că ieși din buclă pe KL)
                    last_loss, last_pol, last_val = losses, pols, vals
                    last_kl, last_ent = kls, ents

                    # // SCHED: Reward-scale și Value-loss weight pe baza semnalului curent
                    try:
                        pol_abs = abs(float(pols))
                        kl_now = float(kls)
                        # Amplifică ușor reward_scale dacă policy update e prea mic
                        if pol_abs < 1e-3:
                            self.cfg.reward_scale = float(min(getattr(self.cfg, 'reward_scale', 1.0) * 1.1, 3.0))
                        # Taie un pic dacă KL sare peste țintă semnificativ
                        elif kl_now > 1.5 * self.cfg.target_kl:
                            self.cfg.reward_scale = float(max(getattr(self.cfg, 'reward_scale', 1.0) / 1.1, 0.5))

                        # Ajustează greutatea value loss ca să nu domine
                        vp = float(vals) / (pol_abs + 1e-6)
                        if vp > 10.0:
                            self.cfg.value_loss_weight = float(max(self.cfg.value_loss_weight * 0.9, 0.1))
                        elif vp < 2.0:
                            self.cfg.value_loss_weight = float(min(self.cfg.value_loss_weight * 1.1, 0.6))
                    except Exception:
                        pass

                    # Adaptive KL pe forward KL
                    kl_val = float(kls.item())
                    self._kl_running.append(kl_val)
                    if kl_val > self.cfg.kl_stop_factor * self.cfg.target_kl:
                        break

                    if len(self._kl_running) >= self.cfg.kl_window:
                        kl_avg = sum(self._kl_running) / len(self._kl_running)
                        if kl_avg > 2.0 * self.cfg.target_kl:
                            self.cfg.kl_coef = min(self.cfg.kl_coef * self.cfg.kl_adapt_rate, self.cfg.max_kl_coef)
                        elif kl_avg < 0.5 * self.cfg.target_kl:
                            self.cfg.kl_coef = max(self.cfg.kl_coef / self.cfg.kl_adapt_rate, self.cfg.min_kl_coef)
                        self.writer.add_scalar("KL/window_avg_forward", kl_avg, step)
                        self.writer.add_scalar("KL/adapted_coef", self.cfg.kl_coef, step)
                        # // SCHED: Entropy weight – crește când KL e mic, scade când e mare
                        if kl_avg < 0.5 * self.cfg.target_kl:
                            self.cfg.entropy_weight = float(min(getattr(self.cfg, 'entropy_weight', 0.0) + 1e-4, 1e-3))
                        elif kl_avg > 1.5 * self.cfg.target_kl:
                            self.cfg.entropy_weight = float(max(getattr(self.cfg, 'entropy_weight', 0.0) * 0.5, 0.0))
                        self.writer.add_scalar('Entropy/weight', getattr(self.cfg, 'entropy_weight', 0.0), step)
                        self._kl_running.clear()

                # Agregare pe epocă
                epoch_loss    += float(last_loss.item())
                epoch_policy  += float(last_pol.item())
                epoch_value   += float(last_val.item())
                epoch_kl      += float(last_kl.item())
                epoch_entropy += float(last_ent.item())
                epoch_reward  += float(rewards.mean().item())
                n_batches     += 1

                # Log periodic
                if step % getattr(self.cfg, "log_every", 50) == 0:
                    avg_loss = epoch_loss / n_batches
                    avg_pol  = epoch_policy / n_batches
                    avg_val  = epoch_value / n_batches
                    avg_kl   = epoch_kl / n_batches
                    avg_ent  = epoch_entropy / n_batches
                    avg_r    = epoch_reward / n_batches

                    tqdm.write(
                        f"[Epoch {epoch}] Step {step} | Loss: {avg_loss:.6f} | Policy: {avg_pol:.6f} | "
                        f"Value: {avg_val:.6f} | KL_fwd(tok): {avg_kl:.6f} | Entropy: {avg_ent:.6f} | Reward: {avg_r:.6f}"
                    )

                    self.writer.add_scalar("Stats/KL_fwd_tok", avg_kl, step)
                    self.writer.add_scalar("Loss/total",   avg_loss, step)
                    self.writer.add_scalar("Loss/policy",  avg_pol,  step)
                    self.writer.add_scalar("Loss/value",   avg_val,  step)
                    self.writer.add_scalar("Stats/entropy",avg_ent,  step)
                    self.writer.add_scalar("Reward/mean",  avg_r,    step)

            # === end epoch: model selection + checkpoint ===
            avg_epoch_reward = (epoch_reward / n_batches) if n_batches > 0 else 0.0
            # // SCHED: Length curriculum pe baza KL mediu al epocii
            avg_epoch_kl = (epoch_kl / n_batches) if n_batches > 0 else 0.0
            try:
                if avg_epoch_kl < self.cfg.target_kl:
                    self.cfg.max_new_tokens = int(min(self.cfg.max_new_tokens + 64, 2048))
                elif avg_epoch_kl > 1.5 * self.cfg.target_kl:
                    self.cfg.max_new_tokens = int(max(self.cfg.max_new_tokens - 32, 512))
            except Exception:
                pass
            self.writer.add_scalar('Gen/max_new_tokens', getattr(self.cfg, 'max_new_tokens', 0), epoch)

            if avg_epoch_reward > getattr(self, "best_reward", -1e9):
                self.best_reward = avg_epoch_reward
                self.no_improve_epochs = 0
                self.policy_model.save_pretrained(f"{self.cfg.save_dir}_best")
                print(f"✅ Model saved at epoch {epoch} with improved reward {avg_epoch_reward:.3f}")
            else:
                self.no_improve_epochs = getattr(self, "no_improve_epochs", 0) + 1

            if (epoch + 1) % getattr(self.cfg, "save_every", 1) == 0 or epoch == 0:
                self.policy_model.save_pretrained(f"{self.cfg.save_dir}_epoch_{epoch}")
                print(f"✅ Model saved at epoch {epoch}")
