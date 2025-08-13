import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import List, Optional, Dict
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
from collections import deque

from .reward_fn_math_reasoning import RewardFunctions


class PPOAgent:
    """PPO pentru LLM cu:
       - Forward pe [prompt + completare]
       - labels = -100 pe PAD și pe tokenii de prompt (contăm doar completarea)
       - Entropie/KL mediate DOAR pe completare via comp_mask
       - AMP, micro-batching, gradient accumulation
       - Off-policy replay (buffer)
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
        self._kl_running = []  # pentru adaptive KL (pe |Δ|)

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

    def compute_logprobs(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Returnează log-probabilități mediate pe tokenii cu label != -100.
           (mask nu e necesară — labels gestionează prompt+PAD).
        """
        if mask is None:
            mask = (labels != -100).float()
        per_token = -F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view(labels.size())
        denom = mask.sum(dim=1).clamp_min(1)
        return (per_token * mask).sum(dim=1) / denom  # [B]

    def compute_entropy(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Entropie mediată PE COMP_MASK (1 = completare, 0 = prompt/PAD)."""
        log_probs = F.log_softmax(logits, dim=-1)   # [B, T, V]
        probs = log_probs.exp()
        ent = -(probs * log_probs).sum(dim=-1)      # [B, T]
        denom = mask.sum(dim=1).clamp_min(1)
        return (ent * mask).sum(dim=1) / denom      # [B]


    # INFO: Neutilizat (un GAE - Artificial pe batch)
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor
    ) -> torch.Tensor:
        advantages = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.cfg.gamma * next_values[t] * mask - values[t]
            gae = delta + self.cfg.gamma * self.cfg.lambda_gae * mask * gae
            advantages[t] = gae
        return advantages

    def value_loss(self, values: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(values, rewards)

    def compute_sentiment_reward(
        self,
        texts: List[str],
        use_margin: bool = False,
        add_length_bonus: bool = True,
        min_words: int = 6,
        bonus: float = 0.05
    ):
        """Reward de sentiment + bonus de lungime + mic bonus de diversitate."""
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

            # mic bonus de „diversitate” (mai puțină repetiție de bigrame)
            def overlap_ratio(text):
                toks = text.split()
                bigrams = [(toks[i], toks[i+1]) for i in range(len(toks)-1)]
                uniq = set(bigrams)
                return 1.0 - len(uniq) / max(1, len(bigrams))
            divers = torch.tensor([overlap_ratio(t) for t in texts], device=self.device)
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

            # ATENȚIE: construim attention_mask prin poziție, nu prin (merged != pad_id)
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
        # prompt mascat integral: nr. de 1 din attention - nr. de 1 din comp_mask <= nr. de -100 (aprox.)
        assert (out["labels"] == -100).sum() >= (out["attention_mask"].sum() - out["comp_mask"].sum()), \
            "Promptul nu este complet mascat în labels."

        return out


    # ===================== Generare =====================

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

    # ===================== PPO Loss =====================

    def ppo_loss(
        self,
        old_policy_logprobs: torch.Tensor,   # logπ_old (no-grad)
        new_logprobs: torch.Tensor,          # logπ_new
        ref_logprobs: torch.Tensor,          # logπ_ref (no-grad)
        rewards: torch.Tensor,
        values: torch.Tensor,
        old_values: torch.Tensor,
        logits: torch.Tensor,
        mask: torch.Tensor                   # comp_mask (1 pe completare)
    ) -> tuple:
        # Bandit: un singur reward pe secvență
        advantages = rewards - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Entropie pe completare & KL_ref
        entropy = self.compute_entropy(logits, mask.float()).mean()
        delta_logp = new_logprobs - ref_logprobs
        kl_ref_signed = delta_logp.mean()
        kl_ref_abs = delta_logp.abs().mean()

        # PPO clipping
        delta_for_ratio = torch.clamp(new_logprobs - old_policy_logprobs, min=-20.0, max=20.0)
        ratio = torch.exp(delta_for_ratio)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.cfg.clip_epsilon, 1 + self.cfg.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value clipping
        clipped_values = old_values + torch.clamp(
            values - old_values, -self.cfg.value_clip_range, self.cfg.value_clip_range
        )
        unclipped_loss = F.mse_loss(values, rewards)
        clipped_loss = F.mse_loss(clipped_values, rewards)
        value_loss = torch.max(unclipped_loss, clipped_loss)

        total_loss = (
            policy_loss
            + self.cfg.value_loss_weight * value_loss
            + self.cfg.kl_coef * kl_ref_abs
            - self.cfg.entropy_weight * entropy
        )
        return total_loss, policy_loss, value_loss, kl_ref_signed, entropy, kl_ref_abs

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

                # 1) Generează completările
                completions = self.generate(batch_inputs)

                # 2) Concat [prompt || completion] + comp_mask
                cat = self._concat_prompt_and_completion(batch_inputs, completions)
                input_ids, attention_mask, labels, comp_mask = (
                    cat["input_ids"], cat["attention_mask"], cat["labels"], cat["comp_mask"]
                )

                clean_completions = completions  # textul pur al completărilor

                # 3) Old policy/ref (no_grad) pe secvența completă
                with torch.inference_mode():
                    with autocast(
                        device_type="cuda" if self.amp_enabled else "cpu",
                        dtype=self.autocast_dtype,
                        enabled=self.amp_enabled
                    ):
                        old_logits, old_values_ = self.policy_model(input_ids, attention_mask, labels)
                        ref_logits = self.policy_ref(input_ids=input_ids, attention_mask=attention_mask).logits

                    old_policy_logprobs = self.compute_logprobs(old_logits, labels)  # [B]
                    ref_logprobs        = self.compute_logprobs(ref_logits, labels)  # [B]
                    if old_values_ is None:
                        old_values_ = torch.zeros_like(old_policy_logprobs)

                old_policy_logprobs = old_policy_logprobs.detach()
                ref_logprobs        = ref_logprobs.detach()
                old_values          = old_values_.detach()

                # 4) Rewards (sentiment + funcții custom)
                raw_sentiment_rewards = self.compute_sentiment_reward(
                    clean_completions,
                    use_margin=getattr(self.cfg, "use_margin_reward", False),
                    add_length_bonus=getattr(self.cfg, "add_length_bonus", True),
                    min_words=getattr(self.cfg, "min_words_bonus", 6),
                    bonus=getattr(self.cfg, "length_bonus", 0.05),
                )
                raw_fn_rewards = self.rfn.compute_all_rewards(
                    prompts=batch["prompt"],
                    completions=clean_completions,
                    answers=batch["solution"]
                )
                raw_rewards = (raw_sentiment_rewards + raw_fn_rewards).to(self.device)

                # Normalizare stabilă
                rewards = (raw_rewards - raw_rewards.mean()) / (raw_rewards.std() + 1e-8)
                rewards = rewards.clamp(-5.0, 5.0)


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
                            f.write(
                                f"Step {step} | Example {i} | Reward: {float(reward):.4f} \n"
                                f"Prompt: {prompt_str} \n"
                                f"Completion: {text}\n {space_star}\n"
                            )
                    self.writer.add_scalar("Reward/raw_mean", rewards.mean().item(), step)
                    self.writer.add_scalar("Reward/raw_std", rewards.std().item(), step)

                # 6) Salvează tranziția în buffer (CPU) — reduce presiunea pe VRAM
                self.replay_buffer.append({
                    "input_ids": input_ids.detach().cpu(),
                    "attention_mask": attention_mask.detach().cpu(),
                    "labels": labels.detach().cpu(),
                    "comp_mask": comp_mask.detach().cpu(),
                    "old_policy_logprobs": old_policy_logprobs.detach().cpu(),
                    "ref_logprobs": ref_logprobs.detach().cpu(),
                    "old_values": old_values.detach().cpu(),
                    "rewards": rewards.detach().cpu()
                })

                # 7) Update PPO cu micro-batching + accum
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
                    old_policy_logprobs, ref_logprobs,
                    rewards, old_values,
                    comp_mask
                ):
                    mb_B = input_ids.size(0)
                    mb_losses = []; mb_pols = []; mb_vals = []; mb_kls_signed = []; mb_ents = []; mb_kls_abs = []
                    mb_count = 0
                    self.optimizer.zero_grad(set_to_none=True)

                    for start in range(0, mb_B, microbatch_size):
                        end = min(start + microbatch_size, mb_B)
                        mb_input_ids           = input_ids[start:end]
                        mb_attention_mask      = attention_mask[start:end]
                        mb_labels              = labels[start:end]
                        mb_comp_mask           = comp_mask[start:end]
                        mb_old_policy_logprobs = old_policy_logprobs[start:end]
                        mb_ref_logprobs        = ref_logprobs[start:end]
                        mb_rewards             = rewards[start:end]
                        mb_old_values          = old_values[start:end]

                        with autocast(
                            device_type="cuda" if self.amp_enabled else "cpu",
                            dtype=self.autocast_dtype,
                            enabled=self.amp_enabled
                        ):
                            logits, values = self.policy_model(mb_input_ids, mb_attention_mask, mb_labels)
                            new_logprobs = self.compute_logprobs(logits, mb_labels)

                            loss, pol_loss, val_loss, kl_ref_signed, entropy, kl_ref_abs = self.ppo_loss(
                                mb_old_policy_logprobs, new_logprobs, mb_ref_logprobs,
                                mb_rewards, values, mb_old_values,
                                logits, mb_comp_mask
                            )

                            loss = loss / accum_steps

                        self.scaler.scale(loss).backward()
                        mb_count += 1

                        mb_losses.append(loss.item() * accum_steps)
                        mb_pols.append(pol_loss.item()); mb_vals.append(val_loss.item())
                        mb_kls_signed.append(kl_ref_signed.item()); mb_ents.append(entropy.item()); mb_kls_abs.append(kl_ref_abs.item())

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
                        torch.tensor(mb_kls_signed, device=self.device).mean().detach().cpu(),
                        torch.tensor(mb_ents,   device=self.device).mean().detach().cpu(),
                        torch.tensor(mb_kls_abs, device=self.device).mean().detach().cpu()
                    )

                # 8) Off-policy replay (dacă buffer plin)
                if len(self.replay_buffer) >= self.cfg.buffer_size:
                    batch_samples = random.sample(self.replay_buffer, self.cfg.replay_batch_size)

                    # concat și mutare pe device
                    buffer_input_ids      = torch.cat([b["input_ids"] for b in batch_samples], dim=0).to(self.device)
                    buffer_attention_mask = torch.cat([b["attention_mask"] for b in batch_samples], dim=0).to(self.device)
                    buffer_labels         = torch.cat([b["labels"] for b in batch_samples], dim=0).to(self.device)
                    buffer_comp_mask      = torch.cat([b["comp_mask"] for b in batch_samples], dim=0).to(self.device)
                    buffer_old_policy_logprobs = torch.cat([b["old_policy_logprobs"] for b in batch_samples], dim=0).to(self.device)
                    buffer_old_values     = torch.cat([b["old_values"] for b in batch_samples], dim=0).to(self.device)
                    buffer_ref_logprobs   = torch.cat([b["ref_logprobs"] for b in batch_samples], dim=0).to(self.device)
                    buffer_rewards        = torch.cat([b["rewards"] for b in batch_samples], dim=0).to(self.device)

                    _ = ppo_update_pass(
                        buffer_input_ids, buffer_attention_mask, buffer_labels,
                        buffer_old_policy_logprobs, buffer_ref_logprobs,
                        buffer_rewards, buffer_old_values,
                        comp_mask=buffer_comp_mask
                    )

                # Inițializează last_* (evită UnboundLocalError)
                last_loss = torch.tensor(0.0)
                last_pol  = torch.tensor(0.0)
                last_val  = torch.tensor(0.0)
                last_kl_signed = torch.tensor(0.0)
                last_ent  = torch.tensor(0.0)
                last_kl_abs = torch.tensor(0.0)

                # 9) PPO epochs pe batch-ul curent
                for _ in range(ppo_epochs):
                    losses, pols, vals, kls_signed, ents, kls_abs = ppo_update_pass(
                        input_ids, attention_mask, labels,
                        old_policy_logprobs, ref_logprobs,
                        rewards, old_values,
                        comp_mask=comp_mask
                    )

                    # salvează imediat (în caz că ieși din buclă pe KL)
                    last_loss, last_pol, last_val = losses, pols, vals
                    last_kl_signed, last_ent, last_kl_abs = kls_signed, ents, kls_abs

                    # Adaptive KL pe |Δ|
                    kl_abs_val = float(kls_abs.item())
                    self._kl_running.append(kl_abs_val)
                    if kl_abs_val > self.cfg.kl_stop_factor * self.cfg.target_kl:
                        break

                    if len(self._kl_running) >= self.cfg.kl_window:
                        kl_avg = sum(self._kl_running) / len(self._kl_running)
                        if kl_avg > 2.0 * self.cfg.target_kl:
                            self.cfg.kl_coef = min(self.cfg.kl_coef * self.cfg.kl_adapt_rate, self.cfg.max_kl_coef)
                        elif kl_avg < 0.5 * self.cfg.target_kl:
                            self.cfg.kl_coef = max(self.cfg.kl_coef / self.cfg.kl_adapt_rate, self.cfg.min_kl_coef)
                        self.writer.add_scalar("KL/window_avg_abs", kl_avg, step)
                        self.writer.add_scalar("KL/adapted_coef", self.cfg.kl_coef, step)
                        self._kl_running.clear()

                # Agregare pe epocă
                epoch_loss    += float(last_loss.item())
                epoch_policy  += float(last_pol.item())
                epoch_value   += float(last_val.item())
                epoch_kl      += float(last_kl_signed.item())
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
                        f"Value: {avg_val:.6f} | KL_ref(tok): {avg_kl:.6f} | Entropy: {avg_ent:.6f} | Reward: {avg_r:.6f}"
                    )

                    self.writer.add_scalar("Stats/KL_ref_tok", avg_kl, step)
                    self.writer.add_scalar("Loss/total",   avg_loss, step)
                    self.writer.add_scalar("Loss/policy",  avg_pol,  step)
                    self.writer.add_scalar("Loss/value",   avg_val,  step)
                    self.writer.add_scalar("Stats/entropy",avg_ent,  step)
                    self.writer.add_scalar("Reward/mean",  avg_r,    step)

            # === end epoch: model selection + checkpoint ===
            avg_epoch_reward = (epoch_reward / n_batches) if n_batches > 0 else 0.0
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
