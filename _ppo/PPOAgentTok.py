# _ppo/ppo_agent_tok.py
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import List, Dict
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
from collections import deque

from .reward_fn_math_reasoning import RewardFunctions


class PPOAgent:
    """PPO (RLHF) cu GAE pe TOKENI, logprobs chunked (low-VRAM), NEW forward pe micro-batch, adaptive-KL, fără entropie în update."""

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
        self.initial_clip_epsilon = float(getattr(self.cfg, "clip_epsilon", 0.2))
        self.final_clip_epsilon   = float(getattr(self.cfg, "clip_epsilon_final", self.initial_clip_epsilon))
        self.device = config.device
        self.scheduler = scheduler

        # Seeds & perf
        random.seed(self.cfg.seed); np.random.seed(self.cfg.seed)
        os.environ["PYTHONHASHSEED"] = str(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.cfg.seed)
        torch.backends.cudnn.deterministic = False
        cudnn.benchmark = True

        # AMP
        self.amp_enabled = torch.cuda.is_available()
        self.autocast_dtype = torch.bfloat16 if (self.amp_enabled and torch.cuda.is_bf16_supported()) else None
        if self.amp_enabled:
            torch.set_float32_matmul_precision("high")

        # Replay (token-level)
        self.replay_buffer = deque(maxlen=getattr(self.cfg, "buffer_size", 0))

        # Models
        self.policy_model = policy_model.to(self.device).train()
        self.policy_ref = policy_ref.to(self.device).eval()
        self.reward_model = reward_model.to(getattr(self.cfg, "reward_device", self.device)).eval()

        # Reward model mapping
        labels_map = getattr(self.reward_model.config, "id2label", {0:"NEGATIVE",1:"POSITIVE"})
        inv = {v.lower(): k for k, v in labels_map.items()}
        self.pos_idx = inv.get("positive", 1)
        self.neg_idx = inv.get("negative", 0 if self.pos_idx != 0 else 1)

        # Logger
        self.scaler = GradScaler(enabled=self.amp_enabled)
        self.writer = SummaryWriter(log_dir=self.cfg.log_dir)
        self.best_reward = -float("inf")
        self._kl_running = []

        # Tokenizer
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        self.tokenizer = tokenizer
        self.reward_tokenizer = reward_tokenizer
        self.optimizer = optimizer

        # Custom reward fns
        self.rfn = RewardFunctions(self.tokenizer, self.device)

    # ------------------------ utils ------------------------

    def compute_token_logprobs(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        Safe gather: înlocuiește -100 cu 0 pentru a evita index OOB; apoi aplică masca.
        """
        assert logits.dim() == 3 and labels.dim() == 2, f"{logits.shape} vs {labels.shape}"
        logp = F.log_softmax(logits, dim=-1)                     # [B,T,V]
        safe_labels = labels.clamp_min(0)                        # -100 -> 0
        token_logp = logp.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)  # [B,T]
        mask = (labels != -100).float()
        return token_logp, mask

    def token_logprobs_chunked(self, logits: torch.Tensor, labels: torch.Tensor, chunk: int = 128):
        """
        Logprobs pe tokeni calculat în benzi pe axa T (reduce peak VRAM).
        """
        B, T, V = logits.shape
        device = logits.device
        out = torch.empty(B, T, device=device, dtype=logits.dtype)
        for s in range(0, T, chunk):
            e = min(s + chunk, T)
            slab = logits[:, s:e, :]                   # [B, t, V]
            logp = F.log_softmax(slab, dim=-1)         # [B, t, V]
            safe_labels = labels[:, s:e].clamp_min(0)  # [B, t]
            out[:, s:e] = logp.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        mask = (labels != -100).float()
        return out, mask

    def _masked_mean(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        denom = m.sum().clamp_min(1.0)
        return (x * m).sum() / denom

    def compute_gae_tokens(self, rewards_tok, values_tok, next_values_tok, dones_tok, mask_tok):
        B, T = rewards_tok.shape
        adv = torch.zeros_like(rewards_tok)
        last_adv = torch.zeros(B, device=rewards_tok.device, dtype=rewards_tok.dtype)
        gamma, lam = self.cfg.gamma, self.cfg.lambda_gae
        for t in reversed(range(T)):
            not_done = (1.0 - dones_tok[:, t])
            delta = rewards_tok[:, t] + gamma * next_values_tok[:, t] * not_done - values_tok[:, t]
            last_adv = delta + gamma * lam * not_done * last_adv
            adv[:, t] = last_adv
        adv = adv * mask_tok
        return adv

    # ------------------------ tokenization & generate ------------------------

    def tokenize_clean(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        tok = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.cfg.policy_max_length,
        )
        tok.pop("token_type_ids", None)
        input_ids = tok["input_ids"]
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": input_ids.to(self.device),
            "attention_mask": tok["attention_mask"].to(self.device),
            "labels": labels.to(self.device),
        }

    def generate(self, batch_inputs, max_new_tokens=None):
        with torch.inference_mode():
            max_new_tokens = max_new_tokens or self.cfg.max_new_tokens
            rep_pen = getattr(self.cfg, "repetition_penalty", 1.0)
            with autocast(device_type="cuda" if self.amp_enabled else "cpu", 
                          dtype=self.autocast_dtype, enabled=self.amp_enabled):
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

    # ------------------------ reward core + shaping ------------------------

    def compute_sentiment_reward(self, texts, use_margin=False, add_length_bonus=True, min_words=6, bonus=0.05):
        with torch.inference_mode():
            inputs = self.reward_tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True,
                max_length=self.cfg.reward_max_length,
            )
            inputs.pop("token_type_ids", None)
            inputs = {k: v.to(getattr(self.cfg, "reward_device", self.device)) for k, v in inputs.items()}
            logits = self.reward_model(**inputs).logits  # [B, C]
            if use_margin and logits.size(-1) >= 2:
                margin = logits[:, self.pos_idx] - logits[:, self.neg_idx]
                raw = torch.sigmoid(margin)
            else:
                probs = F.softmax(logits, dim=-1)
                raw = probs[:, self.pos_idx]
            if add_length_bonus:
                lens = torch.tensor([len(t.split()) for t in texts], device=raw.device, dtype=torch.float32)
                raw = (raw + (lens >= min_words).float() * bonus).clamp(0.0, 1.0)
            raw = (raw * 2.0 - 1.0).clamp(-1.0, 1.0)
        return raw.to(self.device)  # [B]

    @staticmethod
    def _lcs_len(a: List[str], b: List[str]) -> int:
        n, m = len(a), len(b)
        dp = [[0]*(m+1) for _ in range(n+1)]
        for i in range(1, n+1):
            ai = a[i-1]
            row = dp[i]; prev = dp[i-1]
            for j in range(1, m+1):
                row[j] = prev[j-1] + 1 if ai == b[j-1] else max(prev[j], row[j-1])
        return dp[n][m]

    def _rouge_l_f1(self, hyp: str, ref: str) -> float:
        hyp_t, ref_t = hyp.split(), ref.split()
        if not hyp_t or not ref_t:
            return 0.0
        lcs = self._lcs_len(hyp_t, ref_t)
        prec = lcs / max(1, len(hyp_t))
        rec  = lcs / max(1, len(ref_t))
        return 0.0 if (prec+rec)==0 else 2*prec*rec/(prec+rec)

    def build_token_shaping(self, labels_comp: torch.Tensor, gen_mask: torch.Tensor,
                            completions_decoded: List[str], answers: List[str]) -> torch.Tensor:
        """
        R_shape:[B,Tc] = anti-repetiție incrementală (bigrame & n-gram) + ROUGE-L incremental pe prefixe.
        """
        B, Tc = labels_comp.shape
        rep_ngram = int(getattr(self.cfg, "rep_ngram", 3))
        rep_penalty = float(getattr(self.cfg, "rep_penalty", 0.01))
        rouge_coef = float(getattr(self.cfg, "rouge_shaping", 0.1))

        ids = labels_comp.clone()
        ids[labels_comp == -100] = self.tokenizer.pad_token_id
        rep_reward = torch.zeros_like(gen_mask, dtype=torch.float32, device=self.device)

        for b in range(B):
            seq = ids[b].tolist()
            msk = gen_mask[b].bool().tolist()
            seen = set()
            window = []
            for t in range(Tc):
                if not msk[t]:
                    continue
                window.append(seq[t])
                penalize = 0.0
                for n in (2, rep_ngram):
                    if len(window) >= n:
                        ng = tuple(window[-n:])
                        if ng in seen:
                            penalize -= rep_penalty
                        else:
                            seen.add(ng)
                rep_reward[b, t] = penalize

        rouge_reward = torch.zeros_like(gen_mask, dtype=torch.float32, device=self.device)
        for b in range(B):
            ref = answers[b]
            comp = completions_decoded[b]
            toks = comp.split()
            prev = 0.0
            pos = 0
            for t in range(Tc):
                if not gen_mask[b, t]:
                    continue
                pos += 1
                hyp = " ".join(toks[:pos])
                cur = self._rouge_l_f1(hyp, ref)
                rouge_reward[b, t] = rouge_coef * (cur - prev)
                prev = cur

        return (rep_reward + rouge_reward).to(self.device)

    # ------------------------ PPO loss (token-level) ------------------------

    def ppo_loss_tokens(
        self,
        old_logp_tok, new_logp_tok, ref_logp_tok,   # [B,T]
        rewards_tok, values_tok, old_values_tok,    # [B,T]
        mask_tok,                                   # [B,T]
        next_values_tok, dones_tok,                 # [B,T]
        entropy_tok_mean=None                       # scalar opțional (nefolosit dacă None)
    ):
        advantages_tok = self.compute_gae_tokens(
            rewards_tok, values_tok.detach(), next_values_tok.detach(), dones_tok, mask_tok
        )
        flat = advantages_tok[mask_tok.bool()]
        advantages_tok = (advantages_tok - flat.mean()) / (flat.std().clamp_min(1e-8))

        ratio_tok = torch.exp(torch.clamp(new_logp_tok - old_logp_tok, -20, 20))
        surr1 = ratio_tok * advantages_tok
        surr2 = torch.clamp(ratio_tok, 1 - self.cfg.clip_epsilon, 1 + self.cfg.clip_epsilon) * advantages_tok
        policy_loss = -self._masked_mean(torch.min(surr1, surr2), mask_tok)

        vclip = self.cfg.value_clip_range
        v_clipped = old_values_tok + (values_tok - old_values_tok).clamp(-vclip, vclip)
        uncl = (values_tok - rewards_tok).pow(2)
        clp  = (v_clipped   - rewards_tok).pow(2)
        value_loss = self._masked_mean(torch.max(uncl, clp), mask_tok)

        kl_ref_abs = self._masked_mean((new_logp_tok - ref_logp_tok).abs(), mask_tok)

        # fără entropie (stabil/low-VRAM)
        total = (
            policy_loss
            + self.cfg.value_loss_weight * value_loss
            + self.cfg.kl_coef * kl_ref_abs
        )
        entropy_dummy = torch.tensor(0.0, device=self.device)
        return total, policy_loss, value_loss, kl_ref_abs, entropy_dummy

    # ------------------------ train loop ------------------------

    def train(self, dataloader):
        step = 0
        total_updates = self.cfg.n_epochs * len(dataloader)
        initial_clip = float(self.initial_clip_epsilon)
        final_clip   = float(self.final_clip_epsilon)
        device_type = "cuda" if self.amp_enabled else "cpu"
        amp_enabled = self.amp_enabled

        for epoch in range(self.cfg.n_epochs):
            epoch_loss = epoch_policy = epoch_value = epoch_kl_abs = epoch_entropy = epoch_reward = 0.0
            n_batches = 0

            for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
                # schedule ε
                current_update = epoch * len(dataloader) + n_batches + 1
                frac = current_update / max(1, total_updates)
                self.cfg.clip_epsilon = initial_clip * (1 - frac) + final_clip * frac
                step += 1

                batch_inputs = {
                    "input_ids": batch["input_ids"].to(self.device),
                    "attention_mask": batch["attention_mask"].to(self.device),
                }

                # 1) Generate
                completions = self.generate(batch_inputs)
                tok = self.tokenize_clean(completions)
                clean_completions = self.tokenizer.batch_decode(tok["input_ids"], skip_special_tokens=True)

                # 2) Concat prompt+completion
                full_ids  = torch.cat([batch_inputs["input_ids"], tok["input_ids"]], dim=1)
                full_attn = torch.cat([batch_inputs["attention_mask"], tok["attention_mask"]], dim=1)
                Tp = batch_inputs["input_ids"].size(1)
                Tc = tok["input_ids"].size(1)
                T_full = full_ids.size(1)

                prompt_lens = batch_inputs["attention_mask"].sum(dim=1).long()
                B = full_ids.size(0)
                rows = torch.arange(B, device=self.device).unsqueeze(-1)
                cols = prompt_lens.unsqueeze(-1) + torch.arange(Tc, device=self.device).unsqueeze(0)

                # assert-uri de siguranță pentru slicing:
                assert T_full == Tp + Tc, f"T_full={T_full} vs Tp+Tc={Tp+Tc}"
                assert (prompt_lens >= 0).all() and (prompt_lens <= Tp).all(), "prompt_lens out of [0,Tp]"
                max_col = (prompt_lens + Tc - 1).max().item()
                assert max_col < T_full, f"slice col {max_col} >= T_full {T_full}"

                # 3) OLD & REF forward o singură dată, calculăm LOGPROBS CHUNKED și eliberăm logits
                with torch.inference_mode():
                    with autocast(device_type=device_type, dtype=self.autocast_dtype, enabled=amp_enabled):
                        old_logits_full, old_values_full = self.policy_model(full_ids, full_attn, labels=None)
                        ref_logits_full = self.policy_ref(input_ids=full_ids, attention_mask=full_attn).logits

                old_logits_gen = old_logits_full[rows, cols]
                ref_logits_gen = ref_logits_full[rows, cols]

                labels_comp = tok["input_ids"].clone()
                labels_comp[labels_comp == self.tokenizer.pad_token_id] = -100

                V = old_logits_full.size(-1)
                ok_vocab = (labels_comp[labels_comp >= 0] < V).all()
                assert ok_vocab, f"labels conțin id >= vocab (V={V})"

                # logprobs chunked + mask
                old_logp_tok, gen_mask = self.token_logprobs_chunked(old_logits_gen, labels_comp, chunk=128)
                ref_logp_tok, _        = self.token_logprobs_chunked(ref_logits_gen, labels_comp, chunk=128)

                # eliberează logits masive asap
                del old_logits_full, ref_logits_full, old_logits_gen, ref_logits_gen
                torch.cuda.empty_cache()

                # 4) Rewards: RM + custom apoi shaping pe TOKEN
                raw_sentiment_rewards = self.compute_sentiment_reward(
                    completions,
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
                raw_rewards = raw_sentiment_rewards + raw_fn_rewards
                rewards_seq = (raw_rewards - raw_rewards.mean()) / (raw_rewards.std() + 1e-8)
                rewards_seq = rewards_seq.clamp(-5.0, 5.0)  # [B], dtype fp32

                # Reward pe ultimul token (indexare 1D + dtype corect la MB)
                Bc, Tc_check = gen_mask.shape
                assert Tc_check == Tc, "mismatch Tc"
                batch_index = torch.arange(Bc, device=self.device)
                gen_counts = gen_mask.sum(dim=1).to(torch.long)
                valid = gen_counts > 0
                last_idx = (gen_counts - 1).clamp(min=0, max=Tc - 1)
                rewards_tok_fp32 = torch.zeros(Bc, Tc, device=self.device, dtype=torch.float32)
                rewards_tok_fp32[batch_index[valid], last_idx[valid]] = rewards_seq[valid]

                token_shaping = self.build_token_shaping(
                    labels_comp=labels_comp,
                    gen_mask=gen_mask,
                    completions_decoded=clean_completions,
                    answers=batch["solution"]
                ).to(torch.float32)
                rewards_tok_fp32 = rewards_tok_fp32 + token_shaping  # [B,Tc] fp32

                # Dones
                dones_tok = torch.zeros(Bc, Tc, device=self.device, dtype=torch.float32)
                dones_tok[batch_index[valid], last_idx[valid]] = 1.0

                # 5) Micro-batching
                ppo_epochs = getattr(self.cfg, "ppo_epochs", 4)
                seq_lens = gen_mask.sum(dim=1).float()
                avg_len = seq_lens.mean().item()
                microbatch_size = min(
                    self.cfg.microbatch_size,
                    max(1, int(self.cfg.token_microbatch_size / max(1.0, avg_len)))
                )
                accum_steps = getattr(self.cfg, "accum_steps", 1)
                Bcur = Bc
                if microbatch_size is None or microbatch_size >= Bcur:
                    microbatch_size = Bcur

                def ppo_update_pass_tokens(
                    full_ids, full_attn, rows, cols,
                    old_values_full,            # [B,T_full]
                    old_logp_tok, ref_logp_tok, # [B,Tc]
                    labels_comp, gen_mask,      # [B,Tc], [B,Tc]
                    rewards_tok_fp32,           # [B,Tc] fp32
                    dones_tok                   # [B,Tc] fp32
                ):
                    mb_losses = []; mb_pols = []; mb_vals = []; mb_kls = []; mb_ents = []
                    mb_count = 0
                    self.optimizer.zero_grad(set_to_none=True)

                    for start in range(0, Bcur, microbatch_size):
                        end = min(start + microbatch_size, Bcur)
                        mb = slice(start, end)

                        # NEW forward pe micro-batch (full context)
                        with autocast(device_type=device_type, dtype=self.autocast_dtype, enabled=amp_enabled):
                            mb_new_logits_full, mb_new_values_full = self.policy_model(full_ids[mb], full_attn[mb], labels=None)

                        # Taie completarea
                        mb_rows = rows[mb]
                        mb_cols = cols[mb]
                        mb_new_logits = mb_new_logits_full[mb_rows, mb_cols]   # [mb, Tc, V]
                        mb_new_values = mb_new_values_full[mb_rows, mb_cols]   # [mb, Tc]
                        mb_old_values = old_values_full[mb_rows, mb_cols]      # [mb, Tc]

                        # Logprobs NEW (chunked) + eliberează logits asap
                        mb_new_logp, _ = self.token_logprobs_chunked(mb_new_logits, labels_comp[mb], chunk=128)

                        # Pregătește rewards/next/dones local, cu dtype corect
                        mb_rewards = rewards_tok_fp32[mb].to(mb_new_values.dtype)
                        mb_dones   = dones_tok[mb].to(mb_new_values.dtype)

                        mb_next_vals = torch.zeros_like(mb_new_values)
                        mb_next_vals[:, :-1] = mb_new_values[:, 1:]

                        with autocast(device_type=device_type, dtype=self.autocast_dtype, enabled=amp_enabled):
                            loss, pol_loss, val_loss, kl_abs, ent = self.ppo_loss_tokens(
                                old_logp_tok[mb], mb_new_logp, ref_logp_tok[mb],
                                mb_rewards, mb_new_values, mb_old_values,
                                mask_tok=gen_mask[mb],
                                next_values_tok=mb_next_vals, dones_tok=mb_dones,
                                entropy_tok_mean=None  # fără entropie (0)
                            )
                            loss = loss / accum_steps

                        # Free logits imediat
                        del mb_new_logits_full, mb_new_logits
                        torch.cuda.empty_cache()

                        self.scaler.scale(loss).backward()
                        mb_count += 1
                        mb_losses.append(loss.item() * accum_steps)
                        mb_pols.append(pol_loss.item()); mb_vals.append(val_loss.item())
                        mb_kls.append(kl_abs.item());   mb_ents.append(ent.item())

                        if (mb_count % accum_steps == 0) or (end == Bcur):
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            if self.scheduler is not None:
                                self.scheduler.step()
                            self.optimizer.zero_grad(set_to_none=True)

                    return (
                        torch.tensor(mb_losses).mean(),
                        torch.tensor(mb_pols).mean(),
                        torch.tensor(mb_vals).mean(),
                        torch.tensor(mb_kls).mean(),
                        torch.tensor(mb_ents).mean()
                    )

                # 6) Update
                last_loss = last_pol = last_val = last_kl_abs = last_ent = None
                for _ in range(ppo_epochs):
                    losses, pols, vals, kls_abs, ents = ppo_update_pass_tokens(
                        full_ids, full_attn, rows, cols,
                        old_values_full,
                        old_logp_tok, ref_logp_tok,
                        labels_comp, gen_mask,
                        rewards_tok_fp32, dones_tok
                    )
                    last_loss, last_pol, last_val, last_kl_abs, last_ent = losses, pols, vals, kls_abs, ents

                    # adaptive KL
                    kl_abs_val = float(last_kl_abs.item())
                    self._kl_running.append(kl_abs_val)
                    if kl_abs_val > self.cfg.kl_stop_factor * self.cfg.target_kl:
                        break
                    if len(self._kl_running) >= self.cfg.kl_window:
                        kl_avg = sum(self._kl_running) / len(self._kl_running)
                        if kl_avg > 2.0 * self.cfg.target_kl:
                            self.cfg.kl_coef = min(self.cfg.kl_coef * self.cfg.kl_adapt_rate, self.cfg.max_kl_coef)
                        elif kl_avg < 0.5 * self.cfg.target_kl:
                            self.cfg.kl_coef = max(self.cfg.kl_coef / self.cfg.kl_adapt_rate, self.cfg.min_kl_coef)
                        self.writer.add_scalar("KL/window_avg_abs_tok", kl_avg, step)
                        self.writer.add_scalar("KL/adapted_coef", self.cfg.kl_coef, step)
                        self._kl_running.clear()

                # 7) Logging & stats
                epoch_loss    += float(last_loss.item())
                epoch_policy  += float(last_pol.item())
                epoch_value   += float(last_val.item())
                epoch_kl_abs  += float(last_kl_abs.item())
                epoch_entropy += float(last_ent.item())  # 0
                epoch_reward  += float(rewards_seq.mean().item())
                n_batches     += 1

                if step % getattr(self.cfg, "log_every", 50) == 0:
                    avg_loss = epoch_loss / n_batches
                    avg_pol  = epoch_policy / n_batches
                    avg_val  = epoch_value / n_batches
                    avg_kl   = epoch_kl_abs / n_batches
                    avg_ent  = epoch_entropy / n_batches
                    avg_r    = epoch_reward / n_batches

                    tqdm.write(
                        f"[Epoch {epoch}] Step {step} | Loss: {avg_loss:.6f} | Policy: {avg_pol:.6f} | "
                        f"Value: {avg_val:.6f} | KL_ref(tok): {avg_kl:.6f} | Entropy: {avg_ent:.6f} | Reward(seq): {avg_r:.6f}"
                    )

                    self.writer.add_scalar("Loss/total_tok",   avg_loss, step)
                    self.writer.add_scalar("Loss/policy_tok",  avg_pol,  step)
                    self.writer.add_scalar("Loss/value_tok",   avg_val,  step)
                    self.writer.add_scalar("Stats/KL_ref_abs_tok", avg_kl, step)
                    self.writer.add_scalar("Stats/entropy_tok",avg_ent,  step)
                    self.writer.add_scalar("Reward/mean_seq",  avg_r,    step)

            # checkpoint
            avg_epoch_reward = (epoch_reward / n_batches) if n_batches > 0 else 0.0
            if avg_epoch_reward > getattr(self, "best_reward", -1e9):
                self.best_reward = avg_epoch_reward
                self.policy_model.save_pretrained(f"{self.cfg.save_dir}_best")
                print(f"✅ Model saved at epoch {epoch} with improved reward {avg_epoch_reward:.3f}")
            if (epoch + 1) % getattr(self.cfg, "save_every", 1) == 0 or epoch == 0:
                self.policy_model.save_pretrained(f"{self.cfg.save_dir}_epoch_{epoch}")
                print(f"✅ Model saved at epoch {epoch}")
