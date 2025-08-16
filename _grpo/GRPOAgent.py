# ===================== GRPOAgent (token-level, fixed) =====================
import copy
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Callable
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
from collections import deque


class GRPOAgent:
    """
    Group-Relative Policy Optimization (GRPO), token-level:
      - K mostre/completări per prompt (group sampling)
      - baseline de grup + normalizare pe grup => avantaj pe secvență, împărțit pe tokeni
      - concatenare [prompt || completion], labels=-100 pe prompt (optimizăm DOAR completarea)
      - policy_old înghețată per batch de update-uri (PPO-style)
      - forward-KL(π_new || π_ref) din logits, doar pe tokenii completării
      - adaptive KL coef + early stop la depășirea target_kl
      - AMP, micro-batching, grad clip, logging
      - replay buffer DOAR pentru reward model (fără policy replay neinformat)
    """

    def __init__(
        self,
        policy_model,                  # HF causal LM (produce .logits)
        policy_ref,                    # model de referință (eval only, .logits)
        tokenizer,                     # HF tokenizer (pad_token existent)
        optimizer,
        scheduler,
        config,
        reward_functions: List[Callable[[List[str]], torch.Tensor]],  # fiecare fn(list[str]) -> Tensor[B] pe device/CPU
        reward_model: Optional[object] = None,   # opțional, pentru batch_score sau fine-tuning separat
    ):
        self.cfg = config
        self.device = config.device

        # Seeds & CuDNN
        random.seed(config.seed); np.random.seed(config.seed); torch.manual_seed(config.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(config.seed)
        cudnn.deterministic = False; cudnn.benchmark = True

        # Models
        self.policy_model = policy_model.to(self.device).train()
        self.policy_ref   = policy_ref.to(self.device).eval()
        self.policy_old   = copy.deepcopy(self.policy_model).to(self.device).eval()
        self.reward_model = reward_model

        # Gradient checkpointing (dacă există)
        if hasattr(self.policy_model, "gradient_checkpointing_enable"):
            self.policy_model.gradient_checkpointing_enable()

        # Opt & AMP
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler    = GradScaler(enabled=torch.cuda.is_available())

        # Tokenizer
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # Logging
        os.makedirs(config.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=config.log_dir)
        self.step_log_path = os.path.join(config.log_dir, "step_logs.txt")
        open(self.step_log_path, "w").close()

        # Replay buffer (pt. reward model)
        self.replay_buffer = deque(maxlen=config.buffer_size)
        self.reward_functions = reward_functions

        # Clip scheduling
        self.initial_clip_epsilon = getattr(config, "clip_epsilon", 0.2)
        self.final_clip_epsilon   = getattr(config, "clip_epsilon_final", self.initial_clip_epsilon)

        # Adaptive KL
        self._kl_running = []
        self.best_reward = -float("inf")

    # -------------------- Helperi de tokenizare + concat --------------------

    def _concat_prompt_and_completion(
        self,
        prompt_ids: torch.Tensor,           # [B, Tp]
        prompt_attn: torch.Tensor,          # [B, Tp]
        completions: List[str],
        max_len: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Construiește [prompt || completion], labels=-100 pe prompt, attention_mask corect,
        și întoarce input_ids, attention_mask, labels, plus mască 'active' pentru tokenii validați după shift.

        Vom construi:
           merged = [pad ... pad, prompt_tokens, completion_tokens(+eos)]
           labels = merged.clone(); labels[:prompt_len] = -100
        'active' se derivă ulterior după shift în compute_*.
        """
        device = self.device
        B = prompt_ids.size(0)
        max_len = max_len or self.cfg.policy_max_length
        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id

        comp_tok = self.tokenizer(
            completions, add_special_tokens=False, return_tensors=None
        )["input_ids"]

        out_input_ids, out_attn, out_labels = [], [], []

        for i in range(B):
            # lungimea promptului (numărul de 1-uri în attn)
            plen = int(prompt_attn[i].sum().item())
            p_ids = prompt_ids[i, -plen:].to(device)  # strip left-padding

            cids = comp_tok[i]
            if eos_id is not None and (len(cids) == 0 or cids[-1] != eos_id):
                cids = cids + [eos_id]
            c_t = torch.tensor(cids, dtype=torch.long, device=device)

            merged = torch.cat([p_ids, c_t], dim=0)          # [Tp + Tc]
            labels = merged.clone()
            labels[:plen] = -100                              # optimizăm DOAR pe completare

            # Crop de la dreapta dacă depășește max_len
            if merged.size(0) > max_len:
                start = merged.size(0) - max_len
                merged = merged[start:]
                labels = labels[start:]
                # atenție: la crop putem tăia din prompt; labels -100 se păstrează corect

            # Pad la stânga până la max_len
            pad_needed = max_len - merged.size(0)
            if pad_needed > 0:
                pad_vec = torch.full((pad_needed,), pad_id, device=device, dtype=torch.long)
                merged = torch.cat([pad_vec, merged], dim=0)
                labels = torch.cat([torch.full((pad_needed,), -100, device=device, dtype=torch.long), labels], dim=0)

            # attention_mask by position
            attn = torch.zeros_like(merged, dtype=torch.long, device=device)
            attn[pad_needed:] = 1

            out_input_ids.append(merged)
            out_labels.append(labels)
            out_attn.append(attn)

        return {
            "input_ids": torch.stack(out_input_ids, dim=0),     # [B, T]
            "attention_mask": torch.stack(out_attn, dim=0),     # [B, T]
            "labels": torch.stack(out_labels, dim=0),           # [B, T]
        }

    # -------------------- Compute-uri token-level (shift corect) --------------------

    @staticmethod
    def _shift_for_causal(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pentru LM cauzal: logits[:, :-1] prezic labels[:, 1:].
        Returnează (shift_logits, shift_labels).
        """
        return logits[:, :-1, :], labels[:, 1:]

    def token_logprobs(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Log-prob pe token (shape [B, T-1]) cu -inf pe pozițiile inactive (labels==-100).
        """
        shift_logits, shift_labels = self._shift_for_causal(logits, labels)
        logp = F.log_softmax(shift_logits, dim=-1)                    # [B, T-1, V]
        B, Tm1, V = logp.shape
        # pregătim -inf
        out = torch.full((B, Tm1), float("-inf"), device=logp.device, dtype=logp.dtype)
        active = shift_labels.ne(-100)
        if active.any():
            gathered = logp.gather(-1, shift_labels.masked_fill(~active, 0).unsqueeze(-1)).squeeze(-1)  # [B, T-1]
            out[active] = gathered[active]
        return out  # [B, T-1]

    def forward_token_kl(self, logits_new: torch.Tensor, logits_ref: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward KL(π_new || π_ref) pe tokenii de completare, calculat din logits.
        Întoarce [B] (medie pe tokenii activi ai fiecărei secvențe).
        """
        shift_new, shift_labels = self._shift_for_causal(logits_new, labels)
        shift_ref, _            = self._shift_for_causal(logits_ref,  labels)

        logp_new = F.log_softmax(shift_new, dim=-1)       # [B, T-1, V]
        logp_ref = F.log_softmax(shift_ref, dim=-1)       # [B, T-1, V]
        p_new    = logp_new.exp()
        kl_tok   = (p_new * (logp_new - logp_ref)).sum(dim=-1)  # [B, T-1]

        active = shift_labels.ne(-100).float()            # 1 doar pe tokenii de completare (targets)
        tok_count = active.sum(dim=1).clamp_min(1.0)      # [B]
        kl_seq = (kl_tok * active).sum(dim=1) / tok_count
        return kl_seq  # [B]

    def entropy_on_completion(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Entropie mediată DOAR pe tokenii de completare (după shift).
        """
        shift_logits, shift_labels = self._shift_for_causal(logits, labels)
        log_probs = F.log_softmax(shift_logits, dim=-1)      # [B, T-1, V]
        probs     = log_probs.exp()
        ent_tok   = -(probs * log_probs).sum(dim=-1)         # [B, T-1]
        active    = shift_labels.ne(-100).float()
        tok_count = active.sum(dim=1).clamp_min(1.0)
        return (ent_tok * active).sum(dim=1) / tok_count     # [B]

    # -------------------- Generare K completări per prompt --------------------

    def generate(self, prompts: List[str]) -> List[str]:
        """
        Generează o completare per prompt (sampling).
        """
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        amp = torch.cuda.is_available()
        with torch.no_grad():
            with autocast(device_type="cuda" if amp else "cpu", enabled=amp):
                gen = self.policy_model.generate(
                    **inputs,
                    max_new_tokens=self.cfg.max_new_tokens,
                    do_sample=True,
                    top_p=self.cfg.top_p,
                    temperature=self.cfg.temperature,
                    repetition_penalty=getattr(self.cfg, "repetition_penalty", 1.0),
                    pad_token_id=self.tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                )
        gen_only = gen.sequences[:, inputs["input_ids"].size(1):]
        return self.tokenizer.batch_decode(gen_only, skip_special_tokens=True)

    # -------------------- Rewards pe grup --------------------

    def compute_group_rewards(self, completions: List[str]) -> torch.Tensor:
        """
        Agregă o listă de funcții de reward. Fiecare fn: list[str] -> Tensor[B] (pe CPU sau device).
        Returnează Tensor[B] pe device.
        """
        parts = []
        for fn in self.reward_functions:
            r = fn(completions)
            if not isinstance(r, torch.Tensor):
                r = torch.tensor(r)
            parts.append(r.to(self.device, dtype=torch.float32))
        return torch.stack(parts, dim=0).sum(dim=0)  # [B]

    def _update_reward_model(self):
        """
        Ex. update periodic al reward modelului pe (prompt, completion, reward) din buffer.
        Aici doar scheletul; implementați după nevoi.
        """
        if self.reward_model is None:
            return
        bs = getattr(self.cfg, "reward_batch_size", 0)
        if bs <= 0 or len(self.replay_buffer) < bs:
            return
        batch = random.sample(self.replay_buffer, bs)
        prompts, comps, rewards = zip(*batch)
        # TODO: implement training pentru reward_model pe (prompts, comps, rewards)

    # -------------------- Train --------------------

    def train(self, dataloader):
        cfg = self.cfg
        total_updates = max(1, cfg.n_epochs * len(dataloader) * max(1, cfg.update_epochs))
        update_count = 0

        # snapshot inițial
        self.policy_old.load_state_dict(self.policy_model.state_dict(), strict=False)
        print(f"Training for {cfg.n_epochs} epochs (~{total_updates} updates)...")

        for epoch in range(cfg.n_epochs):
            epoch_reward = 0.0
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
                prompts: List[str] = batch["text"]

                # === 1) Group sampling: K completări / prompt ===
                groups: List[List[str]] = [self.generate(prompts) for _ in range(cfg.group_size)]  # list len K of [B] lists

                # === 2) Rewards pe grup + baseline / std per prompt ===
                # raw: [K, B]
                raw = torch.stack([self.compute_group_rewards(g) for g in groups], dim=0)  # [K, B]
                baseline = raw.mean(dim=0)                                 # [B]
                std = raw.std(dim=0, unbiased=False).clamp_min(1e-8)      # [B]
                advantage = (raw - baseline.unsqueeze(0)) / std.unsqueeze(0)  # [K, B]

                # logging + replay pt. reward model
                for k in range(cfg.group_size):
                    for j, prompt in enumerate(prompts):
                        self.replay_buffer.append((prompt, groups[k][j], float(raw[k, j].item())))
                if getattr(cfg, "reward_update_every", 0) and (update_count % cfg.reward_update_every == 0):
                    self._update_reward_model()

                # === 3) Tokenizare prompturi (o singură dată) ===
                prompt_tok = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
                prompt_ids = prompt_tok["input_ids"].to(self.device)        # [B, Tp]
                prompt_attn = prompt_tok["attention_mask"].to(self.device)  # [B, Tp]

                # === 4) Construiește batch-uri [prompt||completion] pe fiecare grup ===
                cat_list = []
                for k in range(cfg.group_size):
                    cat = self._concat_prompt_and_completion(prompt_ids, prompt_attn, groups[k], max_len=cfg.policy_max_length)
                    cat_list.append(cat)

                # Aplatizează peste K (concatenate pe dim batch): [K*B, T]
                input_ids = torch.cat([c["input_ids"] for c in cat_list], dim=0)
                attn      = torch.cat([c["attention_mask"] for c in cat_list], dim=0)
                labels    = torch.cat([c["labels"] for c in cat_list], dim=0)
                flat_adv  = advantage.transpose(0,1).reshape(-1)  # [B, K] -> [B*K] în aceeași ordine ca concatenarea?

                # NOTĂ: cat_list este [k=0..K-1] concatenate; advantage este [K,B].
                # După concatenare pe k, batch-ul este: (k=0, j=0..B-1), (k=1, j=0..B-1), ...
                # advantage.view(-1) ordonează (K,B) ca (k major, b minor)? Torch view folosește C-order (row-major):
                # raw/advantage au shape [K,B], .reshape(-1) => (k=0, b=0..B-1), (k=1, b=0..B-1)... -> corespunzător.

                # === 5) Îngheață policy_old pentru următoarele update-uri pe acest batch ===
                self.policy_old.load_state_dict(self.policy_model.state_dict(), strict=False)

                # === 6) Update-uri μ (token-level PPO pe completare) ===
                ppo_epochs = max(1, getattr(cfg, "update_epochs", 1))

                # micro-batching adaptiv după lungimea medie
                seq_lens = attn.sum(dim=1).float()
                avg_len = max(1.0, float(seq_lens.mean().item()))
                token_budget = max(1, int(getattr(cfg, "token_microbatch_size", 2048)))
                micro = min(getattr(cfg, "microbatch_size", input_ids.size(0)), max(1, int(token_budget / avg_len)))
                accum = max(1, getattr(cfg, "accum_steps", 1))

                for u in range(ppo_epochs):
                    frac = update_count / max(1, total_updates)
                    eps = self.initial_clip_epsilon * (1 - frac) + self.final_clip_epsilon * frac

                    self.optimizer.zero_grad(set_to_none=True)
                    running_loss = []; running_kl = []; running_pol = []

                    for start in range(0, input_ids.size(0), micro):
                        end = min(start + micro, input_ids.size(0))
                        ids_mb   = input_ids[start:end]
                        attn_mb  = attn[start:end]
                        labels_mb= labels[start:end]
                        A_mb     = flat_adv[start:end].to(self.device)    # [mb]
                        A_mb = (A_mb - A_mb.mean()) / (A_mb.std() + 1e-8)  # whitening pe minibatch (stabil)

                        amp = torch.cuda.is_available()
                        with autocast(device_type="cuda" if amp else "cpu", enabled=amp):
                            logits_new = self.policy_model(ids_mb, attention_mask=attn_mb).logits
                            with torch.no_grad():
                                logits_old = self.policy_old(ids_mb, attention_mask=attn_mb).logits
                                logits_ref = self.policy_ref(ids_mb, attention_mask=attn_mb).logits

                            # --- policy ratio pe token (doar completarea) ---
                            logp_new_tok = self.token_logprobs(logits_new, labels_mb)  # [mb, T-1]
                            logp_old_tok = self.token_logprobs(logits_old, labels_mb)  # [mb, T-1]
                            active_tok   = (labels_mb[:, 1:] != -100).float()          # [mb, T-1], 1 doar pe completare (targets)
                            tok_count    = active_tok.sum(dim=1).clamp_min(1.0)        # [mb]

                            # distribuim avantajul pe tokeni activi
                            A_tok = (A_mb[:, None] * active_tok) / tok_count[:, None]

                            delta_tok = (logp_new_tok - logp_old_tok).clamp(-20.0, 20.0)
                            ratio_tok = torch.exp(delta_tok)

                            surr1 = ratio_tok * A_tok
                            surr2 = torch.clamp(ratio_tok, 1 - eps, 1 + eps) * A_tok
                            policy_term = torch.min(surr1, surr2)

                            policy_loss_mb = - (policy_term * active_tok).sum(dim=1) / tok_count
                            policy_loss_mb = policy_loss_mb.mean()

                            # --- forward KL pe tokenii completării ---
                            kl_seq = self.forward_token_kl(logits_new, logits_ref, labels_mb)  # [mb]
                            kl_val = kl_seq.mean()

                            # --- entropie (opțional) ---
                            if getattr(self.cfg, "entropy_weight", 0.0) != 0.0:
                                ent_seq = self.entropy_on_completion(logits_new, labels_mb).mean()
                            else:
                                ent_seq = torch.tensor(0.0, device=self.device, dtype=logits_new.dtype)

                            loss = policy_loss_mb + self.cfg.kl_coef * kl_val - self.cfg.entropy_weight * ent_seq

                        self.scaler.scale(loss / accum).backward()
                        running_loss.append(float(loss.item()))
                        running_kl.append(float(kl_val.item()))
                        running_pol.append(float(policy_loss_mb.item()))

                        if ((start // micro) + 1) % accum == 0 or end == input_ids.size(0):
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            if self.scheduler is not None:
                                self.scheduler.step()
                            self.optimizer.zero_grad(set_to_none=True)

                    update_count += 1
                    batch_R = float(raw.mean().item())
                    epoch_reward += batch_R

                    # ---- Adaptive KL + early stop ----
                    kl_batch_mean = float(np.mean(running_kl)) if running_kl else 0.0
                    self._kl_running.append(kl_batch_mean)
                    # early stop pe update-urile curente
                    if kl_batch_mean > self.cfg.kl_stop_factor * self.cfg.target_kl:
                        pass  # ieșim din bucla de ppo_epochs mai jos
                    # adaptare coeficient pe fereastră
                    if len(self._kl_running) >= getattr(self.cfg, "kl_window", 8):
                        kl_avg = sum(self._kl_running) / len(self._kl_running)
                        if kl_avg > 2.0 * self.cfg.target_kl:
                            self.cfg.kl_coef = min(self.cfg.kl_coef * self.cfg.kl_adapt_rate, self.cfg.max_kl_coef)
                        elif kl_avg < 0.5 * self.cfg.target_kl:
                            self.cfg.kl_coef = max(self.cfg.kl_coef / self.cfg.kl_adapt_rate, self.cfg.min_kl_coef)
                        self.writer.add_scalar("KL/window_avg_forward", kl_avg, update_count)
                        self.writer.add_scalar("KL/adapted_coef", self.cfg.kl_coef, update_count)
                        self._kl_running.clear()

                    # logging periodic
                    if update_count % getattr(cfg, "log_every", 50) == 0:
                        avg_loss = float(np.mean(running_loss)) if running_loss else 0.0
                        avg_pol  = float(np.mean(running_pol))  if running_pol  else 0.0
                        with open(self.step_log_path, "a") as f:
                            f.write(f"Upd {update_count} | Loss {avg_loss:.4f} | Pol {avg_pol:.4f} | KL {kl_batch_mean:.4f} | R {batch_R:.4f}\n")
                        self.writer.add_scalar("GRPO/Loss",   avg_loss,      update_count)
                        self.writer.add_scalar("GRPO/Policy", avg_pol,       update_count)
                        self.writer.add_scalar("GRPO/KL",     kl_batch_mean, update_count)
                        self.writer.add_scalar("GRPO/Reward", batch_R,       update_count)

                    # early stop pe ppo_epochs dacă KL a sărit mult
                    if kl_batch_mean > self.cfg.kl_stop_factor * self.cfg.target_kl:
                        break

                # === end μ updates ===

            # === end epoch: checkpoint ===
            avg_R = epoch_reward / max(1, update_count)
            print(f"Epoch {epoch+1} done | AvgR {avg_R:.4f}")
            if avg_R > self.best_reward:
                self.best_reward = avg_R
                print(f"New best reward {avg_R:.4f}, saving model...")
                self.policy_model.save_pretrained(f"{cfg.save_dir}_best")
            if (epoch + 1) % getattr(cfg, "save_every", 1) == 0 or epoch == 0:
                self.policy_model.save_pretrained(f"{cfg.save_dir}_epoch_{epoch+1}")