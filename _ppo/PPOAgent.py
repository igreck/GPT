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
    """Proximal Policy Optimization agent with mixed-precision, micro-batching, and advanced logging."""

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
        # For dynamic clipping schedule
        self.initial_clip_epsilon = getattr(self.cfg, "clip_epsilon", None)
        self.final_clip_epsilon   = getattr(self.cfg, "clip_epsilon_final", self.initial_clip_epsilon)
        self.device = config.device
        self.scheduler = scheduler
  
        # Reproducibility
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        os.environ["PYTHONHASHSEED"] = str(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.cfg.seed)
        # allow non-deterministic algorithms for speed
        torch.backends.cudnn.deterministic = False
        # enable CuDNN benchmark for fixed input shapes to improve throughput
        cudnn.benchmark = True
        # AMP / dtype setup (once)
        self.amp_enabled = torch.cuda.is_available()
        self.autocast_dtype = torch.bfloat16 if (self.amp_enabled and torch.cuda.is_bf16_supported()) else None
        if self.amp_enabled:
            torch.set_float32_matmul_precision("high")
        # experiențe pentru off-policy replay
        self.replay_buffer = deque(maxlen=self.cfg.buffer_size)

        # Models
        self.policy_model = policy_model.to(self.device).train()
        self.policy_ref = policy_ref.to(self.device).eval()
        self.reward_model = reward_model.to(self.device).eval()

        # Reward-model positive/negative indices
        labels_map = getattr(self.reward_model.config, "id2label", {0:"NEGATIVE",1:"POSITIVE"})
        inv = {v.lower(): k for k, v in labels_map.items()}
        self.pos_idx = inv.get("positive", 1)
        self.neg_idx = inv.get("negative", 0 if self.pos_idx != 0 else 1)

        # AMP scaler and TensorBoard
        self.scaler = GradScaler()
        self.writer = SummaryWriter(log_dir=self.cfg.log_dir)
        self.best_reward = -float("inf")
        self.no_improve_epochs = 0
        self._kl_running = []  # fereastră pentru adaptive KL (pe |Δ|)

        # Enable gradient checkpointing if available
        if hasattr(self.policy_model, "model") and hasattr(self.policy_model.model, "gradient_checkpointing_enable"):
            self.policy_model.model.gradient_checkpointing_enable()

        # Asigură-te că există pad_token și padding_left
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.reward_tokenizer = reward_tokenizer

        self.rfn = RewardFunctions(self.tokenizer, self.device)

    # ---- Core utils ----
    def compute_logprobs(self, logits: torch.Tensor, labels: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            mask = (labels != -100).float()
        per_token = -F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view(labels.size())
        denom = mask.sum(dim=1).clamp_min(1)
        return (per_token * mask).sum(dim=1) / denom

    def compute_sentiment_reward(self, texts, use_margin=False, add_length_bonus=True, min_words=6, bonus=0.05):
        # Tokenizare robustă pentru reward model (max_length separat)
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

            # diversity bonus based on unique bigram ratio
            def overlap_ratio(text):
                toks = text.split()
                bigrams = [(toks[i], toks[i+1]) for i in range(len(toks)-1)]
                uniq = set(bigrams)
                return 1.0 - len(uniq) / max(1, len(bigrams))
            divers = torch.tensor([overlap_ratio(t) for t in texts], device=self.device)
            raw = (raw + divers * self.cfg.diversity_coef).clamp(0.0, 1.0)
            raw = raw * 2.0 - 1.0
            raw = raw.clamp(-1.0, 1.0)
        return raw  # [B]

    def compute_entropy(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        ent = -(probs * log_probs).sum(dim=-1)
        denom = mask.sum(dim=1).clamp_min(1)
        return (ent * mask).sum(dim=1) / denom

    def compute_kl_divergence(self, new_logprobs: torch.Tensor, ref_logprobs: torch.Tensor) -> torch.Tensor:
        """Δlogp = E[log π_new - log π_ref] (medie pe token). Poate fi < 0."""
        return (new_logprobs - ref_logprobs).mean()

    def compute_advantages(self, rewards: torch.Tensor, values: torch.Tensor, next_values: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
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

    def ppo_loss(
        self,
        old_policy_logprobs: torch.Tensor,   # logπ_old (no-grad)
        new_logprobs: torch.Tensor,          # logπ_new
        ref_logprobs: torch.Tensor,          # logπ_ref (no-grad)
        rewards: torch.Tensor,
        values: torch.Tensor,
        old_values: torch.Tensor,
        logits: torch.Tensor,
        mask: torch.Tensor,
        next_values: Optional[torch.Tensor] = None,
        dones: Optional[torch.Tensor] = None
    ) -> tuple:
        # GAE sau 1-step
        if next_values is not None and dones is not None:
            advantages = self.compute_advantages(rewards, values, next_values, dones)
        else:
            advantages = rewards - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Entropy & KL_ref (medie pe token)
        entropy = self.compute_entropy(logits, mask.float()).mean()

        delta_logp   = new_logprobs - ref_logprobs
        kl_ref_signed = delta_logp.mean()         # pentru logging
        kl_ref_abs    = delta_logp.abs().mean()   # pentru penalizare

        # PPO clipping
        delta_for_ratio = torch.clamp(new_logprobs - old_policy_logprobs, min=-20.0, max=20.0)
        ratio = torch.exp(delta_for_ratio)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.cfg.clip_epsilon, 1 + self.cfg.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value clipping
        clipped_values = old_values + torch.clamp(
            values - old_values,
            -self.cfg.value_clip_range,
            self.cfg.value_clip_range
        )
        unclipped_loss = F.mse_loss(values, rewards)
        clipped_loss = F.mse_loss(clipped_values, rewards)
        value_loss = torch.max(unclipped_loss, clipped_loss)

        total_loss = (
            policy_loss
            + self.cfg.value_loss_weight * value_loss
            + self.cfg.kl_coef * kl_ref_abs     # penalizează magnitudinea deviației de la ref
            - self.cfg.entropy_weight * entropy
        )

        return total_loss, policy_loss, value_loss, kl_ref_signed, entropy, kl_ref_abs

    def tokenize_clean(self, texts: List[str], prefix_lens: Optional[List[int]] = None) -> Dict[str, torch.Tensor]:
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
        if prefix_lens is not None:
            for i, plen in enumerate(prefix_lens):
                labels[i, :plen] = -100
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
    
    # ---- Loop de antrenare ----
    def train(self, dataloader):
        step = 0
        # Compute total number of updates for clip epsilon scheduling
        total_updates = self.cfg.n_epochs * len(dataloader)
        initial_clip = self.initial_clip_epsilon
        final_clip   = self.final_clip_epsilon
        if torch.cuda.is_available():
            device_type, amp_enabled = "cuda", True
        else:
            device_type, amp_enabled = "cpu", False 

        for epoch in range(self.cfg.n_epochs):
            epoch_loss = epoch_policy = epoch_value = epoch_kl = epoch_entropy = epoch_reward = 0.0
            n_batches = 0

            for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
                # Update clip_epsilon linearly from initial to final over total updates
                current_update = epoch * len(dataloader) + n_batches + 1
                frac = current_update / total_updates
                self.cfg.clip_epsilon = initial_clip * (1 - frac) + final_clip * frac
                step += 1
                batch_inputs = {
                    "input_ids": batch["input_ids"].to(self.device),
                    "attention_mask": batch["attention_mask"].to(self.device),
                }

                # 1) Generează DOAR completările și tokenizează fără prefix masking
                completions = self.generate(batch_inputs)
                tok = self.tokenize_clean(completions, prefix_lens=None)
                clean_completions = self.tokenizer.batch_decode(tok["input_ids"], skip_special_tokens=True)  # (nefolosit)
                input_ids, attention_mask, labels = tok["input_ids"], tok["attention_mask"], tok["labels"]

                # 2) Old logprobs/policy + old values (no_grad) și ref_logprobs (no_grad)
                with torch.inference_mode():
                    with autocast(device_type="cuda" if self.amp_enabled else "cpu", 
                                    dtype=self.autocast_dtype, enabled=self.amp_enabled):
                        # a) old policy (freeze current policy pentru rollouts)
                        old_logits, old_values_ = self.policy_model(input_ids, attention_mask, labels)
                        # b) modelul de referință (doar pentru shaping KL_ref)
                        ref_logits = self.policy_ref(input_ids=input_ids, attention_mask=attention_mask).logits

                    old_policy_logprobs = self.compute_logprobs(old_logits, labels)  # [B]
                    ref_logprobs        = self.compute_logprobs(ref_logits, labels)  # [B]
                    if old_values_ is None:
                        old_values_ = torch.zeros_like(old_policy_logprobs)

                old_policy_logprobs = old_policy_logprobs.clone().detach()
                ref_logprobs        = ref_logprobs.clone().detach()
                old_values          = old_values_.clone().detach()

                # 3) Rewards BRUTE (fără normalizare pe batch)
                raw_sentiment_rewards = self.compute_sentiment_reward(
                    completions,
                    use_margin=getattr(self.cfg, "use_margin_reward", False),
                    add_length_bonus=getattr(self.cfg, "add_length_bonus", True),
                    min_words=getattr(self.cfg, "min_words_bonus", 6),
                    bonus=getattr(self.cfg, "length_bonus", 0.05),
                )
                # Normalize rewards for stability in NLP setting
                raw_fn_rewards = self.rfn.compute_all_rewards(
                    prompts=batch["prompt"], 
                    completions=clean_completions, 
                    answers=batch["solution"])
                raw_rewards = raw_sentiment_rewards + raw_fn_rewards
                rewards = (raw_rewards - raw_rewards.mean()) / (raw_rewards.std() + 1e-8)
                rewards = rewards.clamp(-5.0, 5.0)
             
                # estimate next values for GAE (shift by one; placeholder)
                nv = torch.cat([raw_rewards.new_tensor([0.0]), old_values[:-1]])  # shift
                dones = torch.zeros_like(raw_rewards)

                # logging opțional per‑exemplu
                if step % getattr(self.cfg, "log_every", 50) == 0:
                    os.makedirs("./logs", exist_ok=True)
                    with open("./logs/rewards_log.txt", "a") as f:
                        for i, (text, reward) in enumerate(zip(completions, raw_rewards)):
                            am = batch_inputs["attention_mask"][i]
                            pl = int(am.sum().item())
                            prompt_ids = batch_inputs["input_ids"][i, -pl:]  # taie paddingul din stânga
                            prompt = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
                            text = self.tokenizer.decode(tok["input_ids"][i], skip_special_tokens=True)
                            space_star = "-" * 20
                            f.write(f"Step {step} | Example {i} | Reward: {reward:.4f} \n Prompt: {prompt} \n Text: {text}\n {space_star}\n")

                    self.writer.add_scalar("Reward/raw_mean", rewards.mean().item(), step)
                    self.writer.add_scalar("Reward/raw_std", rewards.std().item(), step)

                # append to replay buffer
                self.replay_buffer.append({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                    "old_policy_logprobs": old_policy_logprobs,
                    "ref_logprobs": ref_logprobs,
                    "old_values": old_values,
                    "rewards": rewards
                })

                # 4) PPO update cu micro-batch + gradient accumulation
                ppo_epochs = getattr(self.cfg, "ppo_epochs", 4)
                # Dynamic microbatch sizing based on token budget
                seq_lens = attention_mask.sum(dim=1).float()
                avg_len = seq_lens.mean().item()
                microbatch_size = min(
                    self.cfg.microbatch_size,
                    max(1, int(self.cfg.token_microbatch_size / max(1.0, avg_len)))
                )
                accum_steps = getattr(self.cfg, "accum_steps", 1)

                B = input_ids.size(0)
                if microbatch_size is None or microbatch_size >= B:
                    microbatch_size = B  # un singur micro-batch

                last_loss = last_pol = last_val = last_kl_signed = last_ent = last_kl_abs = None

                def ppo_update_pass(input_ids, attention_mask, labels,
                    old_policy_logprobs, ref_logprobs,
                    rewards, old_values, next_values, dones):

                    mb_B = input_ids.size(0)
                    mb_losses = []; mb_pols = []; mb_vals = []; mb_kls_signed = []; mb_ents = []; mb_kls_abs = []
                    mb_count = 0
                    self.optimizer.zero_grad(set_to_none=True)
                    for start in range(0, mb_B, microbatch_size):
                        end = min(start + microbatch_size, mb_B)
                        mb_input_ids           = input_ids[start:end]
                        mb_attention_mask      = attention_mask[start:end]
                        mb_labels              = labels[start:end]
                        mb_old_policy_logprobs = old_policy_logprobs[start:end]
                        mb_ref_logprobs        = ref_logprobs[start:end]
                        mb_rewards             = rewards[start:end]
                        mb_old_values          = old_values[start:end]
                        mb_next_values         = next_values[start:end]
                        mb_dones               = dones[start:end]

                        with autocast(device_type="cuda" if self.amp_enabled else "cpu", 
                                        dtype=self.autocast_dtype, enabled=self.amp_enabled):
                            logits, values = self.policy_model(mb_input_ids, mb_attention_mask, mb_labels)
                            new_logprobs = self.compute_logprobs(logits, mb_labels)
                            loss, pol_loss, val_loss, kl_ref_signed, entropy, kl_ref_abs = self.ppo_loss(
                                mb_old_policy_logprobs, new_logprobs, mb_ref_logprobs,
                                mb_rewards, values, mb_old_values, logits, mb_attention_mask,
                                next_values=mb_next_values, dones=mb_dones
                            )
                            loss = loss / accum_steps  # scale pentru acumulare
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
                        torch.tensor(mb_losses).mean(),
                        torch.tensor(mb_pols).mean(),
                        torch.tensor(mb_vals).mean(),
                        torch.tensor(mb_kls_signed).mean(),
                        torch.tensor(mb_ents).mean(),
                        torch.tensor(mb_kls_abs).mean()
                    )

                # Off-policy replay
                if len(self.replay_buffer) >= self.cfg.buffer_size:
                    batch_samples = random.sample(self.replay_buffer, self.cfg.replay_batch_size)
                    buffer_input_ids = torch.cat([b["input_ids"] for b in batch_samples], dim=0)
                    buffer_attention_mask = torch.cat([b["attention_mask"] for b in batch_samples], dim=0)
                    buffer_labels = torch.cat([b["labels"] for b in batch_samples], dim=0)
                    buffer_old_policy_logprobs = torch.cat([b["old_policy_logprobs"] for b in batch_samples], dim=0)
                    buffer_old_values = torch.cat([b["old_values"] for b in batch_samples], dim=0)
                    buffer_ref_logprobs = torch.cat([b["ref_logprobs"] for b in batch_samples], dim=0)
                    buffer_rewards = torch.cat([b["rewards"] for b in batch_samples], dim=0)
                    buffer_next_values = torch.cat([torch.cat([b["rewards"].new_tensor([0.0]), b["old_values"][:-1]]) for b in batch_samples], dim=0)
                    buffer_dones = torch.zeros_like(buffer_rewards)
                    _ = ppo_update_pass(
                        buffer_input_ids, buffer_attention_mask, buffer_labels,
                        buffer_old_policy_logprobs, buffer_ref_logprobs,
                        buffer_rewards, buffer_old_values, buffer_next_values, buffer_dones
                    )

                for _ in range(ppo_epochs):
                    losses, pols, vals, kls_signed, ents, kls_abs = ppo_update_pass(
                        input_ids, attention_mask, labels,
                        old_policy_logprobs, ref_logprobs,
                        rewards, old_values, nv, dones
                    )
                    last_loss = losses
                    last_pol  = pols
                    last_val  = vals
                    last_ent  = ents
                    last_kl_signed = kls_signed
                    last_kl_abs    = kls_abs

                    # Adaptive KL pe |Δ|
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
                        self.writer.add_scalar("KL/window_avg_abs", kl_avg, step)
                        self.writer.add_scalar("KL/adapted_coef", self.cfg.kl_coef, step)
                        self._kl_running.clear()

                # agregare pe epocă
                epoch_loss    += float(last_loss.item())
                epoch_policy  += float(last_pol.item())
                epoch_value   += float(last_val.item())
                epoch_kl      += float(last_kl_signed.item())   # semnat pentru afișare
                epoch_entropy += float(last_ent.item())
                epoch_reward  += float(rewards.mean().item())
                n_batches     += 1

                # log periodic
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

                    self.writer.add_scalar("Stats/KL_ref_tok", avg_kl, step)     # semnat
                    self.writer.add_scalar("Loss/total",   avg_loss, step)
                    self.writer.add_scalar("Loss/policy",  avg_pol,  step)
                    self.writer.add_scalar("Loss/value",   avg_val,  step)
                    self.writer.add_scalar("Stats/entropy",avg_ent,  step)
                    self.writer.add_scalar("Reward/mean",  avg_r,    step)

            # — end epoch: model selection + checkpoint —
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