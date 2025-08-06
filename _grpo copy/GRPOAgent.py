
# Refactored GRPOAgent implementation matching PPOAgent's structure and efficiency,
# with group-based logic, dynamic clip scheduling, off-policy replay, micro-batching,
# gradient accumulation, adaptive KL, and logging.

import copy
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import List, Callable, Dict
from torch.amp.autocast_mode import autocast 
from torch.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
from collections import deque

class GRPOAgent:
    """
    Group Relative Policy Optimization agent.
    Implements Algorithm 1 with dynamic clipping, off-policy replay,
    micro-batching, adaptive KL, and detailed logging.
    """
    def __init__(
        self,
        policy_model,
        policy_ref,
        tokenizer,
        optimizer,
        scheduler,
        config,
        reward_functions,
    ):
        # Config and device
        self.cfg = config
        self.device = config.device
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(config.seed)
        cudnn.deterministic = False
        cudnn.benchmark = True

        # Models
        self.policy_model = policy_model.to(self.device).train()
        self.policy_ref   = policy_ref.to(self.device).eval()
        self.policy_old   = copy.deepcopy(self.policy_model).to(self.device).eval()

        # Gradient checkpointing
        if hasattr(self.policy_model, 'gradient_checkpointing_enable'):
            self.policy_model.gradient_checkpointing_enable()

        # Optimizer & scheduler
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler    = GradScaler()

        # Tokenizer & logging
        self.tokenizer = tokenizer
        self.writer    = SummaryWriter(log_dir=config.log_dir)
        os.makedirs(config.log_dir, exist_ok=True)

        # Replay buffer & rewards
        self.replay_buffer = deque(maxlen=config.buffer_size)
        self.reward_functions = reward_functions

        # Clip scheduling
        self.initial_clip_epsilon = config.clip_epsilon
        self.final_clip_epsilon   = config.clip_epsilon_final

        # Adaptive KL
        self._kl_running = []
        self.best_reward = -float('inf')
        self.step_log_path = os.path.join(config.log_dir, 'step_logs.txt')
        open(self.step_log_path, 'w').close()

    def generate(self, prompts: List[str]) -> List[str]:
        inputs = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            amp = torch.cuda.is_available()
            with autocast(device_type="cuda" if amp else "cpu", enabled=amp):
                gen = self.policy_model.generate(
                    **inputs,
                    max_new_tokens=self.cfg.max_new_tokens,
                    do_sample=True,
                    top_p=self.cfg.top_p,
                    temperature=self.cfg.temperature,
                    repetition_penalty=self.cfg.repetition_penalty,
                    pad_token_id=self.tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                )
            seqs = gen.sequences[:, inputs['input_ids'].size(1):]
        return self.tokenizer.batch_decode(seqs, skip_special_tokens=True)

    def compute_group_rewards(self, completions: List[str]) -> torch.Tensor:
        scores = [fn(completions).to(self.device) for fn in self.reward_functions]
        return torch.stack(scores).sum(dim=0)

    def compute_token_logprobs(self, logits, labels, mask):
        lp = F.log_softmax(logits, dim=-1)
        token_lp = lp.gather(2, labels.unsqueeze(-1)).squeeze(-1)
        return token_lp * mask

    def train(self, dataloader):
        cfg = self.cfg
        total_updates = cfg.n_epochs * len(dataloader) * cfg.update_epochs
        update_count = 0

        # Copy initial policy
        self.policy_old.load_state_dict(self.policy_model.state_dict(), strict=False)
        print(f"Training for {cfg.n_epochs} epochs ({total_updates} updates)...")

        for epoch in range(cfg.n_epochs):
            epoch_reward = 0.0
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
                prompts = batch['text']
                # Generate group
                groups = [self.generate(prompts) for _ in range(cfg.group_size)]
                raw = torch.stack([self.compute_group_rewards(g) for g in groups])  # [G,B]
                baseline = raw.mean(dim=0)
                advantage = raw - baseline.unsqueeze(0)

                                # logging opțional per‑exemplu
                if update_count % getattr(cfg, "log_every", 50) == 0 and update_count != 0:
                    os.makedirs(cfg.log_dir, exist_ok=True)
                    with open(f"{cfg.log_dir}/rewards_log.txt", "a") as f:
                        for i, (texts, rewards) in enumerate(zip(groups, raw)):
                            space_star = "-" * 20
                            for text, reward in zip(texts, rewards.tolist()):
                                f.write(f"Step {update_count} | Group {i} | Reward: {reward:.4f} \n Prompt: {prompts[i]}\n Text: {text}\n {space_star}\n")
                                # print(f"Step {update_count} | Group {i} | Reward: {reward:.4f} \n Prompt: {prompts[i]}\n Text: {text}\n {space_star}\n")



                # Flatten prompts/comps
                flat_prompts = prompts * cfg.group_size
                flat_comps   = [c for grp in groups for c in grp]
                flat_adv     = advantage.view(-1)

                # Tokenize
                tok = self.tokenizer(flat_prompts, flat_comps,
                                     return_tensors='pt', padding=True,
                                     truncation=True, max_length=cfg.policy_max_length)
                input_ids = tok.input_ids.to(self.device)
                attn      = tok.attention_mask.to(self.device)
                mask      = (input_ids != self.tokenizer.pad_token_id).float()

                # Update policy_old
                self.policy_old.load_state_dict(self.policy_model.state_dict(), strict=False)

                # μ update steps
                for u in range(cfg.update_epochs):
                    frac = update_count / max(1, total_updates)
                    eps  = self.initial_clip_epsilon * (1 - frac) + self.final_clip_epsilon * frac

                    # Micro-batching
                    seq_lens = attn.sum(dim=1).float(); avg_len = seq_lens.mean().item()
                    micro = min(cfg.microbatch_size,
                                int(cfg.token_microbatch_size / max(1.0, avg_len)))
                    accum = cfg.accum_steps

                    self.optimizer.zero_grad()
                    losses = []
                    for start in range(0, input_ids.size(0), micro):
                        end = start + micro
                        ids = input_ids[start:end]
                        am  = attn[start:end]
                        adv_mb = flat_adv[start:end].to(self.device).unsqueeze(1)

                        amp = torch.cuda.is_available()
                        with autocast(device_type="cuda" if amp else "cpu", enabled=amp):
                            logits = self.policy_model(ids, am).logits
                            with torch.no_grad():
                                logits_old = self.policy_old(ids, am).logits
                                logits_ref = self.policy_ref(ids, am).logits

                            # policy ratio
                            lp_new = self.compute_token_logprobs(logits, ids, am)
                            lp_old = self.compute_token_logprobs(logits_old, ids, am)
                            ratio = torch.exp(lp_new - lp_old)
                            r_clip = ratio.clamp(1 - eps, 1 + eps)
                            surrogate = torch.min(ratio * adv_mb, r_clip * adv_mb) * am
                            policy_loss = -surrogate.sum() / am.sum()

                            # token-wise KL
                            p_ref  = F.softmax(logits_ref, dim=-1)
                            logp_new = F.log_softmax(logits, dim=-1)
                            logp_ref = F.log_softmax(logits_ref, dim=-1)
                            kl_token = (p_ref * (logp_ref - logp_new)).sum(dim=-1)
                            kl = (kl_token * am).sum() / am.sum()

                            loss = policy_loss + cfg.kl_coef * kl

                        losses.append(loss.item())
                        self.scaler.scale(loss / accum).backward()
                        if ((start//micro)+1) % accum == 0 or end >= input_ids.size(0):
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            self.optimizer.zero_grad()
                            if self.scheduler: self.scheduler.step()

                    update_count += 1
                    epoch_reward += raw.mean().item()

                    # periodic logging
                    if update_count % cfg.log_every == 0:
                        avg_loss = np.mean(losses)
                        with open(self.step_log_path, 'a') as f:
                            f.write(f"Upd {update_count} | Loss {avg_loss:.4f} | KL {kl.item():.4f} | R {raw.mean().item():.4f}\n")
                        self.writer.add_scalar('GRPO/Loss', avg_loss, update_count)
                        self.writer.add_scalar('GRPO/KL', kl.item(), update_count)
                        self.writer.add_scalar('GRPO/Reward', raw.mean().item(), update_count)

                # end μ updates

            # epoch checkpoint
            avg_R = epoch_reward / max(1, update_count)
            print(f"Epoch {epoch+1} done | AvgR {avg_R:.4f}")
            if avg_R > self.best_reward:
                self.best_reward = avg_R
                print(f"New best reward {avg_R:.4f}, saving model...")
                self.policy_model.save_pretrained(f"{cfg.save_dir}_best")
            if (epoch+1) % cfg.save_every == 0:
                self.policy_model.save_pretrained(f"{cfg.save_dir}_epoch_{epoch+1}")
