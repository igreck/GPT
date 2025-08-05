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
    Group Relative Policy Optimization (GRPO) agent strict implementation.
    Implements Algorithm 1 from the paper with:
      - no critic network
      - token-wise clipping (PPO-style)
      - KL regularization against a fixed reference policy
      - multiple update epochs µ per batch
      - dynamic clipping schedule
      - off-policy replay buffer
      - micro-batching with gradient accumulation
      - adaptive KL coefficient adjustment
      - per-example logging
      - periodic file logging at configured steps
      - checkpointing & model selection
    """
    def __init__(
        self,
        policy_model: torch.nn.Module,
        policy_ref: torch.nn.Module,
        tokenizer,
        optimizer: torch.optim.Optimizer,
        scheduler,
        config,
        reward_functions: List[Callable[[List[str]], torch.Tensor]],
    ):
        # reproducibility & device
        self.device = config.device
        random.seed(config.seed); np.random.seed(config.seed); torch.manual_seed(config.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(config.seed)
        cudnn.deterministic = False; cudnn.benchmark = True

        # models
        self.policy_model = policy_model.to(self.device).train()
        self.policy_ref   = policy_ref.to(self.device).eval()
        self.policy_old   = copy.deepcopy(policy_model).to(self.device).eval()
        if hasattr(self.policy_model, 'gradient_checkpointing_enable'):
            self.policy_model.gradient_checkpointing_enable()

        # tokenizer and optim
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler    = GradScaler()
        self.writer    = SummaryWriter(log_dir=config.log_dir)
        os.makedirs(config.log_dir, exist_ok=True)
        self.config    = config

        # rewards & buffer
        self.reward_functions = reward_functions.copy()
        self.replay_buffer = deque(maxlen=config.buffer_size)

        # adaptive KL
        self._kl_running = []
        self.best_reward = -float('inf')
        self.no_improve_epochs = 0

        # prepare step log file
        self.step_log_path = os.path.join(config.log_dir, 'step_logs.txt')
        with open(self.step_log_path, 'w') as f:
            f.write('')  # clear previous

    def generate(self, prompts: List[str]) -> List[str]:
        inputs = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k,v in inputs.items()}
        with torch.no_grad():
            out = self.policy_model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                top_p=self.config.top_p,
                temperature=self.config.temperature,
                repetition_penalty=self.config.repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
            )
        seqs = out.sequences[:, inputs['input_ids'].shape[1]:]
        return self.tokenizer.batch_decode(seqs, skip_special_tokens=True)

    def compute_group_rewards(self, completions: List[str]) -> torch.Tensor:
        scores = [fn(completions).to(self.device) for fn in self.reward_functions]
        return torch.stack(scores, dim=0).sum(dim=0)

    def compute_token_logprobs(self, logits, labels, mask):
        logp = F.log_softmax(logits, dim=-1)
        lp = logp.gather(2, labels.unsqueeze(-1)).squeeze(-1)
        return lp * mask

    def train(self, dataloader):
        cfg = self.config
        total_updates = cfg.n_epochs * len(dataloader) * cfg.update_epochs
        current_update = 0

        # init old policy
        self.policy_old.load_state_dict(self.policy_model.state_dict())
        print(f"Starting training for {cfg.n_epochs} epochs, ~{total_updates} updates")

        for epoch in range(cfg.n_epochs):
            print(f"\n=== Epoch {epoch+1}/{cfg.n_epochs} ===")
            epoch_loss = epoch_kl = epoch_reward = 0.0
            n_updates = 0
            prog = tqdm(dataloader, desc=f"Epoch {epoch+1}")
            for step, batch in enumerate(prog):
                prompts = batch['text']
                G = cfg.group_size
                if step % cfg.log_every == 0:
                    print(f"Batch {step}/{len(dataloader)}: generating group of size {G}")

                # 1) sample group
                all_comps = [self.generate(prompts) for _ in range(G)]
                raw = torch.stack([self.compute_group_rewards(c) for c in all_comps], dim=0)
                print(f"  Mean raw reward: {raw.mean().item():.4f}")

                # per-instance logging
                with open(os.path.join(cfg.log_dir, 'instance_logs.txt'), 'a') as f:
                    for g in range(G):
                        for i,p in enumerate(prompts):
                            f.write(f"Upd {current_update}|G{g}|i{i}|R{raw[g,i].item():.4f}\n")
                            f.write(f"P:{p}\nC:{all_comps[g][i]}\n{'='*20}\n")
                print(f"  Logged {G*len(prompts)} instances")

                reward = raw
                self.replay_buffer.append({'prompts':prompts,'completions':all_comps,'rewards':reward})

                # update steps
                for mu in range(cfg.update_epochs):
                    frac = current_update / max(1,total_updates)
                    eps = cfg.clip_epsilon*(1-frac) + cfg.clip_epsilon_final*frac
                    print(f"  Step {mu+1}/{cfg.update_epochs}: eps={eps:.4f}, kl_coef={cfg.kl_coef:.4f}")

                    baseline = reward.mean(dim=0)
                    flat_adv = (reward - baseline.unsqueeze(0)).view(-1)
                    flat_prompts = prompts * G
                    flat_comps = [c for grp in all_comps for c in grp]

                    tok = self.tokenizer(flat_prompts, flat_comps, return_tensors='pt',
                                           padding=True, truncation=True, max_length=cfg.policy_max_length)
                    input_ids = tok.input_ids.to(self.device)
                    attn = tok.attention_mask.to(self.device)
                    mask = (input_ids!=self.tokenizer.pad_token_id).float()
                    seq_lens = attn.sum(dim=1).float(); avg_len = seq_lens.mean().item()
                    micro = min(cfg.microbatch_size, max(1,int(cfg.token_microbatch_size/avg_len)))
                    accum = cfg.accum_steps
                    print(f"    Micro={micro}, accum={accum}")

                    self.optimizer.zero_grad(set_to_none=True)
                    losses=[]
                    for start in range(0, input_ids.size(0), micro):
                        end = start+micro
                        ids,input_am = input_ids[start:end],attn[start:end]
                        adv_mb = flat_adv[start:end].to(self.device).unsqueeze(1)
                        with autocast():
                            logits = self.policy_model(ids,input_am).logits
                            with torch.no_grad():
                                logits_old = self.policy_old(ids,input_am).logits
                                logits_ref = self.policy_ref(ids,input_am).logits
                            lp_new = self.compute_token_logprobs(logits,ids,input_am)
                            lp_old = self.compute_token_logprobs(logits_old,ids,input_am)
                            ratio = torch.exp(lp_new-lp_old)
                            r_clip= torch.clamp(ratio,1-eps,1+eps)
                            sur = torch.min(ratio*adv_mb, r_clip*adv_mb)*input_am
                            policy_loss = -sur.sum()/input_am.sum()
                            logp_new = F.log_softmax(logits,dim=-1)
                            p_ref    = F.softmax(logits_ref,dim=-1)
                            kl = F.kl_div(logp_new,p_ref,reduction='batchmean')
                            loss = policy_loss + cfg.kl_coef*kl
                        losses.append(loss.item())
                        self.scaler.scale(loss/accum).backward()
                        if ((start//micro)+1)%accum==0 or end>=input_ids.size(0):
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(),1.0)
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            self.optimizer.zero_grad(set_to_none=True)
                            if self.scheduler: self.scheduler.step()

                    avg_loss = np.mean(losses)
                    print(f"    Loss={avg_loss:.4f}, KL={kl.item():.4f}")

                    # periodic file logging
                    if current_update % cfg.log_every == 0:
                        with open(self.step_log_path, 'a') as sf:
                            sf.write(f"Step {current_update} | Loss {avg_loss:.4f} | KL {kl.item():.4f} | Reward {reward.mean().item():.4f}\n")
                        print(f"    Wrote step metrics to {self.step_log_path}")
                        self.writer.add_scalar('GRPO/Loss', avg_loss, current_update)
                        self.writer.add_scalar('GRPO/KL', kl.item(), current_update)
                        self.writer.add_scalar('GRPO/Reward', reward.mean().item(), current_update)

                    # adaptive KL
                    self._kl_running.append(kl.abs().item())
                    if len(self._kl_running)>=cfg.kl_window:
                        avg_kl=sum(self._kl_running)/len(self._kl_running)
                        print(f"    Adaptive KL avg={avg_kl:.4f}")
                        if avg_kl>2*cfg.target_kl: cfg.kl_coef=min(cfg.kl_coef*cfg.kl_adapt_rate,cfg.max_kl_coef)
                        elif avg_kl<0.5*cfg.target_kl: cfg.kl_coef=max(cfg.kl_coef/cfg.kl_adapt_rate,cfg.min_kl_coef)
                        self._kl_running.clear()

                    epoch_loss+=avg_loss; epoch_kl+=kl.item(); epoch_reward+=reward.mean().item()
                    n_updates+=1; current_update+=1

            # epoch checkpoint
            avg_reward=epoch_reward/max(1,n_updates)
            print(f"Epoch {epoch+1} complete, avg_reward={avg_reward:.4f}")
            if avg_reward>self.best_reward:
                print(f"  New best reward, saving model...")
                self.best_reward=avg_reward; self.no_improve_epochs=0
                self.policy_model.save_pretrained(f"{cfg.save_dir}_best")
            else:
                self.no_improve_epochs+=1
            if (epoch+1)%cfg.save_every==0:
                print(f"  Saving checkpoint for epoch {epoch+1}")
                self.policy_model.save_pretrained(f"{cfg.save_dir}_epoch_{epoch}")
