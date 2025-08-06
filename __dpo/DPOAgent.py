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

from typing import List
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.amp.autocast_mode import autocast 
from torch.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn

class DPOAgent:
    """
    Direct Preference Optimization agent.
    Implements the DPO algorithm (Rafailov et al., 2024).
    """
    def __init__(
        self,
        policy_model,
        policy_ref,
        tokenizer,
        optimizer,
        scheduler,
        config,
    ):
        # Config and device setup
        self.cfg = config
        self.device = config.device
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        cudnn.deterministic = False
        cudnn.benchmark = True

        # Models
        self.policy_model = policy_model.to(self.device).train()
        self.policy_ref   = policy_ref.to(self.device).eval()

        # Optimizer & scheduler
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler    = GradScaler()

        # Tokenizer & logging
        self.tokenizer = tokenizer
        self.writer    = SummaryWriter(log_dir=config.log_dir)
        os.makedirs(config.log_dir, exist_ok=True)

        # DPO hyperparameters
        self.beta = config.beta
        self.max_grad_norm = getattr(config, "max_grad_norm", 1.0)
        self.n_epochs = config.n_epochs

    def compute_token_logprobs(self, logits, labels, mask):
        """
        Compute token-level log probabilities for the given logits and labels,
        masking out padding tokens.
        """
        lp = F.log_softmax(logits, dim=-1)
        token_lp = lp.gather(2, labels.unsqueeze(-1)).squeeze(-1)
        return token_lp * mask

    def generate(self, prompts: List[str]) -> List[str]:
        """
        (Optional) Generation method, unchanged from before.
        """
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

    def train(self, dataloader):
        """
        Train the policy model with Direct Preference Optimization.
        Expects dataloader yielding dicts with keys:
          - 'prompt': List[str]
          - 'y_w': List[str] (preferred completions)
          - 'y_l': List[str] (dispreferred completions)
        """
        self.policy_model.train()
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
                prompts = batch['prompt']
                y_w = batch['y_w']
                y_l = batch['y_l']

                # Tokenize preferred and dispreferred completions
                tok_w = self.tokenizer(prompts, y_w, return_tensors='pt', padding=True, truncation=True)
                tok_l = self.tokenizer(prompts, y_l, return_tensors='pt', padding=True, truncation=True)
                input_ids_w, attn_w = tok_w.input_ids.to(self.device), tok_w.attention_mask.to(self.device)
                input_ids_l, attn_l = tok_l.input_ids.to(self.device), tok_l.attention_mask.to(self.device)

                # Compute logits under current and reference policies
                logits_w     = self.policy_model(input_ids_w, attn_w).logits
                logits_ref_w = self.policy_ref(input_ids_w, attn_w).logits
                logits_l     = self.policy_model(input_ids_l, attn_l).logits
                logits_ref_l = self.policy_ref(input_ids_l, attn_l).logits

                # Compute sequence log-probs
                lp_w     = self.compute_token_logprobs(logits_w, input_ids_w, attn_w).sum(dim=1)
                lp_ref_w = self.compute_token_logprobs(logits_ref_w, input_ids_w, attn_w).sum(dim=1)
                lp_l     = self.compute_token_logprobs(logits_l, input_ids_l, attn_l).sum(dim=1)
                lp_ref_l = self.compute_token_logprobs(logits_ref_l, input_ids_l, attn_l).sum(dim=1)

                # Compute implicit rewards and DPO loss
                r_w = self.beta * (lp_w - lp_ref_w)
                r_l = self.beta * (lp_l - lp_ref_l)
                loss = -torch.log(torch.sigmoid(r_w - r_l)).mean()

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            self.writer.add_scalar('DPO/Loss', avg_loss, epoch+1)
            print(f"Epoch {epoch+1} | Avg loss: {avg_loss:.4f}")
