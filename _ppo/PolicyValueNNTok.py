import os
import json
import contextlib
import torch
import torch.nn as nn
from torch.amp.autocast_mode import autocast
import torch.backends.cudnn as cudnn
from transformers import AutoModelForCausalLM
from transformers.utils.quantization_config import BitsAndBytesConfig
from peft import (
    LoraConfig, get_peft_model, prepare_model_for_kbit_training,
    PeftModel, PeftConfig
)


@contextlib.contextmanager
def temporary_cache(peft_lm, enable_cache: bool = True):
    """
    Activează temporar caching-ul (pentru generate) și oprește checkpointing-ul,
    apoi revine la setările anterioare.
    """
    base = peft_lm
    if hasattr(base, "base_model"):
        base = base.base_model
    if hasattr(base, "model"):
        base = base.model

    prev_cache = getattr(base.config, "use_cache", False)
    had_ckpt = hasattr(base, "is_gradient_checkpointing") and base.is_gradient_checkpointing

    try:
        if had_ckpt and hasattr(base, "gradient_checkpointing_disable"):
            base.gradient_checkpointing_disable()
        base.config.use_cache = enable_cache
        yield
    finally:
        base.config.use_cache = prev_cache
        if had_ckpt and hasattr(base, "gradient_checkpointing_enable"):
            try:
                base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                base.gradient_checkpointing_enable()


class PPOQLoRAWithValueHead(nn.Module):
    """
    QLoRA + value head pe TOKENI (values: [B, T]).
    Poți porni din LoRA SFT existent sau din LoRA nouă; salvează doar adaptoarele + value_head.
    """
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-1.7B",
        *,
        r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05,
        target_modules = ("q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"),
        value_hidden_dim: int = 512,
        use_gradient_checkpointing: bool = True,
        use_cache: bool = False,
        device_map: str = "auto",
        lora_from_sft_dir: str | None = None,
        load_value_head_from: str | None = None,
    ):
        super().__init__()

        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        base = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_cfg,
            device_map=device_map,
        )
        if base.config.pad_token_id is None:
            base.config.pad_token_id = base.config.eos_token_id
        base.config.use_cache = use_cache

        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=use_gradient_checkpointing)
        if use_gradient_checkpointing and hasattr(base, "gradient_checkpointing_enable"):
            try:
                base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                base.gradient_checkpointing_enable()

        if lora_from_sft_dir is None:
            lora_cfg = LoraConfig(
                r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                target_modules=list(target_modules),
                bias="none", task_type="CAUSAL_LM"
            )
            peft_lm = get_peft_model(base, lora_cfg)
        else:
            peft_lm = PeftModel.from_pretrained(base, lora_from_sft_dir)
            for n, p in peft_lm.named_parameters():
                if "lora_" in n:
                    p.requires_grad = True

        self.model = peft_lm

        hidden = getattr(self.model.config, "hidden_size", None) or getattr(self.model.config, "n_embd")
        assert hidden is not None, "Nu pot deduce hidden_size din config."
        self.value_head = nn.Sequential(
            nn.Linear(hidden, value_hidden_dim),
            nn.LayerNorm(value_hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.Linear(value_hidden_dim, 1),
        )

        if load_value_head_from is not None:
            vh_path = os.path.join(load_value_head_from, "value_head.pt")
            if os.path.exists(vh_path):
                state = torch.load(vh_path, map_location="cpu")
                self.value_head.load_state_dict(state, strict=True)
                print(f"[load] value_head loaded from: {vh_path}")
            else:
                print(f"[load] value_head not found at: {vh_path} (random init)")

        self.print_trainable_params()
        if torch.cuda.is_available():
            cudnn.benchmark = True

    def print_trainable_params(self):
        trainable, total = 0, 0
        for p in self.parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
        pct = 100 * trainable / max(total, 1)
        print(f"[QLoRA] Trainable params: {trainable/1e6:.2f}M / {total/1e6:.2f}M ({pct:.2f}%)")

    def forward(self, input_ids, attention_mask, labels=None):
        """
        return:
          logits: [B,T,V]
          values: [B,T]  (value pe FIECARE token)
        """
        amp_enabled = torch.cuda.is_available()
        with autocast(device_type="cuda", enabled=amp_enabled):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=True,
            )
        logits = outputs.logits                # [B,T,V]
        hidden = outputs.hidden_states[-1]     # [B,T,H]

        B, T, H = hidden.shape
        v = self.value_head(hidden.reshape(B*T, H)).view(B, T, 1).squeeze(-1)  # [B,T]
        return logits, v

    @torch.inference_mode()
    def value_forward_last_token(self, input_ids, attention_mask):
        amp = torch.cuda.is_available()
        with autocast(device_type="cuda", enabled=amp):
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            last_h = out.hidden_states[-1]
            last_idx = attention_mask.sum(dim=1) - 1
            pooled = last_h[torch.arange(last_h.size(0), device=last_h.device), last_idx]
            value_seq = self.value_head(pooled).squeeze(-1)
        return value_seq
    
    def generate(self, *args, **kwargs):
        with temporary_cache(self.model, enable_cache=True):
            return self.model.generate(*args, **kwargs)

    def save_pretrained(self, save_directory: str, save_adapters: bool = True, save_value_head: bool = True, meta: dict | None = None):
        os.makedirs(save_directory, exist_ok=True)
        if save_adapters:
            self.model.save_pretrained(save_directory)
            print(f"[save] PEFT adapters saved to: {save_directory}")
        if save_value_head:
            vh_path = os.path.join(save_directory, "value_head.pt")
            torch.save(self.value_head.state_dict(), vh_path)
            print(f"[save] value_head saved to: {vh_path}")

        base_name = None
        try:
            base_name = self.model.base_model.model.name_or_path
        except Exception:
            try:
                base_name = self.model.base_model.name_or_path
            except Exception:
                base_name = None

        meta_obj = {
            "class": "PolicyValue",
            "base_model_name": base_name,
            "hidden_size": getattr(self.model.config, "hidden_size", None),
        }
        if meta:
            meta_obj.update(meta)
        with open(os.path.join(save_directory, "meta.json"), "w") as f:
            json.dump(meta_obj, f, indent=2)

    @classmethod
    def from_pretrained(
        cls,
        peft_or_dir: str,
        *,
        device_map: str = "auto",
        use_gradient_checkpointing: bool = True,
        use_cache: bool = False,
        value_hidden_dim: int = 512,
        override_base_model_name: str | None = None,
        **kwargs,
    ):
        base_name = override_base_model_name
        try:
            peft_cfg = PeftConfig.from_pretrained(peft_or_dir)
            base_name = base_name or peft_cfg.base_model_name_or_path
        except Exception as e:
            if base_name is None:
                raise RuntimeError(
                    f"Could not infer base model name from {peft_or_dir}. Provide override_base_model_name."
                ) from e

        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
        base = AutoModelForCausalLM.from_pretrained(
            base_name,
            quantization_config=bnb_cfg,
            device_map=device_map,
        )
        if base.config.pad_token_id is None:
            base.config.pad_token_id = base.config.eos_token_id
        base.config.use_cache = use_cache

        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=use_gradient_checkpointing)
        if use_gradient_checkpointing and hasattr(base, "gradient_checkpointing_enable"):
            try:
                base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                base.gradient_checkpointing_enable()

        peft_lm = PeftModel.from_pretrained(base, peft_or_dir)
        for n, p in peft_lm.named_parameters():
            if "lora_" in n:
                p.requires_grad = True

        obj = cls.__new__(cls)
        nn.Module.__init__(obj)

        obj.model = peft_lm
        hidden = getattr(obj.model.config, "hidden_size", None) or getattr(obj.model.config, "n_embd")
        if hidden is None:
            raise ValueError("Could not infer hidden_size from config.")
        obj.value_head = nn.Sequential(
            nn.Linear(hidden, value_hidden_dim),
            nn.LayerNorm(value_hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.Linear(value_hidden_dim, 1),
        )

        vh_path = os.path.join(peft_or_dir, "value_head.pt")
        if os.path.exists(vh_path):
            state = torch.load(vh_path, map_location="cpu")
            obj.value_head.load_state_dict(state, strict=True)
            print(f"[load] value_head loaded from: {vh_path}")
        else:
            print(f"[load] value_head.pt not found in {peft_or_dir} (random init).")

        obj.print_trainable_params = getattr(cls, "print_trainable_params").__get__(obj, cls)
        obj.generate = getattr(cls, "generate").__get__(obj, cls)
        obj.forward = getattr(cls, "forward").__get__(obj, cls)
        obj.value_forward_last_token = getattr(cls, "value_forward_last_token").__get__(obj, cls)

        return obj
