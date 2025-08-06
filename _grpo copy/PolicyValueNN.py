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
    Activate temporary caching for generation: disable gradient checkpointing, enable cache,
    then restore.
    """
    base = peft_lm
    if hasattr(base, "base_model"): base = base.base_model
    if hasattr(base, "model"):      base = base.model

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
            base.gradient_checkpointing_enable()


class GRPOQLoRA(nn.Module):
    """
    QLoRA policy model for GRPO: 4-bit NF4 LoRA-adapted causal LM, no value head.
    Supports new or pretrained (SFT) LoRA adapters.
    """
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-1.7B",
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules = ("q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"),
        use_gradient_checkpointing: bool = True,
        use_cache: bool = False,
        device_map: str = "auto",
        lora_from_sft_dir: str | None = None,
    ):
        super().__init__()
        # 4-bit NF4 quantization
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

        # prepare for k-bit training
        base = prepare_model_for_kbit_training(
            base, use_gradient_checkpointing=use_gradient_checkpointing
        )
        if use_gradient_checkpointing and hasattr(base, "gradient_checkpointing_enable"):
            base.gradient_checkpointing_enable()

        # LoRA adapters
        if lora_from_sft_dir is None:
            lora_cfg = LoraConfig(
                r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                target_modules=list(target_modules), bias="none", task_type="CAUSAL_LM"
            )
            peft_lm = get_peft_model(base, lora_cfg)
        else:
            peft_lm = PeftModel.from_pretrained(base, lora_from_sft_dir)
            for n, p in peft_lm.named_parameters():
                if "lora_" in n:
                    p.requires_grad = True

        self.model = peft_lm
        self.print_trainable_params()
        if torch.cuda.is_available(): cudnn.benchmark = True

    def print_trainable_params(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"[QLoRA] Trainable: {trainable/1e6:.2f}M/{total/1e6:.2f}M")

    def forward(self, input_ids, attention_mask):
        amp = torch.cuda.is_available()
        with autocast(device_type="cuda" if amp else "cpu", enabled=amp):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs

    @torch.inference_mode()
    def generate(self, *args, **kwargs):
        with temporary_cache(self.model, enable_cache=True):
            return self.model.generate(*args, **kwargs)

    def save_pretrained(self, save_directory: str, save_adapters: bool = True, meta: dict | None = None):
        os.makedirs(save_directory, exist_ok=True)
        if save_adapters:
            self.model.save_pretrained(save_directory)
            print(f"[save] Adapters saved to {save_directory}")
        meta_obj = {"class": "QwenQLoRA", "base_model": self.model.base_model.name_or_path}
        if meta: meta_obj.update(meta)
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
        override_base_model_name: str | None = None,
        **init_kwargs,
    ):
        # determine base model
        base_name = override_base_model_name
        try:
            peft_cfg = PeftConfig.from_pretrained(peft_or_dir)
            base_name = base_name or peft_cfg.base_model_name_or_path
        except:
            if base_name is None:
                raise RuntimeError("Provide override_base_model_name to load adapters.")
        # load base
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base = AutoModelForCausalLM.from_pretrained(
            base_name, quantization_config=bnb_cfg, device_map=device_map
        )
        if base.config.pad_token_id is None: base.config.pad_token_id = base.config.eos_token_id
        base.config.use_cache = use_cache
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing)
        if use_gradient_checkpointing and hasattr(base, "gradient_checkpointing_enable"):
            base.gradient_checkpointing_enable()
        # attach adapters
        peft_lm = PeftModel.from_pretrained(base, peft_or_dir)
        for n, p in peft_lm.named_parameters():
            if "lora_" in n: p.requires_grad = True
        # build instance
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.model = peft_lm
        obj.print_trainable_params = getattr(cls, "print_trainable_params").__get__(obj, cls)
        obj.generate              = getattr(cls, "generate").__get__(obj, cls)
        obj.forward               = getattr(cls, "forward").__get__(obj, cls)
        obj.save_pretrained       = getattr(cls, "save_pretrained").__get__(obj, cls)
        return obj