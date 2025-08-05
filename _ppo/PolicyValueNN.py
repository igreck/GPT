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
    Activează temporar caching-ul pentru generare, dezactivând gradient checkpointing,
    apoi revine la setările anterioare.
    """
    base = peft_lm
    # PeftModel -> .base_model (HF PreTrainedModel); uneori sub .model
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
            base.gradient_checkpointing_enable()


class PPOQLoRAWithValueHead(nn.Module):
    """
    QLoRA pe Qwen/Qwen3 sau orice model în 4-bit (NF4):
      - fie LoRA nou (ca în codul inițial),
      - fie LoRA încărcat dintr-un director SFT (ex. produs de Unsloth/TRL).
    În ambele cazuri poți continua PPO (antrenezi LoRA + value_head).
    """
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-1.7B",
        r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05,
        target_modules = ("q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"),
        value_hidden_dim: int = 512,
        use_gradient_checkpointing: bool = True,
        use_cache: bool = False,              # la training: False (ckpt on); generate() activează cache temporar
        device_map: str = "auto",
        lora_from_sft_dir: str | None = None, # dacă setezi acest path, atașează adaptoarele SFT existente
        load_value_head_from: str | None = None,  # opțional, cale către value_head.pt
    ):
        super().__init__()

        # ---- 4-bit quantization (modul de load inițial) ----
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        # device_map = {"": "cpu"}

        base = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_cfg,
            device_map=device_map,
        )
        if base.config.pad_token_id is None:
            base.config.pad_token_id = base.config.eos_token_id
        base.config.use_cache = use_cache

        # ---- pregătire k-bit training + gradient checkpointing ----
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=use_gradient_checkpointing)
        if use_gradient_checkpointing and hasattr(base, "gradient_checkpointing_enable"):
            base.gradient_checkpointing_enable()

        # ---- LoRA: fie nou, fie încărcat din SFT ----
        if lora_from_sft_dir is None:
            # (1) LoRA nou (exact ca inițial)
            lora_cfg = LoraConfig(
                r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                target_modules=list(target_modules),
                bias="none", task_type="CAUSAL_LM"
            )
            peft_lm = get_peft_model(base, lora_cfg)   # doar LoRA devin trainable
        else:
            # (2) Încarcă adaptoarele LoRA pre-antrenate (SFT) — compatibile PEFT/Unsloth
            peft_lm = PeftModel.from_pretrained(base, lora_from_sft_dir)
            # asigură-te că LoRA rămân trainabile pentru PPO
            for n, p in peft_lm.named_parameters():
                if "lora_" in n:
                    p.requires_grad = True

        self.model = peft_lm

        # ---- Value head (pentru PPO) ----
        hidden = getattr(self.model.config, "hidden_size", None) or getattr(self.model.config, "n_embd")
        assert hidden is not None, "Nu pot deduce hidden_size din config."
        self.value_head = nn.Sequential(
            nn.Linear(hidden, value_hidden_dim),
            nn.LayerNorm(value_hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.Linear(value_hidden_dim, 1),
        )
        # dacă ai un value_head salvat (ex. dintr-o sesiune PPO anterioară)
        if load_value_head_from is not None:
            vh_path = os.path.join(load_value_head_from, "value_head.pt")
            if os.path.exists(vh_path):
                state = torch.load(vh_path, map_location="cpu")
                self.value_head.load_state_dict(state, strict=True)
                print(f"[load] value_head loaded from: {vh_path}")
            else:
                print(f"[load] value_head not found at: {vh_path} (using random init)")

        self.print_trainable_params()
        if torch.cuda.is_available():
            cudnn.benchmark = True

    # ---------- Utils ----------
    def print_trainable_params(self):
        trainable, total = 0, 0
        for p in self.parameters():
            c = p.numel()
            total += c
            if p.requires_grad:
                trainable += c
        pct = 100 * trainable / max(total, 1)
        print(f"[QLoRA] Trainable params: {trainable/1e6:.2f}M / {total/1e6:.2f}M ({pct:.2f}%)")

    # ---------- Forward ----------
    def forward(self, input_ids, attention_mask, labels=None):
        amp_enabled = torch.cuda.is_available()
        with autocast(device_type="cuda", enabled=amp_enabled):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=True,
            )
        logits = outputs.logits
        hidden = outputs.hidden_states[-1]              # [B, T, H]
        last_idx = attention_mask.sum(dim=1) - 1
        pooled = hidden[torch.arange(hidden.size(0)), last_idx]
        values = self.value_head(pooled).squeeze(-1)    # [B]
        return logits, values

    @torch.inference_mode()
    def value_forward(self, input_ids, attention_mask):
        amp = torch.cuda.is_available()
        with autocast(device_type="cuda", enabled=amp):
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            last_h = out.hidden_states[-1]
            last_idx = attention_mask.sum(dim=1) - 1
            pooled = last_h[torch.arange(last_h.size(0)), last_idx]
            values = self.value_head(pooled).squeeze(-1)
        return values
    
    def generate(self, *args, **kwargs):
        """
        Generare rapidă: activăm temporar caching și dezactivăm checkpointing.
        (în training normal, use_cache=False rămâne)
        """
        with temporary_cache(self.model, enable_cache=True):
            return self.model.generate(*args, **kwargs)

    # ---------- Save / Load ----------
    def save_pretrained(self, save_directory: str, save_adapters: bool = True, save_value_head: bool = True, meta: dict | None = None):
        """
        Salvează adaptoarele LoRA (PEFT) + value_head în `save_directory`.
        Baza (greutățile modelului 4-bit) NU este salvată aici.
        """
        os.makedirs(save_directory, exist_ok=True)

        if save_adapters:
            # self.model este PeftModel -> va scrie adapter_model.bin + adapter_config.json etc.
            self.model.save_pretrained(save_directory)
            print(f"[save] PEFT adapters saved to: {save_directory}")

        if save_value_head:
            vh_path = os.path.join(save_directory, "value_head.pt")
            torch.save(self.value_head.state_dict(), vh_path)
            print(f"[save] value_head saved to: {vh_path}")

        # opțional: scriem și un mic fișier meta
        base_name = None
        try:
            base_name = self.model.base_model.model.name_or_path
        except Exception:
            try:
                base_name = self.model.base_model.name_or_path
            except Exception:
                base_name = None

        meta_obj = {
            "class": "PPOQLoRAWithValueHead",
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
        # dacă vrei să forțezi alt base model decât ce e în adapter_config:
        override_base_model_name: str | None = None,
        # opțional: alte argumente pe care le primește __init__
        **kwargs,
    ):
        """
        Reconstruiește instanța pornind de la un director cu PEFT adapters + value_head.pt (opțional).
        Pași:
          1) Citește adapter_config pentru a afla baza (dacă nu e override).
          2) Încarcă baza în 4‑bit (NF4).
          3) Atașează PEFT adapters din `peft_or_dir`.
          4) Încarcă value_head.pt dacă există.
        """
        # 1) Aflăm baza
        base_name = override_base_model_name
        try:
            peft_cfg = PeftConfig.from_pretrained(peft_or_dir)
            base_name = base_name or peft_cfg.base_model_name_or_path
        except Exception as e:
            if base_name is None:
                raise RuntimeError(
                    f"Could not infer base model name from {peft_or_dir}. "
                    f"Provide override_base_model_name."
                ) from e

        # 2) Încărcăm baza în 4‑bit (modul de load inițial)
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
            base.gradient_checkpointing_enable()

        # 3) Atașăm PEFT adapters
        peft_lm = PeftModel.from_pretrained(base, peft_or_dir)
        # ne asigurăm că LoRA rămâne antrenabilă pentru PPO
        for n, p in peft_lm.named_parameters():
            if "lora_" in n:
                p.requires_grad = True

        # 4) Construim instanța clasei și atașăm value_head
        obj = cls.__new__(cls)  # evităm __init__, că deja am compus base + peft
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
            print(f"[load] value_head.pt not found in {peft_or_dir} (using random init).")

        # reatașăm metodele instanței
        obj.print_trainable_params = getattr(cls, "print_trainable_params").__get__(obj, cls)
        obj.generate = getattr(cls, "generate").__get__(obj, cls)
        obj.forward = getattr(cls, "forward").__get__(obj, cls)
        obj.value_forward = getattr(cls, "value_forward").__get__(obj, cls)

        return obj