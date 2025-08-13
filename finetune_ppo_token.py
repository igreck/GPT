# train_ppo.py
import os
import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.utils.quantization_config import BitsAndBytesConfig
from torch.optim import AdamW

from _ppo.config import Config
from imdb_dataset import build_imdb_dataloader
from openr1_dataset import build_openr1_dataloader

from _ppo.PolicyValueNNTok import PPOQLoRAWithValueHead
from _ppo.PPOAgentTok import PPOAgent


def _prep_tokenizer(tok):
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def build_ref_model_4bit(model_name: str):
    """Încărcare policy_ref înghețat în 4-bit NF4 pentru economie de VRAM."""
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    ref = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_cfg,
        device_map="auto",
    )
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False
    return ref


def main():
    # (opțional, reduce fragmentarea pe GPU moderne)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    cfg = Config()

    # === DataLoader + tokenizer (alege dataset-ul tău) ===
    dataloader, tokenizer = build_openr1_dataloader(
        tokenizer_name=cfg.policy_model_name,
        split=cfg.split,
        batch_size=cfg.batch_size,
        policy_max_length=cfg.policy_max_length,
        prompt_ratio_range=cfg.prompt_ratio_range,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )
    tokenizer = _prep_tokenizer(tokenizer)

    total_steps = cfg.n_epochs * len(dataloader) * cfg.ppo_epochs

    # === Policy model (QLoRA) ===
    if getattr(cfg, "resume_dir", None):
        print(f"[LOAD] Resuming PPO from: {cfg.resume_dir}")
        policy_model = PPOQLoRAWithValueHead.from_pretrained(
            cfg.resume_dir,
            device_map="auto",
            use_gradient_checkpointing=True,
            use_cache=False,
            value_hidden_dim=getattr(cfg, "value_hidden_dim", 512),
        )
    else:
        sft_dir = getattr(cfg, "sft_dir", getattr(cfg, "lora_from_sft_dir", None))
        if sft_dir:
            print(f"[LOAD] Starting PPO from SFT adapters: {sft_dir}")
        policy_model = PPOQLoRAWithValueHead(
            model_name=cfg.policy_model_name,
            r=getattr(cfg, "lora_r", 16),
            lora_alpha=getattr(cfg, "lora_alpha", 32),
            lora_dropout=getattr(cfg, "lora_dropout", 0.05),
            value_hidden_dim=getattr(cfg, "value_hidden_dim", 512),
            use_gradient_checkpointing=True,
            use_cache=False,
            device_map="auto",
            lora_from_sft_dir=sft_dir,
            load_value_head_from=getattr(cfg, "resume_dir", None),
        )
    policy_model.to(cfg.device)

    # === Policy ref (4-bit, frozen) ===
    policy_ref = build_ref_model_4bit(cfg.policy_model_name)

    # === Reward model + tokenizer ===
    reward_model = AutoModelForSequenceClassification.from_pretrained(cfg.reward_model_name)
    reward_tokenizer = AutoTokenizer.from_pretrained(cfg.reward_model_name)
    reward_model.to(getattr(cfg, "reward_device", cfg.device))
    reward_model.eval()

    # === Optimizer & Scheduler (LoRA + value head) ===
    trainable = [p for p in policy_model.parameters() if p.requires_grad]
    adamw_kwargs = {"lr": cfg.lr}
    try:
        if torch.cuda.is_available():
            adamw_kwargs["fused"] = True
    except TypeError:
        pass
    optimizer = AdamW(trainable, **adamw_kwargs)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=total_steps,
    )

    # === PPO Agent ===
    agent = PPOAgent(
        policy_model=policy_model,
        policy_ref=policy_ref,
        reward_model=reward_model,
        reward_tokenizer=reward_tokenizer,
        optimizer=optimizer,
        tokenizer=tokenizer,
        config=cfg,
        scheduler=scheduler,
    )

    # === Train ===
    agent.train(dataloader)

    # === Save last checkpoint (adapters + value_head) ===
    out_dir = os.path.join(getattr(cfg, "save_dir", "rl_ppo_qwen"), "checkpoint_last")
    os.makedirs(out_dir, exist_ok=True)
    policy_model.save_pretrained(out_dir)
    try:
        tokenizer.save_pretrained(out_dir)
    except Exception:
        pass
    print(f"[SAVE] Saved last checkpoint to: {out_dir}")


if __name__ == "__main__":
    main()
