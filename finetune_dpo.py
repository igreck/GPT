import os
import torch
import torch._dynamo
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.utils.quantization_config import BitsAndBytesConfig
from torch.optim import RMSprop

from __dpo.config_dpo import Config
from imdb_dataset import build_imdb_dataloader
from __dpo.DPOAgent import DPOAgent
from __dpo.PolicyValueNN import DPOQLoRA


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
    cfg = Config()

    # DataLoader + tokenizer (policy)
    dataloader, tokenizer = build_imdb_dataloader(
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

    # Total update steps for scheduler
    total_updates = cfg.n_epochs * len(dataloader) * cfg.update_epochs

    # Policy model
    if getattr(cfg, "resume_dir", None):
        print(f"[LOAD] Resuming from: {cfg.resume_dir}")
        policy_model = DPOQLoRA.from_pretrained(
            cfg.resume_dir,
            device_map="auto",
            use_gradient_checkpointing=True,
            use_cache=False,
        )
    else:
        policy_model = DPOQLoRA(
            model_name=cfg.policy_model_name,
            r=getattr(cfg, "lora_r", 16),
            lora_alpha=getattr(cfg, "lora_alpha", 32),
            lora_dropout=getattr(cfg, "lora_dropout", 0.05),
            use_gradient_checkpointing=True,
            use_cache=False,
            device_map="auto",
            lora_from_sft_dir=cfg.sft_dir,
        )
    policy_model.to(cfg.device)
    # torch._dynamo.config.dynamic_shapes = True
    # policy_model = torch.compile(policy_model, dynamic=True)

    # Policy reference
    policy_ref = build_ref_model_4bit(cfg.policy_model_name)

    # Optimizer & Scheduler
    trainable = [p for p in policy_model.parameters() if p.requires_grad]
    optimizer = RMSprop(trainable, lr=cfg.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=total_updates,
    )

    # DPO Agent
    agent = DPOAgent(
        policy_model=policy_model,
        policy_ref=policy_ref,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        config=cfg,
    )

    # Train
    agent.train(dataloader)

    # Save last checkpoint
    out_dir = os.path.join(cfg.save_dir, "checkpoint_last")
    os.makedirs(out_dir, exist_ok=True)
    policy_model.save_pretrained(out_dir)
    try:
        tokenizer.save_pretrained(out_dir)
    except:
        pass
    print(f"[SAVE] Saved last checkpoint to: {out_dir}")


if __name__ == "__main__":
    main()