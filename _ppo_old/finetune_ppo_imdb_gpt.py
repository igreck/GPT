import torch
from transformers import GPT2LMHeadModel, AutoModelForSequenceClassification, AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

from _ppo_old.config import Config
from imdb_dataset import build_imdb_dataloader
from _ppo.PPOAgent import PPOAgent
from _ppo_old.PolicyValueNN import GPT2WithValueHead

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

    # după ce ai construit dataloader-ul
    total_steps = cfg.n_epochs * len(dataloader) * cfg.ppo_epochs

    # Modele
    policy_model = GPT2WithValueHead(cfg.policy_model_name)
    policy_ref   = GPT2LMHeadModel.from_pretrained(cfg.policy_model_name)  # eval/frozen implicit in agent
    reward_model = AutoModelForSequenceClassification.from_pretrained(cfg.reward_model_name)
    reward_tokenizer = AutoTokenizer.from_pretrained(cfg.reward_model_name)

    # Optimizare
    optimizer = AdamW(policy_model.parameters(), lr=cfg.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=total_steps,
)

    agent = PPOAgent(
        policy_model=policy_model,
        policy_ref=policy_ref,
        reward_model=reward_model,
        reward_tokenizer=reward_tokenizer,
        optimizer=optimizer,
        tokenizer=tokenizer,
        config=cfg,
        scheduler=scheduler
    )
    agent.train(dataloader)

if __name__ == "__main__":
    main()