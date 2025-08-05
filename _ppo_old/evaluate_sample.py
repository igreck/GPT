# eval_user_prompts_distilbert_imdb.py
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, GPT2LMHeadModel
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from _ppo_old.config import Config


# ========= CONFIG =========
config = Config()


# Pune aici prompturile tale (fallback: pune câteva manual)
PROMPTS_USER = [
    "The movie starts as a heartfelt drama about family, but",
    "I didn't expect much from this film, however",
    "The plot was thin at first, yet by the end",
]

# ========= MODELE POLICY/Baseline =========
# Tokenizer GPT-2
tok = AutoTokenizer.from_pretrained(config.policy_model_name)
tok.pad_token = tok.eos_token
tok.padding_side = "left"

# PPO cu value head (înlocuiește cu clasa ta dacă diferă)
from _ppo_old.PolicyValueNN import GPT2WithValueHead
ppo_model = GPT2WithValueHead(
    model_name=config.policy_model_name,
    value_hidden_dim=256,
    use_gradient_checkpointing=False,
    use_cache=True
)
ppo_model.load_pretrained("./gpt2_rl_best")
ppo_model.to(config.device).eval()

# Baseline GPT-2 pe IMDB
baseline_model = GPT2LMHeadModel.from_pretrained(config.policy_model_name).to(config.device).eval()

# ========= REWARD MODEL: lvwerra/distilbert-imdb (binar) =========
reward_name = config.reward_model_name
reward_model = AutoModelForSequenceClassification.from_pretrained(reward_name).to(config.device).eval()
reward_tok = AutoTokenizer.from_pretrained(reward_name)

# indexul clasei „POSITIVE” (de regulă 1)
id2label = getattr(reward_model.config, "id2label", {0:"NEGATIVE", 1:"POSITIVE"})
inv = {v.lower(): k for k, v in id2label.items()}
POS_IDX = inv.get("positive", 1)
NEG_IDX = inv.get("negative", 0)

@torch.inference_mode()
def generate_only_completion(model, tokenizer, prompt_texts):
    """Generează DOAR completarea (exclude promptul din reward)."""
    inputs = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.policy_max_length,
    ).to(config.device)

    out = model.generate(
        **inputs,
        max_new_tokens=config.max_new_tokens,
        do_sample=True,
        top_p=config.top_p,
        temperature=config.temperature,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
    )
    seqs = out.sequences
    prompt_lens = inputs["attention_mask"].sum(dim=1)  # [B]
    completions = []
    for i in range(seqs.size(0)):
        gen_only_ids = seqs[i, int(prompt_lens[i].item()):]
        completions.append(tokenizer.decode(gen_only_ids, skip_special_tokens=True))
    return completions

@torch.inference_mode()
def reward_scores_distilbert_imdb(texts, use_margin=False):
    """p(POSITIVE) în [0,1]; opțional variantă margin (mai contrastantă)."""
    inp = reward_tok(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.reward_max_length,
    ).to(config.device)
    logits = reward_model(**inp).logits  # [B,2]
    if use_margin:
        margin = logits[:, POS_IDX] - logits[:, NEG_IDX]
        return torch.sigmoid(margin).cpu().tolist()
    probs = F.softmax(logits, dim=-1)
    return probs[:, POS_IDX].cpu().tolist()

@torch.inference_mode()
def estimate_value(texts):
    """Estimare value head pe completare (dacă ai metoda value_forward)."""
    vals = []
    for t in texts:
        vi = tok(t, return_tensors="pt", padding=True, truncation=True, max_length=config.reward_max_length).to(config.device)
        v = ppo_model.value_forward(vi["input_ids"], vi["attention_mask"])
        vals.append(float(v.item()))
    return vals

# ========= Evaluare pe lista de prompturi =========
ppo_rewards, base_rewards, ppo_values = [], [], []

for prompt in tqdm(PROMPTS_USER, desc="Evaluating"):
    # PPO
    ppo_comp = generate_only_completion(ppo_model.model, tok, [prompt])[0]
    # Baseline
    base_comp = generate_only_completion(baseline_model, tok, [prompt])[0]

    # Rewards DOAR pe completare (alege use_margin=True dacă vrei)
    ppo_r  = reward_scores_distilbert_imdb([ppo_comp], use_margin=False)[0]
    base_r = reward_scores_distilbert_imdb([base_comp], use_margin=False)[0]

    # Value (opțional)
    try:
        ppo_val = estimate_value([ppo_comp])[0]
    except Exception:
        ppo_val = float("nan")

    ppo_rewards.append(ppo_r)
    base_rewards.append(base_r)
    ppo_values.append(ppo_val)

    tqdm.write(f"→ Prompt: {prompt[:60]}{'...' if len(prompt)>60 else ''}")
    tqdm.write(f"   PPO:      {ppo_comp[:100]}{'...' if len(ppo_comp)>100 else ''} | Reward: {ppo_r:.3f} | V≈ {ppo_val:.3f}")
    tqdm.write(f"   Baseline: {base_comp[:100]}{'...' if len(base_comp)>100 else ''} | Reward: {base_r:.3f}")
    tqdm.write("-"*80)

# ========= Rezumat + Grafic =========
mean_ppo = float(np.mean(ppo_rewards)) if ppo_rewards else 0.0
mean_base = float(np.mean(base_rewards)) if base_rewards else 0.0
print("\n✅ Final Comparison:")
print(f"Mean PPO Reward:      {mean_ppo:.4f}")
print(f"Mean Baseline Reward: {mean_base:.4f}")
print(f"Improvement:          {mean_ppo - mean_base:+.4f}")

N = min(10, len(ppo_rewards))
if N > 0:
    idx = np.arange(N)
    w = 0.35
    plt.figure(figsize=(12, 6))
    plt.bar(idx, base_rewards[:N], w, label='Baseline GPT-2')
    plt.bar(idx + w, ppo_rewards[:N], w, label='PPO-tuned')
    for i in range(N):
        plt.text(i, base_rewards[i] + 0.01, f"{base_rewards[i]:.2f}", ha='center', fontsize=8)
        plt.text(i + w, ppo_rewards[i] + 0.01, f"{ppo_rewards[i]:.2f}", ha='center', fontsize=8)
    plt.xlabel("Example")
    plt.ylabel("Sentiment reward (p(pos))")
    plt.title("PPO vs Baseline on user prompts (completion-only)")
    plt.xticks(idx + w/2, [f"Ex {i}" for i in range(N)])
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
    save_path = "ppo_vs_baseline_user_prompts.png"
    plt.savefig(save_path)
    