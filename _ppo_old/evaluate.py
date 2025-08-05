import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, GPT2LMHeadModel
from _ppo_old.PolicyValueNN import GPT2WithValueHead
from datasets import load_dataset
from tqdm import tqdm
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np


device = "cuda"

# === Load tokenizers and models ===
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

ppo_model = GPT2WithValueHead(
    model_name="gpt2",
    value_hidden_dim=256,
    use_gradient_checkpointing=False,
    use_cache=True
)
ppo_model.load_pretrained("./gpt2_rl_best")
ppo_model.to(device)
ppo_model.eval()

baseline_model = GPT2LMHeadModel.from_pretrained("lvwerra/gpt2-imdb")
baseline_model.to(device)
baseline_model.eval()

reward_model = AutoModelForSequenceClassification.from_pretrained(
    "cardiffnlp/twitter-roberta-base-sentiment-latest"
)
reward_model.to(device)
reward_model.eval()
reward_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

# === Load dataset ===
dataset = load_dataset("imdb", split="test[:1%]")  # smaller for test

# === Prompt extractor ===
def extract_prompt(example, ratio=0.2):
    tokens = tokenizer.encode(example["text"])
    cut_idx = int(len(tokens) * ratio)
    prompt = tokenizer.decode(tokens[:cut_idx])
    return {"prompt": prompt}

dataset = dataset.map(extract_prompt)
prompts = dataset["prompt"]

# === Evaluation loop ===
ppo_rewards = []
baseline_rewards = []

for prompt in tqdm(prompts, desc="Evaluating"):

    # Generate from PPO
    ppo_inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    # batch = {}
    # batch['input_ids'] = ppo_inputs['input_ids'].to(device)
    # batch['attention_mask'] = ppo_inputs['attention_mask'].to(device)
    with torch.no_grad():
        ppo_output = ppo_model.generate(
            **ppo_inputs,
            max_new_tokens=128,
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
        )
    ppo_text = tokenizer.decode(ppo_output[0], skip_special_tokens=True)

    # Estimate value for the generated completion
    with torch.no_grad():
        val_inputs = tokenizer(ppo_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        ppo_value = ppo_model.value_forward(val_inputs["input_ids"], val_inputs["attention_mask"])
    tqdm.write(f"   Estimated value: {ppo_value.item():.3f}")

    # Generate from baseline
    base_inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        base_output = baseline_model.generate(
            **base_inputs,
            max_new_tokens=128,
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
        )
    base_text = tokenizer.decode(base_output[0], skip_special_tokens=True)

    # Compute rewards
    with torch.no_grad():
        ppo_reward_inputs = reward_tokenizer(ppo_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        base_reward_inputs = reward_tokenizer(base_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

        ppo_logits = reward_model(**ppo_reward_inputs).logits
        base_logits = reward_model(**base_reward_inputs).logits

        ppo_reward = F.softmax(ppo_logits, dim=-1)[0, 1].item()
        base_reward = F.softmax(base_logits, dim=-1)[0, 1].item()

    ppo_rewards.append(ppo_reward)
    baseline_rewards.append(base_reward)

    tqdm.write(f"→ Prompt: {prompt[:40]}...")
    tqdm.write(f"   PPO:      {ppo_text[len(prompt):][:80]} → Reward: {ppo_reward:.3f}")
    tqdm.write(f"   Baseline: {base_text[len(prompt):][:80]} → Reward: {base_reward:.3f}")
    tqdm.write("-" * 80)

# === Final summary ===
mean_ppo = sum(ppo_rewards) / len(ppo_rewards)
mean_base = sum(baseline_rewards) / len(baseline_rewards)

print("\n✅ Final Comparison:")
print(f"Mean PPO Reward:      {mean_ppo:.4f}")
print(f"Mean Baseline Reward: {mean_base:.4f}")
print(f"Improvement:          {mean_ppo - mean_base:+.4f}")


# Afișează doar primele 10 exemple
N = 10
indices = np.arange(N)
width = 0.35

plt.figure(figsize=(12, 6))
plt.bar(indices, baseline_rewards[:N], width, label='Baseline GPT-2')
plt.bar(indices + width, ppo_rewards[:N], width, label='PPO-tuned GPT-2')

# Adaugă etichete deasupra barelor
for i in range(N):
    plt.text(i, baseline_rewards[i] + 0.01, f"{baseline_rewards[i]:.2f}", ha='center', fontsize=8)
    plt.text(i + width, ppo_rewards[i] + 0.01, f"{ppo_rewards[i]:.2f}", ha='center', fontsize=8)

# Etichete și titlu
plt.xlabel("Exemplu")
plt.ylabel("Sentiment reward")
plt.title("Comparatie PPO vs GPT-2 (reward sentiment pentru completări)")
plt.xticks(indices + width / 2, [f"Ex {i}" for i in range(N)])
plt.ylim(0, 1.1)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()