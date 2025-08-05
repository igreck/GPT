import os
import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from _ppo.config import Config
from _ppo.PolicyValueNN import PPOQLoRAWithValueHead  # adapt modulul tău la Qwen


# ========= CONFIG =========
config = Config()
# Asigură-te că ai în config un atribut policy_model_name și calea către modelul PPO antrenat

# PROMPTS de test user
PROMPTS_USER = [
    "The movie starts as a heartfelt drama about family, but",
    "I didn't expect much from this film, however",
    "The plot was thin at first, yet by the end",
    "This film was so bad that I couldn't even finish it,",
]

# ========= MODELE POLICY / BASELINE =========
# Tokenizer pentru policy și baseline (Qwen)
tok = AutoTokenizer.from_pretrained(config.policy_model_name)
tok = tok if tok.pad_token else tok.add_special_tokens({'pad_token': tok.eos_token})
tok.padding_side = "left"

# Încarcă checkpoint-ul PPO antrenat
policy_model = PPOQLoRAWithValueHead.from_pretrained(
    getattr(config, 'resume_dir', './models/ppo_imdb_qlora_qwen_best_iter2'),
    device_map="auto",
    use_gradient_checkpointing=False,
    use_cache=True,
    value_hidden_dim=getattr(config, "value_hidden_dim", 512),

)
policy_model.to(config.device).eval()

# Baseline Qwen standard (fără value head)
baseline_model = AutoModelForCausalLM.from_pretrained(config.policy_model_name)
baseline_model.to(config.device).eval()

# ========= REWARD MODEL =========
reward_model = AutoModelForSequenceClassification.from_pretrained(config.reward_model_name)
reward_tokenizer = AutoTokenizer.from_pretrained(config.reward_model_name)
reward_model.to(config.device).eval()

# Măsurare performanță
def reset_stats():
    return {'ppo_gen_times': [], 'baseline_gen_times': [], 'gpu_mem_peak': []}
perf_stats = reset_stats()

@torch.inference_mode()
def generate_and_time(model, tokenizer, prompt):
    inputs = tokenizer(
        [prompt], return_tensors="pt", padding=True, truncation=True,
        max_length=config.policy_max_length
    ).to(config.device)
    # Reset memorie și sincronizare
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.time()
    out = model.generate(
        **inputs,
        max_new_tokens=config.max_new_tokens,
        do_sample=True,
        top_p=config.top_p,
        temperature=config.temperature,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1024**2
    else:
        peak = 0.0
    end = time.time()
    duration = end - start
    # Stocare în statistici
    if isinstance(model, PPOQLoRAWithValueHead):
        perf_stats['ppo_gen_times'].append(duration)
    else:
        perf_stats['baseline_gen_times'].append(duration)
    perf_stats['gpu_mem_peak'].append(peak)
    # Extrage completarea
    seq = out.sequences[0]
    prompt_len = inputs['attention_mask'].sum().item()
    completion_ids = seq[prompt_len:]
    return tokenizer.decode(completion_ids, skip_special_tokens=True)

@torch.inference_mode()
def compute_reward(text):
    inputs = reward_tokenizer([text], return_tensors="pt",
        padding=True, truncation=True, max_length=config.reward_max_length
    ).to(config.device)
    logits = reward_model(**inputs).logits
    probs = F.softmax(logits, dim=-1)
    # index pentru eticheta pozitivă
    pos_idx = getattr(reward_model.config, 'id2label', {0:'NEGATIVE', 1:'POSITIVE'})
    # asumăm poziția 1 pentru "POSITIVE"
    return probs[:,1].item()

# ========= Evaluează =========
ppo_rewards, base_rewards = [], []
for prompt in tqdm(PROMPTS_USER, desc="Evaluating Qwen PPO vs Baseline"):
    comp_ppo = generate_and_time(policy_model, tok, prompt)
    comp_base = generate_and_time(baseline_model, tok, prompt)
    r_ppo = compute_reward(comp_ppo)
    r_base = compute_reward(comp_base)
    ppo_rewards.append(r_ppo)
    base_rewards.append(r_base)
    tqdm.write(f"Prompt: {prompt}")
    tqdm.write(f"  PPO  → {comp_ppo[:100]}... | Reward: {r_ppo:.3f} | Time: {perf_stats['ppo_gen_times'][-1]:.3f}s | Mem: {perf_stats['gpu_mem_peak'][-1]:.1f}MB")
    tqdm.write(f"  Base → {comp_base[:100]}... | Reward: {r_base:.3f} | Time: {perf_stats['baseline_gen_times'][-1]:.3f}s")
    tqdm.write('-'*80)

# ========= Rezultate și grafic =========
mean_ppo = np.mean(ppo_rewards)
mean_base = np.mean(base_rewards)
mean_time_ppo = np.mean(perf_stats['ppo_gen_times'])
mean_time_base = np.mean(perf_stats['baseline_gen_times'])
mean_mem = np.mean(perf_stats['gpu_mem_peak'])
print(f"Mean PPO Reward: {mean_ppo:.4f} | Mean Base Reward: {mean_base:.4f}")
print(f"Avg PPO Time: {mean_time_ppo:.3f}s | Avg Base Time: {mean_time_base:.3f}s | Avg GPU Mem: {mean_mem:.1f}MB")

# Plotează timpii de generare
N = len(PROMPTS_USER)
idx = np.arange(N)
w = 0.35
plt.figure(figsize=(10,5))
plt.bar(idx, perf_stats['baseline_gen_times'], w, label='Baseline')
plt.bar(idx+w, perf_stats['ppo_gen_times'], w, label='PPO')
plt.xlabel('Exemplu')
plt.ylabel('Timp generare (s)')
plt.title('Qwen PPO vs Baseline Gen Time')
plt.xticks(idx+w/2, [f"Ex{i}" for i in range(N)])
plt.legend()
plt.grid(axis='y', ls='--', alpha=0.5)
plt.tight_layout()
plt.savefig("qwen_perf_user_prompts.png", dpi=300)
plt.show()

# ======== Plot Mean Positive Probability with SEM ========
labels = ["BASE", "PPO"]
means = [np.mean(base_rewards), np.mean(ppo_rewards)]
sems = [np.std(base_rewards)/np.sqrt(len(base_rewards)), np.std(ppo_rewards)/np.sqrt(len(ppo_rewards))]

plt.figure(figsize=(6,5))
plt.bar(labels, means, yerr=sems, capsize=5, color=['blue', 'orange'])
plt.ylabel('Mean Positive Probability')
plt.title("Base vs PPO: P(positive) on Generated Text")
plt.grid(axis='y', ls='--', alpha=0.5)
plt.tight_layout()
plt.savefig("base_vs_ppo_positive.png", dpi=300)
plt.show()
