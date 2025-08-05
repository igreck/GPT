import os
import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from _ppo.config import Config
from _ppo.PolicyValueNN import PPOQLoRAWithValueHead  # adapta modulul tău la Qwen

# ========= CONFIG =========
config = Config()
# Asigură-te că ai în config un atribut policy_model_name, ex: "Qwen/Qwen3-1.7B"

# PROMPTS de test user
PROMPTS_USER = [
    "The movie starts as a heartfelt drama about family, but",
    "I didn't expect much from this film, however",
    "The plot was thin at first, yet by the end",
    "This film was so bad that I couldn't even finish it,",
]

# ========= MODELE POLICY/Baseline (Qwen) =========
# Tokenizer Qwen
tok = AutoTokenizer.from_pretrained(config.policy_model_name)
tok.pad_token = tok.eos_token
tok.padding_side = "left"

# Policy cu Value Head
policy_model = PPOQLoRAWithValueHead(
    model_name=config.policy_model_name,
    value_hidden_dim=config.value_hidden_dim,
    use_gradient_checkpointing=False,
    use_cache=True
)
policy_model = policy_model.from_pretrained("./models/ppo_imdb_qlora_qwen_best_iter2")
policy_model.to(config.device).eval()

# Baseline Qwen standard
baseline_model = AutoModelForCausalLM.from_pretrained(config.policy_model_name)
baseline_model.to(config.device).eval()

# ========= REWARD MODEL: distilbert-imdb =========
reward_model = AutoModelForSequenceClassification.from_pretrained(config.reward_model_name)
reward_model.to(config.device).eval()
reward_tok = AutoTokenizer.from_pretrained(config.reward_model_name)

# Clase distilbert-imdb
id2label = getattr(reward_model.config, "id2label", {0:"NEGATIVE", 1:"POSITIVE"})
inv = {v.lower(): k for k, v in id2label.items()}
POS_IDX = inv.get("positive", 1)
NEG_IDX = inv.get("negative", 0)

# Timing / Performance storage
perf_stats = {
    'ppo_gen_times': [],
    'baseline_gen_times': [],
    'gpu_mem_peak': []
}

@torch.inference_mode()
def generate_and_time(model, tokenizer, prompt_texts):
    inputs = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.policy_max_length,
    ).to(config.device)
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
    torch.cuda.synchronize()
    end = time.time()
    gen_time = end - start
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    perf_stats['gpu_mem_peak'].append(peak_mem)
    perf_stats['ppo_gen_times'].append(gen_time) if isinstance(model, PPOQLoRAWithValueHead) else perf_stats['baseline_gen_times'].append(gen_time)

    seqs = out.sequences
    prompt_lens = inputs['attention_mask'].sum(dim=1)
    completions = []
    for i in range(seqs.size(0)):
        gen_only_ids = seqs[i, int(prompt_lens[i].item()):]
        completions.append(tokenizer.decode(gen_only_ids, skip_special_tokens=True))
    return completions[0]

@torch.inference_mode()
def reward_scores_imdb(texts, use_margin=False):
    inp = reward_tok(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.reward_max_length,
    ).to(config.device)
    logits = reward_model(**inp).logits
    if use_margin:
        margin = logits[:, POS_IDX] - logits[:, NEG_IDX]
        return torch.sigmoid(margin).cpu().tolist()
    probs = F.softmax(logits, dim=-1)
    return probs[:, POS_IDX].cpu().tolist()

# ========= Evaluare =========
ppo_rewards, base_rewards = [], []
for prompt in tqdm(PROMPTS_USER, desc="Evaluating Qwen Performance"):
    # Policy generate + time
    ppo_comp = generate_and_time(policy_model, tok, [prompt])
    # Baseline generate + time
    base_comp = generate_and_time(baseline_model, tok, [prompt])

    # Reward
    ppo_r  = reward_scores_imdb([ppo_comp])[0]
    base_r = reward_scores_imdb([base_comp])[0]
    ppo_rewards.append(ppo_r)
    base_rewards.append(base_r)

    tqdm.write(f"→ Prompt: {prompt}")
    tqdm.write(f"   Policy:   {ppo_comp[:100]}... | Reward: {ppo_r:.3f} | Gen Time: {perf_stats['ppo_gen_times'][-1]:.3f}s | GPU Peak: {perf_stats['gpu_mem_peak'][-1]:.1f}MB")
    tqdm.write(f"   Baseline: {base_comp[:100]}... | Reward: {base_r:.3f} | Gen Time: {perf_stats['baseline_gen_times'][-1]:.3f}s")
    tqdm.write('-'*80)

# ========= Rezumat & Grafic =========
mean_ppo_reward = np.mean(ppo_rewards)
mean_base_reward = np.mean(base_rewards)
mean_ppo_time = np.mean(perf_stats['ppo_gen_times'])
mean_base_time = np.mean(perf_stats['baseline_gen_times'])
mean_peak_mem = np.mean(perf_stats['gpu_mem_peak'])

print(f"Mean Policy Reward: {mean_ppo_reward:.4f}")
print(f"Mean Baseline Reward: {mean_base_reward:.4f}")
print(f"Mean Policy Gen Time: {mean_ppo_time:.3f}s")
print(f"Mean Baseline Gen Time: {mean_base_time:.3f}s")
print(f"Average GPU Peak Mem: {mean_peak_mem:.1f}MB")

# Plotting
N = len(PROMPTS_USER)
idx = np.arange(N)
w = 0.3
plt.figure(figsize=(12,6))
plt.bar(idx, perf_stats['baseline_gen_times'], w, label='Baseline Gen Time')
plt.bar(idx+w, perf_stats['ppo_gen_times'], w, label='Policy Gen Time')
plt.ylabel('Generation Time (s)')
plt.xlabel('Example')
plt.title('Qwen PPO vs Baseline Generation Time on User Prompts')
plt.xticks(idx+w/2, [f"Ex{i}" for i in range(N)])
plt.legend()
plt.grid(axis='y', ls='--', alpha=0.5)
plt.tight_layout()
plt.savefig("qwen_perf_user_prompts.png", dpi=300)
plt.show()