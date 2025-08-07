import os
import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from _ppo.config import Config
from _ppo.PolicyValueNN import PPOQLoRAWithValueHead  # adapt modulul
from _grpo.PolicyValueNN import GRPOQLoRA
import argparse


def main(method):
    config = Config()
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

    # Load selected policy model
    if method == 'grpo':
        policy_model = GRPOQLoRA.from_pretrained(
            getattr(config, 'resume_dir', './models/grpo_qlora_2025-06-08-17-16-29_best'),
            device_map='auto', use_gradient_checkpointing=False, use_cache=True,
        )
    else:
        policy_model = PPOQLoRAWithValueHead.from_pretrained(
            getattr(config, 'resume_dir', './models/ppo_imdb_qlora_qwen_best_iter2'),
            device_map='auto', use_gradient_checkpointing=False, use_cache=True,
            value_hidden_dim=getattr(config, 'value_hidden_dim', 512),
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
        return {'policy_gen_times': [], 'baseline_gen_times': [], 'gpu_mem_peak': []}
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
        if model is baseline_model:
            perf_stats['baseline_gen_times'].append(duration)
        else:
            perf_stats['policy_gen_times'].append(duration)
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
    policy_rewards, base_rewards = [], []
    for prompt in tqdm(PROMPTS_USER, desc=f"Evaluating {method.upper()} vs Baseline"):
        comp_pol  = generate_and_time(policy_model, tok, prompt)
        comp_base = generate_and_time(baseline_model, tok, prompt)
        r_pol  = compute_reward(comp_pol)
        r_base = compute_reward(comp_base)
        policy_rewards.append(r_pol)
        base_rewards.append(r_base)
        tqdm.write(f"Prompt: {prompt}")
        tqdm.write(f"  {method.upper():<6}→ {comp_pol[:100]}... | Reward: {r_pol:.3f} | Time: {perf_stats['policy_gen_times'][-1]:.3f}s | Mem: {perf_stats['gpu_mem_peak'][-1]:.1f}MB")
        tqdm.write(f"  BASE  → {comp_base[:100]}... | Reward: {r_base:.3f} | Time: {perf_stats['baseline_gen_times'][-1]:.3f}s")
        tqdm.write('-'*80)

    # ========= Rezultate și grafic =========
    mean_pol      = np.mean(policy_rewards)
    mean_base     = np.mean(base_rewards)
    mean_time_pol = np.mean(perf_stats['policy_gen_times'])
    mean_time_base= np.mean(perf_stats['baseline_gen_times'])
    mean_mem      = np.mean(perf_stats['gpu_mem_peak'])
    print(f"Mean {method.upper()} Reward: {mean_pol:.4f} | Mean Base Reward: {mean_base:.4f}")
    print(f"Avg {method.upper()} Time: {mean_time_pol:.3f}s | Avg Base Time: {mean_time_base:.3f}s | Avg GPU Mem: {mean_mem:.1f}MB")

    # Plotează timpii de generare
    N = len(PROMPTS_USER)
    idx = np.arange(N)
    w = 0.35
    plt.figure(figsize=(10,5))
    plt.bar(idx - w/2, perf_stats['baseline_gen_times'], w, label='Baseline')
    plt.bar(idx + w/2, perf_stats['policy_gen_times'],    w, label=method.upper())
    plt.xlabel('Exemplu')
    plt.ylabel('Timp generare (s)')
    plt.title(f"{method.upper()} vs Baseline Gen Time")
    plt.xticks(idx, [f"Ex{i}" for i in range(N)])
    plt.legend()
    plt.grid(axis='y', ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("qwen_perf_user_prompts.png", dpi=300)
    plt.show()

    # ======== Plot Mean Positive Probability with SEM ========
    labels = ['BASE', method.upper()]
    means  = [mean_base, mean_pol]
    sems   = [np.std(base_rewards)/np.sqrt(len(base_rewards)), np.std(policy_rewards)/np.sqrt(len(policy_rewards))]

    plt.figure(figsize=(6,5))
    plt.bar(labels, means, yerr=sems, capsize=5)
    plt.ylabel('Mean Positive Probability')
    plt.title(f"Base vs {method.upper()}: P(positive) on Generated Text")
    plt.grid(axis='y', ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"base_vs_{method}_positive.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main("ppo")