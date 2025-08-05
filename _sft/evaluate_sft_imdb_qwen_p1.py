# gen_and_score.py
import os, json, math, random, gc, time, argparse
from dataclasses import dataclass
from typing import List, Dict
from datetime import datetime

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from unsloth import FastLanguageModel
import numpy as np

@dataclass
class CFG:
    base_name: str = os.environ.get("BASE_NAME", "unsloth/Qwen3-1.7B-Base")
    sft_dir: str = os.environ.get("SFT_DIR", "./sft_imdb_qlora_qwen")
    which: str = os.environ.get("WHICH", "sft")  # "base" sau "sft"
    out_dir: str = os.environ.get("OUT_DIR", "./eval_out")
    # data
    n_prompts: int = int(os.environ.get("N_PROMPTS", 10))
    prompt_prefix_chars: int = int(os.environ.get("PROMPT_PREFIX_CHARS", 300))
    seed: int = int(os.environ.get("SEED", 42))
    # generation
    max_length: int = int(os.environ.get("MAX_LEN", 512))
    max_new_tokens: int = int(os.environ.get("MAX_NEW_TOKENS", 96))
    do_sample: bool = True
    top_p: float = 0.9
    temperature: float = 0.8
    # classifier
    classifier_name: str = os.environ.get("CLASSIFIER_NAME", "lvwerra/distilbert-imdb")
    classifier_device: str = os.environ.get("CLASSIFIER_DEVICE", "cpu")  # "cpu" sau "cuda"

def set_seed(s: int):
    random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def _prep_tokenizer(tok):
    if tok.eos_token is None: tok.eos_token = "<|endoftext|>"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok

def load_classifier(cfg: CFG):
    tok = AutoTokenizer.from_pretrained(cfg.classifier_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(cfg.classifier_name)
    dev = cfg.classifier_device if (cfg.classifier_device=="cuda" and torch.cuda.is_available()) else "cpu"
    mdl.to(dev).eval()
    return mdl, tok, dev

def batched(xs, bs):
    b=[]
    for x in xs:
        b.append(x)
        if len(b)==bs: yield b; b=[]
    if b: yield b

@torch.inference_mode()
def sentiment_scores(texts: List[str], clf, clf_tok, device, batch_size=16) -> List[float]:
    probs=[]
    for batch in batched(texts, batch_size):
        enc = clf_tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        logits = clf(**enc).logits
        p = torch.softmax(logits, dim=-1)[:, -1]  # pozitive index (-1) pentru IMDB
        probs += p.detach().cpu().tolist()
    return probs

def load_lm(cfg: CFG):
    if cfg.which.lower()=="base":
        model, tok = FastLanguageModel.from_pretrained(
            model_name=cfg.base_name, max_seq_length=cfg.max_length,
            load_in_4bit=True, fast_inference=True, max_lora_rank=32, gpu_memory_utilization=0.9)
    elif cfg.which.lower()=="sft":
        base_model, tok = FastLanguageModel.from_pretrained(
            model_name=cfg.base_name, max_seq_length=cfg.max_length,
            load_in_4bit=True, fast_inference=True, max_lora_rank=32, gpu_memory_utilization=0.9)
        model = PeftModel.from_pretrained(base_model, cfg.sft_dir)
    else:
        raise ValueError("which must be 'base' or 'sft'")
    tok = _prep_tokenizer(tok)
    model.eval()
    return model, tok

def unload(*objs):
    for o in objs:
        try: del o
        except: pass
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def ensure_prompts(cfg: CFG) -> List[str]:
    os.makedirs(cfg.out_dir, exist_ok=True)
    p_path = os.path.join(cfg.out_dir, "prompts.json")
    if os.path.exists(p_path):
        with open(p_path, "r") as f:
            return json.load(f)
    # otherwise create deterministic prompts
    ds = load_dataset("imdb")["test"]
    rng = random.Random(cfg.seed)
    idxs = rng.sample(range(len(ds)), k=min(cfg.n_prompts, len(ds)))
    prompts = [ds[i]["text"][:cfg.prompt_prefix_chars] for i in idxs]
    with open(p_path, "w") as f: json.dump(prompts, f, ensure_ascii=False, indent=2)
    return prompts

@torch.inference_mode()
def generate_only_new(model, tok, prompt: str, cfg: CFG) -> str:
    dev = next(model.parameters()).device
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=cfg.max_length).to(dev)
    input_len = int(enc["attention_mask"].sum(dim=1).item())
    out = model.generate(
        **enc, max_new_tokens=cfg.max_new_tokens, do_sample=cfg.do_sample,
        top_p=cfg.top_p, temperature=cfg.temperature,
        eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
        return_dict_in_generate=True,
    )
    seq = out.sequences[0]
    new_tokens = seq[input_len:]
    return tok.decode(new_tokens, skip_special_tokens=True)

def evaluate_sft_imdb_qwen_p1():
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", choices=["base","sft"], default=None)
    args = parser.parse_args()
    cfg = CFG()
    if args.which: cfg.which = args.which
    set_seed(cfg.seed)

    prompts = ensure_prompts(cfg)
    print(f"[{cfg.which.upper()}] Using {len(prompts)} prompts. Out dir: {cfg.out_dir}")

    clf, clf_tok, clf_dev = load_classifier(cfg)

    model, tok = load_lm(cfg)
    gens = [generate_only_new(model, tok, p, cfg) for p in prompts]
    unload(model, tok)  # eliberăm VRAM

    pos = sentiment_scores(gens, clf, clf_tok, clf_dev)
    stats = {
        "name": cfg.which.upper(),
        "mean_pos": float(np.mean(pos)), "std_pos": float(np.std(pos)), "n": len(pos),
        "timestamp": datetime.utcnow().isoformat()+"Z",
        "base_name": cfg.base_name, "sft_dir": cfg.sft_dir,
        "max_new_tokens": cfg.max_new_tokens,
    }

    # salvăm rezultate
    res_path = os.path.join(cfg.out_dir, f"results_{cfg.which.lower()}.json")
    with open(res_path, "w") as f:
        json.dump({
            "cfg": cfg.__dict__,
            "prompts_path": os.path.join(cfg.out_dir, "prompts.json"),
            "generations": gens,
            "pos_probs": pos,
            "stats": stats,
        }, f, ensure_ascii=False, indent=2)

    # câteva mostre text
    txt_path = os.path.join(cfg.out_dir, f"samples_{cfg.which.lower()}.txt")
    with open(txt_path, "w") as f:
        for i in range(min(3, len(prompts))):
            f.write(f"\n=== {cfg.which.upper()} • Prompt {i+1} ===\n{prompts[i]}\n--- ONLY NEW ---\n{gens[i]}\n")

    print(f"[{cfg.which.upper()}] Saved:\n  {res_path}\n  {txt_path}")

if __name__ == "__main__":
    evaluate_sft_imdb_qwen_p1()