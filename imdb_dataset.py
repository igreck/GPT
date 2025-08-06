import random
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from typing import Tuple

def build_imdb_dataloader(
    tokenizer_name: str,
    split: str,
    batch_size: int,
    policy_max_length: int,
    prompt_ratio_range: Tuple[float, float] = (0.45, 0.7),
    shuffle: bool = True,
    num_workers: int = 0,
    seed: int = 42,
    map_batch_size: int = 256,
    drop_last: bool = False,
) -> Tuple[DataLoader, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Reproducibility
    random.seed(seed)
    torch.manual_seed(seed)

    ds = load_dataset("imdb", split=split)
    if shuffle: ds = ds.shuffle(seed=seed)

    def make_prompt(example):
        txt = example["text"]
        ids = tokenizer.encode(txt)
        if len(ids) < 8:
            prompt_ids = ids
        else:
            r_low, r_high = prompt_ratio_range
            cut = max(1, min(len(ids)-1, int(len(ids) * random.uniform(r_low, r_high))))
            prompt_ids = ids[:cut]
        raw = tokenizer.decode(prompt_ids, skip_special_tokens=True)
        instruction = "Write a positive review: "
        prompt = instruction + raw

        return {"prompt": prompt}
    

    ds = ds.map(make_prompt, desc="Building prompts")

    def tok_prompt(batch):
        toks = tokenizer(
            batch["prompt"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=policy_max_length,
        )
        toks.pop("token_type_ids", None)
        # prepare labels for supervised LM (ignore padding)
        labels = toks["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        return {
            "input_ids": toks["input_ids"],
            "attention_mask": toks["attention_mask"],
            "labels": labels,
            "text": batch["prompt"]
        }

    ds = ds.map(tok_prompt, batched=True, batch_size=map_batch_size, desc="Tokenizing prompts")
    ds = ds.remove_columns([c for c in ds.column_names if c not in ("input_ids","attention_mask","labels", "text")])
    ds.set_format(type="torch", columns=["input_ids","attention_mask","labels", "text"])

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(seed),
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
        prefetch_factor=2
    )
    return loader, tokenizer