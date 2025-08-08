import random
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from typing import Tuple
from _ppo.config import Config
import numpy as np


def build_openr1_dataloader(
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

    reasoning_start = "<start_working_out>" # Acts as <think>
    reasoning_end   = "<end_working_out>"   # Acts as </think>
    solution_start  = "<SOLUTION>"
    solution_end    = "</SOLUTION>"

    system_prompt = \
    f"""You are given a problem.
    Think about the problem and provide your working out.
    Place it between {reasoning_start} and {reasoning_end}.
    Then, provide your solution between {solution_start}{solution_end}"""

    chat_template = \
    "{% if messages[0]['role'] == 'system' %}"\
        "{{ messages[0]['content'] + eos_token }}"\
        "{% set loop_messages = messages[1:] %}"\
    "{% else %}"\
        "{{ '{system_prompt}' + eos_token }}"\
        "{% set loop_messages = messages %}"\
    "{% endif %}"\
    "{% for message in loop_messages %}"\
        "{% if message['role'] == 'user' %}"\
            "{{ message['content'] }}"\
        "{% elif message['role'] == 'assistant' %}"\
            "{{ message['content'] + eos_token }}"\
        "{% endif %}"\
    "{% endfor %}"\
    "{% if add_generation_prompt %}{{ '{reasoning_start}' }}"\
    "{% endif %}"

    # Replace with out specific template:
    chat_template = chat_template\
        .replace("'{system_prompt}'",   f"'{system_prompt}'")\
        .replace("'{reasoning_start}'", f"'{reasoning_start}'")
    tokenizer.chat_template = chat_template


    dataset = load_dataset("open-r1/DAPO-Math-17k-Processed", 'en', split=split)
    if shuffle: dataset = dataset.shuffle(seed=seed)


    def make_prompt(example):
        # Build chat messages for math problem
        problem_text = example.get('prompt', example.get('question', example.get('input', '')))
        messages = [
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": problem_text},
        ]
        return {"prompt": messages, "answer": example.get("solution")}
    

    dataset = dataset.map(make_prompt, desc="Building prompts")

    def tok_prompt(batch):
        # Tokenize chat messages via chat_template
        # batch['messages'] is a list of message lists
        token_lists = tokenizer.apply_chat_template(batch['prompt'], add_generation_prompt=True, tokenize=True)
        # Manually pad/truncate to policy_max_length
        input_ids, attention_mask = [], []
        for ids in token_lists:
            ids = ids[:policy_max_length]
            mask = [1]*len(ids)
            if len(ids) < policy_max_length:
                pad_len = policy_max_length - len(ids)
                ids = [tokenizer.pad_token_id]*pad_len + ids
                mask = [0]*pad_len + mask
            input_ids.append(ids)
            attention_mask.append(mask)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

    dataset = dataset.map(tok_prompt, batched=True, batch_size=map_batch_size, desc="Tokenizing prompts")
    dataset = dataset.remove_columns([c for c in dataset.column_names if c not in ("input_ids","attention_mask","labels", "prompt", "solution")])
    dataset.set_format(type="torch", columns=["input_ids","attention_mask","labels"], output_all_columns=True)

    print(tokenizer.decode(dataset[0]['input_ids']))

    # tokenized = dataset.map(
    # lambda x: {"tokens" : tokenizer.apply_chat_template(x["prompt"], add_generation_prompt = True, tokenize = True)},
    # batched = True,
    # )   
    # print(tokenizer.decode(tokenized[0]["tokens"]))
    # tokenized = tokenized.map(lambda x: {"L" : len(x["tokens"])})

    # maximum_length = int(np.quantile(tokenized["L"], 0.9))
    # print("Max Length = ", maximum_length)

    # # Filter only samples smaller than 90% max length
    # dataset = dataset.select(np.where(np.array(tokenized["L"]) <= maximum_length)[0])
    
    loader = DataLoader(
        dataset,
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


if __name__ == "__main__":
    cfg = Config()
        # === DataLoader + tokenizer (policy) ===
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

    for batch in dataloader:
        a = 2 + 3
        
