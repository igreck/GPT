import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from typing import List, Callable


def keyword_reward(keywords: List[str], bonus: float = 1.0) -> Callable[[List[str]], torch.Tensor]:
    """
    Returns a reward function that gives a bonus for occurrences of keywords in completions.
    """
    def fn(completions: List[str]) -> torch.Tensor:
        scores = []
        for text in completions:
            count = sum(text.lower().count(kw.lower()) for kw in keywords)
            scores.append(float(count) * bonus)
        return torch.tensor(scores)
    return fn


def sentiment_reward(
    reward_model: torch.nn.Module,
    reward_tokenizer: AutoTokenizer,
    config
) -> Callable[[List[str]], torch.Tensor]:
    """
    Returns a reward function that uses a sequence classification model to score sentiment.
    Maps positive probability to [-1,1].

    Args:
        reward_model: Pretrained classification model with id2label config.
        reward_tokenizer: Corresponding tokenizer.
        config: Config object providing device and reward_max_length.
    """
    device = getattr(config, 'reward_device', config.device)
    labels_map = getattr(reward_model.config, 'id2label', {})
    inv = {v.lower(): int(k) for k, v in labels_map.items()}
    pos_idx = inv.get('positive', 1)

    def fn(completions: List[str]) -> torch.Tensor:
        with torch.no_grad():
            inputs = reward_tokenizer(
                completions,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=config.reward_max_length
            ) # type: ignore
            inputs.pop('token_type_ids', None)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = reward_model(**inputs).logits
            probs = F.softmax(logits, dim=-1)
            raw = probs[:, pos_idx] * 2.0 - 1.0
        return raw
    return fn