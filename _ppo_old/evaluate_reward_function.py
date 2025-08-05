# test_reward_model.py
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "lvwerra/distilbert-imdb"
MAX_LEN = 512

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_reward_model(model_name=MODEL_NAME, device=None):
    device = device or get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()

    # determină indexul pentru clasa „positive” din config
    id2label = getattr(model.config, "id2label", {0: "NEGATIVE", 1: "POSITIVE"})
    inv = {v.lower(): k for k, v in id2label.items()}
    pos_idx = inv.get("positive", 1)

    return tokenizer, model, device, pos_idx, id2label

@torch.inference_mode()
def score_texts(texts, tokenizer, model, device, pos_idx, max_length=MAX_LEN):
    # tokenizează cu padding+truncation
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    # (opțional) DistilBERT nu necesită token_type_ids; dacă apar, sunt ok.
    inputs = {k: v.to(device) for k, v in inputs.items()}
    logits = model(**inputs).logits                      # [B, 2]
    probs = F.softmax(logits, dim=-1)                   # [B, 2]
    p_pos = probs[:, pos_idx]                           # [B]
    return p_pos.cpu(), probs.cpu(), logits.cpu()

def pretty_print(texts, p_pos, probs, id2label):
    print("id2label:", id2label)
    for t, pp, pr in zip(texts, p_pos.tolist(), probs.tolist()):
        print("-" * 80)
        print(f"Text: {t[:200].replace('/n',' ')}{'...' if len(t)>200 else ''}")
        print(f"p(positive) = {pp:.3f} | probs = {pr}")

if __name__ == "__main__":
    # Exemple rapide
    sample_texts = [
        "I absolutely loved this movie. The performances were outstanding!",
        "This was a waste of time. Poor script and awful acting.",
        "It was okay overall, some parts were enjoyable but others were boring.",
    ]

    tokenizer, model, device, pos_idx, id2label = load_reward_model()
    p_pos, probs, logits = score_texts(sample_texts, tokenizer, model, device, pos_idx)
    pretty_print(sample_texts, p_pos, probs, id2label)