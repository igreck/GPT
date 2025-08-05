# test_policy_qwen.py
import torch
from transformers import AutoTokenizer

# === Config ===
MODEL_ID   = "Qwen/Qwen3-1.7B"   # or "Qwen/Qwen3-1.7B-Base"
USE_QUANT  = True               # True if you have QLoRA class and want 4-bit
VALUE_HDIM = 512

# === Import a Qwen policy with value head ===
# Try quantized version first (if you created it), else fall back to full version.
policy = None
from _ppo.PolicyValueNN import QwenQLoRAWithValueHead as PolicyCls

# 1) Tokenizer (pad + left padding for decoder-only)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# 2) Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Quantized QLoRA variant typically uses device_map="auto"
model = PolicyCls(
    model_name=MODEL_ID,
    value_hidden_dim=VALUE_HDIM,
    use_gradient_checkpointing=False,  # not needed for a quick test
    device_map="auto",
)

# Ensure no-grad and faster gen
model.eval()
if hasattr(model, "model") and hasattr(model.model, "config"):
    # for decoder-only, disable cache warning during training; for this test we want speed
    model.model.config.use_cache = True

# 4) Prompt
prompt = "The future of AI is"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# 5) Generate
print("Generating with Qwen...")
with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=40,
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
        output_scores=True,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    sequences = outputs.sequences

# 6) Decode
generated_text = tokenizer.decode(sequences[0], skip_special_tokens=True)
print("\n✅ Generated text:\n", generated_text)

# 7) Value estimation (scalar per sequence, as in your PPO setup)
print("\nEstimating value for the input sequence:")
with torch.no_grad():
    values = model.value_forward(inputs["input_ids"], inputs["attention_mask"])
print(values)  # tensor([value])