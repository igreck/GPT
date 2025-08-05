import torch
from transformers import GPT2Tokenizer
from _ppo_old.PolicyValueNN import GPT2WithValueHead

# 1. Inițializăm tokenizer și setăm pad_token
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# 2. Inițializăm modelul pe CPU sau CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT2WithValueHead(
    model_name="gpt2",
    value_hidden_dim=256,
    use_gradient_checkpointing=False,
    use_cache=True
).to(device)

# 3. Prompt de test
prompt = "The future of AI is"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# 4. Generare text
print("Generating...")
with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=40,
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
        output_scores=True,
        return_dict_in_generate=True
    )
    sequences = outputs.sequences

# 5. Decodificare și afișare rezultat
generated_text = tokenizer.decode(sequences[0], skip_special_tokens=True)
print("\n✅ Text generat:\n", generated_text)

# Test value estimation
print("Estimating value for each token of the prompt:")
with torch.no_grad():
    values = model.value_forward(inputs["input_ids"], inputs["attention_mask"])
print(values)