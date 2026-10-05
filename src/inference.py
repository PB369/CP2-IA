import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

print("=" * 50)
print("TESTE DA LLM")
print("=" * 50)

print("CUDA disponível:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA não está disponível. "
        "Verifique a instalação do PyTorch."
    )

print("GPU:", torch.cuda.get_device_name(0))

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Modelo carregado!")
print("Dispositivo:", model.device)

messages = [
    {
        "role": "system",
        "content": (
            "Você é um assistente especializado em "
            "programação Python."
        )
    },
    {
        "role": "user",
        "content": "Explique o que é uma lista em Python."
    }
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
)

inputs = {
    key: value.to(model.device)
    for key, value in inputs.items()
}

with torch.no_grad():

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9
    )

input_length = inputs["input_ids"].shape[1]

generated_tokens = outputs[0][input_length:]

response = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True
)

print()
print("=" * 50)
print("RESPOSTA")
print("=" * 50)
print(response)