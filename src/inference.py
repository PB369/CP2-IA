import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_MODEL = "models/python-tutor-qlora"

print("=" * 60)
print("TESTE DA LLM TREINADA COM QLoRA")
print("=" * 60)

print("CUDA disponível:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA não está disponível.")

print("GPU:", torch.cuda.get_device_name(0))

print("\nCarregando tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("Carregando modelo base em 4-bit...")

from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quantization_config,
    device_map="auto",
    dtype=torch.float16,
)

print("Modelo base carregado.")

print("\nCarregando LoRA treinado...")

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_MODEL
)

model.eval()

print("LoRA carregado.")
print("Modelo pronto!")

print("=" * 60)
print("TESTE")
print("=" * 60)

messages = [
    {
        "role": "system",
        "content": (
            "Você é um tutor especializado em programação Python. "
            "Explique os conceitos de forma clara, objetiva e com exemplos."
        )
    },
    {
        "role": "user",
        "content": "Explique o que é uma função em Python."
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
        top_p=0.9,
        do_sample=True
    )

input_length = inputs["input_ids"].shape[1]

generated_tokens = outputs[0][input_length:]

response = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True
)

print("\n" + "=" * 60)
print("RESPOSTA DO MODELO TREINADO")
print("=" * 60)

print(response)

print("\n" + "=" * 60)
print("INFORMAÇÕES DA GPU")
print("=" * 60)

print("GPU:", torch.cuda.get_device_name(0))
print(
    "VRAM utilizada:",
    round(torch.cuda.memory_allocated() / 1024**3, 2),
    "GB"
)

print("=" * 60)