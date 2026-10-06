import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel


BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_MODEL = "models/python-tutor-qlora"


QUESTIONS = [
    "Qual é a diferença entre uma lista e uma tupla em Python?",
    "Explique o que é uma exceção em Python e mostre como usar try e except.",
    "O que é uma função lambda em Python? Mostre um exemplo.",
    "Explique o que é um dicionário em Python e como acessar seus valores.",
    "Qual é a diferença entre for e while em Python?",
    "Explique o que é uma classe em Python e mostre um exemplo simples.",
    "Como ler um arquivo de texto em Python de forma segura?",
    "O que é uma list comprehension? Mostre um exemplo.",
]


print("=" * 70)
print("AVALIAÇÃO DO MODELO TREINADO")
print("=" * 70)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA não está disponível.")

print("CUDA disponível: True")
print("GPU:", torch.cuda.get_device_name(0))


print("\nCarregando tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)


print("Carregando modelo base em 4-bit...")

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


print("Carregando LoRA treinado...")

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_MODEL,
)

model.eval()

print("LoRA carregado.")
print("Modelo pronto.")


for i, question in enumerate(QUESTIONS, start=1):

    print("\n" + "=" * 70)
    print(f"QUESTÃO {i}")
    print("=" * 70)

    print("Pergunta:")
    print(question)

    messages = [
        {
            "role": "system",
            "content": (
                "Você é um tutor especializado em programação Python. "
                "Explique os conceitos de forma clara, objetiva e correta. "
                "Sempre que útil, forneça exemplos de código."
            ),
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    input_length = inputs["input_ids"].shape[1]

    generated_tokens = outputs[0][input_length:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    print("\nResposta:")
    print(response)


print("\n" + "=" * 70)
print("GPU")
print("=" * 70)

print("GPU:", torch.cuda.get_device_name(0))

print(
    "VRAM utilizada:",
    round(torch.cuda.memory_allocated() / 1024**3, 2),
    "GB",
)

print("=" * 70)