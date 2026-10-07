import argparse
import csv
import json
import time
from pathlib import Path

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel


# ============================================================
# CONFIGURAÇÕES
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"


# ============================================================
# PERGUNTAS DE TESTE
# ============================================================

QUESTIONS = [
    {
        "id": 1,
        "topic": "Lista vs tupla",
        "question": (
            "Qual é a diferença entre uma lista e uma tupla em Python? "
            "Explique quando usar cada uma e mostre um exemplo."
        ),
    },

    {
        "id": 2,
        "topic": "Exceções",
        "question": (
            "Explique como funciona try e except em Python. "
            "Mostre um exemplo tratando uma divisão por zero."
        ),
    },

    {
        "id": 3,
        "topic": "Lambda",
        "question": (
            "O que é uma função lambda em Python? "
            "Explique e mostre um exemplo simples."
        ),
    },

    {
        "id": 4,
        "topic": "Dicionários",
        "question": (
            "O que é um dicionário em Python? "
            "Mostre como criar um dicionário, adicionar uma informação "
            "e acessar um valor através de uma chave."
        ),
    },

    {
        "id": 5,
        "topic": "For vs while",
        "question": (
            "Qual é a diferença entre os loops for e while em Python? "
            "Mostre um exemplo de cada."
        ),
    },

    {
        "id": 6,
        "topic": "Classes",
        "question": (
            "O que é uma classe em Python? "
            "Explique o conceito de objeto e mostre uma classe simples "
            "com um atributo e um método."
        ),
    },

    {
        "id": 7,
        "topic": "Arquivos",
        "question": (
            "Como ler um arquivo de texto de forma segura em Python? "
            "Mostre um exemplo utilizando with."
        ),
    },

    {
        "id": 8,
        "topic": "List comprehension",
        "question": (
            "O que é list comprehension em Python? "
            "Explique e mostre um exemplo que gere os quadrados "
            "dos números de 1 a 5."
        ),
    },
]


# ============================================================
# ARGUMENTOS
# ============================================================

parser = argparse.ArgumentParser(
    description="Avaliação qualitativa do modelo treinado"
)

parser.add_argument(
    "--experiment",
    type=str,
    required=True,
    choices=["A", "B"],
    help="Experimento a avaliar",
)

args = parser.parse_args()

EXPERIMENT = args.experiment

MODEL_DIR = MODELS_DIR / f"experimento_{EXPERIMENT}"
RESULTS_OUTPUT_DIR = RESULTS_DIR / f"experimento_{EXPERIMENT}"

RESULTS_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# VERIFICAÇÃO DA GPU
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA não está disponível. "
        "A avaliação deve ser executada utilizando a GPU."
    )

GPU_NAME = torch.cuda.get_device_name(0)
GPU_MEMORY_GB = (
    torch.cuda.get_device_properties(0).total_memory
    / (1024 ** 3)
)


# ============================================================
# INFORMAÇÕES
# ============================================================

print("=" * 70)
print(f"AVALIAÇÃO DO EXPERIMENTO {EXPERIMENT}")
print("=" * 70)

print(f"Modelo base: {MODEL_NAME}")
print(f"Adapter:     {MODEL_DIR}")
print(f"GPU:         {GPU_NAME}")
print(f"VRAM:        {GPU_MEMORY_GB:.2f} GB")
print(f"CUDA:        {torch.version.cuda}")
print(f"PyTorch:     {torch.__version__}")
print()


# ============================================================
# TOKENIZER
# ============================================================

print("Carregando tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# QUANTIZAÇÃO
# ============================================================

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)


# ============================================================
# MODELO BASE
# ============================================================

print("Carregando modelo base...")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    device_map="auto",
    dtype=torch.float16,
    trust_remote_code=True,
)


# ============================================================
# CARREGAR LoRA
# ============================================================

print("Carregando adapter LoRA...")

model = PeftModel.from_pretrained(
    base_model,
    str(MODEL_DIR),
)

model.eval()


# ============================================================
# FUNÇÃO DE GERAÇÃO
# ============================================================

def generate_response(question):

    messages = [
        {
            "role": "system",
            "content": (
                "Você é um tutor especializado em programação Python. "
                "Explique os conceitos de forma clara, didática e objetiva. "
                "Quando apropriado, forneça exemplos de código."
            ),
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    inputs = {
        key: value.to("cuda")
        for key, value in inputs.items()
    }

    with torch.no_grad():

        output = model.generate(
            **inputs,

            max_new_tokens=300,

            temperature=0.7,
            top_p=0.9,

            do_sample=True,

            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = output[0][
        inputs["input_ids"].shape[1]:
    ]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return response.strip()


# ============================================================
# AVALIAÇÃO
# ============================================================

results = []

print()
print("=" * 70)
print("GERANDO RESPOSTAS")
print("=" * 70)

for item in QUESTIONS:

    print()
    print("-" * 70)
    print(
        f"Pergunta {item['id']} - {item['topic']}"
    )
    print("-" * 70)

    print("Pergunta:")
    print(item["question"])

    print()
    print("Gerando resposta...")

    start_time = time.time()

    response = generate_response(
        item["question"]
    )

    generation_time = time.time() - start_time

    print()
    print("Resposta:")
    print(response)

    print()
    print(
        f"Tempo de geração: {generation_time:.2f} segundos"
    )

    results.append(
        {
            "id": item["id"],
            "topic": item["topic"],
            "question": item["question"],
            "response": response,
            "generation_time_seconds": round(
                generation_time,
                2,
            ),

            # Será preenchido manualmente
            "score": "",
            "observation": "",
        }
    )


# ============================================================
# SALVAR RESPOSTAS EM JSON
# ============================================================

json_file = (
    RESULTS_OUTPUT_DIR
    / "evaluation_results.json"
)

with open(
    json_file,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        {
            "experiment": EXPERIMENT,
            "model": MODEL_NAME,
            "gpu": GPU_NAME,
            "cuda": torch.version.cuda,
            "questions": results,
        },
        file,
        indent=4,
        ensure_ascii=False,
    )


# ============================================================
# SALVAR CSV
# ============================================================

csv_file = (
    RESULTS_OUTPUT_DIR
    / "evaluation_results.csv"
)

with open(
    csv_file,
    "w",
    encoding="utf-8-sig",
    newline="",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "id",
            "topic",
            "question",
            "response",
            "generation_time_seconds",
            "score",
            "observation",
        ],
    )

    writer.writeheader()

    writer.writerows(results)


# ============================================================
# INSTRUÇÕES DE AVALIAÇÃO
# ============================================================

instructions_file = (
    RESULTS_OUTPUT_DIR
    / "evaluation_instructions.txt"
)

with open(
    instructions_file,
    "w",
    encoding="utf-8",
) as file:

    file.write(
        f"""AVALIAÇÃO QUALITATIVA - EXPERIMENTO {EXPERIMENT}

Modelo:
{MODEL_NAME}

GPU:
{GPU_NAME}

CUDA:
{torch.version.cuda}


ESCALA DE AVALIAÇÃO

5 - Resposta correta, completa, clara e com exemplo adequado.

4 - Resposta correta, mas apresenta pequena imprecisão,
    omissão ou problema de clareza.

3 - Resposta parcialmente correta, mas possui alguma
    informação incompleta ou imprecisa.

2 - Resposta apresenta erros importantes, embora contenha
    algum conhecimento relacionado ao tema.

1 - Resposta majoritariamente incorreta ou pouco relacionada
    à pergunta.

0 - Não respondeu ou apresentou resposta completamente
    incorreta.


CRITÉRIOS

Considere:

1. Correção técnica
2. Clareza da explicação
3. Completude
4. Qualidade dos exemplos de código
5. Adequação ao nível de um tutor de Python


Após analisar as respostas em evaluation_results.csv,
preencha as colunas:

score
observation


Depois execute novamente o script com:

python src\\evaluate.py --experiment {EXPERIMENT}

ou utilize o arquivo evaluation_results.csv
para calcular a média das notas manualmente.
"""
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("AVALIAÇÃO CONCLUÍDA")
print("=" * 70)

print()
print("Arquivos gerados:")

print(f"1. {json_file}")
print(f"2. {csv_file}")
print(f"3. {instructions_file}")

print()
print("As respostas foram salvas sem alteração.")
print(
    "A coluna 'score' deve ser preenchida manualmente "
    "utilizando a escala de 0 a 5."
)

print()
print("=" * 70)