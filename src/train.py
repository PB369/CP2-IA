import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_FILE = BASE_DIR / "data" / "train.jsonl"
VALIDATION_FILE = BASE_DIR / "data" / "validation.jsonl"

MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"


# ============================================================
# ARGUMENTOS
# ============================================================

parser = argparse.ArgumentParser(
    description="Treinamento QLoRA do Tutor de Python"
)

parser.add_argument(
    "--experiment",
    type=str,
    required=True,
    choices=["A", "B"],
    help="Experimento a executar: A ou B",
)

parser.add_argument(
    "--learning-rate",
    type=float,
    required=True,
    help="Learning rate do experimento",
)

args = parser.parse_args()

EXPERIMENT = args.experiment
LEARNING_RATE = args.learning_rate

MODEL_OUTPUT_DIR = MODELS_DIR / f"experimento_{EXPERIMENT}"
RESULTS_OUTPUT_DIR = RESULTS_DIR / f"experimento_{EXPERIMENT}"

MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# INFORMAÇÕES DO EXPERIMENTO
# ============================================================

print("=" * 70)
print(f"EXPERIMENTO {EXPERIMENT}")
print("=" * 70)

print(f"Modelo:              {MODEL_NAME}")
print(f"Learning rate:       {LEARNING_RATE}")
print(f"Train dataset:       {TRAIN_FILE}")
print(f"Validation dataset:  {VALIDATION_FILE}")
print(f"Modelo de saída:     {MODEL_OUTPUT_DIR}")
print(f"Resultados:          {RESULTS_OUTPUT_DIR}")


# ============================================================
# VERIFICAÇÃO DA GPU
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA não está disponível. "
        "Este treinamento deve ser executado utilizando a GPU."
    )

GPU_NAME = torch.cuda.get_device_name(0)
GPU_MEMORY_GB = torch.cuda.get_device_properties(0).total_memory / (
    1024 ** 3
)

print()
print("GPU:")
print(f"  Nome:       {GPU_NAME}")
print(f"  VRAM:       {GPU_MEMORY_GB:.2f} GB")
print(f"  CUDA:       {torch.version.cuda}")
print(f"  PyTorch:    {torch.__version__}")
print()


# ============================================================
# DATASET
# ============================================================

class PythonTutorDataset(Dataset):

    def __init__(self, file_path, tokenizer, max_length=512):
        self.examples = []

        with open(file_path, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):

                line = line.strip()

                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Erro no JSON da linha {line_number} "
                        f"do arquivo {file_path}: {error}"
                    )

                self.examples.append(data)

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):

        example = self.examples[index]

        # ----------------------------------------------------
        # Compatibilidade com diferentes formatos de JSONL
        # ----------------------------------------------------

        if "messages" in example:

            messages = example["messages"]

        elif "instruction" in example:

            instruction = example["instruction"]

            input_text = example.get("input", "")
            output_text = example.get(
                "output",
                example.get("response", example.get("answer", ""))
            )

            user_content = instruction

            if input_text:
                user_content += f"\n\n{input_text}"

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
                    "content": user_content,
                },
                {
                    "role": "assistant",
                    "content": output_text,
                },
            ]

        else:

            question = (
                example.get("question")
                or example.get("prompt")
                or example.get("input")
                or ""
            )

            answer = (
                example.get("answer")
                or example.get("response")
                or example.get("output")
                or ""
            )

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
                {
                    "role": "assistant",
                    "content": answer,
                },
            ]

        # ----------------------------------------------------
        # Primeiro cria o texto utilizando o chat template
        # ----------------------------------------------------

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # ----------------------------------------------------
        # Depois realiza a tokenização normalmente
        # ----------------------------------------------------

        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        input_ids = torch.tensor(
            encoded["input_ids"],
            dtype=torch.long,
        )

        attention_mask = torch.tensor(
            encoded["attention_mask"],
            dtype=torch.long,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }

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
# DATASETS
# ============================================================

print("Carregando dataset de treinamento...")

train_dataset = PythonTutorDataset(
    TRAIN_FILE,
    tokenizer,
)

print("Carregando dataset de validação...")

validation_dataset = PythonTutorDataset(
    VALIDATION_FILE,
    tokenizer,
)

print()
print(f"Exemplos de treinamento: {len(train_dataset)}")
print(f"Exemplos de validação:   {len(validation_dataset)}")
print()


# ============================================================
# DATA COLLATOR
# ============================================================

def data_collator(features):

    max_length = max(
        feature["input_ids"].size(0)
        for feature in features
    )

    input_ids = []
    attention_masks = []
    labels = []

    for feature in features:

        current_length = feature["input_ids"].size(0)
        padding_length = max_length - current_length

        input_id = torch.nn.functional.pad(
            feature["input_ids"],
            (0, padding_length),
            value=tokenizer.pad_token_id,
        )

        attention_mask = torch.nn.functional.pad(
            feature["attention_mask"],
            (0, padding_length),
            value=0,
        )

        label = torch.nn.functional.pad(
            feature["labels"],
            (0, padding_length),
            value=-100,
        )

        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        labels.append(label)

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
        "labels": torch.stack(labels),
    }


# ============================================================
# QUANTIZAÇÃO 4-BIT
# ============================================================

print("Configurando quantização 4-bit NF4...")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)


# ============================================================
# MODELO
# ============================================================

print("Carregando modelo base...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    device_map="auto",
    dtype=torch.float16,
    trust_remote_code=True,
)

model.config.use_cache = False

# Necessário para treinamento QLoRA
model = prepare_model_for_kbit_training(model)


# ============================================================
# LoRA
# ============================================================

print("Configurando LoRA...")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(
    model,
    lora_config,
)

model.print_trainable_parameters()


# ============================================================
# TRAINING ARGUMENTS
# ============================================================

training_args = TrainingArguments(
    output_dir=str(MODEL_OUTPUT_DIR),

    num_train_epochs=1,

    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,

    gradient_accumulation_steps=8,

    learning_rate=LEARNING_RATE,

    fp16=True,

    gradient_checkpointing=True,

    logging_strategy="steps",
    logging_steps=1,

    save_strategy="epoch",
    eval_strategy="epoch",

    save_total_limit=2,

    report_to="none",

    remove_unused_columns=False,

    dataloader_pin_memory=True,

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    optim="paged_adamw_8bit",

    seed=42,
)


# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(
    model=model,
    args=training_args,

    train_dataset=train_dataset,
    eval_dataset=validation_dataset,

    data_collator=data_collator,

)


# ============================================================
# TREINAMENTO
# ============================================================

print()
print("=" * 70)
print("INICIANDO TREINAMENTO")
print("=" * 70)

start_time = time.time()

train_result = trainer.train()

training_time = time.time() - start_time

print()
print("=" * 70)
print("TREINAMENTO FINALIZADO")
print("=" * 70)

print(f"Tempo total: {training_time:.2f} segundos")


# ============================================================
# AVALIAÇÃO DO DATASET DE VALIDAÇÃO
# ============================================================

print()
print("Executando avaliação no conjunto de validação...")

eval_metrics = trainer.evaluate()

print()
print("MÉTRICAS DE VALIDAÇÃO:")

for key, value in eval_metrics.items():
    print(f"{key}: {value}")


# ============================================================
# SALVAR MODELO FINAL
# ============================================================

print()
print("Salvando modelo final...")

trainer.save_model(str(MODEL_OUTPUT_DIR))
tokenizer.save_pretrained(str(MODEL_OUTPUT_DIR))


# ============================================================
# COLETAR MÉTRICAS
# ============================================================

train_metrics = train_result.metrics

metrics = {
    "experiment": EXPERIMENT,

    "model": MODEL_NAME,

    "learning_rate": LEARNING_RATE,

    "epochs": 1,

    "train_samples": len(train_dataset),
    "validation_samples": len(validation_dataset),

    "batch_size": 1,
    "gradient_accumulation_steps": 8,

    "lora": {
        "r": 8,
        "alpha": 16,
        "dropout": 0.05,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
    },

    "quantization": {
        "bits": 4,
        "type": "NF4",
        "double_quantization": True,
        "compute_dtype": "float16",
    },

    "gpu": {
        "name": GPU_NAME,
        "vram_gb": round(GPU_MEMORY_GB, 2),
        "cuda": torch.version.cuda,
        "pytorch": torch.__version__,
    },

    "training": {
        "train_loss": train_metrics.get("train_loss"),
        "train_runtime_seconds": train_metrics.get(
            "train_runtime",
            training_time,
        ),
        "train_samples_per_second": train_metrics.get(
            "train_samples_per_second"
        ),
        "epoch": train_metrics.get("epoch"),
    },

    "validation": {
        "eval_loss": eval_metrics.get("eval_loss"),
        "eval_runtime_seconds": eval_metrics.get("eval_runtime"),
        "eval_samples_per_second": eval_metrics.get(
            "eval_samples_per_second"
        ),
        "epoch": eval_metrics.get("epoch"),
    },

    "total_training_time_seconds": training_time,
}


# ============================================================
# SALVAR MÉTRICAS JSON
# ============================================================

metrics_file = RESULTS_OUTPUT_DIR / "training_metrics.json"

with open(
    metrics_file,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        metrics,
        file,
        indent=4,
        ensure_ascii=False,
    )


# ============================================================
# SALVAR LOG COMPLETO DO TRAINER
# ============================================================

log_file = RESULTS_OUTPUT_DIR / "training_log.json"

with open(
    log_file,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        trainer.state.log_history,
        file,
        indent=4,
        ensure_ascii=False,
    )


# ============================================================
# SALVAR CONFIGURAÇÃO DO EXPERIMENTO
# ============================================================

config = {
    "experiment": EXPERIMENT,
    "model": MODEL_NAME,
    "learning_rate": LEARNING_RATE,
    "epochs": 1,
    "train_samples": len(train_dataset),
    "validation_samples": len(validation_dataset),
    "batch_size": 1,
    "gradient_accumulation_steps": 8,
    "fp16": True,
    "gradient_checkpointing": True,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "quantization": "4-bit NF4",
    "gpu": GPU_NAME,
    "vram_gb": round(GPU_MEMORY_GB, 2),
    "cuda": torch.version.cuda,
    "pytorch": torch.__version__,
}

config_file = RESULTS_OUTPUT_DIR / "experiment_config.json"

with open(
    config_file,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        config,
        file,
        indent=4,
        ensure_ascii=False,
    )


print()
print("=" * 70)
print("ARQUIVOS GERADOS")
print("=" * 70)

print(f"Modelo:")
print(f"  {MODEL_OUTPUT_DIR}")

print()
print("Resultados:")
print(f"  {metrics_file}")
print(f"  {log_file}")
print(f"  {config_file}")

print()
print("=" * 70)
print(f"EXPERIMENTO {EXPERIMENT} CONCLUÍDO")
print("=" * 70)