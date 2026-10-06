import json
import os

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)

from peft import (
    LoraConfig,
    get_peft_model,
)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

TRAIN_FILE = "data/train.jsonl"
VALIDATION_FILE = "data/validation.jsonl"

OUTPUT_DIR = "models/python-tutor-qlora"

MAX_LENGTH = 512


# ============================================================
# INÍCIO
# ============================================================

print("=" * 70)
print("TREINAMENTO DA LLM LOCAL - QLoRA")
print("=" * 70)

print()


# ============================================================
# VERIFICAÇÃO DA GPU
# ============================================================

if not torch.cuda.is_available():

    raise RuntimeError(
        "CUDA não está disponível. "
        "O treinamento deve utilizar a GPU."
    )


gpu_name = torch.cuda.get_device_name(0)

gpu_memory = (
    torch.cuda.get_device_properties(0).total_memory
    / (1024 ** 3)
)


print("CUDA disponível: SIM")
print(f"GPU: {gpu_name}")
print(f"VRAM: {gpu_memory:.2f} GB")
print(f"CUDA do PyTorch: {torch.version.cuda}")

print()


# ============================================================
# DATASET CUSTOMIZADO
# ============================================================

class PythonTutorDataset(Dataset):

    def __init__(
        self,
        file_path,
        tokenizer,
        max_length=512
    ):

        self.examples = []

        self.tokenizer = tokenizer

        self.max_length = max_length

        print(f"Carregando dataset: {file_path}")

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            for line_number, line in enumerate(
                file,
                start=1
            ):

                line = line.strip()

                if not line:
                    continue

                try:

                    data = json.loads(line)

                except json.JSONDecodeError as error:

                    print(
                        f"Erro no JSON da linha "
                        f"{line_number}: {error}"
                    )

                    continue


                if "messages" not in data:

                    print(
                        f"Aviso: linha {line_number} "
                        "não possui 'messages'."
                    )

                    continue


                messages = data["messages"]


                # ------------------------------------------------
                # Formato de conversa do Qwen
                # ------------------------------------------------

                text = tokenizer.apply_chat_template(

                    messages,

                    tokenize=False,

                    add_generation_prompt=False,
                )


                # ------------------------------------------------
                # Tokenização
                # ------------------------------------------------

                encoded = tokenizer(

                    text,

                    truncation=True,

                    max_length=max_length,

                    padding="max_length",
                )


                input_ids = encoded["input_ids"]

                attention_mask = encoded["attention_mask"]


                # ------------------------------------------------
                # Labels
                # ------------------------------------------------

                labels = input_ids.copy()


                # Ignora padding na loss

                labels = [

                    token if mask == 1 else -100

                    for token, mask
                    in zip(
                        labels,
                        attention_mask
                    )
                ]


                self.examples.append({

                    "input_ids": torch.tensor(

                        input_ids,

                        dtype=torch.long
                    ),

                    "attention_mask": torch.tensor(

                        attention_mask,

                        dtype=torch.long
                    ),

                    "labels": torch.tensor(

                        labels,

                        dtype=torch.long
                    ),
                })


        print(
            f"Exemplos carregados: "
            f"{len(self.examples)}"
        )


    def __len__(self):

        return len(self.examples)


    def __getitem__(self, index):

        return self.examples[index]


# ============================================================
# TOKENIZER
# ============================================================

print("Carregando tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


if tokenizer.pad_token is None:

    tokenizer.pad_token = tokenizer.eos_token


print("Tokenizer carregado.")

print()


# ============================================================
# DATASETS
# ============================================================

train_dataset = PythonTutorDataset(

    TRAIN_FILE,

    tokenizer,

    MAX_LENGTH,
)


validation_dataset = PythonTutorDataset(

    VALIDATION_FILE,

    tokenizer,

    MAX_LENGTH,
)


print()

print(
    "Exemplos de treinamento:",
    len(train_dataset)
)

print(
    "Exemplos de validação:",
    len(validation_dataset)
)

print()


# ============================================================
# QUANTIZAÇÃO 4-BIT
# ============================================================

print("Configurando quantização 4-bit...")

quantization_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_compute_dtype=torch.float16,

    bnb_4bit_use_double_quant=True,
)


print("Quantização 4-bit configurada.")

print()


# ============================================================
# MODELO
# ============================================================

print("Carregando Qwen 2.5 3B em 4-bit...")

model = AutoModelForCausalLM.from_pretrained(

    MODEL_NAME,

    quantization_config=quantization_config,

    device_map="auto",

    dtype=torch.float16,
)


print("Modelo carregado.")

print()


# ============================================================
# CONFIGURAÇÃO LoRA
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


# ============================================================
# APLICA LoRA
# ============================================================

print("Aplicando LoRA...")

model = get_peft_model(

    model,

    lora_config,
)


print()

print("Parâmetros treináveis:")

model.print_trainable_parameters()

print()


# ============================================================
# CONFIGURAÇÕES DO TREINAMENTO
# ============================================================

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    num_train_epochs=1,

    per_device_train_batch_size=1,

    per_device_eval_batch_size=1,

    gradient_accumulation_steps=8,

    learning_rate=1e-4,

    fp16=True,

    gradient_checkpointing=True,

    logging_steps=1,

    save_strategy="epoch",

    eval_strategy="epoch",

    save_total_limit=2,

    report_to="none",

    remove_unused_columns=False,

    dataloader_pin_memory=True,
)


# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=validation_dataset,
)


# ============================================================
# TREINAMENTO
# ============================================================

print("=" * 70)
print("INICIANDO TREINAMENTO")
print("=" * 70)

print()

print("GPU utilizada:")

print(
    torch.cuda.get_device_name(0)
)

print()

trainer.train()


# ============================================================
# SALVAMENTO
# ============================================================

print()

print("=" * 70)
print("SALVANDO MODELO")
print("=" * 70)

print()


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


trainer.save_model(
    OUTPUT_DIR
)


tokenizer.save_pretrained(
    OUTPUT_DIR
)


print()

print("Treinamento concluído!")

print(
    "Modelo salvo em:",
    OUTPUT_DIR
)

print()

print("=" * 70)
print("FIM DO TREINAMENTO")
print("=" * 70)