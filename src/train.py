import torch

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments
)

from peft import LoraConfig

from trl import SFTTrainer


MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

TRAIN_FILE = "data/train.jsonl"
VALIDATION_FILE = "data/validation.jsonl"

OUTPUT_DIR = "models/python-tutor-lora"


print("=" * 60)
print("TREINAMENTO DA LLM")
print("=" * 60)

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA não está disponível. "
        "O treinamento deve ser realizado utilizando GPU."
    )

print("GPU:", torch.cuda.get_device_name(0))


# --------------------------------------------------
# DATASET
# --------------------------------------------------

dataset = load_dataset(
    "json",
    data_files={
        "train": TRAIN_FILE,
        "validation": VALIDATION_FILE
    }
)

train_dataset = dataset["train"]
eval_dataset = dataset["validation"]

print()
print("Exemplos de treinamento:", len(train_dataset))
print("Exemplos de validação:", len(eval_dataset))


# --------------------------------------------------
# TOKENIZER
# --------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# --------------------------------------------------
# MODELO
# --------------------------------------------------

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)


# --------------------------------------------------
# LORA
# --------------------------------------------------

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],

    task_type="CAUSAL_LM"
)


# --------------------------------------------------
# CONFIGURAÇÃO DO TREINAMENTO
# --------------------------------------------------

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    num_train_epochs=1,

    per_device_train_batch_size=1,

    per_device_eval_batch_size=1,

    gradient_accumulation_steps=8,

    learning_rate=1e-4,

    fp16=True,

    logging_steps=1,

    save_strategy="epoch",

    eval_strategy="epoch",

    report_to="none",

    remove_unused_columns=False
)


# --------------------------------------------------
# TRAINER
# --------------------------------------------------

trainer = SFTTrainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=eval_dataset,

    processing_class=tokenizer,

    peft_config=lora_config,

)


# --------------------------------------------------
# TREINAR
# --------------------------------------------------

print()
print("Iniciando treinamento...")

trainer.train()


# --------------------------------------------------
# SALVAR
# --------------------------------------------------

print()
print("Salvando modelo...")

trainer.save_model(OUTPUT_DIR)

tokenizer.save_pretrained(OUTPUT_DIR)

print()
print("Treinamento concluído!")
print("Modelo salvo em:", OUTPUT_DIR)