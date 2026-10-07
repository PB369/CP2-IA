import torch
import gradio as gr

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

# Modelo escolhido após a comparação dos experimentos.
# Experimento A: learning_rate = 1e-4
LORA_PATH = "models/experimento_A"


# ============================================================
# GPU
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA não está disponível. "
        "O frontend precisa ser executado com GPU."
    )

GPU_NAME = torch.cuda.get_device_name(0)

print("=" * 60)
print("TUTOR DE PYTHON - LLM LOCAL")
print("=" * 60)
print(f"GPU: {GPU_NAME}")
print(
    f"VRAM total: "
    f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
)
print(f"CUDA: {torch.version.cuda}")
print("=" * 60)


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
# TOKENIZER
# ============================================================

print("Carregando tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Tokenizer carregado.")


# ============================================================
# MODELO BASE
# ============================================================

print("Carregando modelo base em 4-bit...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    device_map="auto",
    dtype=torch.float16,
)

print("Modelo base carregado.")


# ============================================================
# LoRA
# ============================================================

print("Carregando adaptador LoRA do Experimento A...")

model = PeftModel.from_pretrained(
    model,
    LORA_PATH
)

model.eval()

print("LoRA do Experimento A carregado.")
print("Modelo pronto!")
print("=" * 60)


# ============================================================
# FUNÇÃO DE GERAÇÃO
# ============================================================

def generate_response(prompt, temperature, max_new_tokens):

    if not prompt or not prompt.strip():
        return "Digite uma pergunta sobre Python."

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
            "content": prompt,
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return response.strip()


# ============================================================
# INFORMAÇÕES DA GPU
# ============================================================

def gpu_info():

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3

    return (
        f"**GPU:** {GPU_NAME}\n\n"
        f"**CUDA:** {torch.version.cuda}\n\n"
        f"**VRAM utilizada:** {allocated:.2f} GB\n\n"
        f"**VRAM reservada:** {reserved:.2f} GB"
    )


# ============================================================
# INTERFACE
# ============================================================

with gr.Blocks(
    title="Tutor de Python - LLM Local"
) as app:

    gr.Markdown(
        """
        # 🐍 Tutor de Python - LLM Local

        **Qwen2.5-3B-Instruct + QLoRA**

        **Modelo utilizado: Experimento A — Learning Rate = 1e-4**

        Faça uma pergunta sobre programação Python e receba uma
        explicação do modelo treinado localmente.
        """
    )

    with gr.Row():

        with gr.Column():

            prompt = gr.Textbox(
                label="Pergunta",
                placeholder=(
                    "Ex.: Explique como funciona um loop for em Python."
                ),
                lines=5,
            )

            with gr.Row():

                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=1.5,
                    value=0.7,
                    step=0.1,
                    label="Temperatura",
                )

                max_tokens = gr.Slider(
                    minimum=50,
                    maximum=500,
                    value=300,
                    step=50,
                    label="Máximo de tokens",
                )

            generate_button = gr.Button(
                "Gerar resposta",
                variant="primary",
            )

        with gr.Column():

            response = gr.Textbox(
                label="Resposta do modelo",
                lines=15,
            )

    generate_button.click(
        fn=generate_response,
        inputs=[
            prompt,
            temperature,
            max_tokens,
        ],
        outputs=response,
    )

    gr.Markdown("---")

    gr.Markdown("### Informações da GPU")

    gpu_output = gr.Markdown(
        gpu_info()
    )

    refresh_gpu = gr.Button(
        "Atualizar informações da GPU"
    )

    refresh_gpu.click(
        fn=gpu_info,
        outputs=gpu_output,
    )


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    app.launch(
        inbrowser=True,
    )