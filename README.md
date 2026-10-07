# CP2-IA
Comparação A × B
Métrica	Experimento A	Experimento B
Learning rate	1e-4	5e-5
Épocas	1	1
Treinamento	150 exemplos	150 exemplos
Validação	3 exemplos	3 exemplos
Train loss	2.753	3.071
Eval loss	2.740	3.286
Tempo	92,42 s	87,49 s
GPU	RTX 3050 6 GB	RTX 3050 6 GB
CUDA	13.2	13.2
Parâmetros treináveis	3.686.400	3.686.400
% parâmetros treináveis	0,1193%	0,1193%

----------
Validação Experimento A
#	Tema	Nota	Avaliação
1	Lista vs tupla	5/5	Explicou corretamente mutabilidade e mostrou exemplos de ambas. A resposta foi cortada no final, mas a parte essencial está correta.
2	Exceções / try-except	5/5	Explicação correta e exemplo adequado com ZeroDivisionError.
3	Lambda	4/5	Conceito e exemplo estão corretos, mas há pequenas imprecisões na explicação sobre retorno/estrutura de uma lambda.
4	Dicionários	4/5	Conceito e criação estão corretos; resposta foi interrompida durante o exemplo de adição/acesso.
5	for vs while	5/5	Diferenciação correta e exemplos funcionais dos dois loops.
6	Classes	5/5	Explica classe, objeto, atributo e método corretamente, com exemplo funcional.
7	Arquivos	5/5	Mostra corretamente with open(...) e explica o fechamento automático do arquivo.
8	List comprehension	5/5	Explicação, sintaxe e exemplo dos quadrados estão corretos.

Validação Experimento B

#	Tema	Nota	Avaliação
1	Lista vs tupla	5/5	Diferencia corretamente mutabilidade e mostra exemplos. A resposta foi cortada, mas o conteúdo apresentado está correto.
2	Exceções / try-except	5/5	Explica corretamente try, except e ZeroDivisionError, com exemplo funcional.
3	Lambda	4/5	Conceito e exemplo estão corretos, mas há uma imprecisão ao dizer que o "nome da função é implicitamente o argumento".
4	Dicionários	5/5	Explica criação, adição de chave/valor e acesso. O conteúdo apresentado atende à pergunta.
5	for vs while	5/5	Explicação correta, exemplos funcionais e comparação adequada.
6	Classes	5/5	Explica classe, objeto, atributos e métodos corretamente, com código funcional.
7	Arquivos	4/5	O exemplo com with open() está correto, mas a explicação sobre o comportamento do with contém uma imprecisão.
8	List comprehension	5/5	Conceito e código estão corretos; gera os quadrados solicitados.

------------------------

Conclusão dos experimentos
Foram realizados dois experimentos utilizando QLoRA sobre o modelo Qwen2.5-3B-Instruct, mantendo a mesma arquitetura, conjunto de dados, quantidade de épocas e configuração de hardware. A única variável modificada foi a taxa de aprendizado.
O Experimento A utilizou learning_rate = 1e-4 e obteve train_loss = 2,753 e eval_loss = 2,740. O Experimento B utilizou learning_rate = 5e-5 e obteve train_loss = 3,071 e eval_loss = 3,286.
Na avaliação qualitativa, ambos os experimentos obtiveram média de 4,75/5, considerando oito perguntas sobre conceitos de Python. Dessa forma, o Experimento A foi selecionado como modelo final por apresentar menor perda de validação e desempenho qualitativo equivalente ao Experimento B.
Os treinamentos foram executados utilizando uma NVIDIA GeForce RTX 3050 com 6 GB de VRAM, CUDA 13.2 e PyTorch 2.14.0+cu132.


-------------------------
********************************************************

********************************************************
********************************************************

********************************************************

# CP2 - Treinamento de LLM Local com QLoRA

Projeto desenvolvido para a atividade de treinamento e avaliação de uma LLM local utilizando GPU.

O projeto utiliza o modelo **Qwen2.5-3B-Instruct**, adaptado para atuar como um **tutor de programação Python** através de QLoRA (Quantized Low-Rank Adaptation).

---

## 1. Objetivo

O objetivo deste projeto é realizar o treinamento de uma LLM local para responder perguntas relacionadas à programação Python.

Durante o desenvolvimento foram realizados:

- seleção de uma LLM;
- definição do problema;
- criação de um dataset próprio;
- treinamento utilizando GPU;
- aplicação de quantização 4-bit;
- treinamento utilizando QLoRA;
- experimentação com diferentes learning rates;
- avaliação do treinamento;
- avaliação qualitativa das respostas;
- desenvolvimento de uma interface gráfica;
- verificação do uso da GPU;
- documentação do projeto.

---

## 2. Problema escolhido

O problema escolhido foi o desenvolvimento de um **tutor de programação Python**.

O modelo foi treinado para explicar conceitos da linguagem Python de maneira clara e didática, podendo apresentar exemplos de código quando necessário.

Entre os assuntos presentes no dataset estão:

- tipos de dados;
- listas;
- tuplas;
- conjuntos;
- dicionários;
- operadores;
- estruturas condicionais;
- loops;
- funções;
- lambda;
- exceções;
- list comprehension;
- iteradores;
- arquivos;
- programação orientada a objetos;
- módulos;
- ambientes virtuais;
- testes;
- debugging;
- strings;
- datas;
- APIs;
- CUDA e GPU;
- quantização;
- LoRA;
- QLoRA.

---

## 3. Modelo utilizado

O modelo base escolhido foi:

**Qwen/Qwen2.5-3B-Instruct**

O modelo possui aproximadamente 3 bilhões de parâmetros e foi escolhido por apresentar um bom equilíbrio entre capacidade de geração e possibilidade de execução local na GPU disponível.

O treinamento foi realizado utilizando **LoRA**, permitindo ajustar apenas uma pequena parcela dos parâmetros do modelo.

---

## 4. Hardware e ambiente

### GPU

- **GPU:** NVIDIA GeForce RTX 3050
- **VRAM:** 6 GB
- **CUDA:** 13.2
- **PyTorch:** 2.14.0+cu132

Os testes e treinamentos foram realizados utilizando a GPU.

### Ambiente

- Windows 11
- Python 3.11.3
- Visual Studio Code
- Ambiente virtual Python (`venv`)

---

## 5. Quantização

Para possibilitar a execução do modelo na RTX 3050 com 6 GB de VRAM, foi utilizada quantização de 4 bits.

Configuração utilizada:

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
```

Foi utilizado o formato NF4 (NormalFloat4), juntamente com double quantization.

A quantização reduz o consumo de memória da GPU, tornando possível executar e treinar o modelo utilizando uma quantidade limitada de VRAM.

---

## 6. QLoRA

O treinamento foi realizado utilizando QLoRA, que combina:

- quantização do modelo;
- LoRA;
- treinamento de uma pequena quantidade de parâmetros.

Configuração utilizada:

```python
LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],
    bias="none",
    task_type="CAUSAL_LM",
)
```

Foram treinados 3.686.400 parâmetros de um total de 3.089.625.088 parâmetros. Isso corresponde a aproximadamente 0,1193% dos parâmetros totais do modelo.

---

## 7. Dataset

O dataset foi criado especificamente para o projeto.

Estrutura:

data/
├── train.jsonl
└──validation.jsonl

O conjunto de treinamento possui 150 exemlos. Já o de validação possui 3 exemplos. Os dados foram armazenados no formato JSONL. Cada exemplo contém uma interação entre usuário e assistente relacionada à programação Python.

---

## 8. Estrutura do projeto

CP2-IA/
│
├── data/
│   ├── train.jsonl
│   └── validation.jsonl
│
├── models/
│   ├── experimento_A/
│   └── experimento_B/
│
├── results/
│   ├── experimento_A/
│   └── experimento_B/
│
├── src/
│   ├── train.py
│   ├── inference.py
│   ├── evaluate.py
│   └── test_trained_model.py
│
├── frontend/
│   └── app.py
│
├── screenshots/
│   ├── 01-Cuda_Evidencia.png
│   └── 02-PlavaVideo_Evidencia.png
│
├── requirements.txt
├── .gitignore
└── README.md

---

## 9. Comparação dos Experimentos A e B

Foram realizados dois experimentos alterando apenas o learning rate. As demais configurações foram mantidas iguais para permitir uma comparação justa.

### Comparação A × B

Métrica	| Experimento A | Experimento B
Learning rate | 1e-4 | 5e-5
Épocas | 1 | 1
Treinamento | 150 exemplos | 150 exemplos
Validação | 3 exemplos | 3 exemplos
Train loss | 2.753 | 3.071
Eval loss | 2.740 | 3.286
Tempo | 92,42s | 87,49s
GPU | RTX 3050 6 GB | RTX 3050 6 GB
CUDA | 13.2 | 13.2
Parâmetros treináveis | 3.686.400 | 3.686.400
% parâmetros treináveis | 0,1193% | 0,1193%

## 11. Avaliação qualitativa

Após o treinamento, os dois modelos foram avaliados utilizando as mesmas 8 perguntas sobre programação Python.

As perguntas abordaram:

1. Lista vs tupla;
2. Exceções;
3. Lambda;
4. Dicionários;
5. for vs while;
6. Classes;
7. Leitura de arquivos;
8. List comprehension.

Foi utilizada uma escala de 0 a 5:

Nota | Critério
5 | Resposta correta, completa e clara
4 | Resposta correta com pequena imprecisão
3 | Resposta parcialmente correta
2 | Contém erros importantes
1 | Maior parte incorreta
0 | Sem resposta ou completamente incorreta

### Resultado

Experimento | Média
A | 4,75/5
B | 4,75/5

Os dois experimentos apresentaram desempenho qualitativo equivalente. Por isso, a escolha do modelo final foi realizada considerando principalmente o eval_loss.

### 12. Frontend

Foi desenvolvido um frontend utilizando Gradio. A interface permite ao usuário:

- escrever uma pergunta;
- controlar a temperatura;
- definir a quantidade máxima de tokens;
- gerar uma resposta;
- visualizar informações da GPU.

O frontend utiliza o modelo do Experimento A. O caminho utilizado é "models/experimento_A"

A interface também apresenta GPU, CUDA, VRAM utilizada, VRAM reservada.

Isso permite verificar durante a execução que o modelo está sendo executado utilizando a GPU.

## 13. Execução do projeto

### 13.1 Criar ambiente virtual

No diretório do projeto, execute o comando

```powershell
python -m venv venv
```
E depois execute

```powershell
.\venv\Scripts\activate
```

### 13.2 Instalar PyTorch

Execute no terminal o comando

```powershell
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu132
```

### 13.3 Instalar dependências

Execute o comando 

```powershell
pip install -r requirements.txt
```

## 14. Verificar GPU

Antes de executar o projeto, verificar se a GPU está disponível:

```powershell
nvidia-smi
```

## 15. Treinar os experimentos

### Experimento A

```powershell
python src\train.py --experiment A --learning-rate 0.0001
```

### Experimento B

```powershell
python src\train.py --experiment B --learning-rate 0.00005
```

## 16. Avaliar os modelos

### Experimento A

```powershell
python src\evaluate.py --experiment A
```

### Experimento B

```powershell
python src\evaluate.py --experiment B
```

## 17. Executar o frontend

Na raiz do projeto, execute o comando abaixo:

```powershell
python frontend\app.py
```

## 18. Evidências

Foram realizadas capturas de tela para demonstrar os requisitos da atividade.

As evidências incluem a identificação da GPU instalada e a identificação/configuração do CUDA

Os screenshots utilizados na entrega estão na pasta "screenshots".

## 19. Tecnologias utilizadas

- Python
- PyTorch
- CUDA
- Transformers
- PEFT
- BitsAndBytes
- QLoRA
- Gradio
- Pandas
- Scikit-learn
- Matplotlib
- Visual Studio Code

## 20. Conclusão

O projeto demonstrou a possibilidade de realizar o treinamento e execução local de uma LLM utilizando uma GPU NVIDIA RTX 3050 com 6 GB de VRAM.

A utilização de quantização 4-bit e QLoRA permitiu reduzir significativamente a quantidade de memória necessária para trabalhar com o modelo Qwen2.5-3B-Instruct.

Foram realizados dois experimentos com diferentes learning rates. O Experimento A, utilizando 1e-4, apresentou o melhor resultado de validação, com eval_loss = 2.740, enquanto o Experimento B, utilizando 5e-5, apresentou eval_loss = 3.286.

Na avaliação qualitativa, ambos os modelos obtiveram média de 4,75/5 nas oito perguntas utilizadas.

Com base nesses resultados, o Experimento A foi selecionado como modelo final e integrado ao frontend desenvolvido em Gradio.
O projeto atende aos requisitos de seleção de modelo, definição do problema, treinamento em GPU, experimentação de parâmetros, avaliação dos resultados, desenvolvimento de frontend e documentação.

