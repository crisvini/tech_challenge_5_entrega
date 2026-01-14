# Tech Challenge – Fase 5  
**Detecção de Objetos Cortantes com Visão Computacional**

Este projeto foi desenvolvido como parte do **Tech Challenge – Fase 5** da pós-graduação **IA para Devs (FIAP)**.  
O objetivo é validar a viabilidade de um MVP capaz de detectar **objetos cortantes (facas e tesouras)** em vídeos, utilizando **Visão Computacional e Deep Learning**, com foco na redução de falsos positivos.

---

## 👥 Alunos Participantes

- **Cristian Vinícius Leoncini Lopes** – RM 362011  
- **Júlia de Andrade Bertazzi** – RM 361574  
- **Luiz Henrique Beluci Terra** – RM 363804  
- **Paulo Cesar do Nascimento Silva** – RM 361778  

---

## 📊 Dataset Utilizado

O dataset final utilizado no treinamento foi resultado da **mesclagem e refinamento** dos seguintes datasets públicos do Roboflow:

- Sharp Objects Detection IV  
  https://universe.roboflow.com/hazard-qjwxm/sharp-objects-detection-iv

- Sharp Objects Detection II  
  https://universe.roboflow.com/hazard-qjwxm/sharp-objects-detection-ii

### 🔎 Estratégia de Dados Negativos

- As **imagens negativas** foram extraídas do próprio dataset (imagens que **não continham facas nem tesouras**).
- Foi realizado um **refinamento manual** para ajustar a **proporção entre imagens positivas e negativas**, visando reduzir falsos positivos.
- Imagens negativas que **não possuíam labels** receberam anotações por meio de **scripts auxiliares**, garantindo compatibilidade com o treinamento supervisionado.

### 🔁 Processamento e Padronização

Foram utilizados scripts Python para:

- **Remapear classes** (unificando e padronizando as classes de *faca* e *tesoura*);
- **Criar labels vazias** para imagens negativas sem anotações;
- Garantir consistência entre os datasets mesclados.

---

## 🧰 Tecnologias Utilizadas

- Python 3.11
- PyTorch (CUDA 12.1)
- Ultralytics YOLO
- OpenCV
- Scripts auxiliares em Python para tratamento de dados

---

## ⚙️ Requisitos

- [Python 3.11.9](https://www.python.org/downloads/release/python-3119/)
- GPU compatível com CUDA (recomendado)
- Criar uma pasta chamada **`inputs/`** na raiz do projeto contendo os vídeos de teste:
  - `video.mp4`  
    https://drive.google.com/file/d/1AV6y7OFPgq9UiU0TMUjoaoYQHsvKO__u/view
  - `video2.mp4`  
    https://drive.google.com/file/d/1XBhBKY9QHo0xj8gXMYcq92e-vrECrNH3/view

---

## 🚀 Instalação

```bash
# criar ambiente virtual
python -m venv .venv

# ativar ambiente (Windows)
.\.venv\Scripts\Activate.ps1

# atualizar pip
python -m pip install -U pip

# instalar PyTorch com suporte a CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# instalar YOLO (Ultralytics)
pip install ultralytics
```

---

## 🚀 Execução
