# Fine-tuning pipeline: Computer Vision + LLM

> Transfer learning on Oxford Pets · LoRA fine-tuning on Mistral 7B · RAGAS evaluation · W&B tracking

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7-orange)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Models-yellow)](https://huggingface.co/flaviodell)
[![W&B](https://img.shields.io/badge/Weights%20%26%20Biases-Tracking-yellow)](https://wandb.ai/flaviodell/project-finetuning)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Overview

This project implements a complete end-to-end machine learning pipeline combining **computer vision** and **large language models** on a shared domain: the 37 cat and dog breeds of the Oxford-IIIT Pet dataset.

The pipeline produces two complementary models:

| Model | Architecture | Task | HuggingFace |
|-------|-------------|------|-------------|
| CV classifier | ResNet50 (transfer learning) | 37-class breed classification | [flaviodell/oxford-pets-resnet50](https://huggingface.co/flaviodell/oxford-pets-resnet50) |
| LLM assistant | Mistral 7B + LoRA | Veterinary expert Q&A | [flaviodell/pet-expert-mistral7b-lora](https://huggingface.co/flaviodell/pet-expert-mistral7b-lora) |

These two models are designed to feed into a larger agentic system where an autonomous agent identifies a breed from an image and provides expert veterinary advice.

---

## Results

### CV model — ResNet50

| Metric | Value |
|--------|-------|
| Test accuracy | **90.1%** |
| Test F1 macro | **89.9%** |
| Best val accuracy | 91.9% |
| Training images | ~2,900 |
| Classes | 37 breeds |

### LLM — Mistral 7B + LoRA

| Metric | Value |
|--------|-------|
| Faithfulness (RAGAS) | 0.491 |
| Answer relevancy (RAGAS) | **0.936** |
| Training examples | 185 (synthetic) |
| Quantization | FP16 → 4-bit NF4 (72% size reduction) |

Experiment tracking: [W&B dashboard](https://wandb.ai/flaviodellave/project-finetuning)

---

## Architecture

```
Oxford Pets dataset (37 breeds)
         │
         ├─── CV pipeline ──────────────────────────────────────────────┐
         │    ResNet50 pretrained on ImageNet                           │
         │    Frozen backbone + fine-tuned classifier head              │
         │    torchvision · PyTorch training loop · torchmetrics        │
         │    → best_model.pth (90.1% test accuracy)                    │
         │                                                              │
         └─── LLM pipeline ─────────────────────────────────────────────┘
              Groq API (LLaMA 3 8B) → 185 synthetic Q&A pairs
              Mistral 7B Instruct v0.3 + LoRA (r=16) + QLoRA 4-bit
              HuggingFace PEFT · bitsandbytes · W&B tracking
              → best_lora_adapter/ (relevancy 0.936)
```

---

## Project structure

```
├── configs/
│   └── config.yaml                  # All hyperparameters in one place
├── data/
│   ├── raw/                         # Oxford Pets (auto-downloaded, not tracked)
│   └── processed/                   # Synthetic LLM dataset (train.jsonl, val.jsonl)
├── models/
│   ├── cv/best_model.pth            # CV checkpoint (not tracked, on HuggingFace)
│   └── llm/best_lora_adapter/       # LoRA adapter (not tracked, on HuggingFace)
├── notebooks/
│   ├── project_report.ipynb         # ← main portfolio notebook
│   ├── train_cv_colab.ipynb
│   ├── train_llm_colab.ipynb
│   └── eval_llm_colab.ipynb
├── outputs/
│   ├── cv/                          # Metrics, confusion matrix, per-class accuracy
│   ├── llm/                         # RAGAS results, quantization analysis
│   └── benchmark/                   # Combined summary charts
├── src/
│   ├── data/                        # Dataset download, preprocessing, DataLoaders
│   ├── training/                    # CV training loop, LLM LoRA fine-tuning
│   ├── evaluation/                  # eval_cv.py, eval_llm.py, benchmark.py
│   └── utils/                       # Common utilities, inference, W&B summary
└── README.md
```

---

## Quickstart

### Requirements

```bash
git clone https://github.com/flaviodell/llm-cv-finetuning-pipeline.git
cd llm-cv-finetuning-pipeline
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Mac/Linux
pip install -e .
pip install -r requirements.txt
```

### Environment variables

Copy `.env.example` to `.env` and fill in your API keys:

```
WANDB_API_KEY=...
WANDB_PROJECT=project-finetuning
HF_TOKEN=...
GROQ_API_KEY=...
OPENAI_API_KEY=...
```

### Run the report notebook

```bash
set PYTHONPATH=%CD%
jupyter notebook notebooks/project_report.ipynb
```

### Download models from HuggingFace Hub

**CV model:**

```python
from huggingface_hub import hf_hub_download

checkpoint_path = hf_hub_download(
    repo_id="flaviodell/oxford-pets-resnet50",
    filename="best_model.pth"
)
```

**LLM adapter:**

```python
from huggingface_hub import snapshot_download

snapshot_download(repo_id="flaviodell/pet-expert-mistral7b-lora")
```

---

## Key techniques

**Transfer learning** — ResNet50 backbone frozen, only the classifier head (~75K parameters out of 25M total) is fine-tuned. Stable convergence on small datasets without overfitting.

**LoRA (Low-Rank Adaptation)** — instead of updating all 7B parameters, trainable low-rank matrices are injected into the attention layers (q_proj, v_proj, r=16). Result: 0.1% trainable parameters, full fine-tuning quality.

**QLoRA** — LoRA combined with 4-bit NF4 quantization via bitsandbytes. Reduces the base model from 13.5 GB to 3.8 GB (72% reduction), making 7B fine-tuning feasible on a single T4/P100 GPU.

**Synthetic dataset generation** — 185 instruction/response pairs generated via Groq API (LLaMA 3 8B as teacher model). Covers all 37 Oxford Pets breeds across 5 question templates per breed.

**RAGAS evaluation** — automatic evaluation of LLM output quality using GPT-4o-mini as judge model. Metrics: faithfulness and answer relevancy.

---

## Training infrastructure

| Phase | Where | Hardware | Time |
|-------|-------|----------|------|
| CV training (10 epochs) | Google Colab | T4 GPU (15.6 GB) | ~20 min |
| LLM fine-tuning (3 epochs) | Kaggle | T4 x2 GPU (2×15.6 GB) | ~30 min |
| Evaluation                 | Kaggle | T4 x2 GPU (2×15.6 GB) | ~15 min |
| Everything else | Local | CPU | — |

---

## License

MIT — see [LICENSE](LICENSE) for details.
