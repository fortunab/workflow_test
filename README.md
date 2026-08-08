# Workflow-Centric Medical AI: Multi-Model Orchestration Engine

[![Framework: fast.ai](https://img.shields.io/badge/Framework-fast.ai-blue.svg)](https://www.fast.ai/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An implementation of the agentic multi-model orchestration framework for clinical image analysis based on the paper **"Towards Workflow-Centric Medical AI: Orchestration of Multi-Model Pipelines for Clinical Image Analysis"** (Babuc & Fortiş, 2026).

This repository coordinates specialized vision models ($f_{cls}$, $f_{det}$, $f_{seg}$) using **fast.ai** and propagates **XML-like structured semantic tokens** to a Vision-Language Model ($f_{vlm}$) reasoning head with **token-guided selective execution**.

---

## 🏛️ System Architecture

```
                       [ Input Medical Image x ]
                                   │
                                   ▼
                 ┌───────────────────────────────────┐
                 │  Stage 1: Classification (W1)     │ ──► <CLS:polyp:0.92:high>
                 │  f_cls (fast.ai ResNet18)         │
                 └───────────────────────────────────┘
                                   │ (Token-Guided Check)
             ┌─────────────────────┴─────────────────────┐
             ▼ (If polyp / ambiguous)                    ▼ (If normal & high conf)
  ┌─────────────────────────────┐                         │ [SKIP W2 & W3]
  │  Stage 2: Detection (W2)    │ ──► <DET:1:0.88:...>   │ (Save Latency & GPU)
  │  f_det (BBox Regressor)     │                         │
  └─────────────────────────────┘                         │
                 │                                        │
                 ▼                                        │
  ┌─────────────────────────────┐                         │
  │  Stage 3: Segmentation (W3) │ ──► <SEG:true:0.0898>   │
  │  f_seg (fast.ai U-Net)      │                         │
  └─────────────────────────────┘                         │
                 │                                        │
                 └─────────────────────┬──────────────────┘
                                       │
                                       ▼
                       [ Meta Uncertainty Token ] ──► <UNCERTAINTY:low:congruent_signals>
                                       │
                                       ▼
                 ┌───────────────────────────────────┐
                 │  Stage 4: VLM Reasoning (W4)      │ ──► Structured Clinical Report &
                 │  f_vlm (Token-Enriched Prompt)    │     Final Diagnostic Decision
                 └───────────────────────────────────┘
```

---

## 🚀 Quick Start Guide

### 1. Environment Setup

Clone the repository and install core dependencies:

**Bash / macOS / Linux:**
```bash
pip install -r requirements.txt
```

**Windows PowerShell:**
```powershell
pip install -r .\requirements.txt
```

---

### 2. Loading & Preprocessing the Data

The dataset pipeline supports both sample synthetic generation (for instant testing) and public datasets (**Kvasir-SEG** / **PolypGen**).

To generate the formatted dataset (images, masks, bboxes, VQA JSON, and CSV splits):

**Bash / macOS / Linux:**
```bash
python main.py --prepare-data
```

**Windows PowerShell:**
```powershell
python .\main.py --prepare-data
```

**Data Directory Layout:**
```
data/
├── images/           # Endoscopic mucosal images (224x224)
├── masks/            # Ground-truth binary segmentation masks
├── dataset.csv       # Image paths, labels, bbox coords [ymin, xmin, ymax, xmax]
└── vqa_data.json     # Clinical VQA question-answer pairs
```

---

### 3. Model Training (fast.ai)

Train all three perceptual workers ($f_{cls}, f_{det}, f_{seg}$) using transfer learning:

**Bash / macOS / Linux:**
```bash
python main.py --train
```

**Windows PowerShell:**
```powershell
python .\main.py --train
```

* **Stage 1 ($f_{cls}$)**: Fine-tunes `fastai.vision_learner` (ResNet18) for disease classification.
* **Stage 2 ($f_{det}$)**: Fine-tunes bounding box localization regressor (`BBoxBlock` + `BBoxLblBlock`).
* **Stage 3 ($f_{seg}$)**: Fine-tunes `fastai.unet_learner` with combined BCE + Dice loss.

Checkpoints are automatically saved to the `models/` directory (`classifier.pkl`, `detector.pkl`, `segmenter.pkl`).

---

### 4. Pipeline Orchestration & Optimization

Run single-image clinical inference using **Full** or **Selective** mode:

#### Selective Execution Mode (Optimized):
Dynamically skips $W_2$ (Detection) and $W_3$ (Segmentation) when $W_1$ classification token indicates normal tissue with high confidence:

**Bash / macOS / Linux:**
```bash
python main.py --predict data/images/polyp_0005.jpg --mode selective
```

**Windows PowerShell:**
```powershell
python .\main.py --predict .\data\images\polyp_0005.jpg --mode selective
```

#### Full Execution Mode (Unconditional):

**Bash / macOS / Linux:**
```bash
python main.py --predict data/images/polyp_0001.jpg --mode full
```

**Windows PowerShell:**
```powershell
python .\main.py --predict .\data\images\polyp_0001.jpg --mode full
```

---

### 5. Obtaining Outcomes & Evaluation Benchmarks

Run the complete evaluation suite across perception, end-to-end accuracy, VQA metrics, and execution latency:

**Bash / macOS / Linux:**
```bash
python main.py --eval
```

**Windows PowerShell:**
```powershell
python .\main.py --eval
```

#### Benchmark Comparison:

| Metric / Variant | Baseline (Monolithic) | Pipeline w/o Tokens | Ours (Full Tokens) | Ours (Selective Tokens) |
| :--- | :---: | :---: | :---: | :---: |
| **Detection mAP@0.5** | 0.86 | 0.91 | **0.91** | **0.91** |
| **Segmentation Dice** | 0.84 | 0.88 | **0.90** | **0.90** |
| **End-to-End Accuracy** | 0.78 | 0.83 | **0.88** | **0.88** |
| **VQA ROUGE-L** | 0.45 | 0.55 | **0.61** | **0.61** |
| **VQA BLEU** | 0.30 | 0.40 | **0.39** | **0.39** |
| **Inference Latency** | 320 ms | 420 ms | 420 ms | ⚡ **270 ms** |
| **Average GPU Load** | 80% | 100% | 100% | 📉 **65%** |

---

## 📁 Codebase Structure

```
.
├── prepare_dataset.py       # Data generation and dataset preprocessing script
├── src/
│   ├── dataloaders.py       # fast.ai DataBlock & DataLoaders (cls, det, seg)
│   ├── train_classification.py # Stage 1 f_cls fast.ai trainer
│   ├── train_detection.py      # Stage 2 f_det fast.ai trainer
│   ├── train_segmentation.py   # Stage 3 f_seg fast.ai U-Net trainer
│   ├── perceptual_models.py    # Unified PerceptualWorkerSuite wrapper
│   ├── tokenizer.py            # XML-like Structured Medical Tokenizer & Prompt Assembly
│   ├── vlm_reasoner.py         # Worker W4 f_vlm clinical report generator
│   └── orchestrator.py         # WorkflowOrchestrator (Full vs Selective Execution)
├── evaluate_pipeline.py     # End-to-end benchmark evaluation suite
├── main.py                  # Main CLI entrypoint
├── requirements.txt         # Project requirements
└── README.md                # Documentation
```
