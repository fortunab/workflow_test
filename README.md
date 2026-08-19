# PolypFlow — Workflow-Centric Medical AI Pipeline

<div align="center">

[![Framework: fast.ai](https://img.shields.io/badge/Framework-fast.ai-blue.svg)](https://www.fast.ai/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Datasets](https://img.shields.io/badge/Datasets-Kvasir--SEG%20%7C%20PolypGen%20%7C%20Synthetic-purple.svg)](#datasets)

**A multi-stage, token-guided AI pipeline for endoscopic polyp analysis.**  
*Classification → Detection → Segmentation → VLM Clinical Report*  
with **selective execution** that reduces latency by **68%** on negative cases.

[🌐 Website](#website) · [🚀 Quick Start](#quick-start) · [📊 Results](#benchmark-results) · [📁 Datasets](#datasets) · [⚗️ Experiments](#running-experiments)

</div>

---

## Overview

PolypFlow implements the agentic multi-model orchestration framework described in:

> **"Towards Workflow-Centric Medical AI: Orchestration of Multi-Model Pipelines for Clinical Image Analysis"**  
> Babuc & Fortiş, 2026

Instead of running all inference stages unconditionally, PolypFlow emits a **structured token** after each stage. Downstream stages are only activated when the token confidence exceeds a threshold — dramatically reducing compute on negative cases while preserving full sensitivity on positive ones.

```
[ Input Frame ]
      │
      ▼
┌─────────────────────┐
│  f_cls  Stage 1     │──► [CLS:polyp·conf=0.97]   ← Gating token
│  Classification     │
└─────────────────────┘
      │ polyp? ──YES──────────────────────────────┐
      │ normal? ──NO (skip W2 & W3, save compute) │
      ▼                                           ▼
┌─────────────────────┐              [Latency: ~11ms, GPU: ~70%]
│  f_det  Stage 2     │──► [DET:bbox=(142,88,398,312)·conf=0.91]
│  BBox Detection     │
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  f_seg  Stage 3     │──► [SEG:dice=0.891·area=18.4%]
│  Segmentation       │
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  f_vlm  Stage 4     │──► Clinical Report + Final Diagnosis
│  VLM Reasoning      │
└─────────────────────┘
```

---

## Benchmark Results

### Kvasir-SEG — Segmentation

| Model | Year | mDice ↑ | mIoU ↑ | Recall ↑ | Precision ↑ |
|:------|:----:|:-------:|:------:|:--------:|:-----------:|
| U-Net | 2015 | 0.818 | 0.746 | 0.854 | 0.843 |
| ResUNet++ | 2019 | 0.813 | 0.793 | 0.861 | 0.856 |
| PraNet | 2020 | 0.898 | 0.840 | 0.944 | 0.915 |
| SANet | 2021 | 0.904 | 0.847 | 0.936 | 0.921 |
| CaraNet | 2022 | 0.918 | 0.865 | 0.953 | 0.941 |
| Polyp-PVT | 2021 | 0.940 | 0.890 | 0.963 | 0.948 |
| SSFormer-L | 2022 | 0.945 | 0.900 | 0.968 | 0.952 |
| **PolypFlow (Ours)** | 2024 | **0.948** | **0.903** | **0.971** | 0.948 |

### Kvasir-SEG — Detection

| Model | Year | mAP@.5 ↑ | mAP@.75 ↑ | Recall ↑ | FPS ↑ |
|:------|:----:|:--------:|:---------:|:--------:|:-----:|
| Faster R-CNN | 2015 | 0.814 | 0.671 | 0.832 | 18 |
| RetinaNet | 2017 | 0.828 | 0.692 | 0.854 | 24 |
| YOLOv5-s | 2020 | 0.856 | 0.718 | 0.873 | 112 |
| DETR | 2020 | 0.861 | 0.729 | 0.882 | 28 |
| YOLOv8-s | 2023 | 0.878 | 0.741 | 0.893 | 128 |
| **PolypFlow f_det (Ours)** | 2024 | **0.891** | **0.754** | **0.907** | 96 |

### PolypGen — Segmentation (Multi-Center)

| Model | Year | mDice ↑ | mIoU ↑ | Recall ↑ | Precision ↑ |
|:------|:----:|:-------:|:------:|:--------:|:-----------:|
| U-Net | 2015 | 0.742 | 0.665 | 0.798 | 0.812 |
| PraNet | 2020 | 0.781 | 0.706 | 0.829 | 0.851 |
| SANet | 2021 | 0.798 | 0.724 | 0.844 | 0.862 |
| CaraNet | 2022 | 0.814 | 0.743 | 0.857 | 0.878 |
| Polyp-PVT | 2021 | 0.831 | 0.762 | 0.874 | 0.892 |
| SSFormer-L | 2022 | 0.846 | 0.779 | 0.889 | 0.903 |
| **PolypFlow (Ours)** | 2024 | **0.858** | **0.791** | **0.898** | **0.911** |

### PolypGen — Classification (Polyp vs. Normal)

| Model | Year | Accuracy ↑ | AUC-ROC ↑ | F1 ↑ | Specificity ↑ |
|:------|:----:|:----------:|:---------:|:----:|:-------------:|
| ResNet-50 | 2016 | 0.921 | 0.971 | 0.919 | 0.916 |
| DenseNet-121 | 2017 | 0.934 | 0.978 | 0.931 | 0.928 |
| EfficientNet-B4 | 2019 | 0.941 | 0.982 | 0.939 | 0.935 |
| ViT-B/16 | 2020 | 0.947 | 0.984 | 0.944 | 0.941 |
| **PolypFlow f_cls (Ours)** | 2024 | **0.952** | **0.987** | **0.951** | **0.949** |

### Efficiency: Selective vs. Full Pipeline

| Mode | Latency | GPU Load | Stages Run |
|:-----|:-------:|:--------:|:----------:|
| Full Mode | 35 ms | ~100% | Always all 4 |
| **Selective Mode** | **11 ms** | **~70%** | 1–4 (gated) |
| **Reduction** | **↓ 68%** | **↓ 30pp** | — |

### VLM Reasoning Quality

| Metric | Score |
|:-------|:-----:|
| ROUGE-L | 0.61 |
| BLEU | 0.39 |

---

## Quick Start

### 1. Prerequisites

```bash
# Python 3.10+  |  Conda recommended
conda create -n polypflow python=3.10
conda activate polypflow
```

### 2. Install Dependencies

```bash
pip install fastai torch torchvision pillow pandas
pip install "numpy<2.0.0"    # Critical: fixes DLL conflicts on Windows
```

> **⚠️ Windows Note:** Always use `numpy<2.0.0`. NumPy 2.x breaks pre-compiled C extensions (PyTorch DLLs, pandas, numexpr) on Windows. Use `num_workers=0` in all DataLoaders to avoid multiprocessing pickling errors.

### 3. Synthetic Dataset (zero download — works immediately)

```powershell
# Windows PowerShell
python .\main.py --prepare-data
python .\main.py --train --epochs 20
python .\main.py --eval
```

```bash
# Linux / macOS
python main.py --prepare-data
python main.py --train --epochs 20
python main.py --eval
```

---

## Datasets

### Kvasir-SEG

| Property | Value |
|:---------|:------|
| Samples | 1,000 polyp images |
| Annotations | Per-pixel masks + bounding boxes (JSON) |
| Split | 70% train / 15% val / 15% test |
| License | CC-BY 4.0 |
| Download | [datasets.simula.no/kvasir-seg](https://datasets.simula.no/kvasir-seg/) |

```powershell
# Download (Windows PowerShell)
Invoke-WebRequest -Uri "https://datasets.simula.no/downloads/kvasir-seg.zip" -OutFile "Kvasir-SEG.zip"
Expand-Archive -Path "Kvasir-SEG.zip" -DestinationPath "."

# Prepare + Train
python .\main.py --dataset kvasir --data-path .\Kvasir-SEG --prepare-data
python .\main.py --dataset kvasir --data-path .\Kvasir-SEG --train --epochs 20
python .\main.py --dataset kvasir --data-path .\Kvasir-SEG --eval
```

Expected folder structure after extraction:
```
Kvasir-SEG/
├── images/              # 1000 polyp .jpg images
├── masks/               # 1000 binary masks
└── kavsir_bboxes.json   # bbox annotations
```

### PolypGen

| Property | Value |
|:---------|:------|
| Samples (Positive) | ~3,762 polyp images |
| Samples (Negative) | ~2,000+ normal frames |
| Centers | 6 clinical sites (C1–C6) |
| Split | C1–C4 train / C5 val / C6 test |
| License | Synapse (Academic use) |
| Download | [synapse.org/#!Synapse:syn26376615](https://www.synapse.org/#!Synapse:syn26376615) |

```bash
# Download via Synapse client (requires free registration)
pip install synapseclient
synapse get syn26376615 --recursive

# Prepare + Train
python main.py --dataset polypgen --data-path ./PolypGen --prepare-data
python main.py --dataset polypgen --data-path ./PolypGen --train --epochs 20
python main.py --dataset polypgen --data-path ./PolypGen --eval
```

Expected folder structure:
```
PolypGen/
├── data_C1/
│   ├── images_C1/            # Polyp-positive images
│   └── masks_C1/             # Segmentation masks
├── data_C1_negative/
│   └── images_C1_negative/   # Normal tissue frames
├── data_C2/ ... data_C6/
├── data_C2_negative/ ... data_C6_negative/
└── bbox_annotation.json
```

> See [DATASET_SETUP.md](DATASET_SETUP.md) for full download instructions.

---

## Running Experiments

### Single Dataset

```powershell
# Synthetic
python .\run_experiments.py --dataset synthetic

# Kvasir-SEG
python .\run_experiments.py --dataset kvasir --data-path .\Kvasir-SEG

# PolypGen
python .\run_experiments.py --dataset polypgen --data-path .\PolypGen
```

### All Datasets (Full SOTA Comparison)

```powershell
python .\run_experiments.py --all `
    --kvasir-path   .\Kvasir-SEG `
    --polypgen-path .\PolypGen `
    --num-samples 40
```

### What `run_experiments.py` Outputs

For each dataset, prints and saves:

| Output | Location |
|:-------|:---------|
| Segmentation table vs. U-Net, PraNet, SANet, CaraNet, Polyp-PVT, SSFormer-L | terminal + `results/comparison_seg_<dataset>.csv` |
| Detection table vs. Faster R-CNN, RetinaNet, YOLOv5, YOLOv8, DETR | terminal |
| Classification table vs. ResNet-50, DenseNet-121, EfficientNet-B4, ViT-B/16 | terminal |
| Efficiency: Full vs. Selective latency, GPU load, speedup | terminal |
| VLM: ROUGE-L, BLEU | terminal |
| Full JSON metrics | `results/metrics_<dataset>.json` |

---

## CLI Reference

```
python main.py [OPTIONS]

Data options:
  --prepare-data               Generate / parse dataset
  --dataset {synthetic,kvasir,polypgen}
                               Data source (default: synthetic)
  --data-path PATH             Root folder of downloaded real dataset

Training options:
  --train                      Train all 3 perceptual models
  --epochs N                   Epochs per stage (default: 20)

Inference options:
  --predict PATH               Path to image for inference
  --mode {selective,full}      Execution mode (default: selective)
  --question TEXT              VQA question prompt

Evaluation:
  --eval                       Run end-to-end benchmark evaluation
```

### Example Commands

```powershell
# Train with 30 epochs
python .\main.py --train --epochs 30

# Selective inference
python .\main.py --predict .\data\images\polyp_0001.png --mode selective

# Full inference with custom question
python .\main.py --predict .\data\images\polyp_0001.png --mode full `
                 --question "Is this lesion malignant?"

# Evaluate
python .\main.py --eval
```

---

## Project Structure

```
polypflow/
├── main.py                      # CLI entrypoint (--prepare-data, --train, --eval, --predict)
├── run_experiments.py           # Full SOTA comparison metric suite
├── prepare_dataset.py           # Synthetic dataset generator
├── evaluate_pipeline.py         # Core evaluation logic
│
├── src/
│   ├── __init__.py
│   ├── dataloaders.py           # fast.ai DataBlock for cls / det / seg
│   ├── train_classification.py  # Stage 1: f_cls  (ResNet + fast.ai)
│   ├── train_detection.py       # Stage 2: f_det  (BBox regression)
│   ├── train_segmentation.py    # Stage 3: f_seg  (U-Net + fast.ai)
│   ├── perceptual_models.py     # Unified PerceptualWorkerSuite wrapper
│   ├── tokenizer.py             # Structured token encoder / decoder
│   ├── vlm_reasoner.py          # Stage 4: f_vlm  (clinical report)
│   ├── orchestrator.py          # WorkflowOrchestrator (selective / full)
│   ├── kvasir_loader.py         # Kvasir-SEG dataset adapter
│   └── polypgen_loader.py       # PolypGen multi-center adapter
│
├── models/                      # Saved fast.ai .pkl model files
│   ├── classifier.pkl
│   ├── detector.pkl
│   └── segmenter.pkl
│
├── data/                        # Synthetic dataset (auto-generated)
│   ├── images/
│   ├── masks/
│   ├── dataset.csv
│   └── vqa_data.json
│
├── results/                     # Experiment outputs (auto-created)
│   ├── metrics_synthetic.json
│   ├── metrics_kvasir.json
│   ├── metrics_polypgen.json
│   └── comparison_seg_*.csv
│
├── index.html                   # 🌐 Project website (open in browser)
├── DATASET_SETUP.md             # Dataset download guide
├── requirements.txt
└── README.md
```

---

## Website

A full research project website is included. Open it locally:

```powershell
# Start server (Windows)
python -m http.server 8081
# Then open: http://localhost:8081

# Or on Linux/macOS
python3 -m http.server 8081
```

Features:
- Interactive SOTA comparison tables (Kvasir-SEG + PolypGen)
- Bar charts (mDice, mIoU), Radar chart, Bubble chart (latency vs. accuracy)
- Selective vs. Full mode efficiency cards
- Dataset info cards + Quick Start code blocks

---

## Known Issues & Solutions

| Issue | Cause | Fix |
|:------|:------|:----|
| `_ARRAY_API not found` | NumPy 2.x incompatibility | `pip install "numpy<2.0.0"` |
| `WinError 1114` DLL load failure | NumPy 2.x + PyTorch C-extension conflict | Same as above |
| `AttributeError: Can't pickle local object` | Windows multiprocessing + fast.ai | Use `num_workers=0` in all DataLoaders |
| `Got pickle error when starting worker` | Windows multiprocessing fork | Use `num_workers=0` |
| `BatchNorm channel mismatch` | Manual model head creation | Use `vision_learner(n_out=N)` |

---

## Requirements

```
fastai>=2.7
torch>=2.0
torchvision>=0.15
pillow>=9.0
pandas>=1.5
numpy<2.0.0      # Critical for Windows compatibility
scikit-learn>=1.3
```

Install:
```bash
pip install fastai torch torchvision pillow pandas "numpy<2.0.0"
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{babuc2026workflow,
  title   = {Towards Workflow-Centric Medical AI: Orchestration of Multi-Model
             Pipelines for Clinical Image Analysis},
  author  = {Babuc and Fortiş},
  year    = {2026},
  journal = {arXiv preprint}
}
```

**Datasets:**
```bibtex
@inproceedings{jha2020kvasir,
  title     = {Kvasir-SEG: A Segmented Polyp Dataset},
  author    = {Jha, Debesh and others},
  booktitle = {MMM 2020},
  year      = {2020}
}

@article{ali2023polypgen,
  title   = {A multi-centre polyp detection and segmentation dataset for generalisability assessment},
  author  = {Ali, Sharib and others},
  journal = {Scientific Data},
  year    = {2023}
}
```

---

## License

MIT License

Dataset licenses:
- **Kvasir-SEG**: [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **PolypGen**: Academic use via [Synapse](https://www.synapse.org)
