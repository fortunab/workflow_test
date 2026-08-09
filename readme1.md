# Workflow-Centric Medical AI Pipeline: Multi-Model Orchestration

<div align="center">

[![Framework: fast.ai](https://img.shields.io/badge/Framework-fast.ai-blue.svg)](https://www.fast.ai/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-PathVQA%20%7C%20VQA--RAD-yellow.svg)](https://huggingface.co/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Agentic multi-model orchestration framework coordinating Classification ($f_{cls}$), Detection ($f_{det}$), Segmentation ($f_{seg}$), and VLM Reasoning ($f_{vlm}$) across Gastroenterology, Gynecologic Oncology, and Neurology.**

[🌐 Website Dashboard (localhost:8081)](#website-dashboard) · [🔬 Colorectal Pipeline](#1-colorectal-endoscopy-pipeline-polyp) · [🧫 Cervical Pipeline](#2-cervical-cytology-pipeline-pap-smear) · [🧠 Alzheimer's Pipeline](#3-alzheimers-mri-pipeline-dementia-staging) · [📊 SOTA Benchmarks](#experimental-benchmarks-paper-results)

</div>

---

## Architecture Overview

Instead of running heavy standalone vision models independently, **PolypFlow** organizes models into a directed sequential pipeline emitting standardized XML tokens after each stage:

$$ y = f_{vlm}(f_{seg}(f_{det}(f_{cls}(x)))) $$

```
[ Input Clinical Image (x) ]
          │
          ▼
    ┌───────────┐
    │   f_cls   │──► [CLS:label·conf=0.95]  ← Token Gating Threshold
    └───────────┘
          │ (pathology positive)
          ▼
    ┌───────────┐
    │   f_det   │──► [DET:bbox=(x1,y1,x2,y2)·conf=0.91]
    └───────────┘
          │
          ▼
    ┌───────────┐
    │   f_seg   │──► [SEG:dice=0.901·area=18.4%]
    └───────────┘
          │
          ▼
    ┌───────────┐
    │   f_vlm   │──► Clinical Diagnostic Report + Treatment Guidance
    └───────────┘
```

- **Selective Execution Mode:** If $f_{cls}$ predicts a normal non-pathological frame, downstream detection and segmentation are bypassed, reducing average latency from **420 ms to 270 ms (-35.7%)** and GPU utilization from **100% to 65%** on an NVIDIA Tesla T4.

---

## Use Case 1: Colorectal Endoscopy Pipeline (Polyp)

### 1. Dataset Download
- **Kvasir-SEG (In-Domain):** 1,000 annotated polyp images with bounding boxes and per-pixel masks.
  ```powershell
  Invoke-WebRequest -Uri "https://datasets.simula.no/downloads/kvasir-seg.zip" -OutFile "Kvasir-SEG.zip"
  Expand-Archive -Path "Kvasir-SEG.zip" -DestinationPath "."
  ```
- **PolypGen (Multi-Center Cross-Dataset):** 3,762 positive + 2,000+ normal frames across 6 clinical centers (C1–C6).
  ```bash
  pip install synapseclient
  synapse get syn26376615 --recursive
  ```

### 2. Preprocessing
- Images resized to $224 \times 224$ with ImageNet mean/std normalization.
- Mask values remapped from 0/255 to 0/1 binary float tensors via `_SegDataset`.
- Bounding boxes normalized to $[x_{min}, y_{min}, x_{max}, y_{max}]$ in range $[0, 1]$.
```powershell
python main.py --dataset kvasir --data-path .\Kvasir-SEG --prepare-data
```

### 3. Training & LoRA Adaptation
- **$f_{cls}$ (Classification):** ResNet-50 / FastAI trained with categorical cross-entropy.
- **$f_{det}$ (Detection):** YOLOv8-s single-stage object detector ($mAP@0.5 = 0.910$).
- **$f_{seg}$ (Segmentation):** `facebook/sam2.1-hiera-tiny` + LoRA adapter ($r=8, \alpha=16$, dropout 0.05) trained with BCE + Dice Loss.
```powershell
python main.py --dataset kvasir --data-path .\Kvasir-SEG --train --epochs 20
```

### 4. Processing & Inference
- Runs full or selective token-guided pipeline:
```powershell
python main.py --predict path/to/frame.jpg --mode selective
```

### 5. Optimization
- **Merged Adapter Deployment:** LoRA weights merged into the SAM 2.1 backbone for zero latency penalty during evaluation (**120 ms** latency).
- **Mixed Precision FP16:** Enabled for CUDA evaluation.

---

## Use Case 2: Cervical Cytology Pipeline (Pap Smear)

### 1. Dataset Download & Integration
- **SIPaKMeD / Herlev:** Microscope Pap smear slides covering 3 cell categories (0: Normal, 1: LSIL, 2: HSIL).
- **Hugging Face PathVQA (VLM Reasoning):** Repository **[`flaviagiammarino/path-vqa`](https://huggingface.co/datasets/flaviagiammarino/path-vqa)** (Paper Reference [29]).

### 2. Preprocessing
- Synthetic microscope slide generator [`prepare_cervical_dataset.py`](prepare_cervical_dataset.py) simulates Haematoxylin & Eosin (H&E) staining.
- Nucleus boundary extraction for binary ground-truth segmentation masks.
```powershell
python prepare_cervical_dataset.py
```

### 3. Training & Domain Adaptation
- Dedicated model directory: `models_cervical/`
- **$f_{cls}$:** 3-class cell dysplasia classifier.
- **$f_{seg}$:** SAM 2.1 + LoRA fine-tuned on cervical cell nucleus boundaries (**mDice 0.896**).
- **$f_{vlm}$:** `Qwen3.5-0.8B-LoRA` fine-tuned on PathVQA.
```powershell
python run_experiments.py --dataset cervical
```

### 4. Processing & Inference
- Gated execution skips cell nucleus segmentation if cytology slide is classified as normal.

### 5. Optimization
- Color jitter and stain normalization during training for cross-center slide variance.

---

## Use Case 3: Alzheimer's MRI Pipeline (Dementia Staging)

### 1. Dataset Download & Integration
- **ADNI / HarP Benchmark:** T1-weighted axial brain MRI slices categorized into 3 clinical stages (0: CN - Cognitively Normal, 1: MCI - Mild Cognitive Impairment, 2: AD - Alzheimer's Disease).
- **Hugging Face VQA-RAD (VLM Reasoning):** Repository **[`flaviagiammarino/vqa-rad`](https://huggingface.co/datasets/flaviagiammarino/vqa-rad)** (Paper Reference [26]).

### 2. Preprocessing
- Synthetic MRI slice generator [`prepare_alzheimer_dataset.py`](prepare_alzheimer_dataset.py) models skull stripping, white/grey matter intensity, ventricle enlargement, and hippocampal atrophy.
- Binary masks isolate left/right hippocampus regions.
```powershell
python prepare_alzheimer_dataset.py
```

### 3. Training & Domain Adaptation
- Dedicated model directory: `models_alzheimer/`
- **$f_{cls}$:** Dementia staging classifier.
- **$f_{seg}$:** SAM 2.1 + LoRA fine-tuned on hippocampal volume segmentation (**mDice 0.641 / 0.880 ADNI HarP**).
- **$f_{vlm}$:** `Qwen3.5-0.8B-LoRA` fine-tuned on VQA-RAD.
```powershell
python run_experiments.py --dataset alzheimer
```

### 4. Processing & Inference
- Hippocampal region volume ratio encoded into `<SEG:mask area_ratio=.../>` token for VLM cognitive reporting.

### 5. Optimization
- Noise injection ($\sigma=4.0$) simulating MRI scanner grain for robust feature representation.

---

## Experimental Benchmarks (Paper Results)

### 1. Stage 1: Classification ($f_{cls}$)

| Domain | Dataset | Pathology | Accuracy ↑ | AUC-ROC ↑ | F1-Score ↑ | Specificity ↑ |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Colorectal** | Kvasir / PolypGen | Polyps | **0.952** | **0.987** | **0.951** | **0.949** |
| **Cervical** | SIPaKMeD | Abnormal Cells | **0.950** | **0.950** | **0.940** | **0.950** |
| **Alzheimer's** | ADNI / HarP | MCI / AD Staging | **0.900** | **0.920** | **0.890** | **0.910** |

### 2. Stage 2: Detection ($f_{det}$) — YOLOv8

| Model | mAP@0.5 ↑ | mAP@0.75 ↑ | Recall ↑ | FPS (Tesla T4) ↑ |
| :--- | :---: | :---: | :---: | :---: |
| Faster R-CNN | 0.814 | 0.671 | 0.832 | 18 FPS |
| RetinaNet | 0.828 | 0.692 | 0.854 | 24 FPS |
| YOLOv5-s | 0.856 | 0.718 | 0.873 | 112 FPS |
| **YOLOv8-s (Ours)** | **0.910** | **0.741** | **0.893** | **128 FPS** |

### 3. Stage 3: Segmentation ($f_{seg}$) — SAM 2.1 + LoRA (Table 2)

| Split / Setup | Method | mIoU ↑ | mDice ↑ | mAP ↑ |
| :--- | :--- | :---: | :---: | :---: |
| **In-Domain (Kvasir-SEG)** | U-Net | 0.78 | 0.84 | 0.81 |
| | DeepLabv3 | 0.80 | 0.85 | 0.83 |
| | SAM 2.1 (base) | 0.84 | 0.87 | 0.86 |
| | **SAM 2.1 + LoRA (Ours)** | **0.86** | **0.90** | **0.89** |
| **Cross-Dataset Transfer (PolypGen)** | U-Net | 0.69 | 0.76 | 0.71 |
| | DeepLabv3 | 0.72 | 0.79 | 0.75 |
| | SAM 2.1 (base) | 0.75 | 0.80 | 0.78 |
| | **SAM 2.1 + LoRA (Ours)** | **0.78** | **0.83** | **0.81** |

### 4. Stage 4: VLM Reasoning ($f_{vlm}$) — Qwen3.5-0.8B-LoRA (Tables 4 & 5)

| Regime | Model | ROUGE-1 ↑ | ROUGE-L ↑ | BLEU ↑ | METEOR ↑ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Small-Budget (1k / 30 steps)** | Base Model | 0.1510 | 0.1036 | 0.0084 | 0.1881 |
| | **qwen3.5-0.8B-LoRA (1k)** | **0.4775** | **0.4419** | **0.2443** | **0.4380** |
| **Large-Budget (10k / 500 steps)** | Base Model | 0.1471 | 0.1004 | 0.0084 | 0.1886 |
| | **qwen3.5-0.8B-LoRA (10k)** | **0.6704** | **0.6105** | **0.3912** | **0.6288** |

### 5. Token Orchestration & Efficiency Ablation (Tables 6 & 9)

| Execution Paradigm | End-to-End Acc ↑ | ROUGE-L ↑ | Latency (T4) ↓ | GPU Load (T4) ↓ |
| :--- | :---: | :---: | :---: | :---: |
| Baseline (independent) | 0.78 | 0.45 | 320 ms | 80% |
| Ensemble (parallel) | 0.80 | 0.52 | 560 ms | 95% |
| Pipeline w/o tokens | 0.83 | 0.55 | 420 ms | 100% |
| **Ours (Full Mode)** | **0.88** | **0.61** | 420 ms | 100% |
| **Ours (Selective Mode)** | **0.88** | **0.61** | **270 ms (-35.7%)** | **65% (-35%)** |

---

## Running Experiments via CLI

```powershell
# Run ALL 3 medical domains in sequence (Polyp, Cervical, Alzheimer)
python run_paper_experiments.py

# Run standard single-domain evaluations
python run_experiments.py --dataset synthetic
python run_experiments.py --dataset cervical
python run_experiments.py --dataset alzheimer
```

---

## Website Dashboard

The paper results dashboard is served locally on port 8081:

```powershell
python -m http.server 8081 --bind 127.0.0.1
```

Open **`http://127.0.0.1:8081`** in your browser to view interactive benchmark charts, token routing diagrams, and performance comparisons.

---

## Windows Requirements & Troubleshooting

- **NumPy Version:** Strictly enforce `numpy<2.0.0` to avoid C-extension DLL conflicts with PyTorch/pandas on Windows.
- **Multiprocessing:** Use `num_workers=0` in PyTorch DataLoaders to prevent Windows process fork pickling errors.

---

## Citation

```bibtex
@article{babuc2026workflow,
  title   = {Towards Workflow-Centric Medical AI: Orchestration of Multi-Model Pipelines for Clinical Image Analysis},
  author  = {Diogen Babuc and Teodor-Florin Fortiş},
  journal = {MDPI Future Internet},
  year    = {2026}
}
```
