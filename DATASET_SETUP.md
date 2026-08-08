# Dataset Setup Guide

This guide covers how to download and prepare the **Kvasir-SEG** and **PolypGen** datasets to use with the Workflow-Centric Medical AI pipeline.

---

## Option A: Kvasir-SEG

### 1. Download

Kvasir-SEG is publicly available from Simula Research Laboratory.

**Manual Download:**
Visit: [https://datasets.simula.no/kvasir-seg/](https://datasets.simula.no/kvasir-seg/)  
Click **"Download Dataset"** → Download `Kvasir-SEG.zip` (~44 MB)

**Or via PowerShell (Windows):**
```powershell
Invoke-WebRequest -Uri "https://datasets.simula.no/downloads/kvasir-seg.zip" -OutFile "Kvasir-SEG.zip"
Expand-Archive -Path "Kvasir-SEG.zip" -DestinationPath "."
```

**Or via Bash (Linux/macOS):**
```bash
wget https://datasets.simula.no/downloads/kvasir-seg.zip
unzip kvasir-seg.zip
```

### 2. Expected Folder Structure After Extraction
```
Kvasir-SEG/
├── images/              # 1000 polyp images (.jpg)
├── masks/               # 1000 binary segmentation masks (.jpg)
└── kavsir_bboxes.json   # Bounding box annotations
```

### 3. Run Data Preparation

**Windows PowerShell:**
```powershell
python .\main.py --dataset kvasir --data-path .\Kvasir-SEG --prepare-data
```

**Bash / macOS / Linux:**
```bash
python main.py --dataset kvasir --data-path ./Kvasir-SEG --prepare-data
```

This will generate:
```
Kvasir-SEG/processed/
├── dataset.csv       # Standard pipeline format (1000 rows)
└── vqa_data.json     # Clinical VQA question-answer pairs
```

### 4. Train on Kvasir-SEG

**Windows PowerShell:**
```powershell
python .\main.py --dataset kvasir --data-path .\Kvasir-SEG --train --epochs 20
```

**Bash / macOS / Linux:**
```bash
python main.py --dataset kvasir --data-path ./Kvasir-SEG --train --epochs 20
```

### 5. Evaluate & Predict

```powershell
# Evaluation
python .\main.py --dataset kvasir --data-path .\Kvasir-SEG --eval

# Single image inference (selective mode)
python .\main.py --dataset kvasir --data-path .\Kvasir-SEG --predict .\Kvasir-SEG\images\cju0qkwl9lqtv0993l0dewei2.jpg --mode selective
```

---

## Option B: PolypGen

### 1. Download

PolypGen is hosted on **Synapse** and requires a free account registration.

1. Go to: [https://www.synapse.org/#!Synapse:syn26376615](https://www.synapse.org/#!Synapse:syn26376615)
2. Register / Login with a free Synapse account.
3. Accept the dataset terms of use.
4. Download the full dataset archive.

**Or via Synapse Python Client:**
```bash
pip install synapseclient
synapse get syn26376615 --recursive
```

### 2. Expected Folder Structure After Extraction
```
PolypGen/
├── data_C1/
│   ├── images_C1/            # Polyp-positive images, center 1
│   └── masks_C1/             # Segmentation masks, center 1
├── data_C1_negative/
│   └── images_C1_negative/   # Normal tissue (no polyp), center 1
├── data_C2/ ... data_C6/     # Repeat for centers 2-6
├── data_C2_negative/ ... data_C6_negative/
└── bbox_annotation.json      # Bounding box annotations
```

**Center-Based Split Used:**
| Centers | Split |
|:---|:---|
| C1, C2, C3, C4 | **Train** |
| C5 | **Validation** |
| C6 | **Test** |

### 3. Run Data Preparation

**Windows PowerShell:**
```powershell
python .\main.py --dataset polypgen --data-path .\PolypGen --prepare-data
```

**Bash / macOS / Linux:**
```bash
python main.py --dataset polypgen --data-path ./PolypGen --prepare-data
```

This will generate:
```
PolypGen/processed/
├── dataset.csv       # Standard pipeline format (all centers combined)
└── vqa_data.json     # Clinical VQA question-answer pairs
```

### 4. Train on PolypGen

**Windows PowerShell:**
```powershell
python .\main.py --dataset polypgen --data-path .\PolypGen --train --epochs 20
```

**Bash / macOS / Linux:**
```bash
python main.py --dataset polypgen --data-path ./PolypGen --train --epochs 20
```

### 5. Evaluate & Predict

```powershell
# Evaluation
python .\main.py --dataset polypgen --data-path .\PolypGen --eval

# Single image inference (full mode)
python .\main.py --dataset polypgen --data-path .\PolypGen --predict .\PolypGen\data_C1\images_C1\sample.jpg --mode full
```

---

## Quick Reference: All Dataset Modes

| Dataset | Prepare Data | Train | Eval |
|:---|:---|:---|:---|
| **Synthetic** | `python main.py --prepare-data` | `python main.py --train` | `python main.py --eval` |
| **Kvasir-SEG** | `python main.py --dataset kvasir --data-path ./Kvasir-SEG --prepare-data` | `python main.py --dataset kvasir --data-path ./Kvasir-SEG --train` | `python main.py --dataset kvasir --data-path ./Kvasir-SEG --eval` |
| **PolypGen** | `python main.py --dataset polypgen --data-path ./PolypGen --prepare-data` | `python main.py --dataset polypgen --data-path ./PolypGen --train` | `python main.py --dataset polypgen --data-path ./PolypGen --eval` |

---

> **Note:** All three dataset modes produce the same standard `dataset.csv` schema — the training, tokenization, orchestration, and evaluation scripts run **identically** regardless of which dataset you use.
