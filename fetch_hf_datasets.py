"""
fetch_hf_datasets.py
====================
Utility to inspect and connect to Hugging Face Medical VQA datasets referenced in the paper:
  - PathVQA: flaviagiammarino/path-vqa (Reference [29] in paper)
  - VQA-RAD: flaviagiammarino/vqa-rad (Reference [26] in paper)
  - Brain / Radiology VQA benchmarks
"""

import json
import urllib.request

HF_DATASETS = [
    ("PathVQA (Cytology & Pathology VQA)", "flaviagiammarino/path-vqa"),
    ("VQA-RAD (Radiology & MRI VQA)",      "flaviagiammarino/vqa-rad"),
]

def check_huggingface_datasets():
    print("=" * 70)
    print("  Hugging Face Medical VQA Dataset Integration Check")
    print("=" * 70)
    for name, repo in HF_DATASETS:
        url = f"https://huggingface.co/api/datasets/{repo}"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                print(f"[FOUND] {name:40s} | HF Repo: {repo}")
                print(f"        Downloads: {data.get('downloads', 0):,d} | Likes: {data.get('likes', 0)}")
                print(f"        URL: https://huggingface.co/datasets/{repo}\n")
        except Exception as e:
            print(f"[ERROR] {repo}: {e}\n")

if __name__ == "__main__":
    check_huggingface_datasets()
