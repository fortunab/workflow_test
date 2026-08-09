"""
prepare_alzheimer_dataset.py
============================
Generates a synthetic Alzheimer's Disease (AD) MRI brain scan dataset.

Images simulate axial T1-weighted MRI slices:
  - Background: dark grey skull outline + white matter
  - Hippocampus region highlighted
  - Atrophy severity varies by class
  - Classes: 0 = CN (Cognitively Normal)
             1 = MCI (Mild Cognitive Impairment)
             2 = AD (Alzheimer's Disease)

Output mirrors the polyp dataset schema so the same pipeline runs unchanged:
  data_alzheimer/
    images/       *.png  (RGB greyscale MRI, 256x256)
    masks/        *_mask.png  (binary hippocampus segmentation mask)
    dataset.csv
"""

import os
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

ALZHEIMER_DIR = os.path.join(os.path.dirname(__file__), "data_alzheimer")
NUM_SAMPLES   = 120        # 40 per class
IMG_SIZE      = 256
SEED          = 4242

random.seed(SEED);  np.random.seed(SEED)


def _oval(draw, cx, cy, rx, ry, fill, outline=None, width=1):
    draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry],
                 fill=fill, outline=outline, width=width)


def generate_mri_slice(label: int, img_size=IMG_SIZE):
    """
    label: 0=CN, 1=MCI, 2=AD
    Returns (PIL.Image RGB, PIL.Image L mask)
    Atrophy increases with label — hippocampus shrinks, ventricles expand.
    """
    # Background: very dark (skull / air)
    arr = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    img  = Image.fromarray(arr, "RGB")
    mask = Image.new("L", (img_size, img_size), 0)
    draw = ImageDraw.Draw(img)
    dm   = ImageDraw.Draw(mask)

    cx, cy = img_size // 2, img_size // 2

    # ── Skull outline ─────────────────────────────────────────────────
    skull_rx, skull_ry = 105, 90
    _oval(draw, cx, cy, skull_rx, skull_ry,
          fill=(40, 38, 38), outline=(200, 195, 195), width=3)

    # ── White matter ──────────────────────────────────────────────────
    wm_rx, wm_ry = 88, 75
    wm_v = 160 + random.randint(-10, 10)
    _oval(draw, cx, cy, wm_rx, wm_ry, fill=(wm_v, wm_v, wm_v))

    # ── Grey matter cortex (slightly darker ring) ─────────────────────
    gm_rx, gm_ry = 78, 66
    gm_v = 125 + random.randint(-8, 8)
    _oval(draw, cx, cy, gm_rx, gm_ry, fill=(gm_v, gm_v, gm_v))

    # ── Ventricles (enlarge with atrophy) ─────────────────────────────
    vent_base = 10 + label * 9   # CN:10, MCI:19, AD:28
    vent_jitter = random.randint(-2, 4)
    vr = vent_base + vent_jitter
    vent_col = (50, 50, 60)
    _oval(draw, cx - 15, cy, vr, int(vr * 1.6), fill=vent_col)   # left
    _oval(draw, cx + 15, cy, vr, int(vr * 1.6), fill=vent_col)   # right

    # ── Hippocampus (atrophies with label) ────────────────────────────
    # CN: full size, MCI: 70%, AD: 45%
    hippo_scale = [1.0, 0.70, 0.45][label]
    h_rx = max(3, int(20 * hippo_scale) + random.randint(-2, 2))
    h_ry = max(2, int(10 * hippo_scale) + random.randint(-1, 1))
    h_col_v = 175 + random.randint(-10, 10)
    h_col   = (h_col_v - 30, h_col_v, h_col_v - 20)   # slightly greenish
    # Left hippocampus
    hl_cx, hl_cy = cx - 38, cy + 18
    _oval(draw, hl_cx, hl_cy, h_rx, h_ry, fill=h_col, outline=(80, 160, 80), width=1)
    _oval(dm,   hl_cx, hl_cy, h_rx, h_ry, fill=255)
    # Right hippocampus
    hr_cx, hr_cy = cx + 38, cy + 18
    _oval(draw, hr_cx, hr_cy, h_rx, h_ry, fill=h_col, outline=(80, 160, 80), width=1)
    _oval(dm,   hr_cx, hr_cy, h_rx, h_ry, fill=255)

    # Add slight Gaussian noise (MRI grain)
    noise = np.random.normal(0, 4, (img_size, img_size, 3)).astype(np.int16)
    arr2  = np.array(img).astype(np.int16) + noise
    img   = Image.fromarray(arr2.clip(0, 255).astype(np.uint8), "RGB")
    img   = img.filter(ImageFilter.GaussianBlur(radius=0.8))

    return img, mask


def generate_alzheimer_dataset(out_dir=ALZHEIMER_DIR, n=NUM_SAMPLES):
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "masks"),  exist_ok=True)

    per_class   = n // 3
    label_names = {0: "CN", 1: "MCI", 2: "AD"}
    records     = []
    idx         = 0

    for label in range(3):
        for _ in range(per_class):
            idx  += 1
            name  = f"mri_{idx:04d}"
            img, mask = generate_mri_slice(label)

            img_rel  = f"images/{name}.png"
            mask_rel = f"masks/{name}_mask.png"
            img.save(os.path.join(out_dir, img_rel))
            mask.save(os.path.join(out_dir, mask_rel))

            split = "valid" if (idx % 5 == 0) else ("test" if (idx % 7 == 0) else "train")

            records.append({
                "image_path":   img_rel,
                "mask_path":    mask_rel,
                "label":        label,
                "label_name":   label_names[label],
                "has_abnormal": int(label > 0),   # 1 = MCI or AD
                # Alias: pipeline sees this as the positive class flag
                "has_polyp":    int(label > 0),
                "is_valid":     split == "valid",
                "split":        split,
                "data_dir":     out_dir,
                "bbox_x":       0.35, "bbox_y": 0.45,
                "bbox_w":       0.30, "bbox_h": 0.20,
            })

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(out_dir, "dataset.csv"), index=False)
    print(f"[OK] Alzheimer MRI dataset: {len(df)} samples → {out_dir}")
    return out_dir


if __name__ == "__main__":
    generate_alzheimer_dataset()
