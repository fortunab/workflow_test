"""
prepare_cervical_dataset.py
===========================
Generates a synthetic cervical cytology (Pap smear) dataset.

Images simulate cervical cells under a microscope:
  - Background: pinkish-purple stained slide
  - Cells: circular/oval blobs with nuclei
  - Abnormal cells: enlarged nuclei (high N/C ratio), irregular shapes
  - Classes: 0 = Normal, 1 = LSIL (Low-grade SIL), 2 = HSIL (High-grade SIL)

Output structure mirrors the polyp dataset so the same pipeline works unchanged:
  data_cervical/
    images/       *.png  (RGB, 256x256)
    masks/        *_mask.png  (binary nucleus mask)
    dataset.csv   (image_path, mask_path, label, has_abnormal, is_valid, split, ...)
"""

import os
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

CERVICAL_DIR  = os.path.join(os.path.dirname(__file__), "data_cervical")
IMG_DIR       = os.path.join(CERVICAL_DIR, "images")
MASK_DIR      = os.path.join(CERVICAL_DIR, "masks")
NUM_SAMPLES   = 120          # 0→Normal, 1→LSIL, 2→HSIL (40 each)
IMG_SIZE      = 256
SEED          = 1234

random.seed(SEED);  np.random.seed(SEED)


def _draw_cell(draw, cx, cy, r_cell, r_nucleus, abnormal=False):
    """Draw one cervical cell: cytoplasm circle + nucleus."""
    # Cytoplasm — pale pink
    fill_c = (220 + random.randint(-15, 15),
              175 + random.randint(-15, 15),
              185 + random.randint(-15, 15))
    draw.ellipse([cx - r_cell, cy - r_cell, cx + r_cell, cy + r_cell],
                 fill=fill_c, outline=(160, 100, 110), width=1)

    # Nucleus — dark purple/blue, enlarged if abnormal
    fill_n = (90 + random.randint(-20, 20),
              60 + random.randint(-10, 10),
              140 + random.randint(-20, 20))
    draw.ellipse([cx - r_nucleus, cy - r_nucleus,
                  cx + r_nucleus, cy + r_nucleus],
                 fill=fill_n, outline=(50, 30, 80), width=1)


def generate_cervical_image(label: int, img_size=IMG_SIZE):
    """
    label: 0=Normal, 1=LSIL, 2=HSIL
    Returns (PIL.Image RGB, PIL.Image L mask)
    """
    # Background: pinkish-purple haematoxylin-eosin slide
    bg = np.full((img_size, img_size, 3), [235, 210, 215], dtype=np.int16)
    noise = np.random.randint(-8, 8, bg.shape, dtype=np.int16)
    bg = (bg + noise).clip(0, 255).astype(np.uint8)

    img  = Image.fromarray(bg, "RGB")
    mask = Image.new("L", (img_size, img_size), 0)
    draw_img  = ImageDraw.Draw(img)
    draw_mask = ImageDraw.Draw(mask)

    n_cells = random.randint(6, 14)
    for _ in range(n_cells):
        cx = random.randint(25, img_size - 25)
        cy = random.randint(25, img_size - 25)

        if label == 0:            # Normal — small nucleus, high C/N ratio
            r_cell    = random.randint(14, 22)
            r_nucleus = random.randint(4, 7)
            abnormal  = False
        elif label == 1:          # LSIL — moderately enlarged nucleus
            r_cell    = random.randint(14, 22)
            r_nucleus = random.randint(8, 12)
            abnormal  = True
        else:                     # HSIL — very large nucleus, minimal cytoplasm
            r_cell    = random.randint(14, 22)
            r_nucleus = random.randint(12, 18)
            abnormal  = True

        _draw_cell(draw_img, cx, cy, r_cell, r_nucleus, abnormal)

        # Binary mask: nucleus pixels = 255
        if abnormal or label == 0:
            draw_mask.ellipse([cx - r_nucleus, cy - r_nucleus,
                               cx + r_nucleus, cy + r_nucleus],
                              fill=255)

    img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
    return img, mask


def generate_cervical_dataset(out_dir=CERVICAL_DIR, n=NUM_SAMPLES):
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "masks"),  exist_ok=True)

    per_class = n // 3
    records   = []
    idx       = 0

    label_names = {0: "normal", 1: "lsil", 2: "hsil"}

    for label in range(3):
        for _ in range(per_class):
            idx   += 1
            name   = f"cervical_{idx:04d}"
            img, mask = generate_cervical_image(label)

            img_rel  = f"images/{name}.png"
            mask_rel = f"masks/{name}_mask.png"

            img.save(os.path.join(out_dir, img_rel))
            mask.save(os.path.join(out_dir, mask_rel))

            split = "valid" if (idx % 5 == 0) else ("test" if (idx % 7 == 0) else "train")

            records.append({
                "image_path":    img_rel,
                "mask_path":     mask_rel,
                "label":         label,
                "label_name":    label_names[label],
                "has_abnormal":  int(label > 0),    # 1 = abnormal (LSIL or HSIL)
                # Alias so existing pipeline sees 'has_polyp' = 'has_abnormal'
                "has_polyp":     int(label > 0),
                "is_valid":      split == "valid",
                "split":         split,
                "data_dir":      out_dir,
                "bbox_x":        0.3, "bbox_y": 0.3,
                "bbox_w":        0.4, "bbox_h": 0.4,
            })

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(out_dir, "dataset.csv"), index=False)
    print(f"[OK] Cervical dataset: {len(df)} samples → {out_dir}")
    return out_dir


if __name__ == "__main__":
    generate_cervical_dataset()
