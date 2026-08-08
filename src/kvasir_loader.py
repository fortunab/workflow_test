import os
import json
import pandas as pd
import numpy as np
from PIL import Image

def load_kvasir_seg(dataset_path, output_data_dir=None, train_ratio=0.70, val_ratio=0.15, seed=42):
    """
    Parses the Kvasir-SEG dataset folder structure and converts it to our
    standard dataset.csv + vqa_data.json pipeline format.

    Expected Kvasir-SEG folder layout:
        dataset_path/
        ├── images/              # 1000 polyp .jpg images
        ├── masks/               # 1000 binary .jpg masks
        └── kavsir_bboxes.json   # bbox annotations

    Args:
        dataset_path:   Path to downloaded Kvasir-SEG root folder.
        output_data_dir: Where to write dataset.csv and vqa_data.json.
                         Defaults to dataset_path/processed/
        train_ratio:    Fraction for training split (default 0.70).
        val_ratio:      Fraction for validation split (default 0.15).
        seed:           Random seed for reproducibility.
    """
    np.random.seed(seed)

    images_dir = os.path.join(dataset_path, "images")
    masks_dir  = os.path.join(dataset_path, "masks")
    bbox_json  = os.path.join(dataset_path, "kavsir_bboxes.json")

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"[Kvasir-SEG] 'images/' folder not found in: {dataset_path}")
    if not os.path.isdir(masks_dir):
        raise FileNotFoundError(f"[Kvasir-SEG] 'masks/' folder not found in: {dataset_path}")
    if not os.path.isfile(bbox_json):
        raise FileNotFoundError(f"[Kvasir-SEG] 'kavsir_bboxes.json' not found in: {dataset_path}")

    if output_data_dir is None:
        output_data_dir = os.path.join(dataset_path, "processed")
    os.makedirs(output_data_dir, exist_ok=True)

    with open(bbox_json, "r") as f:
        bbox_data = json.load(f)

    image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    print(f"[Kvasir-SEG] Found {len(image_files)} images.")

    # Assign train/val/test splits
    indices = np.arange(len(image_files))
    np.random.shuffle(indices)
    n_train = int(len(indices) * train_ratio)
    n_val   = int(len(indices) * val_ratio)

    split_map = {}
    for i, idx in enumerate(indices):
        fname = image_files[idx]
        if i < n_train:
            split_map[fname] = "train"
        elif i < n_train + n_val:
            split_map[fname] = "val"
        else:
            split_map[fname] = "test"

    metadata = []
    vqa_data = []

    for img_file in image_files:
        stem = os.path.splitext(img_file)[0]

        # Find matching mask (same stem, any extension)
        mask_file = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = stem + ext
            if os.path.exists(os.path.join(masks_dir, candidate)):
                mask_file = candidate
                break

        if mask_file is None:
            print(f"  [WARN] No mask found for {img_file}, skipping.")
            continue

        img_path_rel  = os.path.join("images", img_file)
        mask_path_rel = os.path.join("masks", mask_file)

        # Load mask to compute area
        mask_np  = np.array(Image.open(os.path.join(masks_dir, mask_file)).convert("L"))
        area_px  = int(np.sum(mask_np > 128))
        total_px = mask_np.shape[0] * mask_np.shape[1]
        rel_area = round(float(area_px) / total_px, 4)

        # Bounding box from JSON (if present)
        bbox_info = bbox_data.get(stem, bbox_data.get(img_file, None))
        if bbox_info:
            xmin = int(bbox_info.get("xmin", 0))
            ymin = int(bbox_info.get("ymin", 0))
            xmax = int(bbox_info.get("xmax", 0))
            ymax = int(bbox_info.get("ymax", 0))
        else:
            # Derive bbox from mask if JSON entry missing
            rows = np.any(mask_np > 128, axis=1)
            cols = np.any(mask_np > 128, axis=0)
            if rows.any():
                ymin, ymax = int(np.where(rows)[0][[0, -1]])
                xmin, xmax = int(np.where(cols)[0][[0, -1]])
            else:
                ymin = xmin = ymax = xmax = 0

        split = split_map.get(img_file, "train")

        metadata.append({
            "image_id":      stem,
            "image_path":    img_path_rel,
            "mask_path":     mask_path_rel,
            "label":         "polyp",
            "has_polyp":     1,
            "ymin":          ymin,
            "xmin":          xmin,
            "ymax":          ymax,
            "xmax":          xmax,
            "area_px":       area_px,
            "relative_area": rel_area,
            "split":         split,
            "source":        "kvasir-seg"
        })

        vqa_data.append({
            "image_id":   stem,
            "image_path": img_path_rel,
            "question":   "Does this endoscopic image show a polyp lesion, and what are its spatial characteristics?",
            "answer":     (
                f"The endoscopic image shows a polyp lesion present at region "
                f"[{xmin}, {ymin}, {xmax}, {ymax}] occupying {rel_area * 100:.1f}% "
                f"of the field of view. Biopsy/resection recommended."
            ),
            "label": "polyp"
        })

    df = pd.DataFrame(metadata)
    csv_path = os.path.join(output_data_dir, "dataset.csv")
    vqa_path = os.path.join(output_data_dir, "vqa_data.json")
    df.to_csv(csv_path, index=False)
    with open(vqa_path, "w") as f:
        json.dump(vqa_data, f, indent=2)

    print(f"[Kvasir-SEG] Processed {len(df)} samples.")
    print(f"  Train: {len(df[df['split']=='train'])}  |  "
          f"Val: {len(df[df['split']=='val'])}  |  "
          f"Test: {len(df[df['split']=='test'])}")
    print(f"[Kvasir-SEG] Saved to: {output_data_dir}")
    return output_data_dir
