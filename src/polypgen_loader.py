import os
import json
import pandas as pd
import numpy as np
from PIL import Image

# Center-based split: C1-C4 = train, C5 = val, C6 = test
CENTER_SPLIT_MAP = {
    "C1": "train", "C2": "train", "C3": "train", "C4": "train",
    "C5": "val",
    "C6": "test"
}

def load_polypgen(dataset_path, output_data_dir=None, seed=42):
    """
    Parses the PolypGen multi-center dataset and converts it to our
    standard dataset.csv + vqa_data.json pipeline format.

    Expected PolypGen folder layout:
        dataset_path/
        ├── data_C1/
        │   ├── images_C1/         # polyp-positive images
        │   └── masks_C1/          # segmentation masks
        ├── data_C1_negative/
        │   └── images_C1_negative/ # normal tissue images (no polyp, no mask)
        ├── ...  (C2 through C6)
        └── bbox_annotation.json   # bounding box annotations per image

    Args:
        dataset_path:    Path to downloaded PolypGen root folder.
        output_data_dir: Where to write dataset.csv and vqa_data.json.
                         Defaults to dataset_path/processed/
        seed:            Random seed.
    """
    np.random.seed(seed)

    if output_data_dir is None:
        output_data_dir = os.path.join(dataset_path, "processed")
    os.makedirs(output_data_dir, exist_ok=True)

    # Load bounding box annotations if present
    bbox_json_path = os.path.join(dataset_path, "bbox_annotation.json")
    if os.path.isfile(bbox_json_path):
        with open(bbox_json_path, "r") as f:
            bbox_data = json.load(f)
    else:
        bbox_data = {}
        print("[PolypGen] Warning: bbox_annotation.json not found. BBoxes will be derived from masks.")

    metadata = []
    vqa_data = []

    for center_id in ["C1", "C2", "C3", "C4", "C5", "C6"]:
        split = CENTER_SPLIT_MAP[center_id]

        # -------------------------------------------------------
        # 1. Polyp-positive samples
        # -------------------------------------------------------
        pos_images_dir = os.path.join(dataset_path, f"data_{center_id}", f"images_{center_id}")
        pos_masks_dir  = os.path.join(dataset_path, f"data_{center_id}", f"masks_{center_id}")

        if os.path.isdir(pos_images_dir):
            image_files = sorted([
                f for f in os.listdir(pos_images_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])
            print(f"[PolypGen] {center_id} positive: {len(image_files)} images -> split={split}")

            for img_file in image_files:
                stem = os.path.splitext(img_file)[0]

                img_path_rel  = os.path.join(f"data_{center_id}", f"images_{center_id}", img_file)
                mask_path_rel = None

                # Find mask
                if os.path.isdir(pos_masks_dir):
                    for ext in [".jpg", ".jpeg", ".png"]:
                        candidate = stem + ext
                        if os.path.exists(os.path.join(pos_masks_dir, candidate)):
                            mask_path_rel = os.path.join(f"data_{center_id}", f"masks_{center_id}", candidate)
                            break

                # Compute area & bbox
                area_px = 0
                rel_area = 0.0
                ymin = xmin = ymax = xmax = 0

                if mask_path_rel and os.path.exists(os.path.join(dataset_path, mask_path_rel)):
                    mask_np  = np.array(Image.open(os.path.join(dataset_path, mask_path_rel)).convert("L"))
                    area_px  = int(np.sum(mask_np > 128))
                    total_px = mask_np.shape[0] * mask_np.shape[1]
                    rel_area = round(float(area_px) / total_px, 4)
                    rows = np.any(mask_np > 128, axis=1)
                    cols = np.any(mask_np > 128, axis=0)
                    if rows.any():
                        ymin, ymax = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
                        xmin, xmax = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])

                # Override bbox from JSON annotation if available
                bbox_entry = bbox_data.get(stem, bbox_data.get(img_file, None))
                if bbox_entry:
                    xmin = int(bbox_entry.get("xmin", xmin))
                    ymin = int(bbox_entry.get("ymin", ymin))
                    xmax = int(bbox_entry.get("xmax", xmax))
                    ymax = int(bbox_entry.get("ymax", ymax))

                metadata.append({
                    "image_id":      f"{center_id}_{stem}",
                    "image_path":    img_path_rel,
                    "mask_path":     mask_path_rel if mask_path_rel else "",
                    "label":         "polyp",
                    "has_polyp":     1,
                    "ymin":          ymin,
                    "xmin":          xmin,
                    "ymax":          ymax,
                    "xmax":          xmax,
                    "area_px":       area_px,
                    "relative_area": rel_area,
                    "split":         split,
                    "source":        f"polypgen-{center_id}"
                })

                vqa_data.append({
                    "image_id":   f"{center_id}_{stem}",
                    "image_path": img_path_rel,
                    "question":   "Does this endoscopic image show a polyp lesion?",
                    "answer":     (
                        f"Yes. A polyp lesion is detected at region [{xmin}, {ymin}, {xmax}, {ymax}], "
                        f"occupying {rel_area * 100:.1f}% of the field of view. Clinical center: {center_id}."
                    ),
                    "label": "polyp"
                })
        else:
            print(f"[PolypGen] {center_id} positive images folder not found, skipping.")

        # -------------------------------------------------------
        # 2. Polyp-negative samples (normal tissue)
        # -------------------------------------------------------
        neg_images_dir = os.path.join(dataset_path, f"data_{center_id}_negative", f"images_{center_id}_negative")

        if os.path.isdir(neg_images_dir):
            neg_files = sorted([
                f for f in os.listdir(neg_images_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])
            print(f"[PolypGen] {center_id} negative: {len(neg_files)} images -> split={split}")

            for img_file in neg_files:
                stem = os.path.splitext(img_file)[0]
                img_path_rel = os.path.join(
                    f"data_{center_id}_negative", f"images_{center_id}_negative", img_file
                )

                metadata.append({
                    "image_id":      f"{center_id}_neg_{stem}",
                    "image_path":    img_path_rel,
                    "mask_path":     "",
                    "label":         "normal",
                    "has_polyp":     0,
                    "ymin":          0, "xmin": 0, "ymax": 0, "xmax": 0,
                    "area_px":       0,
                    "relative_area": 0.0,
                    "split":         split,
                    "source":        f"polypgen-{center_id}-negative"
                })

                vqa_data.append({
                    "image_id":   f"{center_id}_neg_{stem}",
                    "image_path": img_path_rel,
                    "question":   "Is there any mucosal abnormality or polyp lesion detected in this frame?",
                    "answer":     "No polyp lesion or mucosal abnormality detected. Normal endoscopic appearance.",
                    "label": "normal"
                })

    # Save outputs
    df = pd.DataFrame(metadata)
    csv_path = os.path.join(output_data_dir, "dataset.csv")
    vqa_path = os.path.join(output_data_dir, "vqa_data.json")
    df.to_csv(csv_path, index=False)
    with open(vqa_path, "w") as f:
        json.dump(vqa_data, f, indent=2)

    print(f"\n[PolypGen] Total samples processed: {len(df)}")
    print(f"  Polyp-positive: {len(df[df['has_polyp']==1])}")
    print(f"  Polyp-negative: {len(df[df['has_polyp']==0])}")
    print(f"  Train: {len(df[df['split']=='train'])}  |  "
          f"Val: {len(df[df['split']=='val'])}  |  "
          f"Test: {len(df[df['split']=='test'])}")
    print(f"[PolypGen] Saved to: {output_data_dir}")
    return output_data_dir
