import os
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
import cv2

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
MASKS_DIR = os.path.join(DATA_DIR, "masks")

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(MASKS_DIR, exist_ok=True)

def generate_synthetic_polyp_dataset(num_samples=100, img_size=(224, 224), seed=42):
    """
    Generates a realistic synthetic endoscopy polyp dataset (images, masks, bboxes, VQA pairs)
    matching Kvasir-SEG & PolypGen data structures.
    """
    np.random.seed(seed)
    metadata = []
    vqa_data = []

    for i in range(num_samples):
        filename = f"polyp_{i:04d}.jpg"
        mask_filename = f"polyp_{i:04d}_mask.png"
        img_path = os.path.join(IMAGES_DIR, filename)
        mask_path = os.path.join(MASKS_DIR, mask_filename)

        # 1. Base mucosal tissue image (reddish/pinkish background with lighting variation)
        base = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
        base[:, :, 0] = np.random.randint(140, 210, size=img_size, dtype=np.uint8)  # Red channel
        base[:, :, 1] = np.random.randint(40, 90, size=img_size, dtype=np.uint8)    # Green channel
        base[:, :, 2] = np.random.randint(30, 70, size=img_size, dtype=np.uint8)    # Blue channel

        # Add texture & lighting gradient
        x = np.linspace(-1, 1, img_size[0])
        y = np.linspace(-1, 1, img_size[1])
        xx, yy = np.meshgrid(x, y)
        vignette = 1 - 0.4 * (xx**2 + yy**2)
        vignette = np.clip(vignette, 0.2, 1.0)[:, :, None]
        base = (base * vignette).astype(np.uint8)

        # 2. Decide if polyp lesion exists (80% polyp positive, 20% normal mucosal tissue)
        has_polyp = (i % 5 != 0)  # 80% positive
        
        mask = np.zeros((img_size[1], img_size[0]), dtype=np.uint8)
        bbox = [0, 0, 0, 0]

        if has_polyp:
            # Generate random polyp lesion morphology (ellipse/blob)
            cx = np.random.randint(50, img_size[0] - 50)
            cy = np.random.randint(50, img_size[1] - 50)
            rx = np.random.randint(15, 35)
            ry = np.random.randint(15, 35)

            # Draw mask
            pil_mask = Image.new('L', img_size, 0)
            draw_mask = ImageDraw.Draw(pil_mask)
            draw_mask.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)
            mask = np.array(pil_mask)

            # Draw polyp on base image (slightly lighter, raised texture with highlight)
            polyp_layer = Image.fromarray(base)
            draw_img = ImageDraw.Draw(polyp_layer)
            draw_img.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=(220, 110, 90))
            draw_img.ellipse([cx - rx + 3, cy - ry + 3, cx + rx - 3, cy + ry - 3], fill=(235, 130, 105))
            # Highlight spot
            draw_img.ellipse([cx - 5, cy - 5, cx + 2, cy + 2], fill=(255, 230, 220))
            
            base = np.array(polyp_layer)
            # Smooth blur at polyp borders
            base = cv2.GaussianBlur(base, (3, 3), 0)

            # Calculate Bounding Box [ymin, xmin, ymax, xmax] normalized or pixel [ymin, xmin, ymax, xmax]
            ymin, xmin = max(0, cy - ry), max(0, cx - rx)
            ymax, xmax = min(img_size[1], cy + ry), min(img_size[0], cx + rx)
            bbox = [ymin, xmin, ymax, xmax]

        # Save Image & Mask
        Image.fromarray(base).save(img_path)
        Image.fromarray(mask).save(mask_path)

        label = "polyp" if has_polyp else "normal"
        area = int(np.sum(mask > 0))
        relative_area = round(float(area) / (img_size[0] * img_size[1]), 4)

        metadata.append({
            "image_id": f"polyp_{i:04d}",
            "image_path": os.path.join("images", filename),
            "mask_path": os.path.join("masks", mask_filename),
            "label": label,
            "has_polyp": 1 if has_polyp else 0,
            "ymin": bbox[0],
            "xmin": bbox[1],
            "ymax": bbox[2],
            "xmax": bbox[3],
            "area_px": area,
            "relative_area": relative_area,
            "split": "train" if i < 70 else ("val" if i < 85 else "test")
        })

        # VQA entry
        if has_polyp:
            question = "Does this endoscopic image show a polyp lesion, and what are its spatial characteristics?"
            answer = f"The endoscopic image shows a polyp lesion present at region [{bbox[1]}, {bbox[0]}, {bbox[3]}, {bbox[2]}] occupying {relative_area*100:.1f}% of the field of view. Biopsy/resection recommended."
        else:
            question = "Is there any mucosal abnormality or polyp lesion detected in this frame?"
            answer = "No polyp lesion or mucosal abnormality detected. Normal endoscopic appearance."

        vqa_data.append({
            "image_id": f"polyp_{i:04d}",
            "image_path": os.path.join("images", filename),
            "question": question,
            "answer": answer,
            "label": label
        })

    # Save CSV and JSON
    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join(DATA_DIR, "dataset.csv"), index=False)

    with open(os.path.join(DATA_DIR, "vqa_data.json"), "w") as f:
        json.dump(vqa_data, f, indent=2)

    print(f"[OK] Generated dataset successfully with {num_samples} samples.")
    print(f"[PATH] Images: {IMAGES_DIR}")
    print(f"[PATH] Masks: {MASKS_DIR}")
    print(f"[PATH] Metadata CSV: {os.path.join(DATA_DIR, 'dataset.csv')}")

if __name__ == "__main__":
    generate_synthetic_polyp_dataset()
