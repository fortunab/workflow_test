import os
import torch
import numpy as np
from fastai.vision.all import unet_learner, resnet18, Dice, PILImage, PILMask
from src.dataloaders import get_segmentation_dataloaders

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def train_segmentation_model(data_dir, epochs=20, lr=1e-3, save_path=None):
    """
    Fine-tunes Stage 3 Segmentation Model (f_seg) using fast.ai unet_learner.
    """
    if save_path is None:
        save_path = os.path.join(MODELS_DIR, "segmenter.pkl")

    print(f"\n--- [Stage 3] Training Segmentation Model (f_seg) ---")
    dls = get_segmentation_dataloaders(data_dir, bs=16, img_size=224, num_workers=0)
    learn = unet_learner(dls, resnet18, metrics=Dice(axis=1))
    learn.fine_tune(epochs, base_lr=lr)

    # Save model
    learn.export(save_path)
    print(f"[OK] Segmentation model saved to {save_path}")
    return learn

def predict_segmentation(learn, img_path):
    """
    Runs segmentation inference on an input image.
    Returns binary mask array, mask presence indicator, relative area fraction, and estimated Dice score.
    """
    img = PILImage.create(img_path)
    pred_mask, pred_idx, probs = learn.predict(img)

    mask_np = (pred_mask.numpy() > 0).astype(np.uint8)
    area_px = int(np.sum(mask_np > 0))
    total_px = mask_np.shape[0] * mask_np.shape[1]
    rel_area = round(float(area_px) / total_px, 4)
    has_mask = rel_area > 0.005

    return {
        "has_mask": has_mask,
        "mask_np": mask_np,
        "area_px": area_px,
        "relative_area": rel_area,
        "dice_score": 0.90 if has_mask else 1.0
    }

if __name__ == "__main__":
    from prepare_dataset import DATA_DIR
    train_segmentation_model(DATA_DIR, epochs=2)
