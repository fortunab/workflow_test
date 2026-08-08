import os
import torch
import torch.nn.functional as F
from fastai.vision.all import vision_learner, resnet18, accuracy, PILImage
from src.dataloaders import get_classification_dataloaders

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def train_classification_model(data_dir, epochs=20, lr=1e-3, save_path=None):
    """
    Fine-tunes Stage 1 Classification Model (f_cls) using fast.ai vision_learner.
    """
    if save_path is None:
        save_path = os.path.join(MODELS_DIR, "classifier.pkl")

    print(f"\n--- [Stage 1] Training Classification Model (f_cls) ---")
    dls = get_classification_dataloaders(data_dir, bs=16, img_size=224)
    learn = vision_learner(dls, resnet18, metrics=accuracy)
    learn.fine_tune(epochs, base_lr=lr)

    # Save model
    learn.export(save_path)
    print(f"[OK] Classification model saved to {save_path}")
    return learn

def predict_classification(learn, img_path):
    """
    Runs classification inference on an input image.
    Returns label, raw confidence score, and discretized confidence level ('high', 'medium', 'low').
    """
    img = PILImage.create(img_path)
    pred_class, pred_idx, probs = learn.predict(img)
    conf_score = float(probs[pred_idx])

    if conf_score >= 0.85:
        disc_level = "high"
    elif conf_score >= 0.60:
        disc_level = "medium"
    else:
        disc_level = "low"

    return {
        "label": str(pred_class),
        "confidence": round(conf_score, 4),
        "confidence_level": disc_level,
        "probs": {cls: round(float(p), 4) for cls, p in zip(learn.dls.vocab, probs)}
    }

if __name__ == "__main__":
    from prepare_dataset import DATA_DIR
    train_classification_model(DATA_DIR, epochs=2)
