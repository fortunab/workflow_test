import os
import sys
import torch

# Add root directory to sys.path
sys.path.insert(0, os.path.dirname(__file__))

from prepare_dataset import generate_synthetic_polyp_dataset, DATA_DIR
from src.dataloaders import (
    get_classification_dataloaders,
    get_detection_dataloaders,
    get_segmentation_dataloaders
)

def run_tests():
    print("--------------------------------------------------")
    print("  Step 1: Testing Data Pipeline & fast.ai DataLoaders")
    print("--------------------------------------------------")

    # 1. Ensure dataset exists
    csv_path = os.path.join(DATA_DIR, "dataset.csv")
    if not os.path.exists(csv_path):
        print("Dataset missing, generating synthetic dataset...")
        generate_synthetic_polyp_dataset(num_samples=100)

    # 2. Test Classification DataLoaders
    print("\n[1/3] Testing Classification DataLoaders (Stage 1)...")
    cls_dls = get_classification_dataloaders(DATA_DIR, bs=8, img_size=224)
    x_cls, y_cls = cls_dls.one_batch()
    print(f"   Batch Image Shape: {x_cls.shape}")
    print(f"   Batch Label Shape: {y_cls.shape}")
    print(f"   Vocabulary (Classes): {cls_dls.vocab}")
    assert x_cls.shape == (8, 3, 224, 224), f"Unexpected shape {x_cls.shape}"
    print("   [OK] Classification DataLoader verified successfully.")

    # 3. Test Detection DataLoaders
    print("\n[2/3] Testing Detection Bounding Box DataLoaders (Stage 2)...")
    det_dls = get_detection_dataloaders(DATA_DIR, bs=8, img_size=224)
    batch_det = det_dls.one_batch()
    x_det, (bboxes, bbox_lbls) = batch_det[0], batch_det[1:]
    print(f"   Batch Image Shape: {x_det.shape}")
    print(f"   BBoxes Tensor Shape: {bboxes.shape}")
    print(f"   BBox Labels: {bbox_lbls}")
    assert x_det.shape[0] == 8 and x_det.shape[1] == 3, f"Unexpected shape {x_det.shape}"
    print("   [OK] Detection DataLoader verified successfully.")

    # 4. Test Segmentation DataLoaders
    print("\n[3/3] Testing Segmentation Mask DataLoaders (Stage 3)...")
    seg_dls = get_segmentation_dataloaders(DATA_DIR, bs=8, img_size=224)
    x_seg, y_seg = seg_dls.one_batch()
    print(f"   Batch Image Shape: {x_seg.shape}")
    print(f"   Batch Mask Shape: {y_seg.shape}")
    print(f"   Mask unique values: {torch.unique(y_seg)}")
    assert x_seg.shape == (8, 3, 224, 224), f"Unexpected shape {x_seg.shape}"
    assert y_seg.shape == (8, 224, 224), f"Unexpected shape {y_seg.shape}"
    print("   [OK] Segmentation DataLoader verified successfully.")

    print("\n[SUCCESS] STEP 1 VERIFICATION COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    run_tests()
