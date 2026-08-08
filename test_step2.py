import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from prepare_dataset import generate_synthetic_polyp_dataset, DATA_DIR, IMAGES_DIR
from src.train_classification import train_classification_model
from src.train_detection import train_detection_model
from src.train_segmentation import train_segmentation_model
from src.perceptual_models import PerceptualWorkerSuite

def run_step2_tests():
    print("--------------------------------------------------")
    print("  Step 2: Training & Testing Perceptual Models")
    print("--------------------------------------------------")

    # 1. Ensure dataset exists
    csv_path = os.path.join(DATA_DIR, "dataset.csv")
    if not os.path.exists(csv_path):
        print("Dataset missing, generating synthetic dataset...")
        generate_synthetic_polyp_dataset(num_samples=100)

    # 2. Train Classification Model (f_cls)
    print("\n[1/4] Fine-Tuning Stage 1 Classification Model (f_cls)...")
    cls_learn = train_classification_model(DATA_DIR, epochs=2)

    # 3. Train Detection Model (f_det)
    print("\n[2/4] Fine-Tuning Stage 2 Detection Model (f_det)...")
    det_learn = train_detection_model(DATA_DIR, epochs=2)

    # 4. Train Segmentation Model (f_seg)
    print("\n[3/4] Fine-Tuning Stage 3 Segmentation Model (f_seg)...")
    seg_learn = train_segmentation_model(DATA_DIR, epochs=2)

    # 5. Test Unified Perceptual Suite Inference
    print("\n[4/4] Testing Unified PerceptualWorkerSuite Inference...")
    suite = PerceptualWorkerSuite(data_dir=DATA_DIR, auto_train=False)
    
    sample_img = os.path.join(IMAGES_DIR, "polyp_0000.jpg")
    results = suite.run_full_perception_pass(sample_img)

    print(f"\n   [STAGE 1 f_cls] Label: {results['classification']['label']} | Conf: {results['classification']['confidence']} ({results['classification']['confidence_level']})")
    print(f"   [STAGE 2 f_det] BBox: {results['detection']['bounding_box']} | Count: {results['detection']['num_instances']}")
    print(f"   [STAGE 3 f_seg] Has Mask: {results['segmentation']['has_mask']} | Rel Area: {results['segmentation']['relative_area']}")

    assert "label" in results["classification"]
    assert "bounding_box" in results["detection"]
    assert "relative_area" in results["segmentation"]

    print("\n[SUCCESS] STEP 2 PERCEPTUAL MODELS TRAINING & INFERENCE COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    run_step2_tests()
