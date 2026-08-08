import os
from fastai.learner import load_learner
from src.train_classification import predict_classification, train_classification_model
from src.train_detection import predict_detection, train_detection_model
from src.train_segmentation import predict_segmentation, train_segmentation_model

class PerceptualWorkerSuite:
    """
    Unified Manager loading the three trained fast.ai perceptual workers:
    - Worker W1 (f_cls): Classification
    - Worker W2 (f_det): Detection Bounding Box
    - Worker W3 (f_seg): Segmentation Mask
    """
    def __init__(self, models_dir=None, data_dir=None, auto_train=True):
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        self.models_dir = models_dir
        self.data_dir = data_dir

        self.cls_path = os.path.join(models_dir, "classifier.pkl")
        self.det_path = os.path.join(models_dir, "detector.pkl")
        self.seg_path = os.path.join(models_dir, "segmenter.pkl")

        # Train if missing
        if auto_train and data_dir and not (os.path.exists(self.cls_path) and os.path.exists(self.det_path) and os.path.exists(self.seg_path)):
            print("[INFO] Model checkpoints missing. Auto-training perceptual models...")
            train_classification_model(data_dir, epochs=2, save_path=self.cls_path)
            train_detection_model(data_dir, epochs=2, save_path=self.det_path)
            train_segmentation_model(data_dir, epochs=2, save_path=self.seg_path)

        self.cls_learn = load_learner(self.cls_path) if os.path.exists(self.cls_path) else None
        self.det_learn = load_learner(self.det_path) if os.path.exists(self.det_path) else None
        self.seg_learn = load_learner(self.seg_path) if os.path.exists(self.seg_path) else None

    def run_stage1_classification(self, img_path):
        """Worker W1: Classification (f_cls)"""
        if self.cls_learn is None:
            raise RuntimeError("Classification learner is not loaded.")
        return predict_classification(self.cls_learn, img_path)

    def run_stage2_detection(self, img_path):
        """Worker W2: Detection (f_det)"""
        if self.det_learn is None:
            raise RuntimeError("Detection learner is not loaded.")
        return predict_detection(self.det_learn, img_path)

    def run_stage3_segmentation(self, img_path):
        """Worker W3: Segmentation (f_seg)"""
        if self.seg_learn is None:
            raise RuntimeError("Segmentation learner is not loaded.")
        return predict_segmentation(self.seg_learn, img_path)

    def run_full_perception_pass(self, img_path):
        """Executes all 3 perceptual stages sequentially on input image."""
        cls_out = self.run_stage1_classification(img_path)
        det_out = self.run_stage2_detection(img_path)
        seg_out = self.run_stage3_segmentation(img_path)

        return {
            "classification": cls_out,
            "detection": det_out,
            "segmentation": seg_out
        }
