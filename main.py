import os
import argparse
import sys

from prepare_dataset import generate_synthetic_polyp_dataset, DATA_DIR
from src.train_classification import train_classification_model
from src.train_detection import train_detection_model
from src.train_segmentation import train_segmentation_model
from src.orchestrator import WorkflowOrchestrator
from evaluate_pipeline import evaluate_framework

def main():
    parser = argparse.ArgumentParser(
        description="Workflow-Centric Medical AI Pipeline (fast.ai)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--prepare-data", action="store_true",
                        help="Generate or prepare the dataset.\n"
                             "Use --dataset to select source (default: synthetic).")
    parser.add_argument("--dataset",
                        choices=["synthetic", "kvasir", "polypgen"],
                        default="synthetic",
                        help=(
                            "Dataset to use:\n"
                            "  synthetic  - Generate synthetic endoscopy samples (default)\n"
                            "  kvasir     - Parse real Kvasir-SEG dataset\n"
                            "  polypgen   - Parse real PolypGen multi-center dataset"
                        ))
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to the root folder of the downloaded real dataset\n"
                             "(required when --dataset is 'kvasir' or 'polypgen').")
    parser.add_argument("--train", action="store_true",
                        help="Train fast.ai perceptual models (cls, det, seg).")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs per stage (default: 20).")
    parser.add_argument("--eval", action="store_true",
                        help="Run end-to-end benchmark evaluation.")
    parser.add_argument("--predict", type=str,
                        help="Path to input medical image for inference.")
    parser.add_argument("--mode", choices=["full", "selective"], default="selective",
                        help="Orchestration mode (default: selective).")
    parser.add_argument("--question", type=str,
                        default="Does this endoscopic image show a polyp lesion?",
                        help="VQA question prompt for inference.")

    args = parser.parse_args()

    # Resolve the actual data directory to use for training/eval
    resolved_data_dir = DATA_DIR   # default: synthetic output dir

    if args.prepare_data:
        if args.dataset == "synthetic":
            print("[DATA] Generating synthetic endoscopy dataset (100 samples)...")
            generate_synthetic_polyp_dataset(num_samples=100)
            print(f"[DATA] Synthetic dataset ready at: {DATA_DIR}")

        elif args.dataset == "kvasir":
            if not args.data_path:
                print("[ERROR] --data-path is required when using --dataset kvasir")
                print("        Example: python main.py --dataset kvasir --data-path ./Kvasir-SEG --prepare-data")
                sys.exit(1)
            from src.kvasir_loader import load_kvasir_seg
            print(f"[DATA] Parsing Kvasir-SEG from: {args.data_path}")
            resolved_data_dir = load_kvasir_seg(
                dataset_path=args.data_path,
                output_data_dir=os.path.join(args.data_path, "processed")
            )

        elif args.dataset == "polypgen":
            if not args.data_path:
                print("[ERROR] --data-path is required when using --dataset polypgen")
                print("        Example: python main.py --dataset polypgen --data-path ./PolypGen --prepare-data")
                sys.exit(1)
            from src.polypgen_loader import load_polypgen
            print(f"[DATA] Parsing PolypGen from: {args.data_path}")
            resolved_data_dir = load_polypgen(
                dataset_path=args.data_path,
                output_data_dir=os.path.join(args.data_path, "processed")
            )

    # If using a real dataset for training without --prepare-data, resolve path
    if args.dataset in ("kvasir", "polypgen") and args.data_path:
        resolved_data_dir = os.path.join(args.data_path, "processed")

    if args.train:
        print(f"\n[TRAIN] Training Stage 1 Classification (f_cls) for {args.epochs} epochs...")
        train_classification_model(resolved_data_dir, epochs=args.epochs)
        print(f"\n[TRAIN] Training Stage 2 Detection (f_det) for {args.epochs} epochs...")
        train_detection_model(resolved_data_dir, epochs=args.epochs)
        print(f"\n[TRAIN] Training Stage 3 Segmentation (f_seg) for {args.epochs} epochs...")
        train_segmentation_model(resolved_data_dir, epochs=args.epochs)
        print("\n[TRAIN] All perceptual models trained successfully!")

    if args.eval:
        evaluate_framework(data_dir=resolved_data_dir)

    if args.predict:
        if not os.path.exists(args.predict):
            print(f"[ERROR] Image path not found: {args.predict}")
            sys.exit(1)

        orchestrator = WorkflowOrchestrator(data_dir=resolved_data_dir)
        if args.mode == "selective":
            res = orchestrator.run_selective_pipeline(args.predict, question=args.question)
        else:
            res = orchestrator.run_full_pipeline(args.predict, question=args.question)

        print("\n==================================================")
        print(f"  INFERENCE RESULT ({res['mode'].upper()} MODE)")
        print("==================================================")
        print(f"Executed Stages:  {res['executed_stages']}")
        print(f"Skipped Stages:   {res['skipped_stages']}")
        print(f"Latency:          {res['latency_ms']} ms ({res['gpu_load_percent']}% GPU Load)")
        print("--------------------------------------------------")
        print("STRUCTURED TOKENS:")
        print(f"  {res['tokens_info']['tokens_str']}")
        print("--------------------------------------------------")
        print(res['vlm_output']['clinical_report'])
        print("==================================================")

    if not any([args.prepare_data, args.train, args.eval, args.predict]):
        parser.print_help()

if __name__ == "__main__":
    main()
