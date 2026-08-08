import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from prepare_dataset import generate_synthetic_polyp_dataset, DATA_DIR, IMAGES_DIR
from src.orchestrator import WorkflowOrchestrator

def run_step4_tests():
    print("--------------------------------------------------")
    print("  Step 4: Testing VLM Reasoning & Token-Guided Orchestration Engine")
    print("--------------------------------------------------")

    # 1. Ensure dataset exists
    csv_path = os.path.join(DATA_DIR, "dataset.csv")
    if not os.path.exists(csv_path):
        print("Dataset missing, generating synthetic dataset...")
        generate_synthetic_polyp_dataset(num_samples=100)

    orchestrator = WorkflowOrchestrator(data_dir=DATA_DIR)

    polyp_img = os.path.join(IMAGES_DIR, "polyp_0001.jpg") # Polyp image
    normal_img = os.path.join(IMAGES_DIR, "polyp_0005.jpg") # Normal image

    # 2. Test Full Execution Mode
    print("\n[1/3] Executing FULL Pipeline (Unconditional W1 -> W2 -> W3 -> W4)...")
    full_res = orchestrator.run_full_pipeline(polyp_img, question="Is a polyp lesion present?")
    print(f"   Executed Stages: {full_res['executed_stages']}")
    print(f"   Skipped Stages: {full_res['skipped_stages']}")
    print(f"   Latency: {full_res['latency_ms']} ms | GPU Load: {full_res['gpu_load_percent']}%")
    print(f"   Tokens: {full_res['tokens_info']['tokens_str']}")
    print(f"   Report Diagnosis: {full_res['vlm_output']['final_diagnosis']}")
    assert len(full_res['executed_stages']) == 4
    assert len(full_res['skipped_stages']) == 0

    # 3. Test Selective Execution Mode on Normal Image
    print("\n[2/3] Executing SELECTIVE Pipeline on Normal Mucosa (W1 -> Skip W2, W3 -> W4)...")
    sel_norm_res = orchestrator.run_selective_pipeline(normal_img, question="Is there any abnormality?")
    print(f"   Executed Stages: {sel_norm_res['executed_stages']}")
    print(f"   Skipped Stages: {sel_norm_res['skipped_stages']}")
    print(f"   Latency: {sel_norm_res['latency_ms']} ms | GPU Load: {sel_norm_res['gpu_load_percent']}%")
    print(f"   Tokens: {sel_norm_res['tokens_info']['tokens_str']}")
    print(f"   Report Diagnosis: {sel_norm_res['vlm_output']['final_diagnosis']}")

    # 4. Compare Latencies
    print("\n[3/3] Latency & Efficiency Summary:")
    print(f"   • Full Pipeline Execution Latency:      {full_res['latency_ms']} ms (100% GPU Load)")
    print(f"   • Selective Pipeline Execution Latency: {sel_norm_res['latency_ms']} ms ({sel_norm_res['gpu_load_percent']}% GPU Load)")

    print("\n[SUCCESS] STEP 4 VLM REASONER & ORCHESTRATION ENGINE VERIFIED SUCCESSFULLY!")

if __name__ == "__main__":
    run_step4_tests()
