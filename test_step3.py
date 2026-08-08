import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from src.tokenizer import StructuredMedicalTokenizer

def run_step3_tests():
    print("--------------------------------------------------")
    print("  Step 3: Testing Structured Tokenizer Engine")
    print("--------------------------------------------------")

    # 1. Test Congruent Positive Polyp Tokens
    print("\n[1/3] Testing Token Encoding for Positive Polyp Case...")
    polyp_results = {
        "classification": {"label": "polyp", "confidence": 0.92, "confidence_level": "high"},
        "detection": {"num_instances": 1, "avg_confidence": 0.88, "bounding_box": [10, 10, 40, 40]},
        "segmentation": {"has_mask": True, "relative_area": 0.0898}
    }

    polyp_tokens = StructuredMedicalTokenizer.encode_pipeline_tokens(polyp_results)
    print(f"   [CLS TOKEN]: {polyp_tokens['tau_cls']}")
    print(f"   [DET TOKEN]: {polyp_tokens['tau_det']}")
    print(f"   [SEG TOKEN]: {polyp_tokens['tau_seg']}")
    print(f"   [UNC TOKEN]: {polyp_tokens['tau_unc']}")
    print(f"   [COMBINED TOKEN STR]: {polyp_tokens['tokens_str']}")

    assert polyp_tokens['tau_cls'] == "<CLS:polyp:0.92:high>"
    assert polyp_tokens['tau_det'] == "<DET:1:0.88:10,10,40,40>"
    assert polyp_tokens['tau_seg'] == "<SEG:true:0.0898>"
    assert polyp_tokens['tau_unc'] == "<UNCERTAINTY:low:congruent_signals>"

    # 2. Test Normal Case Tokens
    print("\n[2/3] Testing Token Encoding for Normal (No Polyp) Case...")
    normal_results = {
        "classification": {"label": "normal", "confidence": 0.95, "confidence_level": "high"},
        "detection": {"num_instances": 0, "avg_confidence": 0.0, "bounding_box": [0, 0, 0, 0]},
        "segmentation": {"has_mask": False, "relative_area": 0.0}
    }

    normal_tokens = StructuredMedicalTokenizer.encode_pipeline_tokens(normal_results)
    print(f"   [CLS TOKEN]: {normal_tokens['tau_cls']}")
    print(f"   [DET TOKEN]: {normal_tokens['tau_det']}")
    print(f"   [SEG TOKEN]: {normal_tokens['tau_seg']}")
    print(f"   [UNC TOKEN]: {normal_tokens['tau_unc']}")

    assert normal_tokens['tau_cls'] == "<CLS:normal:0.95:high>"
    assert normal_tokens['tau_unc'] == "<UNCERTAINTY:low:congruent_normal>"

    # 3. Test VLM Prompt Formatting
    print("\n[3/3] Testing VLM Prompt Generation...")
    question = "Does this endoscopic image show a polyp lesion?"
    vlm_prompt = StructuredMedicalTokenizer.format_vlm_prompt(question, polyp_results, image_id="polyp_0001")
    print(f"\n--- Formatted VLM Prompt ---\n{vlm_prompt}\n----------------------------")

    assert "<CLS:polyp:0.92:high>" in vlm_prompt
    assert question in vlm_prompt

    print("\n[SUCCESS] STEP 3 STRUCTURED TOKENIZER VERIFICATION COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    run_step3_tests()
