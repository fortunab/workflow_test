import os

class StructuredMedicalTokenizer:
    """
    Deterministic rule engine that converts outputs from visual perception workers (W1, W2, W3)
    into XML-like structured semantic tokens as described in Equations (1)-(2) & Section 3 of the paper.
    """

    @staticmethod
    def encode_classification_token(cls_out):
        """
        Token Format: <CLS:label:confidence:level>
        Example: <CLS:polyp:0.92:high>
        """
        label = cls_out.get("label", "unknown").lower()
        conf = cls_out.get("confidence", 0.0)
        level = cls_out.get("confidence_level", "low").lower()
        return f"<CLS:{label}:{conf:.2f}:{level}>"

    @staticmethod
    def encode_detection_token(det_out):
        """
        Token Format: <DET:num_instances:avg_conf:ymin,xmin,ymax,xmax>
        Example: <DET:1:0.88:10,10,40,40> or <DET:0:0.00:none>
        """
        num_inst = det_out.get("num_instances", 0)
        avg_conf = det_out.get("avg_confidence", 0.0)
        bbox = det_out.get("bounding_box", [0, 0, 0, 0])

        if num_inst > 0 and bbox != [0, 0, 0, 0]:
            coords_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
        else:
            coords_str = "none"

        return f"<DET:{num_inst}:{avg_conf:.2f}:{coords_str}>"

    @staticmethod
    def encode_segmentation_token(seg_out):
        """
        Token Format: <SEG:has_mask:relative_area>
        Example: <SEG:true:0.0898>
        """
        has_mask = str(seg_out.get("has_mask", False)).lower()
        rel_area = seg_out.get("relative_area", 0.0)
        return f"<SEG:{has_mask}:{rel_area:.4f}>"

    @staticmethod
    def compute_uncertainty_token(cls_out, det_out, seg_out):
        """
        Meta-level uncertainty token capturing confidence and consistency of overall prediction.
        Token Format: <UNCERTAINTY:level:signal_type>
        """
        cls_label = cls_out.get("label", "").lower()
        cls_conf = cls_out.get("confidence", 0.0)
        det_instances = det_out.get("num_instances", 0)
        seg_mask = seg_out.get("has_mask", False)

        # Check congruence across stages
        is_polyp_cls = (cls_label == "polyp")
        
        if is_polyp_cls and det_instances > 0 and seg_mask and cls_conf >= 0.75:
            unc_level = "low"
            signal_type = "congruent_signals"
        elif not is_polyp_cls and det_instances == 0 and not seg_mask:
            unc_level = "low"
            signal_type = "congruent_normal"
        elif is_polyp_cls != seg_mask or (is_polyp_cls and det_instances == 0):
            unc_level = "high"
            signal_type = "conflicting_signals"
        else:
            unc_level = "medium"
            signal_type = "moderate_signal"

        return f"<UNCERTAINTY:{unc_level}:{signal_type}>"

    @classmethod
    def encode_pipeline_tokens(cls, perception_results):
        """
        Encodes full dictionary of perception stage outputs into token dictionary and string.
        """
        cls_out = perception_results.get("classification", {})
        det_out = perception_results.get("detection", {})
        seg_out = perception_results.get("segmentation", {})

        tau_cls = cls.encode_classification_token(cls_out)
        tau_det = cls.encode_detection_token(det_out)
        tau_seg = cls.encode_segmentation_token(seg_out)
        tau_unc = cls.compute_uncertainty_token(cls_out, det_out, seg_out)

        tokens_list = [tau_cls, tau_det, tau_seg, tau_unc]
        tokens_str = " ".join(tokens_list)

        return {
            "tau_cls": tau_cls,
            "tau_det": tau_det,
            "tau_seg": tau_seg,
            "tau_unc": tau_unc,
            "tokens_list": tokens_list,
            "tokens_str": tokens_str
        }

    @classmethod
    def format_vlm_prompt(cls, question, perception_results, image_id=None):
        """
        Formats structured prompt incorporating intermediate visual tokens according to Equation (2).
        """
        token_info = cls.encode_pipeline_tokens(perception_results)
        tokens_str = token_info["tokens_str"]

        header = f"[IMAGE_ID: {image_id}]\n" if image_id else ""
        prompt = (
            f"{header}"
            f"Context Structured Tokens: {tokens_str}\n"
            f"Question: {question}\n"
            f"Provide clinical interpretation and diagnostic decision based on the structured evidence."
        )
        return prompt
