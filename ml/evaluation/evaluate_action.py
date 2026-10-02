"""
ASTRA — Action Recognition Evaluator

Evaluates: per-action accuracy, confusion matrix, temporal segment F1.
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List
from ml.evaluation.metrics import (
    confusion_matrix, precision_recall_f1, accuracy,
    compute_segment_f1, compute_edit_distance, top_k_accuracy,
)
from ml.evaluation.dataset_loader import EvalDataset

logger = logging.getLogger(__name__)

def evaluate_action(
    dataset: EvalDataset,
    frame_predictions: List[Dict] = None,
    segment_predictions: List[Dict] = None,
) -> Dict[str, Any]:
    """
    Evaluate action recognition.

    frame_predictions: list of {"action_id": str, "confidence": float, "all_scores": {}}
    segment_predictions: list of {"label": str, "start": float, "end": float}
    """
    results = {}

    # Frame-level evaluation
    if frame_predictions:
        gt_actions = []
        pred_actions = []
        pred_probs = []

        for sample, pred in zip(dataset.samples, frame_predictions):
            if sample.gt_step and "action_id" in (sample.gt_step or {}):
                gt_actions.append(sample.gt_step["action_id"])
                pred_actions.append(pred.get("action_id", "?"))
                if "all_scores" in pred:
                    pred_probs.append(pred["all_scores"])

        if gt_actions:
            results["frame_accuracy"] = round(accuracy(gt_actions, pred_actions), 4)
            results["frame_prf"] = precision_recall_f1(gt_actions, pred_actions)

            labels = sorted(set(gt_actions) | set(pred_actions))
            cm, cm_labels = confusion_matrix(gt_actions, pred_actions, labels)
            results["confusion_matrix"] = {"matrix": cm.tolist(), "labels": cm_labels}

            if pred_probs:
                results["top3_accuracy"] = round(top_k_accuracy(gt_actions, pred_probs, k=3), 4)
        else:
            results["frame_accuracy"] = float("nan")
            results["status"] = "insufficient_data"

    # Segment-level evaluation
    if segment_predictions and dataset.gt_action_segments:
        for iou_t in [0.25, 0.5, 0.75]:
            key = f"segment_f1@{int(iou_t*100)}"
            results[key] = compute_segment_f1(
                dataset.gt_action_segments, segment_predictions, iou_threshold=iou_t
            )

        # Edit distance between action sequences
        gt_seq = [s["label"] for s in sorted(dataset.gt_action_segments, key=lambda x: x["start"])]
        pred_seq = [s["label"] for s in sorted(segment_predictions, key=lambda x: x["start"])]
        ed = compute_edit_distance(gt_seq, pred_seq)
        max_len = max(len(gt_seq), len(pred_seq))
        results["edit_score"] = round(1.0 - ed / max_len, 4) if max_len > 0 else 1.0
    elif not segment_predictions:
        results["segment_note"] = "no_segment_predictions_provided"

    return results
