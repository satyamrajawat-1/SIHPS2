"""
ASTRA — Temporal Model Evaluator

Evaluates: next-action prediction, step candidate accuracy, anomaly score ROC.
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List
from ml.evaluation.metrics import accuracy, precision_recall_f1, compute_ece

logger = logging.getLogger(__name__)

def evaluate_temporal(
    gt_actions: List[str],
    pred_actions: List[str],
    pred_action_confidences: List[float],
    gt_next_actions: List[str] = None,
    pred_next_actions: List[str] = None,
    gt_steps: List[str] = None,
    pred_steps: List[str] = None,
    pred_step_confidences: List[float] = None,
    pred_anomaly_scores: List[float] = None,
    gt_anomaly_labels: List[bool] = None,
) -> Dict[str, Any]:
    """
    Evaluate temporal model predictions.
    """
    results = {}

    # Current action accuracy
    if gt_actions and pred_actions:
        results["action_accuracy"] = round(accuracy(gt_actions, pred_actions), 4)
        results["action_prf"] = precision_recall_f1(gt_actions, pred_actions)

        # Confidence calibration
        correct = [g == p for g, p in zip(gt_actions, pred_actions)]
        results["action_calibration"] = compute_ece(pred_action_confidences, correct)
    else:
        results["action_accuracy"] = float("nan")
        results["action_note"] = "insufficient_data"

    # Next action prediction
    if gt_next_actions and pred_next_actions:
        results["next_action_accuracy"] = round(
            accuracy(gt_next_actions, pred_next_actions), 4)
    else:
        results["next_action_accuracy"] = float("nan")
        results["next_action_note"] = "insufficient_data"

    # Step candidate accuracy
    if gt_steps and pred_steps:
        results["step_accuracy"] = round(accuracy(gt_steps, pred_steps), 4)
        results["step_prf"] = precision_recall_f1(gt_steps, pred_steps)

        if pred_step_confidences:
            correct_steps = [g == p for g, p in zip(gt_steps, pred_steps)]
            results["step_calibration"] = compute_ece(pred_step_confidences, correct_steps)
    else:
        results["step_accuracy"] = float("nan")
        results["step_note"] = "insufficient_data"

    # Anomaly detection ROC (simplified)
    if pred_anomaly_scores and gt_anomaly_labels:
        thresholds = [0.3, 0.5, 0.7]
        results["anomaly_detection"] = {}
        for t in thresholds:
            pred_labels = [s >= t for s in pred_anomaly_scores]
            prf = precision_recall_f1(gt_anomaly_labels, pred_labels, positive_label=True)
            results["anomaly_detection"][f"threshold_{t}"] = prf
    else:
        results["anomaly_detection"] = {"note": "insufficient_data"}

    return results
