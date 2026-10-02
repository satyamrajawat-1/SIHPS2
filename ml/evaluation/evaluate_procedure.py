"""
ASTRA — Procedure Validation Evaluator  (PRIMARY ASTRA METRIC)

Evaluates the complete experiment procedure recognition:
  - Sequence accuracy
  - Step precision / recall / F1
  - False VALID rate
  - UNCERTAIN rate
  - Skipped-step detection
  - Out-of-order detection
  - Final experiment completion correctness
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional
from ml.evaluation.metrics import (
    compute_sequence_accuracy, compute_step_metrics,
    compute_anomaly_detection_metrics, compute_ece,
)

logger = logging.getLogger(__name__)


def evaluate_procedure(
    gt_step_sequence: List[str],
    pred_step_sequence: List[str],
    gt_step_statuses: Dict[str, str],
    pred_step_statuses: Dict[str, str],
    all_step_ids: List[str],
    gt_anomalies: List[Dict] = None,
    pred_anomalies: List[Dict] = None,
    gt_experiment_complete: bool = True,
    pred_experiment_complete: bool = False,
    step_confidences: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    PRIMARY ASTRA EVALUATION.

    Returns comprehensive procedure validation metrics.
    """
    results = {"metric_type": "PROCEDURE_VALIDATION_PERFORMANCE"}

    # 1. Sequence accuracy
    results["sequence"] = compute_sequence_accuracy(gt_step_sequence, pred_step_sequence)

    # 2. Step-level metrics
    results["step_validation"] = compute_step_metrics(
        gt_step_statuses, pred_step_statuses, all_step_ids
    )

    # 3. Anomaly / out-of-order / skipped-step detection
    gt_anom = gt_anomalies or []
    pred_anom = pred_anomalies or []

    # All anomalies
    results["anomaly_detection"] = compute_anomaly_detection_metrics(gt_anom, pred_anom)

    # Separate by type
    for anom_type in ["OUT_OF_ORDER", "SKIPPED_STEP", "TIMEOUT"]:
        gt_typed = [a for a in gt_anom if a.get("type") == anom_type]
        pred_typed = [a for a in pred_anom if a.get("type") == anom_type]
        results[f"{anom_type.lower()}_detection"] = compute_anomaly_detection_metrics(
            gt_typed, pred_typed
        )

    # 4. Experiment completion correctness
    results["experiment_completion"] = {
        "gt_complete": gt_experiment_complete,
        "pred_complete": pred_experiment_complete,
        "correct": gt_experiment_complete == pred_experiment_complete,
    }

    # 5. Confidence calibration for step validation
    if step_confidences:
        confs = []
        correct_flags = []
        for sid in all_step_ids:
            if sid in step_confidences and sid in gt_step_statuses:
                confs.append(step_confidences[sid])
                correct_flags.append(
                    gt_step_statuses[sid] == pred_step_statuses.get(sid, "?")
                )
        if confs:
            results["step_calibration"] = compute_ece(confs, correct_flags)

    # 6. Summary score (weighted composite)
    seq_acc = results["sequence"].get("sequence_accuracy", 0)
    step_f1 = results["step_validation"].get("step_f1", 0)
    if isinstance(step_f1, float) and __import__("math").isnan(step_f1):
        step_f1 = 0
    false_valid = results["step_validation"].get("false_valid_rate", 0)
    completion_correct = 1.0 if results["experiment_completion"]["correct"] else 0.0

    composite = (
        0.30 * seq_acc +
        0.30 * step_f1 +
        0.20 * (1.0 - false_valid) +
        0.20 * completion_correct
    )
    results["composite_score"] = round(composite, 4)

    return results


def compute_failure_cases(
    all_step_ids: List[str],
    gt_step_statuses: Dict[str, str],
    pred_step_statuses: Dict[str, str],
    step_evidence: Optional[Dict[str, Dict]] = None,
    step_confidences: Optional[Dict[str, float]] = None,
) -> List[Dict]:
    """
    For every incorrect decision, compile a failure case record.

    Returns list of failure case dicts with:
      step_id, expected_status, predicted_status, confidence,
      evidence_breakdown, reason
    """
    failures = []

    for sid in all_step_ids:
        gt = gt_step_statuses.get(sid, "NOT_EVALUATED")
        pred = pred_step_statuses.get(sid, "NOT_EVALUATED")

        if gt != pred:
            failure = {
                "step_id": sid,
                "expected_status": gt,
                "predicted_status": pred,
                "confidence": step_confidences.get(sid, None) if step_confidences else None,
                "evidence": step_evidence.get(sid, {}) if step_evidence else {},
            }

            # Determine reason
            if gt == "VALID" and pred == "UNCERTAIN":
                failure["reason"] = "Insufficient evidence to confirm valid step"
                failure["category"] = "false_negative"
            elif gt == "VALID" and pred in ("NOT_STARTED", "NOT_EVALUATED"):
                failure["reason"] = "Step not detected at all"
                failure["category"] = "missed_step"
            elif gt != "VALID" and pred == "VALID":
                failure["reason"] = "Step incorrectly marked as valid"
                failure["category"] = "false_positive"
            elif gt == "SKIPPED" and pred != "SKIPPED":
                failure["reason"] = "Skipped step not detected"
                failure["category"] = "missed_skip"
            else:
                failure["reason"] = f"Status mismatch: expected {gt}, got {pred}"
                failure["category"] = "mismatch"

            failures.append(failure)

    return failures
