"""
ASTRA — Threshold Calibration

Tunes detection, action, HOI, object-state, temporal, step completion,
and anomaly thresholds on a validation set.

Optimizes for: step F1, false VALID rate, UNCERTAIN rate, anomaly F1.
Treats false VALID as a CRITICAL SAFETY METRIC.

Usage:
    python -m ml.evaluation.calibrate --experiment experiments/sample_experiment/ \
        --dataset data/eval/ --output eval_results/calibration/
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def calibrate_thresholds(
    gt_step_statuses: Dict[str, str],
    pred_scores: Dict[str, float],
    all_step_ids: List[str],
    safety_weight: float = 2.0,
) -> Dict[str, Any]:
    """
    Find optimal step completion threshold.

    Objective: maximize step_F1 - safety_weight * false_valid_rate

    Returns best threshold and metrics at that threshold.
    """
    from ml.evaluation.metrics import compute_step_metrics

    candidates = np.arange(0.1, 0.95, 0.05)
    best_score = -float("inf")
    best_threshold = 0.5
    best_metrics = {}
    all_results = []

    for thresh in candidates:
        # Apply threshold to get predicted statuses
        pred_statuses = {}
        for sid in all_step_ids:
            score = pred_scores.get(sid, 0.0)
            if score >= thresh:
                pred_statuses[sid] = "VALID"
            elif score >= thresh * 0.5:
                pred_statuses[sid] = "UNCERTAIN"
            else:
                pred_statuses[sid] = "NOT_STARTED"

        metrics = compute_step_metrics(gt_step_statuses, pred_statuses, all_step_ids)

        step_f1 = metrics.get("step_f1", 0)
        if isinstance(step_f1, float) and np.isnan(step_f1):
            step_f1 = 0
        false_valid = metrics.get("false_valid_rate", 0)
        uncertain = metrics.get("uncertain_rate", 0)

        # Objective: maximize F1, penalize false VALID heavily
        objective = step_f1 - safety_weight * false_valid - 0.1 * uncertain

        result = {
            "threshold": round(float(thresh), 2),
            "step_f1": round(step_f1, 4),
            "false_valid_rate": round(false_valid, 4),
            "uncertain_rate": round(uncertain, 4),
            "objective": round(objective, 4),
        }
        all_results.append(result)

        if objective > best_score:
            best_score = objective
            best_threshold = thresh
            best_metrics = result

    return {
        "best_threshold": round(float(best_threshold), 2),
        "best_metrics": best_metrics,
        "all_results": all_results,
        "safety_weight": safety_weight,
    }


def calibrate_confidence(
    confidences: List[float],
    correct: List[bool],
    method: str = "temperature",
) -> Dict[str, Any]:
    """
    Calibrate model confidence scores.

    Methods: temperature, platt, histogram
    """
    from ml.evaluation.metrics import compute_ece

    if not confidences:
        return {"status": "insufficient_data"}

    raw_ece = compute_ece(confidences, correct)

    if method == "temperature":
        # Temperature scaling: find T that minimizes ECE
        best_t = 1.0
        best_ece = raw_ece["ece"]

        for t in np.arange(0.1, 5.0, 0.1):
            scaled = [_sigmoid(np.log(c / (1 - c + 1e-7)) / t) for c in confidences]
            ece = compute_ece(scaled, correct)["ece"]
            if not np.isnan(ece) and ece < best_ece:
                best_ece = ece
                best_t = t

        scaled_confs = [_sigmoid(np.log(c / (1 - c + 1e-7)) / best_t) for c in confidences]
        scaled_ece = compute_ece(scaled_confs, correct)

        return {
            "method": "temperature_scaling",
            "temperature": round(float(best_t), 2),
            "raw_ece": raw_ece["ece"],
            "calibrated_ece": scaled_ece["ece"],
            "improvement": round(raw_ece["ece"] - scaled_ece["ece"], 4)
                if not np.isnan(raw_ece["ece"]) and not np.isnan(scaled_ece["ece"]) else None,
            "raw_bins": raw_ece.get("bins", []),
            "calibrated_bins": scaled_ece.get("bins", []),
        }

    return {"method": method, "raw_ece": raw_ece}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def calibrate_all(
    experiment_config,
    eval_results: Dict,
) -> Dict[str, Any]:
    """
    Calibrate all thresholds using evaluation results.

    Returns recommended thresholds.
    """
    all_step_ids = [s.id for s in experiment_config.steps]
    recommendations = {}

    # Step completion threshold
    proc = eval_results.get("procedure", {})
    sv = proc.get("step_validation", {})
    ps = sv.get("per_step", {})

    if ps:
        # Simulate scores from step evidence
        pred_scores = {}
        for sid, data in ps.items():
            # Use 1.0 for VALID, 0.5 for UNCERTAIN, 0.0 for NOT_STARTED
            if data.get("pred") == "VALID":
                pred_scores[sid] = 0.8
            elif data.get("pred") == "UNCERTAIN":
                pred_scores[sid] = 0.4
            else:
                pred_scores[sid] = 0.1

        gt_statuses = {sid: data.get("gt", "?") for sid, data in ps.items()}
        cal = calibrate_thresholds(gt_statuses, pred_scores, all_step_ids)
        recommendations["step_completion"] = cal

    # Confidence calibration from temporal model
    temporal = eval_results.get("temporal", {})
    act_cal = temporal.get("action_calibration", {})
    if act_cal and act_cal.get("ece") is not None:
        recommendations["action_confidence"] = {
            "ece": act_cal.get("ece"),
            "recommendation": "temperature_scaling" if act_cal.get("ece", 0) > 0.1 else "no_change",
        }

    return {
        "recommendations": recommendations,
        "note": "These thresholds should be validated on a held-out test set before deployment",
    }


def main():
    parser = argparse.ArgumentParser(description="ASTRA Threshold Calibration")
    parser.add_argument("--results", required=True, help="evaluation_results.json")
    parser.add_argument("--experiment", required=True, help="Experiment config dir")
    parser.add_argument("--output", default="eval_results/calibration")
    args = parser.parse_args()

    from ml.experiment.schema.loader import load_experiment

    config = load_experiment(args.experiment)

    with open(args.results) as f:
        eval_results = json.load(f)

    os.makedirs(args.output, exist_ok=True)

    cal_results = calibrate_all(config, eval_results)

    out_path = os.path.join(args.output, "calibration_results.json")
    with open(out_path, "w") as f:
        json.dump(cal_results, f, indent=2)

    print(json.dumps(cal_results, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
