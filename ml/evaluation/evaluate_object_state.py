"""
ASTRA — Object State Evaluator

Evaluates: state classification accuracy, state transition detection.
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List
from ml.evaluation.metrics import precision_recall_f1, accuracy, confusion_matrix
from ml.evaluation.dataset_loader import EvalDataset

logger = logging.getLogger(__name__)

def evaluate_object_state(
    dataset: EvalDataset,
    predictions: List[List[Dict]],
) -> Dict[str, Any]:
    """
    Evaluate object state classification.

    predictions[i] = list of {"object_id": str, "current_state": str, "is_transition": bool, "confidence": float}
    """
    gt_all = [s.gt_object_states for s in dataset.samples]
    if not any(gt_all):
        return {"status": "insufficient_data", "reason": "No ground truth object states"}

    gt_states = []
    pred_states = []
    gt_transitions = []
    pred_transitions = []
    per_object = {}
    n_evaluated = 0

    for gt_list, pred_list in zip(gt_all, predictions):
        if not gt_list:
            continue
        for gt in gt_list:
            oid = gt.get("object_id", "?")
            gt_state = gt.get("current_state", gt.get("state", "?"))
            gt_states.append(gt_state)
            gt_trans = gt.get("is_transition", False)
            gt_transitions.append(gt_trans)

            # Find matching prediction
            matched = None
            for pred in pred_list:
                if pred.get("object_id") == oid:
                    matched = pred
                    break

            if matched:
                pred_state = matched.get("current_state", "?")
                pred_trans = matched.get("is_transition", False)
            else:
                pred_state = "unknown"
                pred_trans = False

            pred_states.append(pred_state)
            pred_transitions.append(pred_trans)

            if oid not in per_object:
                per_object[oid] = {"gt": [], "pred": []}
            per_object[oid]["gt"].append(gt_state)
            per_object[oid]["pred"].append(pred_state)
            n_evaluated += 1

    results = {"n_evaluated": n_evaluated}

    if gt_states:
        results["state_accuracy"] = round(accuracy(gt_states, pred_states), 4)
        results["state_prf"] = precision_recall_f1(gt_states, pred_states)

        labels = sorted(set(gt_states) | set(pred_states))
        cm, cm_labels = confusion_matrix(gt_states, pred_states, labels)
        results["confusion_matrix"] = {"matrix": cm.tolist(), "labels": cm_labels}

        # Per-object accuracy
        per_obj_acc = {}
        for oid, data in per_object.items():
            per_obj_acc[oid] = round(accuracy(data["gt"], data["pred"]), 4)
        results["per_object_accuracy"] = per_obj_acc

        # Transition detection
        results["transition_prf"] = precision_recall_f1(
            gt_transitions, pred_transitions, positive_label=True
        )
    else:
        results["status"] = "insufficient_data"

    return results
