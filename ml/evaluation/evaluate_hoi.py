"""
ASTRA — HOI Evaluator

Evaluates: interaction detection accuracy, hand assignment, contact detection.
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List
from ml.evaluation.metrics import precision_recall_f1, accuracy, confusion_matrix
from ml.evaluation.dataset_loader import EvalDataset

logger = logging.getLogger(__name__)

def evaluate_hoi(
    dataset: EvalDataset,
    predictions: List[List[Dict]],
) -> Dict[str, Any]:
    """
    Evaluate hand-object interaction detection.

    predictions[i] = list of {"hand": str, "object_id": str, "interaction_type": str, "confidence": float}
    """
    gt_all = [s.gt_hoi for s in dataset.samples]
    if not any(gt_all):
        return {"status": "insufficient_data", "reason": "No ground truth HOI annotations"}

    gt_types = []
    pred_types = []
    gt_hands = []
    pred_hands = []
    contact_gt = []
    contact_pred = []
    n_evaluated = 0

    for gt_list, pred_list in zip(gt_all, predictions):
        if not gt_list:
            continue
        for gt in gt_list:
            gt_type = gt.get("interaction_type", gt.get("type", "?"))
            gt_types.append(gt_type)
            gt_hands.append(gt.get("hand", "?"))
            contact_gt.append(gt_type not in ("no_contact", "none", "idle"))

            # Find best matching prediction
            best_pred = None
            for pred in pred_list:
                if pred.get("object_id") == gt.get("object_id"):
                    best_pred = pred
                    break
            if best_pred is None and pred_list:
                best_pred = pred_list[0]

            if best_pred:
                pred_types.append(best_pred.get("interaction_type", "?"))
                pred_hands.append(best_pred.get("hand", "?"))
                contact_pred.append(best_pred.get("interaction_type", "?") not in ("no_contact", "none"))
            else:
                pred_types.append("no_contact")
                pred_hands.append("?")
                contact_pred.append(False)
            n_evaluated += 1

    results = {"n_evaluated": n_evaluated}

    if gt_types:
        results["interaction_type_accuracy"] = round(accuracy(gt_types, pred_types), 4)
        results["interaction_prf"] = precision_recall_f1(gt_types, pred_types)
        labels = sorted(set(gt_types) | set(pred_types))
        cm, cm_labels = confusion_matrix(gt_types, pred_types, labels)
        results["confusion_matrix"] = {"matrix": cm.tolist(), "labels": cm_labels}
        results["hand_accuracy"] = round(accuracy(gt_hands, pred_hands), 4)
        results["contact_accuracy"] = round(accuracy(contact_gt, contact_pred), 4)

        # Contact precision/recall
        results["contact_prf"] = precision_recall_f1(contact_gt, contact_pred, positive_label=True)
    else:
        results["status"] = "insufficient_data"

    return results
