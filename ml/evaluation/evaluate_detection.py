"""
ASTRA — Object Detection Evaluator

Evaluates: mAP@50, mAP@50:95, per-class AP, class confusion.
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional
from ml.evaluation.metrics import compute_map, confusion_matrix, precision_recall_f1
from ml.evaluation.dataset_loader import EvalDataset

logger = logging.getLogger(__name__)

def evaluate_detection(
    dataset: EvalDataset,
    predictions: List[List[Dict]],
    iou_thresholds: List[float] = None,
) -> Dict[str, Any]:
    """
    Evaluate object detection performance.

    predictions[i] = list of {"bbox": [x1,y1,x2,y2], "class": str, "confidence": float}
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.75]

    gt_boxes = [s.gt_detections for s in dataset.samples]

    if not any(gt_boxes):
        return {"status": "insufficient_data", "reason": "No ground truth detections available"}

    if len(predictions) != len(gt_boxes):
        logger.warning(f"Prediction count {len(predictions)} != GT count {len(gt_boxes)}")
        min_len = min(len(predictions), len(gt_boxes))
        gt_boxes = gt_boxes[:min_len]
        predictions = predictions[:min_len]

    results = {"n_images": len(gt_boxes), "n_gt_total": sum(len(g) for g in gt_boxes),
               "n_pred_total": sum(len(p) for p in predictions)}

    for iou_t in iou_thresholds:
        key = f"mAP@{int(iou_t*100)}"
        results[key] = compute_map(gt_boxes, predictions, iou_threshold=iou_t,
                                    class_names=dataset.class_names)

    # Multi-threshold mAP (COCO-style 50:5:95)
    aps = []
    for t in [i / 100.0 for i in range(50, 100, 5)]:
        r = compute_map(gt_boxes, predictions, iou_threshold=t)
        if not __import__("math").isnan(r["mAP"]):
            aps.append(r["mAP"])
    results["mAP@50:95"] = round(float(__import__("numpy").mean(aps)), 4) if aps else float("nan")

    # Class confusion matrix
    gt_classes = [g["class"] for gts in gt_boxes for g in gts]
    # Match predictions to GT by IoU for confusion
    pred_classes = []
    from ml.evaluation.metrics import compute_iou
    for gts_f, preds_f in zip(gt_boxes, predictions):
        for gt in gts_f:
            best_cls = "MISSED"
            best_iou = 0
            for pred in preds_f:
                iou = compute_iou(gt["bbox"], pred["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_cls = pred["class"] if iou >= 0.5 else "MISSED"
            pred_classes.append(best_cls)

    labels = sorted(set(gt_classes) | set(pred_classes))
    cm, cm_labels = confusion_matrix(gt_classes, pred_classes, labels)
    results["confusion_matrix"] = {"matrix": cm.tolist(), "labels": cm_labels}
    results["per_class_metrics"] = precision_recall_f1(gt_classes, pred_classes)

    return results
