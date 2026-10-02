"""
ASTRA — Core Evaluation Metrics

Reusable metric functions used by all domain evaluators.
Covers classification, detection, temporal segmentation, and sequence metrics.

RULE: Never fabricate metrics. Report NaN / "insufficient_data" when ground truth is absent.
"""

from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_INSUFFICIENT = "insufficient_data"


# ============================================================
# Classification Metrics
# ============================================================

def confusion_matrix(
    y_true: Sequence,
    y_pred: Sequence,
    labels: Optional[List] = None,
) -> Tuple[np.ndarray, List]:
    """
    Compute confusion matrix.

    Returns:
        (matrix, label_list)  matrix[i, j] = count of true=i predicted=j
    """
    if not y_true or not y_pred:
        labs = labels or []
        return np.zeros((len(labs), len(labs)), dtype=int), labs

    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    label_to_idx = {l: i for i, l in enumerate(labels)}
    n = len(labels)
    cm = np.zeros((n, n), dtype=int)

    for t, p in zip(y_true, y_pred):
        ti = label_to_idx.get(t)
        pi = label_to_idx.get(p)
        if ti is not None and pi is not None:
            cm[ti, pi] += 1

    return cm, labels


def precision_recall_f1(
    y_true: Sequence,
    y_pred: Sequence,
    positive_label=None,
) -> Dict[str, float]:
    """
    Compute precision, recall, F1 for a specific class or macro-averaged.

    Returns dict with precision, recall, f1, support.
    """
    if not y_true or not y_pred:
        return {"precision": float("nan"), "recall": float("nan"),
                "f1": float("nan"), "support": 0}

    if positive_label is not None:
        tp = sum(1 for t, p in zip(y_true, y_pred)
                 if t == positive_label and p == positive_label)
        fp = sum(1 for t, p in zip(y_true, y_pred)
                 if t != positive_label and p == positive_label)
        fn = sum(1 for t, p in zip(y_true, y_pred)
                 if t == positive_label and p != positive_label)
        support = tp + fn
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return {"precision": prec, "recall": rec, "f1": f1, "support": support}

    # Macro-averaged
    labels = sorted(set(y_true) | set(y_pred))
    per_class = {}
    for lab in labels:
        per_class[lab] = precision_recall_f1(y_true, y_pred, positive_label=lab)

    macro_prec = np.mean([v["precision"] for v in per_class.values()])
    macro_rec = np.mean([v["recall"] for v in per_class.values()])
    macro_f1 = np.mean([v["f1"] for v in per_class.values()])
    total_support = sum(v["support"] for v in per_class.values())

    return {
        "precision": float(macro_prec),
        "recall": float(macro_rec),
        "f1": float(macro_f1),
        "support": total_support,
        "per_class": per_class,
    }


def accuracy(y_true: Sequence, y_pred: Sequence) -> float:
    """Simple accuracy."""
    if not y_true:
        return float("nan")
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def top_k_accuracy(y_true: Sequence, y_pred_probs: List[Dict[str, float]], k: int = 3) -> float:
    """Top-k accuracy from probability dicts."""
    if not y_true or not y_pred_probs:
        return float("nan")
    correct = 0
    for t, probs in zip(y_true, y_pred_probs):
        top_k_labels = sorted(probs, key=probs.get, reverse=True)[:k]
        if t in top_k_labels:
            correct += 1
    return correct / len(y_true)


# ============================================================
# Detection Metrics  (mAP / IoU)
# ============================================================

def compute_iou(box_a: List[float], box_b: List[float]) -> float:
    """
    Compute IoU between two boxes [x1, y1, x2, y2].
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def compute_ap(precisions: List[float], recalls: List[float]) -> float:
    """
    Compute Average Precision using all-point interpolation.
    """
    if not precisions or not recalls:
        return 0.0

    # Add sentinel values
    mrec = [0.0] + list(recalls) + [1.0]
    mpre = [0.0] + list(precisions) + [0.0]

    # Monotonic decreasing interpolation
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # Compute area under curve
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]

    return ap


def compute_map(
    gt_boxes: List[List[Dict]],
    pred_boxes: List[List[Dict]],
    iou_threshold: float = 0.5,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute mAP across all classes for a set of images.

    gt_boxes[i] = list of {"bbox": [x1,y1,x2,y2], "class": str}
    pred_boxes[i] = list of {"bbox": [x1,y1,x2,y2], "class": str, "confidence": float}

    Returns dict with mAP, per-class AP, and counts.
    """
    if not gt_boxes and not pred_boxes:
        return {"mAP": float("nan"), "per_class": {}, "status": _INSUFFICIENT}

    # Collect all classes
    all_classes = set()
    for gts in gt_boxes:
        for g in gts:
            all_classes.add(g["class"])
    for preds in pred_boxes:
        for p in preds:
            all_classes.add(p["class"])

    if class_names:
        all_classes = set(class_names) | all_classes
    all_classes = sorted(all_classes)

    per_class_ap = {}

    for cls in all_classes:
        # Collect all predictions and ground truths for this class
        all_preds = []
        n_gt = 0

        for img_idx, (gts, preds) in enumerate(zip(gt_boxes, pred_boxes)):
            cls_gts = [g for g in gts if g["class"] == cls]
            cls_preds = [p for p in preds if p["class"] == cls]
            n_gt += len(cls_gts)

            for pred in cls_preds:
                all_preds.append({
                    "img_idx": img_idx,
                    "confidence": pred["confidence"],
                    "bbox": pred["bbox"],
                    "gt_boxes": cls_gts,
                })

        if n_gt == 0:
            per_class_ap[cls] = {"ap": float("nan"), "n_gt": 0, "n_pred": len(all_preds)}
            continue

        # Sort predictions by confidence (descending)
        all_preds.sort(key=lambda x: x["confidence"], reverse=True)

        tp_list = []
        fp_list = []
        matched_gts = defaultdict(set)

        for pred in all_preds:
            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(pred["gt_boxes"]):
                iou = compute_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            key = (pred["img_idx"], best_gt_idx)
            if best_iou >= iou_threshold and key not in matched_gts:
                tp_list.append(1)
                fp_list.append(0)
                matched_gts[pred["img_idx"]].add(best_gt_idx)
            else:
                tp_list.append(0)
                fp_list.append(1)

        # Cumulative sums
        tp_cum = np.cumsum(tp_list)
        fp_cum = np.cumsum(fp_list)

        recalls = (tp_cum / n_gt).tolist()
        precisions = (tp_cum / (tp_cum + fp_cum)).tolist()

        ap = compute_ap(precisions, recalls)
        per_class_ap[cls] = {"ap": round(ap, 4), "n_gt": n_gt, "n_pred": len(all_preds)}

    valid_aps = [v["ap"] for v in per_class_ap.values() if not math.isnan(v["ap"])]
    mean_ap = float(np.mean(valid_aps)) if valid_aps else float("nan")

    return {
        "mAP": round(mean_ap, 4),
        "iou_threshold": iou_threshold,
        "per_class": per_class_ap,
        "n_classes": len(all_classes),
    }


# ============================================================
# Tracking Metrics  (MOTA / IDF1 simplified)
# ============================================================

def compute_tracking_metrics(
    gt_tracks: List[Dict],
    pred_tracks: List[Dict],
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Simplified tracking metrics.

    gt_tracks: list of {frame_id, track_id, bbox}
    pred_tracks: list of {frame_id, track_id, bbox}

    Returns: MOTA, IDF1 approximations, ID switches, FP, FN.
    """
    if not gt_tracks:
        return {"mota": float("nan"), "status": _INSUFFICIENT}

    frames = sorted(set(t["frame_id"] for t in gt_tracks) |
                    set(t["frame_id"] for t in pred_tracks))

    gt_by_frame = defaultdict(list)
    pred_by_frame = defaultdict(list)
    for t in gt_tracks:
        gt_by_frame[t["frame_id"]].append(t)
    for t in pred_tracks:
        pred_by_frame[t["frame_id"]].append(t)

    total_gt = 0
    total_fp = 0
    total_fn = 0
    total_id_sw = 0
    prev_matches = {}

    for frame in frames:
        gts = gt_by_frame[frame]
        preds = pred_by_frame[frame]
        total_gt += len(gts)

        matched_gt = set()
        matched_pred = set()
        current_matches = {}

        for pi, pred in enumerate(preds):
            best_iou = 0
            best_gi = -1
            for gi, gt in enumerate(gts):
                if gi in matched_gt:
                    continue
                iou = compute_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_gi = gi

            if best_gi >= 0:
                matched_gt.add(best_gi)
                matched_pred.add(pi)
                gt_tid = gts[best_gi]["track_id"]
                pred_tid = pred["track_id"]
                current_matches[gt_tid] = pred_tid

                if gt_tid in prev_matches and prev_matches[gt_tid] != pred_tid:
                    total_id_sw += 1

        total_fp += len(preds) - len(matched_pred)
        total_fn += len(gts) - len(matched_gt)
        prev_matches = current_matches

    mota = 1.0 - (total_fp + total_fn + total_id_sw) / max(1, total_gt)

    return {
        "mota": round(float(mota), 4),
        "total_gt": total_gt,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "id_switches": total_id_sw,
    }


# ============================================================
# Pose Metrics  (PCK / MPJPE)
# ============================================================

def compute_pck(
    gt_keypoints: np.ndarray,
    pred_keypoints: np.ndarray,
    threshold_fraction: float = 0.2,
    reference_length: Optional[float] = None,
) -> Dict[str, float]:
    """
    Percentage of Correct Keypoints.

    Args:
        gt_keypoints: (J, 2+) ground truth.
        pred_keypoints: (J, 2+) predictions.
        threshold_fraction: Fraction of reference length.
        reference_length: If None, uses head size.

    Returns: PCK score and per-joint correctness.
    """
    if gt_keypoints.size == 0 or pred_keypoints.size == 0:
        return {"pck": float("nan"), "status": _INSUFFICIENT}

    J = min(gt_keypoints.shape[0], pred_keypoints.shape[0])
    gt = gt_keypoints[:J, :2]
    pred = pred_keypoints[:J, :2]

    if reference_length is None:
        # Use head size (distance between joints 0 and 1 or bbox diagonal)
        reference_length = max(np.ptp(gt[:, 0]), np.ptp(gt[:, 1])) * 0.5
        reference_length = max(reference_length, 1.0)

    threshold = threshold_fraction * reference_length
    dists = np.linalg.norm(gt - pred, axis=1)
    correct = (dists < threshold).astype(float)

    # Handle invisible joints
    if gt_keypoints.shape[1] > 2:
        visibility = gt_keypoints[:J, 2] > 0.1
        correct[~visibility] = float("nan")
        visible_correct = correct[visibility]
        pck = float(np.mean(visible_correct)) if len(visible_correct) > 0 else float("nan")
    else:
        pck = float(np.mean(correct))

    return {"pck": round(pck, 4), "per_joint": correct.tolist(), "threshold": threshold}


def compute_mpjpe(
    gt_keypoints: np.ndarray,
    pred_keypoints: np.ndarray,
) -> Dict[str, float]:
    """Mean Per Joint Position Error (pixels)."""
    if gt_keypoints.size == 0 or pred_keypoints.size == 0:
        return {"mpjpe": float("nan"), "status": _INSUFFICIENT}

    J = min(gt_keypoints.shape[0], pred_keypoints.shape[0])
    dists = np.linalg.norm(
        gt_keypoints[:J, :2] - pred_keypoints[:J, :2], axis=1
    )
    return {
        "mpjpe": round(float(np.mean(dists)), 2),
        "per_joint": [round(float(d), 2) for d in dists],
    }


# ============================================================
# Temporal Segmentation Metrics
# ============================================================

def compute_edit_distance(seq_a: List, seq_b: List) -> int:
    """Levenshtein edit distance between two sequences."""
    n, m = len(seq_a), len(seq_b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    return dp[n][m]


def compute_segment_f1(
    gt_segments: List[Dict],
    pred_segments: List[Dict],
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Temporal segment-level F1.

    Each segment: {"label": str, "start": float, "end": float}

    A predicted segment is correct if it overlaps ≥ iou_threshold
    with a GT segment of the same label.
    """
    if not gt_segments:
        return {"f1": float("nan"), "status": _INSUFFICIENT}

    tp = 0
    fp = 0
    matched_gt = set()

    for pi, pred in enumerate(pred_segments):
        best_iou = 0
        best_gi = -1

        for gi, gt in enumerate(gt_segments):
            if gi in matched_gt:
                continue
            if gt["label"] != pred["label"]:
                continue

            # Temporal IoU
            inter_start = max(gt["start"], pred["start"])
            inter_end = min(gt["end"], pred["end"])
            inter = max(0, inter_end - inter_start)
            union = (gt["end"] - gt["start"]) + (pred["end"] - pred["start"]) - inter

            iou = inter / union if union > 0 else 0
            if iou > best_iou:
                best_iou = iou
                best_gi = gi

        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_gi)
        else:
            fp += 1

    fn = len(gt_segments) - len(matched_gt)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "tp": tp, "fp": fp, "fn": fn,
    }


# ============================================================
# Procedure / Sequence Metrics  (PRIMARY ASTRA METRIC)
# ============================================================

def compute_sequence_accuracy(
    gt_step_sequence: List[str],
    pred_step_sequence: List[str],
) -> Dict[str, Any]:
    """
    PROCEDURE VALIDATION PERFORMANCE — primary ASTRA metric.

    Compares the ground-truth step sequence against the predicted sequence.
    """
    if not gt_step_sequence:
        return {"sequence_accuracy": float("nan"), "status": _INSUFFICIENT}

    # Exact sequence match
    exact_match = gt_step_sequence == pred_step_sequence

    # Edit distance
    edit_dist = compute_edit_distance(gt_step_sequence, pred_step_sequence)
    max_len = max(len(gt_step_sequence), len(pred_step_sequence))
    normalised_edit = 1.0 - edit_dist / max_len if max_len > 0 else 1.0

    # Per-step alignment
    gt_set = set(gt_step_sequence)
    pred_set = set(pred_step_sequence)

    correct_steps = gt_set & pred_set
    missed_steps = gt_set - pred_set
    extra_steps = pred_set - gt_set

    # Order correctness (Kendall tau distance approximation)
    common = [s for s in gt_step_sequence if s in pred_set]
    pred_common = [s for s in pred_step_sequence if s in gt_set]
    order_correct = common == pred_common

    return {
        "sequence_accuracy": round(normalised_edit, 4),
        "exact_match": exact_match,
        "edit_distance": edit_dist,
        "normalised_edit": round(normalised_edit, 4),
        "order_correct": order_correct,
        "correct_steps": sorted(correct_steps),
        "missed_steps": sorted(missed_steps),
        "extra_steps": sorted(extra_steps),
        "gt_length": len(gt_step_sequence),
        "pred_length": len(pred_step_sequence),
    }


def compute_step_metrics(
    gt_step_statuses: Dict[str, str],
    pred_step_statuses: Dict[str, str],
    all_step_ids: List[str],
) -> Dict[str, Any]:
    """
    Per-step precision / recall / F1 for step validation.

    gt_step_statuses: {step_id: "VALID" | "SKIPPED" | ...}
    pred_step_statuses: {step_id: "VALID" | "UNCERTAIN" | ...}
    """
    y_true = []
    y_pred = []
    per_step = {}

    for sid in all_step_ids:
        gt = gt_step_statuses.get(sid, "NOT_EVALUATED")
        pred = pred_step_statuses.get(sid, "NOT_EVALUATED")
        y_true.append(gt)
        y_pred.append(pred)

        gt_valid = gt == "VALID"
        pred_valid = pred == "VALID"

        per_step[sid] = {
            "gt": gt,
            "pred": pred,
            "correct": gt == pred,
            "false_valid": not gt_valid and pred_valid,
            "false_invalid": gt_valid and not pred_valid,
        }

    # Binary: was the step correctly marked VALID?
    tp = sum(1 for s in per_step.values() if s["gt"] == "VALID" and s["pred"] == "VALID")
    fp = sum(1 for s in per_step.values() if s["false_valid"])
    fn = sum(1 for s in per_step.values() if s["false_invalid"])
    tn = sum(1 for s in per_step.values()
             if s["gt"] != "VALID" and s["pred"] != "VALID")

    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else float("nan")

    false_valid_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    uncertain_count = sum(1 for s in per_step.values() if s["pred"] == "UNCERTAIN")
    uncertain_rate = uncertain_count / len(all_step_ids) if all_step_ids else 0.0

    return {
        "step_precision": round(prec, 4) if not math.isnan(prec) else prec,
        "step_recall": round(rec, 4) if not math.isnan(rec) else rec,
        "step_f1": round(f1, 4) if not math.isnan(f1) else f1,
        "false_valid_rate": round(false_valid_rate, 4),
        "uncertain_rate": round(uncertain_rate, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "uncertain_count": uncertain_count,
        "per_step": per_step,
    }


def compute_anomaly_detection_metrics(
    gt_anomalies: List[Dict],
    pred_anomalies: List[Dict],
    time_tolerance: float = 2.0,
) -> Dict[str, Any]:
    """
    Evaluate anomaly / out-of-order / skipped-step detection.

    Each anomaly: {"type": str, "timestamp": float, "step_id": str}
    """
    if not gt_anomalies and not pred_anomalies:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "note": "no_anomalies_in_gt_or_pred"}

    if not gt_anomalies:
        return {"precision": 0.0, "recall": float("nan"), "f1": float("nan"),
                "false_alarms": len(pred_anomalies), "status": _INSUFFICIENT}

    matched_gt = set()
    tp = 0
    fp = 0

    for pred in pred_anomalies:
        found = False
        for gi, gt in enumerate(gt_anomalies):
            if gi in matched_gt:
                continue
            if (gt["type"] == pred["type"] and
                    abs(gt.get("timestamp", 0) - pred.get("timestamp", 0)) < time_tolerance):
                tp += 1
                matched_gt.add(gi)
                found = True
                break
        if not found:
            fp += 1

    fn = len(gt_anomalies) - len(matched_gt)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn,
    }


# ============================================================
# Confidence Calibration
# ============================================================

def compute_ece(
    confidences: List[float],
    correct: List[bool],
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Expected Calibration Error.

    Measures whether confidence scores match actual accuracy.
    """
    if not confidences:
        return {"ece": float("nan"), "status": _INSUFFICIENT}

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_details = []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = [(lo <= c < hi) for c in confidences]
        n = sum(mask)
        if n == 0:
            continue
        avg_conf = np.mean([c for c, m in zip(confidences, mask) if m])
        avg_acc = np.mean([int(cor) for cor, m in zip(correct, mask) if m])
        ece += (n / len(confidences)) * abs(avg_acc - avg_conf)
        bin_details.append({
            "bin": f"{lo:.1f}-{hi:.1f}", "n": n,
            "avg_confidence": round(float(avg_conf), 4),
            "avg_accuracy": round(float(avg_acc), 4),
        })

    return {"ece": round(float(ece), 4), "bins": bin_details}
