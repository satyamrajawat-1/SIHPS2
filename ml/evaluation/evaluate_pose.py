"""
ASTRA — Pose Estimation Evaluator

Evaluates: PCK@0.2, MPJPE, per-joint accuracy.
"""
from __future__ import annotations
import logging
import numpy as np
from typing import Any, Dict, List
from ml.evaluation.metrics import compute_pck, compute_mpjpe
from ml.evaluation.dataset_loader import EvalDataset

logger = logging.getLogger(__name__)

def evaluate_pose(
    dataset: EvalDataset,
    predictions: List[List[Dict]],
) -> Dict[str, Any]:
    """
    Evaluate pose estimation.

    predictions[i] = list of {"keypoints_2d": [[x,y,conf],...], "person_id": int}
    """
    gt_all = [s.gt_poses for s in dataset.samples]
    if not any(gt_all):
        return {"status": "insufficient_data", "reason": "No ground truth poses"}

    pck_scores = []
    mpjpe_scores = []
    per_joint_pck = None
    n_evaluated = 0

    for gt_list, pred_list in zip(gt_all, predictions):
        if not gt_list or not pred_list:
            continue
        for gt, pred in zip(gt_list, pred_list):
            gt_kps = np.array(gt.get("keypoints_2d", []))
            pred_kps = np.array(pred.get("keypoints_2d", []))
            if gt_kps.size == 0 or pred_kps.size == 0:
                continue

            pck_r = compute_pck(gt_kps, pred_kps, threshold_fraction=0.2)
            mpjpe_r = compute_mpjpe(gt_kps, pred_kps)

            if not np.isnan(pck_r.get("pck", float("nan"))):
                pck_scores.append(pck_r["pck"])
                n_evaluated += 1
            if not np.isnan(mpjpe_r.get("mpjpe", float("nan"))):
                mpjpe_scores.append(mpjpe_r["mpjpe"])

            if per_joint_pck is None and "per_joint" in pck_r:
                per_joint_pck = [[] for _ in pck_r["per_joint"]]
            if per_joint_pck is not None and "per_joint" in pck_r:
                for j, v in enumerate(pck_r["per_joint"]):
                    if j < len(per_joint_pck) and not np.isnan(v):
                        per_joint_pck[j].append(v)

    avg_pck = float(np.mean(pck_scores)) if pck_scores else float("nan")
    avg_mpjpe = float(np.mean(mpjpe_scores)) if mpjpe_scores else float("nan")

    joint_names = [
        "nose", "l_eye", "r_eye", "l_ear", "r_ear",
        "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
        "l_wrist", "r_wrist", "l_hip", "r_hip",
        "l_knee", "r_knee", "l_ankle", "r_ankle",
    ]
    per_joint_summary = {}
    if per_joint_pck:
        for j, scores in enumerate(per_joint_pck):
            name = joint_names[j] if j < len(joint_names) else f"joint_{j}"
            per_joint_summary[name] = round(float(np.mean(scores)), 4) if scores else float("nan")

    return {
        "pck@0.2": round(avg_pck, 4),
        "mpjpe_px": round(avg_mpjpe, 2),
        "n_evaluated": n_evaluated,
        "per_joint_pck": per_joint_summary,
    }
