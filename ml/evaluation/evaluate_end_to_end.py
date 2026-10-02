"""
ASTRA — End-to-End Pipeline Evaluator

Runs the full inference pipeline on an evaluation dataset,
collects all intermediate predictions, then evaluates every component
using the domain-specific evaluators.

Also captures failure cases with full evidence traces.
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def evaluate_end_to_end(
    pipeline,
    dataset,
    experiment_config,
    output_dir: str = "eval_results",
    max_frames: int = -1,
) -> Dict[str, Any]:
    """
    Run the full pipeline on an evaluation dataset and evaluate everything.

    Returns a comprehensive results dict covering all 11 evaluation domains.
    """
    from ml.evaluation.evaluate_detection import evaluate_detection
    from ml.evaluation.evaluate_pose import evaluate_pose
    from ml.evaluation.evaluate_action import evaluate_action
    from ml.evaluation.evaluate_hoi import evaluate_hoi
    from ml.evaluation.evaluate_object_state import evaluate_object_state
    from ml.evaluation.evaluate_temporal import evaluate_temporal
    from ml.evaluation.evaluate_procedure import evaluate_procedure, compute_failure_cases
    from ml.evaluation.benchmark import benchmark_pipeline

    os.makedirs(output_dir, exist_ok=True)

    pipeline.reset()
    results = {"dataset_summary": dataset.summary()}

    # ---- Run inference on all samples ----
    logger.info(f"Running inference on {len(dataset.samples)} samples...")

    all_det_preds = []
    all_pose_preds = []
    all_action_preds = []
    all_hoi_preds = []
    all_state_preds = []

    # Temporal predictions
    temporal_gt_actions = []
    temporal_pred_actions = []
    temporal_pred_confs = []
    temporal_gt_steps = []
    temporal_pred_steps = []
    temporal_pred_step_confs = []
    temporal_pred_anomaly = []
    temporal_gt_anomaly = []

    frame_latencies = []
    n_processed = 0

    for i, sample in enumerate(dataset.samples):
        if max_frames > 0 and i >= max_frames:
            break

        # Load frame or generate synthetic
        frame = None
        if sample.image_path and os.path.isfile(sample.image_path):
            import cv2
            frame = cv2.imread(sample.image_path)

        if frame is None:
            # Synthetic frame
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Run pipeline
        t0 = time.perf_counter()
        result = pipeline.process_frame(frame, timestamp=sample.timestamp)
        t1 = time.perf_counter()
        frame_latencies.append((t1 - t0) * 1000)

        sm = result.get("state_machine", {})

        # Collect detection predictions
        dets = pipeline.detector.detect(frame) if hasattr(pipeline, 'detector') else []
        det_preds = [{"bbox": d.bbox, "class": d.class_name, "confidence": d.confidence}
                     for d in dets]
        all_det_preds.append(det_preds)

        # Collect pose predictions
        poses = pipeline.pose_estimator.predict(frame)
        pose_preds = [{"keypoints_2d": p.keypoints_2d.tolist(), "person_id": p.person_id}
                      for p in poses]
        all_pose_preds.append(pose_preds)

        # Collect action predictions
        action_pred = {"action_id": result.get("action", "idle"), "confidence": 0.5}
        all_action_preds.append(action_pred)

        # Collect HOI predictions
        hoi_preds = []
        for pose in poses:
            hois = pipeline.hoi_model.predict(pose, dets, frame)
            for h in hois:
                hoi_preds.append({
                    "hand": h.hand, "object_id": h.object_id,
                    "interaction_type": h.interaction_type,
                    "confidence": h.confidence,
                })
        all_hoi_preds.append(hoi_preds)

        # Collect object state predictions
        state_preds = pipeline.object_state.predict(dets, frame, [])
        state_pred_dicts = [{"object_id": s.object_id, "current_state": s.current_state,
                             "is_transition": s.is_transition, "confidence": s.state_confidence}
                            for s in state_preds]
        all_state_preds.append(state_pred_dicts)

        # Temporal data
        if sample.gt_step:
            gt_action = sample.gt_step.get("action_id", None)
            gt_step = sample.gt_step.get("step_id", None)
            if gt_action:
                temporal_gt_actions.append(gt_action)
                temporal_pred_actions.append(result.get("action", "idle"))
                temporal_pred_confs.append(0.5)
            if gt_step:
                temporal_gt_steps.append(gt_step)
                temporal_pred_steps.append(sm.get("current_step_id", "?"))
                temporal_pred_step_confs.append(0.5)

        n_processed += 1

    logger.info(f"Inference complete: {n_processed} frames processed")

    # ---- Evaluate each component ----

    # 1. Detection
    results["detection"] = evaluate_detection(dataset, all_det_preds)

    # 2. Pose
    results["pose"] = evaluate_pose(dataset, all_pose_preds)

    # 3. Action
    results["action"] = evaluate_action(
        dataset,
        frame_predictions=all_action_preds,
        segment_predictions=None,  # No segments in mock mode
    )

    # 4. HOI
    results["hoi"] = evaluate_hoi(dataset, all_hoi_preds)

    # 5. Object state
    results["object_state"] = evaluate_object_state(dataset, all_state_preds)

    # 6. Temporal
    results["temporal"] = evaluate_temporal(
        gt_actions=temporal_gt_actions,
        pred_actions=temporal_pred_actions,
        pred_action_confidences=temporal_pred_confs,
        gt_steps=temporal_gt_steps,
        pred_steps=temporal_pred_steps,
        pred_step_confidences=temporal_pred_step_confs,
    )

    # 7. Procedure validation (PRIMARY METRIC)
    sm_summary = pipeline.state_machine.get_state_summary()
    pred_step_seq = sm_summary.get("completed_steps", [])
    pred_step_statuses = {}
    step_confidences = {}
    step_evidence = {}
    for sid, ss in pipeline.state_machine.step_states.items():
        pred_step_statuses[sid] = ss.status.value
        step_confidences[sid] = ss.avg_evidence_score
        if ss.evidence_history:
            step_evidence[sid] = ss.evidence_history[-1]

    all_step_ids = [s.id for s in experiment_config.steps]

    results["procedure"] = evaluate_procedure(
        gt_step_sequence=dataset.gt_step_sequence,
        pred_step_sequence=pred_step_seq,
        gt_step_statuses=dataset.gt_step_statuses,
        pred_step_statuses=pred_step_statuses,
        all_step_ids=all_step_ids,
        gt_anomalies=dataset.gt_anomalies,
        pred_anomalies=sm_summary.get("anomalies", []),
        gt_experiment_complete=len(dataset.gt_step_sequence) == len(all_step_ids),
        pred_experiment_complete=sm_summary.get("experiment_status") == "COMPLETE",
        step_confidences=step_confidences,
    )

    # 8. Failure cases
    results["failure_cases"] = compute_failure_cases(
        all_step_ids, dataset.gt_step_statuses, pred_step_statuses,
        step_evidence=step_evidence, step_confidences=step_confidences,
    )

    # 9. Latency
    lat = np.array(frame_latencies)
    results["latency"] = {
        "n_frames": n_processed,
        "mean_ms": round(float(np.mean(lat)), 2),
        "std_ms": round(float(np.std(lat)), 2),
        "p50_ms": round(float(np.percentile(lat, 50)), 2),
        "p95_ms": round(float(np.percentile(lat, 95)), 2),
        "fps": round(1000.0 / float(np.mean(lat)), 1) if np.mean(lat) > 0 else 0,
    }

    # 10. Memory
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        mem = proc.memory_info()
        results["memory"] = {"rss_mb": round(mem.rss / 1024 / 1024, 1),
                             "vms_mb": round(mem.vms / 1024 / 1024, 1)}
    except ImportError:
        results["memory"] = {"note": "psutil not installed"}

    # 11. Known limitations
    results["known_limitations"] = _get_known_limitations(results, dataset)

    # Save JSON results
    json_path = os.path.join(output_dir, "evaluation_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to: {json_path}")

    # Save failure cases
    if results["failure_cases"]:
        fc_path = os.path.join(output_dir, "failure_cases.json")
        with open(fc_path, "w") as f:
            json.dump(results["failure_cases"], f, indent=2, default=str)
        logger.info(f"Failure cases saved to: {fc_path}")

    return results


def _get_known_limitations(results: Dict, dataset) -> List[str]:
    """Compile known limitations based on evaluation results."""
    limitations = []

    summary = dataset.summary()

    if not summary["has_detections"]:
        limitations.append("No ground truth detection annotations — detection metrics are unreliable")
    if not summary["has_poses"]:
        limitations.append("No ground truth pose annotations — pose metrics are unreliable")
    if not summary["has_actions"]:
        limitations.append("No ground truth action segments — action metrics are unreliable")
    if not summary["has_step_sequence"]:
        limitations.append("No ground truth step sequence — procedure metrics are unreliable")

    if summary["n_sessions"] <= 1:
        limitations.append("Single session only — cannot validate cross-session generalization")
    if summary["n_samples"] < 100:
        limitations.append(f"Small dataset ({summary['n_samples']} samples) — metrics may have high variance")

    # Check for models in mock mode
    proc = results.get("procedure", {})
    if proc.get("step_validation", {}).get("uncertain_count", 0) == len(dataset.step_ids):
        limitations.append("All steps are UNCERTAIN — models may not have real weights loaded")

    return limitations
