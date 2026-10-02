"""
ASTRA — Real Model Evaluation Script

Evaluates trained YOLO model on the test split and generates
detection metrics + failure analysis.

Runs YOLO validation, then full pipeline evaluation, then
step-validation against ground truth annotations.

Usage:
    python -m ml.evaluation.evaluate_real \
        --experiment experiments/sample_experiment/ \
        --dataset data/synthetic/ \
        --yolo_weights checkpoints/detection/YOLO11S-ASTRA-v1/weights/best.pt \
        --yolo_data data/yolo_astra/data.yaml \
        --output eval_results/real_v1/
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_yolo_validation(weights_path: str, data_yaml: str, output_dir: str, device: str = "cpu"):
    """Run YOLO validation on test set and return metrics."""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed")
        return None

    logger.info(f"Validating YOLO: {weights_path}")
    model = YOLO(weights_path)

    results = model.val(
        data=data_yaml, split="test", device=device,
        imgsz=320, batch=4, workers=0,
        project=output_dir, name="yolo_val",
    )

    metrics = {
        "mAP50": round(float(results.box.map50), 4),
        "mAP50_95": round(float(results.box.map), 4),
        "precision": round(float(results.box.mp), 4),
        "recall": round(float(results.box.mr), 4),
        "per_class": {},
    }

    # Per-class metrics
    if hasattr(results.box, 'ap_class_index') and results.box.ap_class_index is not None:
        names = model.names
        for i, cls_idx in enumerate(results.box.ap_class_index):
            cls_name = names[int(cls_idx)]
            metrics["per_class"][cls_name] = {
                "AP50": round(float(results.box.ap50[i]), 4),
                "AP50_95": round(float(results.box.ap[i]), 4),
            }

    # Save
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "yolo_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"  mAP@50: {metrics['mAP50']}")
    logger.info(f"  mAP@50:95: {metrics['mAP50_95']}")
    logger.info(f"  Precision: {metrics['precision']}")
    logger.info(f"  Recall: {metrics['recall']}")

    return metrics


def evaluate_step_validation(
    dataset_dir: str,
    test_sessions: list,
    pipeline_results: dict,
) -> dict:
    """
    Compare pipeline step predictions against ground truth annotations.
    """
    results = {
        "total_sessions": 0,
        "step_results": [],
        "false_valid_count": 0,
        "uncertain_count": 0,
        "correct_count": 0,
        "total_steps": 0,
    }

    for session_id in test_sessions:
        ann_dir = os.path.join(dataset_dir, session_id, "annotations")
        steps_file = os.path.join(ann_dir, "steps.jsonl")

        if not os.path.exists(steps_file):
            continue

        # Load GT steps
        gt_steps = []
        with open(steps_file) as f:
            for line in f:
                gt_steps.append(json.loads(line.strip()))

        gt_sequence = [s["step_id"] for s in gt_steps]

        # Get pipeline prediction for this session
        session_pred = pipeline_results.get(session_id, {})
        pred_steps = session_pred.get("completed_steps", [])
        step_statuses = session_pred.get("step_statuses", {})

        results["total_sessions"] += 1
        results["total_steps"] += len(gt_sequence)

        for gt_step in gt_steps:
            sid = gt_step["step_id"]
            gt_status = gt_step.get("status", "completed")
            pred_status = step_statuses.get(sid, "NOT_STARTED")

            is_correct = (
                (gt_status == "completed" and pred_status == "COMPLETED") or
                (gt_status != "completed" and pred_status != "COMPLETED")
            )

            if pred_status == "COMPLETED" and gt_status != "completed":
                results["false_valid_count"] += 1
                category = "FALSE_VALID"
            elif pred_status == "UNCERTAIN":
                results["uncertain_count"] += 1
                category = "UNCERTAIN"
            elif is_correct:
                results["correct_count"] += 1
                category = "CORRECT"
            else:
                category = "MISSED"

            results["step_results"].append({
                "session": session_id,
                "step_id": sid,
                "gt_status": gt_status,
                "pred_status": pred_status,
                "correct": is_correct,
                "category": category,
            })

    if results["total_steps"] > 0:
        results["accuracy"] = round(results["correct_count"] / results["total_steps"], 4)
        results["false_valid_rate"] = round(
            results["false_valid_count"] / results["total_steps"], 4)
        results["uncertain_rate"] = round(
            results["uncertain_count"] / results["total_steps"], 4)
    else:
        results["accuracy"] = None
        results["false_valid_rate"] = None
        results["uncertain_rate"] = None

    return results


def run_full_evaluation(
    experiment_dir: str,
    dataset_dir: str,
    yolo_weights: str,
    yolo_data: str,
    output_dir: str,
    device: str = "cpu",
):
    """Run the complete evaluation pipeline."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. YOLO validation
    logger.info("=" * 60)
    logger.info("  STEP 1: YOLO Validation")
    logger.info("=" * 60)
    yolo_metrics = run_yolo_validation(yolo_weights, yolo_data, output_dir, device)

    # 2. Run pipeline on test videos
    logger.info("=" * 60)
    logger.info("  STEP 2: Pipeline Inference on Test Sessions")
    logger.info("=" * 60)

    # Load split info
    split_path = os.path.join(os.path.dirname(yolo_data), "split_info.json")
    if os.path.exists(split_path):
        with open(split_path) as f:
            split_info = json.load(f)
        test_sessions = split_info.get("test_sessions", [])
    else:
        test_sessions = []
        for entry in sorted(os.listdir(dataset_dir)):
            if entry.startswith("session_"):
                test_sessions.append(entry)
        test_sessions = test_sessions[-3:]  # Use last 3

    logger.info(f"Test sessions: {test_sessions}")

    from ml.pipeline.run_real import run_pipeline_on_video

    pipeline_results = {}
    for session_id in test_sessions:
        video_path = os.path.join(dataset_dir, session_id, f"{session_id}.mp4")
        if not os.path.exists(video_path):
            logger.warning(f"Video not found: {video_path}")
            continue

        session_out = os.path.join(output_dir, f"pipeline_{session_id}")
        result = run_pipeline_on_video(
            experiment_dir=experiment_dir,
            video_path=video_path,
            output_dir=session_out,
            detection_weights=yolo_weights,
            use_mock=False,
            save_annotated=True,
        )
        pipeline_results[session_id] = result

    # 3. Step validation
    logger.info("=" * 60)
    logger.info("  STEP 3: Step Validation")
    logger.info("=" * 60)

    step_metrics = evaluate_step_validation(dataset_dir, test_sessions, pipeline_results)

    with open(os.path.join(output_dir, "step_validation.json"), "w") as f:
        json.dump(step_metrics, f, indent=2)

    logger.info(f"  Accuracy: {step_metrics.get('accuracy')}")
    logger.info(f"  False VALID rate: {step_metrics.get('false_valid_rate')}")
    logger.info(f"  UNCERTAIN rate: {step_metrics.get('uncertain_rate')}")

    # 4. Compile final report
    logger.info("=" * 60)
    logger.info("  STEP 4: Final Report")
    logger.info("=" * 60)

    final = {
        "data_type": "SYNTHETIC",
        "yolo_metrics": yolo_metrics,
        "step_validation": step_metrics,
        "test_sessions": test_sessions,
        "pipeline_sessions": list(pipeline_results.keys()),
        "pipeline_summary": {
            sid: {
                "frames": r.get("frames_processed"),
                "fps": r.get("fps"),
                "status": r.get("final_status"),
                "completed": len(r.get("completed_steps", [])),
            }
            for sid, r in pipeline_results.items()
        },
    }

    with open(os.path.join(output_dir, "evaluation_results.json"), "w") as f:
        json.dump(final, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 60)
    print("  ASTRA REAL EVALUATION SUMMARY")
    print("  DATA TYPE: SYNTHETIC")
    print("=" * 60)
    if yolo_metrics:
        print(f"  YOLO mAP@50:     {yolo_metrics['mAP50']}")
        print(f"  YOLO mAP@50:95:  {yolo_metrics['mAP50_95']}")
        print(f"  YOLO Precision:  {yolo_metrics['precision']}")
        print(f"  YOLO Recall:     {yolo_metrics['recall']}")
    if step_metrics:
        print(f"  Step Accuracy:   {step_metrics.get('accuracy')}")
        print(f"  False VALID:     {step_metrics.get('false_valid_rate')}")
        print(f"  UNCERTAIN:       {step_metrics.get('uncertain_rate')}")
    print("=" * 60)

    return final


def main():
    parser = argparse.ArgumentParser(description="ASTRA Real Evaluation")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--yolo_weights", required=True)
    parser.add_argument("--yolo_data", required=True)
    parser.add_argument("--output", default="eval_results/real_v1")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    run_full_evaluation(
        experiment_dir=args.experiment,
        dataset_dir=args.dataset,
        yolo_weights=args.yolo_weights,
        yolo_data=args.yolo_data,
        output_dir=args.output,
        device=args.device,
    )


if __name__ == "__main__":
    main()
