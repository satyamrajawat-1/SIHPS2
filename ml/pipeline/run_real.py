"""
ASTRA — Run Real Pipeline Inference

Runs the full ASTRA pipeline with real or mock models on video/session data.
Produces evaluation-ready output.

Usage:
    # Real mode with trained YOLO weights
    python -m ml.pipeline.run_real --experiment experiments/sample_experiment/ \
        --video data/synthetic/session_001/session_001.mp4 \
        --detection_weights checkpoints/detection/YOLO11S-ASTRA-v1/weights/best.pt \
        --output eval_results/real_v1/

    # Mock mode comparison
    python -m ml.pipeline.run_real --experiment experiments/sample_experiment/ \
        --video data/synthetic/session_001/session_001.mp4 \
        --mock --output eval_results/mock_v1/
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.experiment.schema.loader import load_experiment
from ml.pipeline.inference import ASTRAPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline_on_video(
    experiment_dir: str,
    video_path: str,
    output_dir: str,
    detection_weights: str = "",
    action_weights: str = "",
    temporal_weights: str = "",
    use_mock: bool = False,
    mock_pose: bool = False,
    max_frames: int = -1,
    save_annotated: bool = True,
) -> dict:
    """Run the full ASTRA pipeline on a video."""
    config = load_experiment(experiment_dir)
    os.makedirs(output_dir, exist_ok=True)

    mode_label = "MOCK" if use_mock else "REAL"
    logger.info(f"Pipeline mode: {mode_label}")
    logger.info(f"Video: {video_path}")

    pipeline = ASTRAPipeline(
        config, device="cpu", use_mock=use_mock, mock_pose=mock_pose,
        log_dir=os.path.join(output_dir, "logs"),
        skeleton_window=30,
    )
    pipeline.load_models(
        detection_weights=detection_weights,
        action_weights=action_weights,
        temporal_weights=temporal_weights,
    )

    if video_path.isdigit():
        cap = cv2.VideoCapture(int(video_path))
    else:
        cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video/camera: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output video
    out_video = None
    if save_annotated:
        out_path = os.path.join(output_dir, "annotated_output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    frame_results = []
    latencies = []
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames > 0 and idx >= max_frames:
            break

        timestamp = idx / fps
        t0 = time.perf_counter()
        result = pipeline.process_frame(frame, timestamp=timestamp)
        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)

        sm = result.get("state_machine", {})
        frame_results.append({
            "frame": idx,
            "timestamp": round(timestamp, 3),
            "detections": result.get("detections", 0),
            "poses": result.get("poses", 0),
            "action": result.get("action"),
            "temporal_action": result.get("temporal_action"),
            "temporal_step": result.get("temporal_step"),
            "temporal_next": result.get("temporal_next"),
            "anomaly_score": round(result.get("anomaly_score", 0.0), 3),
            "step": sm.get("current_step_id"),
            "status": sm.get("experiment_status"),
            "confidence": sm.get("confidence", 0),
            "latency_ms": round(lat, 1),
        })

        # Draw annotations on frame
        if save_annotated and out_video:
            cv2.putText(frame, f"Step: {sm.get('current_step_id', '?')}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Status: {sm.get('experiment_status', '?')}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
            cv2.putText(frame, f"Det: {result.get('detections', 0)} | {mode_label}", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
            t_act = result.get("temporal_action") or "?"
            t_next = result.get("temporal_next") or "?"
            t_anom = result.get("anomaly_score", 0.0)
            cv2.putText(frame, f"T-Act: {t_act} | Next: {t_next} | Anom: {t_anom:.2f}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 100, 255), 1)
            cv2.putText(frame, f"Lat: {lat:.0f}ms", (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            out_video.write(frame)

        idx += 1

    cap.release()
    if out_video:
        out_video.release()

    pipeline.close()

    lat_arr = np.array(latencies) if latencies else np.array([0])
    summary = pipeline.state_machine.get_state_summary()

    output = {
        "mode": mode_label,
        "data_type": "SYNTHETIC",
        "video": video_path,
        "detection_weights": detection_weights if not use_mock else "mock",
        "frames_processed": idx,
        "total_frames": total,
        "mean_latency_ms": round(float(np.mean(lat_arr)), 1),
        "p95_latency_ms": round(float(np.percentile(lat_arr, 95)), 1),
        "fps": round(1000 / float(np.mean(lat_arr)), 1) if np.mean(lat_arr) > 0 else 0,
        "final_status": summary.get("experiment_status"),
        "completed_steps": summary.get("completed_steps", []),
        "current_step": summary.get("current_step_id"),
        "anomalies": summary.get("anomalies", []),
        "step_statuses": {sid: ss.status.value for sid, ss in pipeline.state_machine.step_states.items()},
        "frame_results": frame_results,
    }

    # Save results
    with open(os.path.join(output_dir, "pipeline_results.json"), "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"\nPipeline [{mode_label}] complete:")
    logger.info(f"  Frames: {idx}")
    logger.info(f"  FPS: {output['fps']}")
    logger.info(f"  Status: {output['final_status']}")
    logger.info(f"  Completed: {len(output['completed_steps'])}/{len(config.steps)}")

    return output


def main():
    parser = argparse.ArgumentParser(description="ASTRA Real Pipeline Inference")
    parser.add_argument("--experiment", required=True, help="Experiment config dir")
    parser.add_argument("--video", required=True, help="Input video")
    parser.add_argument("--output", default="eval_results/real_v1")
    parser.add_argument("--detection_weights", default="")
    parser.add_argument("--action_weights", default="")
    parser.add_argument("--temporal_weights", default="", help="Temporal GRU checkpoint")
    parser.add_argument("--mock", action="store_true", help="Use mock models")
    parser.add_argument("--mock_pose", action="store_true", help="Mock pose model to avoid download")
    parser.add_argument("--max_frames", type=int, default=-1)
    parser.add_argument("--no_annotated", action="store_true")
    args = parser.parse_args()

    run_pipeline_on_video(
        experiment_dir=args.experiment,
        video_path=args.video,
        output_dir=args.output,
        detection_weights=args.detection_weights,
        action_weights=args.action_weights,
        temporal_weights=args.temporal_weights,
        use_mock=args.mock,
        mock_pose=args.mock_pose,
        max_frames=args.max_frames,
        save_annotated=not args.no_annotated,
    )


if __name__ == "__main__":
    main()
