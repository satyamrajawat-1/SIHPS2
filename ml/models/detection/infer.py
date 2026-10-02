"""
ASTRA — YOLO Detection Inference

Run real inference with a trained YOLO model.
Never silently falls back to mock mode.

Usage:
    python -m ml.models.detection.infer --input sample.jpg --weights checkpoints/detection/best.pt
    python -m ml.models.detection.infer --input video.mp4 --weights checkpoints/detection/best.pt --save
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def infer_image(weights: str, image_path: str, conf: float = 0.25,
                device: str = "cpu", save_output: bool = False) -> dict:
    """Run YOLO inference on a single image."""
    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError("ultralytics not installed. Run: pip install ultralytics")

    if not os.path.exists(weights):
        raise FileNotFoundError(
            f"Checkpoint not found: {weights}\n"
            f"To use pretrained: yolo export model=yolo11s.pt\n"
            f"To fine-tune: python -m ml.models.detection.train --data <data.yaml>")

    model = YOLO(weights)
    logger.info(f"Model loaded: {weights}")

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    t0 = time.perf_counter()
    results = model.predict(image, conf=conf, device=device, verbose=False)
    latency = (time.perf_counter() - t0) * 1000

    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class_id": int(box.cls[0]),
                "class_name": model.names[int(box.cls[0])],
                "confidence": round(float(box.conf[0]), 4),
                "bbox": [round(float(x), 1) for x in box.xyxy[0].tolist()],
            })

    output = {
        "model": weights,
        "input": image_path,
        "device": device,
        "confidence_threshold": conf,
        "image_shape": list(image.shape[:2]),
        "predictions": len(detections),
        "latency_ms": round(latency, 1),
        "detections": detections,
    }

    if save_output:
        annotated = results[0].plot()
        out_path = image_path.replace(".", "_annotated.")
        cv2.imwrite(out_path, annotated)
        output["annotated_output"] = out_path
        logger.info(f"Annotated image saved: {out_path}")

    return output


def infer_video(weights: str, video_path: str, conf: float = 0.25,
                device: str = "cpu", max_frames: int = -1) -> dict:
    """Run YOLO inference on a video."""
    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError("ultralytics not installed. Run: pip install ultralytics")

    if not os.path.exists(weights):
        raise FileNotFoundError(f"Checkpoint not found: {weights}")

    model = YOLO(weights)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_detections = []
    latencies = []
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames > 0 and idx >= max_frames:
            break

        t0 = time.perf_counter()
        results = model.predict(frame, conf=conf, device=device, verbose=False)
        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)

        dets = []
        for r in results:
            for box in r.boxes:
                dets.append({
                    "class_name": model.names[int(box.cls[0])],
                    "confidence": round(float(box.conf[0]), 4),
                })
        frame_detections.append({"frame": idx, "count": len(dets)})
        idx += 1

    cap.release()
    lat_arr = np.array(latencies) if latencies else np.array([0])

    return {
        "model": weights,
        "video": video_path,
        "frames_processed": idx,
        "total_frames": total,
        "mean_latency_ms": round(float(np.mean(lat_arr)), 1),
        "fps": round(1000 / float(np.mean(lat_arr)), 1) if np.mean(lat_arr) > 0 else 0,
        "mean_detections_per_frame": round(
            np.mean([d["count"] for d in frame_detections]), 1) if frame_detections else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="ASTRA YOLO Detection Inference")
    parser.add_argument("--input", required=True, help="Image or video path")
    parser.add_argument("--weights", default="checkpoints/detection/best.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save", action="store_true", help="Save annotated output")
    parser.add_argument("--max_frames", type=int, default=-1)
    args = parser.parse_args()

    ext = Path(args.input).suffix.lower()
    if ext in (".jpg", ".jpeg", ".png", ".bmp"):
        result = infer_image(args.weights, args.input, args.conf, args.device, args.save)
    elif ext in (".mp4", ".avi", ".mov", ".mkv"):
        result = infer_video(args.weights, args.input, args.conf, args.device, args.max_frames)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
