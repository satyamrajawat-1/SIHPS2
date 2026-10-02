"""
ASTRA — YOLO Model Export

Export a trained YOLO model to ONNX for deployment.

Usage:
    python -m ml.models.detection.export --weights checkpoints/detection/best.pt --format onnx
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def export_model(weights: str, fmt: str = "onnx", imgsz: int = 640, device: str = "cpu") -> dict:
    """Export YOLO model to specified format."""
    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError("ultralytics not installed")

    if not os.path.exists(weights):
        raise FileNotFoundError(f"Checkpoint not found: {weights}")

    model = YOLO(weights)
    export_path = model.export(format=fmt, imgsz=imgsz, device=device)

    # Save metadata
    meta = {
        "source_weights": weights,
        "export_format": fmt,
        "export_path": str(export_path),
        "imgsz": imgsz,
        "model_name": "YOLO11-S",
    }

    meta_path = str(export_path) + ".meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Exported to: {export_path}")
    return meta


def main():
    parser = argparse.ArgumentParser(description="ASTRA YOLO Export")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--format", default="onnx", choices=["onnx", "torchscript", "engine"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    result = export_model(args.weights, args.format, args.imgsz, args.device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
