"""
ASTRA — YOLO Detection Validation

Validate a trained YOLO model on a test set.

Usage:
    python -m ml.models.detection.validate --weights checkpoints/detection/best.pt --data data/detection.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def validate(weights: str, data_yaml: str, device: str = "cpu",
             imgsz: int = 640, batch: int = 16) -> dict:
    """Run YOLO validation."""
    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError("ultralytics not installed. Run: pip install ultralytics")

    if not os.path.exists(weights):
        raise FileNotFoundError(
            f"Checkpoint not found: {weights}\n"
            f"Train first: python -m ml.models.detection.train --data {data_yaml}")

    model = YOLO(weights)
    logger.info(f"Validating: {weights}")

    metrics = model.val(data=data_yaml, device=device, imgsz=imgsz, batch=batch)

    results = {
        "weights": weights,
        "data": data_yaml,
        "device": device,
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50_95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "per_class": {},
    }

    if hasattr(metrics.box, 'maps') and metrics.box.maps is not None:
        for i, ap in enumerate(metrics.box.maps):
            cls_name = model.names.get(i, f"class_{i}")
            results["per_class"][cls_name] = round(float(ap), 4)

    return results


def main():
    parser = argparse.ArgumentParser(description="ASTRA YOLO Validation")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True, help="data.yaml path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--output", help="Save results JSON")
    args = parser.parse_args()

    results = validate(args.weights, args.data, args.device, args.imgsz, args.batch)
    print(json.dumps(results, indent=2))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved: {args.output}")


if __name__ == "__main__":
    main()
