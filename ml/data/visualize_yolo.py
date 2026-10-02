"""
ASTRA — YOLO Dataset Visualizer

Visualize annotated YOLO samples to verify labels are correct.

Usage:
    python -m ml.data.visualize_yolo --dataset data/yolo_astra/ --split train --n 20 --output eval_results/yolo_samples/
"""
from __future__ import annotations

import argparse
import logging
import os
import random
from pathlib import Path

import cv2
import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

COLORS = [
    (80, 60, 200),   # container - red-ish
    (50, 200, 50),   # sample - green
    (200, 200, 60),  # scissors - yellow
    (60, 180, 200),  # tweezers - cyan
    (160, 100, 60),  # tray - brown
]


def visualize_samples(
    dataset_dir: str,
    split: str = "train",
    n: int = 20,
    output_dir: str = None,
    seed: int = 42,
):
    """Visualize YOLO annotated samples."""
    random.seed(seed)

    # Load data.yaml
    data_yaml = os.path.join(dataset_dir, "data.yaml")
    with open(data_yaml) as f:
        config = yaml.safe_load(f)

    class_names = config["names"]
    img_dir = os.path.join(dataset_dir, "images", split)
    lbl_dir = os.path.join(dataset_dir, "labels", split)

    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Image dir not found: {img_dir}")

    images = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
    if not images:
        raise ValueError(f"No images found in {img_dir}")

    random.shuffle(images)
    samples = images[:n]

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    stats = {"total": 0, "by_class": {name: 0 for name in class_names}}

    for img_name in samples:
        img_path = os.path.join(img_dir, img_name)
        lbl_path = os.path.join(lbl_dir, img_name.replace(".jpg", ".txt").replace(".png", ".txt"))

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]

        if os.path.exists(lbl_path):
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    cls_id = int(parts[0])
                    cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

                    # Convert to pixel coords
                    x1 = int((cx - bw / 2) * w)
                    y1 = int((cy - bh / 2) * h)
                    x2 = int((cx + bw / 2) * w)
                    y2 = int((cy + bh / 2) * h)

                    color = COLORS[cls_id % len(COLORS)]
                    cls_name = class_names[cls_id] if cls_id < len(class_names) else f"cls_{cls_id}"

                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, cls_name, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                    stats["total"] += 1
                    stats["by_class"][cls_name] = stats["by_class"].get(cls_name, 0) + 1

        if output_dir:
            out_path = os.path.join(output_dir, f"viz_{img_name}")
            cv2.imwrite(out_path, img)

    logger.info(f"Visualized {len(samples)} samples from {split}")
    logger.info(f"Total annotations: {stats['total']}")
    logger.info(f"Class distribution: {stats['by_class']}")

    if output_dir:
        logger.info(f"Saved to: {output_dir}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Visualize YOLO dataset")
    parser.add_argument("--dataset", required=True, help="YOLO dataset dir")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--n", type=int, default=20, help="Number of samples")
    parser.add_argument("--output", default=None, help="Output dir for visualizations")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    visualize_samples(args.dataset, args.split, args.n, args.output, args.seed)


if __name__ == "__main__":
    main()
