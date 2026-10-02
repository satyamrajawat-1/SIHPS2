"""
ASTRA — Convert Synthetic/Annotated Dataset to YOLO Format

Converts COCO-format detection annotations to YOLO format
with session-aware train/val/test split.

Usage:
    python -m ml.data.convert_to_yolo --input data/synthetic/ --output data/yolo_astra/ --split
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def convert_coco_to_yolo(coco_data: dict, img_w: int, img_h: int) -> Dict[int, List[str]]:
    """
    Convert COCO annotations to YOLO format.

    Returns dict mapping image_id to list of YOLO annotation lines.
    """
    yolo_annotations = {}

    for ann in coco_data.get("annotations", []):
        img_id = ann["image_id"]
        cat_id = ann["category_id"]

        # COCO bbox is [x, y, w, h]
        bx, by, bw, bh = ann["bbox"]

        # Convert to YOLO: center_x, center_y, width, height (normalized)
        cx = (bx + bw / 2) / img_w
        cy = (by + bh / 2) / img_h
        nw = bw / img_w
        nh = bh / img_h

        # Clamp to [0, 1]
        cx = max(0, min(1, cx))
        cy = max(0, min(1, cy))
        nw = max(0, min(1, nw))
        nh = max(0, min(1, nh))

        line = f"{cat_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"

        if img_id not in yolo_annotations:
            yolo_annotations[img_id] = []
        yolo_annotations[img_id].append(line)

    return yolo_annotations


def convert_dataset(
    input_dir: str,
    output_dir: str,
    split_ratios: tuple = (0.6, 0.2, 0.2),
    seed: int = 42,
):
    """
    Convert complete dataset to YOLO format with session-aware split.
    """
    random.seed(seed)

    # Find all sessions
    sessions = []
    for entry in sorted(os.listdir(input_dir)):
        session_dir = os.path.join(input_dir, entry)
        if not os.path.isdir(session_dir):
            continue
        det_path = os.path.join(session_dir, "annotations", "detections.json")
        if os.path.exists(det_path):
            sessions.append(entry)

    if not sessions:
        raise ValueError(f"No sessions with detections found in {input_dir}")

    logger.info(f"Found {len(sessions)} sessions")

    # Session-aware split
    random.shuffle(sessions)
    n = len(sessions)
    n_train = max(1, int(n * split_ratios[0]))
    n_val = max(1, int(n * split_ratios[1]))

    train_sessions = sessions[:n_train]
    val_sessions = sessions[n_train:n_train + n_val]
    test_sessions = sessions[n_train + n_val:]

    # Ensure test has at least 1 session
    if not test_sessions and val_sessions:
        test_sessions = [val_sessions.pop()]

    logger.info(f"Split: train={len(train_sessions)}, val={len(val_sessions)}, test={len(test_sessions)}")
    logger.info(f"  Train: {train_sessions}")
    logger.info(f"  Val:   {val_sessions}")
    logger.info(f"  Test:  {test_sessions}")

    # Create YOLO directory structure
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    # Get class names from first session
    first_det = os.path.join(input_dir, sessions[0], "annotations", "detections.json")
    with open(first_det) as f:
        first_coco = json.load(f)
    categories = {c["id"]: c["name"] for c in first_coco.get("categories", [])}

    stats = {"train": 0, "val": 0, "test": 0, "total_labels": 0, "empty_labels": 0}
    class_counts = {name: 0 for name in categories.values()}

    split_map = {}
    for s in train_sessions:
        split_map[s] = "train"
    for s in val_sessions:
        split_map[s] = "val"
    for s in test_sessions:
        split_map[s] = "test"

    # Convert each session
    for session_id in sessions:
        split = split_map[session_id]
        session_dir = os.path.join(input_dir, session_id)

        det_path = os.path.join(session_dir, "annotations", "detections.json")
        with open(det_path) as f:
            coco_data = json.load(f)

        # Image dimensions from first image
        if coco_data["images"]:
            img_w = coco_data["images"][0].get("width", 640)
            img_h = coco_data["images"][0].get("height", 480)
        else:
            img_w, img_h = 640, 480

        yolo_anns = convert_coco_to_yolo(coco_data, img_w, img_h)

        frames_dir = os.path.join(session_dir, "frames")

        for img_info in coco_data["images"]:
            img_id = img_info["id"]
            src_img = os.path.join(frames_dir, img_info["file_name"])

            if not os.path.exists(src_img):
                continue

            # Unique filename: session_frameId.jpg
            dst_name = f"{session_id}_{img_info['file_name']}"
            label_name = dst_name.replace(".jpg", ".txt")

            dst_img = os.path.join(output_dir, "images", split, dst_name)
            dst_lbl = os.path.join(output_dir, "labels", split, label_name)

            shutil.copy2(src_img, dst_img)

            # Write label file
            lines = yolo_anns.get(img_id, [])
            with open(dst_lbl, "w") as f:
                f.write("\n".join(lines))

            if lines:
                for line in lines:
                    cls_id = int(line.split()[0])
                    cls_name = categories.get(cls_id, f"class_{cls_id}")
                    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                stats["total_labels"] += len(lines)
            else:
                stats["empty_labels"] += 1

            stats[split] += 1

    # Create data.yaml
    data_yaml = {
        "path": os.path.abspath(output_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(categories),
        "names": [categories[i] for i in sorted(categories.keys())],
    }

    yaml_path = os.path.join(output_dir, "data.yaml")
    import yaml
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    # Save split info
    split_info = {
        "data_type": "SYNTHETIC",
        "split_method": "session_aware",
        "seed": seed,
        "train_sessions": train_sessions,
        "val_sessions": val_sessions,
        "test_sessions": test_sessions,
        "stats": stats,
        "class_distribution": class_counts,
    }
    with open(os.path.join(output_dir, "split_info.json"), "w") as f:
        json.dump(split_info, f, indent=2)

    logger.info(f"\nYOLO dataset created: {output_dir}")
    logger.info(f"  Images:  train={stats['train']}, val={stats['val']}, test={stats['test']}")
    logger.info(f"  Labels:  {stats['total_labels']} total, {stats['empty_labels']} empty")
    logger.info(f"  Classes: {class_counts}")

    return split_info


def main():
    parser = argparse.ArgumentParser(description="Convert to YOLO format")
    parser.add_argument("--input", required=True, help="Input dataset dir")
    parser.add_argument("--output", required=True, help="Output YOLO dir")
    parser.add_argument("--ratios", nargs=3, type=float, default=[0.6, 0.2, 0.2])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    convert_dataset(args.input, args.output, tuple(args.ratios), args.seed)


if __name__ == "__main__":
    main()
