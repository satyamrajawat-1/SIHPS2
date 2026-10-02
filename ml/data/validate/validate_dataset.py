"""
ASTRA — Dataset Validation Tool

Validates annotation consistency, schema compliance, and cross-reference integrity.

Usage:
    python -m ml.data.validate.validate_dataset --dataset data/datasets/sample/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ValidationReport:
    """Accumulates validation errors and warnings."""

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []
        self.stats: Dict[str, Any] = {}

    def add_error(self, msg: str):
        self.errors.append(msg)
        logger.error(f"  ERROR: {msg}")

    def add_warning(self, msg: str):
        self.warnings.append(msg)
        logger.warning(f"  WARNING: {msg}")

    def add_info(self, msg: str):
        self.info.append(msg)
        logger.info(f"  INFO: {msg}")

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "VALIDATION REPORT",
            "=" * 60,
            f"Errors:   {len(self.errors)}",
            f"Warnings: {len(self.warnings)}",
            f"Status:   {'PASS' if self.is_valid else 'FAIL'}",
            "",
        ]
        if self.errors:
            lines.append("ERRORS:")
            for e in self.errors:
                lines.append(f"  ✗ {e}")
            lines.append("")
        if self.warnings:
            lines.append("WARNINGS:")
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")
            lines.append("")
        if self.stats:
            lines.append("STATISTICS:")
            for k, v in self.stats.items():
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.info,
            "stats": self.stats,
        }


def validate_frame_manifest(manifest_path: str, report: ValidationReport):
    """Validate a frame extraction manifest."""
    if not os.path.exists(manifest_path):
        report.add_warning(f"Frame manifest not found: {manifest_path}")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    frames = manifest.get("frames", [])
    report.stats["total_frames"] = len(frames)

    # Check for required fields
    required_fields = ["frame_id", "filename", "timestamp"]
    for i, frame in enumerate(frames):
        for field in required_fields:
            if field not in frame:
                report.add_error(f"Frame {i} missing required field: {field}")

    # Check for duplicate frame IDs
    frame_ids = [f.get("frame_id") for f in frames]
    duplicates = [fid for fid in set(frame_ids) if frame_ids.count(fid) > 1]
    if duplicates:
        report.add_error(f"Duplicate frame IDs: {duplicates[:10]}")

    # Check timestamps are monotonically increasing
    timestamps = [f.get("timestamp", 0) for f in frames]
    for i in range(1, len(timestamps)):
        if timestamps[i] < timestamps[i - 1]:
            report.add_warning(
                f"Non-monotonic timestamp at frame {i}: "
                f"{timestamps[i]} < {timestamps[i-1]}"
            )
            break

    report.add_info(f"Frame manifest: {len(frames)} frames validated")


def validate_detection_annotations(annotations_path: str, report: ValidationReport):
    """Validate COCO-style detection annotations."""
    if not os.path.exists(annotations_path):
        report.add_warning(f"Detection annotations not found: {annotations_path}")
        return

    with open(annotations_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    annotations = data.get("annotations", [])
    categories = data.get("categories", [])
    images = data.get("images", [])

    report.stats["detection_annotations"] = len(annotations)
    report.stats["detection_categories"] = len(categories)
    report.stats["detection_images"] = len(images)

    # Validate categories
    cat_ids = set(c["id"] for c in categories)
    image_ids = set(img["id"] for img in images)

    for ann in annotations:
        if ann.get("category_id") not in cat_ids:
            report.add_error(
                f"Annotation {ann.get('id')} references unknown category "
                f"{ann.get('category_id')}"
            )
        if ann.get("image_id") not in image_ids:
            report.add_error(
                f"Annotation {ann.get('id')} references unknown image "
                f"{ann.get('image_id')}"
            )

        # Validate bbox format
        bbox = ann.get("bbox", [])
        if len(bbox) != 4:
            report.add_error(f"Annotation {ann.get('id')} has invalid bbox: {bbox}")
        elif any(v < 0 for v in bbox[:2]) or any(v <= 0 for v in bbox[2:]):
            report.add_warning(
                f"Annotation {ann.get('id')} has suspicious bbox values: {bbox}"
            )

    report.add_info(f"Detection annotations: {len(annotations)} validated")


def validate_pose_annotations(annotations_path: str, report: ValidationReport):
    """Validate pose keypoint annotations (JSONL format)."""
    if not os.path.exists(annotations_path):
        report.add_warning(f"Pose annotations not found: {annotations_path}")
        return

    count = 0
    with open(annotations_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                ann = json.loads(line)
            except json.JSONDecodeError:
                report.add_error(f"Pose annotation line {line_num}: invalid JSON")
                continue

            count += 1

            # Check required fields
            for field in ["frame_id", "person_id", "keypoints_2d"]:
                if field not in ann:
                    report.add_error(
                        f"Pose annotation line {line_num}: missing field '{field}'"
                    )

            # Validate keypoint format
            kps = ann.get("keypoints_2d", [])
            if kps and isinstance(kps[0], list):
                for kp_idx, kp in enumerate(kps):
                    if len(kp) < 2:
                        report.add_error(
                            f"Pose line {line_num}, keypoint {kp_idx}: "
                            f"needs at least [x, y], got {kp}"
                        )

    report.stats["pose_annotations"] = count
    report.add_info(f"Pose annotations: {count} entries validated")


def validate_action_annotations(annotations_path: str, report: ValidationReport):
    """Validate temporal action segment annotations (JSONL format)."""
    if not os.path.exists(annotations_path):
        report.add_warning(f"Action annotations not found: {annotations_path}")
        return

    count = 0
    actions_seen = set()

    with open(annotations_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                ann = json.loads(line)
            except json.JSONDecodeError:
                report.add_error(f"Action annotation line {line_num}: invalid JSON")
                continue

            count += 1
            actions_seen.add(ann.get("action_id", "unknown"))

            # Check temporal consistency
            start = ann.get("start_frame", 0)
            end = ann.get("end_frame", 0)
            if end <= start:
                report.add_error(
                    f"Action line {line_num}: end_frame ({end}) <= start_frame ({start})"
                )

    report.stats["action_annotations"] = count
    report.stats["unique_actions"] = len(actions_seen)
    report.add_info(f"Action annotations: {count} segments, {len(actions_seen)} unique actions")


def validate_hoi_annotations(annotations_path: str, report: ValidationReport):
    """Validate hand-object interaction annotations (JSONL format)."""
    if not os.path.exists(annotations_path):
        report.add_warning(f"HOI annotations not found: {annotations_path}")
        return

    count = 0
    valid_hands = {"left", "right"}

    with open(annotations_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                ann = json.loads(line)
            except json.JSONDecodeError:
                report.add_error(f"HOI annotation line {line_num}: invalid JSON")
                continue

            count += 1

            hand = ann.get("hand", "")
            if hand not in valid_hands:
                report.add_error(
                    f"HOI line {line_num}: invalid hand '{hand}', "
                    f"must be 'left' or 'right'"
                )

    report.stats["hoi_annotations"] = count
    report.add_info(f"HOI annotations: {count} entries validated")


def validate_step_annotations(annotations_path: str, report: ValidationReport):
    """Validate experiment step annotations (JSONL format)."""
    if not os.path.exists(annotations_path):
        report.add_warning(f"Step annotations not found: {annotations_path}")
        return

    count = 0
    valid_statuses = {"completed", "skipped", "failed", "partial", "in_progress"}

    with open(annotations_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                ann = json.loads(line)
            except json.JSONDecodeError:
                report.add_error(f"Step annotation line {line_num}: invalid JSON")
                continue

            count += 1

            status = ann.get("status", "")
            if status not in valid_statuses:
                report.add_warning(
                    f"Step line {line_num}: unusual status '{status}'"
                )

    report.stats["step_annotations"] = count
    report.add_info(f"Step annotations: {count} entries validated")


def validate_dataset(dataset_dir: str) -> ValidationReport:
    """
    Validate an entire ASTRA dataset directory.

    Expected structure:
        dataset_dir/
            manifest.json or frame_manifest.json
            annotations/
                detections.json       (COCO-style)
                poses.jsonl
                actions.jsonl
                hoi.jsonl
                object_states.jsonl
                steps.jsonl
            frames/               (optional, if frames are stored here)
            split.json            (optional)
    """
    report = ValidationReport()
    dataset_dir = os.path.abspath(dataset_dir)

    logger.info(f"Validating dataset: {dataset_dir}")

    if not os.path.isdir(dataset_dir):
        report.add_error(f"Dataset directory does not exist: {dataset_dir}")
        return report

    # Check manifest
    manifest_found = False
    for mname in ["manifest.json", "frame_manifest.json"]:
        mpath = os.path.join(dataset_dir, mname)
        if os.path.exists(mpath):
            validate_frame_manifest(mpath, report)
            manifest_found = True
            break

    if not manifest_found:
        report.add_warning("No manifest file found (manifest.json or frame_manifest.json)")

    # Check annotation files
    ann_dir = os.path.join(dataset_dir, "annotations")
    if os.path.isdir(ann_dir):
        validate_detection_annotations(
            os.path.join(ann_dir, "detections.json"), report
        )
        validate_pose_annotations(
            os.path.join(ann_dir, "poses.jsonl"), report
        )
        validate_action_annotations(
            os.path.join(ann_dir, "actions.jsonl"), report
        )
        validate_hoi_annotations(
            os.path.join(ann_dir, "hoi.jsonl"), report
        )
        validate_step_annotations(
            os.path.join(ann_dir, "steps.jsonl"), report
        )
    else:
        report.add_warning(f"Annotations directory not found: {ann_dir}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="ASTRA Dataset Validation Tool",
    )
    parser.add_argument("--dataset", required=True, help="Path to dataset directory")
    parser.add_argument("--output", help="Save report to JSON file")

    args = parser.parse_args()

    report = validate_dataset(args.dataset)
    print(report.summary())

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info(f"Report saved to {args.output}")

    exit(0 if report.is_valid else 1)


if __name__ == "__main__":
    main()
