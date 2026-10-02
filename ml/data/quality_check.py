"""
ASTRA — Dataset Quality Check

Validates annotations, reports statistics, detects issues.
Rejects malformed annotations.

Usage:
    python -m ml.data.quality_check --dataset data/raw/session_001/
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def check_dataset_quality(dataset_dir: str) -> dict:
    """Run full quality check on a dataset directory."""
    report = {
        "dataset_dir": os.path.abspath(dataset_dir),
        "statistics": {},
        "issues": [],
        "warnings": [],
    }

    # Check manifest
    manifest_path = os.path.join(dataset_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        report["issues"].append("CRITICAL: manifest.json not found")
        return report

    with open(manifest_path) as f:
        manifest = json.load(f)

    frames = manifest.get("frames", [])
    report["statistics"]["total_frames"] = len(frames)
    report["statistics"]["duration_seconds"] = manifest.get("duration_seconds", 0)
    report["statistics"]["extract_fps"] = manifest.get("extract_fps", 0)

    # Check frames exist
    frames_dir = os.path.join(dataset_dir, "frames")
    missing_frames = 0
    for fr in frames:
        fpath = os.path.join(frames_dir, fr["filename"])
        if not os.path.exists(fpath):
            missing_frames += 1
    if missing_frames > 0:
        report["issues"].append(f"Missing {missing_frames}/{len(frames)} frame files")

    # Check annotations
    ann_dir = os.path.join(dataset_dir, "annotations")
    if not os.path.isdir(ann_dir):
        report["warnings"].append("No annotations directory found")
        return report

    # Check each annotation type
    ann_counts = {}
    for name in ["actions", "steps", "object_states", "anomalies", "poses", "hoi"]:
        path = os.path.join(ann_dir, f"{name}.jsonl")
        if os.path.exists(path):
            items = []
            errors = 0
            with open(path) as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        items.append(item)
                    except json.JSONDecodeError:
                        errors += 1
                        report["issues"].append(f"Malformed JSON in {name}.jsonl line {line_no}")

            ann_counts[name] = len(items)
            if errors > 0:
                report["issues"].append(f"{errors} malformed lines in {name}.jsonl")

            # Validate annotation fields
            _validate_annotations(name, items, frames, report)
        else:
            ann_counts[name] = 0
            report["warnings"].append(f"{name}.jsonl not found")

    report["statistics"]["annotations"] = ann_counts

    # Check detections (COCO format)
    det_path = os.path.join(ann_dir, "detections.json")
    if os.path.exists(det_path):
        try:
            with open(det_path) as f:
                det_data = json.load(f)
            n_anns = len(det_data.get("annotations", []))
            n_cats = len(det_data.get("categories", []))
            ann_counts["detections"] = n_anns
            report["statistics"]["detection_categories"] = n_cats

            # Class distribution
            class_counts = Counter(
                a.get("category_id") for a in det_data.get("annotations", [])
            )
            report["statistics"]["detection_class_distribution"] = dict(class_counts)
        except Exception as e:
            report["issues"].append(f"Error reading detections.json: {e}")

    # Check step sequence
    seq_path = os.path.join(ann_dir, "step_sequence.json")
    if os.path.exists(seq_path):
        with open(seq_path) as f:
            seq = json.load(f)
        report["statistics"]["step_sequence_length"] = len(seq.get("sequence", []))

    # Session manifest
    session_path = os.path.join(dataset_dir, "session_manifest.json")
    if os.path.exists(session_path):
        with open(session_path) as f:
            session = json.load(f)
        report["statistics"]["session_id"] = session.get("session_id")
        report["statistics"]["performer_id"] = session.get("performer_id")
        report["statistics"]["experiment_id"] = session.get("experiment_id")

    # Compute class imbalance
    if "actions" in ann_counts and ann_counts["actions"] > 0:
        action_path = os.path.join(ann_dir, "actions.jsonl")
        action_labels = []
        with open(action_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    action_labels.append(json.loads(line).get("action_id", "?"))
        counts = Counter(action_labels)
        if counts:
            max_c = max(counts.values())
            min_c = min(counts.values())
            ratio = max_c / max(min_c, 1)
            report["statistics"]["action_class_distribution"] = dict(counts)
            report["statistics"]["action_imbalance_ratio"] = round(ratio, 1)
            if ratio > 10:
                report["warnings"].append(f"High action class imbalance: {ratio:.1f}x")

    # Check overlapping segments
    for name in ["actions", "steps"]:
        path = os.path.join(ann_dir, f"{name}.jsonl")
        if os.path.exists(path):
            items = []
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        items.append(json.loads(line))
            overlaps = _check_overlaps(items)
            if overlaps:
                report["warnings"].append(f"{len(overlaps)} overlapping segments in {name}")

    # Summary
    n_issues = len(report["issues"])
    n_warnings = len(report["warnings"])
    report["summary"] = {
        "issues": n_issues,
        "warnings": n_warnings,
        "status": "FAIL" if n_issues > 0 else ("WARN" if n_warnings > 0 else "PASS"),
    }

    return report


def _validate_annotations(name, items, frames, report):
    """Validate annotation fields."""
    max_frame = len(frames) - 1 if frames else 0

    for i, item in enumerate(items):
        sf = item.get("start_frame")
        ef = item.get("end_frame")

        if sf is not None and ef is not None:
            if sf > ef:
                report["issues"].append(
                    f"{name}[{i}]: start_frame ({sf}) > end_frame ({ef})")
            if sf < 0 or (max_frame > 0 and ef > max_frame * 1.5):
                report["warnings"].append(
                    f"{name}[{i}]: frame range [{sf}-{ef}] may be out of bounds")


def _check_overlaps(items):
    """Check for overlapping temporal segments."""
    sorted_items = sorted(items, key=lambda x: x.get("start_frame", 0))
    overlaps = []
    for i in range(len(sorted_items) - 1):
        a = sorted_items[i]
        b = sorted_items[i + 1]
        a_end = a.get("end_frame", 0)
        b_start = b.get("start_frame", 0)
        if a_end > b_start:
            overlaps.append((i, i + 1))
    return overlaps


def print_report(report):
    """Print quality check report."""
    print("\n" + "=" * 60)
    print("  ASTRA Dataset Quality Check")
    print("=" * 60)

    stats = report.get("statistics", {})
    print(f"\n  Frames:        {stats.get('total_frames', 0)}")
    print(f"  Duration:      {stats.get('duration_seconds', 0)}s")
    print(f"  Session:       {stats.get('session_id', '?')}")
    print(f"  Performer:     {stats.get('performer_id', '?')}")

    anns = stats.get("annotations", {})
    if anns:
        print(f"\n  Annotations:")
        for name, count in anns.items():
            print(f"    {name:20s} {count}")

    issues = report.get("issues", [])
    if issues:
        print(f"\n  ISSUES ({len(issues)}):")
        for iss in issues:
            print(f"    [ERROR] {iss}")

    warnings = report.get("warnings", [])
    if warnings:
        print(f"\n  WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"    [WARN]  {w}")

    status = report.get("summary", {}).get("status", "?")
    print(f"\n  Status: {status}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="ASTRA Dataset Quality Check")
    parser.add_argument("--dataset", required=True, help="Dataset/session directory")
    parser.add_argument("--output", help="Save report JSON")
    args = parser.parse_args()

    report = check_dataset_quality(args.dataset)
    print_report(report)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n  Report saved: {args.output}")


if __name__ == "__main__":
    main()
