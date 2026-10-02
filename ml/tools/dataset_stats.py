"""
ASTRA — Dataset Statistics Tool

Generate comprehensive statistics about a dataset.

Usage:
    python -m ml.tools.dataset_stats --dataset data/datasets/sample/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_stats(dataset_dir: str) -> Dict[str, Any]:
    """
    Compute comprehensive statistics for an ASTRA dataset.

    Args:
        dataset_dir: Path to dataset directory.

    Returns:
        Dict with statistics.
    """
    stats = {
        "dataset_dir": dataset_dir,
        "frames": {},
        "detections": {},
        "poses": {},
        "actions": {},
        "hoi": {},
        "steps": {},
    }

    # Frame statistics
    for mname in ["manifest.json", "frame_manifest.json"]:
        mpath = os.path.join(dataset_dir, mname)
        if os.path.exists(mpath):
            with open(mpath, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            frames = manifest.get("frames", [])
            stats["frames"] = {
                "total_frames": len(frames),
                "sessions": len(set(f.get("session_id", "?") for f in frames)),
                "videos": len(set(f.get("video_id", "?") for f in frames)),
                "cameras": len(set(f.get("camera_id", "?") for f in frames)),
            }
            if frames:
                timestamps = [f.get("timestamp", 0) for f in frames]
                stats["frames"]["duration_seconds"] = max(timestamps) - min(timestamps)
            break

    # Detection statistics
    det_path = os.path.join(dataset_dir, "annotations", "detections.json")
    if os.path.exists(det_path):
        with open(det_path, "r", encoding="utf-8") as f:
            det_data = json.load(f)
        anns = det_data.get("annotations", [])
        cats = det_data.get("categories", [])
        cat_map = {c["id"]: c["name"] for c in cats}

        class_counts = Counter(cat_map.get(a["category_id"], "?") for a in anns)
        stats["detections"] = {
            "total_annotations": len(anns),
            "categories": len(cats),
            "class_distribution": dict(class_counts.most_common()),
            "images_with_annotations": len(set(a["image_id"] for a in anns)),
            "avg_annotations_per_image": round(
                len(anns) / max(1, len(set(a["image_id"] for a in anns))), 2
            ),
        }

    # Pose statistics
    pose_path = os.path.join(dataset_dir, "annotations", "poses.jsonl")
    if os.path.exists(pose_path):
        pose_count = 0
        persons = set()
        with open(pose_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ann = json.loads(line)
                    pose_count += 1
                    persons.add(ann.get("person_id", 0))
                except json.JSONDecodeError:
                    pass
        stats["poses"] = {
            "total_annotations": pose_count,
            "unique_persons": len(persons),
        }

    # Action statistics
    action_path = os.path.join(dataset_dir, "annotations", "actions.jsonl")
    if os.path.exists(action_path):
        action_count = 0
        action_types = Counter()
        durations = []
        with open(action_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ann = json.loads(line)
                    action_count += 1
                    action_types[ann.get("action_id", "?")] += 1
                    start = ann.get("start_time", 0)
                    end = ann.get("end_time", 0)
                    if end > start:
                        durations.append(end - start)
                except json.JSONDecodeError:
                    pass
        stats["actions"] = {
            "total_segments": action_count,
            "unique_actions": len(action_types),
            "action_distribution": dict(action_types.most_common()),
            "avg_duration_seconds": round(
                sum(durations) / max(1, len(durations)), 2
            ) if durations else 0,
            "min_duration_seconds": round(min(durations), 2) if durations else 0,
            "max_duration_seconds": round(max(durations), 2) if durations else 0,
        }

    # HOI statistics
    hoi_path = os.path.join(dataset_dir, "annotations", "hoi.jsonl")
    if os.path.exists(hoi_path):
        hoi_count = 0
        interaction_types = Counter()
        hand_dist = Counter()
        with open(hoi_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ann = json.loads(line)
                    hoi_count += 1
                    interaction_types[ann.get("interaction_type", "?")] += 1
                    hand_dist[ann.get("hand", "?")] += 1
                except json.JSONDecodeError:
                    pass
        stats["hoi"] = {
            "total_annotations": hoi_count,
            "interaction_distribution": dict(interaction_types.most_common()),
            "hand_distribution": dict(hand_dist),
        }

    # Step statistics
    step_path = os.path.join(dataset_dir, "annotations", "steps.jsonl")
    if os.path.exists(step_path):
        step_count = 0
        step_types = Counter()
        step_statuses = Counter()
        with open(step_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ann = json.loads(line)
                    step_count += 1
                    step_types[ann.get("step_id", "?")] += 1
                    step_statuses[ann.get("status", "?")] += 1
                except json.JSONDecodeError:
                    pass
        stats["steps"] = {
            "total_annotations": step_count,
            "step_distribution": dict(step_types.most_common()),
            "status_distribution": dict(step_statuses),
        }

    # Split statistics
    split_path = os.path.join(dataset_dir, "split.json")
    if os.path.exists(split_path):
        with open(split_path, "r", encoding="utf-8") as f:
            split_data = json.load(f)
        stats["split"] = split_data.get("counts", {})

    return stats


def format_stats(stats: Dict[str, Any]) -> str:
    """Format statistics as a human-readable report."""
    lines = [
        "=" * 60,
        "ASTRA DATASET STATISTICS",
        "=" * 60,
        f"Directory: {stats.get('dataset_dir', '?')}",
        "",
    ]

    # Frames
    f = stats.get("frames", {})
    if f:
        lines.extend([
            "--- Frames ---",
            f"  Total frames:     {f.get('total_frames', 0)}",
            f"  Sessions:         {f.get('sessions', 0)}",
            f"  Videos:           {f.get('videos', 0)}",
            f"  Cameras:          {f.get('cameras', 0)}",
            f"  Duration (s):     {f.get('duration_seconds', 0):.1f}",
            "",
        ])

    # Detections
    d = stats.get("detections", {})
    if d:
        lines.extend([
            "--- Object Detection ---",
            f"  Total annotations: {d.get('total_annotations', 0)}",
            f"  Categories:        {d.get('categories', 0)}",
            f"  Avg per image:     {d.get('avg_annotations_per_image', 0)}",
            "  Class distribution:",
        ])
        for cls, cnt in d.get("class_distribution", {}).items():
            lines.append(f"    {cls}: {cnt}")
        lines.append("")

    # Poses
    p = stats.get("poses", {})
    if p:
        lines.extend([
            "--- Pose ---",
            f"  Total annotations: {p.get('total_annotations', 0)}",
            f"  Unique persons:    {p.get('unique_persons', 0)}",
            "",
        ])

    # Actions
    a = stats.get("actions", {})
    if a:
        lines.extend([
            "--- Actions ---",
            f"  Total segments:    {a.get('total_segments', 0)}",
            f"  Unique actions:    {a.get('unique_actions', 0)}",
            f"  Avg duration (s):  {a.get('avg_duration_seconds', 0)}",
            "  Action distribution:",
        ])
        for act, cnt in a.get("action_distribution", {}).items():
            lines.append(f"    {act}: {cnt}")
        lines.append("")

    # HOI
    h = stats.get("hoi", {})
    if h:
        lines.extend([
            "--- Hand-Object Interaction ---",
            f"  Total annotations: {h.get('total_annotations', 0)}",
            "  Interaction types:",
        ])
        for it, cnt in h.get("interaction_distribution", {}).items():
            lines.append(f"    {it}: {cnt}")
        lines.append("")

    # Steps
    s = stats.get("steps", {})
    if s:
        lines.extend([
            "--- Experiment Steps ---",
            f"  Total annotations: {s.get('total_annotations', 0)}",
            "  Step distribution:",
        ])
        for st, cnt in s.get("step_distribution", {}).items():
            lines.append(f"    {st}: {cnt}")
        lines.append("  Status distribution:")
        for st, cnt in s.get("status_distribution", {}).items():
            lines.append(f"    {st}: {cnt}")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="ASTRA Dataset Statistics Tool",
    )
    parser.add_argument("--dataset", required=True, help="Path to dataset directory")
    parser.add_argument("--output", help="Save stats to JSON file")

    args = parser.parse_args()

    stats = compute_stats(args.dataset)

    print(format_stats(stats))

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Statistics saved to {args.output}")


if __name__ == "__main__":
    main()
