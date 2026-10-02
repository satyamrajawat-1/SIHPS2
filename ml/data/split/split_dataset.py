"""
ASTRA — Dataset Split Tool

Split datasets by session/person/video (NOT by individual frame).
This prevents temporal leakage.

Usage:
    python -m ml.data.split.split_dataset --dataset data/datasets/sample/ --output data/datasets/sample/
    python -m ml.data.split.split_dataset --dataset data/datasets/sample/ --strategy session --ratios 0.7 0.15 0.15
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_manifest(dataset_dir: str) -> Dict:
    """Load dataset manifest."""
    manifest_path = os.path.join(dataset_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        # Try frame manifest
        manifest_path = os.path.join(dataset_dir, "frame_manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"No manifest found in {dataset_dir}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_by_session(
    sessions: List[str],
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> Dict[str, List[str]]:
    """
    Split session IDs into train/val/test sets.

    Args:
        sessions: List of session IDs.
        ratios: (train, val, test) ratios.
        seed: Random seed for reproducibility.

    Returns:
        Dict with 'train', 'val', 'test' keys mapping to session ID lists.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, f"Ratios must sum to 1.0, got {sum(ratios)}"

    rng = random.Random(seed)
    sessions = sorted(sessions)  # Deterministic ordering
    rng.shuffle(sessions)

    n = len(sessions)
    n_train = max(1, int(n * ratios[0]))
    n_val = max(1, int(n * ratios[1])) if n > 2 else 0
    # Remaining goes to test
    n_test = n - n_train - n_val

    if n_test < 0:
        n_val = n - n_train
        n_test = 0

    split = {
        "train": sessions[:n_train],
        "val": sessions[n_train:n_train + n_val],
        "test": sessions[n_train + n_val:],
    }

    return split


def split_by_video(
    videos: List[Dict],
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> Dict[str, List[str]]:
    """
    Split video IDs into train/val/test sets.

    Args:
        videos: List of video dicts with 'video_id' key.
        ratios: (train, val, test) ratios.
        seed: Random seed.

    Returns:
        Dict with split assignments.
    """
    video_ids = sorted(set(v.get("video_id", str(i)) for i, v in enumerate(videos)))
    return split_by_session(video_ids, ratios, seed)


def split_by_person(
    person_ids: List[str],
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> Dict[str, List[str]]:
    """
    Split person IDs into train/val/test sets (leave-one-out style).

    Args:
        person_ids: List of person identifiers.
        ratios: (train, val, test) ratios.
        seed: Random seed.

    Returns:
        Dict with split assignments.
    """
    return split_by_session(sorted(set(person_ids)), ratios, seed)


def split_by_camera(
    camera_ids: List[str],
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> Dict[str, List[str]]:
    """
    Split camera IDs into train/val/test sets.

    For multi-camera evaluation: train on some cameras, test on others.

    Args:
        camera_ids: List of camera identifiers.
        ratios: (train, val, test) ratios.
        seed: Random seed.

    Returns:
        Dict with split assignments.
    """
    return split_by_session(sorted(set(camera_ids)), ratios, seed)


def assign_frames_to_split(
    frames: List[Dict],
    split: Dict[str, List[str]],
    split_key: str = "session_id",
) -> Dict[str, List[Dict]]:
    """
    Assign individual frames to splits based on their group (session/video/person).

    Args:
        frames: List of frame dicts.
        split: Dict mapping split names to lists of group IDs.
        split_key: Key in frame dict to match against split groups.

    Returns:
        Dict mapping split names to lists of frames.
    """
    # Build reverse lookup
    group_to_split = {}
    for split_name, group_ids in split.items():
        for gid in group_ids:
            group_to_split[gid] = split_name

    result = {"train": [], "val": [], "test": [], "unassigned": []}

    for frame in frames:
        group_id = frame.get(split_key, "unknown")
        split_name = group_to_split.get(group_id, "unassigned")
        result[split_name].append(frame)

    if result["unassigned"]:
        logger.warning(
            f"{len(result['unassigned'])} frames not assigned to any split"
        )
    else:
        del result["unassigned"]

    return result


def save_split(
    split: Dict[str, List[str]],
    output_dir: str,
    split_name: str = "split",
    metadata: Optional[Dict] = None,
) -> str:
    """Save split assignments to JSON."""
    os.makedirs(output_dir, exist_ok=True)

    split_data = {
        "split_name": split_name,
        "strategy": metadata.get("strategy", "unknown") if metadata else "unknown",
        "seed": metadata.get("seed", 42) if metadata else 42,
        "ratios": metadata.get("ratios", [0.7, 0.15, 0.15]) if metadata else [0.7, 0.15, 0.15],
        "split": split,
        "counts": {k: len(v) for k, v in split.items()},
    }

    output_path = os.path.join(output_dir, f"{split_name}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(split_data, f, indent=2)

    logger.info(f"Split saved to {output_path}")
    for k, v in split.items():
        logger.info(f"  {k}: {len(v)} groups")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="ASTRA Dataset Split Tool — Split by session/video/person to prevent temporal leakage",
    )
    parser.add_argument("--dataset", required=True, help="Path to dataset directory")
    parser.add_argument("--output", help="Output directory (default: same as dataset)")
    parser.add_argument(
        "--strategy",
        choices=["session", "video", "person", "camera"],
        default="session",
        help="Split strategy",
    )
    parser.add_argument(
        "--ratios",
        nargs=3,
        type=float,
        default=[0.7, 0.15, 0.15],
        help="Train/val/test ratios (must sum to 1.0)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    output_dir = args.output or args.dataset
    ratios = tuple(args.ratios)

    logger.info(f"Splitting dataset: {args.dataset}")
    logger.info(f"Strategy: {args.strategy}, Ratios: {ratios}, Seed: {args.seed}")

    # Load manifest
    try:
        manifest = load_manifest(args.dataset)
    except FileNotFoundError:
        logger.error(f"No manifest found in {args.dataset}")
        logger.info("Generate a manifest first using the extraction tools.")
        return

    # Extract group IDs based on strategy
    frames = manifest.get("frames", [])
    if not frames:
        logger.warning("No frames found in manifest. Creating demo split.")
        # Create a demo split from sessions if available
        sessions = manifest.get("sessions", [])
        if sessions:
            session_ids = [s.get("session_id", f"session_{i}") for i, s in enumerate(sessions)]
        else:
            session_ids = ["session_00"]
        split = split_by_session(session_ids, ratios, args.seed)
    else:
        if args.strategy == "session":
            group_ids = sorted(set(f.get("session_id", "unknown") for f in frames))
            split = split_by_session(group_ids, ratios, args.seed)
        elif args.strategy == "video":
            split = split_by_video(frames, ratios, args.seed)
        elif args.strategy == "person":
            person_ids = sorted(set(f.get("person_id", "unknown") for f in frames))
            split = split_by_person(person_ids, ratios, args.seed)
        elif args.strategy == "camera":
            camera_ids = sorted(set(f.get("camera_id", "unknown") for f in frames))
            split = split_by_camera(camera_ids, ratios, args.seed)

    metadata = {
        "strategy": args.strategy,
        "seed": args.seed,
        "ratios": list(ratios),
    }

    save_split(split, output_dir, split_name="split", metadata=metadata)


if __name__ == "__main__":
    main()
