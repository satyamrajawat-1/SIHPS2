"""
ASTRA — Skeleton Action Dataset Generator

Generates synthetic skeleton sequences for training ST-GCN++ on the
5-action vocabulary: idle, reach, pick, move, place, release.

Each sample is a temporal window of T frames × J joints × C channels.

Uses kinematically plausible trajectories (not random noise):
- reach: hand moves toward object position
- pick: hand closes, object starts moving up
- move: hand + object translate laterally
- place: hand + object move downward
- release: hand retracts from object
- idle: minimal movement
"""
from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# COCO-17 skeleton topology
COCO_JOINTS = 17
COCO_JOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Action class mapping
ACTION_CLASSES = {
    "idle": 0,
    "reach": 1,
    "pick": 2,
    "move": 3,
    "place": 4,
    "release": 5,
}
NUM_CLASSES = len(ACTION_CLASSES)


def generate_base_skeleton(height: float = 0.5, center_x: float = 0.5) -> np.ndarray:
    """Generate a standing COCO-17 skeleton. Returns (17, 2) array."""
    cx = center_x
    # Approximate normalized positions (0-1 range)
    skeleton = np.array([
        [cx, 0.15],          # nose
        [cx - 0.02, 0.13],   # left_eye
        [cx + 0.02, 0.13],   # right_eye
        [cx - 0.04, 0.15],   # left_ear
        [cx + 0.04, 0.15],   # right_ear
        [cx - 0.10, 0.25],   # left_shoulder
        [cx + 0.10, 0.25],   # right_shoulder
        [cx - 0.15, 0.35],   # left_elbow
        [cx + 0.15, 0.35],   # right_elbow
        [cx - 0.18, 0.45],   # left_wrist
        [cx + 0.18, 0.45],   # right_wrist
        [cx - 0.08, 0.50],   # left_hip
        [cx + 0.08, 0.50],   # right_hip
        [cx - 0.08, 0.65],   # left_knee
        [cx + 0.08, 0.65],   # right_knee
        [cx - 0.08, 0.80],   # left_ankle
        [cx + 0.08, 0.80],   # right_ankle
    ], dtype=np.float32)
    return skeleton


def generate_action_sequence(
    action: str,
    n_frames: int = 30,
    hand: str = "right",  # which hand performs the action
    noise_std: float = 0.005,
    seed: int = None,
) -> np.ndarray:
    """
    Generate a kinematically plausible skeleton sequence for a given action.

    Returns: (T, J, C) where C = [x, y, confidence]
    """
    if seed is not None:
        np.random.seed(seed)

    wrist_idx = 10 if hand == "right" else 9  # right_wrist or left_wrist
    elbow_idx = 8 if hand == "right" else 7

    base = generate_base_skeleton()
    object_pos = np.array([0.7, 0.45])  # object position (right side of frame)
    target_pos = np.array([0.3, 0.45])  # target zone (left side)

    frames = np.zeros((n_frames, COCO_JOINTS, 3), dtype=np.float32)

    for t in range(n_frames):
        progress = t / max(n_frames - 1, 1)
        skeleton = base.copy()

        if action == "idle":
            # Minimal movement — just small sway
            skeleton += np.random.randn(COCO_JOINTS, 2) * 0.003

        elif action == "reach":
            # Hand moves from resting position toward object
            rest_pos = base[wrist_idx].copy()
            skeleton[wrist_idx] = rest_pos + (object_pos - rest_pos) * progress
            # Elbow follows partially
            skeleton[elbow_idx] = base[elbow_idx] + (skeleton[wrist_idx] - base[wrist_idx]) * 0.5

        elif action == "pick":
            # Hand is near object, slight upward movement
            skeleton[wrist_idx] = object_pos + np.array([0, -0.05 * progress])
            skeleton[elbow_idx] = base[elbow_idx] + (skeleton[wrist_idx] - base[wrist_idx]) * 0.5

        elif action == "move":
            # Hand + object translate from object_pos toward target_pos
            current = object_pos + (target_pos - object_pos) * progress
            current[1] -= 0.05  # slightly elevated
            skeleton[wrist_idx] = current
            skeleton[elbow_idx] = base[elbow_idx] + (skeleton[wrist_idx] - base[wrist_idx]) * 0.5

        elif action == "place":
            # Hand near target_pos, moving downward
            skeleton[wrist_idx] = target_pos + np.array([0, -0.05 * (1 - progress)])
            skeleton[elbow_idx] = base[elbow_idx] + (skeleton[wrist_idx] - base[wrist_idx]) * 0.5

        elif action == "release":
            # Hand retracts from target_pos back toward rest
            rest_pos = base[wrist_idx].copy()
            skeleton[wrist_idx] = target_pos + (rest_pos - target_pos) * progress
            skeleton[elbow_idx] = base[elbow_idx] + (skeleton[wrist_idx] - base[wrist_idx]) * 0.5

        # Add noise to all joints
        skeleton += np.random.randn(COCO_JOINTS, 2) * noise_std

        # Set confidence (high for all joints)
        conf = np.ones((COCO_JOINTS, 1), dtype=np.float32) * (0.85 + np.random.rand() * 0.15)

        frames[t, :, :2] = skeleton
        frames[t, :, 2:] = conf

    return frames


def augment_skeleton(sequence: np.ndarray, seed: int = None) -> np.ndarray:
    """Apply augmentation to skeleton sequence. Does NOT change action label."""
    if seed is not None:
        np.random.seed(seed)

    aug = sequence.copy()
    T, J, C = aug.shape

    # Random horizontal flip (50% chance)
    if np.random.rand() > 0.5:
        aug[:, :, 0] = 1.0 - aug[:, :, 0]
        # Swap left/right joint pairs
        pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
        for l, r in pairs:
            aug[:, [l, r]] = aug[:, [r, l]]

    # Random scale (0.8 - 1.2)
    scale = 0.8 + np.random.rand() * 0.4
    center = aug[:, :, :2].mean(axis=(0, 1), keepdims=True)
    aug[:, :, :2] = center + (aug[:, :, :2] - center) * scale

    # Random translate
    tx = (np.random.rand() - 0.5) * 0.1
    ty = (np.random.rand() - 0.5) * 0.1
    aug[:, :, 0] += tx
    aug[:, :, 1] += ty

    # Random temporal speed (resample to same length)
    if np.random.rand() > 0.5:
        speed = 0.8 + np.random.rand() * 0.4
        new_len = int(T * speed)
        if new_len > 2:
            indices = np.linspace(0, new_len - 1, T).astype(int)
            indices = np.clip(indices, 0, new_len - 1)
            # Resample from temporary stretched version
            temp_indices = np.linspace(0, T - 1, new_len).astype(int)
            temp = aug[temp_indices]
            aug = temp[indices]

    return aug


def generate_action_dataset(
    output_dir: str,
    n_samples_per_class: int = 100,
    n_frames: int = 30,
    n_augmentations: int = 3,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict:
    """
    Generate complete action recognition dataset.

    Output structure:
        data/action/
        ├── train/
        │   ├── samples.npz       # (N, T, J, C) arrays
        │   └── labels.json       # metadata
        ├── val/
        │   ├── samples.npz
        │   └── labels.json
        ├── test/
        │   ├── samples.npz
        │   └── labels.json
        └── dataset_info.json
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(seed)

    all_sequences = []
    all_labels = []
    all_session_ids = []
    all_metadata = []

    session_counter = 0

    for action_name, class_id in ACTION_CLASSES.items():
        for i in range(n_samples_per_class):
            session_id = f"session_{session_counter:04d}"
            session_counter += 1

            # Vary parameters
            hand = rng.choice(["right", "left"])
            noise = 0.003 + rng.rand() * 0.008
            n_f = n_frames + rng.randint(-5, 6)
            n_f = max(10, n_f)

            # Generate base sequence
            seq = generate_action_sequence(
                action=action_name,
                n_frames=n_f,
                hand=hand,
                noise_std=noise,
                seed=rng.randint(0, 100000),
            )

            # Pad/trim to fixed length
            seq = _pad_or_trim(seq, n_frames)

            all_sequences.append(seq)
            all_labels.append(class_id)
            all_session_ids.append(session_id)
            all_metadata.append({
                "action": action_name,
                "class_id": class_id,
                "session_id": session_id,
                "hand": hand,
                "n_original_frames": n_f,
                "is_augmented": False,
            })

            # Generate augmented versions
            for aug_i in range(n_augmentations):
                aug_seq = augment_skeleton(seq, seed=rng.randint(0, 100000))
                aug_session = f"{session_id}_aug{aug_i}"

                all_sequences.append(aug_seq)
                all_labels.append(class_id)
                all_session_ids.append(aug_session)
                all_metadata.append({
                    "action": action_name,
                    "class_id": class_id,
                    "session_id": aug_session,
                    "hand": hand,
                    "n_original_frames": n_f,
                    "is_augmented": True,
                    "augmentation_index": aug_i,
                    "parent_session": session_id,
                })

    # Convert to arrays
    sequences = np.array(all_sequences, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int64)

    logger.info(f"Generated {len(sequences)} samples: {sequences.shape}")
    logger.info(f"  Per class: {n_samples_per_class} base + {n_augmentations} aug each")

    # Session-aware split
    unique_base_sessions = sorted(set(
        m["parent_session"] if m["is_augmented"] else m["session_id"]
        for m in all_metadata
    ))
    rng.shuffle(unique_base_sessions)

    n_val = max(1, int(len(unique_base_sessions) * val_ratio))
    n_test = max(1, int(len(unique_base_sessions) * test_ratio))
    n_train = len(unique_base_sessions) - n_val - n_test

    train_sessions = set(unique_base_sessions[:n_train])
    val_sessions = set(unique_base_sessions[n_train:n_train + n_val])
    test_sessions = set(unique_base_sessions[n_train + n_val:])

    # Assign split
    train_idx, val_idx, test_idx = [], [], []
    for i, meta in enumerate(all_metadata):
        base = meta["parent_session"] if meta["is_augmented"] else meta["session_id"]
        if base in train_sessions:
            train_idx.append(i)
        elif base in val_sessions:
            val_idx.append(i)
        else:
            test_idx.append(i)

    # Save splits
    for split_name, indices in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        split_seqs = sequences[indices]
        split_labels = labels[indices]
        split_meta = [all_metadata[i] for i in indices]

        np.savez_compressed(
            os.path.join(split_dir, "samples.npz"),
            sequences=split_seqs,
            labels=split_labels,
        )

        with open(os.path.join(split_dir, "labels.json"), "w") as f:
            json.dump({
                "samples": split_meta,
                "class_map": ACTION_CLASSES,
                "n_samples": len(indices),
            }, f, indent=2)

        logger.info(f"  {split_name}: {len(indices)} samples")

    # Save dataset info
    info = {
        "name": "astra_action_v1",
        "version": "1.0",
        "data_type": "SYNTHETIC",
        "n_total": len(sequences),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "n_classes": NUM_CLASSES,
        "classes": ACTION_CLASSES,
        "n_joints": COCO_JOINTS,
        "n_frames": n_frames,
        "channels": ["x", "y", "confidence"],
        "tensor_shape": f"({n_frames}, {COCO_JOINTS}, 3)",
        "split_method": "session_aware",
        "n_base_sessions": len(unique_base_sessions),
        "augmentations_per_sample": n_augmentations,
        "seed": seed,
    }

    with open(os.path.join(output_dir, "dataset_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    logger.info(f"Dataset saved to: {output_dir}")
    return info


def _pad_or_trim(seq: np.ndarray, target_len: int) -> np.ndarray:
    """Pad or trim sequence to target_len frames."""
    T = seq.shape[0]
    if T == target_len:
        return seq
    elif T > target_len:
        start = (T - target_len) // 2
        return seq[start:start + target_len]
    else:
        # Pad by repeating last frame
        pad = np.repeat(seq[-1:], target_len - T, axis=0)
        return np.concatenate([seq, pad], axis=0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    info = generate_action_dataset(
        output_dir="data/action",
        n_samples_per_class=100,
        n_frames=30,
        n_augmentations=3,
        seed=42,
    )
    print(f"\nDataset generated: {info['n_total']} samples ({info['n_classes']} classes)")
    print(f"  Train: {info['n_train']}, Val: {info['n_val']}, Test: {info['n_test']}")
