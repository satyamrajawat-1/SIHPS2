"""
ASTRA — Evaluation Dataset Loader

Loads annotated datasets for evaluation.
Enforces session/video/person-aware splits — NEVER random frame splits.

Supports:
  - Real annotated datasets (COCO-style, JSONL)
  - Synthetic/mock datasets for pipeline validation
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvalSample:
    """A single evaluation sample (frame or segment)."""
    sample_id: str
    session_id: str
    video_id: str
    frame_id: int = 0
    timestamp: float = 0.0
    image_path: Optional[str] = None

    # Ground truth annotations (populated by loaders)
    gt_detections: List[Dict] = field(default_factory=list)
    gt_poses: List[Dict] = field(default_factory=list)
    gt_actions: List[Dict] = field(default_factory=list)
    gt_hoi: List[Dict] = field(default_factory=list)
    gt_object_states: List[Dict] = field(default_factory=list)
    gt_step: Optional[Dict] = None


@dataclass
class EvalDataset:
    """Complete evaluation dataset."""
    name: str
    split: str  # "test", "val"
    samples: List[EvalSample] = field(default_factory=list)

    # Sequence-level annotations
    gt_step_sequence: List[str] = field(default_factory=list)
    gt_step_statuses: Dict[str, str] = field(default_factory=dict)
    gt_anomalies: List[Dict] = field(default_factory=list)
    gt_action_segments: List[Dict] = field(default_factory=list)

    # Metadata
    sessions: List[str] = field(default_factory=list)
    videos: List[str] = field(default_factory=list)
    class_names: List[str] = field(default_factory=list)
    action_names: List[str] = field(default_factory=list)
    step_ids: List[str] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "split": self.split,
            "n_samples": len(self.samples),
            "n_sessions": len(self.sessions),
            "n_videos": len(self.videos),
            "n_classes": len(self.class_names),
            "n_actions": len(self.action_names),
            "n_steps": len(self.step_ids),
            "has_detections": any(s.gt_detections for s in self.samples),
            "has_poses": any(s.gt_poses for s in self.samples),
            "has_actions": bool(self.gt_action_segments),
            "has_step_sequence": bool(self.gt_step_sequence),
            "has_anomalies": bool(self.gt_anomalies),
        }


def load_eval_dataset(
    dataset_dir: str,
    split: str = "test",
    split_file: Optional[str] = None,
) -> EvalDataset:
    """
    Load an annotated evaluation dataset.

    Expected directory structure:
        dataset_dir/
            manifest.json
            split.json
            annotations/
                detections.json
                poses.jsonl
                actions.jsonl
                hoi.jsonl
                object_states.jsonl
                steps.jsonl
                anomalies.jsonl
                step_sequence.json
            frames/  (optional)
    """
    dataset_dir = os.path.abspath(dataset_dir)
    ds = EvalDataset(name=os.path.basename(dataset_dir), split=split)

    if not os.path.isdir(dataset_dir):
        logger.warning(f"Dataset directory not found: {dataset_dir}")
        return ds

    # Load split
    split_sessions = None
    sf = split_file or os.path.join(dataset_dir, "split.json")
    if os.path.exists(sf):
        with open(sf, "r") as f:
            split_data = json.load(f)
        split_sessions = set(split_data.get("split", {}).get(split, []))
        logger.info(f"Split loaded: {split} → {len(split_sessions)} sessions")

    # Load manifest
    manifest_path = None
    for mname in ["manifest.json", "frame_manifest.json"]:
        mp = os.path.join(dataset_dir, mname)
        if os.path.exists(mp):
            manifest_path = mp
            break

    frames_data = []
    if manifest_path:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        frames_data = manifest.get("frames", [])

    # Build samples from manifest
    for fd in frames_data:
        sid = fd.get("session_id", "session_00")
        if split_sessions and sid not in split_sessions:
            continue

        sample = EvalSample(
            sample_id=f"{fd.get('video_id', 'v')}_{fd.get('frame_id', 0)}",
            session_id=sid,
            video_id=fd.get("video_id", "video_00"),
            frame_id=fd.get("frame_id", 0),
            timestamp=fd.get("timestamp", 0.0),
            image_path=fd.get("filename"),
        )
        ds.samples.append(sample)

    ds.sessions = sorted(set(s.session_id for s in ds.samples))
    ds.videos = sorted(set(s.video_id for s in ds.samples))

    # Load annotations
    ann_dir = os.path.join(dataset_dir, "annotations")

    # Detections (COCO-style)
    det_path = os.path.join(ann_dir, "detections.json")
    if os.path.exists(det_path):
        _load_detection_annotations(det_path, ds)

    # Poses (JSONL)
    pose_path = os.path.join(ann_dir, "poses.jsonl")
    if os.path.exists(pose_path):
        _load_jsonl_annotations(pose_path, ds, "gt_poses")

    # Actions (JSONL segments)
    action_path = os.path.join(ann_dir, "actions.jsonl")
    if os.path.exists(action_path):
        _load_action_segments(action_path, ds)

    # HOI (JSONL)
    hoi_path = os.path.join(ann_dir, "hoi.jsonl")
    if os.path.exists(hoi_path):
        _load_jsonl_annotations(hoi_path, ds, "gt_hoi")

    # Object states (JSONL)
    state_path = os.path.join(ann_dir, "object_states.jsonl")
    if os.path.exists(state_path):
        _load_jsonl_annotations(state_path, ds, "gt_object_states")

    # Steps (JSONL)
    step_path = os.path.join(ann_dir, "steps.jsonl")
    if os.path.exists(step_path):
        _load_step_annotations(step_path, ds)

    # Anomalies (JSONL)
    anomaly_path = os.path.join(ann_dir, "anomalies.jsonl")
    if os.path.exists(anomaly_path):
        _load_anomaly_annotations(anomaly_path, ds)

    # Step sequence (JSON)
    seq_path = os.path.join(ann_dir, "step_sequence.json")
    if os.path.exists(seq_path):
        with open(seq_path, "r") as f:
            seq_data = json.load(f)
        ds.gt_step_sequence = seq_data.get("sequence", [])

    logger.info(f"Dataset loaded: {ds.summary()}")
    return ds


def _load_detection_annotations(path: str, ds: EvalDataset):
    """Load COCO-style detection annotations into samples."""
    with open(path, "r") as f:
        data = json.load(f)

    cats = {c["id"]: c["name"] for c in data.get("categories", [])}
    ds.class_names = sorted(cats.values())

    img_to_anns = defaultdict(list)
    for ann in data.get("annotations", []):
        img_to_anns[ann["image_id"]].append({
            "bbox": ann["bbox"],  # [x, y, w, h] COCO format
            "class": cats.get(ann["category_id"], "?"),
            "category_id": ann["category_id"],
        })

    # Map image IDs to samples
    images = {img["id"]: img for img in data.get("images", [])}
    for sample in ds.samples:
        # Try to match by frame_id or filename
        for img_id, img in images.items():
            if (img.get("frame_id") == sample.frame_id or
                    img.get("file_name") == sample.image_path):
                # Convert COCO [x,y,w,h] to [x1,y1,x2,y2]
                for ann in img_to_anns.get(img_id, []):
                    x, y, w, h = ann["bbox"]
                    sample.gt_detections.append({
                        "bbox": [x, y, x + w, y + h],
                        "class": ann["class"],
                    })
                break


def _load_jsonl_annotations(path: str, ds: EvalDataset, attr: str):
    """Load JSONL annotations into matching samples."""
    frame_anns = defaultdict(list)
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ann = json.loads(line)
            fid = ann.get("frame_id", -1)
            frame_anns[fid].append(ann)

    for sample in ds.samples:
        anns = frame_anns.get(sample.frame_id, [])
        setattr(sample, attr, anns)


def _load_action_segments(path: str, ds: EvalDataset):
    """Load temporal action segment annotations."""
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ann = json.loads(line)
            ds.gt_action_segments.append({
                "label": ann.get("action_id", "?"),
                "start": ann.get("start_time", 0),
                "end": ann.get("end_time", 0),
            })
            action_name = ann.get("action_id", "?")
            if action_name not in ds.action_names:
                ds.action_names.append(action_name)


def _load_step_annotations(path: str, ds: EvalDataset):
    """Load step status annotations."""
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ann = json.loads(line)
            step_id = ann.get("step_id", "?")
            status = ann.get("status", "?")
            ds.gt_step_statuses[step_id] = status
            if step_id not in ds.step_ids:
                ds.step_ids.append(step_id)
            if step_id not in ds.gt_step_sequence:
                if status == "completed":
                    ds.gt_step_sequence.append(step_id)


def _load_anomaly_annotations(path: str, ds: EvalDataset):
    """Load anomaly annotations."""
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ann = json.loads(line)
            ds.gt_anomalies.append({
                "type": ann.get("type", "?"),
                "timestamp": ann.get("timestamp", 0),
                "step_id": ann.get("step_id"),
            })


# ============================================================
# Synthetic / Mock Dataset Generator
# ============================================================

def generate_mock_eval_dataset(
    experiment_config,
    n_frames: int = 100,
    seed: int = 42,
) -> EvalDataset:
    """
    Generate a mock evaluation dataset from an ExperimentConfig.
    Used for pipeline validation when no real data exists.
    """
    rng = np.random.RandomState(seed)
    ds = EvalDataset(name="mock_eval", split="test")
    ds.step_ids = [s.id for s in experiment_config.steps]
    ds.class_names = [o.obj_class for o in experiment_config.objects]
    ds.action_names = [a.id for a in experiment_config.actions]

    # Generate frames
    frames_per_step = n_frames // max(1, len(experiment_config.steps))

    for step_idx, step in enumerate(experiment_config.steps):
        for fi in range(frames_per_step):
            global_fi = step_idx * frames_per_step + fi
            t = global_fi / 30.0

            sample = EvalSample(
                sample_id=f"mock_{global_fi}",
                session_id="session_mock",
                video_id="video_mock",
                frame_id=global_fi,
                timestamp=t,
            )

            # Generate mock detections
            for obj in experiment_config.objects:
                if obj.id in step.required_objects or rng.random() > 0.3:
                    x1, y1 = rng.randint(50, 400, 2)
                    sample.gt_detections.append({
                        "bbox": [float(x1), float(y1), float(x1 + 80), float(y1 + 60)],
                        "class": obj.obj_class,
                    })

            # Generate mock pose
            kps = rng.rand(17, 3).astype(float)
            kps[:, 0] *= 640
            kps[:, 1] *= 480
            kps[:, 2] = rng.uniform(0.5, 1.0, 17)
            sample.gt_poses = [{"keypoints_2d": kps.tolist(), "person_id": 0}]

            # Generate mock step annotation
            sample.gt_step = {"step_id": step.id, "status": "in_progress"}

            ds.samples.append(sample)

        # Step completed
        ds.gt_step_statuses[step.id] = "VALID"
        ds.gt_step_sequence.append(step.id)

    # Generate action segments
    for step_idx, step in enumerate(experiment_config.steps):
        if step.required_actions:
            t_start = step_idx * frames_per_step / 30.0
            t_end = (step_idx + 1) * frames_per_step / 30.0
            ds.gt_action_segments.append({
                "label": step.required_actions[-1],
                "start": t_start,
                "end": t_end,
            })

    ds.sessions = ["session_mock"]
    ds.videos = ["video_mock"]

    logger.info(f"Mock dataset generated: {len(ds.samples)} samples, "
                f"{len(ds.step_ids)} steps")
    return ds
