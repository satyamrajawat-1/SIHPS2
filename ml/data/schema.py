"""
ASTRA Dataset Schema — Annotation Format Definitions

Every annotation in the ASTRA dataset follows these standardized schemas.
Formats: COCO-style JSON for detections, JSONL for temporal annotations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# Common Metadata
# ============================================================

@dataclass
class FrameMetadata:
    """Metadata for a single video frame."""
    experiment_id: str
    session_id: str
    video_id: str
    camera_id: str
    frame_id: int
    timestamp: float  # seconds from video start
    frame_width: int
    frame_height: int
    annotation_source: str = "manual"  # "manual", "model", "synthetic"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "session_id": self.session_id,
            "video_id": self.video_id,
            "camera_id": self.camera_id,
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "annotation_source": self.annotation_source,
        }


# ============================================================
# A. Object Detection Annotations (COCO-style)
# ============================================================

@dataclass
class BBoxAnnotation:
    """COCO-style bounding box annotation."""
    annotation_id: int
    image_id: int  # maps to frame_id
    category_id: int
    category_name: str
    bbox: List[float]  # [x, y, width, height] in pixels
    area: float
    confidence: float = 1.0
    track_id: Optional[int] = None  # object tracking ID across frames
    is_crowd: int = 0

    def to_coco_dict(self) -> Dict[str, Any]:
        return {
            "id": self.annotation_id,
            "image_id": self.image_id,
            "category_id": self.category_id,
            "bbox": self.bbox,
            "area": self.area,
            "iscrowd": self.is_crowd,
            "attributes": {
                "track_id": self.track_id,
                "confidence": self.confidence,
            },
        }


# ============================================================
# B. Pose Keypoint Annotations
# ============================================================

@dataclass
class PoseAnnotation:
    """Human pose keypoint annotation."""
    annotation_id: int
    frame_id: int
    timestamp: float
    person_id: int
    bbox: List[float]  # person bounding box [x, y, w, h]
    keypoints_2d: List[List[float]]  # [[x, y, conf], ...] — body keypoints
    hand_keypoints_left: Optional[List[List[float]]] = None  # [[x, y, conf], ...]
    hand_keypoints_right: Optional[List[List[float]]] = None
    keypoint_scores: Optional[List[float]] = None
    tracking_confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "annotation_id": self.annotation_id,
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "person_id": self.person_id,
            "bbox": self.bbox,
            "keypoints_2d": self.keypoints_2d,
            "tracking_confidence": self.tracking_confidence,
        }
        if self.hand_keypoints_left is not None:
            d["hand_keypoints_left"] = self.hand_keypoints_left
        if self.hand_keypoints_right is not None:
            d["hand_keypoints_right"] = self.hand_keypoints_right
        if self.keypoint_scores is not None:
            d["keypoint_scores"] = self.keypoint_scores
        return d


# ============================================================
# C. Action Annotations (Temporal Segments)
# ============================================================

@dataclass
class ActionAnnotation:
    """Temporal action segment annotation."""
    annotation_id: int
    video_id: str
    session_id: str
    person_id: int
    action_id: str
    action_name: str
    start_frame: int
    end_frame: int
    start_time: float  # seconds
    end_time: float
    confidence: float = 1.0
    annotation_source: str = "manual"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "annotation_id": self.annotation_id,
            "video_id": self.video_id,
            "session_id": self.session_id,
            "person_id": self.person_id,
            "action_id": self.action_id,
            "action_name": self.action_name,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "confidence": self.confidence,
            "annotation_source": self.annotation_source,
        }


# ============================================================
# D. Hand-Object Interaction Annotations
# ============================================================

class InteractionType(str, Enum):
    NONE = "none"
    REACH = "reach"
    GRASP = "grasp"
    HOLD = "hold"
    RELEASE = "release"
    USE = "use"
    CONTACT = "contact"


@dataclass
class HOIAnnotation:
    """Hand-object interaction annotation."""
    annotation_id: int
    frame_id: int
    timestamp: float
    person_id: int
    hand: str  # "left", "right"
    object_id: str
    object_class: str
    interaction_type: str  # InteractionType value
    contact: bool
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "annotation_id": self.annotation_id,
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "person_id": self.person_id,
            "hand": self.hand,
            "object_id": self.object_id,
            "object_class": self.object_class,
            "interaction_type": self.interaction_type,
            "contact": self.contact,
            "confidence": self.confidence,
        }


# ============================================================
# E. Object State Annotations
# ============================================================

@dataclass
class ObjectStateAnnotation:
    """Object state annotation at a point in time."""
    annotation_id: int
    frame_id: int
    timestamp: float
    object_id: str
    object_class: str
    state: str
    previous_state: Optional[str] = None
    is_transition: bool = False
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "annotation_id": self.annotation_id,
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "object_id": self.object_id,
            "object_class": self.object_class,
            "state": self.state,
            "previous_state": self.previous_state,
            "is_transition": self.is_transition,
            "confidence": self.confidence,
        }


# ============================================================
# F. Experiment Step Annotations
# ============================================================

@dataclass
class StepAnnotation:
    """Experiment step annotation (temporal segment)."""
    annotation_id: int
    video_id: str
    session_id: str
    experiment_id: str
    step_id: str
    step_name: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    status: str  # "completed", "skipped", "failed", "partial"
    confidence: float = 1.0
    annotation_source: str = "manual"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "annotation_id": self.annotation_id,
            "video_id": self.video_id,
            "session_id": self.session_id,
            "experiment_id": self.experiment_id,
            "step_id": self.step_id,
            "step_name": self.step_name,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "status": self.status,
            "confidence": self.confidence,
            "annotation_source": self.annotation_source,
        }


# ============================================================
# G. Anomaly Annotations
# ============================================================

@dataclass
class AnomalyAnnotation:
    """Anomaly annotation for unexpected events."""
    annotation_id: int
    video_id: str
    session_id: str
    frame_id: int
    timestamp: float
    anomaly_type: str  # "skipped_step", "out_of_order", "unexpected_action", etc.
    description: str
    related_step_id: Optional[str] = None
    severity: str = "warning"  # "info", "warning", "critical"
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "annotation_id": self.annotation_id,
            "video_id": self.video_id,
            "session_id": self.session_id,
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "anomaly_type": self.anomaly_type,
            "description": self.description,
            "related_step_id": self.related_step_id,
            "severity": self.severity,
            "confidence": self.confidence,
        }


# ============================================================
# H. Camera Metadata
# ============================================================

@dataclass
class CameraMetadata:
    """Camera configuration and calibration data."""
    camera_id: str
    name: str
    resolution: Tuple[int, int]  # (width, height)
    fps: float
    intrinsics: Optional[Dict[str, Any]] = None  # fx, fy, cx, cy, distortion
    extrinsics: Optional[Dict[str, Any]] = None  # rotation, translation
    position_description: str = ""  # e.g., "overhead", "frontal", "side"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "name": self.name,
            "resolution": list(self.resolution),
            "fps": self.fps,
            "intrinsics": self.intrinsics,
            "extrinsics": self.extrinsics,
            "position_description": self.position_description,
        }


# ============================================================
# I. Dataset Manifest
# ============================================================

@dataclass
class DatasetManifest:
    """Top-level manifest for an ASTRA dataset."""
    dataset_id: str
    dataset_name: str
    version: str
    description: str
    experiment_id: str
    creation_date: str
    sessions: List[Dict[str, Any]]  # [{session_id, videos: [...]}]
    cameras: List[CameraMetadata]
    split: Optional[Dict[str, List[str]]] = None  # {"train": [...], "val": [...], "test": [...]}
    statistics: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "version": self.version,
            "description": self.description,
            "experiment_id": self.experiment_id,
            "creation_date": self.creation_date,
            "sessions": self.sessions,
            "cameras": [c.to_dict() for c in self.cameras],
            "split": self.split,
            "statistics": self.statistics,
        }
