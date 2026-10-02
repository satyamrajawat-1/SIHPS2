"""
ASTRA — Model Interfaces (Stable API Contracts)

Every model in ASTRA must implement one of these interfaces.
This ensures models can be replaced independently without breaking the pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ============================================================
# Common Output Types
# ============================================================

@dataclass
class Detection:
    """A single object detection."""
    object_id: int  # Tracking ID
    class_name: str
    class_id: int
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    frame_id: int = 0
    timestamp: float = 0.0
    camera_id: str = "cam_00"


@dataclass
class PoseResult:
    """Pose estimation result for a single person."""
    person_id: int
    bbox: List[float]  # [x1, y1, x2, y2]
    keypoints_2d: np.ndarray  # (J, 3) — [x, y, confidence] per joint
    hand_keypoints_left: Optional[np.ndarray] = None  # (21, 3)
    hand_keypoints_right: Optional[np.ndarray] = None  # (21, 3)
    tracking_confidence: float = 1.0
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "person_id": self.person_id,
            "bbox": self.bbox,
            "keypoints_2d": self.keypoints_2d.tolist(),
            "tracking_confidence": self.tracking_confidence,
            "timestamp": self.timestamp,
        }
        if self.hand_keypoints_left is not None:
            d["hand_keypoints_left"] = self.hand_keypoints_left.tolist()
        if self.hand_keypoints_right is not None:
            d["hand_keypoints_right"] = self.hand_keypoints_right.tolist()
        return d


@dataclass
class ActionResult:
    """Action recognition result."""
    action_id: str
    action_name: str
    confidence: float
    person_id: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    all_scores: Optional[Dict[str, float]] = None


@dataclass
class HOIResult:
    """Hand-object interaction result."""
    hand: str  # "left" or "right"
    object_id: str
    object_class: str
    interaction_type: str  # "grasp", "reach", "hold", etc.
    confidence: float
    contact_probability: float = 0.0
    person_id: int = 0


@dataclass
class ObjectStateResult:
    """Object state observation."""
    object_id: str
    object_class: str
    current_state: str
    previous_state: Optional[str] = None
    state_confidence: float = 0.0
    is_transition: bool = False
    timestamp: float = 0.0


@dataclass
class TemporalResult:
    """Temporal model output."""
    current_action: Optional[str] = None
    current_action_confidence: float = 0.0
    predicted_next_action: Optional[str] = None
    next_action_confidence: float = 0.0
    current_step_candidate: Optional[str] = None
    step_confidence: float = 0.0
    anomaly_score: float = 0.0
    embeddings: Optional[np.ndarray] = None
    supporting_evidence: Optional[Dict[str, float]] = None


@dataclass
class FusedEvidence:
    """Fused evidence from all perception sources."""
    step_id: Optional[str] = None
    validity_score: float = 0.0
    confidence: float = 0.0
    uncertainty: float = 1.0
    evidence_breakdown: Dict[str, float] = field(default_factory=dict)
    status: str = "UNCERTAIN"  # "VALID", "UNCERTAIN", "INVALID"


# ============================================================
# Model Interface ABCs
# ============================================================

class ObjectDetector(ABC):
    """Interface for object detection + tracking models."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect objects in a single frame.

        Args:
            frame: BGR image (H, W, 3).

        Returns:
            List of Detection objects with tracking IDs.
        """
        ...

    @abstractmethod
    def load_model(self, checkpoint: str, device: str = "cpu"):
        """Load model weights."""
        ...

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata."""
        return {"name": self.__class__.__name__, "version": "unknown"}


class PoseEstimator(ABC):
    """Interface for human pose estimation models."""

    @abstractmethod
    def predict(self, frame: np.ndarray, detections: Optional[List[Detection]] = None) -> List[PoseResult]:
        """
        Estimate pose for detected persons.

        Args:
            frame: BGR image (H, W, 3).
            detections: Optional person detections for top-down pose.

        Returns:
            List of PoseResult objects.
        """
        ...

    @abstractmethod
    def load_model(self, checkpoint: str, device: str = "cpu"):
        """Load model weights."""
        ...


class ActionRecognizer(ABC):
    """Interface for skeleton-based action recognition models."""

    @abstractmethod
    def predict(self, skeleton_sequence: np.ndarray) -> ActionResult:
        """
        Recognize action from a skeleton sequence.

        Args:
            skeleton_sequence: (T, J, C) array.

        Returns:
            ActionResult with action class and confidence.
        """
        ...

    @abstractmethod
    def load_model(self, checkpoint: str, device: str = "cpu"):
        """Load model weights."""
        ...


class HOIModel(ABC):
    """Interface for hand-object interaction models."""

    @abstractmethod
    def predict(
        self,
        pose: PoseResult,
        detections: List[Detection],
        frame: Optional[np.ndarray] = None,
    ) -> List[HOIResult]:
        """
        Predict hand-object interactions.

        Args:
            pose: Pose result with hand keypoints.
            detections: Object detections in the same frame.
            frame: Optional frame for visual features.

        Returns:
            List of HOIResult objects.
        """
        ...


class ObjectStateModel(ABC):
    """Interface for object state recognition."""

    @abstractmethod
    def predict(
        self,
        detections: List[Detection],
        frame: Optional[np.ndarray] = None,
        hoi_results: Optional[List[HOIResult]] = None,
    ) -> List[ObjectStateResult]:
        """
        Predict object states.

        Args:
            detections: Object detections.
            frame: Optional frame for visual features.
            hoi_results: Optional HOI context.

        Returns:
            List of ObjectStateResult objects.
        """
        ...


class TemporalModel(ABC):
    """Interface for temporal multimodal reasoning models."""

    @abstractmethod
    def predict(self, feature_sequence: np.ndarray) -> TemporalResult:
        """
        Process a temporal sequence of multimodal features.

        Args:
            feature_sequence: (T, D) array of fused features.

        Returns:
            TemporalResult with predictions.
        """
        ...

    @abstractmethod
    def load_model(self, checkpoint: str, device: str = "cpu"):
        """Load model weights."""
        ...


class VisualFeatureExtractor(ABC):
    """Interface for visual feature extraction models."""

    @abstractmethod
    def extract(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract visual features from a frame.

        Args:
            frame: BGR image (H, W, 3).

        Returns:
            Feature vector (D,).
        """
        ...

    @abstractmethod
    def load_model(self, checkpoint: str, device: str = "cpu"):
        """Load model weights."""
        ...


class EvidenceFusion(ABC):
    """Interface for multi-source evidence fusion."""

    @abstractmethod
    def fuse(self, evidence: Dict[str, Any]) -> FusedEvidence:
        """
        Fuse evidence from multiple perception sources.

        Args:
            evidence: Dict with scores from each source.

        Returns:
            FusedEvidence with aggregated assessment.
        """
        ...
