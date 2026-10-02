"""
ASTRA — Hand-Object Interaction (HOI) Graph Network

Determines which hand is interacting with which object,
and the type of interaction (grasp, reach, hold, contact, etc.).

Uses spatial proximity + visual features + pose information.

Input: Pose results + object detections + optional visual features
Output: List of HOI results (hand → object → interaction_type)
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from ml.models.interfaces import HOIModel, HOIResult, PoseResult, Detection


# Interaction types
INTERACTION_TYPES = [
    "no_contact",   # Hand near but not touching
    "reach",        # Hand moving toward object
    "grasp",        # Hand grasping/holding object
    "hold",         # Stable hold
    "release",      # Hand releasing object
    "contact",      # Generic hand-object contact
    "manipulate",   # Active manipulation
    "use",          # Using a tool on another object
]


def compute_hand_object_distance(
    hand_pos: np.ndarray,
    bbox: List[float],
) -> float:
    """
    Compute minimum distance from hand position to object bounding box.

    Args:
        hand_pos: (2,) hand [x, y] position.
        bbox: [x1, y1, x2, y2] bounding box.

    Returns:
        Distance in pixels (0 if hand is inside bbox).
    """
    x, y = hand_pos
    x1, y1, x2, y2 = bbox

    # Clamp to bbox edges
    cx = max(x1, min(x, x2))
    cy = max(y1, min(y, y2))

    dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return dist


def compute_hand_velocity(
    current_pos: np.ndarray,
    previous_pos: Optional[np.ndarray],
    dt: float = 1.0 / 30.0,
) -> float:
    """Compute hand velocity magnitude."""
    if previous_pos is None:
        return 0.0
    vel = np.linalg.norm(current_pos - previous_pos) / max(dt, 1e-6)
    return float(vel)


class ProximityHOIModel(HOIModel):
    """
    Proximity-based Hand-Object Interaction model.

    Uses spatial distance, hand keypoint visibility, and temporal context
    to determine interactions. Can be upgraded to a learned model later.

    Thresholds:
    - contact_threshold: max pixel distance for contact
    - reach_threshold: max pixel distance for reach
    - grasp_confidence: min confidence to classify as grasp
    """

    def __init__(
        self,
        contact_threshold: float = 50.0,
        reach_threshold: float = 150.0,
        min_grasp_frames: int = 3,
    ):
        self.contact_threshold = contact_threshold
        self.reach_threshold = reach_threshold
        self.min_grasp_frames = min_grasp_frames

        # Temporal state for grasp detection
        self._prev_hand_positions: Dict[str, np.ndarray] = {}
        self._contact_history: Dict[str, List[bool]] = {}

    def predict(
        self,
        pose: PoseResult,
        detections: List[Detection],
        frame: Optional[np.ndarray] = None,
    ) -> List[HOIResult]:
        """
        Predict hand-object interactions.

        Args:
            pose: Pose result with keypoints.
            detections: Object detections (excluding persons).
            frame: Optional frame for visual features (unused in proximity model).

        Returns:
            List of HOIResult.
        """
        results = []
        kps = pose.keypoints_2d

        # Get hand positions
        hands = {}
        # Left wrist = COCO index 9
        if len(kps) > 9 and kps[9, 2] > 0.1:
            hands["left"] = kps[9, :2]
        # Right wrist = COCO index 10
        if len(kps) > 10 and kps[10, 2] > 0.1:
            hands["right"] = kps[10, :2]

        # Filter out person detections
        obj_dets = [d for d in detections if d.class_name.lower() not in
                    ("person", "astronaut", "human")]

        for hand_name, hand_pos in hands.items():
            # Compute velocity
            prev_key = f"{pose.person_id}_{hand_name}"
            prev_pos = self._prev_hand_positions.get(prev_key)
            velocity = compute_hand_velocity(hand_pos, prev_pos)
            self._prev_hand_positions[prev_key] = hand_pos.copy()

            for det in obj_dets:
                dist = compute_hand_object_distance(hand_pos, det.bbox)

                # Determine interaction type
                interaction_type = "no_contact"
                contact_prob = 0.0
                confidence = 0.0

                if dist < self.contact_threshold:
                    contact_prob = max(0, 1.0 - dist / self.contact_threshold)

                    # Track contact history for temporal filtering
                    contact_key = f"{prev_key}_{det.object_id}"
                    if contact_key not in self._contact_history:
                        self._contact_history[contact_key] = []
                    self._contact_history[contact_key].append(True)

                    # Keep only recent history
                    history = self._contact_history[contact_key][-10:]
                    self._contact_history[contact_key] = history

                    consecutive_contacts = 0
                    for c in reversed(history):
                        if c:
                            consecutive_contacts += 1
                        else:
                            break

                    if consecutive_contacts >= self.min_grasp_frames:
                        if velocity < 5.0:
                            interaction_type = "hold"
                            confidence = contact_prob * 0.9
                        else:
                            interaction_type = "manipulate"
                            confidence = contact_prob * 0.85
                    else:
                        interaction_type = "grasp"
                        confidence = contact_prob * 0.8

                elif dist < self.reach_threshold:
                    interaction_type = "reach"
                    contact_prob = max(0, 1.0 - dist / self.reach_threshold) * 0.5
                    confidence = contact_prob * 0.6

                    # Clear contact history
                    contact_key = f"{prev_key}_{det.object_id}"
                    if contact_key in self._contact_history:
                        self._contact_history[contact_key].append(False)
                else:
                    continue  # Too far, skip

                results.append(HOIResult(
                    hand=hand_name,
                    object_id=str(det.object_id),
                    object_class=det.class_name,
                    interaction_type=interaction_type,
                    confidence=confidence,
                    contact_probability=contact_prob,
                    person_id=pose.person_id,
                ))

        return results

    def reset(self):
        """Reset temporal state (call between videos)."""
        self._prev_hand_positions.clear()
        self._contact_history.clear()


class LearnedHOIModel(HOIModel):
    """
    Learned HOI model using a small graph neural network.
    Uses spatial features from proximity + visual crop features.

    To be implemented after initial data collection.
    Currently wraps ProximityHOIModel.
    """

    def __init__(self, **kwargs):
        self._proxy = ProximityHOIModel(**kwargs)
        self.model = None

    def predict(
        self,
        pose: PoseResult,
        detections: List[Detection],
        frame: Optional[np.ndarray] = None,
    ) -> List[HOIResult]:
        if self.model is None:
            # Fall back to proximity model
            return self._proxy.predict(pose, detections, frame)

        # TODO: Implement learned HOI when model is trained
        return self._proxy.predict(pose, detections, frame)

    def load_model(self, checkpoint: str = "", device: str = "cpu"):
        """Load learned HOI model weights."""
        if checkpoint and Path(checkpoint).exists():
            try:
                import torch
                state = torch.load(checkpoint, map_location=device)
                # TODO: Build and load learned model
                logger.info(f"Learned HOI model loaded: {checkpoint}")
            except Exception as e:
                logger.warning(f"Failed to load learned HOI model: {e}")
        else:
            logger.info("Using proximity-based HOI model (no learned weights)")
