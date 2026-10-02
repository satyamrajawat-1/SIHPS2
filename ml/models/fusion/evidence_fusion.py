"""
ASTRA — Evidence Fusion Layer

Combines evidence from ALL perception sources into a single
Evidence packet that is fed to the Experiment State Machine.

Sources:
  1. Object Detection  →  detected_objects
  2. Pose Estimation   →  hand positions, body pose confidence
  3. Action (ST-GCN++) →  detected_action
  4. HOI Graph         →  hand_object_interactions
  5. Object State      →  object_states, state_transitions
  6. Temporal Model    →  temporal action, step candidate, anomaly
  7. Visual Features   →  scene context (used by temporal model)

Rules:
  - Fusion is DETERMINISTIC. No LLM in this path.
  - Evidence weights are configurable.
  - Individual source failures degrade gracefully (do not crash).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from ml.experiment.state_graph.state_machine import Evidence
from ml.models.interfaces import (
    Detection, PoseResult, ActionResult, HOIResult,
    ObjectStateResult, TemporalResult, FusedEvidence,
)

logger = logging.getLogger(__name__)


class EvidenceFusionLayer:
    """
    Aggregates perception outputs into Evidence packets for the state machine.

    This is a deterministic transformation — no learned parameters.
    """

    def __init__(
        self,
        object_class_to_id_map: Optional[Dict[str, str]] = None,
        frame_interval: float = 1.0 / 30.0,
    ):
        """
        Args:
            object_class_to_id_map: Maps detection class names to experiment object IDs.
                Example: {"sample_container": "container_01", "scissors": "tool_scissors"}
            frame_interval: Time between frames (seconds).
        """
        self.class_to_id = object_class_to_id_map or {}
        self.frame_interval = frame_interval
        self._frame_count = 0

    def fuse(
        self,
        frame_id: int,
        timestamp: float,
        camera_id: str = "cam_00",
        detections: Optional[List[Detection]] = None,
        poses: Optional[List[PoseResult]] = None,
        action_result: Optional[ActionResult] = None,
        hoi_results: Optional[List[HOIResult]] = None,
        object_states: Optional[List[ObjectStateResult]] = None,
        temporal_result: Optional[TemporalResult] = None,
    ) -> Evidence:
        """
        Fuse all perception outputs into a single Evidence packet.

        Each source is optional — missing sources contribute zero evidence.
        """
        ev = Evidence(
            timestamp=timestamp,
            frame_id=frame_id,
            camera_id=camera_id,
        )

        self._frame_count += 1

        # 1. Object detections
        if detections:
            for det in detections:
                obj_id = self._map_class_to_id(det.class_name, det.object_id)
                ev.detected_objects[obj_id] = det.confidence
                ev.object_classes[obj_id] = det.class_name

        # 2. Pose data
        if poses:
            best_pose = max(poses, key=lambda p: p.tracking_confidence)
            ev.pose_confidence = best_pose.tracking_confidence

            # Compute hand-object proximity
            if detections:
                for hand_name, wrist_idx in [("left", 9), ("right", 10)]:
                    kps = best_pose.keypoints_2d
                    if wrist_idx < len(kps) and kps[wrist_idx, 2] > 0.1:
                        hand_pos = kps[wrist_idx, :2]
                        for det in detections:
                            obj_id = self._map_class_to_id(det.class_name, det.object_id)
                            dist = self._point_to_bbox_distance(hand_pos, det.bbox)
                            # Convert distance to proximity score (0-1)
                            proximity = max(0, 1.0 - dist / 200.0)
                            # Keep max proximity across hands
                            ev.hand_near_objects[obj_id] = max(
                                ev.hand_near_objects.get(obj_id, 0.0),
                                proximity,
                            )

        # 3. Action recognition
        if action_result:
            ev.detected_action = action_result.action_id
            ev.action_confidence = action_result.confidence

        # 4. HOI results
        if hoi_results:
            for hoi in hoi_results:
                obj_id = self._map_class_to_id(hoi.object_class, hoi.object_id)
                ev.hand_object_interactions.append({
                    "hand": hoi.hand,
                    "object_id": obj_id,
                    "type": hoi.interaction_type,
                    "confidence": hoi.confidence,
                })

        # 5. Object states
        if object_states:
            for os_result in object_states:
                obj_id = self._map_class_to_id(os_result.object_class, os_result.object_id)
                ev.object_states[obj_id] = os_result.current_state
                ev.object_state_confidences[obj_id] = os_result.state_confidence
                if os_result.is_transition:
                    ev.object_state_transitions.append({
                        "object_id": obj_id,
                        "from": os_result.previous_state,
                        "to": os_result.current_state,
                        "confidence": os_result.state_confidence,
                    })

        # 6. Temporal model results
        if temporal_result:
            ev.temporal_action = temporal_result.current_action
            ev.temporal_action_confidence = temporal_result.current_action_confidence
            ev.temporal_step_candidate = temporal_result.current_step_candidate
            ev.temporal_step_confidence = temporal_result.step_confidence
            ev.anomaly_score = temporal_result.anomaly_score

            # Compute temporal consistency from entropy
            if temporal_result.supporting_evidence:
                action_entropy = temporal_result.supporting_evidence.get("action_entropy", 1.0)
                # Lower entropy = higher consistency
                ev.temporal_consistency = max(0, 1.0 - action_entropy / 3.0)
            else:
                ev.temporal_consistency = temporal_result.current_action_confidence * 0.8

            # Set fused step ID from temporal model
            if temporal_result.step_confidence > 0.3:
                ev.fused_step_id = temporal_result.current_step_candidate
                ev.fused_confidence = temporal_result.step_confidence

        return ev

    def _map_class_to_id(self, class_name: str, fallback_id: Any) -> str:
        """Map a detection class name to an experiment object ID."""
        if class_name in self.class_to_id:
            return self.class_to_id[class_name]
        # Try with underscores/lowercase
        normalized = class_name.lower().replace(" ", "_").replace("-", "_")
        if normalized in self.class_to_id:
            return self.class_to_id[normalized]
        return str(fallback_id)

    @staticmethod
    def _point_to_bbox_distance(point: np.ndarray, bbox: List[float]) -> float:
        """Compute distance from a point to the nearest edge of a bbox."""
        x, y = point
        x1, y1, x2, y2 = bbox
        cx = max(x1, min(x, x2))
        cy = max(y1, min(y, y2))
        return float(np.sqrt((x - cx) ** 2 + (y - cy) ** 2))


class FeatureAggregator:
    """
    Aggregates per-frame features into the multimodal feature vector
    that is fed to the temporal model.

    Concatenates: visual_features + pose_features + action_scores + hoi_features
    """

    def __init__(
        self,
        visual_dim: int = 256,
        pose_dim: int = 51,      # 17 joints × 3 (x, y, conf)
        action_dim: int = 18,     # Number of action classes
        hoi_dim: int = 32,        # HOI summary features
        object_state_dim: int = 16,
    ):
        self.visual_dim = visual_dim
        self.pose_dim = pose_dim
        self.action_dim = action_dim
        self.hoi_dim = hoi_dim
        self.object_state_dim = object_state_dim
        self.total_dim = visual_dim + pose_dim + action_dim + hoi_dim + object_state_dim

    def aggregate(
        self,
        visual_features: Optional[np.ndarray] = None,
        pose_results: Optional[List[PoseResult]] = None,
        action_result: Optional[ActionResult] = None,
        hoi_results: Optional[List[HOIResult]] = None,
        object_states: Optional[List[ObjectStateResult]] = None,
    ) -> np.ndarray:
        """
        Aggregate all per-frame features into a single vector.

        Returns:
            (total_dim,) feature vector.
        """
        parts = []

        # Visual features
        if visual_features is not None:
            vf = visual_features[:self.visual_dim]
            if len(vf) < self.visual_dim:
                vf = np.pad(vf, (0, self.visual_dim - len(vf)))
            parts.append(vf)
        else:
            parts.append(np.zeros(self.visual_dim, dtype=np.float32))

        # Pose features (flatten keypoints)
        if pose_results and len(pose_results) > 0:
            best_pose = max(pose_results, key=lambda p: p.tracking_confidence)
            kps = best_pose.keypoints_2d.flatten()[:self.pose_dim]
            if len(kps) < self.pose_dim:
                kps = np.pad(kps, (0, self.pose_dim - len(kps)))
            parts.append(kps.astype(np.float32))
        else:
            parts.append(np.zeros(self.pose_dim, dtype=np.float32))

        # Action scores
        if action_result and action_result.all_scores:
            scores = np.zeros(self.action_dim, dtype=np.float32)
            for i, (k, v) in enumerate(action_result.all_scores.items()):
                if i < self.action_dim:
                    scores[i] = v
            parts.append(scores)
        else:
            parts.append(np.zeros(self.action_dim, dtype=np.float32))

        # HOI summary features
        hoi_vec = np.zeros(self.hoi_dim, dtype=np.float32)
        if hoi_results:
            # Encode: count of interactions, max confidence, hand distribution
            hoi_vec[0] = len(hoi_results)
            if hoi_results:
                hoi_vec[1] = max(h.confidence for h in hoi_results)
                hoi_vec[2] = sum(1 for h in hoi_results if h.hand == "left")
                hoi_vec[3] = sum(1 for h in hoi_results if h.hand == "right")
                hoi_vec[4] = max(h.contact_probability for h in hoi_results)
                # Interaction type one-hot (up to hoi_dim - 5 types)
                for h in hoi_results[:5]:
                    type_hash = hash(h.interaction_type) % (self.hoi_dim - 5)
                    hoi_vec[5 + type_hash] = h.confidence
        parts.append(hoi_vec)

        # Object state features
        state_vec = np.zeros(self.object_state_dim, dtype=np.float32)
        if object_states:
            state_vec[0] = len(object_states)
            for i, os_r in enumerate(object_states[:self.object_state_dim - 2]):
                state_vec[i + 1] = os_r.state_confidence
                if os_r.is_transition:
                    state_vec[-1] += 1  # Count transitions
        parts.append(state_vec)

        return np.concatenate(parts)
