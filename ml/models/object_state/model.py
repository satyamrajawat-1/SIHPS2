"""
ASTRA — Object State Recognition

Classifies the current state of each experiment object
(open/closed, empty/full, docked/in-hand, etc.)

Uses object crop features + HOI context.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from ml.models.interfaces import ObjectStateModel, ObjectStateResult, Detection, HOIResult


class ObjectStateClassifier(ObjectStateModel):
    """
    Object state classifier.

    For each detected object, predicts its current state from a
    predefined set of states (loaded from experiment config).

    Architecture:
    - Crop object region from frame
    - Extract visual features (shared backbone or lightweight CNN)
    - Classify into state categories
    - Track state transitions temporally

    Initially: rule-based + visual similarity
    Later: fine-tuned lightweight classifier
    """

    def __init__(
        self,
        state_definitions: Optional[Dict[str, List[str]]] = None,
        crop_size: int = 64,
        transition_min_frames: int = 3,
    ):
        """
        Args:
            state_definitions: {class_name: [state1, state2, ...]}.
            crop_size: Size to resize object crops.
            transition_min_frames: Min frames of consistent observation for state change.
        """
        self.state_definitions = state_definitions or {}
        self.crop_size = crop_size
        self.transition_min_frames = transition_min_frames

        # Temporal state tracking
        self._current_states: Dict[str, str] = {}
        self._state_history: Dict[str, List[str]] = {}
        self._previous_crops: Dict[str, np.ndarray] = {}

        self.model = None
        self.device = "cpu"

    def load_config_states(self, experiment_config) -> None:
        """
        Load state definitions from an ExperimentConfig.

        Args:
            experiment_config: ExperimentConfig with objects that have state definitions.
        """
        for obj in experiment_config.objects:
            if obj.possible_states:
                self.state_definitions[obj.obj_class] = obj.possible_states
                self._current_states[obj.id] = obj.expected_initial_state

    def predict(
        self,
        detections: List[Detection],
        frame: Optional[np.ndarray] = None,
        hoi_results: Optional[List[HOIResult]] = None,
    ) -> List[ObjectStateResult]:
        """
        Predict object states.

        Uses a combination of:
        1. Visual features (crop changes)
        2. HOI context (interactions suggest state changes)
        3. Temporal consistency (require multiple frames to change state)
        """
        results = []

        for det in detections:
            obj_key = f"{det.class_name}_{det.object_id}"

            # Get state definition for this class
            valid_states = self.state_definitions.get(det.class_name, [])
            if not valid_states:
                continue

            # Current known state
            current_state = self._current_states.get(obj_key, valid_states[0])

            # Extract crop if frame available
            crop = None
            if frame is not None:
                crop = self._extract_crop(frame, det.bbox)

            # Predict new state
            predicted_state = current_state
            confidence = 0.5

            if self.model is not None:
                # Use learned model
                predicted_state, confidence = self._predict_with_model(crop, det, valid_states)
            else:
                # Rule-based: use HOI and visual cues
                predicted_state, confidence = self._predict_rule_based(
                    obj_key, det, crop, hoi_results, valid_states, current_state,
                )

            # Temporal filtering
            is_transition = False
            if obj_key not in self._state_history:
                self._state_history[obj_key] = []

            self._state_history[obj_key].append(predicted_state)
            # Keep only recent history
            self._state_history[obj_key] = self._state_history[obj_key][-20:]

            # Check for consistent state change
            recent = self._state_history[obj_key][-self.transition_min_frames:]
            if (len(recent) >= self.transition_min_frames and
                    all(s == predicted_state for s in recent) and
                    predicted_state != current_state):
                is_transition = True
                self._current_states[obj_key] = predicted_state

            results.append(ObjectStateResult(
                object_id=obj_key,
                object_class=det.class_name,
                current_state=self._current_states.get(obj_key, predicted_state),
                previous_state=current_state if is_transition else None,
                state_confidence=confidence,
                is_transition=is_transition,
            ))

        return results

    def _extract_crop(self, frame: np.ndarray, bbox: List[float]) -> np.ndarray:
        """Extract and resize object crop from frame."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)

        return cv2.resize(crop, (self.crop_size, self.crop_size))

    def _predict_rule_based(
        self,
        obj_key: str,
        det: Detection,
        crop: Optional[np.ndarray],
        hoi_results: Optional[List[HOIResult]],
        valid_states: List[str],
        current_state: str,
    ) -> tuple:
        """
        Rule-based state prediction.

        Uses HOI signals and visual change detection.
        """
        predicted = current_state
        confidence = 0.5

        # Check HOI context
        if hoi_results:
            active_interactions = [
                hoi for hoi in hoi_results
                if hoi.object_id == str(det.object_id) and hoi.confidence > 0.5
            ]

            if active_interactions:
                for hoi in active_interactions:
                    # Infer state changes from interaction type
                    if hoi.interaction_type == "grasp" and "in_hand" in valid_states:
                        predicted = "in_hand"
                        confidence = hoi.confidence * 0.8
                    elif hoi.interaction_type == "release" and "on_surface" in valid_states:
                        predicted = "on_surface"
                        confidence = hoi.confidence * 0.7
                    elif hoi.interaction_type in ("manipulate", "use"):
                        # Check for open/close actions
                        if "open" in valid_states and current_state == "closed":
                            predicted = "open"
                            confidence = hoi.confidence * 0.6
                        elif "closed" in valid_states and current_state == "open":
                            predicted = "closed"
                            confidence = hoi.confidence * 0.6

        # Visual change detection (crop comparison)
        if crop is not None and obj_key in self._previous_crops:
            prev_crop = self._previous_crops[obj_key]
            similarity = self._compute_crop_similarity(prev_crop, crop)
            if similarity < 0.7:
                # Significant visual change detected
                confidence = max(confidence, 0.6)

        if crop is not None:
            self._previous_crops[obj_key] = crop.copy()

        return predicted, confidence

    def _predict_with_model(self, crop, det, valid_states):
        """Use learned model for state prediction."""
        # TODO: Implement when model is trained
        return valid_states[0], 0.5

    def _compute_crop_similarity(
        self,
        crop1: np.ndarray,
        crop2: np.ndarray,
    ) -> float:
        """Compute visual similarity between two crops."""
        if crop1.shape != crop2.shape:
            return 0.5

        # Simple histogram comparison
        hist1 = cv2.calcHist([crop1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([crop2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()

        return float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))

    def reset(self):
        """Reset temporal state (call between videos)."""
        self._state_history.clear()
        self._previous_crops.clear()
        # Keep current_states as they represent the known initial state
