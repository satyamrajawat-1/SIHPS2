"""
ASTRA — Action Temporal Smoother

Implements temporal stability for raw frame-level action predictions:
- Sliding window majority voting
- Minimum persistence before state change
- Confidence thresholding
- Cooldown period after transitions

Prevents flickering between actions in the pipeline.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SmoothedAction:
    """Result from the temporal smoother."""
    action_id: str
    action_name: str
    confidence: float
    raw_action: str       # The unsmoothed prediction
    is_stable: bool       # Whether minimum persistence is met
    frames_held: int      # How many frames this action has been the smoothed output
    transition: bool      # Whether this frame is a transition


class ActionSmoother:
    """
    Temporal smoothing for action predictions.

    Uses sliding-window majority voting with minimum persistence
    to produce stable action outputs for the state machine.
    """

    def __init__(
        self,
        window_size: int = 15,
        min_persistence: int = 5,
        confidence_threshold: float = 0.3,
        cooldown_frames: int = 10,
        action_classes: Optional[List[str]] = None,
    ):
        """
        Args:
            window_size: Number of frames for majority voting.
            min_persistence: Minimum consecutive frames before accepting new action.
            confidence_threshold: Minimum confidence to accept a prediction.
            cooldown_frames: Minimum frames between transitions.
        """
        self.window_size = window_size
        self.min_persistence = min_persistence
        self.confidence_threshold = confidence_threshold
        self.cooldown_frames = cooldown_frames
        self.action_classes = action_classes or []

        # State
        self._history: deque = deque(maxlen=window_size)
        self._confidence_history: deque = deque(maxlen=window_size)
        self._current_action: str = "idle"
        self._frames_held: int = 0
        self._candidate_action: str = "idle"
        self._candidate_count: int = 0
        self._cooldown_remaining: int = 0
        self._frame_count: int = 0

    def reset(self):
        """Reset smoother state."""
        self._history.clear()
        self._confidence_history.clear()
        self._current_action = "idle"
        self._frames_held = 0
        self._candidate_action = "idle"
        self._candidate_count = 0
        self._cooldown_remaining = 0
        self._frame_count = 0

    def update(self, action_id: str, confidence: float) -> SmoothedAction:
        """
        Process a new frame prediction and return smoothed result.

        Args:
            action_id: Raw predicted action.
            confidence: Prediction confidence.

        Returns:
            SmoothedAction with temporally stable output.
        """
        self._frame_count += 1

        # Apply confidence threshold — below threshold → idle
        if confidence < self.confidence_threshold:
            action_id = "idle"
            confidence = 0.0

        self._history.append(action_id)
        self._confidence_history.append(confidence)

        # Majority vote over window
        vote_action, vote_conf = self._majority_vote()

        # Manage cooldown
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        transition = False

        # Check if majority vote suggests a different action
        if vote_action != self._current_action:
            if vote_action == self._candidate_action:
                self._candidate_count += 1
            else:
                self._candidate_action = vote_action
                self._candidate_count = 1

            # Accept transition only if:
            # 1. Candidate has been seen for min_persistence frames
            # 2. Cooldown has expired
            if (self._candidate_count >= self.min_persistence
                    and self._cooldown_remaining <= 0):
                self._current_action = vote_action
                self._frames_held = 1
                self._cooldown_remaining = self.cooldown_frames
                self._candidate_count = 0
                transition = True
                logger.debug(
                    f"Action transition → {vote_action} "
                    f"(conf={vote_conf:.2f}, frame={self._frame_count})"
                )
        else:
            self._frames_held += 1
            self._candidate_action = self._current_action
            self._candidate_count = 0

        return SmoothedAction(
            action_id=self._current_action,
            action_name=self._current_action.capitalize(),
            confidence=vote_conf,
            raw_action=action_id,
            is_stable=self._frames_held >= self.min_persistence,
            frames_held=self._frames_held,
            transition=transition,
        )

    def _majority_vote(self) -> tuple:
        """Compute majority vote and average confidence over window."""
        if not self._history:
            return "idle", 0.0

        # Count votes
        votes: Dict[str, int] = {}
        confs: Dict[str, List[float]] = {}
        for action, conf in zip(self._history, self._confidence_history):
            votes[action] = votes.get(action, 0) + 1
            if action not in confs:
                confs[action] = []
            confs[action].append(conf)

        # Winner
        winner = max(votes, key=votes.get)
        avg_conf = sum(confs[winner]) / len(confs[winner])
        return winner, avg_conf

    def get_state(self) -> Dict:
        """Return current smoother state for logging."""
        return {
            "current_action": self._current_action,
            "frames_held": self._frames_held,
            "is_stable": self._frames_held >= self.min_persistence,
            "cooldown_remaining": self._cooldown_remaining,
            "window_size": len(self._history),
            "frame_count": self._frame_count,
        }
