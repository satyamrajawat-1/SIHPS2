"""
ASTRA — Experiment State Graph (Deterministic Finite State Machine)

This is the CORE procedural reasoning module. It does NOT use neural networks.
It receives evidence from the AI perception pipeline and makes deterministic
procedural decisions based on the experiment configuration.

RULES:
  1. One frame can never complete an experiment step.
  2. Step completion requires temporal evidence.
  3. Required objects must be present.
  4. Required interactions must be observed.
  5. Required actions must be observed.
  6. Required object state transitions must happen.
  7. Step ordering must be respected.
  8. Low-confidence observations → UNCERTAIN (never forced VALID).
  9. Unexpected actions must be detected.
  10. An LLM can never override these rules.
  11. Every state transition stores its evidence.
  12. The system can explain "Why was this step marked valid?"
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class StepStatus(str, Enum):
    """Possible statuses for an experiment step."""
    EXPECTED = "EXPECTED"
    IN_PROGRESS = "IN_PROGRESS"
    VALID = "VALID"
    SKIPPED = "SKIPPED"
    OUT_OF_ORDER = "OUT_OF_ORDER"
    FAILED = "FAILED"
    UNCERTAIN = "UNCERTAIN"
    TIMEOUT = "TIMEOUT"
    COMPLETE = "COMPLETE"
    NOT_STARTED = "NOT_STARTED"


class ExperimentStatus(str, Enum):
    """Overall experiment status."""
    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    ABORTED = "ABORTED"


@dataclass
class Evidence:
    """Evidence packet from the perception pipeline."""
    timestamp: float
    frame_id: int
    camera_id: str = "cam_00"

    # Detection evidence
    detected_objects: Dict[str, float] = field(default_factory=dict)  # {obj_id: confidence}
    object_classes: Dict[str, str] = field(default_factory=dict)  # {obj_id: class_name}

    # Pose evidence
    pose_confidence: float = 0.0
    hand_near_objects: Dict[str, float] = field(default_factory=dict)  # {obj_id: proximity_score}

    # Action evidence
    detected_action: Optional[str] = None
    action_confidence: float = 0.0

    # HOI evidence
    hand_object_interactions: List[Dict[str, Any]] = field(default_factory=list)
    # [{hand: "right", object_id: "x", type: "grasp", confidence: 0.9}, ...]

    # Object state evidence
    object_states: Dict[str, str] = field(default_factory=dict)  # {obj_id: state}
    object_state_confidences: Dict[str, float] = field(default_factory=dict)
    object_state_transitions: List[Dict[str, Any]] = field(default_factory=list)
    # [{object_id: "x", from: "closed", to: "open", confidence: 0.95}, ...]

    # Temporal evidence
    temporal_action: Optional[str] = None
    temporal_action_confidence: float = 0.0
    temporal_step_candidate: Optional[str] = None
    temporal_step_confidence: float = 0.0
    temporal_consistency: float = 0.0
    anomaly_score: float = 0.0

    # Fused evidence (computed by EvidenceFusion layer)
    fused_step_id: Optional[str] = None
    fused_confidence: float = 0.0
    fused_evidence_breakdown: Dict[str, float] = field(default_factory=dict)


@dataclass
class StepState:
    """State tracking for a single experiment step."""
    step_id: str
    status: StepStatus = StepStatus.NOT_STARTED
    evidence_frames: int = 0  # Number of frames with supporting evidence
    total_evidence_score: float = 0.0
    start_timestamp: Optional[float] = None
    end_timestamp: Optional[float] = None
    attempts: int = 0
    evidence_history: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def avg_evidence_score(self) -> float:
        if self.evidence_frames == 0:
            return 0.0
        return self.total_evidence_score / self.evidence_frames


@dataclass
class StateTransitionRecord:
    """Record of a state transition with evidence."""
    timestamp: float
    step_id: str
    from_status: StepStatus
    to_status: StepStatus
    evidence: Dict[str, Any]
    reason: str


class ExperimentStateMachine:
    """
    Deterministic experiment state machine.

    Receives evidence from the AI pipeline and makes procedural decisions
    based on the experiment configuration. Every transition is evidence-backed
    and explainable.
    """

    def __init__(self, experiment_config):
        """
        Initialize the state machine from an ExperimentConfig.

        Args:
            experiment_config: An ExperimentConfig loaded from YAML.
        """
        self.config = experiment_config
        self.rules = experiment_config.rules

        # Initialize step states
        self.step_states: Dict[str, StepState] = {}
        for step in experiment_config.steps:
            self.step_states[step.id] = StepState(step_id=step.id)

        # Current expected step
        first_step = experiment_config.get_first_step()
        self.current_step_id: Optional[str] = first_step.id if first_step else None
        if self.current_step_id:
            self.step_states[self.current_step_id].status = StepStatus.EXPECTED

        # Object states
        self.object_states: Dict[str, str] = {}
        for obj in experiment_config.objects:
            self.object_states[obj.id] = obj.expected_initial_state

        # Overall status
        self.experiment_status = ExperimentStatus.NOT_STARTED
        self.start_time: Optional[float] = None

        # Transition history
        self.transition_history: List[StateTransitionRecord] = []

        # Anomaly log
        self.anomalies: List[Dict[str, Any]] = []

        # Completed steps (in order)
        self.completed_steps: List[str] = []

    def update(self, evidence: Evidence) -> Dict[str, Any]:
        """
        Process new evidence and update the experiment state.

        This is the main entry point. Called once per temporal window
        with aggregated evidence from the perception pipeline.

        Args:
            evidence: Evidence packet from perception pipeline.

        Returns:
            Dict with current state, step status, next step, anomalies,
            confidence, and evidence trail.
        """
        if self.experiment_status == ExperimentStatus.NOT_STARTED:
            self.experiment_status = ExperimentStatus.IN_PROGRESS
            self.start_time = evidence.timestamp

        result = {
            "timestamp": evidence.timestamp,
            "frame_id": evidence.frame_id,
            "experiment_status": self.experiment_status.value,
            "current_step_id": self.current_step_id,
            "current_step_status": None,
            "next_valid_steps": [],
            "anomalies": [],
            "confidence": 0.0,
            "evidence": {},
            "object_states": dict(self.object_states),
            "events": [],
        }

        if self.experiment_status == ExperimentStatus.COMPLETE:
            result["current_step_status"] = StepStatus.COMPLETE.value
            return result

        if self.current_step_id is None:
            return result

        current_step = self.config.get_step(self.current_step_id)
        if current_step is None:
            return result

        step_state = self.step_states[self.current_step_id]
        result["current_step_status"] = step_state.status.value

        # Check for timeout
        if step_state.start_timestamp is not None:
            elapsed = evidence.timestamp - step_state.start_timestamp
            if elapsed > current_step.timeout_seconds:
                self._transition_step(
                    step_state, StepStatus.TIMEOUT,
                    evidence.timestamp,
                    {"reason": f"Step timed out after {elapsed:.1f}s"},
                    f"Step {current_step.id} exceeded timeout of {current_step.timeout_seconds}s",
                )
                result["events"].append({
                    "event": "STEP_TIMEOUT",
                    "step_id": current_step.id,
                    "elapsed": elapsed,
                })

        # --- Evaluate evidence against current expected step ---
        step_evidence = self._evaluate_step_evidence(current_step, evidence)
        result["evidence"] = step_evidence

        # Check for out-of-order action
        anomaly = self._check_out_of_order(evidence)
        if anomaly:
            result["anomalies"].append(anomaly)
            self.anomalies.append(anomaly)

        # Is there enough evidence for the current step?
        if step_evidence["score"] >= self.rules.uncertain_threshold:
            # Mark as IN_PROGRESS if not already
            if step_state.status == StepStatus.EXPECTED:
                self._transition_step(
                    step_state, StepStatus.IN_PROGRESS,
                    evidence.timestamp,
                    step_evidence,
                    f"Evidence detected for step {current_step.id}",
                )
                step_state.start_timestamp = evidence.timestamp
                result["events"].append({
                    "event": "STEP_STARTED",
                    "step_id": current_step.id,
                })

            # Accumulate evidence
            step_state.evidence_frames += 1
            step_state.total_evidence_score += step_evidence["score"]
            step_state.evidence_history.append({
                "timestamp": evidence.timestamp,
                "frame_id": evidence.frame_id,
                "score": step_evidence["score"],
                "breakdown": step_evidence.get("breakdown", {}),
            })

        # Check completion conditions
        if self._check_step_completion(current_step, step_state, evidence):
            self._transition_step(
                step_state, StepStatus.VALID,
                evidence.timestamp,
                step_evidence,
                f"Step {current_step.id} completed with sufficient evidence",
            )
            self.completed_steps.append(current_step.id)

            # Update object states
            for obj_id, new_state in current_step.object_state_effects.items():
                self.object_states[obj_id] = new_state

            result["events"].append({
                "event": "STEP_COMPLETED",
                "step_id": current_step.id,
                "confidence": step_state.avg_evidence_score,
            })

            # Advance to next step
            self._advance_to_next_step(current_step, evidence.timestamp)

        elif step_evidence["score"] < self.rules.abstain_threshold and \
                step_state.status == StepStatus.IN_PROGRESS:
            # Evidence dropped too low — mark uncertain
            if step_state.evidence_frames > self.rules.min_temporal_frames_for_step:
                self._transition_step(
                    step_state, StepStatus.UNCERTAIN,
                    evidence.timestamp,
                    step_evidence,
                    f"Evidence for step {current_step.id} is below threshold",
                )
                result["events"].append({
                    "event": "STEP_UNCERTAIN",
                    "step_id": current_step.id,
                })

        # Update result
        result["current_step_status"] = step_state.status.value
        result["current_step_id"] = self.current_step_id
        result["confidence"] = step_state.avg_evidence_score
        result["next_valid_steps"] = self._get_next_valid_steps()
        result["experiment_status"] = self.experiment_status.value

        return result

    def _evaluate_step_evidence(
        self,
        step,
        evidence: Evidence,
    ) -> Dict[str, Any]:
        """
        Evaluate how well the evidence matches the expected step.


        Returns a score and breakdown of individual evidence sources.
        """
        breakdown = {}
        weights = self.rules.evidence_weights

        # 1. Object detection: are required objects present?
        obj_scores = []
        for obj_id in step.required_objects:
            score = evidence.detected_objects.get(obj_id, 0.0)
            obj_scores.append(score)
        obj_score = sum(obj_scores) / max(1, len(obj_scores))
        breakdown["object_detection"] = round(obj_score, 4)

        # 2. Pose: are hands near required objects?
        pose_scores = []
        for obj_id in step.required_objects:
            score = evidence.hand_near_objects.get(obj_id, 0.0)
            pose_scores.append(score)
        pose_score = max(
            sum(pose_scores) / max(1, len(pose_scores)),
            evidence.pose_confidence * 0.5,
        )
        breakdown["pose"] = round(pose_score, 4)

        # 3. Action recognition: does detected action match required actions?
        action_score = 0.0
        if evidence.detected_action in step.required_actions:
            action_score = evidence.action_confidence
        elif evidence.temporal_action in step.required_actions:
            action_score = evidence.temporal_action_confidence
        breakdown["action_recognition"] = round(action_score, 4)

        # 4. HOI: is the right interaction happening?
        hoi_score = 0.0
        if evidence.hand_object_interactions:
            for interaction in evidence.hand_object_interactions:
                if interaction.get("object_id") in step.required_objects:
                    hoi_score = max(hoi_score, interaction.get("confidence", 0.0))
        else:
            # If no HOI data is provided, check if this step even requires
            # hand manipulation. For non-manipulation steps (e.g., visual_inspection),
            # HOI is not a meaningful signal.
            manipulation_actions = {
                "grasp", "release", "move", "place", "pick", "remove",
                "insert", "open", "close", "rotate", "press", "pull",
                "push", "transfer", "use",
            }
            has_manipulation = any(
                a in manipulation_actions for a in step.required_actions
            )
            if not has_manipulation:
                # Non-manipulation step: HOI is not required, use neutral score
                hoi_score = evidence.pose_confidence * 0.8
            # else: hoi_score remains 0.0 (HOI data was expected but not provided)
        breakdown["hoi"] = round(hoi_score, 4)

        # 5. Object state: check state conditions
        state_score = 0.0
        state_checks = 0

        if step.object_state_effects:
            # Step has expected state changes — check for those transitions
            for obj_id, expected_state in step.object_state_effects.items():
                state_checks += 1
                # Check if state transition evidence exists
                for trans in evidence.object_state_transitions:
                    if (trans.get("object_id") == obj_id and
                            trans.get("to") == expected_state):
                        state_score += trans.get("confidence", 0.5)
                        break
                else:
                    # Check current observed state
                    if evidence.object_states.get(obj_id) == expected_state:
                        state_score += evidence.object_state_confidences.get(obj_id, 0.5)
            state_score /= state_checks
        elif step.object_state_preconditions:
            # Step has preconditions but no effects — verify preconditions are met
            for obj_id, required_state in step.object_state_preconditions.items():
                state_checks += 1
                observed = evidence.object_states.get(obj_id)
                if observed == required_state:
                    state_score += evidence.object_state_confidences.get(obj_id, 0.8)
                elif observed is not None:
                    state_score += 0.0  # Wrong state
                else:
                    state_score += 0.3  # Unknown state
            state_score /= max(1, state_checks)
        else:
            # No state requirements at all — neutral score
            state_score = obj_score * 0.9 if obj_score > 0 else 0.5
        breakdown["object_state"] = round(state_score, 4)

        # 6. Temporal consistency
        temporal_score = evidence.temporal_consistency
        if evidence.temporal_step_candidate == step.id:
            temporal_score = max(temporal_score, evidence.temporal_step_confidence)
        breakdown["temporal_consistency"] = round(temporal_score, 4)

        # Weighted fusion
        total_score = sum(
            breakdown.get(key, 0.0) * weights.get(key, 0.0)
            for key in weights
        )

        return {
            "score": round(total_score, 4),
            "breakdown": breakdown,
            "step_id": step.id,
        }

    def _check_step_completion(
        self,
        step,
        step_state: StepState,
        evidence: Evidence,
    ) -> bool:
        """
        Check if a step has accumulated enough evidence to be marked VALID.

        RULE 1: One frame can never complete a step.
        RULE 2: Temporal evidence is required.
        RULE 8: Low confidence → UNCERTAIN, not VALID.
        """
        cc = step.completion_conditions

        # Must have enough temporal frames
        if step_state.evidence_frames < cc.min_temporal_frames:
            return False

        # Check recent evidence window (last min_temporal_frames).
        # For steps with state transitions, early frames naturally have lower
        # scores before the transition occurs, so the all-time average is
        # unfairly penalized. Use the recent window for a fairer check.
        recent_history = step_state.evidence_history[-cc.min_temporal_frames:]
        if recent_history:
            recent_avg = sum(h["score"] for h in recent_history) / len(recent_history)
        else:
            recent_avg = step_state.avg_evidence_score

        if recent_avg < cc.min_evidence_score:
            return False

        # Check required object states
        for obj_id, required_state in cc.required_object_states.items():
            observed = evidence.object_states.get(obj_id)
            if observed != required_state:
                return False

        # Check object state preconditions are met
        for obj_id, required_state in step.object_state_preconditions.items():
            if self.object_states.get(obj_id) != required_state:
                return False

        return True

    def _check_out_of_order(self, evidence: Evidence) -> Optional[Dict[str, Any]]:
        """
        Check if the observed action/step is out of order.

        If the temporal model or evidence suggests a step that is NOT the
        current expected step, flag it.
        """
        if not evidence.fused_step_id:
            return None

        observed_step_id = evidence.fused_step_id

        if observed_step_id == self.current_step_id:
            return None  # Expected

        # Check if it's a valid next step
        current_step = self.config.get_step(self.current_step_id)
        if current_step and observed_step_id in current_step.allowed_next_steps:
            return None  # It's an allowed next step

        # Check if it's a completed step (repeat)
        if observed_step_id in self.completed_steps:
            return None  # Already done, ignore

        # Check if this step requires a predecessor that hasn't been done
        observed_step = self.config.get_step(observed_step_id)
        if observed_step:
            required_prev = set(observed_step.allowed_previous_steps) - {"START"}
            completed = set(self.completed_steps)
            if required_prev and not required_prev.intersection(completed):
                # This step requires a previous step that hasn't been completed
                anomaly = {
                    "type": "OUT_OF_ORDER",
                    "timestamp": evidence.timestamp,
                    "observed_step": observed_step_id,
                    "expected_step": self.current_step_id,
                    "confidence": evidence.fused_confidence,
                    "reason": (
                        f"Step '{observed_step_id}' observed but requires "
                        f"completion of {required_prev} first"
                    ),
                }
                return anomaly

        # Check if we skipped steps
        if self.rules.strict_ordering:
            expected_idx = self._get_step_index(self.current_step_id)
            observed_idx = self._get_step_index(observed_step_id)
            if observed_idx is not None and expected_idx is not None:
                if observed_idx > expected_idx + 1:
                    # Steps were skipped
                    skipped = [
                        self.config.steps[i].id
                        for i in range(expected_idx, observed_idx)
                        if self.config.steps[i].id not in self.completed_steps
                    ]
                    return {
                        "type": "SKIPPED_STEPS",
                        "timestamp": evidence.timestamp,
                        "observed_step": observed_step_id,
                        "expected_step": self.current_step_id,
                        "skipped_steps": skipped,
                        "confidence": evidence.fused_confidence,
                        "reason": f"Steps {skipped} appear to have been skipped",
                    }
                elif observed_idx < expected_idx:
                    return {
                        "type": "OUT_OF_ORDER",
                        "timestamp": evidence.timestamp,
                        "observed_step": observed_step_id,
                        "expected_step": self.current_step_id,
                        "confidence": evidence.fused_confidence,
                        "reason": (
                            f"Step '{observed_step_id}' (index {observed_idx}) "
                            f"observed before expected step "
                            f"'{self.current_step_id}' (index {expected_idx})"
                        ),
                    }

        return None

    def _advance_to_next_step(self, completed_step, timestamp: float):
        """Advance to the next expected step after completing one."""
        next_ids = completed_step.allowed_next_steps

        if "COMPLETE" in next_ids and len(next_ids) == 1:
            # Experiment is complete
            self.experiment_status = ExperimentStatus.COMPLETE
            self.current_step_id = None
            return

        # Find the first non-completed next step
        for next_id in next_ids:
            if next_id == "COMPLETE":
                continue
            if next_id not in self.completed_steps:
                self.current_step_id = next_id
                self.step_states[next_id].status = StepStatus.EXPECTED
                return

        # All next steps are completed or only COMPLETE remains
        if "COMPLETE" in next_ids:
            self.experiment_status = ExperimentStatus.COMPLETE
            self.current_step_id = None
        else:
            logger.warning(f"No valid next step found after {completed_step.id}")

    def _get_next_valid_steps(self) -> List[str]:
        """Get list of valid next steps from current position."""
        if self.current_step_id is None:
            return []
        current_step = self.config.get_step(self.current_step_id)
        if current_step is None:
            return []
        return [
            sid for sid in current_step.allowed_next_steps
            if sid != "START"
        ]

    def _get_step_index(self, step_id: str) -> Optional[int]:
        """Get the index of a step in the step list."""
        for i, step in enumerate(self.config.steps):
            if step.id == step_id:
                return i
        return None

    def _transition_step(
        self,
        step_state: StepState,
        new_status: StepStatus,
        timestamp: float,
        evidence: Dict[str, Any],
        reason: str,
    ):
        """Record a state transition with full evidence trail."""
        old_status = step_state.status
        step_state.status = new_status

        record = StateTransitionRecord(
            timestamp=timestamp,
            step_id=step_state.step_id,
            from_status=old_status,
            to_status=new_status,
            evidence=evidence,
            reason=reason,
        )
        self.transition_history.append(record)

        logger.info(
            f"Step {step_state.step_id}: {old_status.value} → {new_status.value} "
            f"| {reason}"
        )

    def get_state_summary(self) -> Dict[str, Any]:
        """Get a complete summary of the current experiment state."""
        return {
            "experiment_id": self.config.experiment_id,
            "experiment_status": self.experiment_status.value,
            "current_step_id": self.current_step_id,
            "completed_steps": list(self.completed_steps),
            "step_states": {
                sid: {
                    "status": ss.status.value,
                    "evidence_frames": ss.evidence_frames,
                    "avg_evidence_score": round(ss.avg_evidence_score, 4),
                }
                for sid, ss in self.step_states.items()
            },
            "object_states": dict(self.object_states),
            "anomalies": list(self.anomalies),
            "transition_count": len(self.transition_history),
        }

    def explain_step_decision(self, step_id: str) -> str:
        """
        Explain why a step was marked with its current status.
        Returns human-readable explanation with evidence.
        """
        step_state = self.step_states.get(step_id)
        if step_state is None:
            return f"Unknown step: {step_id}"

        lines = [
            f"Step: {step_id}",
            f"Status: {step_state.status.value}",
            f"Evidence frames: {step_state.evidence_frames}",
            f"Average evidence score: {step_state.avg_evidence_score:.4f}",
            "",
            "Transition History:",
        ]

        for record in self.transition_history:
            if record.step_id == step_id:
                lines.append(
                    f"  [{record.timestamp:.2f}] "
                    f"{record.from_status.value} → {record.to_status.value}: "
                    f"{record.reason}"
                )
                if "breakdown" in record.evidence:
                    for source, score in record.evidence["breakdown"].items():
                        lines.append(f"    {source}: {score:.4f}")

        return "\n".join(lines)

    def reset(self):
        """Reset the state machine to initial state."""
        self.__init__(self.config)
