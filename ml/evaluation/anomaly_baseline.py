"""
ASTRA — Deterministic Anomaly Baseline

Uses SOP state + observed action + object evidence + temporal persistence + confidence
to classify step status.

Classifications:
    VALID, UNCERTAIN, SKIPPED, OUT_OF_ORDER, UNEXPECTED_ACTION, TIMEOUT, FAILED

This is the FIRST anomaly model — purely deterministic, no neural weights.
Must be beaten before a learned model is justified.

Usage:
    from ml.evaluation.anomaly_baseline import DeterministicAnomalyClassifier
    classifier = DeterministicAnomalyClassifier(thresholds)
    result = classifier.classify(step_evidence)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AnomalyThresholds:
    """Tunable thresholds for anomaly classification."""
    # Step completion
    step_completion_threshold: float = 0.6
    step_uncertain_threshold: float = 0.3

    # Action confidence
    action_confidence_min: float = 0.3
    action_match_threshold: float = 0.5

    # Object evidence
    object_presence_threshold: float = 0.4

    # HOI
    hoi_confidence_threshold: float = 0.3

    # Temporal
    temporal_persistence_frames: int = 5
    timeout_seconds: float = 120.0

    # Anomaly scoring
    out_of_order_threshold: float = 0.5
    skip_detection_threshold: float = 0.5


@dataclass
class StepEvidence:
    """Evidence packet for a single step evaluation."""
    step_id: str
    expected_step_id: str
    action_detected: Optional[str] = None
    action_confidence: float = 0.0
    required_actions: List[str] = field(default_factory=list)
    objects_detected: List[str] = field(default_factory=list)
    required_objects: List[str] = field(default_factory=list)
    hoi_detected: bool = False
    hoi_confidence: float = 0.0
    temporal_confidence: float = 0.0
    persistence_frames: int = 0
    elapsed_seconds: float = 0.0
    step_order_index: int = 0
    expected_order_index: int = 0


@dataclass
class AnomalyResult:
    """Classification result."""
    status: str  # VALID, UNCERTAIN, SKIPPED, OUT_OF_ORDER, UNEXPECTED_ACTION, TIMEOUT, FAILED
    confidence: float
    reasons: List[str] = field(default_factory=list)
    evidence_scores: Dict[str, float] = field(default_factory=dict)


class DeterministicAnomalyClassifier:
    """
    Rule-based anomaly classifier.

    No neural weights. No learning. Pure deterministic logic.
    """

    def __init__(self, thresholds: Optional[AnomalyThresholds] = None):
        self.thresholds = thresholds or AnomalyThresholds()

    def classify(self, evidence: StepEvidence) -> AnomalyResult:
        """
        Classify a step's status based on available evidence.

        Returns AnomalyResult with status and reasons.
        """
        t = self.thresholds
        reasons = []
        scores = {}

        # 1. Check for TIMEOUT
        if evidence.elapsed_seconds > t.timeout_seconds:
            return AnomalyResult(
                status="TIMEOUT", confidence=0.9,
                reasons=[f"Step exceeded timeout ({evidence.elapsed_seconds:.0f}s > {t.timeout_seconds}s)"],
                evidence_scores={"timeout": 1.0},
            )

        # 2. Check for OUT_OF_ORDER
        if evidence.step_id != evidence.expected_step_id:
            if evidence.step_order_index > evidence.expected_order_index:
                return AnomalyResult(
                    status="OUT_OF_ORDER", confidence=0.8,
                    reasons=[f"Step {evidence.step_id} observed before expected {evidence.expected_step_id}"],
                    evidence_scores={"order_mismatch": 1.0},
                )

        # 3. Action matching
        action_score = 0.0
        if evidence.action_detected and evidence.required_actions:
            if evidence.action_detected in evidence.required_actions:
                action_score = evidence.action_confidence
            else:
                if evidence.action_confidence > t.action_confidence_min:
                    reasons.append(
                        f"Unexpected action: {evidence.action_detected} "
                        f"(expected: {evidence.required_actions})")
                    scores["unexpected_action"] = evidence.action_confidence
        elif not evidence.required_actions:
            action_score = 0.5  # No action requirement
        scores["action"] = action_score

        # 4. Object evidence
        object_score = 0.0
        if evidence.required_objects:
            present = set(evidence.objects_detected) & set(evidence.required_objects)
            object_score = len(present) / len(evidence.required_objects)
            if object_score < t.object_presence_threshold:
                missing = set(evidence.required_objects) - set(evidence.objects_detected)
                reasons.append(f"Missing objects: {missing}")
        else:
            object_score = 0.5
        scores["objects"] = object_score

        # 5. HOI evidence
        hoi_score = evidence.hoi_confidence if evidence.hoi_detected else 0.0
        scores["hoi"] = hoi_score

        # 6. Temporal persistence
        persistence_score = min(1.0, evidence.persistence_frames / max(1, t.temporal_persistence_frames))
        scores["persistence"] = persistence_score

        # 7. Temporal model confidence
        scores["temporal"] = evidence.temporal_confidence

        # 8. Composite score
        composite = (
            0.25 * action_score +
            0.20 * object_score +
            0.15 * hoi_score +
            0.20 * persistence_score +
            0.20 * evidence.temporal_confidence
        )
        scores["composite"] = round(composite, 4)

        # 9. Classify
        if "unexpected_action" in scores and scores["unexpected_action"] > t.action_match_threshold:
            return AnomalyResult(
                status="UNEXPECTED_ACTION", confidence=scores["unexpected_action"],
                reasons=reasons, evidence_scores=scores)

        if composite >= t.step_completion_threshold:
            return AnomalyResult(
                status="VALID", confidence=composite,
                reasons=["All evidence supports step completion"],
                evidence_scores=scores)

        if composite >= t.step_uncertain_threshold:
            reasons.append(f"Composite score {composite:.3f} below completion threshold {t.step_completion_threshold}")
            return AnomalyResult(
                status="UNCERTAIN", confidence=composite,
                reasons=reasons, evidence_scores=scores)

        reasons.append(f"Very low composite score: {composite:.3f}")
        return AnomalyResult(
            status="FAILED", confidence=composite,
            reasons=reasons, evidence_scores=scores)

    def classify_skipped(
        self,
        expected_step_id: str,
        completed_steps: List[str],
        current_step_id: str,
        all_step_ids: List[str],
    ) -> Optional[AnomalyResult]:
        """Check if a step was skipped."""
        if expected_step_id in completed_steps:
            return None  # Not skipped

        try:
            expected_idx = all_step_ids.index(expected_step_id)
            current_idx = all_step_ids.index(current_step_id)
        except ValueError:
            return None

        if current_idx > expected_idx + 1:
            # Steps between expected and current were skipped
            skipped = all_step_ids[expected_idx:current_idx]
            skipped = [s for s in skipped if s not in completed_steps]
            if skipped:
                return AnomalyResult(
                    status="SKIPPED", confidence=0.85,
                    reasons=[f"Steps skipped: {skipped}"],
                    evidence_scores={"skipped_count": len(skipped)})

        return None
