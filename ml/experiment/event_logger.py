"""
ASTRA — Lightweight Event Logger

Produces human-readable JSONL event logs for experiment tracking.
Every validated event is timestamped and evidence-backed.

Event Types:
    STEP_STARTED
    STEP_COMPLETED
    STEP_SKIPPED
    STEP_OUT_OF_ORDER
    STEP_FAILED
    STEP_UNCERTAIN
    STEP_TIMEOUT
    ANOMALY
    NEXT_STEP_SUGGESTED
    VOICE_ALERT
    EXPERIMENT_STARTED
    EXPERIMENT_COMPLETED
    OBJECT_STATE_CHANGED
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm."""
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


@dataclass
class ExperimentEvent:
    """A single experiment event."""
    timestamp: float  # seconds from experiment start
    event: str  # event type
    step_id: Optional[str] = None
    action: Optional[str] = None
    objects: Optional[List[str]] = None
    status: Optional[str] = None
    confidence: Optional[float] = None
    evidence: Optional[Dict[str, float]] = None
    message: Optional[str] = None
    anomaly_type: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a clean dict, omitting None values."""
        d = {"timestamp": format_timestamp(self.timestamp), "event": self.event}
        if self.step_id is not None:
            d["step_id"] = self.step_id
        if self.action is not None:
            d["action"] = self.action
        if self.objects is not None:
            d["objects"] = self.objects
        if self.status is not None:
            d["status"] = self.status
        if self.confidence is not None:
            d["confidence"] = round(self.confidence, 4)
        if self.evidence is not None:
            d["evidence"] = {
                k: round(v, 4) for k, v in self.evidence.items()
            }
        if self.message is not None:
            d["message"] = self.message
        if self.anomaly_type is not None:
            d["anomaly_type"] = self.anomaly_type
        if self.details is not None:
            d["details"] = self.details
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


class EventLogger:
    """
    Lightweight experiment event logger.

    Writes JSONL files with one event per line.
    Also maintains in-memory event history for programmatic access.
    """

    def __init__(
        self,
        log_dir: str,
        experiment_id: str,
        session_id: str = "session_00",
    ):
        self.log_dir = log_dir
        self.experiment_id = experiment_id
        self.session_id = session_id
        self.events: List[ExperimentEvent] = []

        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(
            log_dir,
            f"{experiment_id}_{session_id}_events.jsonl",
        )

        # Open file for appending
        self._file = open(self.log_path, "a", encoding="utf-8")

    def log_event(self, event: ExperimentEvent):
        """Log a single event."""
        self.events.append(event)
        self._file.write(event.to_json() + "\n")
        self._file.flush()
        logger.debug(f"Event: {event.event} | {event.step_id} | {event.message}")

    def log_step_started(
        self,
        timestamp: float,
        step_id: str,
        step_name: str = "",
    ):
        self.log_event(ExperimentEvent(
            timestamp=timestamp,
            event="STEP_STARTED",
            step_id=step_id,
            message=f"Started: {step_name or step_id}",
        ))

    def log_step_completed(
        self,
        timestamp: float,
        step_id: str,
        action: str = "",
        objects: Optional[List[str]] = None,
        confidence: float = 0.0,
        evidence: Optional[Dict[str, float]] = None,
    ):
        self.log_event(ExperimentEvent(
            timestamp=timestamp,
            event="STEP_COMPLETED",
            step_id=step_id,
            action=action,
            objects=objects,
            status="VALID",
            confidence=confidence,
            evidence=evidence,
        ))

    def log_step_skipped(
        self,
        timestamp: float,
        step_id: str,
        reason: str = "",
    ):
        self.log_event(ExperimentEvent(
            timestamp=timestamp,
            event="STEP_SKIPPED",
            step_id=step_id,
            status="SKIPPED",
            message=reason,
        ))

    def log_step_out_of_order(
        self,
        timestamp: float,
        observed_step: str,
        expected_step: str,
        confidence: float = 0.0,
    ):
        self.log_event(ExperimentEvent(
            timestamp=timestamp,
            event="STEP_OUT_OF_ORDER",
            step_id=observed_step,
            status="OUT_OF_ORDER",
            confidence=confidence,
            message=f"Expected '{expected_step}', observed '{observed_step}'",
            details={"expected_step": expected_step},
        ))

    def log_step_failed(
        self,
        timestamp: float,
        step_id: str,
        reason: str = "",
    ):
        self.log_event(ExperimentEvent(
            timestamp=timestamp,
            event="STEP_FAILED",
            step_id=step_id,
            status="FAILED",
            message=reason,
        ))

    def log_step_uncertain(
        self,
        timestamp: float,
        step_id: str,
        confidence: float = 0.0,
    ):
        self.log_event(ExperimentEvent(
            timestamp=timestamp,
            event="STEP_UNCERTAIN",
            step_id=step_id,
            status="UNCERTAIN",
            confidence=confidence,
            message="Insufficient evidence to confirm step",
        ))

    def log_anomaly(
        self,
        timestamp: float,
        anomaly_type: str,
        message: str,
        step_id: Optional[str] = None,
        confidence: float = 0.0,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.log_event(ExperimentEvent(
            timestamp=timestamp,
            event="ANOMALY",
            step_id=step_id,
            anomaly_type=anomaly_type,
            confidence=confidence,
            message=message,
            details=details,
        ))

    def log_next_step_suggested(
        self,
        timestamp: float,
        next_step_id: str,
        next_step_name: str = "",
    ):
        self.log_event(ExperimentEvent(
            timestamp=timestamp,
            event="NEXT_STEP_SUGGESTED",
            step_id=next_step_id,
            message=f"Next step: {next_step_name or next_step_id}",
        ))

    def log_voice_alert(
        self,
        timestamp: float,
        message: str,
        step_id: Optional[str] = None,
    ):
        self.log_event(ExperimentEvent(
            timestamp=timestamp,
            event="VOICE_ALERT",
            step_id=step_id,
            message=message,
        ))

    def log_experiment_started(self, timestamp: float):
        self.log_event(ExperimentEvent(
            timestamp=timestamp,
            event="EXPERIMENT_STARTED",
            message=f"Experiment {self.experiment_id} started",
        ))

    def log_experiment_completed(self, timestamp: float, confidence: float = 0.0):
        self.log_event(ExperimentEvent(
            timestamp=timestamp,
            event="EXPERIMENT_COMPLETED",
            status="COMPLETE",
            confidence=confidence,
            message=f"Experiment {self.experiment_id} completed",
        ))

    def log_object_state_changed(
        self,
        timestamp: float,
        object_id: str,
        from_state: str,
        to_state: str,
        confidence: float = 0.0,
    ):
        self.log_event(ExperimentEvent(
            timestamp=timestamp,
            event="OBJECT_STATE_CHANGED",
            objects=[object_id],
            confidence=confidence,
            message=f"{object_id}: {from_state} → {to_state}",
            details={"from_state": from_state, "to_state": to_state},
        ))

    def get_events(self, event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get logged events, optionally filtered by type."""
        events = self.events
        if event_type:
            events = [e for e in events if e.event == event_type]
        return [e.to_dict() for e in events]

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all logged events."""
        from collections import Counter
        event_counts = Counter(e.event for e in self.events)
        return {
            "experiment_id": self.experiment_id,
            "session_id": self.session_id,
            "total_events": len(self.events),
            "event_counts": dict(event_counts),
            "log_path": self.log_path,
        }

    def close(self):
        """Close the log file."""
        if self._file and not self._file.closed:
            self._file.close()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
