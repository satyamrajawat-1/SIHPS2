"""
ASTRA Experiment Schema Loader

Loads and validates experiment configuration from YAML files.
Provides structured access to experiment steps, objects, actions, and rules.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class CompletionConditions:
    """Conditions required to mark a step as complete."""
    min_evidence_score: float = 0.8
    min_temporal_frames: int = 10
    required_object_states: Dict[str, str] = field(default_factory=dict)


@dataclass
class FailureConditions:
    """Conditions that mark a step as failed."""
    max_attempts: int = 3
    timeout: bool = True
    forbidden_actions: List[str] = field(default_factory=list)


@dataclass
class ExperimentStep:
    """A single step in the experiment procedure."""
    id: str
    name: str
    description: str
    required_actions: List[str]
    required_objects: List[str]
    object_state_preconditions: Dict[str, str]
    object_state_effects: Dict[str, str]
    allowed_previous_steps: List[str]
    allowed_next_steps: List[str]
    timeout_seconds: float
    is_optional: bool
    completion_conditions: CompletionConditions
    failure_conditions: FailureConditions

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentStep":
        cc = data.get("completion_conditions", {})
        fc = data.get("failure_conditions", {})
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            required_actions=data.get("required_actions", []),
            required_objects=data.get("required_objects", []),
            object_state_preconditions=data.get("object_state_preconditions", {}),
            object_state_effects=data.get("object_state_effects", {}),
            allowed_previous_steps=data.get("allowed_previous_steps", []),
            allowed_next_steps=data.get("allowed_next_steps", []),
            timeout_seconds=data.get("timeout_seconds", 120.0),
            is_optional=data.get("is_optional", False),
            completion_conditions=CompletionConditions(
                min_evidence_score=cc.get("min_evidence_score", 0.8),
                min_temporal_frames=cc.get("min_temporal_frames", 10),
                required_object_states=cc.get("required_object_states", {}),
            ),
            failure_conditions=FailureConditions(
                max_attempts=fc.get("max_attempts", 3),
                timeout=fc.get("timeout", True),
                forbidden_actions=fc.get("forbidden_actions", []),
            ),
        )


@dataclass
class StateTransition:
    """A valid state transition for an object."""
    from_state: str
    to_state: str
    trigger_actions: List[str]


@dataclass
class ExperimentObject:
    """An object involved in the experiment."""
    id: str
    name: str
    obj_class: str
    expected_initial_state: str
    possible_states: List[str]
    state_transitions: List[StateTransition]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentObject":
        transitions = []
        for t in data.get("state_transitions", []):
            transitions.append(StateTransition(
                from_state=t["from"],
                to_state=t["to"],
                trigger_actions=t.get("trigger_actions", []),
            ))
        return cls(
            id=data["id"],
            name=data["name"],
            obj_class=data.get("class", data["name"].lower()),
            expected_initial_state=data.get("expected_initial_state", "unknown"),
            possible_states=data.get("possible_states", []),
            state_transitions=transitions,
        )

    def get_valid_transitions(self, current_state: str) -> List[StateTransition]:
        """Get all valid transitions from the current state."""
        return [t for t in self.state_transitions if t.from_state == current_state]


@dataclass
class PrimitiveAction:
    """A primitive action the system can recognize."""
    id: str
    name: str
    primitive_type: str
    description: str
    typical_duration_seconds: float
    involves_hand: str
    requires_object: bool

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PrimitiveAction":
        return cls(
            id=data["id"],
            name=data["name"],
            primitive_type=data.get("primitive_type", "unknown"),
            description=data.get("description", ""),
            typical_duration_seconds=data.get("typical_duration_seconds", 1.0),
            involves_hand=data.get("involves_hand", "any"),
            requires_object=data.get("requires_object", True),
        )


@dataclass
class ExperimentRules:
    """Validation rules for the experiment."""
    min_temporal_frames_for_step: int = 10
    min_evidence_score: float = 0.75
    uncertain_threshold: float = 0.5
    max_step_timeout_seconds: float = 120.0
    allow_step_retry: bool = True
    max_retries_per_step: int = 3
    strict_ordering: bool = True
    confidence_calibration: str = "temperature_scaling"
    abstain_enabled: bool = True
    abstain_threshold: float = 0.4
    multi_camera_fusion: str = "confidence_weighted"
    evidence_fusion_method: str = "rule_based"
    evidence_weights: Dict[str, float] = field(default_factory=lambda: {
        "object_detection": 0.20,
        "pose": 0.10,
        "action_recognition": 0.20,
        "hoi": 0.20,
        "object_state": 0.20,
        "temporal_consistency": 0.10,
    })

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentRules":
        rules_data = data.get("rules", data)
        return cls(
            min_temporal_frames_for_step=rules_data.get("min_temporal_frames_for_step", 10),
            min_evidence_score=rules_data.get("min_evidence_score", 0.75),
            uncertain_threshold=rules_data.get("uncertain_threshold", 0.5),
            max_step_timeout_seconds=rules_data.get("max_step_timeout_seconds", 120.0),
            allow_step_retry=rules_data.get("allow_step_retry", True),
            max_retries_per_step=rules_data.get("max_retries_per_step", 3),
            strict_ordering=rules_data.get("strict_ordering", True),
            confidence_calibration=rules_data.get("confidence_calibration", "temperature_scaling"),
            abstain_enabled=rules_data.get("abstain_enabled", True),
            abstain_threshold=rules_data.get("abstain_threshold", 0.4),
            multi_camera_fusion=rules_data.get("multi_camera_fusion", "confidence_weighted"),
            evidence_fusion_method=rules_data.get("evidence_fusion_method", "rule_based"),
            evidence_weights=rules_data.get("evidence_weights", {}),
        )


@dataclass
class ExperimentConfig:
    """Complete experiment configuration loaded from YAML files."""
    experiment_id: str
    experiment_name: str
    description: str
    version: str
    steps: List[ExperimentStep]
    objects: List[ExperimentObject]
    actions: List[PrimitiveAction]
    rules: ExperimentRules

    # Lookup maps (built after loading)
    _steps_by_id: Dict[str, ExperimentStep] = field(default_factory=dict, repr=False)
    _objects_by_id: Dict[str, ExperimentObject] = field(default_factory=dict, repr=False)
    _actions_by_id: Dict[str, PrimitiveAction] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._steps_by_id = {s.id: s for s in self.steps}
        self._objects_by_id = {o.id: o for o in self.objects}
        self._actions_by_id = {a.id: a for a in self.actions}

    def get_step(self, step_id: str) -> Optional[ExperimentStep]:
        return self._steps_by_id.get(step_id)

    def get_object(self, object_id: str) -> Optional[ExperimentObject]:
        return self._objects_by_id.get(object_id)

    def get_action(self, action_id: str) -> Optional[PrimitiveAction]:
        return self._actions_by_id.get(action_id)

    def get_step_ids(self) -> List[str]:
        return [s.id for s in self.steps]

    def get_object_ids(self) -> List[str]:
        return [o.id for o in self.objects]

    def get_action_ids(self) -> List[str]:
        return [a.id for a in self.actions]

    def get_object_classes(self) -> List[str]:
        """Get unique detection class names for all objects."""
        return list(set(o.obj_class for o in self.objects))

    def get_first_step(self) -> Optional[ExperimentStep]:
        """Get the first step (one with START in allowed_previous_steps)."""
        for step in self.steps:
            if "START" in step.allowed_previous_steps:
                return step
        return self.steps[0] if self.steps else None

    def get_final_steps(self) -> List[ExperimentStep]:
        """Get steps that can lead to COMPLETE."""
        return [s for s in self.steps if "COMPLETE" in s.allowed_next_steps]

    def validate(self) -> List[str]:
        """
        Validate the experiment configuration for consistency.
        Returns a list of error messages (empty if valid).
        """
        errors = []

        # Check step references
        step_ids = set(self.get_step_ids())
        special_ids = {"START", "COMPLETE"}

        for step in self.steps:
            for prev_id in step.allowed_previous_steps:
                if prev_id not in step_ids and prev_id not in special_ids:
                    errors.append(
                        f"Step {step.id}: allowed_previous_steps references "
                        f"unknown step '{prev_id}'"
                    )
            for next_id in step.allowed_next_steps:
                if next_id not in step_ids and next_id not in special_ids:
                    errors.append(
                        f"Step {step.id}: allowed_next_steps references "
                        f"unknown step '{next_id}'"
                    )

            # Check required objects exist
            obj_ids = set(self.get_object_ids())
            for obj_id in step.required_objects:
                if obj_id not in obj_ids:
                    errors.append(
                        f"Step {step.id}: required_objects references "
                        f"unknown object '{obj_id}'"
                    )

            # Check required actions exist
            action_ids = set(self.get_action_ids())
            for act_id in step.required_actions:
                if act_id not in action_ids:
                    errors.append(
                        f"Step {step.id}: required_actions references "
                        f"unknown action '{act_id}'"
                    )

            # Check object state preconditions reference valid objects and states
            for obj_id, state in step.object_state_preconditions.items():
                if obj_id not in obj_ids:
                    errors.append(
                        f"Step {step.id}: precondition references "
                        f"unknown object '{obj_id}'"
                    )
                else:
                    obj = self.get_object(obj_id)
                    if obj and state not in obj.possible_states:
                        errors.append(
                            f"Step {step.id}: precondition '{obj_id}' state "
                            f"'{state}' not in possible_states"
                        )

        # Check for unreachable steps
        reachable = {"START"}
        changed = True
        while changed:
            changed = False
            for step in self.steps:
                if step.id not in reachable:
                    if any(prev in reachable for prev in step.allowed_previous_steps):
                        reachable.add(step.id)
                        changed = True
        unreachable = step_ids - reachable
        for uid in unreachable:
            errors.append(f"Step {uid} is unreachable from START")

        # Check that at least one step leads to COMPLETE
        if not self.get_final_steps():
            errors.append("No step has COMPLETE in allowed_next_steps")

        return errors


def load_experiment(experiment_dir: str | Path) -> ExperimentConfig:
    """
    Load a complete experiment configuration from a directory.

    The directory must contain:
        experiment.yaml
        objects.yaml
        actions.yaml
        rules.yaml

    Args:
        experiment_dir: Path to the experiment configuration directory.

    Returns:
        ExperimentConfig with all components loaded and validated.

    Raises:
        FileNotFoundError: If required files are missing.
        ValueError: If configuration is invalid.
    """
    experiment_dir = Path(experiment_dir)

    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

    # Load experiment.yaml
    exp_path = experiment_dir / "experiment.yaml"
    if not exp_path.exists():
        raise FileNotFoundError(f"experiment.yaml not found in {experiment_dir}")
    with open(exp_path, "r", encoding="utf-8") as f:
        exp_data = yaml.safe_load(f)

    # Load objects.yaml
    obj_path = experiment_dir / "objects.yaml"
    if not obj_path.exists():
        raise FileNotFoundError(f"objects.yaml not found in {experiment_dir}")
    with open(obj_path, "r", encoding="utf-8") as f:
        obj_data = yaml.safe_load(f)

    # Load actions.yaml
    act_path = experiment_dir / "actions.yaml"
    if not act_path.exists():
        raise FileNotFoundError(f"actions.yaml not found in {experiment_dir}")
    with open(act_path, "r", encoding="utf-8") as f:
        act_data = yaml.safe_load(f)

    # Load rules.yaml
    rules_path = experiment_dir / "rules.yaml"
    if not rules_path.exists():
        raise FileNotFoundError(f"rules.yaml not found in {experiment_dir}")
    with open(rules_path, "r", encoding="utf-8") as f:
        rules_data = yaml.safe_load(f)

    # Parse components
    steps = [ExperimentStep.from_dict(s) for s in exp_data.get("steps", [])]
    objects = [ExperimentObject.from_dict(o) for o in obj_data.get("objects", [])]
    actions = [PrimitiveAction.from_dict(a) for a in act_data.get("actions", [])]
    rules = ExperimentRules.from_dict(rules_data)

    config = ExperimentConfig(
        experiment_id=exp_data.get("experiment_id", "unknown"),
        experiment_name=exp_data.get("experiment_name", "Unknown Experiment"),
        description=exp_data.get("description", ""),
        version=exp_data.get("version", "0.0.0"),
        steps=steps,
        objects=objects,
        actions=actions,
        rules=rules,
    )

    # Validate
    errors = config.validate()
    if errors:
        error_msg = "\n".join(f"  - {e}" for e in errors)
        raise ValueError(
            f"Experiment configuration validation failed:\n{error_msg}"
        )

    return config


def list_experiments(experiments_root: str | Path) -> List[str]:
    """List all available experiment directories."""
    experiments_root = Path(experiments_root)
    if not experiments_root.is_dir():
        return []
    return [
        d.name
        for d in experiments_root.iterdir()
        if d.is_dir() and (d / "experiment.yaml").exists()
    ]
