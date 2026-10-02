# ASTRA — Experiment Configuration Schema
#
# This file documents the experiment YAML configuration format.
# Every experiment is defined by YAML files in experiments/<experiment_id>/
#
# Files:
#   experiment.yaml  — Experiment metadata, steps, and flow
#   objects.yaml     — Object definitions and states
#   actions.yaml     — Primitive action taxonomy
#   rules.yaml       — Validation rules and constraints

# ============================================================
# experiment.yaml
# ============================================================
#
# experiment_id: str          — Unique experiment identifier
# experiment_name: str        — Human-readable name
# description: str            — Description of the experiment
# version: str                — Configuration version
#
# steps:
#   - id: str                 — Unique step identifier (e.g., "step_01")
#     name: str               — Human-readable step name
#     description: str        — What this step involves
#     required_actions: list  — Action IDs that must be observed
#     required_objects: list  — Object IDs that must be present
#     object_state_preconditions: dict  — Required object states before step
#     object_state_effects: dict        — Expected object states after step
#     allowed_previous_steps: list      — Valid predecessor step IDs
#     allowed_next_steps: list          — Valid successor step IDs
#     timeout_seconds: float            — Max duration for this step
#     is_optional: bool                 — Whether step can be skipped
#     completion_conditions:            — Conditions that mark step complete
#       min_evidence_score: float
#       min_temporal_frames: int
#       required_object_states: dict
#     failure_conditions:               — Conditions that mark step failed
#       max_attempts: int
#       timeout: bool
#       forbidden_actions: list

# ============================================================
# objects.yaml
# ============================================================
#
# objects:
#   - id: str                 — Unique object identifier
#     name: str               — Human-readable name
#     class: str              — Detection class name (for YOLO)
#     expected_initial_state: str  — State at experiment start
#     possible_states: list   — All possible states
#     state_transitions: list — Valid state transitions
#       - from: str
#         to: str
#         trigger_actions: list

# ============================================================
# actions.yaml
# ============================================================
#
# actions:
#   - id: str                 — Unique action identifier
#     name: str               — Human-readable name
#     primitive_type: str     — Category (reach, grasp, etc.)
#     description: str        — What this action looks like
#     typical_duration_seconds: float  — Expected duration
#     involves_hand: str      — "left", "right", "both", "any"
#     requires_object: bool   — Whether an object must be involved

# ============================================================
# rules.yaml
# ============================================================
#
# rules:
#   min_temporal_frames_for_step: int     — Minimum frames to confirm step
#   min_evidence_score: float             — Minimum fused evidence for VALID
#   uncertain_threshold: float            — Below this → UNCERTAIN
#   max_step_timeout_seconds: float       — Global step timeout
#   allow_step_retry: bool                — Whether steps can be retried
#   max_retries_per_step: int             — Max retry count
#   strict_ordering: bool                 — Whether ordering is strictly enforced
