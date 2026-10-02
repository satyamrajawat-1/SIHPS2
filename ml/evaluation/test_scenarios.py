"""
ASTRA — Realistic Test Scenarios

10 predefined scenarios for evaluating the state machine + anomaly detection.
Each scenario produces a different state outcome.

Usage:
    python -m ml.evaluation.test_scenarios --experiment experiments/sample_experiment/
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_scenario(scenario_fn, config, label):
    """Run a scenario and return results."""
    from ml.experiment.state_graph.state_machine import ExperimentStateMachine

    sm = ExperimentStateMachine(config)
    result = scenario_fn(sm, config)
    summary = sm.get_state_summary()

    return {
        "scenario": label,
        "experiment_status": summary.get("experiment_status"),
        "completed_steps": summary.get("completed_steps", []),
        "current_step": summary.get("current_step_id"),
        "anomalies": summary.get("anomalies", []),
        "step_statuses": {sid: ss.status.value for sid, ss in sm.step_states.items()},
        **result,
    }


def _make_evidence(step, config, confidence=0.8, hoi=True, objects=True, frame_id=0):
    """Create an Evidence dataclass for completing a step."""
    from ml.experiment.state_graph.state_machine import Evidence

    detected_objects = {}
    hand_near = {}
    if objects and step.required_objects:
        for obj_id in step.required_objects:
            detected_objects[obj_id] = confidence
            hand_near[obj_id] = confidence * 0.9

    hoi_list = []
    if hoi and step.required_objects:
        hoi_list = [{"hand": "right", "object_id": step.required_objects[0],
                      "type": "manipulating", "confidence": confidence}]

    action_id = step.required_actions[-1] if step.required_actions else "idle"

    return Evidence(
        timestamp=frame_id / 30.0,
        frame_id=frame_id,
        detected_objects=detected_objects,
        hand_near_objects=hand_near,
        detected_action=action_id,
        action_confidence=confidence,
        hand_object_interactions=hoi_list,
        temporal_action=action_id,
        temporal_action_confidence=confidence,
        temporal_step_candidate=step.id,
        temporal_step_confidence=confidence,
        temporal_consistency=confidence,
        pose_confidence=confidence,
    )


def scenario_1_normal(sm, config):
    """SCENARIO 1: Normal experiment — all steps completed in order."""
    fid = 0
    for step in config.steps:
        for _ in range(8):
            ev = _make_evidence(step, config, confidence=0.85, frame_id=fid)
            sm.update(ev)
            fid += 1
    return {"expected_outcome": "COMPLETE"}


def scenario_2_skipped(sm, config):
    """SCENARIO 2: Skipped step — step_02 is skipped."""
    fid = 0
    for i, step in enumerate(config.steps):
        if i == 1:  # Skip step_02
            continue
        for _ in range(8):
            ev = _make_evidence(step, config, confidence=0.85, frame_id=fid)
            sm.update(ev)
            fid += 1
    return {"expected_outcome": "INCOMPLETE", "skipped": config.steps[1].id if len(config.steps) > 1 else None}


def scenario_3_out_of_order(sm, config):
    """SCENARIO 3: Out-of-order — step_03 before step_02."""
    steps = list(config.steps)
    if len(steps) >= 3:
        order = [steps[0], steps[2], steps[1]] + steps[3:]
    else:
        order = steps
    fid = 0
    for step in order:
        for _ in range(8):
            ev = _make_evidence(step, config, confidence=0.85, frame_id=fid)
            sm.update(ev)
            fid += 1
    return {"expected_outcome": "OUT_OF_ORDER_DETECTED"}


def scenario_4_unexpected_action(sm, config):
    """SCENARIO 4: Unexpected action during a step."""
    step = config.steps[0]
    fid = 0
    for _ in range(5):
        ev = _make_evidence(step, config, confidence=0.85, frame_id=fid)
        sm.update(ev)
        fid += 1
    # Insert unexpected action
    ev = _make_evidence(step, config, confidence=0.85, frame_id=fid)
    ev.detected_action = "unknown_action_xyz"
    sm.update(ev)
    fid += 1
    for _ in range(5):
        ev = _make_evidence(step, config, confidence=0.85, frame_id=fid)
        sm.update(ev)
        fid += 1
    return {"expected_outcome": "UNEXPECTED_ACTION_LOGGED"}


def scenario_5_object_occlusion(sm, config):
    """SCENARIO 5: Object occlusion — required objects missing intermittently."""
    step = config.steps[0]
    for i in range(12):
        ev = _make_evidence(step, config, confidence=0.85, objects=(i % 3 != 0), frame_id=i)
        sm.update(ev)
    return {"expected_outcome": "INTERMITTENT_OBJECTS"}


def scenario_6_hand_occlusion(sm, config):
    """SCENARIO 6: Hand occlusion — no HOI evidence."""
    step = config.steps[0]
    for i in range(12):
        ev = _make_evidence(step, config, confidence=0.7, hoi=False, frame_id=i)
        sm.update(ev)
    return {"expected_outcome": "NO_HOI_EVIDENCE"}


def scenario_7_low_confidence(sm, config):
    """SCENARIO 7: Low confidence — all evidence below threshold."""
    step = config.steps[0]
    for i in range(15):
        ev = _make_evidence(step, config, confidence=0.15, frame_id=i)
        sm.update(ev)
    return {"expected_outcome": "UNCERTAIN_OR_NOT_STARTED"}


def scenario_8_repeated_action(sm, config):
    """SCENARIO 8: Repeated action — same step evidence repeated many times."""
    step = config.steps[0]
    for i in range(30):
        ev = _make_evidence(step, config, confidence=0.85, frame_id=i)
        sm.update(ev)
    return {"expected_outcome": "STEP_COMPLETED_ONCE"}


def scenario_9_timeout(sm, config):
    """SCENARIO 9: Timeout — very long step with no progression."""
    step = config.steps[0]
    for i in range(5):
        ev = _make_evidence(step, config, confidence=0.2, frame_id=i)
        sm.update(ev)
    return {"expected_outcome": "TIMEOUT_WARNING"}


def scenario_10_incomplete(sm, config):
    """SCENARIO 10: Incomplete experiment — only first 3 steps."""
    fid = 0
    for step in config.steps[:3]:
        for _ in range(8):
            ev = _make_evidence(step, config, confidence=0.85, frame_id=fid)
            sm.update(ev)
            fid += 1
    return {"expected_outcome": "INCOMPLETE"}


ALL_SCENARIOS = [
    ("1_normal", scenario_1_normal),
    ("2_skipped_step", scenario_2_skipped),
    ("3_out_of_order", scenario_3_out_of_order),
    ("4_unexpected_action", scenario_4_unexpected_action),
    ("5_object_occlusion", scenario_5_object_occlusion),
    ("6_hand_occlusion", scenario_6_hand_occlusion),
    ("7_low_confidence", scenario_7_low_confidence),
    ("8_repeated_action", scenario_8_repeated_action),
    ("9_timeout", scenario_9_timeout),
    ("10_incomplete", scenario_10_incomplete),
]


def main():
    parser = argparse.ArgumentParser(description="ASTRA Test Scenarios")
    parser.add_argument("--experiment", required=True, help="Experiment config dir")
    parser.add_argument("--output", default="eval_results/scenarios")
    parser.add_argument("--scenario", type=int, help="Run specific scenario (1-10)")
    args = parser.parse_args()

    from ml.experiment.schema.loader import load_experiment
    config = load_experiment(args.experiment)

    os.makedirs(args.output, exist_ok=True)

    if args.scenario:
        scenarios = [ALL_SCENARIOS[args.scenario - 1]]
    else:
        scenarios = ALL_SCENARIOS

    results = []
    print("\n" + "=" * 64)
    print("  ASTRA Test Scenarios")
    print("=" * 64)

    for label, fn in scenarios:
        try:
            result = run_scenario(fn, config, label)
            results.append(result)
            status = result["experiment_status"]
            completed = len(result["completed_steps"])
            total = len(config.steps)
            anomalies = len(result["anomalies"])
            expected = result.get("expected_outcome", "?")
            print(f"\n  Scenario {label}:")
            print(f"    Status:     {status}")
            print(f"    Completed:  {completed}/{total}")
            print(f"    Anomalies:  {anomalies}")
            print(f"    Expected:   {expected}")
        except Exception as e:
            print(f"\n  Scenario {label}: FAILED — {e}")
            results.append({"scenario": label, "error": str(e)})

    out_path = os.path.join(args.output, "scenario_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'=' * 64}")
    print(f"  Results saved: {out_path}")
    print(f"{'=' * 64}\n")


if __name__ == "__main__":
    main()
