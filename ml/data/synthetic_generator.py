"""
ASTRA — Synthetic Experiment Video Generator

Generates synthetic experiment videos with rendered objects and hands,
along with EXACT ground truth annotations for every frame.

This is SYNTHETIC DATA — not real camera footage.
All metrics from this data must be labeled SYNTHETIC.

Generates:
  - MP4 videos with colored object shapes
  - COCO JSON detection annotations (exact ground truth)
  - JSONL action annotations
  - JSONL step annotations
  - JSONL object state annotations
  - JSONL HOI annotations

Usage:
    python -m ml.data.synthetic_generator --output data/synthetic/ --n_sessions 10
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Object visual definitions
OBJECT_DEFS = {
    "container": {"color": (80, 60, 200), "size": (120, 80), "label_color": (255, 255, 255)},
    "sample":    {"color": (50, 200, 50), "size": (40, 30), "label_color": (0, 0, 0)},
    "scissors":  {"color": (200, 200, 60), "size": (80, 25), "label_color": (0, 0, 0)},
    "tweezers":  {"color": (60, 180, 200), "size": (70, 20), "label_color": (0, 0, 0)},
    "tray":      {"color": (160, 100, 60), "size": (140, 90), "label_color": (255, 255, 255)},
}

CLASS_MAP = {"container": 0, "sample": 1, "scissors": 2, "tweezers": 3, "tray": 4}

# Step definitions: (step_id, name, duration_frames, actions, active_objects)
NORMAL_STEPS = [
    ("step_01", "Prepare Workspace", (30, 50), ["visual_inspection"], ["container", "scissors", "tweezers", "tray"]),
    ("step_02", "Open Container", (25, 40), ["reach", "grasp", "open"], ["container"]),
    ("step_03", "Remove Sample", (30, 50), ["reach", "grasp", "remove"], ["container", "sample", "tweezers"]),
    ("step_04", "Place on Tray", (25, 40), ["move", "place", "release"], ["sample", "tray", "tweezers"]),
    ("step_05", "Cut Sample", (35, 55), ["reach", "grasp", "use"], ["sample", "scissors"]),
    ("step_06", "Return Sample", (30, 50), ["reach", "grasp", "move", "place"], ["sample", "container", "tweezers"]),
    ("step_07", "Close Container", (20, 35), ["reach", "grasp", "close"], ["container"]),
]

# Scenario types
SCENARIOS = {
    "normal": "Normal complete execution",
    "skipped_step": "Step 03 (Remove Sample) skipped",
    "out_of_order": "Step 05 before Step 04",
    "incomplete": "Only steps 01-04 completed",
    "repeated_action": "Step 02 attempted twice",
    "occluded": "Random object occlusion in some frames",
}


def _random_jitter(base_pos, jitter=15):
    """Add random jitter to a position."""
    return (base_pos[0] + random.randint(-jitter, jitter),
            base_pos[1] + random.randint(-jitter, jitter))


def _draw_hand(frame, pos, hand="right", confidence=0.9):
    """Draw a simplified hand shape."""
    x, y = pos
    color = (220, 180, 140)  # Skin tone
    cv2.circle(frame, (x, y), 18, color, -1)
    # Fingers
    for angle_off in [-20, -10, 0, 10, 20]:
        angle = math.radians(-90 + angle_off)
        fx = int(x + 22 * math.cos(angle))
        fy = int(y + 22 * math.sin(angle))
        cv2.line(frame, (x, y), (fx, fy), color, 4)
    return (x - 20, y - 25, x + 20, y + 20)


def _draw_object(frame, obj_class, pos, state=None, occluded=False):
    """Draw an object on the frame. Returns bbox [x1,y1,x2,y2]."""
    x, y = pos
    defn = OBJECT_DEFS[obj_class]
    w, h = defn["size"]
    color = defn["color"]

    if occluded:
        # 50% opacity
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    else:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)

    # State indicator
    if state:
        cv2.putText(frame, state[:8], (x + 2, y + h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, defn["label_color"], 1)

    # Label
    cv2.putText(frame, obj_class[:6], (x + 2, y + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, defn["label_color"], 1)

    return [x, y, x + w, y + h]


def generate_session(
    session_id: str,
    scenario: str,
    output_dir: str,
    fps: int = 15,
    frame_size: tuple = (640, 480),
    camera_angle: int = 0,
    seed: int = None,
) -> Dict[str, Any]:
    """
    Generate one synthetic experiment session.

    Returns session manifest with all annotations.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    w, h = frame_size
    session_dir = os.path.join(output_dir, session_id)
    frames_dir = os.path.join(session_dir, "frames")
    ann_dir = os.path.join(session_dir, "annotations")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)

    # Base object positions (with slight randomization per session)
    base_positions = {
        "container": (80 + random.randint(-20, 20), 180 + random.randint(-20, 20)),
        "sample":    (110 + random.randint(-10, 10), 210 + random.randint(-10, 10)),
        "scissors":  (400 + random.randint(-30, 30), 120 + random.randint(-20, 20)),
        "tweezers":  (380 + random.randint(-30, 30), 300 + random.randint(-20, 20)),
        "tray":      (280 + random.randint(-20, 20), 300 + random.randint(-20, 20)),
    }

    # Determine step order based on scenario
    if scenario == "normal":
        steps = list(NORMAL_STEPS)
    elif scenario == "skipped_step":
        steps = [s for s in NORMAL_STEPS if s[0] != "step_03"]
    elif scenario == "out_of_order":
        steps = list(NORMAL_STEPS)
        if len(steps) >= 5:
            steps[3], steps[4] = steps[4], steps[3]  # Swap step_04 and step_05
    elif scenario == "incomplete":
        steps = list(NORMAL_STEPS)[:4]
    elif scenario == "repeated_action":
        steps = list(NORMAL_STEPS)
        steps.insert(2, NORMAL_STEPS[1])  # Repeat step_02
    elif scenario == "occluded":
        steps = list(NORMAL_STEPS)
    else:
        steps = list(NORMAL_STEPS)

    # Generate frames
    all_detections = []
    action_anns = []
    step_anns = []
    state_anns = []
    hoi_anns = []
    anomaly_anns = []
    frame_list = []

    # Object states
    obj_states = {
        "container": "closed", "sample": "inside_container",
        "scissors": "docked", "tweezers": "docked", "tray": "empty"
    }

    frame_idx = 0
    hand_pos = (w // 2, h - 60)
    ann_id = 0

    # Video writer
    video_path = os.path.join(session_dir, f"{session_id}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

    for step_def in steps:
        step_id, step_name, dur_range, actions, active_objs = step_def
        n_frames = random.randint(dur_range[0], dur_range[1])
        step_start = frame_idx

        # Determine target hand position (near first active object)
        target_obj = active_objs[0]
        target_pos = base_positions[target_obj]
        hand_target = (target_pos[0] + 30, target_pos[1] - 20)

        # Action segments within this step
        action_dur = max(3, n_frames // len(actions))
        for a_idx, action_id in enumerate(actions):
            a_start = step_start + a_idx * action_dur
            a_end = min(step_start + (a_idx + 1) * action_dur, step_start + n_frames)
            action_anns.append({
                "action_id": action_id, "start_frame": a_start, "end_frame": a_end,
            })

        for i in range(n_frames):
            # Create frame
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            # Background with slight variation
            bg_shade = 35 + random.randint(-5, 5)
            frame[:] = (bg_shade, bg_shade + 5, bg_shade + 2)

            # Draw workspace surface
            cv2.rectangle(frame, (20, 100), (w - 20, h - 40), (50, 45, 40), -1)
            cv2.rectangle(frame, (20, 100), (w - 20, h - 40), (80, 75, 70), 2)

            # Step label
            cv2.putText(frame, f"{step_id}: {step_name}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
            cv2.putText(frame, f"Frame {frame_idx} | {session_id}", (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)

            # Update object states based on step progress
            progress = i / max(1, n_frames - 1)
            if step_id == "step_02" and progress > 0.6:
                obj_states["container"] = "open"
            elif step_id == "step_03" and progress > 0.5:
                obj_states["sample"] = "held_by_tweezers"
            elif step_id == "step_04" and progress > 0.6:
                obj_states["sample"] = "on_tray"
                obj_states["tray"] = "occupied"
            elif step_id == "step_05" and progress > 0.7:
                obj_states["sample"] = "processed"
            elif step_id == "step_06" and progress > 0.6:
                obj_states["sample"] = "inside_container"
                obj_states["tray"] = "empty"
            elif step_id == "step_07" and progress > 0.6:
                obj_states["container"] = "closed"

            # Draw objects
            frame_dets = []
            for obj_class, pos in base_positions.items():
                is_occluded = (scenario == "occluded" and
                               random.random() < 0.15 and
                               obj_class in active_objs)
                # Move sample based on state
                draw_pos = pos
                if obj_class == "sample":
                    if obj_states["sample"] == "held_by_tweezers":
                        draw_pos = (hand_pos[0] - 10, hand_pos[1] + 5)
                    elif obj_states["sample"] == "on_tray":
                        tray_pos = base_positions["tray"]
                        draw_pos = (tray_pos[0] + 40, tray_pos[1] + 25)

                jittered = _random_jitter(draw_pos, jitter=3)
                bbox = _draw_object(frame, obj_class, jittered,
                                     state=obj_states.get(obj_class), occluded=is_occluded)

                conf = 0.95 - (0.5 if is_occluded else 0.0) + random.uniform(-0.03, 0.03)
                frame_dets.append({
                    "id": ann_id, "image_id": frame_idx,
                    "category_id": CLASS_MAP[obj_class],
                    "bbox": [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
                    "bbox_xyxy": bbox,
                    "area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                    "iscrowd": 0, "class_name": obj_class,
                    "occluded": is_occluded,
                    "confidence": round(max(0.1, min(1.0, conf)), 3),
                })
                ann_id += 1

            all_detections.extend(frame_dets)

            # Move hand toward target
            alpha = min(1.0, (i + 1) / (n_frames * 0.4))
            hx = int(hand_pos[0] + alpha * (hand_target[0] - hand_pos[0]))
            hy = int(hand_pos[1] + alpha * (hand_target[1] - hand_pos[1]))
            hand_pos = _random_jitter((hx, hy), jitter=5)
            _draw_hand(frame, hand_pos)

            writer.write(frame)

            fname = f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(os.path.join(frames_dir, fname), frame)
            frame_list.append({
                "frame_id": frame_idx, "filename": fname,
                "timestamp": round(frame_idx / fps, 4),
                "source_frame": frame_idx,
            })
            frame_idx += 1

        # Step annotation
        step_anns.append({
            "step_id": step_id, "step_name": step_name,
            "start_frame": step_start, "end_frame": frame_idx - 1,
            "status": "completed",
        })

        # State transitions
        for obj, state in obj_states.items():
            state_anns.append({
                "object_id": obj, "state": state,
                "frame": frame_idx - 1, "step_id": step_id,
            })

        # HOI
        if len(active_objs) > 0:
            hoi_anns.append({
                "hand": "right", "object": active_objs[0],
                "interaction": actions[-1] if actions else "idle",
                "start_frame": step_start, "end_frame": frame_idx - 1,
                "step_id": step_id,
            })

    writer.release()

    # Anomaly annotations for non-normal scenarios
    if scenario == "skipped_step":
        anomaly_anns.append({
            "anomaly_type": "skipped_step", "expected_step": "step_03",
            "observed_step": "step_04", "start_frame": 0, "end_frame": frame_idx - 1,
        })
    elif scenario == "out_of_order":
        anomaly_anns.append({
            "anomaly_type": "out_of_order", "expected_step": "step_04",
            "observed_step": "step_05", "start_frame": 0, "end_frame": frame_idx - 1,
        })
    elif scenario == "incomplete":
        anomaly_anns.append({
            "anomaly_type": "incomplete", "expected_step": "step_05",
            "observed_step": "none", "start_frame": 0, "end_frame": frame_idx - 1,
        })

    # Save manifest
    manifest = {
        "session_id": session_id, "scenario": scenario,
        "data_type": "SYNTHETIC",
        "video_path": os.path.abspath(video_path),
        "frames": frame_list,
        "extracted_frames": frame_idx,
        "duration_seconds": round(frame_idx / fps, 2),
        "source_fps": fps, "extract_fps": fps,
        "generation_date": datetime.now().isoformat(),
    }
    with open(os.path.join(session_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # Save COCO detections
    coco = {
        "info": {"description": f"ASTRA Synthetic - {session_id}", "data_type": "SYNTHETIC"},
        "images": [{"id": fl["frame_id"], "file_name": fl["filename"],
                     "width": w, "height": h}
                    for fl in frame_list],
        "annotations": all_detections,
        "categories": [{"id": v, "name": k} for k, v in CLASS_MAP.items()],
    }
    with open(os.path.join(ann_dir, "detections.json"), "w") as f:
        json.dump(coco, f, indent=2)

    # Save JSONL annotations
    for name, data in [("actions", action_anns), ("steps", step_anns),
                        ("object_states", state_anns), ("hoi", hoi_anns),
                        ("anomalies", anomaly_anns)]:
        with open(os.path.join(ann_dir, f"{name}.jsonl"), "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

    # Step sequence
    step_seq = {"experiment_id": "sample_experiment_001", "session_id": session_id,
                "sequence": [s["step_id"] for s in step_anns]}
    with open(os.path.join(ann_dir, "step_sequence.json"), "w") as f:
        json.dump(step_seq, f, indent=2)

    # Session manifest
    session_meta = {
        "experiment_id": "sample_experiment_001", "session_id": session_id,
        "performer_id": f"performer_{random.randint(1, 3):03d}",
        "scenario": scenario, "data_type": "SYNTHETIC",
        "video_path": video_path, "duration": manifest["duration_seconds"],
        "extracted_frames": frame_idx, "extract_fps": fps,
        "annotation_status": "auto_generated",
        "created": datetime.now().isoformat(),
    }
    with open(os.path.join(session_dir, "session_manifest.json"), "w") as f:
        json.dump(session_meta, f, indent=2)

    logger.info(f"Generated {session_id}: {scenario} | {frame_idx} frames | {manifest['duration_seconds']}s")
    return manifest


def generate_dataset(output_dir: str, n_sessions: int = 15, seed: int = 42):
    """
    Generate a complete synthetic dataset with multiple scenarios.
    """
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # Session plan
    plan = []
    # At least 5 normal
    for i in range(5):
        plan.append(("normal", i))
    # At least 2 skipped
    for i in range(2):
        plan.append(("skipped_step", i))
    # At least 2 out-of-order
    for i in range(2):
        plan.append(("out_of_order", i))
    # 1 incomplete
    plan.append(("incomplete", 0))
    # 1 repeated
    plan.append(("repeated_action", 0))
    # 2 occluded
    for i in range(2):
        plan.append(("occluded", i))

    # Pad to n_sessions
    while len(plan) < n_sessions:
        plan.append(("normal", len(plan)))

    plan = plan[:n_sessions]
    random.shuffle(plan)

    sessions = []
    for idx, (scenario, var) in enumerate(plan):
        session_id = f"session_{idx + 1:03d}"
        manifest = generate_session(
            session_id=session_id, scenario=scenario,
            output_dir=output_dir, seed=seed + idx * 100,
            camera_angle=random.choice([0, 15, -15]),
        )
        manifest["scenario"] = scenario
        sessions.append(manifest)

    # Dataset manifest
    dataset_meta = {
        "dataset_name": "astra_synthetic_v1",
        "data_type": "SYNTHETIC",
        "description": "Synthetic experiment data for ASTRA development",
        "n_sessions": len(sessions),
        "scenarios": {s: sum(1 for p in plan if p[0] == s) for s in set(p[0] for p in plan)},
        "total_frames": sum(s["extracted_frames"] for s in sessions),
        "generation_date": datetime.now().isoformat(),
        "seed": seed,
        "sessions": [{
            "session_id": s["session_id"], "scenario": s["scenario"],
            "frames": s["extracted_frames"], "duration": s["duration_seconds"],
        } for s in sessions],
    }
    with open(os.path.join(output_dir, "dataset_manifest.json"), "w") as f:
        json.dump(dataset_meta, f, indent=2)

    logger.info(f"\nDataset generated: {len(sessions)} sessions, "
                f"{dataset_meta['total_frames']} total frames")
    logger.info(f"Scenarios: {dataset_meta['scenarios']}")

    return dataset_meta


def main():
    parser = argparse.ArgumentParser(description="ASTRA Synthetic Data Generator")
    parser.add_argument("--output", default="data/synthetic", help="Output directory")
    parser.add_argument("--n_sessions", type=int, default=15, help="Number of sessions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    meta = generate_dataset(args.output, args.n_sessions, args.seed)
    print(f"\nDataset: {args.output}")
    print(f"Sessions: {meta['n_sessions']}")
    print(f"Frames: {meta['total_frames']}")
    print(f"Scenarios: {meta['scenarios']}")


if __name__ == "__main__":
    main()
