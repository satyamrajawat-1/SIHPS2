import json
import logging
import os
import random
import math
from typing import Dict, List, Tuple
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

COCO_JOINTS = 17
ACTION_CLASSES = {"idle": 0, "reach": 1, "pick": 2, "move": 3, "place": 4, "release": 5}

def random_body_proportions(rng) -> Dict[str, float]:
    return {
        "torso": rng.uniform(0.85, 1.15),
        "shoulder_width": rng.uniform(0.85, 1.15),
        "upper_arm": rng.uniform(0.85, 1.15),
        "forearm": rng.uniform(0.85, 1.15),
        "thigh": rng.uniform(0.85, 1.15),
        "shin": rng.uniform(0.85, 1.15),
    }

def generate_base_skeleton(props: Dict[str, float], center_x: float = 0.5) -> np.ndarray:
    cx = center_x
    s = props
    # Build a skeleton with randomized limb lengths
    skeleton = np.zeros((17, 2), dtype=np.float32)
    skeleton[0] = [cx, 0.15] # nose
    skeleton[1] = [cx - 0.02, 0.13]
    skeleton[2] = [cx + 0.02, 0.13]
    skeleton[3] = [cx - 0.04, 0.15]
    skeleton[4] = [cx + 0.04, 0.15]
    
    sw = 0.10 * s["shoulder_width"]
    skeleton[5] = [cx - sw, 0.25]
    skeleton[6] = [cx + sw, 0.25]
    
    ua = 0.10 * s["upper_arm"]
    skeleton[7] = [cx - sw - ua*0.5, 0.25 + ua] # left_elbow
    skeleton[8] = [cx + sw + ua*0.5, 0.25 + ua] # right_elbow
    
    fa = 0.10 * s["forearm"]
    skeleton[9] = [skeleton[7][0] - fa*0.3, skeleton[7][1] + fa] # left_wrist
    skeleton[10] = [skeleton[8][0] + fa*0.3, skeleton[8][1] + fa] # right_wrist
    
    th = 0.15 * s["thigh"]
    sh = 0.15 * s["shin"]
    hw = 0.08
    torso_y = 0.25 + 0.25 * s["torso"]
    skeleton[11] = [cx - hw, torso_y]
    skeleton[12] = [cx + hw, torso_y]
    skeleton[13] = [cx - hw, torso_y + th]
    skeleton[14] = [cx + hw, torso_y + th]
    skeleton[15] = [cx - hw, torso_y + th + sh]
    skeleton[16] = [cx + hw, torso_y + th + sh]
    return skeleton

def bezier_curve(p0, p1, p2, t):
    return (1 - t)**2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2

def generate_continuous_episode(rng, props: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    base = generate_base_skeleton(props)
    
    # Randomize object, target, and orientations
    obj_pos = np.array([rng.uniform(0.6, 0.8), rng.uniform(0.4, 0.6)])
    tgt_pos = np.array([rng.uniform(0.2, 0.4), rng.uniform(0.4, 0.6)])
    
    hand = rng.choice(["right", "left"])
    wrist_idx = 10 if hand == "right" else 9
    elbow_idx = 8 if hand == "right" else 7
    
    # Action durations (frames)
    durations = {
        "idle_1": rng.randint(10, 50),
        "reach": rng.randint(10, 40),
        "pick": rng.randint(5, 25),
        "move": rng.randint(15, 50),
        "place": rng.randint(10, 35),
        "release": rng.randint(5, 20),
        "idle_2": rng.randint(10, 50),
    }
    
    total_frames = sum(durations.values())
    frames = np.zeros((total_frames, 17, 3), dtype=np.float32)
    labels = np.zeros(total_frames, dtype=np.int64)
    
    curr_frame = 0
    rest_pos = base[wrist_idx].copy()
    
    def fill_frames(action_name, num_f, pos_func):
        nonlocal curr_frame
        cid = ACTION_CLASSES.get(action_name.split("_")[0], 0)
        for t in range(num_f):
            progress = t / max(num_f - 1, 1)
            # Add speed variation (non-linear progress)
            speed_curve = rng.choice(["linear", "fast_start", "slow_start"])
            if speed_curve == "fast_start":
                p = 1 - (1 - progress)**2
            elif speed_curve == "slow_start":
                p = progress**2
            else:
                p = progress
                
            sk = base.copy()
            # Idle sway
            sk += rng.randn(17, 2) * 0.003
            
            # Update arm
            new_wrist, new_elbow = pos_func(p)
            if new_wrist is not None:
                sk[wrist_idx] = new_wrist
                sk[elbow_idx] = new_elbow
                
            # Random noise
            noise_level = rng.choice([0.001, 0.005, 0.015, 0.03])
            sk += rng.randn(17, 2) * noise_level
            
            # Confidence dropouts
            conf = np.ones((17, 1), dtype=np.float32)
            if rng.rand() < 0.05: # dropout
                drop_idx = rng.randint(0, 17)
                conf[drop_idx] = rng.uniform(0.0, 0.2)
                
            frames[curr_frame, :, :2] = sk
            frames[curr_frame, :, 2:] = conf
            labels[curr_frame] = cid
            curr_frame += 1

    # IDLE 1
    def idle_func(p):
        return None, None
    fill_frames("idle_1", durations["idle_1"], idle_func)
    
    # REACH (Curved/Straight)
    reach_cp = rest_pos + (obj_pos - rest_pos) * 0.5 + rng.uniform(-0.1, 0.1, 2)
    def reach_func(p):
        w = bezier_curve(rest_pos, reach_cp, obj_pos, p)
        e = base[elbow_idx] + (w - base[wrist_idx]) * 0.5
        return w, e
    fill_frames("reach", durations["reach"], reach_func)
    
    # PICK (Hesitation/Lifting)
    pick_end = obj_pos + np.array([0, -0.05 - rng.uniform(0, 0.05)])
    def pick_func(p):
        w = obj_pos + (pick_end - obj_pos) * p
        e = base[elbow_idx] + (w - base[wrist_idx]) * 0.5
        return w, e
    fill_frames("pick", durations["pick"], pick_func)
    
    # MOVE (Curved)
    move_cp = pick_end + (tgt_pos - pick_end) * 0.5 + np.array([0, -rng.uniform(0.1, 0.3)])
    def move_func(p):
        w = bezier_curve(pick_end, move_cp, tgt_pos, p)
        e = base[elbow_idx] + (w - base[wrist_idx]) * 0.5
        return w, e
    fill_frames("move", durations["move"], move_func)
    
    # PLACE (Lowering)
    place_end = tgt_pos + np.array([0, 0.05 + rng.uniform(0, 0.05)])
    def place_func(p):
        w = tgt_pos + (place_end - tgt_pos) * p
        e = base[elbow_idx] + (w - base[wrist_idx]) * 0.5
        return w, e
    fill_frames("place", durations["place"], place_func)
    
    # RELEASE (Retract)
    def release_func(p):
        w = place_end + (rest_pos - place_end) * p
        e = base[elbow_idx] + (w - base[wrist_idx]) * 0.5
        return w, e
    fill_frames("release", durations["release"], release_func)
    
    # IDLE 2
    fill_frames("idle_2", durations["idle_2"], idle_func)
    
    # ----------------------------------------------------
    # V3 FIX: Coordinate Normalization
    # 1. Translate to reference joint (0=nose)
    # 2. Normalize scale (distance between shoulders 5 and 6)
    # ----------------------------------------------------
    for t in range(total_frames):
        kps = frames[t]
        ref = kps[0, :2].copy()
        
        # Translate
        kps[:, 0] -= ref[0]
        kps[:, 1] -= ref[1]
        
        # Scale
        scale = np.linalg.norm(kps[5, :2] - kps[6, :2])
        if scale > 1e-6:
            kps[:, :2] /= scale
            
        frames[t] = kps

    # Apply global augmentations
    # Rotate 0-180 (Rotation now happens around (0,0) which is the root joint)
    if rng.rand() > 0.3:
        angle = rng.uniform(0, np.pi)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        for t in range(total_frames):
            xy = frames[t, :, :2].copy()
            nx = xy[:, 0] * cos_a - xy[:, 1] * sin_a
            ny = xy[:, 0] * sin_a + xy[:, 1] * cos_a
            frames[t, :, 0] = nx
            frames[t, :, 1] = ny
            
    # We remove random translation and scaling because the normalization completely
    # cancels them out anyway. The model should be translation/scale invariant.
    
    return frames, labels

def extract_windows(frames, labels, rng, n_windows=20):
    windows = []
    window_labels = []
    T = len(frames)
    sizes = [16, 24, 30, 40, 48, 60]
    for _ in range(n_windows):
        size = rng.choice(sizes)
        if T <= size:
            start = 0
            size = T
        else:
            start = rng.randint(0, T - size)
            
        win = frames[start:start+size].copy()
        
        padded = np.zeros((60, 17, 3), dtype=np.float32)
        padded[:size] = win
        for i in range(size, 60):
            padded[i] = win[-1]
            
        recent = labels[start:start+size][-15:]
        win_label = np.bincount(recent).argmax()
        
        windows.append(padded)
        window_labels.append(win_label)
        
    return windows, window_labels

def main():
    out_dir = Path("data/action_v3")
    
    rng = np.random.RandomState(42)
    
    def generate_split(split_name, n_episodes, out_path):
        out_path.mkdir(parents=True, exist_ok=True)
        all_windows = []
        all_labels = []
        
        for ep in range(n_episodes):
            props = random_body_proportions(rng)
            frames, labels = generate_continuous_episode(rng, props)
            wins, w_labels = extract_windows(frames, labels, rng, n_windows=30)
            all_windows.extend(wins)
            all_labels.extend(w_labels)
            
        np.savez(out_path / "samples.npz", sequences=np.array(all_windows), labels=np.array(all_labels))
        with open(out_path / "labels.json", "w") as f:
            json.dump({"class_map": ACTION_CLASSES, "n_samples": len(all_labels)}, f)
        return all_windows, all_labels

    print("Generating train...")
    train_x, train_y = generate_split("train", 300, out_dir / "train")
    print("Generating val...")
    generate_split("val", 50, out_dir / "val")
    print("Generating test...")
    generate_split("test", 50, out_dir / "test")
    print("Generating synthetic_hard_test...")
    generate_split("synthetic_hard_test", 100, out_dir / "synthetic_hard_test")

    # Generate basic stats for training set
    stats = {
        "class_distribution": {k: int(np.sum(np.array(train_y) == v)) for k, v in ACTION_CLASSES.items()},
        "total_windows": len(train_y)
    }
    with open("data/action_v3/synthetic_v3_stats.json", "w") as f:
        json.dump(stats, f)
    print("Stats written.")

if __name__ == "__main__":
    main()
