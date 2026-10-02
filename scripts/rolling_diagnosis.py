"""
ASTRA - ST-GCN V3 Rolling Window Diagnosis

Updated to strictly use V3 preprocessing, V3 classes, and the canonical model.
"""

import numpy as np
import torch
import os
import sys
from collections import deque

sys.path.append('.')
from ml.models.action.stgcn.model import STGCNPP
from ml.models.pose.model import normalize_skeleton
from ml.scripts.generate_action_data_v3 import generate_continuous_episode, random_body_proportions, ACTION_CLASSES

class ActionSmoother:
    def __init__(self, window_size=15, min_count=8):
        self.buffer = deque(maxlen=window_size)
        self.min_count = min_count
        self.current = 'idle'
        
    def update(self, new_action):
        self.buffer.append(new_action)
        counts = {}
        for a in self.buffer:
            counts[a] = counts.get(a, 0) + 1
        
        best_action, count = max(counts.items(), key=lambda x: x[1])
        if count >= self.min_count:
            self.current = best_action
            
        return self.current

def main():
    CHECKPOINT_PATH = "checkpoints/action/STGCNPP-ASTRA-v3/best.pt"
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"ERROR: No checkpoint found at {CHECKPOINT_PATH}")
        return
        
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    classes = ckpt.get('classes', list(ACTION_CLASSES.keys()))
    print(f"Loaded epoch {ckpt.get('epoch', '?')}, val_acc={ckpt.get('val_acc', '?')}")
    print(f"Classes: {classes}")
    
    model = STGCNPP(action_classes=classes, num_classes=len(classes))
    model.load_model(CHECKPOINT_PATH, device=DEVICE)
    model.model.eval()
    
    rng = np.random.RandomState(777)
    props = random_body_proportions(rng)
    ep_frames, ep_labels = generate_continuous_episode(rng, props)
    T_total = len(ep_frames)
    
    print(f"\n{'='*80}")
    print(f"TEST EPISODE: {T_total} frames")
    print(f"{'='*80}")
    
    smoother = ActionSmoother(window_size=15, min_count=8)
    transitions = []
    last_smoothed = 'idle'
    
    buffer = deque(maxlen=60)
    
    print(f"\n{'---'*25}")
    print(f"ROLLING INFERENCE (60-FRAME BUFFER)")
    print(f"{'---'*25}")
    
    for t in range(T_total):
        # 1. Normalize
        norm_kps = normalize_skeleton(ep_frames[t].copy())
        
        # 2. Append to buffer
        buffer.append(norm_kps)
        
        # 3. Predict if enough frames
        if len(buffer) >= 16:
            seq = np.array(list(buffer))
            
            # Padding
            padded = np.zeros((60, 17, 3), dtype=np.float32)
            padded[:len(seq)] = seq
            for i in range(len(seq), 60):
                padded[i] = seq[-1]
                
            res = model.predict(padded)
            probs = res.all_scores
            # Find max
            raw_pred = max(probs, key=probs.get)
            conf = probs[raw_pred]
        else:
            raw_pred = 'idle'
            conf = 1.0
            
        smoothed = smoother.update(raw_pred)
        
        true_label = classes[ep_labels[t]]
        
        if smoothed != last_smoothed:
            transitions.append((t, last_smoothed, smoothed, true_label))
            last_smoothed = smoothed
            
        if t % 20 == 0:
            print(f"  F{t:3d} | True: {true_label:7s} | Raw: {raw_pred:7s} ({conf:.2f}) | Smooth: {smoothed:7s}")
            
    detected_seq = ['idle'] + [to for _, _, to, _ in transitions]
    expected = ['idle', 'reach', 'pick', 'move', 'place', 'release', 'idle']
    
    print(f"\n  Transitions:")
    for t, frm, to, true in transitions:
        match = 'Y' if to == true else 'N'
        print(f"    Frame {t:3d}: {frm} -> {to}  (true: {true}) {match}")
        
    print(f"\n  Expected:  {expected}")
    print(f"  Detected:  {detected_seq}")
    
    # Filter detected sequence to remove consecutive duplicates (just in case)
    final_detected = []
    for d in detected_seq:
        if not final_detected or final_detected[-1] != d:
            final_detected.append(d)
            
    correct = 0
    # simple match count
    idx = 0
    for e in expected:
        if idx < len(final_detected) and final_detected[idx] == e:
            correct += 1
            idx += 1
            
    score = correct / len(expected)
    print(f"  Match: {correct}/{len(expected)} = {score:.0%}")
    
    if score >= 6/7:
        print("\nDECISION: ST-GCN READY (acceptable sequence detection)")
    else:
        print("\nDECISION: ROLLING INFERENCE FAILS")

if __name__ == '__main__':
    main()
