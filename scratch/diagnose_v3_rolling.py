"""
ASTRA - ST-GCN Rolling Window Diagnosis

Tasks:
1. Trace exact window composition at key frames
2. Check label semantics (training vs inference)
3. Test multiple window lengths on same episode
4. Check warm-up / padding behavior
5. Print full probability vectors
6. Compare training vs inference preprocessing
7. Determine if any window config produces correct sequence

Run on Colab after training, or locally with checkpoint.
"""

import numpy as np
import torch
import torch.nn as nn
from collections import deque, Counter
import os
import json

# ============================================================
# CONFIG
# ============================================================
ACTION_CLASSES = {"idle": 0, "reach": 1, "pick": 2, "move": 3, "place": 4, "release": 5}
CLASS_NAMES = list(ACTION_CLASSES.keys())
NUM_CLASSES = len(ACTION_CLASSES)

# Adjust this path for local vs Colab
CHECKPOINT_PATH = None
for p in [
    "checkpoints/action/STGCNPP-ASTRA-v2/best.pt",
    "checkpoints/action/STGCNPP-ASTRA-v1/best.pt",
]:
    if os.path.exists(p):
        CHECKPOINT_PATH = p
        break

if CHECKPOINT_PATH is None:
    print("ERROR: No checkpoint found. Set CHECKPOINT_PATH manually.")
    exit(1)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()
print(f"Device: {DEVICE}")
print(f"Checkpoint: {CHECKPOINT_PATH}")

# ============================================================
# MODEL DEFINITION (identical to training)
# ============================================================
def get_spatial_graph(num_joints=17):
    edges = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (0, 5), (0, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16),
    ]
    self_loops = [(i, i) for i in range(num_joints)]
    all_edges = edges + self_loops
    A = np.zeros((num_joints, num_joints), dtype=np.float32)
    for i, j in all_edges:
        A[i, j] = 1; A[j, i] = 1
    D = np.sum(A, axis=1)
    D_inv = np.where(D > 0, 1.0 / np.sqrt(D), 0)
    return D_inv[:, None] * A * D_inv[None, :]

class SpatialGraphConv(nn.Module):
    def __init__(self, in_ch, out_ch, A):
        super().__init__()
        self.A = nn.Parameter(torch.from_numpy(A).float(), requires_grad=False)
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        self.bn = nn.BatchNorm2d(out_ch)
    def forward(self, x):
        x_a = torch.einsum('nctv,vw->nctw', x, self.A)
        return self.bn(self.conv(x_a))

class TemporalConv(nn.Module):
    def __init__(self, ch, ks=9):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, (ks, 1), padding=((ks-1)//2, 0))
        self.bn = nn.BatchNorm2d(ch)
    def forward(self, x):
        return self.bn(self.conv(x))

class STGCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, A, stride=1, dropout=0.1):
        super().__init__()
        self.gcn = SpatialGraphConv(in_ch, out_ch, A)
        self.tcn = TemporalConv(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        self.residual = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1), nn.BatchNorm2d(out_ch)
        ) if in_ch != out_ch else nn.Identity()
        self.pool = nn.AvgPool2d((stride, 1)) if stride > 1 else nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        x = self.relu(self.gcn(x))
        x = self.drop(self.tcn(x))
        x = self.pool(x); res = self.pool(res)
        return self.relu(x + res)

class STGCNPlusPlus(nn.Module):
    def __init__(self, in_channels=3, num_classes=6, num_joints=17, A=None, hidden_dim=64, num_layers=10, dropout=0.15):
        super().__init__()
        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)
        layers = []
        ch = in_channels
        for i in range(num_layers):
            out_ch = hidden_dim * (2 ** min(i // 3, 2))
            stride = 2 if i in [3, 6] else 1
            layers.append(STGCNBlock(ch, out_ch, A, stride, dropout))
            ch = out_ch
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(ch, num_classes)

    def forward(self, x):
        N, C, T, V = x.shape
        x_bn = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x_bn = self.data_bn(x_bn)
        x = x_bn.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()
        x = self.layers(x)
        x = x.mean(dim=-1).mean(dim=-1)
        return self.fc(x)

# ============================================================
# LOAD MODEL
# ============================================================
A_MATRIX = get_spatial_graph(17)
model = STGCNPlusPlus(in_channels=3, num_classes=NUM_CLASSES, num_joints=17,
                      A=A_MATRIX, hidden_dim=64, num_layers=10, dropout=0.15).to(DEVICE)

ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(ckpt['model'])
model.eval()
print(f"Loaded epoch {ckpt.get('epoch', '?')}, val_acc={ckpt.get('val_acc', '?')}")
print(f"Classes: {ckpt.get('classes', CLASS_NAMES)}")

# ============================================================
# GENERATE TEST EPISODE
# ============================================================
def generate_test_episode(rng):
    """Generate one continuous IDLE->REACH->PICK->MOVE->PLACE->RELEASE->IDLE episode."""
    cx = 0.5
    base = np.zeros((17, 2), dtype=np.float32)
    base[0] = [cx, 0.15]; base[1] = [cx-0.02, 0.13]; base[2] = [cx+0.02, 0.13]
    base[3] = [cx-0.04, 0.15]; base[4] = [cx+0.04, 0.15]
    base[5] = [cx-0.10, 0.25]; base[6] = [cx+0.10, 0.25]
    base[7] = [cx-0.175, 0.35]; base[8] = [cx+0.175, 0.35]
    base[9] = [cx-0.21, 0.45]; base[10] = [cx+0.21, 0.45]
    base[11] = [cx-0.08, 0.50]; base[12] = [cx+0.08, 0.50]
    base[13] = [cx-0.08, 0.65]; base[14] = [cx+0.08, 0.65]
    base[15] = [cx-0.08, 0.80]; base[16] = [cx+0.08, 0.80]

    obj_pos = np.array([0.72, 0.48]); tgt_pos = np.array([0.28, 0.47])
    wrist, elbow = 10, 8
    rest = base[wrist].copy()

    actions = [
        ('idle',    rng.randint(15, 40)),
        ('reach',   rng.randint(15, 35)),
        ('pick',    rng.randint(10, 20)),
        ('move',    rng.randint(20, 45)),
        ('place',   rng.randint(10, 30)),
        ('release', rng.randint(8, 18)),
        ('idle',    rng.randint(15, 40)),
    ]
    total = sum(d for _, d in actions)
    frames = np.zeros((total, 17, 3), dtype=np.float32)
    labels = np.zeros(total, dtype=np.int64)

    t = 0
    for action, dur in actions:
        cid = ACTION_CLASSES[action]
        for i in range(dur):
            p = i / max(dur-1, 1)
            sk = base.copy() + rng.randn(17, 2) * 0.003
            if action == 'reach':
                sk[wrist] = rest + (obj_pos - rest) * p
            elif action == 'pick':
                sk[wrist] = obj_pos + np.array([0, -0.06*p])
            elif action == 'move':
                sk[wrist] = obj_pos + (tgt_pos - obj_pos) * p + np.array([0, -0.05])
            elif action == 'place':
                sk[wrist] = tgt_pos + np.array([0, -0.05*(1-p)])
            elif action == 'release':
                sk[wrist] = tgt_pos + (rest - tgt_pos) * p
            sk[elbow] = base[elbow] + (sk[wrist] - base[wrist]) * 0.5
            conf = np.ones((17, 1), dtype=np.float32) * 0.9
            frames[t, :, :2] = sk; frames[t, :, 2:] = conf
            labels[t] = cid; t += 1
    return frames[:t], labels[:t], actions

rng = np.random.RandomState(777)
ep_frames, ep_labels, ep_actions = generate_test_episode(rng)
T_total = len(ep_frames)

print(f"\n{'='*80}")
print(f"TEST EPISODE: {T_total} frames")
print(f"Actions: {[(a, d) for a, d in ep_actions]}")

# Print action boundaries
offset = 0
boundaries = []
for action, dur in ep_actions:
    boundaries.append((offset, offset + dur - 1, action))
    print(f"  Frames {offset:3d}-{offset+dur-1:3d}: {action} ({dur} frames)")
    offset += dur

# ============================================================
# TASK 1: TRACE EXACT WINDOW at key frames
# ============================================================
print(f"\n{'='*80}")
print("TASK 1: TRACE EXACT WINDOW COMPOSITION")
print(f"{'='*80}")

check_frames = [20, 40, 60, 80, 100, 120, 140, 160, 180]
check_frames = [f for f in check_frames if f < T_total]

for frame_idx in check_frames:
    buf_start = max(0, frame_idx - 59)
    buf_end = frame_idx + 1
    buf_len = buf_end - buf_start

    window_labels = ep_labels[buf_start:buf_end]
    label_counts = Counter(CLASS_NAMES[l] for l in window_labels)
    
    recent_15 = window_labels[-15:]
    train_label = CLASS_NAMES[np.bincount(recent_15, minlength=NUM_CLASSES).argmax()]
    
    center = buf_start + buf_len // 2
    center_label = CLASS_NAMES[ep_labels[center]]
    
    end_label = CLASS_NAMES[ep_labels[frame_idx]]
    
    padded_len = 60
    valid_frames = min(buf_len, 60)
    padding_frames = padded_len - valid_frames

    print(f"\nFrame {frame_idx:3d}:")
    print(f"  Window:       [{buf_start}, {buf_end}) = {buf_len} frames")
    print(f"  Valid frames: {valid_frames}")
    print(f"  Padded with:  {padding_frames} edge-replicated frames")
    print(f"  True label (end):     {end_label}")
    print(f"  True label (center):  {center_label}")
    print(f"  Train label (mode-15): {train_label}")
    print(f"  Label distribution:   {dict(label_counts)}")


# ============================================================
# TASK 2: LABEL SEMANTICS
# ============================================================
print(f"\n{'='*80}")
print("TASK 2: LABEL SEMANTICS ANALYSIS")
print(f"{'='*80}")

print("""
TRAINING label assignment (from generate_action_data_v2.py):
  - Windows are extracted from continuous episodes
  - Window sizes vary: [16, 24, 30, 40, 48, 60] frames
  - Label = MODE of the LAST 15 frames of the window
  - This means the label describes the RECENT action at the END of the window

INFERENCE (rolling window in notebook Part 10):
  - Uses a deque(maxlen=60) that grows from 0 to 60
  - Predicts when buffer has >= 16 frames
  - Pads buffer to 60 frames using edge replication
  - Feeds the ENTIRE padded 60-frame sequence to the model
  - The model produces ONE prediction for the whole window

MISMATCH ANALYSIS:
  1. Training uses variable-length windows (16-60) padded to 60
  2. Inference uses a GROWING window from 0->60, then a SLIDING window
  3. During training, the label corresponds to the last 15 frames
     of the ORIGINAL (pre-padded) window
  4. During inference, the window includes ALL accumulated history
  
  The model was trained on windows where the action in the last 15 frames
  defines the label. But during inference, the ENTIRE 60 frames go through GAP
  (Global Average Pooling), which means ALL frames contribute equally to the
  output, not just the last 15.
  
  THIS IS THE ROOT CAUSE: The model learns to classify based on the dominant
  pattern across the entire window via GAP, but the label only reflects the
  last 15 frames. At test time, with a continuously growing/sliding window,
  old frames from previous actions contaminate the current prediction.
""")


# ============================================================
# TASK 3: TEST MULTIPLE WINDOW LENGTHS
# ============================================================
print(f"\n{'='*80}")
print("TASK 3: MULTI-WINDOW INFERENCE TEST")
print(f"{'='*80}")

def predict_with_window(frames_buf, target_len):
    """Pad/truncate to target_len and predict."""
    L = len(frames_buf)
    padded = np.zeros((target_len, 17, 3), dtype=np.float32)
    use = min(L, target_len)
    padded[:use] = frames_buf[-use:]
    for i in range(use, target_len):
        padded[i] = frames_buf[-1]
    
    tensor = torch.from_numpy(padded).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=-1).cpu().numpy().squeeze()
    return probs

class ActionSmoother:
    def __init__(self, window_size=15, min_count=8):
        self.window = deque(maxlen=window_size)
        self.min_count = min_count
        self.current = "idle"
    def update(self, action):
        self.window.append(action)
        if len(self.window) >= self.min_count:
            counts = Counter(self.window)
            top = counts.most_common(1)[0]
            if top[1] >= self.min_count:
                self.current = top[0]
        return self.current

best_config = None
best_score = 0

for window_len in [20, 30, 40, 60]:
    print(f"\n{'---'*25}")
    print(f"WINDOW LENGTH = {window_len}")
    print(f"{'---'*25}")
    
    smoother = ActionSmoother(window_size=15, min_count=8)
    buf = deque(maxlen=window_len)
    transitions = []
    last_smoothed = 'idle'
    
    for t in range(T_total):
        buf.append(ep_frames[t])
        
        if len(buf) >= min(16, window_len):
            probs = predict_with_window(np.array(list(buf)), window_len)
            idx = int(np.argmax(probs))
            raw_pred = CLASS_NAMES[idx]
            conf = probs[idx]
        else:
            raw_pred = 'idle'
            conf = 0.0
            probs = np.zeros(NUM_CLASSES)
        
        smoothed = smoother.update(raw_pred)
        
        if smoothed != last_smoothed:
            transitions.append((t, last_smoothed, smoothed, CLASS_NAMES[ep_labels[t]]))
            last_smoothed = smoothed
        
        if t % 20 == 0:
            true_label = CLASS_NAMES[ep_labels[t]]
            prob_str = " ".join([f"{CLASS_NAMES[i]}={probs[i]:.2f}" for i in range(NUM_CLASSES)])
            print(f"  F{t:3d} | True: {true_label:7s} | Raw: {raw_pred:7s} ({conf:.2f}) | Smooth: {smoothed:7s} | {prob_str}")
    
    detected_seq = ['idle'] + [to for _, _, to, _ in transitions]
    expected = ['idle', 'reach', 'pick', 'move', 'place', 'release', 'idle']
    
    print(f"\n  Transitions:")
    for t, frm, to, true in transitions:
        match = 'Y' if to == true else 'N'
        print(f"    Frame {t:3d}: {frm} -> {to}  (true: {true}) {match}")
    
    print(f"\n  Expected:  {expected}")
    print(f"  Detected:  {detected_seq}")
    
    correct = sum(1 for e, d in zip(expected, detected_seq) if e == d)
    score = correct / len(expected)
    print(f"  Match: {correct}/{len(expected)} = {score:.0%}")
    
    if score > best_score:
        best_score = score
        best_config = f"window={window_len}, smoother(15,8)"


# ============================================================
# TASK 4: WARM-UP / PADDING ANALYSIS
# ============================================================
print(f"\n{'='*80}")
print("TASK 4: WARM-UP / PADDING ANALYSIS")
print(f"{'='*80}")

print("\nAnalyzing early frames (0-60) with 60-frame window:")
for t in range(0, min(65, T_total), 5):
    buf_len = t + 1
    valid = min(buf_len, 60)
    padded = 60 - valid
    pad_ratio = padded / 60
    
    buf = ep_frames[:t+1]
    probs = predict_with_window(buf, 60)
    idx = int(np.argmax(probs))
    raw_pred = CLASS_NAMES[idx]
    
    true_label = CLASS_NAMES[ep_labels[t]]
    print(f"  Frame {t:3d} | Valid: {valid:2d}/60 | Padded: {padded:2d} ({pad_ratio:.0%}) | "
          f"True: {true_label:7s} | Pred: {raw_pred:7s} ({probs[idx]:.2f})")


# ============================================================
# TASK 5: FULL PROBABILITY VECTORS
# ============================================================
print(f"\n{'='*80}")
print("TASK 5: FULL PROBABILITY VECTORS (60-frame window)")
print(f"{'='*80}")

header = f"{'Frame':>5s} | {'True':>7s} | " + " | ".join([f"{c:>7s}" for c in CLASS_NAMES]) + " | Pred"
print(header)
print("-" * len(header))

buf = deque(maxlen=60)
for t in range(T_total):
    buf.append(ep_frames[t])
    
    if t % 10 == 0 or t < 20:
        if len(buf) >= 16:
            probs = predict_with_window(np.array(list(buf)), 60)
            idx = int(np.argmax(probs))
            raw_pred = CLASS_NAMES[idx]
            
            true_label = CLASS_NAMES[ep_labels[t]]
            prob_str = " | ".join([f"{probs[i]:7.3f}" for i in range(NUM_CLASSES)])
            correct_prob = probs[ep_labels[t]]
            marker = "  Y" if raw_pred == true_label else f"  N (true has {correct_prob:.3f})"
            print(f"{t:5d} | {true_label:>7s} | {prob_str} | {raw_pred:>7s}{marker}")


# ============================================================
# TASK 6: PREPROCESSING COMPARISON
# ============================================================
print(f"\n{'='*80}")
print("TASK 6: PREPROCESSING COMPARISON")
print(f"{'='*80}")

data_dirs = ["data/action_v2/train", "data/action_v2/val"]
for ddir in data_dirs:
    path = os.path.join(ddir, "samples.npz")
    if os.path.exists(path):
        d = np.load(path)
        X = d['sequences']
        print(f"\n[{ddir}]")
        print(f"  Shape: {X.shape}")
        print(f"  dtype: {X.dtype}")
        print(f"  Mean:  {X.mean():.6f}")
        print(f"  Std:   {X.std():.6f}")
        print(f"  Min:   {X.min():.6f}")
        print(f"  Max:   {X.max():.6f}")
        print(f"  Channel 0 (x) range: [{X[:,:,:,0].min():.4f}, {X[:,:,:,0].max():.4f}]")
        print(f"  Channel 1 (y) range: [{X[:,:,:,1].min():.4f}, {X[:,:,:,1].max():.4f}]")
        print(f"  Channel 2 (conf) range: [{X[:,:,:,2].min():.4f}, {X[:,:,:,2].max():.4f}]")
        break

print(f"\n[Rolling inference episode]")
print(f"  Shape: {ep_frames.shape}")
print(f"  dtype: {ep_frames.dtype}")
print(f"  Mean:  {ep_frames.mean():.6f}")
print(f"  Std:   {ep_frames.std():.6f}")
print(f"  Min:   {ep_frames.min():.6f}")
print(f"  Max:   {ep_frames.max():.6f}")
print(f"  Channel 0 (x) range: [{ep_frames[:,:,0].min():.4f}, {ep_frames[:,:,0].max():.4f}]")
print(f"  Channel 1 (y) range: [{ep_frames[:,:,1].min():.4f}, {ep_frames[:,:,1].max():.4f}]")
print(f"  Channel 2 (conf) range: [{ep_frames[:,:,2].min():.4f}, {ep_frames[:,:,2].max():.4f}]")

print(f"\n[Augmentation impact on training data]")
print(f"  Training v2 generator applies:")
print(f"  - Random rotation (0-180 degrees) with 70% probability")
print(f"  - Scale: uniform(0.7, 1.3)")
print(f"  - Translation: uniform(-0.2, 0.2) on both axes")
print(f"  This means training data has MUCH wider coordinate range than raw episode")
print(f"  The rolling episode is UN-AUGMENTED (no rotation, no scale, no translate)")
print(f"  This creates a domain gap even though the motion patterns are similar")


# ============================================================
# TASK 3b: SHORT WINDOW WITH LAST-N-FRAMES ONLY
# ============================================================
print(f"\n{'='*80}")
print("TASK 3b: SHORT WINDOW - USE ONLY LAST N FRAMES (no history contamination)")
print(f"{'='*80}")

for window_len in [20, 30]:
    print(f"\n{'---'*25}")
    print(f"LAST-{window_len} FRAMES ONLY (no growing buffer)")
    print(f"{'---'*25}")
    
    smoother = ActionSmoother(window_size=10, min_count=6)
    transitions = []
    last_smoothed = 'idle'
    
    for t in range(T_total):
        start = max(0, t - window_len + 1)
        window = ep_frames[start:t+1]
        
        if len(window) >= 10:
            probs = predict_with_window(window, window_len)
            idx = int(np.argmax(probs))
            raw_pred = CLASS_NAMES[idx]
            conf = probs[idx]
        else:
            raw_pred = 'idle'
            conf = 0.0
            probs = np.zeros(NUM_CLASSES)
        
        smoothed = smoother.update(raw_pred)
        
        if smoothed != last_smoothed:
            transitions.append((t, last_smoothed, smoothed, CLASS_NAMES[ep_labels[t]]))
            last_smoothed = smoothed
        
        if t % 20 == 0:
            true_label = CLASS_NAMES[ep_labels[t]]
            print(f"  F{t:3d} | True: {true_label:7s} | Raw: {raw_pred:7s} ({conf:.2f}) | Smooth: {smoothed:7s}")
    
    detected_seq = ['idle'] + [to for _, _, to, _ in transitions]
    expected = ['idle', 'reach', 'pick', 'move', 'place', 'release', 'idle']
    
    print(f"\n  Transitions:")
    for t, frm, to, true in transitions:
        match = 'Y' if to == true else 'N'
        print(f"    Frame {t:3d}: {frm} -> {to}  (true: {true}) {match}")
    
    print(f"\n  Expected:  {expected}")
    print(f"  Detected:  {detected_seq}")
    
    correct = sum(1 for e, d in zip(expected, detected_seq) if e == d)
    score = correct / len(expected)
    print(f"  Match: {correct}/{len(expected)} = {score:.0%}")
    
    if score > best_score:
        best_score = score
        best_config = f"last-{window_len}, smoother(10,6)"


# ============================================================
# FINAL SUMMARY
# ============================================================
print(f"\n{'='*80}")
print("DIAGNOSIS SUMMARY")
print(f"{'='*80}")

print(f"""
BEST CONFIGURATION FOUND: {best_config}
BEST SEQUENCE MATCH: {best_score:.0%}

KEY FINDINGS:

1. LABEL SEMANTICS MISMATCH:
   - Training: label = mode of LAST 15 frames of a variable-length window
   - Model: uses Global Average Pooling over ALL temporal frames
   - This means the model sees equal contribution from all frames,
     but the label only describes the last 15 frames
   - For held-out evaluation (same window extraction), this works fine
   - For continuous rolling inference, old frames contaminate predictions

2. WARM-UP CONTAMINATION:
   - Early frames (0-59) use heavy zero/edge padding
   - The edge replication creates an artificial static pattern
   - The model may learn to associate heavy padding with 'idle'

3. DOMAIN GAP:
   - Training data has rotation/scale/translation augmentation
   - Rolling test episode has NO augmentation
   - The BatchNorm statistics may not match

4. POTENTIAL SOLUTIONS (in order of effort):
   a. Use shorter sliding window (20-30 frames) instead of growing buffer
   b. Use only last N frames, no history accumulation
   c. Add stride/hop to avoid overlapping contamination
   d. Retrain with center-frame labels instead of mode-15 labels
   e. Replace GAP with attention to weight recent frames more
""")

if best_score >= 6/7:
    print("DECISION: ST-GCN READY (acceptable sequence detection)")
    print(f"USE CONFIG: {best_config}")
elif best_score >= 4/7:
    print("DECISION: PARTIAL - needs inference tuning, no retrain needed")
    print(f"BEST SO FAR: {best_config}")
else:
    print("DECISION: ROLLING INFERENCE FAILS - see docs/STGCN_ROLLING_FAILURE_REPORT.md")
