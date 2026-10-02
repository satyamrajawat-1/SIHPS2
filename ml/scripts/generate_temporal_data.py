"""
generate_temporal_data.py
Generates a synthetic multimodal temporal dataset suitable for training the Temporal GRU model.
"""

import numpy as np
import os
import json
import argparse
from pathlib import Path

ACTION_CLASSES = {'idle': 0, 'reach': 1, 'pick': 2, 'move': 3, 'place': 4, 'release': 5}
NUM_ACTIONS = len(ACTION_CLASSES)

NORMAL_PROCEDURE = ['idle', 'reach', 'pick', 'move', 'place', 'release', 'idle']
NUM_STEPS = len(NORMAL_PROCEDURE)

def generate_prototypes(dim=512, seed=42):
    rng = np.random.RandomState(seed)
    prototypes = {}
    for action in ACTION_CLASSES:
        vec = rng.randn(dim).astype(np.float32)
        vec /= np.linalg.norm(vec)
        vec *= 5.0 # reasonable scale
        prototypes[action] = vec
    return prototypes

def generate_sequence(
    rng,
    prototypes,
    seq_def,
    dim=512,
    transition_len=5
):
    """
    seq_def: list of tuples (action_name, duration)
    """
    T = sum(d for _, d in seq_def)
    features = np.zeros((T, dim), dtype=np.float32)
    y_action = np.zeros(T, dtype=np.int64)
    y_step = np.zeros(T, dtype=np.int64)
    y_next_action = np.zeros(T, dtype=np.int64)
    y_anomaly = np.zeros(T, dtype=np.float32)
    
    # Generate feature sequence
    base_drift = rng.randn(dim).astype(np.float32) * 0.1 # slow feature drift
    seq_variation = rng.randn(dim).astype(np.float32) * 0.5 # sequence specific variation
    
    idx = 0
    pure_features = np.zeros((T, dim), dtype=np.float32)
    
    for action, duration in seq_def:
        proto = prototypes[action]
        for _ in range(duration):
            pure_features[idx] = proto
            y_action[idx] = ACTION_CLASSES[action]
            idx += 1
            
    # Smooth transitions
    smoothed_features = pure_features.copy()
    for i in range(1, T):
        smoothed_features[i] = 0.7 * smoothed_features[i-1] + 0.3 * pure_features[i]
        
    # Add noise and drift
    for i in range(T):
        drift = base_drift * (i / T)
        noise = rng.randn(dim).astype(np.float32) * 0.2
        features[i] = smoothed_features[i] + seq_variation + drift + noise
        
    # Generate targets
    # We maintain a state machine of the normal procedure
    curr_step = 0
    for i in range(T):
        current_action = list(ACTION_CLASSES.keys())[y_action[i]]
        expected_action = NORMAL_PROCEDURE[curr_step]
        
        # Check if we advanced to the next valid step
        if curr_step + 1 < NUM_STEPS and current_action == NORMAL_PROCEDURE[curr_step + 1]:
            curr_step += 1
            expected_action = NORMAL_PROCEDURE[curr_step]
            
        is_anomaly = 0.0
        if current_action != expected_action:
            is_anomaly = 1.0
            
        y_step[i] = curr_step
        y_anomaly[i] = is_anomaly
        
        # Next action is the one following the current valid step
        next_step = min(curr_step + 1, NUM_STEPS - 1)
        next_a = NORMAL_PROCEDURE[next_step]
        y_next_action[i] = ACTION_CLASSES[next_a]
        
    return features, y_action, y_step, y_next_action, y_anomaly

def make_normal_def(rng):
    return [
        ('idle', rng.randint(15, 30)),
        ('reach', rng.randint(20, 35)),
        ('pick', rng.randint(10, 20)),
        ('move', rng.randint(30, 50)),
        ('place', rng.randint(15, 30)),
        ('release', rng.randint(10, 20)),
        ('idle', rng.randint(20, 40)),
    ]

def make_anomalous_def(rng, anomaly_type):
    if anomaly_type == 'skipped_step':
        # skip pick
        return [
            ('idle', rng.randint(15, 30)),
            ('reach', rng.randint(20, 35)),
            ('move', rng.randint(30, 50)),
            ('place', rng.randint(15, 30)),
            ('release', rng.randint(10, 20)),
            ('idle', rng.randint(20, 40)),
        ]
    elif anomaly_type == 'out_of_order':
        # idle -> pick -> reach
        return [
            ('idle', rng.randint(15, 30)),
            ('pick', rng.randint(10, 20)),
            ('reach', rng.randint(20, 35)),
            ('move', rng.randint(30, 50)),
            ('place', rng.randint(15, 30)),
            ('release', rng.randint(10, 20)),
            ('idle', rng.randint(20, 40)),
        ]
    elif anomaly_type == 'wrong_action':
        # idle -> place
        return [
            ('idle', rng.randint(15, 30)),
            ('place', rng.randint(15, 30)),
            ('idle', rng.randint(20, 40)),
        ]
    elif anomaly_type == 'premature_release':
        # skip place
        return [
            ('idle', rng.randint(15, 30)),
            ('reach', rng.randint(20, 35)),
            ('pick', rng.randint(10, 20)),
            ('move', rng.randint(30, 50)),
            ('release', rng.randint(10, 20)),
            ('idle', rng.randint(20, 40)),
        ]
    elif anomaly_type == 'repeated_action':
        # reach -> reach
        return [
            ('idle', rng.randint(15, 30)),
            ('reach', rng.randint(10, 15)),
            ('idle', rng.randint(5, 10)),
            ('reach', rng.randint(20, 35)),
            ('pick', rng.randint(10, 20)),
            ('move', rng.randint(30, 50)),
            ('place', rng.randint(15, 30)),
            ('release', rng.randint(10, 20)),
            ('idle', rng.randint(20, 40)),
        ]
    elif anomaly_type == 'unexpected_idle':
        return [
            ('idle', rng.randint(15, 30)),
            ('reach', rng.randint(20, 35)),
            ('idle', rng.randint(20, 40)),
        ]
    return make_normal_def(rng)

def generate_dataset(num_seqs, split, prototypes, rng):
    X = []
    Y_action = []
    Y_step = []
    Y_next_action = []
    Y_anomaly = []
    
    anomaly_types = ['skipped_step', 'out_of_order', 'wrong_action', 'premature_release', 'repeated_action', 'unexpected_idle']
    
    for i in range(num_seqs):
        if rng.rand() < 0.7:
            # 70% normal
            seq_def = make_normal_def(rng)
        else:
            # 30% anomalous
            atype = rng.choice(anomaly_types)
            seq_def = make_anomalous_def(rng, atype)
            
        f, a, s, n, an = generate_sequence(rng, prototypes, seq_def)
        X.append(f)
        Y_action.append(a)
        Y_step.append(s)
        Y_next_action.append(n)
        Y_anomaly.append(an)
        
    return X, Y_action, Y_step, Y_next_action, Y_anomaly

def pad_sequences(seqs, max_len=None):
    if max_len is None:
        max_len = max(len(s) for s in seqs)
    
    # For features, pad with zeros
    # For labels, pad with -100 (standard ignore index)
    padded_X = np.zeros((len(seqs), max_len, 512), dtype=np.float32)
    padded_action = np.full((len(seqs), max_len), -100, dtype=np.int64)
    padded_step = np.full((len(seqs), max_len), -100, dtype=np.int64)
    padded_next = np.full((len(seqs), max_len), -100, dtype=np.int64)
    padded_anomaly = np.full((len(seqs), max_len), -100, dtype=np.float32)
    
    for i, s in enumerate(seqs):
        length = min(len(s), max_len)
        padded_X[i, :length] = s[:length]
        
    return padded_X, padded_action, padded_step, padded_next, padded_anomaly

def save_split(path, X, Y_a, Y_s, Y_n, Y_an):
    os.makedirs(path, exist_ok=True)
    # Convert lists to padded arrays
    pX, pA, pS, pN, pAn = pad_sequences(X)
    
    max_len = max(len(x) for x in X)
    pX, pA, pS, pN, pAn = pad_sequences(X, max_len)
    for i in range(len(X)):
        length = min(len(X[i]), max_len)
        pA[i, :length] = Y_a[i][:length]
        pS[i, :length] = Y_s[i][:length]
        pN[i, :length] = Y_n[i][:length]
        pAn[i, :length] = Y_an[i][:length]
        
    np.savez_compressed(
        os.path.join(path, "samples.npz"),
        features=pX,
        action=pA,
        step=pS,
        next_action=pN,
        anomaly=pAn,
        lengths=np.array([len(x) for x in X], dtype=np.int32)
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_train', type=int, default=700)
    parser.add_argument('--num_val', type=int, default=150)
    parser.add_argument('--num_test', type=int, default=150)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out_dir', type=str, default='data/temporal_v1')
    args = parser.parse_args()
    
    rng = np.random.RandomState(args.seed)
    prototypes = generate_prototypes(dim=512, seed=args.seed)
    
    print(f"Generating temporal dataset v1 at {args.out_dir}...")
    
    # Generate splits
    X_train, Ya_train, Ys_train, Yn_train, Yan_train = generate_dataset(args.num_train, 'train', prototypes, rng)
    X_val, Ya_val, Ys_val, Yn_val, Yan_val = generate_dataset(args.num_val, 'val', prototypes, rng)
    X_test, Ya_test, Ys_test, Yn_test, Yan_test = generate_dataset(args.num_test, 'test', prototypes, rng)
    
    save_split(os.path.join(args.out_dir, 'train'), X_train, Ya_train, Ys_train, Yn_train, Yan_train)
    save_split(os.path.join(args.out_dir, 'val'), X_val, Ya_val, Ys_val, Yn_val, Yan_val)
    save_split(os.path.join(args.out_dir, 'test'), X_test, Ya_test, Ys_test, Yn_test, Yan_test)
    
    # Save metadata
    meta = {
        'version': 'temporal_v1',
        'seed': args.seed,
        'input_dim': 512,
        'num_actions': NUM_ACTIONS,
        'num_steps': NUM_STEPS,
        'action_classes': ACTION_CLASSES,
        'normal_procedure': NORMAL_PROCEDURE,
        'splits': {
            'train': args.num_train,
            'val': args.num_val,
            'test': args.num_test
        }
    }
    with open(os.path.join(args.out_dir, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)
        
    print("Generation complete.")
    
    # Sanity checks
    print("\n--- SANITY CHECKS ---")
    d = np.load(os.path.join(args.out_dir, 'train', 'samples.npz'))
    print(f"Train features shape: {d['features'].shape}")
    print(f"Train labels shape: {d['action'].shape}")
    print(f"Feature min/max: {d['features'].min():.4f}, {d['features'].max():.4f}")
    
    # Print one normal and one anomalous example
    for i in range(5):
        anomalies = d['anomaly'][i]
        length = d['lengths'][i]
        acts = d['action'][i][:length]
        is_anom = (anomalies[:length] > 0).any()
        anom_str = "ANOMALOUS" if is_anom else "NORMAL"
        act_names = [list(ACTION_CLASSES.keys())[a] for a in acts]
        
        # Just print transitions to make it readable
        trans = []
        last_a = None
        for a in act_names:
            if a != last_a:
                trans.append(a)
                last_a = a
        print(f"Example {i} ({anom_str}): {' -> '.join(trans)}")
        if is_anom:
            anom_idx = np.where(anomalies[:length] > 0)[0][0]
            print(f"  Anomaly started at frame {anom_idx} (action: {act_names[anom_idx]})")
            print(f"  Next expected action was: {list(ACTION_CLASSES.keys())[d['next_action'][i][anom_idx]]}")

if __name__ == '__main__':
    main()
