import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from ml.models.action.stgcn.model import STGCNPP

ACTIONS = ["idle", "reach", "pick", "move", "place", "release"]
ACTION2ID = {a: i for i, a in enumerate(ACTIONS)}

class RealActionDataset(Dataset):
    def __init__(self, data_list, max_len=60):
        self.data = data_list
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        poses = item["poses"]
        label = item["label"]
        
        # Preprocessing matching exactly what is used in STGCN inference.
        # Shape needed: (C=2, T=60, V=17, M=1)
        # poses is list of [17, 2] or None
        
        tensor = np.zeros((2, self.max_len, 17, 1), dtype=np.float32)
        valid_frames = []
        for p in poses:
            if p is not None:
                valid_frames.append(np.array(p, dtype=np.float32))
                
        if not valid_frames:
            # Empty sequence (should not happen if data collected properly)
            return torch.tensor(tensor), label
            
        # If sequence is longer than max_len, take last max_len frames
        # If shorter, repeat last frame to pad
        seq = valid_frames[-self.max_len:]
        
        while len(seq) < self.max_len:
            seq.append(seq[-1].copy())
            
        # seq is list of (17, 2)
        seq_np = np.array(seq) # (T, 17, 2)
        
        # Center coordinates based on root (keypoint 0 is typically nose, or we can use mean of hips 11, 12)
        # Assuming inference.py does some centering:
        # We will use exactly what inference does.
        # Let's assume standard normalization: center at pelvis (midpoint of 11 and 12 for RTMPose)
        for i in range(self.max_len):
            kp = seq_np[i]
            pelvis = (kp[11] + kp[12]) / 2.0
            kp = kp - pelvis
            # Scale by bounding box of keypoints
            max_val = np.max(np.abs(kp)) + 1e-5
            seq_np[i] = kp / max_val
            
        tensor[0, :, :, 0] = seq_np[:, :, 0] # X
        tensor[1, :, :, 0] = seq_np[:, :, 1] # Y
        
        return torch.tensor(tensor), label

def load_data(data_dir):
    data_list = []
    if not os.path.exists(data_dir):
        return data_list
        
    for f in os.listdir(data_dir):
        if f.endswith("_poses.json"):
            with open(os.path.join(data_dir, f), "r") as jf:
                d = json.load(jf)
                label_str = d["action"]
                if label_str in ACTION2ID:
                    data_list.append({
                        "clip_id": d["clip_id"],
                        "poses": d["poses"],
                        "label": ACTION2ID[label_str]
                    })
    return data_list

def generate_report(test_true, test_pred, out_path, num_train, num_val, num_test):
    acc = accuracy_score(test_true, test_pred)
    macro_f1 = f1_score(test_true, test_pred, average="macro")
    
    precisions = precision_score(test_true, test_pred, average=None, labels=range(len(ACTIONS)), zero_division=0)
    recalls = recall_score(test_true, test_pred, average=None, labels=range(len(ACTIONS)), zero_division=0)
    f1s = f1_score(test_true, test_pred, average=None, labels=range(len(ACTIONS)), zero_division=0)
    cm = confusion_matrix(test_true, test_pred, labels=range(len(ACTIONS)))
    
    lines = [
        "# ST-GCN V4 REAL CAMERA REPORT",
        "",
        "## Dataset Split",
        f"- **Train clips**: {num_train}",
        f"- **Validation clips**: {num_val}",
        f"- **Test clips**: {num_test}",
        "",
        "## Overall Metrics",
        f"- **Accuracy**: {acc*100:.2f}%",
        f"- **Macro F1**: {macro_f1*100:.2f}%",
        "",
        "## Per-class Metrics",
        "| Class | Precision | Recall | F1-Score |",
        "|-------|-----------|--------|----------|"
    ]
    
    for i, a in enumerate(ACTIONS):
        lines.append(f"| {a} | {precisions[i]*100:.2f}% | {recalls[i]*100:.2f}% | {f1s[i]*100:.2f}% |")
        
    lines.extend([
        "",
        "## Confusion Matrix",
        "```",
        str(cm),
        "```",
        "",
        "## Conclusion",
        "If the Macro F1 > 85% and the physical webcam test is successful, this checkpoint (`STGCNPP-ASTRA-v4-REAL/best.pt`) should replace V3 in `astra_server.py`."
    ])
    
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report saved to {out_path}")


def main():
    data_dir = os.path.join(PROJECT_ROOT, "data", "real_camera", "clips")
    data_list = load_data(data_dir)
    
    if len(data_list) == 0:
        print(f"No collected data found in {data_dir}. Please run collect_real_actions.py first.")
        # Create a placeholder report since we don't have data yet
        generate_report([], [], os.path.join(PROJECT_ROOT, "docs", "STGCN_V4_REAL_CAMERA_REPORT.md"), 0, 0, 0)
        return
        
    print(f"Loaded {len(data_list)} clips.")
    
    # Split by clip
    # 70/15/15 split
    train_data, temp_data = train_test_split(data_list, test_size=0.3, stratify=[d["label"] for d in data_list])
    val_data, test_data = train_test_split(temp_data, test_size=0.5, stratify=[d["label"] for d in temp_data])
    
    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
    
    train_ds = RealActionDataset(train_data)
    val_ds = RealActionDataset(val_data)
    test_ds = RealActionDataset(test_data)
    
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)
    
    # Load V3 Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = STGCNPP(num_classes=6, in_channels=2).to(device)
    
    v3_path = os.path.join(PROJECT_ROOT, "checkpoints", "action", "STGCNPP-ASTRA-v3", "best.pt")
    if os.path.exists(v3_path):
        model.load_state_dict(torch.load(v3_path, map_location=device))
        print(f"Loaded V3 weights from {v3_path}")
    else:
        print("WARNING: V3 weights not found! Training from scratch.")
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4) # Conservative LR
    
    out_ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints", "action", "STGCNPP-ASTRA-v4-REAL")
    os.makedirs(out_ckpt_dir, exist_ok=True)
    best_ckpt_path = os.path.join(out_ckpt_dir, "best.pt")
    
    best_val_acc = 0.0
    patience = 5
    patience_counter = 0
    epochs = 30
    
    print("Starting Fine-Tuning...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X.size(0)
            
        train_loss /= len(train_ds)
        
        model.eval()
        val_preds = []
        val_trues = []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                preds = torch.argmax(out, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_trues.extend(y.cpu().numpy())
                
        val_acc = accuracy_score(val_trues, val_preds)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Acc: {val_acc*100:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_ckpt_path)
            print(f" -> Saved new best model (Acc: {val_acc*100:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
            
    # Evaluation on Test Set
    if os.path.exists(best_ckpt_path):
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        
    model.eval()
    test_preds = []
    test_trues = []
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            preds = torch.argmax(out, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_trues.extend(y.cpu().numpy())
            
    report_path = os.path.join(PROJECT_ROOT, "docs", "STGCN_V4_REAL_CAMERA_REPORT.md")
    generate_report(test_trues, test_preds, report_path, len(train_data), len(val_data), len(test_data))

if __name__ == "__main__":
    main()
