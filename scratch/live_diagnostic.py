import cv2
import time
import numpy as np
import json
import torch
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.experiment.schema.loader import load_experiment
from ml.pipeline.inference import ASTRAPipeline

def record_action(pipeline, action_name, duration=10):
    print(f"\n>>> Get ready to perform: {action_name.upper()}")
    for i in range(3, 0, -1):
        print(f"Starting in {i}...")
        time.sleep(1)
    
    print(f"--- RECORDING {action_name.upper()} FOR {duration} SECONDS ---")
    cap = cv2.VideoCapture(0)
    
    data = []
    start_time = time.time()
    
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break
            
        res = pipeline.process_frame(frame)
        
        # We need to capture the normalized ST-GCN input.
        # It's in pipeline.skeleton_buffer
        if len(pipeline.skeleton_buffer) > 0:
            seq = np.array(list(pipeline.skeleton_buffer))
            
            # Re-run stgcn manually to get logits
            stgcn = pipeline.action_recognizer
            T, J, C = seq.shape
            tensor = torch.from_numpy(seq).permute(2, 0, 1).unsqueeze(0).float()
            
            # Pad C if needed
            if C < stgcn.in_channels:
                extra = torch.zeros((1, stgcn.in_channels - C, T, J), dtype=torch.float32)
                tensor = torch.cat([tensor, extra], dim=1)
            elif C > stgcn.in_channels:
                tensor = tensor[:, :stgcn.in_channels, :, :]
                
            tensor = tensor.to(stgcn.device)
            with torch.no_grad():
                logits = stgcn.model(tensor)
                probs = torch.softmax(logits, dim=-1).cpu().numpy().squeeze(0)
                
            top_3_idx = np.argsort(probs)[-3:][::-1]
            top_3 = {stgcn.action_classes[i]: float(probs[i]) for i in top_3_idx}
            
            data.append({
                "action_performed": action_name,
                "skeleton_seq": seq.tolist(),
                "predicted_class": stgcn.action_classes[top_3_idx[0]],
                "confidence": float(probs[top_3_idx[0]]),
                "top_3": top_3,
                "raw_probs": probs.tolist(),
                "pose_confidence": res.get("pose_confidence", 0.0),
                "temporal_prediction": res.get("state_machine", {}).get("temporal_action", "N/A")
            })
            
        cv2.imshow("Diagnostic Recording", frame)
        cv2.waitKey(1)
        
    cap.release()
    cv2.destroyAllWindows()
    return data

def main():
    print("Loading pipeline...")
    config = load_experiment("experiments/temporal_demo")
    pipeline = ASTRAPipeline(config, device="cpu", use_mock=False)
    pipeline.load_models(
        detection_weights="checkpoints/detection/YOLO11S-ASTRA-v1/weights/best.pt",
        action_weights="checkpoints/action/STGCNPP-ASTRA-v3/best.pt",
        temporal_weights="checkpoints/action/TemporalGRU-ASTRA-v1/best.pt"
    )
    
    actions = ["idle", "reach", "pick", "move", "place", "release"]
    all_data = []
    
    for action in actions:
        data = record_action(pipeline, action, duration=10)
        all_data.extend(data)
        
    # Generate report
    report = ["# LIVE ST-GCN DIAGNOSTIC V2\n"]
    report.append("| Performed | ST-GCN prediction | Confidence | Top-3 | GRU prediction |")
    report.append("|---|---|---|---|---|")
    
    for action in actions:
        action_data = [d for d in all_data if d["action_performed"] == action]
        if not action_data:
            continue
            
        # Get middle frame of the recording as representative
        rep = action_data[len(action_data)//2]
        
        top3_str = "<br>".join([f"{k}: {v:.2f}" for k, v in rep["top_3"].items()])
        report.append(f"| {action.upper()} | {rep['predicted_class'].upper()} | {rep['confidence']:.2f} | {top3_str} | {rep['temporal_prediction']} |")
        
    with open("docs/LIVE_STGCN_DIAGNOSTIC_V2.md", "w") as f:
        f.write("\n".join(report))
        
    # Save dataset for adaptation
    os.makedirs("data/real_adaptation", exist_ok=True)
    with open("data/real_adaptation/raw_data.json", "w") as f:
        json.dump(all_data, f)
        
    print("Diagnostic complete. Report saved.")

if __name__ == "__main__":
    main()
