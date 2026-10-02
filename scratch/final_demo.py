import os
import sys
import time
import numpy as np
import cv2

PROJECT_ROOT = r"c:\Users\asus\Desktop\SIH"
sys.path.insert(0, PROJECT_ROOT)

from ml.experiment.schema.loader import load_experiment
from ml.pipeline.inference import ASTRAPipeline

print("REAL RTMPOSE E2E TEST")
print("=====================")

# Setup
exp_dir = os.path.join(PROJECT_ROOT, "experiments/sample_experiment")
video_path = os.path.join(PROJECT_ROOT, "data/synthetic/session_001/session_001.mp4")
det_weights = os.path.join(PROJECT_ROOT, "checkpoints/detection/YOLO11S-ASTRA-v1/weights/best.pt")
act_weights = os.path.join(PROJECT_ROOT, "checkpoints/action/STGCNPP-ASTRA-v3/best.pt")
temp_weights = os.path.join(PROJECT_ROOT, "checkpoints/action/TemporalGRU-ASTRA-v1/best.pt")
out_dir = os.path.join(PROJECT_ROOT, "eval_results/real_e2e_demo")
os.makedirs(out_dir, exist_ok=True)

config = load_experiment(exp_dir)
pipeline = ASTRAPipeline(config, device="cpu", use_mock=False, mock_pose=False)
pipeline.load_models(detection_weights=det_weights, action_weights=act_weights, temporal_weights=temp_weights)

# Monkey-patch to measure latencies
stats = {"yolo": [], "rtmpose": [], "stgcn": [], "feature_agg": [], "temporal_gru": []}

pose_confidences = []
has_valid_pose_count = 0

def measure(name, func):
    def wrapper(*args, **kwargs):
        global has_valid_pose_count
        t0 = time.perf_counter()
        res = func(*args, **kwargs)
        stats[name].append((time.perf_counter() - t0) * 1000)
        
        if name == "rtmpose" and res:
            has_valid_pose_count += 1
            for p in res:
                pose_confidences.extend(p.keypoints_2d[:, 2].tolist())
                
        return res
    return wrapper

pipeline.detector.detect = measure("yolo", pipeline.detector.detect)
pipeline.pose_estimator.predict = measure("rtmpose", pipeline.pose_estimator.predict)
pipeline.action_recognizer.predict = measure("stgcn", pipeline.action_recognizer.predict)
pipeline.feature_aggregator.aggregate = measure("feature_agg", pipeline.feature_aggregator.aggregate)
pipeline.temporal_model.predict = measure("temporal_gru", pipeline.temporal_model.predict)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 15
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

out_path = os.path.join(out_dir, "annotated_output.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

frame_latencies = []
pose_confidences = []
has_valid_pose_count = 0
idx = 0
t_start = time.perf_counter()

print(f"\nProcessing {total_frames} frames...")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    t0 = time.perf_counter()
    result = pipeline.process_frame(frame, timestamp=idx/fps)
    lat = (time.perf_counter() - t0) * 1000
    frame_latencies.append(lat)
    
    num_poses = result.get("poses", 0)
    
    # Render professional GUI
    # Top banner
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (40, 40, 40), -1)
    cv2.putText(frame, "LIVE E2E INFERENCE MODE", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Extract data
    sm = result.get("state_machine", {})
    raw_action = result.get("raw_action", "?")
    val_action = result.get("action", "?")
    step_id = sm.get('current_step_id', '?')
    pose_conf = result.get("pose_confidence", 0.0)
    
    anomalies = sm.get("anomalies", [])
    anomaly_text = anomalies[-1]["type"] if anomalies else "NONE"
    
    # Draw background panels
    cv2.rectangle(frame, (10, 50), (400, 240), (30, 30, 30), -1)
    cv2.rectangle(frame, (10, 50), (400, 240), (100, 100, 100), 2)
    
    y = 75
    dy = 25
    
    # 1. RAW PERCEPTION
    cv2.putText(frame, "1. RAW PERCEPTION", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    y += dy
    color = (0, 0, 255) if raw_action in ["release", "place"] and val_action in ["invalid", "uncertain"] else (255, 255, 255)
    cv2.putText(frame, f"Raw Action: {raw_action}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    y += dy
    cv2.putText(frame, f"Pose Conf:  {pose_conf:.3f}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y += dy
    
    # 2. VALIDATED PROCEDURE ACTION
    cv2.putText(frame, "2. VALIDATED ACTION", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    y += dy
    color = (0, 0, 255) if val_action in ["invalid", "uncertain"] else (0, 255, 0)
    cv2.putText(frame, f"Action: {val_action.upper() if val_action else '?'}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    y += dy
    cv2.putText(frame, f"Anomaly: {anomaly_text}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    y += dy
    
    # 3. PROCEDURE STATE
    cv2.putText(frame, "3. PROCEDURE STATE", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    y += dy
    cv2.putText(frame, f"Step: {step_id}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    t_act = result.get("temporal_action") or "?"
    t_step = result.get("temporal_step") or "?"
    t_next = result.get("temporal_next") or "?"
    t_anom = result.get("anomaly_score", 0.0)
    
    out_video.write(frame)
    
    if idx % 15 == 0:
        print(f"Frame {idx:03d} | Latency: {lat:.1f}ms | YOLO: {result.get('detections', 0)} | Poses: {num_poses} | "
              f"ST-GCN: {result.get('action')} | T-Act: {t_act} | T-Step: {t_step} | T-Next: {t_next} | Anom: {t_anom:.3f} | State: {sm.get('experiment_status')}")
        
    idx += 1

total_runtime = time.perf_counter() - t_start
cap.release()
out_video.release()
pipeline.close()

# Compute final stats
fps_actual = idx / total_runtime if total_runtime > 0 else 0
mean_lat = np.mean(frame_latencies)
p95_lat = np.percentile(frame_latencies, 95)
pose_mean = np.mean(pose_confidences) if pose_confidences else 0
pose_median = np.median(pose_confidences) if pose_confidences else 0
pose_valid_pct = (has_valid_pose_count / idx) * 100 if idx > 0 else 0

def get_mean(lst): return np.mean(lst) if lst else 0

print("\nPerformance:")
print(f"Total frames: {idx}")
print(f"Total runtime: {total_runtime:.2f}s")
print(f"FPS: {fps_actual:.2f}")
print(f"Mean latency: {mean_lat:.1f}ms")
print(f"P95 latency: {p95_lat:.1f}ms")
print("\nComponent Mean Latencies:")
print(f"YOLO: {get_mean(stats['yolo']):.1f}ms")
print(f"RTMPose: {get_mean(stats['rtmpose']):.1f}ms")
print(f"ST-GCN: {get_mean(stats['stgcn']):.1f}ms")
print(f"Feature aggregation: {get_mean(stats['feature_agg']):.1f}ms")
print(f"Temporal GRU: {get_mean(stats['temporal_gru']):.1f}ms")

print("\nPose confidence:")
print(f"Mean keypoint confidence: {pose_mean:.3f}")
print(f"Median keypoint confidence: {pose_median:.3f}")
print(f"Valid person pose frame %: {pose_valid_pct:.1f}%")

print(f"\nAnnotated video saved to: {out_path}")
