import os
import sys
import time
import threading
import cv2
import numpy as np

PROJECT_ROOT = r"c:\Users\asus\Desktop\SIH"
sys.path.insert(0, PROJECT_ROOT)

from ml.experiment.schema.loader import load_experiment
from ml.pipeline.inference import ASTRAPipeline
from ml.experiment.state_graph.state_machine import ExperimentStateMachine
from ml.pipeline.action_smoother import ActionSmoother

class ThreadSafeData:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_timestamp = 0.0
        self.latest_result = {}
        self.processing_fps = 0.0
        self.is_running = True
        self.reset_requested = False

shared_data = ThreadSafeData()

def inference_worker():
    print("[Worker] Initializing ASTRA Pipeline...")
    exp_dir = os.path.join(PROJECT_ROOT, "experiments/sample_experiment")
    det_weights = os.path.join(PROJECT_ROOT, "checkpoints/detection/YOLO11S-ASTRA-v1/weights/best.pt")
    act_weights = os.path.join(PROJECT_ROOT, "checkpoints/action/STGCNPP-ASTRA-v3/best.pt")
    temp_weights = os.path.join(PROJECT_ROOT, "checkpoints/action/TemporalGRU-ASTRA-v1/best.pt")
    
    config = load_experiment(exp_dir)
    pipeline = ASTRAPipeline(config, device="cpu", use_mock=False, mock_pose=False)
    pipeline.load_models(detection_weights=det_weights, action_weights=act_weights, temporal_weights=temp_weights)
    print("[Worker] Models loaded.")
    
    while True:
        with shared_data.lock:
            if not shared_data.is_running:
                break
            frame = shared_data.latest_frame.copy() if shared_data.latest_frame is not None else None
            timestamp = shared_data.latest_timestamp
            reset = shared_data.reset_requested
            shared_data.reset_requested = False
            
        if reset:
            print("[Worker] Resetting pipeline...")
            pipeline.state_machine = ExperimentStateMachine(config)
            action_classes = [a.id for a in config.actions]
            pipeline.action_smoother = ActionSmoother(
                window_size=15, min_persistence=5,
                confidence_threshold=0.3, cooldown_frames=10,
                action_classes=action_classes
            )
            pipeline.skeleton_buffer.clear()
            pipeline.feature_buffer.clear()
            pipeline.frame_count = 0
            with shared_data.lock:
                shared_data.latest_result = {}
                
        if frame is None:
            time.sleep(0.01)
            continue
            
        t0 = time.perf_counter()
        
        try:
            result = pipeline.process_frame(frame, timestamp=timestamp)
            action = result.get("action")
            raw_action = result.get("raw_action")
            if action and action != "idle":
                print(f"[Worker] T={timestamp:.1f} | Raw: {raw_action} | Validated: {action} | Step: {result.get('state_machine', {}).get('current_step_id')}")
        except Exception as e:
            print(f"[Worker] Error during inference: {e}")
            result = {}
            time.sleep(1)
            
        t1 = time.perf_counter()
        processing_fps = 1.0 / (t1 - t0) if t1 > t0 else 0.0
        
        with shared_data.lock:
            shared_data.latest_result = result
            shared_data.processing_fps = processing_fps

    pipeline.close()
    print("[Worker] Terminated.")

def draw_gui(frame, result, processing_fps, cam_fps):
    h, w = frame.shape[:2]
    
    # Draw top banner
    cv2.rectangle(frame, (0, 0), (w, 50), (30, 30, 30), -1)
    cv2.putText(frame, "ASTRA - LIVE HUMAN ACTIVITY RECOGNITION", (15, 33), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Cam FPS: {cam_fps:.1f} | Infer FPS: {processing_fps:.1f}", (w - 350, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    
    if not result:
        cv2.putText(frame, "Initializing Models...", (w//2 - 150, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        return

    sm = result.get("state_machine", {})
    raw_action = result.get("raw_action", "idle")
    if raw_action is None: raw_action = "idle"
    
    val_action = result.get("action", "idle")
    if val_action is None: val_action = "idle"
    
    step_id = sm.get('current_step_id', 'step_01')
    pose_conf = result.get("pose_confidence", 0.0)
    
    anomalies = sm.get("anomalies", [])
    anomaly_text = anomalies[-1]["type"] if anomalies else "NONE"
    
    t_act = result.get("temporal_action")
    if t_act is None: t_act = "N/A"
    
    t_next = result.get("temporal_next")
    if t_next is None: t_next = "N/A"
    
    # Left Panel: Data
    panel_w = 420
    panel_h = 320
    cv2.rectangle(frame, (10, 60), (10 + panel_w, 60 + panel_h), (40, 40, 40), -1)
    cv2.rectangle(frame, (10, 60), (10 + panel_w, 60 + panel_h), (150, 150, 150), 2)
    
    y = 90
    dy = 28
    
    # PERCEPTION
    cv2.putText(frame, "1. PERCEPTION", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
    y += dy
    color_raw = (0, 0, 255) if raw_action in ["release", "place"] and val_action in ["invalid", "uncertain"] else (255, 255, 255)
    cv2.putText(frame, f"Raw Action: {raw_action.upper()}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_raw, 1)
    y += dy
    cv2.putText(frame, f"Pose Conf:  {pose_conf:.3f}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    y += int(dy * 1.5)
    
    # VALIDATION
    cv2.putText(frame, "2. VALIDATED ACTION", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
    y += dy
    color_val = (0, 0, 255) if val_action in ["invalid", "uncertain"] else (0, 255, 0)
    cv2.putText(frame, f"Action: {val_action.upper()}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_val, 2)
    y += dy
    cv2.putText(frame, f"Anomaly: {anomaly_text}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_val, 1)
    y += dy
    cv2.putText(frame, f"Temp Act: {t_act.upper()}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    y += int(dy * 1.5)
    
    # STATE MACHINE
    cv2.putText(frame, "3. PROCEDURE STATE", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
    y += dy
    cv2.putText(frame, f"Step: {step_id}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    y += dy
    cv2.putText(frame, f"Next Expected: {t_next.upper()}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

def main():
    print("Starting ASTRA Live Prototype...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    worker = threading.Thread(target=inference_worker, daemon=True)
    worker.start()
    
    print("\nControls:")
    print(" 'q' - Quit")
    print(" 'r' - Reset Experiment")
    print(" 's' - Save Screenshot\n")
    
    t0 = time.perf_counter()
    frames = 0
    cam_fps = 0.0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera.")
            break
            
        frame = cv2.flip(frame, 1) # Mirror for user convenience
        current_time = time.perf_counter()
        
        frames += 1
        if current_time - t0 > 1.0:
            cam_fps = frames / (current_time - t0)
            frames = 0
            t0 = current_time
            
        with shared_data.lock:
            shared_data.latest_frame = frame
            shared_data.latest_timestamp = current_time
            res = shared_data.latest_result
            fps_inf = shared_data.processing_fps
            
        display_frame = frame.copy()
        draw_gui(display_frame, res, fps_inf, cam_fps)
        
        cv2.imshow("ASTRA Live", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            with shared_data.lock:
                shared_data.reset_requested = True
        elif key == ord('s'):
            filename = f"astra_screenshot_{int(time.time())}.png"
            cv2.imwrite(filename, display_frame)
            print(f"Screenshot saved to {filename}")

    print("Shutting down...")
    with shared_data.lock:
        shared_data.is_running = False
        
    worker.join(timeout=5.0)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
