import os
import sys
import time
import numpy as np
import cv2

PROJECT_ROOT = r"c:\Users\asus\Desktop\SIH"
sys.path.insert(0, PROJECT_ROOT)

from ml.experiment.schema.loader import load_experiment
from ml.pipeline.inference import ASTRAPipeline

def draw_gui(frame, mode_title, result):
    # professional UI style
    # Draw top banner
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (40, 40, 40), -1)
    cv2.putText(frame, mode_title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Extract data
    sm = result.get("state_machine", {})
    raw_action = result.get("raw_action", "?")
    val_action = result.get("action", "?")
    step_id = sm.get('current_step_id', '?')
    pose_conf = result.get("pose_confidence", 0.0)
    
    anomalies = sm.get("anomalies", [])
    anomaly_text = anomalies[-1]["type"] if anomalies else "NONE"
    
    t_act = result.get("temporal_action") or "?"
    t_anom = result.get("anomaly_score", 0.0)
    
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

class MockStep:
    def __init__(self, step_id, required, allowed_next):
        self.id = step_id
        self.required_actions = required
        self.allowed_next_steps = allowed_next
        self.allowed_previous_steps = ["START"] if step_id == "step_01" else []
        self.name = step_id
        self.timeout_seconds = 100
        self.object_state_effects = {}
        self.object_state_preconditions = {}
        class CC:
            min_temporal_frames = 5
            min_evidence_score = 0.5
            required_object_states = {}
        self.completion_conditions = CC()

def run_replay(mode_title, sequence, out_filename):
    print(f"\n--- RUNNING {mode_title} ---")
    exp_dir = os.path.join(PROJECT_ROOT, "experiments/sample_experiment")
    out_dir = os.path.join(PROJECT_ROOT, "eval_results/real_e2e_demo")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_filename)
    
    config = load_experiment(exp_dir)
    config.steps = [
        MockStep("step_01", ["idle"], ["step_02"]),
        MockStep("step_02", ["reach"], ["step_03"]),
        MockStep("step_03", ["pick"], ["step_04"]),
        MockStep("step_04", ["move"], ["step_05"]),
        MockStep("step_05", ["place"], ["step_06"]),
        MockStep("step_06", ["release"], ["step_07"]),
        MockStep("step_07", ["idle"], ["COMPLETE"])
    ]
    config.step_map = {s.id: s for s in config.steps}
    
    pipeline = ASTRAPipeline(config, device="cpu", use_mock=True, mock_pose=True)
    pipeline.load_models(
        temporal_weights=os.path.join(PROJECT_ROOT, "checkpoints/action/TemporalGRU-ASTRA-v1/best.pt")
    )
    
    # Create a blank frame for GUI
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(out_path, fourcc, 15, (640, 480))
    
    class MockActionResult:
        def __init__(self, action_id):
            self.action_id = action_id
            self.confidence = 0.95
            self.all_scores = {action_id: 0.95}

    # We set mock pose confidence high to pass threshold unless it's a specific test
    pipeline.pose_estimator.predict = lambda *args: [
        type("MockPose", (), {
            "keypoints_2d": np.ones((17, 3), dtype=np.float32), 
            "tracking_confidence": 0.9, 
            "person_id": 1,
            "bbox": [0,0,10,10]
        })()
    ]
    
    # Prime the pipeline so skeleton buffer fills up naturally
    pipeline.action_recognizer.predict = lambda x, a="idle": MockActionResult(a)
    for _ in range(16):
        pipeline.process_frame(frame, timestamp=0.0)

    # Iterate sequence
    for i, raw_act in enumerate(sequence):
        # We need to bypass actual perception models to feed our raw_act directly.
        # But we still want ActionSmoother -> ActionValidator -> FeatureAgg -> TemporalGRU
        
        # We will mock the ST-GCN predict
        pipeline.action_recognizer.predict = lambda x, a=raw_act: MockActionResult(a)
        
        # In mock mode, process_frame uses mock models. The mock ST-GCN might not use our injected action.
        # Let's see how process_frame works. It uses pipeline.action_recognizer.predict(skeleton_seq).
        # We patched it above.
        
        # Process multiple frames per sequence item to allow smoother persistence (15 frames)
        for _ in range(15):
            # Create fresh frame canvas
            display_frame = frame.copy()
            
            result = pipeline.process_frame(display_frame, timestamp=pipeline.frame_count/15.0)
            
            # Since we bypass real detection, we ensure raw_action is set in result 
            # (if it's not set by process_frame correctly due to mock differences)
            
            draw_gui(display_frame, mode_title, result)
            out_video.write(display_frame)
            
            if pipeline.frame_count % 15 == 0:
                print(f"Frame {pipeline.frame_count:03d} | Raw: {raw_act} | Validated: {result.get('action')} | Step: {result.get('state_machine', {}).get('current_step_id')}")

    out_video.release()
    pipeline.close()
    print(f"Saved {mode_title} to {out_path}")

if __name__ == "__main__":
    normal_sequence = ["idle", "reach", "pick", "move", "place", "release", "idle"]
    run_replay("REPLAY / DEMO MODE - Normal Sequence", normal_sequence, "demo_normal.mp4")
    
    anomaly_sequence = ["idle", "reach", "release", "release", "place", "idle"]
    run_replay("REPLAY / DEMO MODE - Anomaly Sequence", anomaly_sequence, "demo_anomaly.mp4")
