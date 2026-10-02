import os
import time
import cv2
import json
import uuid
import argparse

ACTIONS = ["idle", "reach", "pick", "move", "place", "release"]
REPETITIONS_TARGET = 20
RECORD_SECONDS = 6

def main():
    print("========================================")
    print(" ASTRA Real-Camera Action Data Collector")
    print("========================================")

    session_id = uuid.uuid4().hex[:8]
    print(f"Session ID: {session_id}")
    
    out_dir = os.path.join("data", "real_camera", "clips")
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    print(f"\nWebcam started: {fw}x{fh} @ {fps}fps")

    for action in ACTIONS:
        action_dir = os.path.join(out_dir, action)
        os.makedirs(action_dir, exist_ok=True)
        
        print(f"\n==============================================")
        print(f" NEXT ACTION TO RECORD: {action.upper()}")
        print(f"==============================================")
        
        reps_done = 0
        while reps_done < REPETITIONS_TARGET:
            input(f"\nPress ENTER to start recording repetition {reps_done + 1}/{REPETITIONS_TARGET} for '{action}' (or Ctrl+C to quit)...")
            
            for i in range(3, 0, -1):
                print(f"Starting in {i}...")
                time.sleep(1)
                
            print(">>> RECORDING NOW! <<<")
            
            clip_id = f"{action}_{uuid.uuid4().hex[:8]}"
            vid_path = os.path.join(action_dir, f"{clip_id}.mp4")
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(vid_path, fourcc, fps, (fw, fh))
            
            start_time = time.time()
            frames_recorded = 0
            
            while time.time() - start_time < RECORD_SECONDS:
                ret, frame = cap.read()
                if not ret:
                    continue
                writer.write(frame)
                frames_recorded += 1
                
                disp = frame.copy()
                cv2.putText(disp, f"RECORDING: {action.upper()}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                cv2.imshow("Collector", disp)
                cv2.waitKey(1)
                
            writer.release()
            duration = time.time() - start_time
            print("--- Recording stopped ---")
            
            # Save metadata
            meta_path = os.path.join(action_dir, f"{clip_id}.json")
            with open(meta_path, "w") as jf:
                json.dump({
                    "label": action,
                    "clip_id": clip_id,
                    "session_id": session_id,
                    "fps": fps,
                    "resolution": [fw, fh],
                    "duration": duration,
                    "frames": frames_recorded
                }, jf)
                
            print(f"Saved clip {clip_id}.mp4 and metadata.")
            reps_done += 1

    cap.release()
    cv2.destroyAllWindows()
    print("\nCollection Complete!")
    print(f"Please upload the {out_dir} folder to your Google Drive under 'real_camera_astra/clips/'")

if __name__ == "__main__":
    main()
