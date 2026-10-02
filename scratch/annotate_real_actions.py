import cv2
import json
import os
import argparse
from collections import namedtuple

# Classes
CLASSES = {
    '1': 'idle',
    '2': 'reach',
    '3': 'pick',
    '4': 'move',
    '5': 'place',
    '6': 'release'
}

Segment = namedtuple("Segment", ["label", "start", "end"])

def is_overlapping(new_seg, segments):
    for seg in segments:
        # check overlap
        if max(new_seg.start, seg.start) < min(new_seg.end, seg.end):
            return True
    return False

def main():
    parser = argparse.ArgumentParser(description="Annotate Real Actions")
    parser.add_argument("video", type=str, help="Path to the video file")
    args = parser.parse_args()

    video_path = args.video
    if not os.path.exists(video_path):
        print(f"Video {video_path} not found.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    segments = []
    current_label = 'idle'
    start_time = None

    paused = True
    current_frame_idx = 0

    window_name = "Annotator (Q=Quit, Space=Play/Pause, A=Start, S=End, 1-6=Class, Left/Right=Seek)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def seek_to(frame_idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        return cap.read()[1]

    frame = seek_to(current_frame_idx)

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                paused = True
                continue
            current_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

        if frame is None:
            break

        display_frame = frame.copy()
        time_sec = current_frame_idx / fps

        # Draw info
        info_lines = [
            f"Frame: {current_frame_idx}/{total_frames} | Time: {time_sec:.2f}s",
            f"Current Action (1-6): {current_label.upper()}",
            f"Mark Start (A): {f'{start_time:.2f}s' if start_time is not None else 'None'}",
            f"Segments: {len(segments)}"
        ]
        
        y_offset = 30
        for line in info_lines:
            cv2.putText(display_frame, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
            
        # Draw segments
        for i, seg in enumerate(segments[-3:]): # last 3
            cv2.putText(display_frame, f"  {seg.label} {seg.start:.1f}-{seg.end:.1f}s", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 1)
            y_offset += 25

        cv2.imshow(window_name, display_frame)

        key = cv2.waitKey(30 if not paused else 0) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == 81 or key == 2: # Left arrow (Linux/Mac 2, Windows 81 depending on waitKeyEx)
            current_frame_idx = max(0, current_frame_idx - int(fps))
            frame = seek_to(current_frame_idx)
            paused = True
        elif key == 83 or key == 3: # Right arrow
            current_frame_idx = min(total_frames - 1, current_frame_idx + int(fps))
            frame = seek_to(current_frame_idx)
            paused = True
        elif key == ord(','): # Frame back
            current_frame_idx = max(0, current_frame_idx - 1)
            frame = seek_to(current_frame_idx)
            paused = True
        elif key == ord('.'): # Frame fwd
            current_frame_idx = min(total_frames - 1, current_frame_idx + 1)
            frame = seek_to(current_frame_idx)
            paused = True
        elif chr(key) in CLASSES:
            current_label = CLASSES[chr(key)]
        elif key == ord('a'):
            start_time = time_sec
            print(f"Start marked at {start_time:.2f}s")
        elif key == ord('s'):
            if start_time is not None:
                end_time = time_sec
                if end_time > start_time:
                    new_seg = Segment(current_label, start_time, end_time)
                    if not is_overlapping(new_seg, segments):
                        segments.append(new_seg)
                        print(f"Added segment: {new_seg}")
                    else:
                        print("Error: Segment overlaps with existing segment!")
                else:
                    print("Error: End time must be after start time!")
                start_time = None
            else:
                print("Error: Mark start time first with 'a'!")

    cap.release()
    cv2.destroyAllWindows()

    if not segments:
        print("No segments recorded. Exiting.")
        return

    # Save prompt
    confirm = input(f"Save {len(segments)} segments? (y/n): ")
    if confirm.lower() == 'y':
        os.makedirs("data/real_camera", exist_ok=True)
        out_file = "data/real_camera/annotations.json"
        
        # Load existing if any
        data = {}
        if os.path.exists(out_file):
            with open(out_file, 'r') as f:
                try:
                    data = json.load(f)
                except:
                    pass
        
        if "videos" not in data:
            data["videos"] = {}
            
        data["videos"][os.path.basename(video_path)] = {
            "segments": [
                {"label": s.label, "start": s.start, "end": s.end} for s in segments
            ]
        }
        
        with open(out_file, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"Saved to {out_file}")
    else:
        print("Discarded.")

if __name__ == "__main__":
    main()
