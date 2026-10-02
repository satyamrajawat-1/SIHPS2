import re

file_path = "scratch/astra_web/index.html"
with open(file_path, "r", encoding="utf-8") as f:
    html = f.read()

# 1. Fix WebSocket message type ('result' instead of 'inference')
html = html.replace("else if (data.type === 'inference')", "else if (data.type === 'result' || data.type === 'inference')")

# 2. Fix 'd.detections' -> 'd.detection_boxes'
html = html.replace("if (d.detections && Array.isArray(d.detections)) {", "if (d.detection_boxes && Array.isArray(d.detection_boxes)) {")
html = html.replace("d.detections.forEach(det => {", "d.detection_boxes.forEach(det => {")

# 3. Fix 'd.poses' -> 'd.pose_keypoints_raw'
html = html.replace("if (d.poses && Array.isArray(d.poses)) {", "if (d.pose_keypoints_raw && Array.isArray(d.pose_keypoints_raw)) {")
html = html.replace("d.poses.forEach(pose => {", "d.pose_keypoints_raw.forEach(pose => {")

# 4. In astra_server.py: "detections_count" and "poses_count" is used
# I already have:
# safeSetText('detCount', d.detections_count || '0');
# safeSetText('poseCount', d.poses_count || '0');
# This is correct.

# Wait, in the overlay drawing logic, there's another place:
# drawOverlays(d) takes the width and height of the OVERLAY canvas which matches the VIDEO,
# but the backend bbox/keypoints are normalized [0,1]?
# Let's check: in STGCN we normalize coords. In YOLO/RTMPose, `d.bbox` and `pose.keypoints_2d` might NOT be normalized!
# If they are NOT normalized, multiplying by `sx` and `sy` will make them huge!
# I need to check if inference.py normalizes them.
# I'll just write this script for now.

with open(file_path, "w", encoding="utf-8") as f:
    f.write(html)
print("index.html fixed.")
