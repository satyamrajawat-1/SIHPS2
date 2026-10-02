"""
ASTRA — Lightweight Local Annotation Tool

A simple local annotation interface for experiment video data.
Runs as a local web server with a minimal UI.

Capabilities:
  1. Play/seek video frames
  2. Create action segments
  3. Label experiment steps
  4. Mark object states
  5. Mark anomalies
  6. Associate object IDs
  7. Export JSON/JSONL

Usage:
    python -m ml.data.annotate --session data/raw/session_001/ --port 8501
"""
from __future__ import annotations

import argparse
import http.server
import json
import logging
import os
import sys
import urllib.parse
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_annotation_html(session_dir: str) -> str:
    """Generate the annotation tool HTML."""
    manifest_path = os.path.join(session_dir, "manifest.json")
    frames = []
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        frames = manifest.get("frames", [])

    # Load existing annotations
    ann_dir = os.path.join(session_dir, "annotations")
    existing_actions = []
    existing_steps = []
    existing_states = []
    existing_anomalies = []

    for name, target in [("actions.jsonl", existing_actions), ("steps.jsonl", existing_steps),
                          ("object_states.jsonl", existing_states), ("anomalies.jsonl", existing_anomalies)]:
        path = os.path.join(ann_dir, name)
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        target.append(json.loads(line))

    frames_json = json.dumps(frames)
    actions_json = json.dumps(existing_actions)
    steps_json = json.dumps(existing_steps)
    states_json = json.dumps(existing_states)
    anomalies_json = json.dumps(existing_anomalies)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ASTRA Annotator</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: system-ui, sans-serif; background: #1a1a2e; color: #e0e0e0; padding: 16px; }}
h1 {{ color: #00d4ff; margin-bottom: 12px; font-size: 20px; }}
.container {{ display: flex; gap: 16px; }}
.viewer {{ flex: 2; }}
.panel {{ flex: 1; max-height: 80vh; overflow-y: auto; }}
img#frame {{ width: 100%; border: 2px solid #333; border-radius: 4px; background: #000; }}
.controls {{ margin: 8px 0; display: flex; gap: 8px; align-items: center; }}
button {{ padding: 6px 12px; border: none; border-radius: 4px; cursor: pointer; font-size: 13px; }}
.btn-primary {{ background: #00d4ff; color: #000; }}
.btn-danger {{ background: #ff4444; color: #fff; }}
.btn-success {{ background: #00cc66; color: #000; }}
.btn-sm {{ padding: 4px 8px; font-size: 12px; }}
input, select {{ padding: 4px 8px; border: 1px solid #444; border-radius: 4px; background: #2a2a3e; color: #e0e0e0; font-size: 13px; }}
input[type=range] {{ flex: 1; }}
.section {{ background: #2a2a3e; padding: 12px; border-radius: 6px; margin-bottom: 8px; }}
.section h3 {{ color: #00d4ff; margin-bottom: 8px; font-size: 14px; }}
.ann-item {{ background: #333; padding: 6px 8px; border-radius: 3px; margin: 4px 0; font-size: 12px; display: flex; justify-content: space-between; }}
.info {{ color: #888; font-size: 12px; }}
#timeline {{ width: 100%; height: 40px; background: #222; border-radius: 4px; margin: 8px 0; position: relative; cursor: pointer; }}
.timeline-segment {{ position: absolute; height: 100%; border-radius: 2px; opacity: 0.7; }}
</style>
</head>
<body>
<h1>ASTRA Annotation Tool</h1>
<div class="container">
  <div class="viewer">
    <img id="frame" src="" alt="Frame">
    <div class="controls">
      <button class="btn-sm" onclick="prevFrame()">&lt;</button>
      <input type="range" id="slider" min="0" max="0" value="0" oninput="seekFrame(this.value)">
      <button class="btn-sm" onclick="nextFrame()">&gt;</button>
      <span id="frameInfo" class="info">0/0 | 0.00s</span>
    </div>
    <div id="timeline"></div>
    <div class="section">
      <h3>Add Annotation</h3>
      <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px">
        <select id="annType">
          <option value="action">Action</option>
          <option value="step">Experiment Step</option>
          <option value="state">Object State</option>
          <option value="anomaly">Anomaly</option>
        </select>
        <input id="annLabel" placeholder="Label/ID" style="width:120px">
        <input id="annObject" placeholder="Object ID" style="width:100px">
        <button class="btn-primary" onclick="markStart()">Mark Start</button>
        <button class="btn-success" onclick="markEnd()">Mark End</button>
      </div>
      <div class="info" id="markInfo">Click 'Mark Start' at the beginning of a segment</div>
    </div>
  </div>
  <div class="panel">
    <div class="section">
      <h3>Actions (<span id="actionCount">0</span>)</h3>
      <div id="actionList"></div>
    </div>
    <div class="section">
      <h3>Steps (<span id="stepCount">0</span>)</h3>
      <div id="stepList"></div>
    </div>
    <div class="section">
      <h3>Object States (<span id="stateCount">0</span>)</h3>
      <div id="stateList"></div>
    </div>
    <div class="section">
      <h3>Anomalies (<span id="anomalyCount">0</span>)</h3>
      <div id="anomalyList"></div>
    </div>
    <div style="margin-top:12px">
      <button class="btn-primary" onclick="saveAnnotations()">Save Annotations</button>
      <button class="btn-danger" onclick="exportJSON()">Export JSON</button>
    </div>
  </div>
</div>
<script>
const frames = {frames_json};
let actions = {actions_json};
let steps = {steps_json};
let states = {states_json};
let anomalies = {anomalies_json};
let currentFrame = 0;
let markStartFrame = null;

const slider = document.getElementById('slider');
slider.max = frames.length - 1;

function loadFrame(idx) {{
  if (idx < 0 || idx >= frames.length) return;
  currentFrame = idx;
  const f = frames[idx];
  document.getElementById('frame').src = '/frame/' + f.filename;
  document.getElementById('frameInfo').textContent =
    `${{idx+1}}/${{frames.length}} | ${{f.timestamp.toFixed(2)}}s | frame ${{f.source_frame}}`;
  slider.value = idx;
}}

function seekFrame(v) {{ loadFrame(parseInt(v)); }}
function prevFrame() {{ loadFrame(currentFrame - 1); }}
function nextFrame() {{ loadFrame(currentFrame + 1); }}

function markStart() {{
  markStartFrame = currentFrame;
  document.getElementById('markInfo').textContent = `Start marked at frame ${{currentFrame}} (${{frames[currentFrame]?.timestamp.toFixed(2)}}s)`;
}}

function markEnd() {{
  if (markStartFrame === null) {{ alert('Mark start first'); return; }}
  const type = document.getElementById('annType').value;
  const label = document.getElementById('annLabel').value || 'unlabeled';
  const objId = document.getElementById('annObject').value;
  const startT = frames[markStartFrame]?.timestamp || 0;
  const endT = frames[currentFrame]?.timestamp || 0;

  const ann = {{
    start_frame: markStartFrame, end_frame: currentFrame,
    start_time: startT, end_time: endT,
  }};

  if (type === 'action') {{ ann.action_id = label; actions.push(ann); }}
  else if (type === 'step') {{ ann.step_id = label; ann.status = 'completed'; steps.push(ann); }}
  else if (type === 'state') {{ ann.object_id = objId; ann.state = label; states.push(ann); }}
  else if (type === 'anomaly') {{ ann.type = label; ann.observed_step = objId; anomalies.push(ann); }}

  markStartFrame = null;
  document.getElementById('markInfo').textContent = 'Annotation added';
  document.getElementById('annLabel').value = '';
  document.getElementById('annObject').value = '';
  renderAnnotations();
}}

function renderAnnotations() {{
  const render = (list, id, countId, labelFn) => {{
    document.getElementById(countId).textContent = list.length;
    document.getElementById(id).innerHTML = list.map((a, i) =>
      `<div class="ann-item"><span>${{labelFn(a)}} [${{a.start_frame}}-${{a.end_frame}}]</span>` +
      `<button class="btn-sm btn-danger" onclick="${{id}}Del(${{i}})">x</button></div>`
    ).join('');
  }};
  render(actions, 'actionList', 'actionCount', a => a.action_id || '?');
  render(steps, 'stepList', 'stepCount', a => a.step_id || '?');
  render(states, 'stateList', 'stateCount', a => `${{a.object_id}}: ${{a.state}}`);
  render(anomalies, 'anomalyList', 'anomalyCount', a => a.type || '?');
}}

function actionListDel(i) {{ actions.splice(i,1); renderAnnotations(); }}
function stepListDel(i) {{ steps.splice(i,1); renderAnnotations(); }}
function stateListDel(i) {{ states.splice(i,1); renderAnnotations(); }}
function anomalyListDel(i) {{ anomalies.splice(i,1); renderAnnotations(); }}

function saveAnnotations() {{
  fetch('/save', {{
    method: 'POST', headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{ actions, steps, states, anomalies }})
  }}).then(r => r.json()).then(d => alert(d.message || 'Saved'));
}}

function exportJSON() {{
  const data = {{ actions, steps, states, anomalies }};
  const blob = new Blob([JSON.stringify(data, null, 2)], {{type: 'application/json'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'annotations.json';
  a.click();
}}

document.addEventListener('keydown', e => {{
  if (e.key === 'ArrowLeft') prevFrame();
  else if (e.key === 'ArrowRight') nextFrame();
  else if (e.key === 's') markStart();
  else if (e.key === 'e') markEnd();
}});

if (frames.length > 0) loadFrame(0);
renderAnnotations();
</script>
</body>
</html>"""


class AnnotationHandler(http.server.BaseHTTPRequestHandler):
    session_dir = ""

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            html = generate_annotation_html(self.session_dir)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))
        elif self.path.startswith("/frame/"):
            fname = self.path[7:]
            fpath = os.path.join(self.session_dir, "frames", urllib.parse.unquote(fname))
            if os.path.exists(fpath):
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.end_headers()
                with open(fpath, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/save":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            ann_dir = os.path.join(self.session_dir, "annotations")
            os.makedirs(ann_dir, exist_ok=True)

            for name, data in body.items():
                path = os.path.join(ann_dir, f"{name}.jsonl")
                with open(path, "w") as f:
                    for item in data:
                        f.write(json.dumps(item) + "\n")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"message": "Saved", "files": list(body.keys())}).encode())
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass  # Suppress default logging


def main():
    parser = argparse.ArgumentParser(description="ASTRA Annotation Tool")
    parser.add_argument("--session", required=True, help="Session directory")
    parser.add_argument("--port", type=int, default=8501, help="Server port")
    args = parser.parse_args()

    AnnotationHandler.session_dir = os.path.abspath(args.session)
    server = http.server.HTTPServer(("localhost", args.port), AnnotationHandler)
    print(f"\nASTRA Annotator running at: http://localhost:{args.port}")
    print(f"Session: {args.session}")
    print("Press Ctrl+C to stop.\n")
    print("Keyboard: Left/Right = navigate | S = mark start | E = mark end\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nAnnotator stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
