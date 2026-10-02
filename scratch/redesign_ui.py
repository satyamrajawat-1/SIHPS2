import re

file_path = "scratch/astra_web/index.html"

with open(file_path, "r", encoding="utf-8") as f:
    html = f.read()

# 1. Update .main grid
html = re.sub(
    r"\.main\s*\{[^}]*grid-template-columns:[^}]*\}",
    ".main {\n    display: grid;\n    grid-template-columns: 1fr 1fr;\n    gap: 16px;\n    padding: 16px 24px;\n    height: calc(100vh - 64px - 180px);\n    min-height: 420px;\n  }",
    html
)

# 2. Update HTML structure for .main
new_main = """<div class="main">
  <!-- Person Panel (Who is present?) -->
  <div class="camera-panel" id="personPanel">
    <div style="position:absolute; top:16px; left:16px; z-index:10; background:rgba(0,0,0,0.6); padding:6px 12px; border-radius:6px; display:flex; align-items:center; gap:8px; font-size:12px; font-weight:700; letter-spacing:1px; border:1px solid var(--border-light); backdrop-filter:blur(4px);">
      <div style="width:8px; height:8px; border-radius:50%; background:var(--cyan);"></div> WHO IS PRESENT?
    </div>
    <video id="video" autoplay playsinline muted style="display:none;"></video>
    <canvas id="personCanvas"></canvas>
    <div class="camera-placeholder" id="cameraPlaceholder">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9A2.25 2.25 0 0013.5 5.25h-9A2.25 2.25 0 002.25 7.5v9A2.25 2.25 0 004.5 18.75z"/></svg>
      <div style="font-size:16px; font-weight:600">Click "Start Camera" to begin</div>
    </div>
  </div>

  <!-- Action Panel (What is being done?) -->
  <div class="camera-panel" id="actionPanel">
    <div style="position:absolute; top:16px; left:16px; z-index:10; background:rgba(0,0,0,0.6); padding:6px 12px; border-radius:6px; display:flex; align-items:center; gap:8px; font-size:12px; font-weight:700; letter-spacing:1px; border:1px solid var(--border-light); backdrop-filter:blur(4px);">
      <div style="width:8px; height:8px; border-radius:50%; background:var(--purple);"></div> WHAT IS BEING DONE?
    </div>
    <canvas id="actionCanvas"></canvas>
    
    <div id="actionOverlay" style="position:absolute; bottom:24px; left:24px; z-index:10; background:rgba(17,24,39,0.85); border:1px solid var(--border); border-radius:var(--radius); padding:20px; min-width:240px; backdrop-filter:blur(8px); display:none;">
       <div id="rawActionBig" class="action-big idle" style="font-size:36px; padding:12px; margin-bottom:4px; text-transform:uppercase;">—</div>
       <div id="rawConfidence" style="font-size:13px; color:var(--text-dim); text-align:center; margin-bottom:16px; font-family:'JetBrains Mono', monospace;">0.00 confidence</div>
       
       <div style="display:flex; justify-content:space-between; align-items:center; padding:6px 0; border-top:1px solid var(--border-light);">
         <span style="font-size:12px; color:var(--text-dim); font-weight:500;">Expected:</span>
         <span id="overlayExpected" style="font-family:'JetBrains Mono', monospace; font-size:13px; font-weight:600; color:var(--text);">—</span>
       </div>
       <div style="display:flex; justify-content:space-between; align-items:center; padding:6px 0;">
         <span style="font-size:12px; color:var(--text-dim); font-weight:500;">Validated:</span>
         <span id="overlayValidated" style="font-family:'JetBrains Mono', monospace; font-size:12px; font-weight:700; padding:4px 8px; border-radius:4px; background:var(--bg-card); border:1px solid var(--border);">—</span>
       </div>
    </div>
  </div>
</div>

<div style="display:none;" id="oldAnalysisPanel">"""

# Replace from `<div class="main">` up to the bottom-bar start
html = re.sub(
    r'<div class="main">.*?(?=<!-- Bottom Bar -->)',
    new_main,
    html,
    flags=re.DOTALL
)

html = html.replace('<!-- Bottom Bar -->', '</div>\n<!-- Bottom Bar -->')

# 3. JS Updates for variables
js_vars_repl = """let videoEl = document.getElementById('video');
let personCanvas = document.getElementById('personCanvas');
let personCtx = personCanvas.getContext('2d');
let actionCanvas = document.getElementById('actionCanvas');
let actionCtx = actionCanvas.getContext('2d');"""

html = re.sub(
    r"let videoEl = document.getElementById\('video'\);\s*let overlayCanvas = document.getElementById\('overlay'\);\s*let overlayCtx = overlayCanvas.getContext\('2d'\);",
    js_vars_repl,
    html
)

# 4. Canvas sizing in JS
js_size_repl = """personCanvas.width = videoEl.videoWidth;
    personCanvas.height = videoEl.videoHeight;
    actionCanvas.width = videoEl.videoWidth;
    actionCanvas.height = videoEl.videoHeight;
    
    // Draw background video to both
    personCtx.drawImage(videoEl, 0, 0);
    actionCtx.drawImage(videoEl, 0, 0);"""

html = re.sub(
    r"overlayCanvas\.width = videoEl\.videoWidth;\s*overlayCanvas\.height = videoEl\.videoHeight;\s*overlayCtx\.drawImage\(videoEl, 0, 0\);",
    js_size_repl,
    html
)

# 5. Clear UI
html = html.replace(
    "overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);",
    "personCtx.clearRect(0, 0, personCanvas.width, personCanvas.height);\n  actionCtx.clearRect(0, 0, actionCanvas.width, actionCanvas.height);\n  document.getElementById('actionOverlay').style.display = 'none';"
)

html = html.replace(
    "document.getElementById('cameraPlaceholder').style.display = 'none';",
    "document.getElementById('cameraPlaceholder').style.display = 'none';\n  document.getElementById('actionOverlay').style.display = 'block';"
)

# 6. handleResult extraction and UI update
ui_update_code = """
  // Update Large Overlay Badges
  const rawBig = document.getElementById('rawActionBig');
  const validBig = document.getElementById('overlayValidated');
  const expectedText = document.getElementById('overlayExpected');
  const confText = document.getElementById('rawConfidence');
  
  if (d.raw_action) {
    rawBig.textContent = d.raw_action;
    confText.textContent = (d.confidence || 0).toFixed(2) + " confidence";
    
    // Color code raw action based on validation
    if (d.state_machine && d.state_machine.validated_action) {
       const v = d.state_machine.validated_action.toLowerCase();
       if (v === 'invalid' || v === 'out_of_sequence') {
           rawBig.className = 'action-big invalid';
           validBig.textContent = 'INVALID';
           validBig.style.color = 'var(--red)';
           validBig.style.borderColor = 'var(--red)';
       } else if (v === 'uncertain') {
           rawBig.className = 'action-big idle';
           validBig.textContent = 'UNCERTAIN';
           validBig.style.color = 'var(--yellow)';
           validBig.style.borderColor = 'var(--yellow)';
       } else {
           rawBig.className = 'action-big valid';
           validBig.textContent = 'VALID';
           validBig.style.color = 'var(--green)';
           validBig.style.borderColor = 'var(--green)';
       }
    }
  }
  
  if (d.state_machine && d.state_machine.next_expected) {
     expectedText.textContent = d.state_machine.next_expected;
  }
"""

html = html.replace(
    "// Perception\n  if (d.raw_action) {",
    ui_update_code + "\n  // Perception\n  if (d.raw_action) {"
)

# 7. Drawing logic: substitute overlayCtx with appropriate ctx
draw_logic = """
  const sx = personCanvas.width;
  const sy = personCanvas.height;

  // Person Panel: Draw Bounding Boxes
  if (d.detections && Array.isArray(d.detections)) {
    d.detections.forEach(det => {
      if (det.class_name.toLowerCase() !== 'person') return;
      
      const [x1, y1, x2, y2] = det.bbox;
      const w = x2 - x1;
      const h = y2 - y1;

      personCtx.strokeStyle = 'var(--green)';
      personCtx.lineWidth = 2;
      personCtx.strokeRect(x1 * sx, y1 * sy, w * sx, h * sy);

      personCtx.fillStyle = 'rgba(0,0,0,0.6)';
      personCtx.fillRect(x1 * sx, y1 * sy - 24, 120, 24);
      
      personCtx.fillStyle = '#fff';
      personCtx.font = '12px Inter';
      personCtx.fillText(`PERSON ${(det.confidence||0).toFixed(2)}`, x1 * sx + 6, y1 * sy - 6);
    });
  }

  // Action Panel: Draw Skeletons
  if (d.poses && Array.isArray(d.poses)) {
    d.poses.forEach(pose => {
      const kps = pose.keypoints_2d;
      if (!kps || kps.length < 17) return;

      const pairs = [
        [0,1],[1,3],[0,2],[2,4],[5,7],[7,9],[6,8],[8,10],
        [5,11],[6,12],[11,13],[13,15],[12,14],[14,16],[5,6],[11,12]
      ];

      actionCtx.strokeStyle = 'rgba(96, 165, 250, 0.7)';
      actionCtx.lineWidth = 3;

      pairs.forEach(([i, j]) => {
        if (kps[i][2] > 0.2 && kps[j][2] > 0.2) {
          actionCtx.beginPath();
          actionCtx.moveTo(kps[i][0] * sx, kps[i][1] * sy);
          actionCtx.lineTo(kps[j][0] * sx, kps[j][1] * sy);
          actionCtx.stroke();
        }
      });

      kps.forEach(([x, y, c]) => {
        if (c > 0.2) {
          actionCtx.fillStyle = c > 0.5 ? 'rgba(96, 165, 250, 0.95)' : 'rgba(96, 165, 250, 0.5)';
          actionCtx.beginPath();
          actionCtx.arc(x * sx, y * sy, 4, 0, Math.PI * 2);
          actionCtx.fill();
        }
      });
    });
  }
"""

html = re.sub(
    r"const sx = overlayCanvas\.width;.*?}\s*}\s*}\s*}\s*}\s*}\s*}",
    draw_logic + "\n}",
    html,
    flags=re.DOTALL
)

# 8. Fix the extra curly bracket from regex replacing too much
html = html.replace("\n}\n}\n// ── Anomaly toast ──", "\n// ── Anomaly toast ──")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(html)
print("UI redesign applied successfully.")
