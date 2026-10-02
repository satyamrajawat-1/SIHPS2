import re

html_content = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ASTRA — Live Human Activity Recognition</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0a0e17;
    --bg-card: #111827;
    --bg-card-alt: #151d2e;
    --border: #1f2937;
    --border-light: #374151;
    --text: #f3f4f6;
    --text-dim: #9ca3af;
    --text-muted: #6b7280;
    --accent: #3b82f6;
    --accent-glow: rgba(59, 130, 246, 0.25);
    --green: #22c55e;
    --green-dim: rgba(34, 197, 94, 0.15);
    --red: #ef4444;
    --red-dim: rgba(239, 68, 68, 0.15);
    --yellow: #eab308;
    --cyan: #06b6d4;
    --purple: #a855f7;
    --radius: 12px;
    --radius-sm: 8px;
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    font-family: 'Inter', sans-serif;
    background: var(--bg);
    color: var(--text);
    overflow-x: hidden;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }

  /* ── Top Bar ─────────────────────────────────────── */
  .top-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 24px;
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    border-bottom: 1px solid var(--border);
    position: sticky;
    top: 0;
    z-index: 100;
  }
  .top-bar .logo { display: flex; align-items: center; gap: 12px; }
  .top-bar .logo-icon {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, var(--accent), var(--purple));
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-weight: 900; font-size: 18px; color: white;
  }
  .top-bar h1 {
    font-size: 20px; font-weight: 800; letter-spacing: 2px;
    background: linear-gradient(135deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .top-bar .subtitle { font-size: 11px; color: var(--text-dim); font-weight: 400; letter-spacing: 0.5px; }
  
  .top-bar .controls { display: flex; gap: 10px; align-items: center; }
  .fps-display { font-family: 'JetBrains Mono', monospace; font-size: 12px; color: var(--text-dim); padding: 6px 12px; background: var(--bg-card); border-radius: var(--radius-sm); border: 1px solid var(--border); }
  .fps-display span { color: var(--green); font-weight: 600; }

  .btn { font-family: 'Inter', sans-serif; font-size: 13px; font-weight: 600; padding: 8px 16px; border: 1px solid var(--border-light); border-radius: var(--radius-sm); background: var(--bg-card); color: var(--text); cursor: pointer; transition: all 0.2s; }
  .btn:hover { background: var(--bg-card-alt); border-color: var(--accent); }
  .btn-primary { background: var(--accent); border-color: var(--accent); color: white; }
  .btn-primary:hover { background: #2563eb; }

  /* ── Main Layout ─────────────────────────────────── */
  .main {
    flex: 1;
    display: flex;
    flex-direction: column;
    padding: 16px 24px;
    gap: 16px;
  }

  /* ── Single Camera View ──────────────────────────────── */
  .camera-container {
    position: relative;
    width: 100%;
    max-width: 1400px;
    margin: 0 auto;
    background: #000;
    border-radius: var(--radius);
    overflow: hidden;
    border: 1px solid var(--border);
    display: flex;
    justify-content: center;
    align-items: center;
    aspect-ratio: 16/9;
  }
  
  .camera-container video {
    position: absolute;
    top: 0; left: 0; width: 100%; height: 100%;
    object-fit: contain;
    z-index: 1;
  }
  
  .camera-container canvas {
    position: absolute;
    top: 0; left: 0; width: 100%; height: 100%;
    z-index: 2;
    pointer-events: none;
  }

  .live-badge {
    position: absolute;
    top: 16px; left: 16px;
    display: flex; align-items: center; gap: 8px;
    padding: 6px 14px;
    background: rgba(0,0,0,0.7); backdrop-filter: blur(8px);
    border-radius: 20px; font-size: 12px; font-weight: 600;
    z-index: 10;
  }
  .live-dot {
    width: 8px; height: 8px; border-radius: 50%; background: var(--red);
    animation: pulse-dot 1.5s infinite;
  }
  @keyframes pulse-dot {
    0%, 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.6); }
    50% { box-shadow: 0 0 0 6px rgba(239, 68, 68, 0); }
  }

  .camera-placeholder {
    position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; color: var(--text-dim); gap: 16px; z-index: 3;
  }
  .camera-placeholder svg { width: 64px; height: 64px; opacity: 0.4; }

  /* ── Compact Info Bar ────────────────────────────── */
  .info-bar {
    display: flex;
    gap: 16px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 12px 20px;
    align-items: center;
    justify-content: space-between;
    max-width: 1400px;
    margin: 0 auto;
    width: 100%;
  }
  .info-item {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .info-item .label { font-size: 10px; text-transform: uppercase; color: var(--text-muted); font-weight: 700; letter-spacing: 1px; }
  .info-item .value { font-family: 'JetBrains Mono', monospace; font-size: 14px; font-weight: 600; color: var(--text); }
  .info-item.anomaly .value { color: var(--red); }

  /* ── Timeline ────────────────────────────────────── */
  .timeline-container {
    max-width: 1400px;
    margin: 0 auto;
    width: 100%;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px 24px;
  }
  .timeline {
    display: flex;
    align-items: center;
    width: 100%;
  }
  .timeline-step {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    position: relative;
  }
  .timeline-step .step-circle {
    width: 32px; height: 32px;
    border-radius: 50%;
    border: 2px solid var(--border-light);
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; font-weight: 700; background: var(--bg); z-index: 2;
    transition: all 0.4s;
  }
  .timeline-step .step-label {
    font-size: 10px; font-weight: 600; color: var(--text-muted); margin-top: 6px; text-transform: uppercase; letter-spacing: 0.5px; transition: all 0.3s;
  }
  .timeline-step.completed .step-circle { background: var(--green); border-color: var(--green); color: white; }
  .timeline-step.completed .step-label { color: var(--green); }
  .timeline-step.current .step-circle { background: var(--accent); border-color: var(--accent); color: white; box-shadow: 0 0 12px var(--accent-glow); animation: pulse-step 2s infinite; }
  .timeline-step.current .step-label { color: var(--accent); font-weight: 700; }
  @keyframes pulse-step { 0%, 100% { box-shadow: 0 0 8px var(--accent-glow); } 50% { box-shadow: 0 0 20px var(--accent-glow); } }
  .timeline-connector { flex: 0.6; height: 2px; background: var(--border-light); margin-bottom: 20px; transition: background 0.4s; }
  .timeline-connector.done { background: var(--green); }

</style>
</head>
<body>

<!-- Top Bar -->
<div class="top-bar">
  <div class="logo">
    <div class="logo-icon">A</div>
    <div>
      <h1>ASTRA</h1>
      <div class="subtitle">Onboard Human Activity Recognition</div>
    </div>
  </div>
  <div class="controls">
    <div class="fps-display">Camera <span id="camFps">0</span> FPS &nbsp;|&nbsp; Inference <span id="infFps">0</span> FPS</div>
    <button class="btn btn-primary" id="btnStart" onclick="startCamera()">Start Camera</button>
    <button class="btn" id="btnStop" onclick="stopCamera()" style="display:none">Stop Camera</button>
    <button class="btn" onclick="resetExperiment()">Reset</button>
  </div>
</div>

<div class="main">
  <!-- Single Live Camera -->
  <div class="camera-container">
    <div class="live-badge" id="liveBadge" style="display:none">
      <div class="live-dot"></div>
      <span>LIVE</span>
      <span style="color:var(--text-muted); font-weight:400">Camera 0</span>
    </div>
    
    <video id="video" autoplay playsinline muted style="display:none;"></video>
    <canvas id="overlayCanvas"></canvas>
    
    <div class="camera-placeholder" id="cameraPlaceholder">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9A2.25 2.25 0 0013.5 5.25h-9A2.25 2.25 0 002.25 7.5v9A2.25 2.25 0 004.5 18.75z"/></svg>
      <div style="font-size:16px; font-weight:600">Click "Start Camera" to begin</div>
    </div>
  </div>

  <!-- Info Bar -->
  <div class="info-bar">
    <div class="info-item">
      <div class="label">Current Step</div>
      <div class="value" id="infoStep">STEP_01 / IDLE</div>
    </div>
    <div class="info-item">
      <div class="label">Expected Action</div>
      <div class="value" id="infoExpected">REACH</div>
    </div>
    <div class="info-item">
      <div class="label">Validated Action</div>
      <div class="value" id="infoValidated">—</div>
    </div>
    <div class="info-item anomaly">
      <div class="label">Anomaly Status</div>
      <div class="value" id="infoAnomaly" style="color:var(--text-dim)">NONE</div>
    </div>
  </div>

  <!-- Timeline -->
  <div class="timeline-container">
    <div class="timeline" id="timeline"></div>
  </div>
</div>

<script>
const ACTIONS = ['idle', 'reach', 'pick', 'move', 'place', 'release', 'idle'];
const STEP_TO_ACTION = {
  'step_01': 'idle',
  'step_02': 'reach',
  'step_03': 'pick',
  'step_04': 'move',
  'step_05': 'place',
  'step_06': 'release',
  'step_07': 'idle'
};

var videoEl = document.getElementById('video');
var overlayCanvas = document.getElementById('overlayCanvas');
var overlayCtx = overlayCanvas.getContext('2d');
var sendCanvas = document.createElement('canvas');
var sendCtx = sendCanvas.getContext('2d');

var ws = null;
var mediaStream = null;
var cameraRunning = false;
var sending = false;
var camFrames = 0;
var lastCamTime = performance.now();
var prevStep = null;
var prevAnomaly = null;

// Latest data from backend
var currentMLData = null;

// ── DOM Helpers ──
function safeSetText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

// ── Camera ──
async function startCamera() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
      audio: false
    });
    videoEl.srcObject = mediaStream;
    cameraRunning = true;
    
    document.getElementById('cameraPlaceholder').style.display = 'none';
    document.getElementById('liveBadge').style.display = 'flex';
    document.getElementById('btnStart').style.display = 'none';
    document.getElementById('btnStop').style.display = '';
    
    connectWS();
    requestAnimationFrame(camLoop);
  } catch(e) {
    console.error('Camera error:', e);
  }
}

function stopCamera() {
  cameraRunning = false;
  if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());
  mediaStream = null;
  videoEl.srcObject = null;
  currentMLData = null;
  
  document.getElementById('cameraPlaceholder').style.display = '';
  document.getElementById('liveBadge').style.display = 'none';
  document.getElementById('btnStart').style.display = '';
  document.getElementById('btnStop').style.display = 'none';
  
  if (ws) ws.close();
}

function camLoop() {
  if (!cameraRunning) return;
  camFrames++;
  const now = performance.now();
  if (now - lastCamTime > 1000) {
    safeSetText('camFps', camFrames);
    camFrames = 0;
    lastCamTime = now;
  }

  // Handle Canvas Resize & Drawing
  if (videoEl.videoWidth) {
    overlayCanvas.width = videoEl.videoWidth;
    overlayCanvas.height = videoEl.videoHeight;
    
    // Draw Video Background
    overlayCtx.drawImage(videoEl, 0, 0, overlayCanvas.width, overlayCanvas.height);
    
    // Draw ML Overlays
    if (currentMLData) {
      drawOverlays(currentMLData);
    }
  }

  requestAnimationFrame(camLoop);
}

// ── Drawing Logic ──
const COCO_SKELETON = [
  [0,1],[0,2],[1,3],[2,4],[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]
];

function drawOverlays(d) {
  const sx = overlayCanvas.width;
  const sy = overlayCanvas.height;
  
  // 1. Draw Object/Person Detections
  if (d.detections && Array.isArray(d.detections)) {
    d.detections.forEach(det => {
      const [x1, y1, x2, y2] = det.bbox;
      const w = x2 - x1;
      const h = y2 - y1;
      
      const isPerson = det.class_name.toLowerCase() === 'person';
      const color = isPerson ? 'var(--cyan)' : 'var(--purple)';

      overlayCtx.strokeStyle = color;
      overlayCtx.lineWidth = 2;
      overlayCtx.strokeRect(x1 * sx, y1 * sy, w * sx, h * sy);

      // Label background
      overlayCtx.fillStyle = 'rgba(0,0,0,0.7)';
      const labelText = `${det.class_name.toUpperCase()} ${(det.confidence||0).toFixed(2)}`;
      overlayCtx.font = '600 14px Inter';
      const textWidth = overlayCtx.measureText(labelText).width;
      overlayCtx.fillRect(x1 * sx, y1 * sy - 28, textWidth + 16, 28);
      
      // Label text
      overlayCtx.fillStyle = color;
      overlayCtx.fillText(labelText, x1 * sx + 8, y1 * sy - 8);
    });
  }

  // 2. Draw Skeleton
  if (d.poses && Array.isArray(d.poses)) {
    d.poses.forEach(pose => {
      const kps = pose.keypoints_2d;
      if (!kps || kps.length < 17) return;

      overlayCtx.strokeStyle = 'rgba(255, 255, 255, 0.6)';
      overlayCtx.lineWidth = 3;

      COCO_SKELETON.forEach(([i, j]) => {
        if (kps[i][2] > 0.2 && kps[j][2] > 0.2) {
          overlayCtx.beginPath();
          overlayCtx.moveTo(kps[i][0] * sx, kps[i][1] * sy);
          overlayCtx.lineTo(kps[j][0] * sx, kps[j][1] * sy);
          overlayCtx.stroke();
        }
      });

      kps.forEach(([x, y, c]) => {
        if (c > 0.2) {
          overlayCtx.fillStyle = c > 0.5 ? 'var(--green)' : 'rgba(255,255,255,0.8)';
          overlayCtx.beginPath();
          overlayCtx.arc(x * sx, y * sy, 4, 0, Math.PI * 2);
          overlayCtx.fill();
        }
      });
    });
  }

  // 3. Draw Info Panel (Bottom Left)
  drawInfoPanel(d, sx, sy);
  
  // 4. Draw Procedure Panel (Top Right)
  drawProcedurePanel(d, sx, sy);
  
  // 5. Draw Anomaly Warning (Center)
  drawAnomalyWarning(d, sx, sy);
}

function drawInfoPanel(d, width, height) {
  const panelX = 30;
  const panelY = height - 130;
  
  overlayCtx.fillStyle = 'rgba(17, 24, 39, 0.85)';
  overlayCtx.beginPath();
  overlayCtx.roundRect(panelX, panelY, 300, 100, 8);
  overlayCtx.fill();
  overlayCtx.strokeStyle = 'var(--border)';
  overlayCtx.lineWidth = 1;
  overlayCtx.stroke();

  // Action & Confidence
  const rawAction = (d.raw_action || 'IDLE').toUpperCase();
  const conf = (d.pose_confidence || 0).toFixed(2);
  
  overlayCtx.fillStyle = '#fff';
  overlayCtx.font = '800 24px JetBrains Mono';
  overlayCtx.fillText(`ACTION: ${rawAction}`, panelX + 20, panelY + 40);
  
  overlayCtx.fillStyle = 'var(--text-dim)';
  overlayCtx.font = '500 13px Inter';
  overlayCtx.fillText(`CONFIDENCE: ${conf}`, panelX + 20, panelY + 65);

  // Expected & Validated
  const nextStepId = (d.next_valid_steps && d.next_valid_steps.length > 0) ? d.next_valid_steps[0] : null;
  const expAction = nextStepId ? (STEP_TO_ACTION[nextStepId] || 'N/A').toUpperCase() : 'N/A';
  
  const va = (d.validated_action || 'IDLE').toUpperCase();
  const isInvalid = va === 'INVALID' || va === 'UNCERTAIN';
  const valColor = isInvalid ? 'var(--red)' : (va === 'IDLE' ? 'var(--text-dim)' : 'var(--green)');
  const valStatus = d.validation_reason ? d.validation_reason.toUpperCase() : (isInvalid ? 'INVALID' : 'VALID');

  overlayCtx.fillStyle = 'var(--text)';
  overlayCtx.font = '600 12px Inter';
  overlayCtx.fillText(`Expected: ${expAction}`, panelX + 20, panelY + 85);
  
  overlayCtx.fillStyle = valColor;
  overlayCtx.fillText(`Status: ${valStatus}`, panelX + 160, panelY + 85);
}

function drawProcedurePanel(d, width, height) {
  const panelX = width - 200;
  const panelY = 30;
  
  overlayCtx.fillStyle = 'rgba(0, 0, 0, 0.7)';
  overlayCtx.beginPath();
  overlayCtx.roundRect(panelX, panelY, 170, 70, 8);
  overlayCtx.fill();
  
  const stepId = d.current_step || 'step_01';
  const stepNum = parseInt(stepId.replace('step_',''), 10) || 1;
  const nextStepId = (d.next_valid_steps && d.next_valid_steps.length > 0) ? d.next_valid_steps[0] : null;
  const expAction = nextStepId ? (STEP_TO_ACTION[nextStepId] || 'N/A').toUpperCase() : 'N/A';

  overlayCtx.fillStyle = '#fff';
  overlayCtx.font = '700 16px Inter';
  overlayCtx.fillText(`STEP 0${stepNum} / 06`, panelX + 16, panelY + 30);
  
  overlayCtx.fillStyle = 'var(--accent)';
  overlayCtx.font = '600 13px Inter';
  overlayCtx.fillText(`Expected: ${expAction}`, panelX + 16, panelY + 54);
}

function drawAnomalyWarning(d, width, height) {
  if (d.anomaly && d.anomaly !== 'NONE') {
    const rawAction = (d.raw_action || 'UNKNOWN').toUpperCase();
    const nextStepId = (d.next_valid_steps && d.next_valid_steps.length > 0) ? d.next_valid_steps[0] : null;
    const expAction = nextStepId ? (STEP_TO_ACTION[nextStepId] || 'UNKNOWN').toUpperCase() : 'UNKNOWN';

    const panelW = 400;
    const panelH = 120;
    const panelX = (width - panelW) / 2;
    const panelY = height * 0.2; // upper middle
    
    overlayCtx.fillStyle = 'rgba(220, 38, 38, 0.9)'; // Red transparent
    overlayCtx.beginPath();
    overlayCtx.roundRect(panelX, panelY, panelW, panelH, 12);
    overlayCtx.fill();
    overlayCtx.strokeStyle = 'var(--red)';
    overlayCtx.lineWidth = 2;
    overlayCtx.stroke();

    overlayCtx.fillStyle = '#fff';
    overlayCtx.textAlign = 'center';
    
    overlayCtx.font = '800 24px Inter';
    overlayCtx.fillText('⚠ OUT OF SEQUENCE', panelX + panelW/2, panelY + 40);
    
    overlayCtx.font = '600 16px Inter';
    overlayCtx.fillText(`Observed: ${rawAction}`, panelX + panelW/2, panelY + 75);
    overlayCtx.fillText(`Expected: ${expAction}`, panelX + panelW/2, panelY + 100);
    
    overlayCtx.textAlign = 'left'; // reset
  }
}

// ── WebSocket ──
function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws/inference`);
  ws.binaryType = 'arraybuffer';

  ws.onopen = () => { startSending(); };
  ws.onclose = () => { sending = false; };
  ws.onmessage = (ev) => {
    const data = JSON.parse(ev.data);
    if (data.type === 'inference') {
      currentMLData = data;
      handleResult(data);
      if (sending && cameraRunning) sendFrame();
    }
  };
}

function startSending() {
  sending = true;
  sendFrame();
}

function sendFrame() {
  if (!ws || ws.readyState !== WebSocket.OPEN || !cameraRunning) return;
  if (!videoEl.videoWidth) { setTimeout(sendFrame, 100); return; }

  sendCanvas.width = videoEl.videoWidth;
  sendCanvas.height = videoEl.videoHeight;
  sendCtx.drawImage(videoEl, 0, 0);

  sendCanvas.toBlob(blob => {
    if (blob && ws && ws.readyState === WebSocket.OPEN) {
      blob.arrayBuffer().then(buf => ws.send(buf));
    }
  }, 'image/jpeg', 0.7);
}

// ── Handle Info Bar Updates ──
function handleResult(d) {
  safeSetText('infFps', d.inference_fps || '0');
  
  const stepAction = (STEP_TO_ACTION[d.current_step] || 'idle').toUpperCase();
  safeSetText('infoStep', `${d.current_step || 'step_01'} / ${stepAction}`);
  
  const nextStepId = (d.next_valid_steps && d.next_valid_steps.length > 0) ? d.next_valid_steps[0] : null;
  const nextAction = nextStepId ? (STEP_TO_ACTION[nextStepId] || 'N/A').toUpperCase() : 'N/A';
  safeSetText('infoExpected', nextAction);
  
  safeSetText('infoValidated', (d.validated_action || 'IDLE').toUpperCase());
  
  safeSetText('infoAnomaly', d.anomaly || 'NONE');
  const anomEl = document.getElementById('infoAnomaly');
  if (anomEl) {
    anomEl.style.color = (d.anomaly && d.anomaly !== 'NONE') ? 'var(--red)' : 'var(--text-dim)';
  }
  
  updateTimeline(d.current_step);
}

// ── Timeline ──
function buildTimeline() {
  const tl = document.getElementById('timeline');
  tl.innerHTML = '';
  ACTIONS.forEach((a, i) => {
    const step = document.createElement('div');
    step.className = 'timeline-step';
    step.id = `tlStep_${i}`;
    step.innerHTML = `<div class="step-circle">${i+1}</div><div class="step-label">${a.toUpperCase()}</div>`;
    tl.appendChild(step);
    if (i < ACTIONS.length - 1) {
      const conn = document.createElement('div');
      conn.className = 'timeline-connector';
      conn.id = `tlConn_${i}`;
      tl.appendChild(conn);
    }
  });
}
buildTimeline();

function updateTimeline(stepId) {
  const stepNum = parseInt((stepId || 'step_01').replace('step_', ''), 10) - 1;
  ACTIONS.forEach((_, i) => {
    const el = document.getElementById(`tlStep_${i}`);
    const conn = document.getElementById(`tlConn_${i}`);
    if(!el) return;
    el.className = 'timeline-step';
    if (i < stepNum) { el.classList.add('completed'); if(conn) conn.classList.add('done'); }
    else if (i === stepNum) { el.classList.add('current'); if(conn) conn.classList.remove('done'); }
    else { if(conn) conn.classList.remove('done'); }
  });
}

// ── Reset ──
async function resetExperiment() {
  try {
    const res = await fetch('/api/reset', { method: 'POST' });
    const data = await res.json();
    if (data.ok && data.state) {
      currentMLData = null; // Clear overlay
      handleResult(data.state); // Use state machine summary
      // clear canvas
      overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    }
  } catch(e) {
    console.error('Reset failed:', e);
  }
}

// Initialize timeline at Step 1
updateTimeline('step_01');
</script>
</body>
</html>
"""

with open("scratch/astra_web/index.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("Redesigned to a single live camera view.")
