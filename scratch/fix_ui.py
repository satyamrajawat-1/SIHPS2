import re

file_path = "scratch/astra_web/index.html"
with open(file_path, "r", encoding="utf-8") as f:
    html = f.read()

# Add safe helpers at the top of the JS block
helpers = """
// ── Safe DOM Helpers ─────────────────────────────────
function safeSetText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}
function safeSetHTML(id, htmlStr) {
  const el = document.getElementById(id);
  if (el) el.innerHTML = htmlStr;
}
function safeSetClass(id, className) {
  const el = document.getElementById(id);
  if (el) el.className = className;
}
function safeSetStyle(id, key, val) {
  const el = document.getElementById(id);
  if (el) el.style[key] = val;
}
"""

if "function safeSetText" not in html:
    html = html.replace("// ── Clear UI ───────────────────────────────────────", helpers + "\n// ── Clear UI ───────────────────────────────────────")

# Replace document.getElementById('X').textContent = Y with safeSetText('X', Y)
# This is tricky with regex, let's just do targeted replacements.

# clearUI replacements
clearUI_fixes = [
    ("document.getElementById('poseConf').textContent = '—';", "safeSetText('poseConf', '—');"),
    ("document.getElementById('detCount').textContent = '—';", "safeSetText('detCount', '—');"),
    ("document.getElementById('poseCount').textContent = '—';", "safeSetText('poseCount', '—');"),
    
    ("const vaBig = document.getElementById('validatedActionBig');", "const vaBig = document.getElementById('validatedActionBig');"),
    ("vaBig.textContent = '—';", "if (vaBig) vaBig.textContent = '—';"),
    ("vaBig.className = 'action-big idle';", "if (vaBig) vaBig.className = 'action-big idle';"),
    
    ("const valReason = document.getElementById('valReason');", "const valReason = document.getElementById('valReason');"),
    ("valReason.textContent = '—';", "if (valReason) valReason.textContent = '—';"),
    ("valReason.style.color = 'var(--text-dim)';", "if (valReason) valReason.style.color = 'var(--text-dim)';"),
    
    ("document.getElementById('tempAction').textContent = '—';", "safeSetText('tempAction', '—');"),
    ("document.getElementById('tempAnom').textContent = '—';", "safeSetText('tempAnom', '—');"),
    
    ("const anomLabel = document.getElementById('anomalyLabel');", "const anomLabel = document.getElementById('anomalyLabel');"),
    ("anomLabel.textContent = 'NONE';", "if (anomLabel) anomLabel.textContent = 'NONE';"),
    ("anomLabel.style.color = 'var(--text-dim)';", "if (anomLabel) anomLabel.style.color = 'var(--text-dim)';"),
    
    ("document.getElementById('curStep').textContent = 'step_01';", "safeSetText('curStep', 'step_01');"),
    ("document.getElementById('curAction').textContent = 'IDLE';", "safeSetText('curAction', 'IDLE');"),
    ("document.getElementById('nextExpected').textContent = 'REACH';", "safeSetText('nextExpected', 'REACH');"),
    ("document.getElementById('debugState').textContent = 'step_01 / IDLE';", "safeSetText('debugState', 'step_01 / IDLE');"),
    ("document.getElementById('debugExpected').textContent = 'REACH';", "safeSetText('debugExpected', 'REACH');"),
    ("document.getElementById('debugRaw').textContent = '—';", "safeSetText('debugRaw', '—');"),
    ("document.getElementById('debugValidated').textContent = '—';", "safeSetText('debugValidated', '—');"),

    ("document.getElementById('curStep').textContent = '—';", "safeSetText('curStep', '—');"),
    ("document.getElementById('curAction').textContent = 'PAUSED';", "safeSetText('curAction', 'PAUSED');"),
    ("document.getElementById('nextExpected').textContent = '—';", "safeSetText('nextExpected', '—');"),
    ("document.getElementById('debugState').textContent = '— / PAUSED';", "safeSetText('debugState', '— / PAUSED');"),
]

# handleResult fixes
handleResult_fixes = [
    ("document.getElementById('infFps').textContent = d.inference_fps || '0';", "safeSetText('infFps', d.inference_fps || '0');"),
    ("document.getElementById('rawAction').textContent = (d.raw_action || 'idle').toUpperCase();", "safeSetText('rawAction', (d.raw_action || 'idle').toUpperCase());"),
    ("document.getElementById('poseConf').textContent = d.pose_confidence || '0';", "safeSetText('poseConf', d.pose_confidence || '0');"),
    ("document.getElementById('detCount').textContent = d.detections_count || '0';", "safeSetText('detCount', d.detections_count || '0');"),
    ("document.getElementById('poseCount').textContent = d.poses_count || '0';", "safeSetText('poseCount', d.poses_count || '0');"),
    
    ("vaBig.textContent = va;", "if (vaBig) vaBig.textContent = va;"),
    ("vaBig.className = 'action-big ' + (isInvalid ? 'invalid' : (d.validated_action === 'idle' ? 'idle' : 'valid'));", "if (vaBig) vaBig.className = 'action-big ' + (isInvalid ? 'invalid' : (d.validated_action === 'idle' ? 'idle' : 'valid'));"),
    
    ("document.getElementById('valReason').textContent =", "safeSetText('valReason', d.validation_reason ? d.validation_reason.toUpperCase() : (isInvalid ? 'OUT OF SEQUENCE' : 'VALID')); //"),
]

# resetExperiment fixes
reset_fixes = [
    ("document.getElementById('curStep').textContent = data.state.current_step;", "safeSetText('curStep', data.state.current_step);"),
    ("document.getElementById('curAction').textContent = data.state.current_action;", "safeSetText('curAction', data.state.current_action);"),
    ("document.getElementById('nextExpected').textContent = data.state.next_expected;", "safeSetText('nextExpected', data.state.next_expected);"),
    ("document.getElementById('rawAction').textContent = data.state.raw_action;", "safeSetText('rawAction', data.state.raw_action);"),
    
    ("vaBig.textContent = data.state.validated_action;", "if (vaBig) vaBig.textContent = data.state.validated_action;"),
    ("vaBig.className = 'action-big idle';", "if (vaBig) vaBig.className = 'action-big idle';"),
    
    ("valReason.textContent = data.state.validation_reason;", "if (valReason) valReason.textContent = data.state.validation_reason;"),
    ("valReason.style.color = 'var(--text-dim)';", "if (valReason) valReason.style.color = 'var(--text-dim)';"),
    
    ("document.getElementById('tempAction').textContent = data.state.temporal_action;", "safeSetText('tempAction', data.state.temporal_action);"),
    ("document.getElementById('tempAnom').textContent = data.state.temporal_anomaly;", "safeSetText('tempAnom', data.state.temporal_anomaly);"),
    
    ("anomLabel.textContent = data.state.anomaly;", "if (anomLabel) anomLabel.textContent = data.state.anomaly;"),
    ("anomLabel.style.color = 'var(--text-dim)';", "if (anomLabel) anomLabel.style.color = 'var(--text-dim)';"),
]

for orig, new in clearUI_fixes + handleResult_fixes + reset_fixes:
    html = html.replace(orig, new)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(html)
print("DOM elements safely patched.")
