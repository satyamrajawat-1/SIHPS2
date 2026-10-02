import re

file_path = "scratch/astra_web/index.html"
with open(file_path, "r", encoding="utf-8") as f:
    html = f.read()

# Change video object-fit to contain to avoid cropping
html = html.replace("object-fit: cover;", "object-fit: contain;")

# Add coordinate transformation logic inside drawOverlays
transform_logic = """
function getTransform(video, canvas, frameW, frameH) {
  // video uses object-fit: contain
  // It maintains aspect ratio inside the canvas bounds.
  // We need to find the actual rendered size of the video inside the canvas.
  
  const canvasW = canvas.width;
  const canvasH = canvas.height;
  
  const videoRatio = frameW / frameH;
  const canvasRatio = canvasW / canvasH;
  
  let drawW, drawH, offsetX, offsetY;
  
  if (canvasRatio > videoRatio) {
    // Canvas is wider than video (pillarboxing)
    drawH = canvasH;
    drawW = canvasH * videoRatio;
    offsetX = (canvasW - drawW) / 2;
    offsetY = 0;
  } else {
    // Canvas is taller than video (letterboxing)
    drawW = canvasW;
    drawH = canvasW / videoRatio;
    offsetX = 0;
    offsetY = (canvasH - drawH) / 2;
  }
  
  return {
    scaleX: drawW / frameW,
    scaleY: drawH / frameH,
    offsetX: offsetX,
    offsetY: offsetY
  };
}
"""

if "function getTransform" not in html:
    html = html.replace("// ── OVERLAYS DRAWING LOGIC ──", "// ── OVERLAYS DRAWING LOGIC ──\n" + transform_logic)

# Now inject the transform into drawOverlays
old_draw_func = """function drawOverlays(d) {
  const sx = overlayCanvas.width;
  const sy = overlayCanvas.height;"""

new_draw_func = """function drawOverlays(d) {
  const fw = d.frame_w || 640;
  const fh = d.frame_h || 480;
  const t = getTransform(videoEl, overlayCanvas, fw, fh);
  
  // Helper to map absolute backend coords to canvas coords
  const mapX = (x) => (x * t.scaleX) + t.offsetX;
  const mapY = (y) => (y * t.scaleY) + t.offsetY;
  
  // Keep sx, sy for the fixed overlays (badges) which are drawn relative to canvas bounds
  const sx = overlayCanvas.width;
  const sy = overlayCanvas.height;"""

html = html.replace(old_draw_func, new_draw_func)

# Fix object/person detections
html = html.replace(
    "const [x1, y1, x2, y2] = det.bbox;",
    "const x1 = mapX(det.bbox[0]); const y1 = mapY(det.bbox[1]); const x2 = mapX(det.bbox[2]); const y2 = mapY(det.bbox[3]);"
)
html = html.replace("x1 * sx", "x1")
html = html.replace("y1 * sy", "y1")
html = html.replace("w * sx", "w")
html = html.replace("h * sy", "h")

# Fix skeleton
html = html.replace(
    "overlayCtx.moveTo(kps[i][0] * sx, kps[i][1] * sy);",
    "overlayCtx.moveTo(mapX(kps[i][0]), mapY(kps[i][1]));"
)
html = html.replace(
    "overlayCtx.lineTo(kps[j][0] * sx, kps[j][1] * sy);",
    "overlayCtx.lineTo(mapX(kps[j][0]), mapY(kps[j][1]));"
)
html = html.replace(
    "overlayCtx.arc(x * sx, y * sy, 4, 0, Math.PI * 2);",
    "overlayCtx.arc(mapX(x), mapY(y), 4, 0, Math.PI * 2);"
)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(html)
print("Transform logic injected.")
