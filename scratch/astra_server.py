"""
ASTRA — Live Web Server (FastAPI + WebSocket)

Architecture:
    Browser (getUserMedia) → WebSocket (JPEG frames) → FastAPI
        → ASTRA Pipeline (YOLO, RTMPose, ST-GCN, TemporalGRU, etc.)
        → WebSocket (JSON results + annotated detections)
        → Browser overlay

Launch:
    python scratch/astra_server.py
"""
import os
import sys
import time
import json
import base64
import asyncio
import threading
import logging
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from ml.experiment.schema.loader import load_experiment
from ml.pipeline.inference import ASTRAPipeline
from ml.experiment.state_graph.state_machine import ExperimentStateMachine
from ml.pipeline.action_smoother import ActionSmoother

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("astra_server")

# ── Global pipeline state ──────────────────────────────────────
pipeline: ASTRAPipeline = None
pipeline_lock = threading.Lock()
pipeline_ready = threading.Event()
config_ref = None  # hold experiment config for reset

ACTION_SEQUENCE = ["idle", "reach", "pick", "move", "place", "release"]


def init_pipeline():
    """Heavy model loading – runs once in a background thread."""
    global pipeline, config_ref
    logger.info("Loading ASTRA models …")

    exp_dir = os.path.join(PROJECT_ROOT, "experiments", "temporal_demo")
    det_w = os.path.join(PROJECT_ROOT, "checkpoints", "detection", "YOLO11S-ASTRA-v1", "weights", "best.pt")
    act_w = os.path.join(PROJECT_ROOT, "checkpoints", "action", "STGCNPP-ASTRA-v3", "best.pt")
    tmp_w = os.path.join(PROJECT_ROOT, "checkpoints", "action", "TemporalGRU-ASTRA-v1", "best.pt")

    config_ref = load_experiment(exp_dir)
    pipeline = ASTRAPipeline(config_ref, device="cpu", use_mock=False, mock_pose=False)
    pipeline.load_models(detection_weights=det_w, action_weights=act_w, temporal_weights=tmp_w)

    pipeline_ready.set()
    logger.info("Pipeline ready.")


def reset_pipeline():
    """Reset state machine / buffers without reloading weights."""
    global pipeline
    with pipeline_lock:
        action_classes = [a.id for a in config_ref.actions]
        pipeline.state_machine = ExperimentStateMachine(config_ref)
        pipeline.action_smoother = ActionSmoother(
            window_size=15, min_persistence=5,
            confidence_threshold=0.3, cooldown_frames=10,
            action_classes=action_classes,
        )
        pipeline.skeleton_buffer.clear()
        pipeline.feature_buffer.clear()
        pipeline.frame_count = 0
        pipeline.start_time = None
    logger.info("Pipeline reset.")


# ── FastAPI ─────────────────────────────────────────────────────
app = FastAPI(title="ASTRA Live Server")

# Serve static frontend
STATIC_DIR = os.path.join(os.path.dirname(__file__), "astra_web")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/live")
async def live_page():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/status")
async def status():
    return {"ready": pipeline_ready.is_set()}


@app.post("/api/reset")
async def reset():
    if pipeline_ready.is_set():
        reset_pipeline()
        with pipeline_lock:
            # Re-fetch the newly initialized state
            sm = pipeline.state_machine.get_state_summary()
            current_step = sm.get("current_step_id") or "step_01"
            next_valid_steps = pipeline.state_machine._get_next_valid_steps()
            
            next_action = "REACH"
            if next_valid_steps:
                ns_id = next_valid_steps[0]
                ns_obj = config_ref.get_step(ns_id)
                if ns_obj and ns_obj.required_actions:
                    next_action = ns_obj.required_actions[0].upper()
            
        return {
            "ok": True,
            "state": {
                "current_step": current_step,
                "current_action": "IDLE",  # first step is idle
                "next_expected": next_action,
                "anomaly": "NONE",
                "raw_action": "—",
                "validated_action": "—",
                "validation_reason": "—",
                "temporal_action": "—",
                "temporal_anomaly": 0
            }
        }
    return {"ok": False, "reason": "Pipeline not ready"}


@app.websocket("/ws/inference")
async def ws_inference(ws: WebSocket):
    await ws.accept()
    logger.info("WebSocket client connected")

    if not pipeline_ready.is_set():
        await ws.send_json({"type": "status", "ready": False, "message": "Loading models…"})
        await asyncio.to_thread(pipeline_ready.wait)
    await ws.send_json({"type": "status", "ready": True, "message": "Pipeline ready"})

    inference_count = 0
    t_last_log = time.time()

    try:
        while True:
            raw = await ws.receive_bytes()

            # Decode JPEG
            arr = np.frombuffer(raw, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            t0 = time.perf_counter()

            def _run_inference(f):
                with pipeline_lock:
                    return pipeline.process_frame(f, timestamp=time.time())

            result = await asyncio.to_thread(_run_inference, frame)

            dt = time.perf_counter() - t0
            inference_count += 1

            sm = result.get("state_machine", {})
            anomalies = sm.get("anomalies", [])
            latest_anomaly = anomalies[-1] if anomalies else None

            # Get frame dimensions for normalizing overlay coords
            fh, fw = frame.shape[:2]

            response = {
                "type": "result",
                "frame_id": result.get("frame_id", 0),
                "timestamp": result.get("timestamp", 0),
                "inference_ms": round(dt * 1000, 1),
                "inference_fps": round(1.0 / dt, 2) if dt > 0 else 0,
                "frame_w": fw,
                "frame_h": fh,

                # Perception
                "raw_action": result.get("raw_action") or "idle",
                "pose_confidence": round(result.get("pose_confidence", 0), 4),
                "detections_count": result.get("detections", 0),
                "poses_count": result.get("poses", 0),

                # Validation
                "validated_action": result.get("action") or "idle",
                "validation_reason": result.get("validation_reason", ""),

                # Temporal
                "temporal_action": result.get("temporal_action") or "N/A",
                "temporal_action_confidence": round(result.get("temporal_action_confidence", 0), 4),
                "temporal_next": result.get("temporal_next") or "N/A",
                "anomaly_score": round(result.get("anomaly_score", 0), 4),

                # State Machine
                "current_step": sm.get("current_step_id", "step_01"),
                "experiment_status": sm.get("experiment_status", "IN_PROGRESS"),
                "next_valid_steps": sm.get("next_valid_steps", []),

                # Anomaly
                "anomaly": latest_anomaly.get("type") if latest_anomaly else "NONE",
                "anomaly_detail": latest_anomaly.get("reason", "") if latest_anomaly else "",

                # Overlay data (detection bboxes + pose keypoints in pixel coords)
                "detection_boxes": result.get("detection_boxes", []),
                "pose_keypoints_raw": result.get("pose_keypoints_raw", []),
            }

            await ws.send_json(response)

            # Periodic log
            now = time.time()
            if now - t_last_log > 5.0:
                logger.info(
                    f"Inferences: {inference_count} | "
                    f"Last: {dt*1000:.0f}ms | "
                    f"Raw={result.get('raw_action')} Validated={result.get('action')} "
                    f"Step={sm.get('current_step_id')}"
                )
                t_last_log = now

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)


# ── Main entry ──────────────────────────────────────────────────
if __name__ == "__main__":
    # Start model loading in background
    threading.Thread(target=init_pipeline, daemon=True).start()

    logger.info("Starting ASTRA server on http://localhost:8765")
    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="info")
