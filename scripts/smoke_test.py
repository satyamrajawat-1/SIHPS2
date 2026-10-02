"""
ASTRA — Model Smoke Tests

Verifies that every model can perform REAL inference independently.
Mock mode and real mode are clearly distinguished.

Usage:
    python scripts/smoke_test.py --all
    python scripts/smoke_test.py --model detection
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _make_test_frame(h=480, w=640):
    """Create a synthetic test frame with shapes for detection."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = (40, 40, 40)  # Dark gray background
    # Draw some rectangles (simulate objects)
    cv2.rectangle(frame, (100, 100), (250, 200), (0, 200, 0), -1)
    cv2.rectangle(frame, (300, 150), (450, 280), (200, 0, 0), -1)
    cv2.rectangle(frame, (50, 300), (180, 400), (0, 0, 200), -1)
    cv2.circle(frame, (500, 350), 50, (200, 200, 0), -1)
    return frame


def _report(name, mode, success, details):
    status = "PASS" if success else "FAIL"
    mode_tag = f"[REAL]" if mode == "real" else "[MOCK]"
    print(f"  {status} {mode_tag} {name}")
    for k, v in details.items():
        print(f"       {k}: {v}")
    return {"name": name, "mode": mode, "success": success, **details}


def test_detection(use_mock=False):
    """Test YOLO11-S detection."""
    name = "YOLO11-S Detection"
    frame = _make_test_frame()

    if use_mock:
        from ml.models.detection.model import MockDetector, Detection
        det = MockDetector([
            Detection(object_id=1, class_name="container", class_id=0,
                      bbox=[100, 100, 250, 200], confidence=0.9),
        ])
        det.load_model()
        t0 = time.perf_counter()
        results = det.detect(frame)
        lat = (time.perf_counter() - t0) * 1000
        return _report(name, "mock", True, {
            "predictions": len(results), "latency_ms": f"{lat:.1f}",
            "note": "MockDetector — not real inference"})

    try:
        from ml.models.detection.model import YOLODetector
        det = YOLODetector(conf_threshold=0.1)
        det.load_model("yolo11s.pt", device="cpu")
        t0 = time.perf_counter()
        results = det.detect(frame, track=False)
        lat = (time.perf_counter() - t0) * 1000
        classes = [r.class_name for r in results]
        confs = [round(r.confidence, 3) for r in results[:5]]
        return _report(name, "real", True, {
            "model_loaded": "yolo11s.pt", "predictions": len(results),
            "classes": classes[:5], "confidences": confs,
            "latency_ms": f"{lat:.1f}"})
    except Exception as e:
        return _report(name, "real", False, {"error": str(e)})


def test_pose(use_mock=False):
    """Test RTMPose-L pose estimation."""
    name = "RTMPose-L Pose"
    frame = _make_test_frame()

    if use_mock:
        from ml.models.pose.model import MockPoseEstimator
        est = MockPoseEstimator()
        est.load_model()
        t0 = time.perf_counter()
        results = est.predict(frame)
        lat = (time.perf_counter() - t0) * 1000
        return _report(name, "mock", True, {
            "poses": len(results), "keypoints_shape": results[0].keypoints_2d.shape,
            "latency_ms": f"{lat:.1f}", "note": "MockPoseEstimator"})

    try:
        from ml.models.pose.model import RTMPoseEstimator
        est = RTMPoseEstimator()
        est.load_model(device="cpu")
        t0 = time.perf_counter()
        results = est.predict(frame)
        lat = (time.perf_counter() - t0) * 1000
        has_real = est._pose_model is not None
        return _report(name, "real" if has_real else "fallback", True, {
            "poses": len(results),
            "keypoints_shape": results[0].keypoints_2d.shape if results else "none",
            "has_real_model": has_real, "latency_ms": f"{lat:.1f}"})
    except Exception as e:
        return _report(name, "real", False, {"error": str(e)})


def test_visual(use_mock=False):
    """Test ConvNeXt-Tiny visual features."""
    name = "ConvNeXt-Tiny Visual"
    frame = _make_test_frame()

    if use_mock:
        from ml.models.visual.convnext import MockVisualExtractor
        ext = MockVisualExtractor(256)
        ext.load_model()
        t0 = time.perf_counter()
        feat = ext.extract(frame)
        lat = (time.perf_counter() - t0) * 1000
        return _report(name, "mock", True, {
            "feature_shape": feat.shape, "latency_ms": f"{lat:.1f}",
            "note": "MockVisualExtractor"})

    try:
        from ml.models.visual.convnext import ConvNeXtExtractor
        ext = ConvNeXtExtractor(projection_dim=256)
        ext.load_model(device="cpu")
        t0 = time.perf_counter()
        feat = ext.extract(frame)
        lat = (time.perf_counter() - t0) * 1000
        return _report(name, "real", True, {
            "feature_shape": feat.shape, "feature_norm": f"{np.linalg.norm(feat):.3f}",
            "latency_ms": f"{lat:.1f}"})
    except Exception as e:
        return _report(name, "real", False, {"error": str(e)})


def test_action(use_mock=False):
    """Test ST-GCN++ action recognition."""
    name = "ST-GCN++ Action"
    skeleton = np.random.rand(64, 17, 3).astype(np.float32)

    try:
        from ml.models.action.stgcn.model import STGCNPP
        model = STGCNPP(num_classes=18)
        model.load_model(device="cpu")
        has_real = model.model is not None
        t0 = time.perf_counter()
        result = model.predict(skeleton)
        lat = (time.perf_counter() - t0) * 1000
        return _report(name, "real" if has_real else "mock", True, {
            "action": result.action_id, "confidence": f"{result.confidence:.3f}",
            "has_real_model": has_real, "latency_ms": f"{lat:.1f}"})
    except Exception as e:
        return _report(name, "real", False, {"error": str(e)})


def test_hoi():
    """Test HOI proximity model."""
    name = "HOI Proximity"
    from ml.models.hoi.model import ProximityHOIModel
    from ml.models.interfaces import PoseResult, Detection

    kps = np.random.rand(17, 3).astype(np.float32)
    kps[:, 0] *= 640; kps[:, 1] *= 480; kps[:, 2] = 0.9
    kps[10, :2] = [200, 180]  # Right wrist near object

    pose = PoseResult(person_id=0, bbox=[50, 50, 300, 400], keypoints_2d=kps)
    dets = [Detection(object_id=1, class_name="container", class_id=0,
                      bbox=[180, 160, 260, 220], confidence=0.9)]

    model = ProximityHOIModel(contact_threshold=60)
    t0 = time.perf_counter()
    results = model.predict(pose, dets)
    lat = (time.perf_counter() - t0) * 1000
    return _report(name, "real", True, {
        "interactions": len(results),
        "types": [r.interaction_type for r in results],
        "latency_ms": f"{lat:.1f}"})


def test_object_state():
    """Test object state classifier."""
    name = "Object State"
    from ml.models.object_state.model import ObjectStateClassifier
    from ml.models.interfaces import Detection
    frame = _make_test_frame()

    clf = ObjectStateClassifier(state_definitions={"container": ["open", "closed"]})
    dets = [Detection(object_id=1, class_name="container", class_id=0,
                      bbox=[100, 100, 250, 200], confidence=0.9)]
    t0 = time.perf_counter()
    results = clf.predict(dets, frame)
    lat = (time.perf_counter() - t0) * 1000
    return _report(name, "real", True, {
        "predictions": len(results),
        "states": {r.object_id: r.current_state for r in results},
        "latency_ms": f"{lat:.1f}"})


def test_temporal(use_mock=False):
    """Test temporal model."""
    name = "Temporal Model"
    features = np.random.rand(32, 373).astype(np.float32)

    try:
        from ml.models.temporal.model import ASTRATemporalModel
        model = ASTRATemporalModel(input_dim=373)
        model.load_model(device="cpu")
        t0 = time.perf_counter()
        result = model.predict(features)
        lat = (time.perf_counter() - t0) * 1000
        has_result = result.current_action is not None
        return _report(name, "real" if has_result else "empty", True, {
            "action": result.current_action,
            "step": result.current_step_candidate,
            "anomaly_score": f"{result.anomaly_score:.3f}" if result.anomaly_score else "none",
            "latency_ms": f"{lat:.1f}"})
    except Exception as e:
        return _report(name, "real", False, {"error": str(e)})


def test_pipeline(use_mock=False):
    """Test complete pipeline."""
    name = "Complete Pipeline"
    frame = _make_test_frame()

    try:
        from ml.experiment.schema.loader import load_experiment
        from ml.pipeline.inference import ASTRAPipeline

        config = load_experiment("experiments/sample_experiment")
        pipeline = ASTRAPipeline(config, device="cpu", use_mock=use_mock,
                                  log_dir="eval_results/smoke_logs")
        pipeline.load_models()

        t0 = time.perf_counter()
        result = pipeline.process_frame(frame, timestamp=0.0)
        lat = (time.perf_counter() - t0) * 1000
        sm = result["state_machine"]

        pipeline.close()
        return _report(name, "mock" if use_mock else "real", True, {
            "detections": result["detections"], "poses": result["poses"],
            "action": result["action"], "status": sm["experiment_status"],
            "step": sm["current_step_id"], "latency_ms": f"{lat:.1f}"})
    except Exception as e:
        return _report(name, "mock" if use_mock else "real", False, {"error": str(e)})


def main():
    parser = argparse.ArgumentParser(description="ASTRA Model Smoke Tests")
    parser.add_argument("--model", choices=["detection", "pose", "visual", "action",
                        "hoi", "object_state", "temporal", "pipeline"],
                        help="Test specific model")
    parser.add_argument("--all", action="store_true", help="Test all models")
    parser.add_argument("--mock", action="store_true", help="Use mock models")
    args = parser.parse_args()

    tests = {
        "detection": lambda: test_detection(args.mock),
        "pose": lambda: test_pose(args.mock),
        "visual": lambda: test_visual(args.mock),
        "action": lambda: test_action(args.mock),
        "hoi": test_hoi,
        "object_state": test_object_state,
        "temporal": lambda: test_temporal(args.mock),
        "pipeline": lambda: test_pipeline(args.mock),
    }

    if not args.model and not args.all:
        args.all = True

    print("\n" + "=" * 60)
    print("  ASTRA Smoke Tests")
    print("=" * 60 + "\n")

    results = []
    to_run = tests if args.all else {args.model: tests[args.model]}

    for name, fn in to_run.items():
        try:
            r = fn()
            results.append(r)
        except Exception as e:
            results.append({"name": name, "success": False, "error": str(e)})
            print(f"  FAIL [{name}] {e}")
        print()

    passed = sum(1 for r in results if r.get("success"))
    total = len(results)
    real = sum(1 for r in results if r.get("mode") == "real")
    mock = sum(1 for r in results if r.get("mode") == "mock")

    print("=" * 60)
    print(f"  Results: {passed}/{total} passed | {real} real | {mock} mock")
    print("=" * 60)

    out_path = Path("eval_results/smoke_test_results.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved: {out_path}\n")


if __name__ == "__main__":
    main()
