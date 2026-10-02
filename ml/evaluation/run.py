"""
ASTRA — Evaluation Runner

Main entry point for running the complete evaluation framework.
Supports explicit --mock and --real modes. Never silently falls back.

Usage:
    # Mock evaluation
    python -m ml.evaluation.run --experiment experiments/sample_experiment/ --mock --output eval_results/mock/

    # REAL evaluation with trained weights
    python -m ml.evaluation.run --experiment experiments/sample_experiment/ \
        --real \
        --dataset data/synthetic/ \
        --weights checkpoints/detection/YOLO11S-ASTRA-v1/weights/best.pt \
        --output eval_results/real_v1/
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="ASTRA Evaluation Runner")
    parser.add_argument("--experiment", required=True, help="Experiment config directory")
    parser.add_argument("--dataset", help="Evaluation dataset directory (sessions with annotations)")
    parser.add_argument("--weights", help="Detection weights path (required for --real)")

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--mock", action="store_true", help="Use mock dataset + mock models")
    mode_group.add_argument("--real", action="store_true", help="Use real dataset + real models")

    parser.add_argument("--output", default="eval_results", help="Output directory")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--max_frames", type=int, default=-1, help="Max frames per session")
    parser.add_argument("--benchmark", action="store_true", help="Run latency/memory benchmark")
    parser.add_argument("--n_mock_frames", type=int, default=100, help="Mock dataset size")
    parser.add_argument("--split_file", help="Path to split_info.json for session selection")

    args = parser.parse_args()

    # ---- Validate arguments ----
    if args.real and not args.weights:
        parser.error("--real requires --weights <path_to_checkpoint>")
    if args.real and not args.dataset:
        parser.error("--real requires --dataset <path_to_dataset>")
    if args.real and args.weights and not os.path.isfile(args.weights):
        logger.error(f"FATAL: Weights file not found: {args.weights}")
        sys.exit(1)

    from ml.experiment.schema.loader import load_experiment
    from ml.pipeline.inference import ASTRAPipeline
    from ml.evaluation.dataset_loader import generate_mock_eval_dataset
    from ml.evaluation.evaluate_end_to_end import evaluate_end_to_end
    from ml.evaluation.report import generate_report

    config = load_experiment(args.experiment)
    mode_label = "REAL" if args.real else "MOCK"
    logger.info(f"Mode: {mode_label}")
    logger.info(f"Experiment: {config.experiment_id} | {len(config.steps)} steps")

    # ---- Load dataset ----
    if args.mock:
        logger.info(f"Generating mock evaluation dataset ({args.n_mock_frames} frames)...")
        dataset = generate_mock_eval_dataset(config, n_frames=args.n_mock_frames)
    else:
        logger.info(f"Loading REAL dataset from: {args.dataset}")
        dataset = _load_real_dataset(args.dataset, args.split_file)

    logger.info(f"Dataset: {dataset.summary()}")

    # ---- Create pipeline ----
    pipeline = ASTRAPipeline(
        experiment_config=config,
        device=args.device,
        use_mock=args.mock,
        log_dir=os.path.join(args.output, "logs"),
    )

    if args.real:
        logger.info(f"Loading REAL weights: {args.weights}")
        pipeline.load_models(detection_weights=args.weights)
        # Verify detector is NOT mock
        det_type = type(pipeline.detector).__name__
        if "Mock" in det_type:
            logger.error(f"FATAL: --real specified but detector is {det_type}. Aborting.")
            sys.exit(1)
        logger.info(f"Detector: {det_type} (REAL)")
    else:
        pipeline.load_models()
        logger.info(f"Detector: {type(pipeline.detector).__name__} (MOCK)")

    # Report model modes
    model_modes = _get_model_modes(pipeline)
    logger.info(f"Model modes: {model_modes}")

    # ---- Run evaluation ----
    logger.info("Running end-to-end evaluation...")
    results = evaluate_end_to_end(
        pipeline=pipeline,
        dataset=dataset,
        experiment_config=config,
        output_dir=args.output,
        max_frames=args.max_frames,
    )

    # Inject metadata
    results["evaluation_mode"] = mode_label
    results["model_modes"] = model_modes
    results["weights_path"] = args.weights if args.real else "mock"
    results["dataset_path"] = args.dataset if args.real else "mock_generated"

    # ---- Validate predictions ----
    det = results.get("detection", {})
    n_pred = det.get("n_pred_total", 0)
    if args.real and n_pred == 0:
        logger.error("=" * 60)
        logger.error("  REAL evaluation produced ZERO predictions.")
        logger.error("  Check checkpoint/data/model wiring.")
        logger.error("=" * 60)
        results["CRITICAL_ERROR"] = "REAL evaluation produced zero predictions"

    # ---- Fix misleading metrics ----
    results = _fix_insufficient_data_metrics(results, dataset)

    # ---- Benchmark ----
    if args.benchmark:
        logger.info("Running benchmark...")
        from ml.evaluation.benchmark import benchmark_pipeline
        bench_results = benchmark_pipeline(pipeline, n_frames=50)
        results["benchmark"] = bench_results
        bench_results["mode"] = mode_label

        bench_path = os.path.join(args.output, "benchmark_results.json")
        with open(bench_path, "w") as f:
            json.dump(bench_results, f, indent=2, default=str)

    # ---- Re-save with metadata ----
    json_path = os.path.join(args.output, "evaluation_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # ---- Generate report ----
    report_path = generate_report(results, output_dir=args.output)
    logger.info(f"Report generated: {report_path}")

    # ---- Print summary ----
    proc = results.get("procedure", {})
    det = results.get("detection", {})
    lat = results.get("latency", {})

    print()
    print("=" * 60)
    print(f"  ASTRA EVALUATION SUMMARY [{mode_label}]")
    print("=" * 60)
    print(f"  Dataset:              {results.get('dataset_path', '?')}")
    print(f"  Weights:              {results.get('weights_path', '?')}")
    print(f"  Predictions:          {det.get('n_pred_total', '?')}")
    print(f"  mAP@50:               {det.get('mAP50', '?')}")
    print(f"  Composite Score:      {proc.get('composite_score', '?')}")
    print(f"  Sequence Accuracy:    {proc.get('sequence', {}).get('sequence_accuracy', '?')}")
    print(f"  Step F1:              {proc.get('step_validation', {}).get('step_f1', '?')}")
    print(f"  False VALID Rate:     {proc.get('step_validation', {}).get('false_valid_rate', '?')}")
    print(f"  UNCERTAIN Rate:       {proc.get('step_validation', {}).get('uncertain_rate', '?')}")
    print(f"  Completion Correct:   {proc.get('experiment_completion', {}).get('correct', '?')}")
    print(f"  FPS:                  {lat.get('fps', '?')}")
    print(f"  Mean Latency:         {lat.get('mean_ms', '?')} ms")

    print(f"\n  Model Modes:")
    for k, v in model_modes.items():
        print(f"    {k:20s} {v}")

    limits = results.get("known_limitations", [])
    if limits:
        print(f"\n  Known Limitations ({len(limits)}):")
        for lim in limits:
            print(f"    - {lim}")

    if "CRITICAL_ERROR" in results:
        print(f"\n  ⚠️  CRITICAL: {results['CRITICAL_ERROR']}")

    print("=" * 60)
    print(f"  Full report: {report_path}")
    print("=" * 60)

    pipeline.close()


def _load_real_dataset(dataset_dir: str, split_file: str = None):
    """Load real dataset from session directories with annotations."""
    from ml.evaluation.dataset_loader import EvalDataset, EvalSample

    # Determine test sessions
    test_sessions = []
    if split_file and os.path.isfile(split_file):
        with open(split_file) as f:
            split_info = json.load(f)
        test_sessions = split_info.get("test_sessions", [])
    else:
        # Look for split_info.json in dataset or yolo dir
        for candidate in [
            os.path.join(dataset_dir, "split_info.json"),
            os.path.join(dataset_dir, "..", "yolo_astra", "split_info.json"),
            "data/yolo_astra/split_info.json",
        ]:
            if os.path.isfile(candidate):
                with open(candidate) as f:
                    split_info = json.load(f)
                test_sessions = split_info.get("test_sessions", [])
                logger.info(f"Found split_info: {candidate} → test={test_sessions}")
                break

    if not test_sessions:
        # Fall back to all sessions
        for entry in sorted(os.listdir(dataset_dir)):
            session_dir = os.path.join(dataset_dir, entry)
            if os.path.isdir(session_dir) and entry.startswith("session_"):
                test_sessions.append(entry)
        logger.warning(f"No split_info found. Using ALL sessions: {test_sessions}")

    # Load samples from test sessions
    samples = []
    gt_step_sequence = []
    gt_step_statuses = {}
    gt_anomalies = []
    step_ids = set()
    n_det = 0
    n_actions = 0
    n_steps = 0

    for session_id in test_sessions:
        session_dir = os.path.join(dataset_dir, session_id)
        if not os.path.isdir(session_dir):
            logger.warning(f"Session dir not found: {session_dir}")
            continue

        # Load manifest
        manifest_path = os.path.join(session_dir, "manifest.json")
        if not os.path.isfile(manifest_path):
            logger.warning(f"No manifest in {session_dir}")
            continue

        with open(manifest_path) as f:
            manifest = json.load(f)

        frames_dir = os.path.join(session_dir, "frames")

        # Load annotations
        ann_dir = os.path.join(session_dir, "annotations")
        det_data = _load_json(os.path.join(ann_dir, "detections.json"))
        steps_data = _load_jsonl(os.path.join(ann_dir, "steps.jsonl"))
        actions_data = _load_jsonl(os.path.join(ann_dir, "actions.jsonl"))
        states_data = _load_jsonl(os.path.join(ann_dir, "object_states.jsonl"))
        hoi_data = _load_jsonl(os.path.join(ann_dir, "hoi.jsonl"))
        anomaly_data = _load_jsonl(os.path.join(ann_dir, "anomalies.jsonl"))

        # Index detection annotations by image_id
        det_by_frame = {}
        if det_data and "annotations" in det_data:
            categories = {c["id"]: c["name"] for c in det_data.get("categories", [])}
            for ann in det_data["annotations"]:
                fid = ann["image_id"]
                if fid not in det_by_frame:
                    det_by_frame[fid] = []
                det_by_frame[fid].append({
                    # Convert COCO [x, y, w, h] → [x1, y1, x2, y2] for compute_iou/compute_map
                    "bbox": [
                        ann["bbox"][0],
                        ann["bbox"][1],
                        ann["bbox"][0] + ann["bbox"][2],
                        ann["bbox"][1] + ann["bbox"][3],
                    ],
                    "class": categories.get(ann["category_id"], f"cls_{ann['category_id']}"),
                    "class_id": ann["category_id"],
                    "confidence": ann.get("confidence", 1.0),
                })
            n_det += len(det_data["annotations"])

        # Index steps/actions by frame range
        step_by_frame = {}
        for step in steps_data:
            step_ids.add(step["step_id"])
            for f in range(step.get("start_frame", 0), step.get("end_frame", 0) + 1):
                step_by_frame[f] = step

        action_by_frame = {}
        for act in actions_data:
            for f in range(act.get("start_frame", 0), act.get("end_frame", 0) + 1):
                action_by_frame[f] = act
            n_actions += 1

        # Build step sequence for this session
        session_steps = [s["step_id"] for s in steps_data]
        if not gt_step_sequence:  # Use first session as reference
            gt_step_sequence = session_steps
        for s in steps_data:
            gt_step_statuses[s["step_id"]] = s.get("status", "completed")
            n_steps += 1

        # Anomalies
        gt_anomalies.extend(anomaly_data)

        # Create samples
        for frame_info in manifest.get("frames", []):
            fid = frame_info["frame_id"]
            fname = frame_info["filename"]
            img_path = os.path.join(frames_dir, fname)

            gt_dets = det_by_frame.get(fid, [])
            gt_step = step_by_frame.get(fid)
            gt_action = action_by_frame.get(fid)

            sample = EvalSample(
                sample_id=f"{session_id}_f{fid}",
                session_id=session_id,
                video_id=session_id,
                frame_id=fid,
                image_path=img_path if os.path.isfile(img_path) else None,
                timestamp=frame_info.get("timestamp", fid / 15.0),
                gt_detections=gt_dets,
                gt_step=gt_step,
                gt_actions=[gt_action] if gt_action else [],
            )
            samples.append(sample)

    data_type = "SYNTHETIC"
    # Check if any manifest says REAL
    for session_id in test_sessions:
        meta_path = os.path.join(dataset_dir, session_id, "session_manifest.json")
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get("data_type") == "REAL":
                data_type = "REAL"
                break

    dataset = EvalDataset(
        name=f"astra_{data_type.lower()}_v1",
        split="test",
        samples=samples,
        step_ids=sorted(step_ids),
        gt_step_sequence=gt_step_sequence,
        gt_step_statuses=gt_step_statuses,
        gt_anomalies=gt_anomalies,
        sessions=test_sessions,
        videos=test_sessions,
        class_names=["container", "sample", "scissors", "tweezers", "tray"],
        action_names=[],
    )

    logger.info(f"Loaded {len(samples)} samples from {len(test_sessions)} test sessions")
    logger.info(f"  GT detections: {n_det}, steps: {n_steps}, actions: {n_actions}")
    logger.info(f"  Data type: {data_type}")

    return dataset


def _load_json(path):
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return None


def _load_jsonl(path):
    if not os.path.isfile(path):
        return []
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _get_model_modes(pipeline) -> dict:
    """Determine actual mode of each model component."""
    modes = {}
    det_type = type(pipeline.detector).__name__
    modes["YOLO"] = "REAL" if "Mock" not in det_type else "MOCK"

    pose_type = type(pipeline.pose_estimator).__name__
    # RTMPoseEstimator silently falls back to mock when rtmlib is unavailable
    if "Mock" in pose_type:
        modes["Pose"] = "FALLBACK"
    elif hasattr(pipeline.pose_estimator, '_pose_model') and pipeline.pose_estimator._pose_model is None:
        modes["Pose"] = "FALLBACK"
    else:
        modes["Pose"] = "REAL"

    modes["ConvNeXt"] = "REAL" if hasattr(pipeline.visual_extractor, 'model') and pipeline.visual_extractor.model is not None else "MOCK"
    modes["ST-GCN++"] = "REAL"  # Always uses PyTorch model (no pretrained weights yet)
    modes["HOI"] = "REAL"  # Proximity model, no external weights
    modes["ObjectState"] = "REAL"  # Rule-based
    modes["Temporal"] = "REAL"  # GRU always loaded

    return modes


def _fix_insufficient_data_metrics(results, dataset):
    """Replace misleading perfect/zero metrics with INSUFFICIENT_DATA."""
    summary = dataset.summary()

    # If no anomaly GT, don't report anomaly metrics as real
    proc = results.get("procedure", {})
    anomaly = proc.get("anomaly", {})
    if not dataset.gt_anomalies:
        if anomaly:
            anomaly["note"] = "INSUFFICIENT_DATA — no anomaly ground truth"
            if anomaly.get("f1", 0) == 1.0 or anomaly.get("f1", 0) == 0.0:
                anomaly["f1"] = "INSUFFICIENT_DATA"

    # HOI: if no HOI GT
    hoi = results.get("hoi", {})
    if not summary.get("has_hoi", False):
        hoi["note"] = "INSUFFICIENT_DATA — no HOI ground truth"

    # Object state: if no state GT
    obj_state = results.get("object_state", {})
    if not summary.get("has_object_states", False):
        obj_state["note"] = "INSUFFICIENT_DATA — no object state ground truth"

    # Pose: if using fallback
    pose = results.get("pose", {})
    pose_type = type(results).__name__  # Will be dict, but check model_modes
    model_modes = results.get("model_modes", {})
    if model_modes.get("Pose") == "FALLBACK":
        pose["note"] = "INSUFFICIENT_DATA — pose using fallback (no RTMPose)"

    # Latency label
    lat = results.get("latency", {})
    lat["mode"] = results.get("evaluation_mode", "UNKNOWN")

    return results


if __name__ == "__main__":
    main()
