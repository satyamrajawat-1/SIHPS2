"""
ASTRA — Main Inference Pipeline

Orchestrates the complete perception → reasoning → decision pipeline.
This is the top-level entry point for live or recorded video inference.

Architecture:
  Frame → [Detection] → [Pose] → [Action] → [HOI] → [State]
             ↓            ↓         ↓          ↓        ↓
          [Visual] → [Feature Aggregation] → [Temporal Model]
                              ↓
                    [Evidence Fusion Layer]
                              ↓
                 [Experiment State Machine]
                              ↓
                      [Event Logger]

Usage:
    python -m ml.pipeline.inference --experiment experiments/sample_experiment/ --video video.mp4
    python -m ml.pipeline.inference --experiment experiments/sample_experiment/ --camera 0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.experiment.schema.loader import load_experiment, ExperimentConfig
from ml.experiment.state_graph.state_machine import (
    Evidence, ExperimentStateMachine, ExperimentStatus,
)
from ml.experiment.event_logger import EventLogger

from ml.models.interfaces import Detection, PoseResult, ActionResult
from ml.models.detection.model import YOLODetector, MockDetector
from ml.models.pose.model import RTMPoseEstimator, normalize_skeleton, MockPoseEstimator
from ml.models.action.stgcn.model import STGCNPP
from ml.models.hoi.model import ProximityHOIModel
from ml.models.object_state.model import ObjectStateClassifier
from ml.models.visual.convnext import ConvNeXtExtractor, MockVisualExtractor
from ml.models.temporal.model import ASTRATemporalModel
from ml.models.fusion.evidence_fusion import EvidenceFusionLayer, FeatureAggregator
from ml.pipeline.action_smoother import ActionSmoother
from ml.pipeline.voice import VoiceAlertModule

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ASTRAPipeline:
    """
    Main ASTRA inference pipeline.

    Processes video frames through all perception models and feeds
    evidence to the experiment state machine.
    """

    def __init__(
        self,
        experiment_config: ExperimentConfig,
        device: str = "cpu",
        use_mock: bool = False,
        mock_pose: bool = False,
        log_dir: str = "logs",
        skeleton_window: int = 64,
    ):
        self.config = experiment_config
        self.device = device
        self.use_mock = use_mock
        self.mock_pose = mock_pose
        self.skeleton_window = skeleton_window

        # Build class-to-ID mapping from experiment config
        self.class_to_id = {}
        for obj in experiment_config.objects:
            self.class_to_id[obj.obj_class] = obj.id
            self.class_to_id[obj.obj_class.lower()] = obj.id

        # Feature aggregator (must be created before _init_models)
        action_classes = [a.id for a in experiment_config.actions]
        self.feature_aggregator = FeatureAggregator(
            visual_dim=256,
            pose_dim=51,
            action_dim=len(action_classes),
        )

        # Initialize models
        self._init_models()

        # State machine
        self.state_machine = ExperimentStateMachine(experiment_config)

        # Event logger
        os.makedirs(log_dir, exist_ok=True)
        self.event_logger = EventLogger(
            log_dir, experiment_config.experiment_id, "live"
        )

        # Evidence fusion
        self.evidence_fusion = EvidenceFusionLayer(
            object_class_to_id_map=self.class_to_id,
        )

        # Action smoother
        self.action_smoother = ActionSmoother(
            window_size=15,
            min_persistence=5,
            confidence_threshold=0.3,
            cooldown_frames=10,
            action_classes=action_classes
        )

        # Action validator
        from ml.pipeline.action_validator import ActionValidator
        self.action_validator = ActionValidator(
            experiment_config=experiment_config,
            pose_confidence_threshold=0.15
        )

        # Voice alerts
        self.voice = VoiceAlertModule(enabled=True)

        # Skeleton buffer for action recognition
        self.skeleton_buffer = deque(maxlen=skeleton_window)

        # Feature buffer for temporal model
        self.feature_buffer = deque(maxlen=128)

        # Frame counter
        self.frame_count = 0
        self.start_time = None

        logger.info(f"ASTRA Pipeline initialized | device={device} | mock={use_mock}")

    def _init_models(self):
        """Initialize all perception models."""
        if self.use_mock:
            self.detector = MockDetector()
            self.visual_extractor = MockVisualExtractor()
        else:
            self.detector = YOLODetector(device=self.device)
            self.visual_extractor = ConvNeXtExtractor()

        if self.use_mock or self.mock_pose:
            self.pose_estimator = MockPoseEstimator()
        else:
            self.pose_estimator = RTMPoseEstimator(device=self.device)

        self.action_recognizer = STGCNPP(
            action_classes=[a.id for a in self.config.actions],
            num_classes=len(self.config.actions),
        )

        self.hoi_model = ProximityHOIModel()

        self.object_state = ObjectStateClassifier()
        self.object_state.load_config_states(self.config)

        step_names = [s.id for s in self.config.steps]
        action_names = [a.id for a in self.config.actions]
        self.temporal_model = ASTRATemporalModel(
            input_dim=self.feature_aggregator.total_dim,
            num_actions=len(action_names),
            num_steps=len(step_names),
            action_names=action_names,
            step_names=step_names,
        )

    def load_models(
        self,
        detection_weights: str = "",
        pose_weights: str = "",
        action_weights: str = "",
        visual_weights: str = "",
        temporal_weights: str = "",
    ):
        """Load all model weights."""
        try:
            self.detector.load_model(detection_weights or "yolo11s.pt", self.device)
        except Exception as e:
            logger.warning(f"Detection model load failed: {e}. Using mock.")
            self.detector = MockDetector()
            self.detector.load_model()

        try:
            self.pose_estimator.load_model(pose_weights, self.device)
        except Exception as e:
            logger.warning(f"Pose model load failed: {e}. Using mock.")
            self.pose_estimator = MockPoseEstimator()
            self.pose_estimator.load_model()

        try:
            self.visual_extractor.load_model(visual_weights, self.device)
        except Exception as e:
            logger.warning(f"Visual model load failed: {e}. Using mock.")
            self.visual_extractor = MockVisualExtractor()
            self.visual_extractor.load_model()

        try:
            self.action_recognizer.load_model(action_weights, self.device)
        except Exception as e:
            logger.warning(f"Action model load failed: {e}")

        try:
            self.temporal_model.load_model(temporal_weights, self.device)
        except Exception as e:
            logger.warning(f"Temporal model load failed: {e}")

        logger.info("All models loaded.")

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None,
        camera_id: str = "cam_00",
    ) -> Dict[str, Any]:
        """
        Process a single frame through the full pipeline.

        Args:
            frame: BGR image (H, W, 3).
            timestamp: Frame timestamp (seconds). Auto-incremented if None.
            camera_id: Camera identifier.

        Returns:
            Dict with state machine result, events, and perception outputs.
        """
        self.frame_count += 1

        if self.start_time is None:
            self.start_time = timestamp or 0.0
            self.event_logger.log_experiment_started(self.start_time)

        if timestamp is None:
            timestamp = (self.frame_count - 1) / 30.0

        # ---- Perception Pipeline ----

        # 1. Object Detection
        detections = self.detector.detect(frame)

        # 2. Pose Estimation
        poses = self.pose_estimator.predict(frame, detections)

        # 3. Visual Features
        visual_features = self.visual_extractor.extract(frame)

        # 4. Action Recognition (from skeleton buffer)
        action_result = None
        raw_action = None
        raw_confidence = 0.0
        validated_action_str = None
        
        if poses:
            best_pose = max(poses, key=lambda p: p.tracking_confidence)
            norm_kps = normalize_skeleton(best_pose.keypoints_2d)
            self.skeleton_buffer.append(norm_kps)
            
            # compute pose mean confidence
            pose_confs = [p.keypoints_2d[:, 2].mean() for p in poses if p.keypoints_2d.shape[1] > 2]
            mean_pose_conf = sum(pose_confs) / len(pose_confs) if pose_confs else 0.0

            if len(self.skeleton_buffer) >= 16:  # Minimum window for action
                skeleton_seq = np.array(list(self.skeleton_buffer))
                raw_action_result = self.action_recognizer.predict(skeleton_seq)
                raw_action = raw_action_result.action_id
                raw_confidence = raw_action_result.confidence
                
                # Apply temporal smoothing
                smoothed = self.action_smoother.update(
                    raw_action, raw_confidence
                )
                
                # Validate the smoothed action against the procedure state
                val_action, is_valid, reason = self.action_validator.validate(
                    raw_action=smoothed.action_id,
                    raw_confidence=smoothed.confidence,
                    pose_confidence=mean_pose_conf,
                    current_step_id=self.state_machine.current_step_id
                )
                validated_action_str = val_action
                
                # Log the debug instrumentation
                logger.debug(
                    f"Frame {self.frame_count} | Raw: {raw_action} ({raw_confidence:.2f}) | "
                    f"PoseConf: {mean_pose_conf:.2f} | Smoothed: {smoothed.action_id} | "
                    f"Validated: {val_action} | Valid: {is_valid} | Reason: {reason} | "
                    f"Step: {self.state_machine.current_step_id}"
                )
                
                # Protect Temporal GRU semantic input:
                # If validated action differs from raw_action, we construct a new all_scores
                # that ensures the Temporal GRU only sees the validated action.
                final_scores = {}
                if val_action in ["invalid", "uncertain"]:
                    # Create an all-zero score vector or default to idle
                    for k in raw_action_result.all_scores.keys():
                        final_scores[k] = 1.0 if k == "idle" else 0.0
                elif val_action != raw_action:
                    for k in raw_action_result.all_scores.keys():
                        final_scores[k] = 1.0 if k == val_action else 0.0
                else:
                    final_scores = raw_action_result.all_scores.copy()

                # Replace with validated result
                from ml.models.interfaces import ActionResult
                action_result = ActionResult(
                    action_id=val_action,
                    action_name=val_action,
                    confidence=smoothed.confidence if is_valid else 0.0,
                    all_scores=final_scores
                )

        # 5. HOI
        hoi_results = []
        if poses:
            for pose in poses:
                hoi_results.extend(
                    self.hoi_model.predict(pose, detections, frame)
                )

        # 6. Object State
        obj_states = self.object_state.predict(detections, frame, hoi_results)

        # 7. Feature Aggregation → Temporal Model
        frame_features = self.feature_aggregator.aggregate(
            visual_features=visual_features,
            pose_results=poses,
            action_result=action_result,
            hoi_results=hoi_results,
            object_states=obj_states,
        )
        self.feature_buffer.append(frame_features)

        temporal_result = None
        if len(self.feature_buffer) >= 8:  # Minimum sequence for temporal
            feature_seq = np.array(list(self.feature_buffer))
            temporal_result = self.temporal_model.predict(feature_seq)

        # ---- Evidence Fusion ----
        evidence = self.evidence_fusion.fuse(
            frame_id=self.frame_count,
            timestamp=timestamp,
            camera_id=camera_id,
            detections=detections,
            poses=poses,
            action_result=action_result,
            hoi_results=hoi_results,
            object_states=obj_states,
            temporal_result=temporal_result,
        )

        # ---- State Machine Update ----
        sm_result = self.state_machine.update(evidence)

        if validated_action_str == "invalid":
            anomaly = {
                "type": "OUT_OF_ORDER_ACTION",
                "timestamp": timestamp,
                "observed_action": raw_action,
                "expected_step": self.state_machine.current_step_id,
                "reason": f"ST-GCN emitted {raw_action} which is invalid for current state",
                "confidence": raw_confidence,
            }
            if "anomalies" not in sm_result:
                sm_result["anomalies"] = []
            sm_result["anomalies"].append(anomaly)
            self.state_machine.anomalies.append(anomaly)

        # ---- Event Logging ----
        for event in sm_result.get("events", []):
            ev_type = event.get("event", "")
            step_id = event.get("step_id", "")
            step = self.config.get_step(step_id) if step_id else None

            if ev_type == "STEP_STARTED":
                self.event_logger.log_step_started(
                    timestamp, step_id, step.name if step else "",
                )
            elif ev_type == "STEP_COMPLETED":
                self.event_logger.log_step_completed(
                    timestamp, step_id,
                    action=evidence.detected_action or "",
                    confidence=event.get("confidence", 0),
                )
                self.voice.on_step_complete(step.name if step else step_id)

                # Log next step suggestion
                next_steps = sm_result.get("next_valid_steps", [])
                if next_steps and next_steps[0] != "COMPLETE":
                    next_step = self.config.get_step(next_steps[0])
                    self.event_logger.log_next_step_suggested(
                        timestamp, next_steps[0],
                        next_step.name if next_step else "",
                    )
                    self.voice.on_next_step(next_step.name if next_step else next_steps[0])
            elif ev_type == "STEP_UNCERTAIN":
                self.event_logger.log_step_uncertain(timestamp, step_id)
                self.voice.on_uncertain()
            elif ev_type == "STEP_TIMEOUT":
                self.event_logger.log_step_failed(
                    timestamp, step_id, "Step timed out",
                )
                self.voice.on_timeout_warning(step.name if step else step_id)

        for anomaly in sm_result.get("anomalies", []):
            self.event_logger.log_anomaly(
                timestamp,
                anomaly.get("type", "UNKNOWN"),
                anomaly.get("reason", ""),
                step_id=anomaly.get("observed_step"),
                confidence=anomaly.get("confidence", 0),
            )
            if anomaly.get("type") == "OUT_OF_ORDER_ACTION":
                expected = anomaly.get("expected_step", "")
                detected = anomaly.get("observed_action", anomaly.get("observed_step", ""))
                self.voice.on_out_of_order(expected, detected)
            elif anomaly.get("type") == "SKIPPED_STEP":
                self.voice.on_skipped_step(anomaly.get("observed_step", ""))

        # Check experiment completion
        if sm_result.get("experiment_status") == "COMPLETE":
            self.event_logger.log_experiment_completed(timestamp)
            if "EXPERIMENT_COMPLETE" not in [e.get("event") for e in sm_result.get("events", [])]:
                self.voice.on_experiment_complete()

        return {
            "frame_id": self.frame_count,
            "timestamp": timestamp,
            "state_machine": sm_result,
            "detections": len(detections),
            "poses": len(poses),
            "action": action_result.action_id if action_result else None,
            "raw_action": raw_action,
            "pose_confidence": mean_pose_conf if 'mean_pose_conf' in locals() else 0.0,
            "validation_reason": reason if 'reason' in locals() else "",
            "hoi_count": len(hoi_results),
            "object_states": {s.object_id: s.current_state for s in obj_states},
            "temporal_action": temporal_result.current_action if temporal_result else None,
            "temporal_action_confidence": temporal_result.current_action_confidence if temporal_result else 0.0,
            "temporal_step": temporal_result.current_step_candidate if temporal_result else None,
            "temporal_next": temporal_result.predicted_next_action if temporal_result else None,
            "anomaly_score": temporal_result.anomaly_score if temporal_result else 0.0,
            # Raw overlay data for web frontend
            "detection_boxes": [
                {"bbox": d.bbox, "class_name": d.class_name, "confidence": round(d.confidence, 3)}
                for d in detections
            ],
            "pose_keypoints_raw": [p.to_dict() for p in poses],
        }

    def process_video(
        self,
        video_path: str,
        max_frames: int = -1,
        skip_frames: int = 0,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Process an entire video file.

        Args:
            video_path: Path to video file.
            max_frames: Maximum frames to process (-1 = all).
            skip_frames: Skip every N frames (0 = process all).
            show_progress: Print progress to stdout.

        Returns:
            Final experiment state summary.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Processing video: {video_path} | {total} frames @ {fps} FPS")

        frame_idx = 0
        processed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            if skip_frames > 0 and frame_idx % (skip_frames + 1) != 0:
                continue

            if max_frames > 0 and processed >= max_frames:
                break

            timestamp = frame_idx / fps
            result = self.process_frame(frame, timestamp)
            processed += 1

            if show_progress and processed % 100 == 0:
                status = result["state_machine"]["experiment_status"]
                step = result["state_machine"]["current_step_id"]
                logger.info(
                    f"Frame {processed}/{total} | "
                    f"Status: {status} | Step: {step}"
                )

        cap.release()

        summary = self.state_machine.get_state_summary()
        summary["frames_processed"] = processed
        summary["video_path"] = video_path

        logger.info(f"Processing complete: {processed} frames")
        return summary

    def get_guidance(self) -> Dict[str, Any]:
        """
        Get current guidance for the astronaut.

        Returns:
            Dict with:
              - current_step: what they should be doing
              - next_step: what comes next
              - warnings: any issues detected
              - voice_message: suggested voice prompt
        """
        summary = self.state_machine.get_state_summary()
        current_step_id = summary.get("current_step_id")

        guidance = {
            "experiment_status": summary["experiment_status"],
            "current_step_id": current_step_id,
            "current_step_name": "",
            "next_step_id": None,
            "next_step_name": "",
            "progress": f"{len(summary['completed_steps'])}/{len(self.config.steps)}",
            "warnings": [],
            "voice_message": "",
        }

        if current_step_id:
            step = self.config.get_step(current_step_id)
            if step:
                guidance["current_step_name"] = step.name
                next_steps = [s for s in step.allowed_next_steps if s != "COMPLETE"]
                if next_steps:
                    next_step = self.config.get_step(next_steps[0])
                    guidance["next_step_id"] = next_steps[0]
                    guidance["next_step_name"] = next_step.name if next_step else ""

        # Generate voice message
        if summary["experiment_status"] == "COMPLETE":
            guidance["voice_message"] = "Experiment complete. All steps verified."
        elif current_step_id:
            step = self.config.get_step(current_step_id)
            if step:
                guidance["voice_message"] = f"Current step: {step.name}."
                if guidance["next_step_name"]:
                    guidance["voice_message"] += f" Next: {guidance['next_step_name']}."

        # Add anomaly warnings
        for anomaly in summary.get("anomalies", [])[-3:]:
            guidance["warnings"].append(anomaly.get("reason", ""))

        return guidance

    def reset(self):
        """Reset pipeline state for a new experiment session."""
        self.state_machine.reset()
        self.skeleton_buffer.clear()
        self.feature_buffer.clear()
        self.hoi_model.reset()
        self.object_state.reset()
        self.temporal_model.reset_hidden()
        self.frame_count = 0
        self.start_time = None

    def close(self):
        """Clean up resources."""
        self.event_logger.close()


def main():
    parser = argparse.ArgumentParser(description="ASTRA Inference Pipeline")
    parser.add_argument("--experiment", required=True, help="Experiment config directory")
    parser.add_argument("--video", help="Input video path")
    parser.add_argument("--camera", type=int, help="Camera device index for live mode")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--mock", action="store_true", help="Use mock models (no real weights)")
    parser.add_argument("--log_dir", default="logs", help="Log directory")
    parser.add_argument("--max_frames", type=int, default=-1, help="Max frames to process")
    parser.add_argument("--skip_frames", type=int, default=0, help="Skip every N frames")
    parser.add_argument("--detection_weights", default="", help="Detection model weights")
    parser.add_argument("--temporal_weights", default="", help="Temporal model weights")

    args = parser.parse_args()

    # Load experiment config
    config = load_experiment(args.experiment)
    logger.info(f"Experiment: {config.experiment_id} | {len(config.steps)} steps")

    # Create pipeline
    pipeline = ASTRAPipeline(
        experiment_config=config,
        device=args.device,
        use_mock=args.mock,
        log_dir=args.log_dir,
    )

    # Load models
    pipeline.load_models(
        detection_weights=args.detection_weights,
        temporal_weights=args.temporal_weights,
    )

    try:
        if args.video:
            # Process video file
            summary = pipeline.process_video(
                args.video,
                max_frames=args.max_frames,
                skip_frames=args.skip_frames,
            )
            print(json.dumps(summary, indent=2))

        elif args.camera is not None:
            # Live camera mode
            cap = cv2.VideoCapture(args.camera)
            if not cap.isOpened():
                logger.error(f"Cannot open camera {args.camera}")
                return

            logger.info(f"Live camera mode | Camera {args.camera}")
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                timestamp = frame_idx / 30.0

                result = pipeline.process_frame(frame, timestamp)

                # Print guidance periodically
                if frame_idx % 30 == 0:
                    guidance = pipeline.get_guidance()
                    step_name = guidance.get("current_step_name", "?")
                    progress = guidance.get("progress", "?")
                    print(f"\r[{progress}] {step_name}", end="", flush=True)

                # Exit on 'q' key
                cv2.imshow("ASTRA", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            cap.release()
            cv2.destroyAllWindows()

            summary = pipeline.state_machine.get_state_summary()
            print(json.dumps(summary, indent=2))

        else:
            parser.print_help()
            logger.info("Use --video or --camera to start inference.")

    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
