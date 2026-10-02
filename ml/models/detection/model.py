"""
ASTRA — YOLO11-S Object Detection + Tracking

Detects experiment-specific objects and maintains tracking across frames.
Uses Ultralytics YOLO11 with built-in BoT-SORT/ByteTrack.

Usage:
    # Inference
    from ml.models.detection.model import YOLODetector
    detector = YOLODetector()
    detector.load_model("checkpoints/yolo11s-astra-v1.pt")
    detections = detector.detect(frame)

    # Training
    python -m ml.models.detection.train --config configs/detection.yaml
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies
_ultralytics = None


def _get_ultralytics():
    global _ultralytics
    if _ultralytics is None:
        try:
            import ultralytics
            _ultralytics = ultralytics
        except ImportError:
            raise ImportError(
                "ultralytics is required for YOLO detection. "
                "Install with: pip install ultralytics"
            )
    return _ultralytics


# Import interfaces
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from ml.models.interfaces import ObjectDetector, Detection


class YOLODetector(ObjectDetector):
    """
    YOLO11-S object detector with built-in tracking.

    Responsibilities:
    - Detect experiment-specific objects per frame
    - Maintain object identity across frames via tracking
    - Output bounding boxes, class IDs, confidence, tracking IDs

    Model: YOLO11-S (Ultralytics)
    Input: RGB frame (resized to img_size)
    Output: List[Detection]
    """

    def __init__(
        self,
        img_size: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "auto",
        tracker: str = "botsort.yaml",
        max_det: int = 50,
    ):
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.tracker = tracker
        self.max_det = max_det
        self.model = None
        self.class_names: Dict[int, str] = {}
        self._track_history: Dict[int, List] = {}

    def load_model(self, checkpoint: str = "yolo11s.pt", device: str = None):
        """
        Load YOLO model from checkpoint.

        Args:
            checkpoint: Path to .pt weights or model name.
            device: "cpu", "cuda", "auto", or device index.
        """
        ultralytics = _get_ultralytics()
        self.model = ultralytics.YOLO(checkpoint)
        if device is not None:
            self.device = device
        self.class_names = self.model.names if hasattr(self.model, 'names') else {}
        logger.info(f"YOLO model loaded: {checkpoint} | Classes: {len(self.class_names)}")

    def detect(self, frame: np.ndarray, track: bool = True) -> List[Detection]:
        """
        Detect and optionally track objects in a single frame.

        Args:
            frame: BGR image (H, W, 3).
            track: If True, use tracking to maintain object IDs.

        Returns:
            List of Detection objects.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if track:
            results = self.model.track(
                frame,
                imgsz=self.img_size,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                max_det=self.max_det,
                persist=True,
                tracker=self.tracker,
                verbose=False,
                device=self.device,
            )
        else:
            results = self.model.predict(
                frame,
                imgsz=self.img_size,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                max_det=self.max_det,
                verbose=False,
                device=self.device,
            )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy().tolist()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                cls_name = self.class_names.get(cls_id, f"class_{cls_id}")

                # Get tracking ID if available
                track_id = -1
                if track and boxes.id is not None:
                    track_id = int(boxes.id[i].cpu().numpy())

                detections.append(Detection(
                    object_id=track_id,
                    class_name=cls_name,
                    class_id=cls_id,
                    bbox=bbox,
                    confidence=conf,
                ))

        return detections

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """Detect objects in a batch of frames (no tracking)."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        results = self.model.predict(
            frames,
            imgsz=self.img_size,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=self.max_det,
            verbose=False,
        )

        all_detections = []
        for result in results:
            frame_dets = []
            boxes = result.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    bbox = boxes.xyxy[i].cpu().numpy().tolist()
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    cls_name = self.class_names.get(cls_id, f"class_{cls_id}")
                    frame_dets.append(Detection(
                        object_id=-1,
                        class_name=cls_name,
                        class_id=cls_id,
                        bbox=bbox,
                        confidence=conf,
                    ))
            all_detections.append(frame_dets)

        return all_detections

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": "YOLO11-S",
            "framework": "ultralytics",
            "img_size": self.img_size,
            "conf_threshold": self.conf_threshold,
            "num_classes": len(self.class_names),
            "class_names": dict(self.class_names),
            "tracker": self.tracker,
        }

    def reset_tracker(self):
        """Reset tracking state (call between videos)."""
        self._track_history.clear()
        if self.model is not None:
            self.model.predictor = None


class MockDetector(ObjectDetector):
    """
    Mock detector for testing without YOLO weights.
    Returns configurable synthetic detections.
    """

    def __init__(self, mock_detections: Optional[List[Detection]] = None):
        self._mock = mock_detections or []
        self._frame_count = 0

    def load_model(self, checkpoint: str = "", device: str = "cpu"):
        logger.info("MockDetector loaded (no real model)")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        self._frame_count += 1
        # Return mock detections with slight position jitter
        result = []
        for det in self._mock:
            jittered = Detection(
                object_id=det.object_id,
                class_name=det.class_name,
                class_id=det.class_id,
                bbox=[b + np.random.normal(0, 2) for b in det.bbox],
                confidence=max(0, min(1, det.confidence + np.random.normal(0, 0.05))),
                frame_id=self._frame_count,
            )
            result.append(jittered)
        return result

    def set_mock_detections(self, detections: List[Detection]):
        self._mock = detections
