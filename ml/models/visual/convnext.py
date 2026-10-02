"""
ASTRA — ConvNeXt-Tiny Visual Feature Extractor

Extracts global scene features for multimodal fusion.
NOT a replacement for YOLO — provides holistic visual context.

Output: 768-d or projected-d frame embedding.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from ml.models.interfaces import VisualFeatureExtractor


class ConvNeXtExtractor(VisualFeatureExtractor):
    """
    ConvNeXt-Tiny visual feature extractor.

    Input: RGB frame (resized to 224×224)
    Output: 768-d feature vector (or projected to projection_dim)

    Initially: frozen backbone + trainable projection head.
    Later: optional full fine-tuning.
    """

    def __init__(
        self,
        projection_dim: int = 256,
        input_size: int = 224,
        pretrained: bool = True,
    ):
        self.projection_dim = projection_dim
        self.input_size = input_size
        self.pretrained = pretrained
        self.model = None
        self.projection = None
        self.device = "cpu"
        self._backbone_dim = 768  # ConvNeXt-Tiny output dim

    def load_model(self, checkpoint: str = "", device: str = "cpu"):
        """
        Load ConvNeXt-Tiny backbone.

        Args:
            checkpoint: Custom checkpoint path (empty = pretrained from timm).
            device: "cpu" or "cuda".
        """
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError("torch is required. Install with: pip install torch")

        self.device = device

        if checkpoint and Path(checkpoint).exists():
            # Load custom checkpoint
            state = torch.load(checkpoint, map_location=device)
            if "backbone" in state:
                self._load_from_state(state)
                logger.info(f"ConvNeXt loaded from checkpoint: {checkpoint}")
                return
            # Fall through to timm loading

        try:
            import timm
            self.model = timm.create_model(
                "convnext_tiny.fb_in22k_ft_in1k",
                pretrained=self.pretrained,
                num_classes=0,  # Remove classification head → feature extractor
            )
        except ImportError:
            try:
                from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
                backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
                backbone.classifier = nn.Identity()
                self.model = backbone
            except Exception:
                raise ImportError(
                    "Either timm or torchvision >= 0.13 is required for ConvNeXt. "
                    "Install with: pip install timm"
                )

        # Freeze backbone by default
        for param in self.model.parameters():
            param.requires_grad = False

        # Create projection head
        import torch.nn as nn
        self.projection = nn.Sequential(
            nn.Linear(self._backbone_dim, self.projection_dim),
            nn.GELU(),
            nn.Linear(self.projection_dim, self.projection_dim),
        )

        self.model = self.model.to(device)
        self.projection = self.projection.to(device)
        self.model.eval()

        logger.info(
            f"ConvNeXt-Tiny loaded | backbone_dim={self._backbone_dim} "
            f"→ projection_dim={self.projection_dim} | device={device}"
        )

    def _load_from_state(self, state):
        """Load from a custom ASTRA checkpoint."""
        import torch
        import torch.nn as nn

        # Reconstruct model
        self.load_model("", self.device)  # Load pretrained first
        if "projection" in state:
            self.projection.load_state_dict(state["projection"])
        if "backbone" in state:
            self.model.load_state_dict(state["backbone"], strict=False)

    def extract(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract visual features from a frame.

        Args:
            frame: BGR image (H, W, 3).

        Returns:
            Feature vector (projection_dim,).
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        import torch
        import cv2

        # Preprocess: resize, normalize, convert to tensor
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_size, self.input_size))
        img = img.astype(np.float32) / 255.0

        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std

        # To tensor: (H, W, C) → (1, C, H, W)
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        tensor = tensor.to(self.device)

        with torch.no_grad():
            features = self.model(tensor)  # (1, 768)
            if self.projection is not None:
                features = self.projection(features)  # (1, projection_dim)

        return features.cpu().numpy().squeeze(0)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": "ConvNeXt-Tiny",
            "backbone_dim": self._backbone_dim,
            "projection_dim": self.projection_dim,
            "input_size": self.input_size,
            "device": self.device,
        }


class MockVisualExtractor(VisualFeatureExtractor):
    """Mock visual feature extractor for testing."""

    def __init__(self, output_dim: int = 256):
        self.output_dim = output_dim

    def load_model(self, checkpoint: str = "", device: str = "cpu"):
        logger.info("MockVisualExtractor loaded")

    def extract(self, frame: np.ndarray) -> np.ndarray:
        return np.random.randn(self.output_dim).astype(np.float32)
