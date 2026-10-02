"""
ASTRA — ST-GCN++ Skeleton Action Recognition

Recognizes PRIMITIVE actions (reach, grasp, lift, place, etc.)
from temporal skeleton sequences.

Input: T × J × C skeleton sequence
Output: Action class + confidence

Does NOT directly classify experiment steps — only primitive actions.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from ml.models.interfaces import ActionRecognizer, ActionResult


# Default primitive action classes
DEFAULT_ACTION_CLASSES = [
    "idle", "reach", "grasp", "release", "move", "place", "pick",
    "remove", "insert", "open", "close", "rotate", "press",
    "pull", "push", "transfer", "use", "visual_inspection",
]


def get_spatial_graph(num_joints: int = 17, layout: str = "coco") -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Get the adjacency matrix and edge list for the skeleton graph.

    Args:
        num_joints: Number of joints.
        layout: Skeleton layout ("coco" for COCO-17).

    Returns:
        (adjacency_matrix, edge_list)
    """
    if layout == "coco":
        edges = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (0, 5), (0, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16),
        ]
        # Add self-loops
        self_loops = [(i, i) for i in range(num_joints)]
        all_edges = edges + self_loops
    else:
        # Generic fully-connected
        all_edges = [(i, j) for i in range(num_joints) for j in range(num_joints)]

    # Build adjacency matrix
    A = np.zeros((num_joints, num_joints), dtype=np.float32)
    for i, j in all_edges:
        if i < num_joints and j < num_joints:
            A[i, j] = 1
            A[j, i] = 1

    # Normalize
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.where(D > 0, 1.0 / np.sqrt(D), 0)
    A_norm = D_inv_sqrt[:, None] * A * D_inv_sqrt[None, :]

    return A_norm, edges


class STGCNPP(ActionRecognizer):
    """
    ST-GCN++ skeleton-based action recognition model.

    Architecture:
    - Spatial graph convolutions on joint topology
    - Temporal convolutions along time axis
    - Multi-scale temporal modeling

    Input: (T, J, C) skeleton sequence
        T = window length (e.g., 64 frames)
        J = number of joints (17 for COCO)
        C = channels (x, y, conf, [velocity, bone_vectors])

    Output: ActionResult with primitive action class
    """

    def __init__(
        self,
        num_classes: int = 18,
        in_channels: int = 3,
        num_joints: int = 17,
        window_size: int = 64,
        graph_layout: str = "coco",
        action_classes: Optional[List[str]] = None,
        hidden_dim: int = 64,
        num_layers: int = 10,
    ):
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_joints = num_joints
        self.window_size = window_size
        self.graph_layout = graph_layout
        self.action_classes = action_classes or DEFAULT_ACTION_CLASSES[:num_classes]
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.model = None
        self.device = "cpu"

        # Get spatial graph
        self.A, self.edges = get_spatial_graph(num_joints, graph_layout)

    def load_model(self, checkpoint: str = "", device: str = "cpu"):
        """Load ST-GCN++ model weights."""
        self.device = device

        try:
            import torch
            import torch.nn as nn
        except ImportError:
            logger.warning("PyTorch not available. Using mock ST-GCN++.")
            return

        if checkpoint and Path(checkpoint).exists():
            state = torch.load(checkpoint, map_location=device, weights_only=False)
            
            # Read metadata if available
            if isinstance(state, dict):
                if "classes" in state:
                    self.action_classes = state["classes"]
                    self.num_classes = len(self.action_classes)
                elif "num_classes" in state:
                    self.num_classes = state["num_classes"]

            self.model = self._build_model()
            state_dict = state.get("model_state_dict", state.get("model", state))
            self.model.load_state_dict(state_dict)
            self.model.to(device)
            self.model.eval()
            logger.info(f"ST-GCN++ loaded from: {checkpoint}")
        else:
            self.model = self._build_model()
            self.model.to(device)
            self.model.eval()
            logger.info("ST-GCN++ initialized (no pretrained weights)")
            self.model.eval()
            logger.info(f"ST-GCN++ initialized (no pretrained weights)")

    def _build_model(self):
        """Build the ST-GCN++ PyTorch model."""
        import torch
        import torch.nn as nn

        class SpatialGraphConv(nn.Module):
            def __init__(self, in_channels, out_channels, A):
                super().__init__()
                self.A = nn.Parameter(torch.from_numpy(A).float(), requires_grad=False)
                self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
                self.bn = nn.BatchNorm2d(out_channels)

            def forward(self, x):
                # x: (N, C, T, V)
                N, C, T, V = x.shape
                # Graph convolution: multiply by adjacency
                x_a = torch.einsum('nctv,vw->nctw', x, self.A)
                x_a = self.conv(x_a)
                x_a = self.bn(x_a)
                return x_a

        class TemporalConv(nn.Module):
            def __init__(self, channels, kernel_size=9):
                super().__init__()
                padding = (kernel_size - 1) // 2
                self.conv = nn.Conv2d(channels, channels, (kernel_size, 1), padding=(padding, 0))
                self.bn = nn.BatchNorm2d(channels)

            def forward(self, x):
                return self.bn(self.conv(x))

        class STGCNBlock(nn.Module):
            def __init__(self, in_channels, out_channels, A, stride=1):
                super().__init__()
                self.gcn = SpatialGraphConv(in_channels, out_channels, A)
                self.tcn = TemporalConv(out_channels)
                self.relu = nn.ReLU(inplace=True)
                self.residual = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels),
                ) if in_channels != out_channels else nn.Identity()
                if stride > 1:
                    self.pool = nn.AvgPool2d((stride, 1))
                else:
                    self.pool = nn.Identity()

            def forward(self, x):
                res = self.residual(x)
                x = self.relu(self.gcn(x))
                x = self.tcn(x)
                x = self.pool(x)
                res = self.pool(res)
                return self.relu(x + res)

        class STGCNPlusPlus(nn.Module):
            def __init__(self, in_channels, num_classes, num_joints, A, hidden_dim, num_layers):
                super().__init__()
                self.data_bn = nn.BatchNorm1d(in_channels * num_joints)

                layers = []
                ch = in_channels
                for i in range(num_layers):
                    out_ch = hidden_dim * (2 ** min(i // 3, 2))
                    stride = 2 if i in [3, 6] else 1
                    layers.append(STGCNBlock(ch, out_ch, A, stride))
                    ch = out_ch
                self.layers = nn.Sequential(*layers)
                self.fc = nn.Linear(ch, num_classes)
                self.output_dim = ch

            def forward(self, x):
                # x: (N, C, T, V)
                N, C, T, V = x.shape
                x_bn = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
                x_bn = self.data_bn(x_bn)
                x = x_bn.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()

                x = self.layers(x)
                # Spatial pooling
                x = x.mean(dim=-1)  # (N, C, T)
                # Temporal pooling over the final 15 positions
                pool_frames = min(15, x.size(-1))
                x = x[:, :, -pool_frames:].mean(dim=-1)  # (N, C)
                return self.fc(x)

        return STGCNPlusPlus(
            self.in_channels, self.num_classes, self.num_joints,
            self.A, self.hidden_dim, self.num_layers,
        )

    def predict(self, skeleton_sequence: np.ndarray) -> ActionResult:
        """
        Recognize action from skeleton sequence.

        Args:
            skeleton_sequence: (T, J, C) array.

        Returns:
            ActionResult with predicted action.
        """
        if self.model is None:
            # No model — return mock result
            return ActionResult(
                action_id="idle", action_name="Idle",
                confidence=0.5, all_scores={c: 0.05 for c in self.action_classes},
            )

        import torch

        # Do NOT pad/truncate dynamically. Use original sequence.
        # WAIT: The model was trained with EXACTLY 60 frames padded by repeating the last frame!
        T, J, C = skeleton_sequence.shape
        seq = np.zeros((60, J, C), dtype=np.float32)
        if T <= 60:
            seq[:T] = skeleton_sequence
            for i in range(T, 60):
                seq[i] = skeleton_sequence[-1]
        else:
            seq = skeleton_sequence[-60:]
        T = 60
        if C < self.in_channels:
            extra = np.zeros((T, J, self.in_channels - C), dtype=np.float32)
            seq = np.concatenate([seq, extra], axis=-1)
        elif C > self.in_channels:
            seq = seq[:, :, :self.in_channels]

        # To tensor: (T, J, C) -> (1, C, T, J)
        tensor = torch.from_numpy(seq).permute(2, 0, 1).unsqueeze(0).float()
        tensor = tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)  # (1, num_classes)
            probs = torch.softmax(logits, dim=-1).cpu().numpy().squeeze(0)

        # Get top prediction
        top_idx = int(np.argmax(probs))
        action_id = self.action_classes[top_idx] if top_idx < len(self.action_classes) else f"action_{top_idx}"

        all_scores = {}
        for i, cls in enumerate(self.action_classes):
            if i < len(probs):
                all_scores[cls] = float(probs[i])

        return ActionResult(
            action_id=action_id,
            action_name=action_id.replace("_", " ").title(),
            confidence=float(probs[top_idx]),
            all_scores=all_scores,
        )

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": "ST-GCN++",
            "num_classes": self.num_classes,
            "in_channels": self.in_channels,
            "num_joints": self.num_joints,
            "window_size": self.window_size,
            "action_classes": self.action_classes,
        }
