"""
ASTRA — Temporal Model (Mamba-2 SSM + CPU Fallback)

Processes the multimodal feature sequence (visual + pose + action + HOI + state)
over time to produce:
  - Refined current action estimate
  - Predicted next action
  - Current step candidate
  - Anomaly / deviation score
  - Temporal embeddings

GPU: Mamba-2 (selective state space model)
CPU Fallback: 1D causal convolutions + GRU (lightweight)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from ml.models.interfaces import TemporalModel, TemporalResult


class CausalGRUFallback:
    """
    Lightweight CPU-friendly temporal model.

    Architecture: 1D causal convolution → GRU → MLP head
    Used as a fallback when Mamba-2 is not available (CPU inference).
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_actions: int = 18,
        num_steps: int = 10,
        dropout: float = 0.1,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_actions = num_actions
        self.num_steps = num_steps
        self.model = None
        self.device = "cpu"

    def build(self):
        """Build the PyTorch model."""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            logger.warning("PyTorch not available for CausalGRUFallback")
            return

        class CausalConv1d(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size):
                super().__init__()
                self.padding = kernel_size - 1
                self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)

            def forward(self, x):
                # x: (N, C, T)
                x = nn.functional.pad(x, (self.padding, 0))
                return self.conv(x)

        class TemporalGRUModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_layers,
                         num_actions, num_steps, dropout):
                super().__init__()
                # Causal conv pre-processing
                self.conv1 = CausalConv1d(input_dim, hidden_dim, kernel_size=3)
                self.conv2 = CausalConv1d(hidden_dim, hidden_dim, kernel_size=3)
                self.bn1 = nn.BatchNorm1d(hidden_dim)

                # GRU for temporal modeling
                self.gru = nn.GRU(
                    hidden_dim, hidden_dim, num_layers,
                    batch_first=True, dropout=dropout if num_layers > 1 else 0,
                )

                # Output heads
                self.action_head = nn.Linear(hidden_dim, num_actions)
                self.step_head = nn.Linear(hidden_dim, num_steps)
                self.anomaly_head = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, 1),
                    nn.Sigmoid(),
                )
                self.next_action_head = nn.Linear(hidden_dim, num_actions)

            def forward(self, x, hidden=None):
                # x: (N, T, D)
                N, T, D = x.shape

                # Conv: (N, T, D) → (N, D, T) → conv → (N, H, T) → (N, T, H)
                x_conv = x.permute(0, 2, 1)
                x_conv = torch.relu(self.bn1(self.conv1(x_conv)))
                x_conv = torch.relu(self.conv2(x_conv))
                x_conv = x_conv.permute(0, 2, 1)

                # GRU
                out, hidden = self.gru(x_conv, hidden)

                # Use last timestep output
                last = out[:, -1, :]  # (N, H)

                return {
                    "action_logits": self.action_head(last),
                    "step_logits": self.step_head(last),
                    "anomaly_score": self.anomaly_head(last).squeeze(-1),
                    "next_action_logits": self.next_action_head(last),
                    "embeddings": last,
                    "hidden": hidden,
                }

        import torch
        self.model = TemporalGRUModel(
            self.input_dim, self.hidden_dim, self.num_layers,
            self.num_actions, self.num_steps, 0.1,
        ).to(self.device)
        self.model.eval()

    def predict(self, features: np.ndarray, hidden=None):
        """Run inference on a feature sequence."""
        import torch

        if self.model is None:
            self.build()
            if self.model is None:
                return None

        tensor = torch.from_numpy(features).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            output = self.model(tensor, hidden)

        return output


class ASTRATemporalModel(TemporalModel):
    """
    ASTRA Temporal Model.

    GPU: Mamba-2 selective SSM for long-range temporal modeling
    CPU: Causal convolution + GRU fallback

    Both share the same input/output interface.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_actions: int = 18,
        num_steps: int = 10,
        action_names: Optional[List[str]] = None,
        step_names: Optional[List[str]] = None,
        use_mamba: bool = True,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.num_steps = num_steps
        self.action_names = action_names or [f"action_{i}" for i in range(num_actions)]
        self.step_names = step_names or [f"step_{i:02d}" for i in range(num_steps)]
        self.use_mamba = use_mamba
        self.device = "cpu"

        self._backend = None  # Will be CausalGRUFallback or Mamba2 model
        self._hidden = None   # Recurrent hidden state

    def load_model(self, checkpoint: str = "", device: str = "cpu"):
        """Load temporal model.

        When a checkpoint is provided, reads its metadata (num_actions, num_steps,
        classes) and builds the GRU backend with those dimensions so that
        load_state_dict succeeds even if the experiment config defines more
        action classes than the checkpoint was trained on.  A mapping from
        checkpoint action indices to experiment action names is stored so that
        predict() can translate outputs correctly.
        """
        self.device = device

        # Try Mamba-2 first if requested and GPU available
        if self.use_mamba and device != "cpu":
            try:
                self._load_mamba(checkpoint, device)
                return
            except (ImportError, RuntimeError) as e:
                logger.warning(f"Mamba-2 not available: {e}. Falling back to GRU.")

        # Determine dimensions — use checkpoint metadata when available
        ckpt_num_actions = self.num_actions
        ckpt_num_steps = self.num_steps
        ckpt_input_dim = self.input_dim
        self._ckpt_action_names: Optional[List[str]] = None  # set if checkpoint has fewer classes

        if checkpoint and Path(checkpoint).exists():
            try:
                import torch
                state = torch.load(checkpoint, map_location=device, weights_only=False)
                if isinstance(state, dict):
                    if "num_actions" in state:
                        ckpt_num_actions = state["num_actions"]
                    if "num_steps" in state:
                        ckpt_num_steps = state["num_steps"]
                    if "input_dim" in state:
                        ckpt_input_dim = state["input_dim"]
                    if "classes" in state:
                        self._ckpt_action_names = list(state["classes"])
                        logger.info(
                            f"Checkpoint trained on {ckpt_num_actions} actions: "
                            f"{self._ckpt_action_names}"
                        )
                    logger.info(
                        f"Checkpoint dimensions: input_dim={ckpt_input_dim}, "
                        f"num_actions={ckpt_num_actions}, num_steps={ckpt_num_steps}"
                    )
            except Exception as e:
                logger.warning(f"Could not read checkpoint metadata: {e}")

        # Update input_dim so predict() pads features to the correct size
        self.input_dim = ckpt_input_dim

        # CPU fallback — build with checkpoint-compatible dimensions
        self._backend = CausalGRUFallback(
            input_dim=ckpt_input_dim,
            hidden_dim=self.hidden_dim,
            num_actions=ckpt_num_actions,
            num_steps=ckpt_num_steps,
        )
        self._backend.device = device
        self._backend.build()

        if checkpoint and Path(checkpoint).exists():
            try:
                import torch
                state = torch.load(checkpoint, map_location=device, weights_only=False)
                if isinstance(state, dict) and "model_state_dict" in state:
                    self._backend.model.load_state_dict(state["model_state_dict"])
                logger.info(f"Temporal model loaded from: {checkpoint}")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
        else:
            logger.info(f"Temporal model initialized (GRU fallback, no weights)")

    def _load_mamba(self, checkpoint: str, device: str):
        """Attempt to load Mamba-2 model."""
        try:
            from mamba_ssm import Mamba2
        except ImportError:
            raise ImportError(
                "mamba-ssm not installed. Install with: pip install mamba-ssm causal-conv1d"
            )

        import torch
        import torch.nn as nn

        class Mamba2Temporal(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_actions, num_steps):
                super().__init__()
                self.proj_in = nn.Linear(input_dim, hidden_dim)
                self.mamba = Mamba2(d_model=hidden_dim, d_state=64, d_conv=4, expand=2)
                self.action_head = nn.Linear(hidden_dim, num_actions)
                self.step_head = nn.Linear(hidden_dim, num_steps)
                self.anomaly_head = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, 1),
                    nn.Sigmoid(),
                )
                self.next_action_head = nn.Linear(hidden_dim, num_actions)

            def forward(self, x, hidden=None):
                x = self.proj_in(x)  # (N, T, D)
                x = self.mamba(x)    # (N, T, D)
                last = x[:, -1, :]
                return {
                    "action_logits": self.action_head(last),
                    "step_logits": self.step_head(last),
                    "anomaly_score": self.anomaly_head(last).squeeze(-1),
                    "next_action_logits": self.next_action_head(last),
                    "embeddings": last,
                    "hidden": None,
                }

        model = Mamba2Temporal(
            self.input_dim, self.hidden_dim, self.num_actions, self.num_steps,
        ).to(device)

        if checkpoint and Path(checkpoint).exists():
            state = torch.load(checkpoint, map_location=device)
            if isinstance(state, dict) and "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"])

        model.eval()
        self._backend = type('MambaBackend', (), {
            'model': model, 'device': device,
            'predict': lambda self, features, hidden=None: self._run(features, hidden),
            '_run': lambda self, features, hidden: model(
                __import__('torch').from_numpy(features).unsqueeze(0).float().to(device)
            ),
        })()
        logger.info(f"Mamba-2 temporal model loaded on {device}")

    def predict(self, feature_sequence: np.ndarray) -> TemporalResult:
        """
        Process temporal feature sequence.

        Args:
            feature_sequence: (T, D) multimodal feature array.

        Returns:
            TemporalResult with action, step, anomaly predictions.
        """
        if self._backend is None:
            # No model loaded — return empty result
            return TemporalResult()

        # Ensure correct dimensions
        if feature_sequence.ndim == 1:
            feature_sequence = feature_sequence.reshape(1, -1)

        # Pad/truncate input dim if needed
        T, D = feature_sequence.shape
        if D < self.input_dim:
            pad = np.zeros((T, self.input_dim - D), dtype=np.float32)
            feature_sequence = np.concatenate([feature_sequence, pad], axis=-1)
        elif D > self.input_dim:
            feature_sequence = feature_sequence[:, :self.input_dim]

        try:
            output = self._backend.predict(feature_sequence.astype(np.float32), self._hidden)
        except Exception as e:
            logger.warning(f"Temporal model inference failed: {e}")
            return TemporalResult()

        if output is None:
            return TemporalResult()

        import torch

        # Parse output
        action_logits = output["action_logits"]
        step_logits = output["step_logits"]
        anomaly = output["anomaly_score"]
        next_logits = output["next_action_logits"]
        embeddings = output["embeddings"]

        if hasattr(output.get("hidden", None), 'detach'):
            self._hidden = output["hidden"].detach()

        # Convert to numpy
        action_probs = torch.softmax(action_logits, dim=-1).cpu().numpy().squeeze(0)
        step_probs = torch.softmax(step_logits, dim=-1).cpu().numpy().squeeze(0)
        next_probs = torch.softmax(next_logits, dim=-1).cpu().numpy().squeeze(0)
        anomaly_val = float(anomaly.cpu().numpy()) if anomaly.dim() > 0 else float(anomaly.item())
        emb = embeddings.cpu().numpy().squeeze(0)

        top_action_idx = int(np.argmax(action_probs))
        top_step_idx = int(np.argmax(step_probs))
        top_next_idx = int(np.argmax(next_probs))

        # Use checkpoint action names for index→name mapping when available
        act_names = self._ckpt_action_names if self._ckpt_action_names else self.action_names

        return TemporalResult(
            current_action=act_names[top_action_idx] if top_action_idx < len(act_names) else None,
            current_action_confidence=float(action_probs[top_action_idx]),
            predicted_next_action=act_names[top_next_idx] if top_next_idx < len(act_names) else None,
            next_action_confidence=float(next_probs[top_next_idx]),
            current_step_candidate=self.step_names[top_step_idx] if top_step_idx < len(self.step_names) else None,
            step_confidence=float(step_probs[top_step_idx]),
            anomaly_score=anomaly_val,
            embeddings=emb,
            supporting_evidence={
                "action_entropy": float(-np.sum(action_probs * np.log(action_probs + 1e-10))),
                "step_entropy": float(-np.sum(step_probs * np.log(step_probs + 1e-10))),
            },
        )

    def reset_hidden(self):
        """Reset recurrent hidden state (call between videos)."""
        self._hidden = None

    def get_model_info(self) -> Dict[str, Any]:
        backend_name = "mamba2" if self.use_mamba else "gru"
        if self._backend is not None:
            backend_name = type(self._backend).__name__
        return {
            "name": "ASTRATemporalModel",
            "backend": backend_name,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_actions": self.num_actions,
            "num_steps": self.num_steps,
            "device": self.device,
        }
