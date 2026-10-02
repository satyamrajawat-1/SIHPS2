"""
ASTRA — YOLO11-S Training Script

Fine-tune YOLO11-S on experiment-specific objects.

Usage:
    python -m ml.models.detection.train --config configs/detection.yaml
    python -m ml.models.detection.train --data data/datasets/sample/detection.yaml --epochs 100
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train(
    data_config: str,
    pretrained: str = "yolo11s.pt",
    epochs: int = 100,
    img_size: int = 640,
    batch_size: int = 16,
    lr: float = 0.01,
    device: str = "auto",
    project: str = "experiments",
    name: str = None,
    resume: bool = False,
    seed: int = 42,
    patience: int = 20,
    save_period: int = 10,
    workers: int = 4,
    amp: bool = True,
):
    """
    Train YOLO11-S on custom dataset.

    Args:
        data_config: Path to YOLO dataset YAML (images, labels, classes).
        pretrained: Pretrained weights path or model name.
        epochs: Number of training epochs.
        img_size: Training image size.
        batch_size: Batch size.
        lr: Initial learning rate.
        device: Device string.
        project: Output project directory.
        name: Run name.
        resume: Whether to resume from last checkpoint.
        seed: Random seed.
        patience: Early stopping patience (epochs).
        save_period: Save checkpoint every N epochs.
        workers: DataLoader workers.
        amp: Automatic mixed precision.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    if name is None:
        name = f"yolo11s_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    logger.info(f"Training YOLO11-S")
    logger.info(f"  Data: {data_config}")
    logger.info(f"  Pretrained: {pretrained}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Image size: {img_size}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Seed: {seed}")

    # Load model
    model = YOLO(pretrained)

    # Train
    results = model.train(
        data=data_config,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        lr0=lr,
        device=device if device != "auto" else None,
        project=project,
        name=name,
        resume=resume,
        seed=seed,
        patience=patience,
        save_period=save_period,
        workers=workers,
        amp=amp,
        verbose=True,
        exist_ok=True,
    )

    # Save training metadata
    # Ultralytics may save to a different path than project/name
    actual_save_dir = Path(str(results.save_dir)) if hasattr(results, 'save_dir') else Path(project) / name
    run_dir = actual_save_dir

    metadata = {
        "model": "YOLO11-S",
        "pretrained": pretrained,
        "data_config": data_config,
        "epochs": epochs,
        "img_size": img_size,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "patience": patience,
        "amp": amp,
        "timestamp": datetime.now().isoformat(),
        "run_dir": str(run_dir),
    }

    os.makedirs(run_dir, exist_ok=True)
    metadata_path = run_dir / "training_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Training complete. Results saved to: {run_dir}")
    return results


def validate(
    model_path: str,
    data_config: str,
    img_size: int = 640,
    batch_size: int = 16,
    device: str = "auto",
    split: str = "val",
):
    """Validate a trained model and report metrics."""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed.")
        sys.exit(1)

    model = YOLO(model_path)
    results = model.val(
        data=data_config,
        imgsz=img_size,
        batch=batch_size,
        device=device if device != "auto" else None,
        split=split,
        verbose=True,
    )

    # Extract metrics
    metrics = {
        "mAP50": float(results.box.map50) if hasattr(results.box, 'map50') else None,
        "mAP50_95": float(results.box.map) if hasattr(results.box, 'map') else None,
        "precision": float(results.box.mp) if hasattr(results.box, 'mp') else None,
        "recall": float(results.box.mr) if hasattr(results.box, 'mr') else None,
    }

    logger.info(f"Validation Results:")
    for k, v in metrics.items():
        if v is not None:
            logger.info(f"  {k}: {v:.4f}")

    return metrics


def export_model(
    model_path: str,
    export_format: str = "onnx",
    img_size: int = 640,
    half: bool = False,
    int8: bool = False,
):
    """Export model to ONNX or other formats."""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed.")
        sys.exit(1)

    model = YOLO(model_path)
    export_path = model.export(
        format=export_format,
        imgsz=img_size,
        half=half,
        int8=int8,
    )
    logger.info(f"Model exported to: {export_path}")
    return export_path


def main():
    parser = argparse.ArgumentParser(description="ASTRA YOLO11-S Training")
    sub = parser.add_subparsers(dest="command", help="Command")

    # Train
    train_p = sub.add_parser("train", help="Train model")
    train_p.add_argument("--data", required=True, help="Dataset YAML path")
    train_p.add_argument("--pretrained", default="yolo11s.pt", help="Pretrained weights")
    train_p.add_argument("--epochs", type=int, default=100)
    train_p.add_argument("--img_size", type=int, default=640)
    train_p.add_argument("--batch_size", type=int, default=16)
    train_p.add_argument("--lr", type=float, default=0.01)
    train_p.add_argument("--device", default="auto")
    train_p.add_argument("--project", default="experiments")
    train_p.add_argument("--name", default=None)
    train_p.add_argument("--resume", action="store_true")
    train_p.add_argument("--seed", type=int, default=42)
    train_p.add_argument("--patience", type=int, default=20)

    # Validate
    val_p = sub.add_parser("val", help="Validate model")
    val_p.add_argument("--model", required=True, help="Model checkpoint")
    val_p.add_argument("--data", required=True, help="Dataset YAML")
    val_p.add_argument("--img_size", type=int, default=640)
    val_p.add_argument("--device", default="auto")

    # Export
    exp_p = sub.add_parser("export", help="Export model")
    exp_p.add_argument("--model", required=True, help="Model checkpoint")
    exp_p.add_argument("--format", default="onnx", choices=["onnx", "torchscript", "engine"])
    exp_p.add_argument("--img_size", type=int, default=640)
    exp_p.add_argument("--half", action="store_true")

    args = parser.parse_args()

    if args.command == "train":
        train(
            data_config=args.data, pretrained=args.pretrained,
            epochs=args.epochs, img_size=args.img_size,
            batch_size=args.batch_size, lr=args.lr,
            device=args.device, project=args.project,
            name=args.name, resume=args.resume,
            seed=args.seed, patience=args.patience,
        )
    elif args.command == "val":
        validate(args.model, args.data, args.img_size, device=args.device)
    elif args.command == "export":
        export_model(args.model, args.format, args.img_size, args.half)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
