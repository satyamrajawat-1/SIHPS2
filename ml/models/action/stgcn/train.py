"""
ASTRA — ST-GCN++ Action Recognition Training

Trains ST-GCN++ on the skeleton action dataset for the 5-action vocabulary.

Usage:
    python -m ml.models.action.stgcn.train \
        --data data/action/ \
        --output checkpoints/action/STGCNPP-ASTRA-v1/ \
        --epochs 50 --lr 0.01 --batch_size 32
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class SkeletonActionDataset(Dataset):
    """PyTorch dataset for skeleton action sequences."""

    def __init__(self, data_dir: str):
        data = np.load(os.path.join(data_dir, "samples.npz"))
        self.sequences = torch.from_numpy(data["sequences"])  # (N, T, J, C)
        self.labels = torch.from_numpy(data["labels"]).long()  # (N,)

        with open(os.path.join(data_dir, "labels.json")) as f:
            self.meta = json.load(f)

        logger.info(f"Loaded {len(self.sequences)} samples from {data_dir}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # ST-GCN++ inner model expects (N, C, T, V)
        seq = self.sequences[idx]  # (T, J, C)
        # Reshape to (C, T, V)
        seq = seq.permute(2, 0, 1)  # (C, T, V)
        return seq, self.labels[idx]


def train_stgcn(args):
    """Train ST-GCN++ model."""
    from ml.models.action.stgcn.model import STGCNPP

    device = torch.device(args.device)
    logger.info(f"Device: {device}")

    # Load datasets
    train_ds = SkeletonActionDataset(os.path.join(args.data, "train"))
    val_ds = SkeletonActionDataset(os.path.join(args.data, "val"))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0)

    # Load dataset info
    with open(os.path.join(args.data, "dataset_info.json")) as f:
        dataset_info = json.load(f)

    num_classes = dataset_info["n_classes"]
    action_classes = list(dataset_info["classes"].keys())

    logger.info(f"Classes ({num_classes}): {action_classes}")
    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Create model — STGCNPP is a wrapper, use its _build_model for the nn.Module
    wrapper = STGCNPP(
        action_classes=action_classes,
        num_classes=num_classes,
    )
    model = wrapper._build_model().to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model params: {n_params:,} total, {n_trainable:,} trainable")

    # Optimizer and scheduler
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr,
        momentum=0.9, weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Training state
    os.makedirs(args.output, exist_ok=True)
    best_val_acc = 0.0
    best_epoch = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # Resume if checkpoint exists
    start_epoch = 0
    resume_path = os.path.join(args.output, "latest.pt")
    if args.resume and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0)
        history = ckpt.get("history", history)
        logger.info(f"Resumed from epoch {start_epoch}")

    # Training loop
    t_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            _, preds = logits.max(dim=1)
            train_correct += (preds == batch_y).sum().item()
            train_total += batch_x.size(0)

        scheduler.step()

        train_loss /= max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                _, preds = logits.max(dim=1)
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_x.size(0)

        val_loss /= max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        lr = scheduler.get_last_lr()[0]
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} | "
            f"lr={lr:.6f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_path = os.path.join(args.output, "best.pt")
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "val_loss": val_loss,
                "classes": action_classes,
                "num_classes": num_classes,
            }, best_path)
            logger.info(f"  ★ New best: val_acc={val_acc:.4f} (epoch {epoch+1})")

        # Save latest (for resume)
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_acc": best_val_acc,
            "history": history,
        }, resume_path)

    elapsed = time.time() - t_start
    logger.info(f"\nTraining complete in {elapsed:.1f}s")
    logger.info(f"Best val accuracy: {best_val_acc:.4f} at epoch {best_epoch}")

    # Save training metadata
    metadata = {
        "model": "STGCNPP",
        "version": "ASTRA-v1",
        "dataset": dataset_info["name"],
        "data_type": dataset_info["data_type"],
        "classes": action_classes,
        "num_classes": num_classes,
        "input_shape": dataset_info["tensor_shape"],
        "n_joints": dataset_info["n_joints"],
        "n_frames": dataset_info["n_frames"],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "optimizer": "SGD",
        "scheduler": "CosineAnnealing",
        "seed": args.seed,
        "device": args.device,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "training_time_seconds": elapsed,
        "n_params": n_params,
        "timestamp": datetime.now().isoformat(),
        "history": history,
    }

    meta_path = os.path.join(args.output, "training_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info(f"Metadata saved to: {meta_path}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Train ST-GCN++ for ASTRA")
    parser.add_argument("--data", default="data/action", help="Action dataset directory")
    parser.add_argument("--output", default="checkpoints/action/STGCNPP-ASTRA-v1",
                        help="Output directory")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()
    train_stgcn(args)


if __name__ == "__main__":
    main()
