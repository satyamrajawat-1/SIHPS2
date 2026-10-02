"""
ASTRA — Restore Colab Checkpoint Locally

Downloads and restores a Colab-trained checkpoint ZIP to the
local checkpoints directory.

Usage:
    python scripts/restore_colab_checkpoint.py STGCNPP-ASTRA-v2.zip
    python scripts/restore_colab_checkpoint.py path/to/downloaded.zip --target checkpoints/action/STGCNPP-ASTRA-v2
"""

import argparse
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path


def restore_checkpoint(zip_path: str, target_dir: str = None):
    """Restore a Colab checkpoint ZIP to the local checkpoints directory."""
    if not os.path.exists(zip_path):
        print(f"ERROR: ZIP file not found: {zip_path}")
        sys.exit(1)

    # Determine target directory
    if target_dir is None:
        base = Path(zip_path).stem  # e.g., STGCNPP-ASTRA-v2
        target_dir = os.path.join("checkpoints", "action", base)

    print(f"Source:  {zip_path}")
    print(f"Target:  {target_dir}")

    # Create target directory
    os.makedirs(target_dir, exist_ok=True)

    # Extract
    with zipfile.ZipFile(zip_path, 'r') as z:
        # List contents
        names = z.namelist()
        print(f"Files in ZIP: {len(names)}")
        for name in names:
            print(f"  {name}")

        # Extract all files, flattening the directory structure
        for name in names:
            if name.endswith('/'):
                continue  # Skip directories

            # Get just the filename (flatten nested paths)
            basename = os.path.basename(name)
            if not basename:
                continue

            target_path = os.path.join(target_dir, basename)

            # Extract
            with z.open(name) as src, open(target_path, 'wb') as dst:
                shutil.copyfileobj(src, dst)

    print(f"\nRestored files:")
    for f in sorted(os.listdir(target_dir)):
        size = os.path.getsize(os.path.join(target_dir, f))
        print(f"  {f:40s} {size:>12,} bytes")

    # Verify critical files
    critical = ['best.pt']
    for cf in critical:
        path = os.path.join(target_dir, cf)
        if os.path.exists(path):
            print(f"\n✓ {cf} found ({os.path.getsize(path):,} bytes)")
        else:
            print(f"\n✗ WARNING: {cf} not found!")

    # Print metadata if available
    meta_path = os.path.join(target_dir, 'training_metadata.json')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"\n--- Checkpoint Metadata ---")
        print(f"  Model:          {meta.get('model', '?')}")
        print(f"  Version:        {meta.get('version', '?')}")
        print(f"  Dataset:        {meta.get('dataset', '?')}")
        print(f"  Classes:        {meta.get('classes', '?')}")
        print(f"  Best Val Acc:   {meta.get('best_val_acc', '?')}")
        print(f"  Best Epoch:     {meta.get('best_epoch', '?')}")
        print(f"  PyTorch:        {meta.get('pytorch_version', '?')}")
        print(f"  CUDA:           {meta.get('cuda_version', '?')}")
        print(f"  GPU:            {meta.get('gpu', '?')}")
        print(f"  Git Commit:     {meta.get('git_commit', '?')}")
        print(f"  Trained at:     {meta.get('timestamp', '?')}")

    # Verify PyTorch compatibility
    try:
        import torch
        ckpt = torch.load(os.path.join(target_dir, 'best.pt'), map_location='cpu')
        print(f"\n✓ Checkpoint loads successfully on local PyTorch {torch.__version__}")
        print(f"  Epoch: {ckpt.get('epoch', '?')}")
        print(f"  Val Acc: {ckpt.get('val_acc', '?')}")
        print(f"  Classes: {ckpt.get('classes', '?')}")
    except Exception as e:
        print(f"\n✗ WARNING: Could not load checkpoint: {e}")

    print(f"\n{'=' * 60}")
    print(f"Checkpoint restored to: {target_dir}")
    print(f"Ready for local inference.")


def main():
    parser = argparse.ArgumentParser(description="Restore Colab checkpoint locally")
    parser.add_argument("zip_path", help="Path to downloaded checkpoint ZIP")
    parser.add_argument("--target", default=None, help="Target directory (default: auto from ZIP name)")
    args = parser.parse_args()
    restore_checkpoint(args.zip_path, args.target)


if __name__ == "__main__":
    main()
