# ASTRA — Google Colab Training Notebooks

This directory contains Jupyter notebooks designed to run on Google Colab for GPU-accelerated model training.

## Notebooks

| Notebook | Model | Status |
|---|---|---|
| `ASTRA_STGCN_training.ipynb` | ST-GCN++ (Action Recognition) | ✅ Ready |
| `ASTRA_GRU_training.ipynb` | Temporal GRU | 🔜 Pending ST-GCN validation |

## Quick Start

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload the notebook
3. Set runtime to **GPU** (Runtime → Change runtime type)
4. Upload the dataset ZIP (generated locally)
5. Run all cells

See [`docs/COLAB_TRAINING_GUIDE.md`](../docs/COLAB_TRAINING_GUIDE.md) for detailed instructions.

## Dataset Preparation (Local)

Before using Colab, generate and package the dataset on the local machine:

```powershell
# Generate synthetic_v2 dataset
python ml/scripts/generate_action_data_v2.py

# Package for upload
python -c "import shutil; shutil.make_archive('astra_dataset_v2', 'zip', '.', 'data/action_v2')"
```

## Checkpoint Restore (Local)

After training on Colab, download the checkpoint ZIP and restore:

```powershell
python scripts/restore_colab_checkpoint.py path/to/STGCNPP-ASTRA-v2.zip
```

## Architecture Decision

- **Local machine** → development, preprocessing, inference, debugging, demo
- **Google Colab** → GPU training (ST-GCN++, Temporal GRU, future models)

The notebooks are self-contained and include the full model definitions. No need to recreate the entire ASTRA environment on Colab.

## Stop Conditions

Do NOT proceed to GRU training until ST-GCN v2 passes:
- [ ] Preprocessing validation
- [ ] No-leakage validation
- [ ] Hard test evaluation
- [ ] Continuous rolling-window test
- [ ] Action transition stability test
