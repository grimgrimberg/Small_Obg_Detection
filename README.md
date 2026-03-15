# Small Object Detection

This repository contains a modular, notebook-friendly implementation of a custom anchor-based object detector for four small-object classes: fish, fly, honeybee, and seagull.

## Current project constraints

- Use original-image indexing before tiling or augmentation.
- Use leakage-safe 2-fold cross-validation on labeled original images.
- Keep the notebook thin and import reusable logic from Python modules.
- Treat `seagull` as a few-shot, high-risk class throughout analysis and evaluation.

## Dataset note

The Kaggle dataset at `daenys2000/small-object-dataset` was audited on 2026-03-15. The labeled bounding-box `.mat` files are present under the dataset's `train/*/gt-bbox` subtrees. The provided `test/*` subtrees contain images and dot maps but no bbox `.mat` annotations, so cross-validation will be built from the 167 labeled original images under `train/*`.

## Quickstart

```bash
python -m pip install -r requirements-colab.txt
pytest
```

## Layout

- `src/data/`: indexing, verification, EDA, folds, tiling, datasets
- `src/models/`: anchors, detector modules, losses, inference utilities
- `src/train/`: training loops and experiment runners
- `src/eval/`: metrics and qualitative evaluation helpers
- `src/utils/`: shared configs and reproducibility utilities
- `notebooks/`: Colab-ready workflow notebook
- `outputs/`: reports, plots, metrics, checkpoints
- `tests/`: smoke tests and focused regression checks
