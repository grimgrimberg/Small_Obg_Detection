# Stage 2: Project Scaffold

## What was implemented
- Repository scaffold with `src/data`, `src/models`, `src/train`, `src/eval`, `src/utils`, `notebooks`, `outputs`, and `tests`.
- Packaging files: `.gitignore`, `pyproject.toml`, and `requirements-colab.txt`.
- A thin Colab-ready notebook skeleton at `notebooks/small_object_detector_workbench.ipynb`.
- Shared config dataclasses and reproducibility helpers in `src/utils/`.
- Import smoke tests in `tests/test_scaffold_imports.py`.

## What was run
- `python -m pip install -e .[dev]`
- `pytest -q`
- Notebook JSON parse check
- Notebook bootstrap probe from the `notebooks/` directory
- Import-time probe for `src.utils.config` and `torch`

## What was verified
- Editable install succeeds.
- The scaffolded package imports from the repository root.
- The notebook is valid JSON and resolves the repository root correctly when launched from `notebooks/`.
- Output directories are created on demand.
- The dataset path can be validated explicitly before later stages run.

## Evidence collected
- `pytest` passed with 4 tests.
- Notebook parse check reported `nbformat=4.5` and 4 cells.
- Dataset validation resolves to the downloaded Kaggle path.
- The config import path dropped from about 28.6 seconds to about 0.33 seconds after hardening.

## Failure modes discovered
- Top-level `src.utils` initialization eagerly imported `torch`, which made config-only imports unnecessarily slow.

## Patch applied
- Reworked `src/utils/__init__.py` so `set_seed()` imports the torch-backed seeding helper lazily.
- Added explicit dataset-availability checks to `ProjectPaths` and surfaced them in the notebook.

## Safe to proceed
- Yes. The scaffold is importable, notebook-friendly, and fast enough to support the next stage.

## Recommended next step
- Implement labeled-image indexing plus visual bbox verification so the pipeline can trust its annotations before EDA and folding.
