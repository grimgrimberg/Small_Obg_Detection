# Stage 11: Colab Training Notebook

## What was implemented
- A dedicated Colab training notebook at `notebooks/colab_train_detector.ipynb`.
- Colab-specific setup for:
  - cloning the repo from GitHub
  - optional Drive-backed outputs
  - dependency installation
  - Kaggle dataset download
  - full pipeline rebuild from records through tiling
  - fold training via `fit_detector()`

## What was run
- Notebook JSON parse check
- Structural audit for:
  - repo-clone setup cell
  - GPU runtime guard
  - training cell calling `fit_detector()`

## What was verified
- The notebook is valid `nbformat 4.5`.
- It contains the expected setup, pipeline, dataloader, and training cells.
- It explicitly rejects TPU/CPU misuse by requiring a GPU runtime.
- It explicitly rejects the placeholder GitHub URL before any heavy setup work starts.

## Evidence collected
- Notebook path: `notebooks/colab_train_detector.ipynb`

## Failure modes discovered
- Without an early guard, the placeholder `REPO_URL` would have caused a later and noisier setup failure in Colab.

## Patch applied
- Added an explicit `REPO_URL` guard in the setup cell so the notebook fails fast with a concrete instruction.

## What remains uncertain
- The notebook assumes the repository is available on GitHub; if the repo stays local-only, the user will need to push it first or adapt the setup cell to unzip an archive from Drive.
- Training duration in Colab will depend on the selected runtime and memory limits; the notebook defaults are reasonable for a T4 GPU but may still need tuning.

## Safe to proceed
- Yes. The Colab notebook is structurally ready to use.

## Recommended next step
- Push the repo to GitHub, open the new notebook in Colab, fill in `REPO_URL`, and run it on a GPU runtime.
