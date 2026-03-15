# Stage 5: Fold Splitting

## What was implemented
- Image-level 2-fold splitter in `src/data/folds.py` using stratified original-image assignments.
- Fold summary export and validation-balance plotting.
- Fold validator that checks assignment completeness, train/validation separation, and per-class validation coverage.

## What was run
- Trusted-pool rebuild from the Stage 3 audit.
- 2-fold assignment generation with `random_state=42`.
- Fold summary export and validation-balance plotting.
- `pytest -q`

## What was verified
- All 166 trusted original images received a validation fold assignment.
- Each fold contains exactly 83 validation images and 83 training images.
- Validation class counts are perfectly balanced across folds:
  - `fish=32`
  - `fly=25`
  - `honeybee=25`
  - `seagull=1`
- No image-level leakage occurs between train and validation within a fold.

## Evidence collected
- Fold assignments: `outputs/metrics/stage_05_fold_assignments.csv`
- Fold summary: `outputs/reports/stage_05_fold_summary.json`
- Fold-balance plot: `outputs/plots/stage_05_fold_balance.png`

## Failure modes discovered
- None during the real-data split, but the stage exposed the need for a reusable assignment validator before later tiling and training steps start multiplying examples per image.

## Patch applied
- Added `validate_fold_assignments()` so later stages can prove that original-image leakage has not been introduced.

## What remains uncertain
- The seagull class is technically present in both folds, but each validation fold still rests on exactly one original seagull image, so fold-wise seagull metrics will remain extremely unstable.

## Safe to proceed
- Yes. The leakage-safe original-image folds are saved, balanced, and validated.

## Recommended next step
- Implement overlapping tiling on top of these image-level fold assignments, making sure tile generation never crosses fold boundaries.
