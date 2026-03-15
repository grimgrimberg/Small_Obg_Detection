# Stage 4: EDA

## What was implemented
- Box-level EDA expansion in `src/data/eda.py`.
- Summary generation for image sizes, boxes per image, bbox dimensions, clipped area ratios, and class-wise totals.
- A compact matplotlib dashboard for the trusted labeled pool.

## What was run
- Trusted-pool rebuild from the Stage 3 index and audit.
- Box-level table generation with center-based bbox conversion and image-bound clipping.
- EDA summary export and dashboard plotting.
- `pytest -q`

## What was verified
- The trusted pool contains 166 images and 10,464 boxes.
- Trusted image counts are `fish=64`, `fly=50`, `honeybee=50`, `seagull=2`.
- Box counts are `fish=3581`, `fly=3607`, `honeybee=1543`, `seagull=1733`.
- About `5.76%` of boxes require clipping at image boundaries.
- The seagull class is severely imbalanced at `32x` fewer images than the majority class.

## Evidence collected
- Box-level EDA table: `outputs/metrics/stage_04_box_level_eda.csv`
- EDA summary JSON: `outputs/reports/stage_04_eda_summary.json`
- EDA dashboard: `outputs/plots/stage_04_eda_dashboard.png`

## Failure modes discovered
- The original image-size scatter under-represented duplicate resolutions because repeated sizes collapsed onto the same points.

## Patch applied
- Updated the dashboard to use bubble sizes for repeated image resolutions and added an explicit seagull-risk callout on the class-count panel.

## What remains uncertain
- Seagull has many boxes despite only two images, so any fold-level metric for that class will still be highly image-dependent and unstable.
- The long tail in boxes-per-image means the training sampler and loss normalization will need to handle dense-object scenes carefully.

## Safe to proceed
- Yes. The trusted-pool EDA is complete and clearly documents the few-shot and density risks that need to inform fold creation.

## Recommended next step
- Create a leakage-safe 2-fold split on the 166 trusted original images and verify that the seagull class is handled transparently in both folds.
