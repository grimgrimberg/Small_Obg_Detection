# Stage 3: Data Indexing And BBox Verification

## What was implemented
- Original-image records indexing in `src/data/indexing.py`.
- MATLAB bbox loading with flexible `[x, y, w, h] -> [x1, y1, x2, y2]` conversion for both `top_left` and `center` coordinate modes.
- Annotation audit helpers in `src/data/verification.py` for dot coverage, out-of-bounds checks, mismatch detection, and overlay rendering.
- A trusted-records filter that excludes structurally inconsistent samples from the supervised pool.

## What was run
- Full labeled-record indexing over the downloaded Kaggle dataset.
- End-to-end annotation audit against the dot maps.
- Overlay gallery generation for four classes.
- Side-by-side semantics comparison for `fish065`.
- `pytest -q`

## What was verified
- The labeled pool contains 167 original images and 10,464 boxes before filtering.
- `center` semantics vastly outperform `top_left` semantics on dot coverage (`0.9204` vs `0.2744` mean coverage).
- Visual inspection of `fish065` confirms that `center` boxes align with the fish while `top_left` boxes are shifted.
- Exactly one structural annotation mismatch was found: `fish001` has no bbox boxes but a non-empty dot map.
- The trusted supervised pool for later stages contains 166 images after excluding that mismatch.

## Evidence collected
- Records table: `outputs/metrics/stage_03_labeled_records.csv`
- Trusted records table: `outputs/metrics/stage_03_trusted_labeled_records.csv`
- Audit table: `outputs/metrics/stage_03_annotation_audit.csv`
- Audit summary: `outputs/reports/stage_03_annotation_audit_summary.json`
- Overlay gallery: `outputs/plots/stage_03_bbox_verification/`
- Visual semantics comparison: `outputs/plots/stage_03_bbox_verification/fish_065_semantics_comparison.png`

## Failure modes discovered
- The raw bbox coordinates behave like `center_x, center_y, width, height`, not top-left `x, y, w, h`.
- `fish001` is structurally inconsistent between bbox and dot annotations.
- Partial edge overflow is common, so later stages must clip decoded boxes instead of assuming all boxes are fully in-bounds.

## Patch applied
- Added explicit coordinate-mode support and set the project default to `center`.
- Added mismatch detection and a trusted-records filter so noisy samples are excluded before EDA, folds, tiling, and training.
- Added a visual semantics comparison helper so later reviewers can re-check the coordinate interpretation quickly.

## What remains uncertain
- Eleven samples still have center-mode dot coverage below `0.8`, so the dataset likely contains some annotation noise beyond the single hard mismatch.
- Forty-nine samples have more than `10%` of boxes crossing image bounds, which is acceptable for small-object scenes near edges but will need careful clipping in later stages.

## Safe to proceed
- Yes. The coordinate semantics are verified, the single known corrupted sample is isolated, and the trusted labeled pool is now explicit.

## Recommended next step
- Run EDA on the trusted 166-image labeled pool and make the seagull imbalance explicit before building the 2-fold split.
