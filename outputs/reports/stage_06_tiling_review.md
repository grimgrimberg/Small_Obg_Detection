# Stage 6: Tiling

## What was implemented
- Overlapping tile-window generation in `src/data/tiling.py`.
- Canonical box-to-tile assignment using validated bbox centers and a best-visible-area tie-breaker.
- Tile-level metadata and tile-local bbox table generation.
- Tile previews for both the original-image grid and a tile crop with local boxes.
- Tile validator that checks fold inheritance, in-bounds local boxes, and one-to-one source-box ownership.

## What was run
- Trusted-pool rebuild plus fold-assignment rebuild.
- Full tile-table generation with `tile_size=512` and `overlap=128`.
- Tile validation.
- Tile preview rendering.
- `pytest -q`

## What was verified
- 276 non-empty tiles were generated from the 166 trusted original images.
- The tile-local bbox table contains exactly 10,464 boxes, matching the trusted original box count after duplicate-assignment hardening.
- Fold inheritance remains balanced after tiling: `138` tiles in fold `0` and `138` tiles in fold `1`.
- Tile-local boxes stay within tile bounds.
- Visual previews show sensible overlap coverage and aligned local boxes.

## Evidence collected
- Tile table: `outputs/metrics/stage_06_tile_table.csv`
- Tile bbox table: `outputs/metrics/stage_06_tile_box_table.csv`
- Tiling summary: `outputs/reports/stage_06_tiling_summary.json`
- Tile grid preview: `outputs/plots/stage_06_tile_grid_preview.png`
- Tile crop preview: `outputs/plots/stage_06_tile_crop_preview.png`

## Failure modes discovered
- The first tiling implementation duplicated source boxes across overlapping tiles because “center inside tile” alone was not a unique ownership rule.

## Patch applied
- Reworked box ownership so each source box is assigned to exactly one tile, preferring the tile with the largest visible clipped area.
- Added `validate_tile_tables()` to catch future fold-inheritance or duplication regressions immediately.

## What remains uncertain
- The current setup drops empty tiles by default, which is efficient for the first baseline but may reduce background-negative diversity; later ablations may want to turn `keep_empty_tiles=True` for comparison.
- Dense seagull scenes produce very crowded tiles, so the dataset pipeline and loss normalization will need to handle high object counts per tile cleanly.

## Safe to proceed
- Yes. Tiling is now fold-safe, box-safe, and visually verified.

## Recommended next step
- Build the dataset class and collate function on top of the validated tile tables, then verify a real dataloader batch before moving into the detector model.
