# Stage 7: Dataset And Collate Function

## What was implemented
- Tile-based dataset in `src/data/dataset.py`.
- Train and validation transform builders with padding to `512x512`, optional color jitter, and train-time horizontal flips.
- Fold-aware dataset builders and a stacking detection collate function.

## What was run
- Full trusted-pool rebuild, fold rebuild, and tile rebuild.
- Fold-0 train/validation dataset construction.
- Real `DataLoader` batch extraction with `batch_size=4`.
- Synthetic empty-target dataset regression test.
- `pytest -q`

## What was verified
- Fold-0 produces `138` train tiles and `138` validation tiles.
- A real batch stacks to `(4, 3, 512, 512)`.
- Target dictionaries preserve variable box counts per tile.
- The empty-target path returns `(0, 4)` boxes and `(0,)` labels without crashing.
- Visual batch preview shows aligned boxes after crop, padding, and collation.

## Evidence collected
- Dataloader summary: `outputs/reports/stage_07_dataloader_summary.json`
- Dataloader batch preview: `outputs/plots/stage_07_dataloader_batch_preview.png`

## Failure modes discovered
- A test incorrectly assumed that `torch.dtype` exposes a `.name` attribute.
- The real tiling setup currently yields no empty tiles with `keep_empty_tiles=False`, so the empty-target path needed explicit synthetic coverage.

## Patch applied
- Fixed the dtype assertion to use `torch.int64`.
- Added a synthetic empty-target regression test so future negative-tile experiments are covered.

## What remains uncertain
- The current train transform is intentionally simple; later ablations may want stronger augmentations, especially for the few-shot seagull class, but only within the training fold.
- Padding preserves variable crop sizes, so later model code must read the true tile size from targets when needed.

## Safe to proceed
- Yes. The dataloader path is verified with real batches and an explicit empty-target regression.

## Recommended next step
- Implement the custom anchor generator, pretrained backbone detector, and inference/loss plumbing, then verify a forward pass on a real batch.
