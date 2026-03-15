# Stage 10: Tiny Smoke Test

## What was implemented
- End-to-end training/evaluation execution through `fit_detector()` using the real dataloader, detector, losses, decode path, metrics, checkpointing, and gallery export.

## What was run
- A 1-epoch smoke run on fold 0 using 4 train tiles and 4 validation tiles.
- Optimizer: `AdamW`
- Scheduler: `ReduceLROnPlateau`
- Device: `cpu`

## What was verified
- The training loop completes without crashing.
- Validation runs after training and produces losses plus a prediction gallery.
- A checkpoint is written successfully.
- The saved smoke summary contains the full nested history structure needed for later baseline runs.

## Evidence collected
- Smoke summary: `outputs/reports/stage_10_smoke_summary.json`
- Smoke checkpoint: `outputs/checkpoints/stage_10_smoke_best.pt`
- Smoke gallery: `outputs/plots/stage_10_smoke_epoch_01.png`

## Failure modes discovered
- The first smoke attempt completed without leaving a readable artifact trail, which made the result impossible to trust.

## Patch applied
- Reran the smoke stage with tighter subsets plus explicit progress and save logging so the run leaves durable evidence even on CPU.

## What remains uncertain
- The smoke subset covers only fish tiles, so the resulting qualitative predictions are structural checks rather than representative class-performance evidence.
- The detector is still effectively untrained after one tiny epoch, so zero predictions on the gallery are expected and not yet diagnostic.

## Safe to proceed
- Yes. The full training/evaluation stack has now completed at least one bounded end-to-end run with saved outputs.

## Recommended next step
- Run a real fold-0 baseline on the full train/validation tile sets and save fold-wise metrics, plots, and checkpoint artifacts.
