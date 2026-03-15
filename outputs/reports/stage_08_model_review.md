# Stage 8: Model And Anchors

## What was implemented
- Configurable multi-scale anchor generator in `src/models/anchors.py`.
- Pretrained ResNet-18 backbone feature extractor.
- Lightweight FPN-style neck across the `1/8`, `1/16`, and `1/32` feature maps.
- Shared-tower detection head for classification logits and box deltas.
- Custom detector wrapper returning logits, deltas, anchors, and feature-map shapes.

## What was run
- Real fold-0 dataloader batch extraction.
- Detector forward pass on a `(2, 3, 512, 512)` batch.
- `pytest -q tests/test_model_forward.py`

## What was verified
- Feature-map shapes are `[64, 64]`, `[32, 32]`, and `[16, 16]`.
- The model produces `48,384` anchors for a `512x512` tile.
- Classification logits and box deltas both align with the anchor count.
- The forward pass works with the custom detector stack on real tile batches.

## Evidence collected
- Forward summary: `outputs/reports/stage_08_model_forward_summary.json`

## Failure modes discovered
- None in the actual forward path, but the initial head parameters would have started training with overly neutral logits.

## Patch applied
- Added focal-loss-friendly head initialization with a low-positive prior for classification logits and small regression weights.

## What remains uncertain
- The detector is still untrained, so anchor coverage quality and class-score calibration will only become clear once the loss and decode path are exercised.

## Safe to proceed
- Yes. The model and anchor geometry are stable enough to wire into target assignment, losses, and inference.

## Recommended next step
- Implement IoU matching, focal/smooth-L1 losses, box decoding, and NMS, then verify the full loss-and-inference plumbing on a real batch.
