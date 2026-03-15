# Stage 9: Losses, Decoding, And NMS

## What was implemented
- IoU computation, anchor matching, box encoding, and box decoding in `src/models/losses.py`.
- Sigmoid focal classification loss and Smooth L1 regression loss for anchor-based training.
- Prediction decoding, image clipping, and per-class NMS in `src/models/inference.py`.

## What was run
- Real fold-0 batch forward pass through the custom detector.
- Detection loss computation on real targets.
- Prediction decoding with clipping and NMS.
- `pytest -q tests/test_loss_and_inference.py`

## What was verified
- The loss path produces finite values on a real batch.
- The sample batch produced `378` positive anchors in the latest run.
- Decoding returns per-image prediction dictionaries with clipped `xyxy` boxes, scores, and labels.
- Post-NMS predictions are capped cleanly at `50` detections per image.

## Evidence collected
- Loss/decode summary: `outputs/reports/stage_09_loss_decode_summary.json`

## Failure modes discovered
- The first inference test timed out because untrained logits left too many low-confidence anchors for NMS to process.

## Patch applied
- Added a configurable pre-NMS `topk_candidates` cap and tightened the test threshold so decode/NMS stays tractable even on untrained outputs.

## What remains uncertain
- The decoded predictions are structurally valid but not semantically meaningful yet because the detector has not been trained.
- Positive-anchor counts will vary by batch and may need monitoring once the training loop starts sampling different tile mixtures.

## Safe to proceed
- Yes. The forward, loss, and decode stack is now verified end to end on real data.

## Recommended next step
- Implement the training loop, checkpointing, and evaluation metrics, then run a tiny smoke test before attempting full-fold baselines.
