"""Box decoding, thresholding, and non-max suppression helpers."""

from __future__ import annotations

import torch
from torchvision.ops import nms

from .losses import decode_box_deltas


def clip_boxes_to_image(boxes: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    """Clip xyxy boxes to the provided `(height, width)` image size."""
    height, width = image_size
    clipped = boxes.clone()
    clipped[:, 0::2] = clipped[:, 0::2].clamp(min=0, max=width)
    clipped[:, 1::2] = clipped[:, 1::2].clamp(min=0, max=height)
    return clipped


def decode_predictions(
    class_logits: torch.Tensor,
    box_deltas: torch.Tensor,
    anchors: torch.Tensor,
    image_sizes: list[tuple[int, int]],
    score_threshold: float = 0.05,
    nms_threshold: float = 0.5,
    topk_candidates: int = 1000,
    max_detections: int = 100,
) -> list[dict[str, torch.Tensor]]:
    """Decode detector outputs into per-image boxes, scores, and class labels."""
    batch_boxes = [decode_box_deltas(box_deltas[i], anchors) for i in range(class_logits.shape[0])]
    batch_scores = class_logits.sigmoid()
    predictions: list[dict[str, torch.Tensor]] = []

    for boxes, scores, image_size in zip(batch_boxes, batch_scores, image_sizes):
        boxes = clip_boxes_to_image(boxes, image_size=image_size)
        image_boxes = []
        image_scores = []
        image_labels = []

        for class_index in range(scores.shape[1]):
            class_scores = scores[:, class_index]
            keep = class_scores > score_threshold
            if keep.sum() == 0:
                continue
            class_boxes = boxes[keep]
            class_scores = class_scores[keep]
            if len(class_scores) > topk_candidates:
                class_scores, topk_indices = class_scores.topk(topk_candidates)
                class_boxes = class_boxes[topk_indices]
            keep_indices = nms(class_boxes, class_scores, nms_threshold)
            keep_indices = keep_indices[:max_detections]
            image_boxes.append(class_boxes[keep_indices])
            image_scores.append(class_scores[keep_indices])
            image_labels.append(
                torch.full((len(keep_indices),), class_index, device=class_scores.device, dtype=torch.int64)
            )

        if image_boxes:
            boxes_out = torch.cat(image_boxes, dim=0)
            scores_out = torch.cat(image_scores, dim=0)
            labels_out = torch.cat(image_labels, dim=0)
            top_scores, top_indices = scores_out.sort(descending=True)
            top_indices = top_indices[:max_detections]
            predictions.append(
                {
                    "boxes": boxes_out[top_indices],
                    "scores": top_scores[:max_detections],
                    "labels": labels_out[top_indices],
                }
            )
        else:
            device = scores.device
            predictions.append(
                {
                    "boxes": torch.zeros((0, 4), device=device),
                    "scores": torch.zeros((0,), device=device),
                    "labels": torch.zeros((0,), dtype=torch.int64, device=device),
                }
            )
    return predictions
