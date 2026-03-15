"""Detection metrics and fold aggregation helpers."""

from __future__ import annotations

import math

import torch

from src.models.losses import pairwise_iou


def _compute_average_precision(recalls: torch.Tensor, precisions: torch.Tensor) -> float:
    mrec = torch.cat([torch.tensor([0.0]), recalls, torch.tensor([1.0])])
    mpre = torch.cat([torch.tensor([0.0]), precisions, torch.tensor([0.0])])
    for index in range(mpre.numel() - 1, 0, -1):
        mpre[index - 1] = torch.maximum(mpre[index - 1], mpre[index])
    changing_points = torch.where(mrec[1:] != mrec[:-1])[0]
    ap = torch.sum((mrec[changing_points + 1] - mrec[changing_points]) * mpre[changing_points + 1])
    return float(ap.item())


def _metrics_for_class(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor | str]],
    class_index: int,
    iou_threshold: float,
) -> dict[str, float]:
    detections: list[tuple[float, int, torch.Tensor]] = []
    gt_by_image: dict[int, dict[str, torch.Tensor]] = {}
    total_gt = 0

    for image_index, (prediction, target) in enumerate(zip(predictions, targets)):
        gt_mask = target["labels"].cpu() == class_index
        gt_boxes = target["boxes"].cpu()[gt_mask]
        gt_by_image[image_index] = {
            "boxes": gt_boxes,
            "matched": torch.zeros((len(gt_boxes),), dtype=torch.bool),
        }
        total_gt += len(gt_boxes)

        pred_mask = prediction["labels"].cpu() == class_index
        pred_boxes = prediction["boxes"].cpu()[pred_mask]
        pred_scores = prediction["scores"].cpu()[pred_mask]
        for score, box in zip(pred_scores.tolist(), pred_boxes):
            detections.append((float(score), image_index, box))

    detections.sort(key=lambda item: item[0], reverse=True)
    if total_gt == 0:
        return {
            "precision": math.nan,
            "recall": math.nan,
            "f1": math.nan,
            "ap": math.nan,
            "num_gt": 0,
            "num_predictions": len(detections),
        }

    true_positives = []
    false_positives = []
    for _, image_index, pred_box in detections:
        gt_info = gt_by_image[image_index]
        gt_boxes = gt_info["boxes"]
        if len(gt_boxes) == 0:
            true_positives.append(0.0)
            false_positives.append(1.0)
            continue

        ious = pairwise_iou(pred_box.unsqueeze(0), gt_boxes).squeeze(0)
        best_iou, best_index = ious.max(dim=0)
        if best_iou >= iou_threshold and not gt_info["matched"][best_index]:
            gt_info["matched"][best_index] = True
            true_positives.append(1.0)
            false_positives.append(0.0)
        else:
            true_positives.append(0.0)
            false_positives.append(1.0)

    if not true_positives:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "ap": 0.0,
            "num_gt": total_gt,
            "num_predictions": 0,
        }

    tp_cumsum = torch.tensor(true_positives).cumsum(dim=0)
    fp_cumsum = torch.tensor(false_positives).cumsum(dim=0)
    recalls = tp_cumsum / max(total_gt, 1)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum).clamp(min=1e-6)
    precision = float(precisions[-1].item())
    recall = float(recalls[-1].item())
    f1 = float((2 * precision * recall) / max(precision + recall, 1e-6))
    ap = _compute_average_precision(recalls, precisions)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ap": ap,
        "num_gt": total_gt,
        "num_predictions": len(detections),
    }


def compute_detection_metrics(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor | str]],
    class_names: tuple[str, ...],
    iou_threshold: float = 0.5,
) -> dict[str, object]:
    """Compute per-class and mean detection metrics at a fixed IoU threshold."""
    per_class = {
        class_name: _metrics_for_class(predictions, targets, class_index, iou_threshold)
        for class_index, class_name in enumerate(class_names)
    }
    valid_aps = [values["ap"] for values in per_class.values() if not math.isnan(values["ap"])]
    valid_precisions = [values["precision"] for values in per_class.values() if not math.isnan(values["precision"])]
    valid_recalls = [values["recall"] for values in per_class.values() if not math.isnan(values["recall"])]
    valid_f1 = [values["f1"] for values in per_class.values() if not math.isnan(values["f1"])]
    return {
        "per_class": per_class,
        "mAP@0.5": float(sum(valid_aps) / max(len(valid_aps), 1)),
        "mean_precision": float(sum(valid_precisions) / max(len(valid_precisions), 1)),
        "mean_recall": float(sum(valid_recalls) / max(len(valid_recalls), 1)),
        "mean_f1": float(sum(valid_f1) / max(len(valid_f1), 1)),
    }
