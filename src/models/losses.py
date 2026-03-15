"""Target assignment and loss computation helpers."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
    heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    return widths * heights


def pairwise_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute IoU between two sets of boxes."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device, dtype=boxes1.dtype)

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    top_left = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    inter_wh = (bottom_right - top_left).clamp(min=0)
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter_area
    return inter_area / union.clamp(min=1e-6)


def encode_box_targets(gt_boxes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """Encode matched boxes as anchor-relative regression targets."""
    anchor_widths = anchors[:, 2] - anchors[:, 0]
    anchor_heights = anchors[:, 3] - anchors[:, 1]
    anchor_ctr_x = anchors[:, 0] + 0.5 * anchor_widths
    anchor_ctr_y = anchors[:, 1] + 0.5 * anchor_heights

    gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1]
    gt_ctr_x = gt_boxes[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_boxes[:, 1] + 0.5 * gt_heights

    dx = (gt_ctr_x - anchor_ctr_x) / anchor_widths.clamp(min=1e-6)
    dy = (gt_ctr_y - anchor_ctr_y) / anchor_heights.clamp(min=1e-6)
    dw = torch.log(gt_widths.clamp(min=1e-6) / anchor_widths.clamp(min=1e-6))
    dh = torch.log(gt_heights.clamp(min=1e-6) / anchor_heights.clamp(min=1e-6))
    return torch.stack((dx, dy, dw, dh), dim=1)


def decode_box_deltas(box_deltas: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """Decode anchor-relative regression outputs into xyxy boxes."""
    anchor_widths = anchors[:, 2] - anchors[:, 0]
    anchor_heights = anchors[:, 3] - anchors[:, 1]
    anchor_ctr_x = anchors[:, 0] + 0.5 * anchor_widths
    anchor_ctr_y = anchors[:, 1] + 0.5 * anchor_heights

    dx = box_deltas[:, 0]
    dy = box_deltas[:, 1]
    dw = box_deltas[:, 2].clamp(max=math.log(1000.0 / 16.0))
    dh = box_deltas[:, 3].clamp(max=math.log(1000.0 / 16.0))

    pred_ctr_x = dx * anchor_widths + anchor_ctr_x
    pred_ctr_y = dy * anchor_heights + anchor_ctr_y
    pred_w = torch.exp(dw) * anchor_widths
    pred_h = torch.exp(dh) * anchor_heights

    x1 = pred_ctr_x - 0.5 * pred_w
    y1 = pred_ctr_y - 0.5 * pred_h
    x2 = pred_ctr_x + 0.5 * pred_w
    y2 = pred_ctr_y + 0.5 * pred_h
    return torch.stack((x1, y1, x2, y2), dim=1)


def match_anchors(
    anchors: torch.Tensor,
    gt_boxes: torch.Tensor,
    matched_iou_threshold: float = 0.5,
    unmatched_iou_threshold: float = 0.4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign anchors to gt boxes with positive, negative, and ignore states."""
    num_anchors = anchors.shape[0]
    if gt_boxes.numel() == 0:
        matched_indices = torch.zeros((num_anchors,), dtype=torch.long, device=anchors.device)
        match_labels = torch.zeros((num_anchors,), dtype=torch.long, device=anchors.device)
        return matched_indices, match_labels

    ious = pairwise_iou(anchors, gt_boxes)
    matched_ious, matched_indices = ious.max(dim=1)

    match_labels = torch.full((num_anchors,), -1, dtype=torch.long, device=anchors.device)
    match_labels[matched_ious < unmatched_iou_threshold] = 0
    match_labels[matched_ious >= matched_iou_threshold] = 1

    gt_best_anchor_indices = ious.argmax(dim=0)
    match_labels[gt_best_anchor_indices] = 1
    matched_indices[gt_best_anchor_indices] = torch.arange(gt_boxes.shape[0], device=anchors.device)
    return matched_indices, match_labels


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Compute sigmoid focal loss without requiring torchvision ops."""
    prob = logits.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    alpha_factor = alpha * targets + (1 - alpha) * (1 - targets)
    modulating_factor = (1 - p_t) ** gamma
    return (alpha_factor * modulating_factor * ce_loss).sum()


def compute_detection_losses(
    class_logits: torch.Tensor,
    box_deltas: torch.Tensor,
    anchors: torch.Tensor,
    targets: list[dict[str, torch.Tensor | str]],
    num_classes: int,
    matched_iou_threshold: float = 0.5,
    unmatched_iou_threshold: float = 0.4,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    smooth_l1_beta: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute classification and regression losses for one detector batch."""
    device = class_logits.device
    total_cls_loss = torch.tensor(0.0, device=device)
    total_reg_loss = torch.tensor(0.0, device=device)
    total_positive = 0
    total_negative = 0
    total_ignored = 0

    for batch_index, target in enumerate(targets):
        gt_boxes = target["boxes"].to(device)
        gt_labels = target["labels"].to(device)
        matched_indices, match_labels = match_anchors(
            anchors=anchors,
            gt_boxes=gt_boxes,
            matched_iou_threshold=matched_iou_threshold,
            unmatched_iou_threshold=unmatched_iou_threshold,
        )

        cls_targets = torch.zeros((anchors.shape[0], num_classes), device=device, dtype=class_logits.dtype)
        positive_mask = match_labels == 1
        negative_mask = match_labels == 0
        valid_mask = match_labels >= 0

        if positive_mask.any():
            matched_gt_labels = gt_labels[matched_indices[positive_mask]]
            cls_targets[positive_mask, matched_gt_labels] = 1.0
            reg_targets = encode_box_targets(gt_boxes[matched_indices[positive_mask]], anchors[positive_mask])
            total_reg_loss = total_reg_loss + F.smooth_l1_loss(
                box_deltas[batch_index, positive_mask],
                reg_targets,
                beta=smooth_l1_beta,
                reduction="sum",
            )

        total_cls_loss = total_cls_loss + sigmoid_focal_loss(
            class_logits[batch_index, valid_mask],
            cls_targets[valid_mask],
            alpha=focal_alpha,
            gamma=focal_gamma,
        )

        total_positive += int(positive_mask.sum().item())
        total_negative += int(negative_mask.sum().item())
        total_ignored += int((match_labels < 0).sum().item())

    normalizer = max(total_positive, 1)
    classification_loss = total_cls_loss / normalizer
    regression_loss = total_reg_loss / normalizer
    total_loss = classification_loss + regression_loss

    return {
        "classification_loss": classification_loss,
        "regression_loss": regression_loss,
        "total_loss": total_loss,
        "num_positive_anchors": torch.tensor(total_positive, device=device),
        "num_negative_anchors": torch.tensor(total_negative, device=device),
        "num_ignored_anchors": torch.tensor(total_ignored, device=device),
    }
