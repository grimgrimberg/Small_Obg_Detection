"""Training and validation loops for the detector."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.eval.metrics import compute_detection_metrics
from src.eval.visualization import save_prediction_gallery
from src.models.inference import decode_predictions
from src.models.losses import compute_detection_losses


def _to_device_targets(targets: list[dict[str, torch.Tensor | str]], device: torch.device) -> list[dict[str, torch.Tensor | str]]:
    moved_targets = []
    for target in targets:
        moved = {}
        for key, value in target.items():
            moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
        moved_targets.append(moved)
    return moved_targets


def _mean_dict(values: list[dict[str, float]]) -> dict[str, float]:
    if not values:
        return {}
    keys = values[0].keys()
    return {key: sum(item[key] for item in values) / len(values) for key in keys}


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int,
    grad_clip_norm: float | None = None,
) -> dict[str, float]:
    model.train()
    batch_logs = []

    for images, targets in loader:
        images = images.to(device)
        device_targets = _to_device_targets(targets, device)
        outputs = model(images)
        losses = compute_detection_losses(
            outputs.class_logits,
            outputs.box_deltas,
            outputs.anchors,
            device_targets,
            num_classes=num_classes,
        )
        optimizer.zero_grad(set_to_none=True)
        losses["total_loss"].backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        batch_logs.append(
            {
                "classification_loss": float(losses["classification_loss"].item()),
                "regression_loss": float(losses["regression_loss"].item()),
                "total_loss": float(losses["total_loss"].item()),
            }
        )

    return _mean_dict(batch_logs)


def evaluate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    num_classes: int,
    class_names: tuple[str, ...],
    score_threshold: float = 0.05,
    nms_threshold: float = 0.5,
    topk_candidates: int = 500,
    max_detections: int = 100,
    gallery_path: str | Path | None = None,
) -> dict[str, object]:
    model.eval()
    batch_logs = []
    all_predictions = []
    all_targets = []
    gallery_payload = None

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            device_targets = _to_device_targets(targets, device)
            outputs = model(images)
            losses = compute_detection_losses(
                outputs.class_logits,
                outputs.box_deltas,
                outputs.anchors,
                device_targets,
                num_classes=num_classes,
            )
            image_sizes = [tuple(int(v) for v in target["tile_size"].tolist()) for target in targets]
            predictions = decode_predictions(
                outputs.class_logits,
                outputs.box_deltas,
                outputs.anchors,
                image_sizes=image_sizes,
                score_threshold=score_threshold,
                nms_threshold=nms_threshold,
                topk_candidates=topk_candidates,
                max_detections=max_detections,
            )

            batch_logs.append(
                {
                    "classification_loss": float(losses["classification_loss"].item()),
                    "regression_loss": float(losses["regression_loss"].item()),
                    "total_loss": float(losses["total_loss"].item()),
                }
            )
            all_predictions.extend(
                [
                    {key: value.detach().cpu() for key, value in prediction.items()}
                    for prediction in predictions
                ]
            )
            all_targets.extend(
                [
                    {key: value.detach().cpu() if isinstance(value, torch.Tensor) else value for key, value in target.items()}
                    for target in targets
                ]
            )
            if gallery_payload is None:
                gallery_payload = (images.detach().cpu(), all_targets[: len(targets)], all_predictions[: len(predictions)])

    metrics = compute_detection_metrics(all_predictions, all_targets, class_names=class_names)
    summary = {
        "losses": _mean_dict(batch_logs),
        "metrics": metrics,
    }
    if gallery_path is not None and gallery_payload is not None:
        save_prediction_gallery(
            images=gallery_payload[0],
            targets=gallery_payload[1],
            predictions=gallery_payload[2],
            output_path=gallery_path,
        )
        summary["gallery_path"] = str(gallery_path)
    return summary


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch_index: int,
    summary: dict[str, object],
    output_path: str | Path,
) -> Path:
    """Persist a checkpoint for later inspection or resumed training."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch_index,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "summary": summary,
        },
        destination,
    )
    return destination


def fit_detector(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    num_classes: int,
    class_names: tuple[str, ...],
    epochs: int,
    checkpoint_dir: str | Path,
    run_name: str,
    grad_clip_norm: float | None = None,
    score_threshold: float = 0.05,
    nms_threshold: float = 0.5,
    topk_candidates: int = 500,
    max_detections: int = 100,
) -> dict[str, object]:
    """Train the detector and keep a compact history plus the best checkpoint."""
    history = []
    best_map = float("-inf")
    best_checkpoint_path = None

    for epoch_index in range(epochs):
        train_summary = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            num_classes=num_classes,
            grad_clip_norm=grad_clip_norm,
        )
        gallery_path = Path(checkpoint_dir).parent / "plots" / f"{run_name}_epoch_{epoch_index + 1:02d}.png"
        val_summary = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            num_classes=num_classes,
            class_names=class_names,
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            topk_candidates=topk_candidates,
            max_detections=max_detections,
            gallery_path=gallery_path,
        )
        if scheduler is not None:
            scheduler.step(val_summary["losses"]["total_loss"])

        epoch_summary = {
            "epoch": epoch_index + 1,
            "train": train_summary,
            "val": val_summary,
        }
        history.append(epoch_summary)

        current_map = val_summary["metrics"]["mAP@0.5"]
        if current_map >= best_map:
            best_map = current_map
            best_checkpoint_path = save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch_index=epoch_index + 1,
                summary=epoch_summary,
                output_path=Path(checkpoint_dir) / f"{run_name}_best.pt",
            )

    return {
        "history": history,
        "best_map": best_map,
        "best_checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path is not None else None,
    }


def save_history(summary: dict[str, object], output_path: str | Path) -> Path:
    """Persist training history to JSON."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return destination
