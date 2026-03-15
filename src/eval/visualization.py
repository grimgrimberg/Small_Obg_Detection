"""Prediction gallery and plotting helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _unnormalize(image: torch.Tensor) -> torch.Tensor:
    return (image.detach().cpu() * _IMAGENET_STD + _IMAGENET_MEAN).clamp(0, 1)


def save_prediction_gallery(
    images: torch.Tensor,
    targets: list[dict[str, torch.Tensor | str]],
    predictions: list[dict[str, torch.Tensor]],
    output_path: str | Path,
    max_items: int = 4,
) -> Path:
    """Save a side-by-side qualitative gallery of predictions and ground truth."""
    num_items = min(max_items, len(targets))
    fig, axes = plt.subplots(num_items, 2, figsize=(10, 4 * num_items))
    if num_items == 1:
        axes = [axes]

    for row_index in range(num_items):
        image = _unnormalize(images[row_index]).permute(1, 2, 0).numpy()
        target = targets[row_index]
        prediction = predictions[row_index]

        for axis_index, (ax, box_source, title, color) in enumerate(
            [
                (axes[row_index][0], target["boxes"], "Ground Truth", "#7CFC00"),
                (axes[row_index][1], prediction["boxes"], "Predictions", "#FF595E"),
            ]
        ):
            ax.imshow(image)
            for box in box_source:
                x1, y1, x2, y2 = [float(value) for value in box]
                ax.add_patch(
                    Rectangle(
                        (x1, y1),
                        x2 - x1,
                        y2 - y1,
                        fill=False,
                        edgecolor=color,
                        linewidth=1.0,
                    )
                )
            ax.set_title(f"{title}: {target['tile_id']}")
            ax.axis("off")

    fig.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return destination
