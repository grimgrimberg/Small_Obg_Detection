"""Anchor generation helpers."""

from __future__ import annotations

import math

import torch
from torch import nn


class AnchorGenerator(nn.Module):
    """Generate multi-scale anchors for a list of feature maps."""

    def __init__(
        self,
        sizes: tuple[tuple[int, ...], ...],
        aspect_ratios: tuple[tuple[float, ...], ...],
        strides: tuple[int, ...],
    ) -> None:
        super().__init__()
        if not (len(sizes) == len(aspect_ratios) == len(strides)):
            raise ValueError("sizes, aspect_ratios, and strides must have the same length.")
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.strides = strides

    @property
    def num_anchors_per_location(self) -> tuple[int, ...]:
        return tuple(len(level_sizes) * len(level_ratios) for level_sizes, level_ratios in zip(self.sizes, self.aspect_ratios))

    def _generate_base_anchors(
        self,
        sizes: tuple[int, ...],
        aspect_ratios: tuple[float, ...],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        anchors = []
        for size in sizes:
            area = float(size * size)
            for ratio in aspect_ratios:
                width = math.sqrt(area / ratio)
                height = ratio * width
                anchors.append(
                    [
                        -width / 2.0,
                        -height / 2.0,
                        width / 2.0,
                        height / 2.0,
                    ]
                )
        return torch.tensor(anchors, device=device, dtype=dtype)

    def _grid_anchors(
        self,
        feature_shape: tuple[int, int],
        stride: int,
        base_anchors: torch.Tensor,
    ) -> torch.Tensor:
        height, width = feature_shape
        shifts_x = (torch.arange(width, device=base_anchors.device, dtype=base_anchors.dtype) + 0.5) * stride
        shifts_y = (torch.arange(height, device=base_anchors.device, dtype=base_anchors.dtype) + 0.5) * stride
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=-1).reshape(-1, 4)
        return (shifts[:, None, :] + base_anchors[None, :, :]).reshape(-1, 4)

    def forward(
        self,
        feature_maps: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        anchors = []
        for feature_map, sizes, ratios, stride in zip(feature_maps, self.sizes, self.aspect_ratios, self.strides):
            base_anchors = self._generate_base_anchors(
                sizes=sizes,
                aspect_ratios=ratios,
                device=feature_map.device,
                dtype=feature_map.dtype,
            )
            anchors.append(self._grid_anchors(feature_map.shape[-2:], stride, base_anchors))
        return anchors
