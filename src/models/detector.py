"""Backbone, neck, and detection-head modules."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from .anchors import AnchorGenerator


class ConvNormAct(nn.Module):
    """Small conv block used by the neck and heads."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResNet18Backbone(nn.Module):
    """Expose intermediate feature maps from a pretrained ResNet-18 backbone."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(images)
        x = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c3, c4, c5]


class LightweightFPN(nn.Module):
    """A lightweight top-down neck over three backbone stages."""

    def __init__(self, in_channels: tuple[int, ...], out_channels: int) -> None:
        super().__init__()
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(level_channels, out_channels, kernel_size=1) for level_channels in in_channels]
        )
        self.output_convs = nn.ModuleList(
            [ConvNormAct(out_channels, out_channels, kernel_size=3) for _ in in_channels]
        )

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        lateral_features = [conv(feature) for conv, feature in zip(self.lateral_convs, features)]
        pyramid: list[torch.Tensor] = [lateral_features[-1]]
        for lateral in reversed(lateral_features[:-1]):
            top_down = nn.functional.interpolate(pyramid[0], size=lateral.shape[-2:], mode="nearest")
            pyramid.insert(0, lateral + top_down)
        return [conv(feature) for conv, feature in zip(self.output_convs, pyramid)]


class DetectionHead(nn.Module):
    """Shared conv tower with separate classification and regression outputs."""

    def __init__(
        self,
        in_channels: int,
        num_anchors_per_location: tuple[int, ...],
        num_classes: int,
        head_channels: int,
        num_convs: int = 2,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors_per_location = num_anchors_per_location
        self.shared_towers = nn.ModuleList()
        self.cls_heads = nn.ModuleList()
        self.box_heads = nn.ModuleList()

        for anchors_per_location in num_anchors_per_location:
            tower_layers = []
            current_channels = in_channels
            for _ in range(num_convs):
                tower_layers.append(ConvNormAct(current_channels, head_channels, kernel_size=3))
                current_channels = head_channels
            self.shared_towers.append(nn.Sequential(*tower_layers))
            self.cls_heads.append(nn.Conv2d(head_channels, anchors_per_location * num_classes, kernel_size=3, padding=1))
            self.box_heads.append(nn.Conv2d(head_channels, anchors_per_location * 4, kernel_size=3, padding=1))
        self._init_parameters()

    def _init_parameters(self) -> None:
        prior_probability = 0.01
        bias_value = -math.log((1 - prior_probability) / prior_probability)
        for module in self.shared_towers.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for cls_head in self.cls_heads:
            nn.init.normal_(cls_head.weight, std=0.01)
            nn.init.constant_(cls_head.bias, bias_value)
        for box_head in self.box_heads:
            nn.init.normal_(box_head.weight, std=0.01)
            nn.init.zeros_(box_head.bias)

    def forward(self, features: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        class_logits: list[torch.Tensor] = []
        box_deltas: list[torch.Tensor] = []
        for feature, tower, cls_head, box_head, anchors_per_location in zip(
            features,
            self.shared_towers,
            self.cls_heads,
            self.box_heads,
            self.num_anchors_per_location,
        ):
            hidden = tower(feature)
            cls = cls_head(hidden)
            box = box_head(hidden)
            batch_size, _, height, width = cls.shape
            class_logits.append(
                cls.view(batch_size, anchors_per_location, self.num_classes, height, width)
                .permute(0, 3, 4, 1, 2)
                .reshape(batch_size, -1, self.num_classes)
            )
            box_deltas.append(
                box.view(batch_size, anchors_per_location, 4, height, width)
                .permute(0, 3, 4, 1, 2)
                .reshape(batch_size, -1, 4)
            )
        return class_logits, box_deltas


@dataclass
class DetectorOutput:
    class_logits: torch.Tensor
    box_deltas: torch.Tensor
    anchors: torch.Tensor
    feature_map_shapes: list[tuple[int, int]]


class SmallObjectDetector(nn.Module):
    """Custom anchor-based detector built on a pretrained ResNet-18 backbone."""

    def __init__(
        self,
        num_classes: int,
        anchor_generator: AnchorGenerator,
        pretrained_backbone: bool = True,
        neck_channels: int = 128,
        head_channels: int = 128,
    ) -> None:
        super().__init__()
        self.backbone = ResNet18Backbone(pretrained=pretrained_backbone)
        self.neck = LightweightFPN(in_channels=(128, 256, 512), out_channels=neck_channels)
        self.anchor_generator = anchor_generator
        self.head = DetectionHead(
            in_channels=neck_channels,
            num_anchors_per_location=anchor_generator.num_anchors_per_location,
            num_classes=num_classes,
            head_channels=head_channels,
        )

    def forward(self, images: torch.Tensor) -> DetectorOutput:
        features = self.backbone(images)
        pyramid = self.neck(features)
        class_logits_per_level, box_deltas_per_level = self.head(pyramid)
        anchors_per_level = self.anchor_generator(pyramid)
        return DetectorOutput(
            class_logits=torch.cat(class_logits_per_level, dim=1),
            box_deltas=torch.cat(box_deltas_per_level, dim=1),
            anchors=torch.cat(anchors_per_level, dim=0),
            feature_map_shapes=[tuple(feature.shape[-2:]) for feature in pyramid],
        )
