"""Shared project utilities."""

from .config import (
    AnchorConfig,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    ProjectPaths,
    TileConfig,
    TrainConfig,
)


def set_seed(seed: int) -> None:
    """Import the heavier torch-based seeding helper only when needed."""
    from .reproducibility import set_seed as _set_seed

    _set_seed(seed)

__all__ = [
    "AnchorConfig",
    "DataConfig",
    "ExperimentConfig",
    "ModelConfig",
    "ProjectPaths",
    "TileConfig",
    "TrainConfig",
    "set_seed",
]
