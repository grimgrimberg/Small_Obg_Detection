"""Configuration dataclasses used across notebook and modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_CLASSES = ("fish", "fly", "honeybee", "seagull")
FEW_SHOT_CLASSES = ("seagull",)


def default_dataset_root() -> Path:
    """Return the default local path created by `kagglehub` for this dataset."""
    return (
        Path.home()
        / ".cache"
        / "kagglehub"
        / "datasets"
        / "daenys2000"
        / "small-object-dataset"
        / "versions"
        / "1"
        / "Small Object dataset"
    )


@dataclass(slots=True)
class ProjectPaths:
    """Resolve project-relative directories while staying notebook-friendly."""

    root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    dataset_root: Path = field(default_factory=default_dataset_root)
    outputs_dir: Path | None = None
    reports_dir: Path | None = None
    metrics_dir: Path | None = None
    plots_dir: Path | None = None
    checkpoints_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.outputs_dir is None:
            self.outputs_dir = self.root / "outputs"
        if self.reports_dir is None:
            self.reports_dir = self.outputs_dir / "reports"
        if self.metrics_dir is None:
            self.metrics_dir = self.outputs_dir / "metrics"
        if self.plots_dir is None:
            self.plots_dir = self.outputs_dir / "plots"
        if self.checkpoints_dir is None:
            self.checkpoints_dir = self.outputs_dir / "checkpoints"

    def ensure(self) -> "ProjectPaths":
        """Create the writable output directories used throughout the project."""
        for path in (
            self.outputs_dir,
            self.reports_dir,
            self.metrics_dir,
            self.plots_dir,
            self.checkpoints_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self

    def dataset_is_available(self) -> bool:
        """Return whether the expected dataset root exists on disk."""
        return self.dataset_root.exists()

    def validate_dataset_root(self) -> Path:
        """Fail fast with a clear message when the dataset is not available."""
        if not self.dataset_is_available():
            raise FileNotFoundError(
                "Dataset root does not exist. Download the Kaggle dataset with "
                "`kagglehub.dataset_download('daenys2000/small-object-dataset')` "
                f"or point `ProjectPaths.dataset_root` to the extracted 'Small Object dataset' directory. "
                f"Expected path: {self.dataset_root}"
            )
        return self.dataset_root


@dataclass(slots=True)
class DataConfig:
    """Dataset and split-level settings."""

    classes: tuple[str, ...] = DEFAULT_CLASSES
    few_shot_classes: tuple[str, ...] = FEW_SHOT_CLASSES
    bbox_mat_key: str = "bbox_all"
    bbox_format: str = "xywh"
    bbox_coordinate_mode: str = "center"
    num_folds: int = 2
    fold_seed: int = 42
    use_labeled_subtree_only: bool = True


@dataclass(slots=True)
class TileConfig:
    """Small-object tiling defaults."""

    tile_size: int = 512
    overlap: int = 128
    min_box_area: float = 4.0
    keep_empty_tiles: bool = False


@dataclass(slots=True)
class AnchorConfig:
    """Anchor generator defaults tuned for small objects."""

    sizes: tuple[tuple[int, ...], ...] = ((8, 12, 16), (20, 28, 36), (48, 64, 96))
    aspect_ratios: tuple[tuple[float, ...], ...] = (
        (0.5, 1.0, 2.0),
        (0.5, 1.0, 2.0),
        (0.5, 1.0, 2.0),
    )
    strides: tuple[int, ...] = (8, 16, 32)


@dataclass(slots=True)
class ModelConfig:
    """Detector architecture defaults."""

    backbone_name: str = "resnet18"
    pretrained: bool = True
    feature_channels: tuple[int, ...] = (128, 256, 512)
    neck_channels: int = 128
    head_channels: int = 128


@dataclass(slots=True)
class TrainConfig:
    """Training-loop defaults for reproducible baseline runs."""

    batch_size: int = 4
    num_workers: int = 0
    epochs: int = 3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    smooth_l1_beta: float = 1.0
    grad_clip_norm: float = 5.0
    scheduler_patience: int = 2
    scheduler_factor: float = 0.5


@dataclass(slots=True)
class ExperimentConfig:
    """Bundle the main config groups into a single notebook-friendly object."""

    paths: ProjectPaths = field(default_factory=ProjectPaths)
    data: DataConfig = field(default_factory=DataConfig)
    tiles: TileConfig = field(default_factory=TileConfig)
    anchors: AnchorConfig = field(default_factory=AnchorConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
