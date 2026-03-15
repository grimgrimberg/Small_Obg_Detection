from pathlib import Path

from src.utils import ExperimentConfig, ProjectPaths, set_seed
from src.utils.config import DEFAULT_CLASSES, FEW_SHOT_CLASSES


def test_project_paths_resolve_from_repo_root() -> None:
    paths = ProjectPaths()
    assert paths.root == Path(__file__).resolve().parents[1]
    assert (paths.root / "src").exists()
    assert paths.dataset_root.name == "Small Object dataset"


def test_project_paths_ensure_creates_output_dirs(tmp_path: Path) -> None:
    paths = ProjectPaths(root=tmp_path, dataset_root=tmp_path / "dataset")
    paths.ensure()
    assert paths.reports_dir.exists()
    assert paths.metrics_dir.exists()
    assert paths.plots_dir.exists()
    assert paths.checkpoints_dir.exists()


def test_validate_dataset_root_raises_clear_error(tmp_path: Path) -> None:
    paths = ProjectPaths(root=tmp_path, dataset_root=tmp_path / "missing-dataset")
    try:
        paths.validate_dataset_root()
    except FileNotFoundError as exc:
        assert "daenys2000/small-object-dataset" in str(exc)
        assert "missing-dataset" in str(exc)
    else:
        raise AssertionError("Expected validate_dataset_root() to raise for a missing dataset.")


def test_experiment_config_marks_seagull_as_few_shot() -> None:
    set_seed(42)
    config = ExperimentConfig()
    assert config.data.classes == DEFAULT_CLASSES
    assert FEW_SHOT_CLASSES == ("seagull",)
    assert "seagull" in config.data.few_shot_classes
