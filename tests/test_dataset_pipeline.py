import pytest
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader

from src.data.dataset import SmallObjectTileDataset, build_fold_tile_datasets, build_val_transforms, detection_collate_fn
from src.data.folds import make_image_level_folds
from src.data.indexing import build_labeled_records
from src.data.tiling import build_tile_tables
from src.data.verification import audit_annotations, filter_trusted_records
from src.utils.config import default_dataset_root


def test_fold_tile_dataset_returns_padded_tensors_and_targets() -> None:
    dataset_root = default_dataset_root()
    if not dataset_root.exists():
        pytest.skip("Dataset not available locally.")

    records = build_labeled_records(dataset_root)
    audit_df = audit_annotations(records)
    trusted_records = filter_trusted_records(records, audit_df)
    assignments = make_image_level_folds(trusted_records, num_folds=2, random_state=42)
    tile_table, tile_box_table = build_tile_tables(trusted_records, assignments, tile_size=512, overlap=128)
    train_dataset, val_dataset = build_fold_tile_datasets(
        trusted_records,
        tile_table,
        tile_box_table,
        fold_index=0,
        tile_size=512,
    )

    image, target = train_dataset[0]
    assert image.shape == (3, 512, 512)
    assert target["boxes"].shape[1] == 4
    assert target["labels"].dtype == torch.int64
    assert len(train_dataset) > 0
    assert len(val_dataset) > 0


def test_detection_collate_fn_stacks_images() -> None:
    dataset_root = default_dataset_root()
    if not dataset_root.exists():
        pytest.skip("Dataset not available locally.")

    records = build_labeled_records(dataset_root)
    audit_df = audit_annotations(records)
    trusted_records = filter_trusted_records(records, audit_df)
    assignments = make_image_level_folds(trusted_records, num_folds=2, random_state=42)
    tile_table, tile_box_table = build_tile_tables(trusted_records, assignments, tile_size=512, overlap=128)
    train_dataset, _ = build_fold_tile_datasets(
        trusted_records,
        tile_table,
        tile_box_table,
        fold_index=0,
        tile_size=512,
    )
    loader = DataLoader(train_dataset, batch_size=2, shuffle=False, collate_fn=detection_collate_fn)
    images, targets = next(iter(loader))
    assert images.shape == (2, 3, 512, 512)
    assert len(targets) == 2


def test_dataset_handles_empty_targets_with_synthetic_tile(tmp_path) -> None:
    image_path = tmp_path / "synthetic.jpg"
    Image.new("RGB", (300, 300), color=(32, 48, 64)).save(image_path)

    records = pd.DataFrame(
        [
            {
                "image_index": 0,
                "image_path": str(image_path),
                "class_id": 0,
                "class_name": "fish",
                "sample_id": "000",
                "is_few_shot": False,
            }
        ]
    )
    tile_table = pd.DataFrame(
        [
            {
                "tile_id": "0000_000",
                "image_index": 0,
                "validation_fold": 0,
                "class_id": 0,
                "class_name": "fish",
                "sample_id": "000",
                "is_few_shot": False,
                "tile_index": 0,
                "x0": 0,
                "y0": 0,
                "x1": 300,
                "y1": 300,
                "tile_width": 300,
                "tile_height": 300,
                "num_boxes": 0,
                "has_boxes": False,
            }
        ]
    )
    tile_box_table = pd.DataFrame(columns=["tile_id", "x1", "y1", "x2", "y2", "class_id"])

    dataset = SmallObjectTileDataset(records, tile_table, tile_box_table, transform=build_val_transforms(512))
    image, target = dataset[0]

    assert image.shape == (3, 512, 512)
    assert target["boxes"].shape == (0, 4)
    assert target["labels"].shape == (0,)
