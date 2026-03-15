import pytest
import torch
from torch.utils.data import DataLoader

from src.data.dataset import build_fold_tile_datasets, detection_collate_fn
from src.data.folds import make_image_level_folds
from src.data.indexing import build_labeled_records
from src.data.tiling import build_tile_tables
from src.data.verification import audit_annotations, filter_trusted_records
from src.models.anchors import AnchorGenerator
from src.models.detector import SmallObjectDetector
from src.utils.config import AnchorConfig, default_dataset_root


def test_detector_forward_pass_matches_anchor_count() -> None:
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
    images, _ = next(iter(DataLoader(train_dataset, batch_size=2, shuffle=False, collate_fn=detection_collate_fn)))

    anchor_config = AnchorConfig()
    anchor_generator = AnchorGenerator(
        sizes=anchor_config.sizes,
        aspect_ratios=anchor_config.aspect_ratios,
        strides=anchor_config.strides,
    )
    model = SmallObjectDetector(
        num_classes=4,
        anchor_generator=anchor_generator,
        pretrained_backbone=False,
        neck_channels=128,
        head_channels=128,
    ).eval()

    with torch.no_grad():
        outputs = model(images)

    assert outputs.class_logits.shape[0] == 2
    assert outputs.class_logits.shape[1] == outputs.anchors.shape[0]
    assert outputs.class_logits.shape[2] == 4
    assert outputs.box_deltas.shape == (2, outputs.anchors.shape[0], 4)
