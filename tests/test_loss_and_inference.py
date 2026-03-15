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
from src.models.inference import decode_predictions
from src.models.losses import compute_detection_losses, decode_box_deltas, encode_box_targets
from src.utils.config import AnchorConfig, default_dataset_root


def test_encode_decode_round_trip() -> None:
    anchors = torch.tensor([[0.0, 0.0, 10.0, 10.0], [10.0, 10.0, 30.0, 30.0]])
    gt_boxes = torch.tensor([[1.0, 2.0, 11.0, 12.0], [12.0, 15.0, 28.0, 27.0]])
    deltas = encode_box_targets(gt_boxes, anchors)
    decoded = decode_box_deltas(deltas, anchors)
    assert torch.allclose(decoded, gt_boxes, atol=1e-4)


def test_loss_and_decode_on_real_batch() -> None:
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
    images, targets = next(iter(DataLoader(train_dataset, batch_size=2, shuffle=False, collate_fn=detection_collate_fn)))

    anchor_config = AnchorConfig()
    anchor_generator = AnchorGenerator(anchor_config.sizes, anchor_config.aspect_ratios, anchor_config.strides)
    model = SmallObjectDetector(num_classes=4, anchor_generator=anchor_generator, pretrained_backbone=False).eval()
    outputs = model(images)

    losses = compute_detection_losses(
        outputs.class_logits,
        outputs.box_deltas,
        outputs.anchors,
        targets,
        num_classes=4,
    )
    assert torch.isfinite(losses["total_loss"])
    assert losses["num_positive_anchors"].item() > 0

    image_sizes = [tuple(int(v) for v in target["tile_size"].tolist()) for target in targets]
    predictions = decode_predictions(
        outputs.class_logits,
        outputs.box_deltas,
        outputs.anchors,
        image_sizes=image_sizes,
        score_threshold=0.01,
        nms_threshold=0.5,
        topk_candidates=500,
        max_detections=50,
    )
    assert len(predictions) == 2
    assert all(prediction["boxes"].shape[1] == 4 for prediction in predictions)
