from pathlib import Path

import numpy as np
import pytest

from src.data.indexing import build_labeled_records, extract_sample_id, xywh_to_xyxy
from src.data.verification import audit_annotations, filter_trusted_records
from src.utils.config import default_dataset_root


def test_extract_sample_id_handles_image_and_annotation_names() -> None:
    assert extract_sample_id("fish065.jpg") == "065"
    assert extract_sample_id("bbox_all_065.mat") == "065"
    assert extract_sample_id(Path("dots003.png")) == "003"


def test_xywh_to_xyxy_supports_center_and_top_left_modes() -> None:
    boxes = np.array([[10, 20, 4, 8]], dtype=np.float32)
    center = xywh_to_xyxy(boxes, coordinate_mode="center")
    top_left = xywh_to_xyxy(boxes, coordinate_mode="top_left")
    assert np.allclose(center, np.array([[8, 16, 12, 24]], dtype=np.float32))
    assert np.allclose(top_left, np.array([[10, 20, 14, 28]], dtype=np.float32))


def test_build_labeled_records_matches_audited_dataset_counts() -> None:
    dataset_root = default_dataset_root()
    if not dataset_root.exists():
        pytest.skip("Dataset not available locally.")

    records = build_labeled_records(dataset_root)
    assert len(records) == 167
    assert int(records["num_boxes"].sum()) == 10464
    assert int((~records["has_boxes"]).sum()) == 1
    assert int(records["is_few_shot"].sum()) == 2


def test_audit_prefers_center_mode_on_real_samples() -> None:
    dataset_root = default_dataset_root()
    if not dataset_root.exists():
        pytest.skip("Dataset not available locally.")

    records = build_labeled_records(dataset_root)
    sampled = (
        records[records["has_boxes"]]
        .groupby("class_name", group_keys=False)
        .head(1)
        .reset_index(drop=True)
    )
    audit_df = audit_annotations(sampled)
    assert (audit_df["dot_coverage_center"] > audit_df["dot_coverage_top_left"]).all()
    assert (audit_df["preferred_coordinate_mode"] == "center").all()


def test_filter_trusted_records_excludes_dot_bbox_mismatches() -> None:
    dataset_root = default_dataset_root()
    if not dataset_root.exists():
        pytest.skip("Dataset not available locally.")

    records = build_labeled_records(dataset_root)
    audit_df = audit_annotations(records)
    trusted = filter_trusted_records(records, audit_df)
    assert len(trusted) == 166
    assert not ((trusted["class_name"] == "fish") & (trusted["sample_id"] == "001")).any()
