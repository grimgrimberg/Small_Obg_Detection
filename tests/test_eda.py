from pathlib import Path

import pytest

from src.data.eda import build_box_level_table, summarize_eda
from src.data.indexing import build_labeled_records
from src.data.verification import audit_annotations, filter_trusted_records
from src.utils.config import default_dataset_root


def test_eda_summary_reflects_trusted_pool_counts() -> None:
    dataset_root = default_dataset_root()
    if not dataset_root.exists():
        pytest.skip("Dataset not available locally.")

    records = build_labeled_records(dataset_root)
    audit_df = audit_annotations(records)
    trusted_records = filter_trusted_records(records, audit_df)
    box_table = build_box_level_table(trusted_records)
    summary = summarize_eda(trusted_records, box_table)

    assert summary["trusted_image_count"] == 166
    assert summary["class_image_counts"] == {
        "fish": 64,
        "fly": 50,
        "honeybee": 50,
        "seagull": 2,
    }
    assert summary["seagull_imbalance"]["majority_to_seagull_image_ratio"] >= 25.0
    assert summary["trusted_box_count"] == int(box_table.shape[0])
