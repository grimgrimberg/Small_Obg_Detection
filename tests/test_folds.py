import pytest

from src.data.folds import (
    get_fold_split,
    make_image_level_folds,
    summarize_folds,
    validate_fold_assignments,
)
from src.data.indexing import build_labeled_records
from src.data.verification import audit_annotations, filter_trusted_records
from src.utils.config import default_dataset_root


def test_two_fold_split_keeps_seagull_in_each_validation_fold() -> None:
    dataset_root = default_dataset_root()
    if not dataset_root.exists():
        pytest.skip("Dataset not available locally.")

    records = build_labeled_records(dataset_root)
    audit_df = audit_annotations(records)
    trusted_records = filter_trusted_records(records, audit_df)
    assignments = make_image_level_folds(trusted_records, num_folds=2, random_state=42)
    summary = summarize_folds(trusted_records, assignments)
    validate_fold_assignments(trusted_records, assignments)

    assert summary["folds"]["0"]["val_class_counts"]["seagull"] == 1
    assert summary["folds"]["1"]["val_class_counts"]["seagull"] == 1
    assert summary["folds"]["0"]["val_class_counts"]["fish"] == 32
    assert summary["folds"]["1"]["val_class_counts"]["fish"] == 32


def test_get_fold_split_preserves_image_level_separation() -> None:
    dataset_root = default_dataset_root()
    if not dataset_root.exists():
        pytest.skip("Dataset not available locally.")

    records = build_labeled_records(dataset_root)
    audit_df = audit_annotations(records)
    trusted_records = filter_trusted_records(records, audit_df)
    assignments = make_image_level_folds(trusted_records, num_folds=2, random_state=42)

    train_records, val_records = get_fold_split(trusted_records, assignments, fold_index=0)
    assert set(train_records["image_index"]).isdisjoint(set(val_records["image_index"]))
    assert len(train_records) + len(val_records) == len(trusted_records)
