import pytest

from src.data.folds import make_image_level_folds
from src.data.indexing import build_labeled_records
from src.data.tiling import build_tile_tables, generate_tile_windows, validate_tile_tables
from src.data.verification import audit_annotations, filter_trusted_records
from src.utils.config import default_dataset_root


def test_generate_tile_windows_covers_small_image_with_single_tile() -> None:
    windows = generate_tile_windows(image_width=300, image_height=410, tile_size=512, overlap=128)
    assert windows == [(0, 0, 300, 410)]


def test_tile_tables_preserve_fold_assignments_and_bounds() -> None:
    dataset_root = default_dataset_root()
    if not dataset_root.exists():
        pytest.skip("Dataset not available locally.")

    records = build_labeled_records(dataset_root)
    audit_df = audit_annotations(records)
    trusted_records = filter_trusted_records(records, audit_df)
    assignments = make_image_level_folds(trusted_records, num_folds=2, random_state=42)
    tile_table, tile_box_table = build_tile_tables(
        trusted_records,
        assignments,
        tile_size=512,
        overlap=128,
        coordinate_mode="center",
        min_box_area=4.0,
        keep_empty_tiles=False,
    )

    assert not tile_table.empty
    assert not tile_box_table.empty
    assert set(tile_table["validation_fold"].unique()) == {0, 1}
    validate_tile_tables(assignments, tile_table, tile_box_table)
    assert (tile_box_table["x1"] >= 0).all()
    assert (tile_box_table["y1"] >= 0).all()
    tile_width_map = tile_table.set_index("tile_id")["tile_width"].to_dict()
    tile_height_map = tile_table.set_index("tile_id")["tile_height"].to_dict()
    assert (tile_box_table["x2"] <= tile_box_table["tile_id"].map(tile_width_map)).all()
    assert (tile_box_table["y2"] <= tile_box_table["tile_id"].map(tile_height_map)).all()
