import math

import torch

from src.eval.metrics import compute_detection_metrics


def test_compute_detection_metrics_reports_perfect_single_class_match() -> None:
    predictions = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([0]),
        }
    ]
    targets = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
            "labels": torch.tensor([0]),
        }
    ]
    metrics = compute_detection_metrics(predictions, targets, class_names=("fish",), iou_threshold=0.5)
    assert math.isclose(metrics["mAP@0.5"], 1.0, rel_tol=1e-6)
    assert math.isclose(metrics["mean_precision"], 1.0, rel_tol=1e-6)
    assert math.isclose(metrics["mean_recall"], 1.0, rel_tol=1e-6)
