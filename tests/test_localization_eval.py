"""Unit tests for the Temporal Action Localization evaluation module."""
import numpy as np
import pytest

from src.eval.localization import (
    segment_iou,
    compute_average_precision,
    compute_average_recall,
    TemporalLocalizationEvaluator,
    check_dataset_status,
)


def test_segment_iou():
    target = np.array([[0.0, 2.0], [3.0, 5.0]])
    candidate = np.array([[0.0, 2.0], [1.0, 2.0], [6.0, 8.0]])
    iou = segment_iou(target, candidate)
    assert iou.shape == (2, 3)
    # target 0 vs candidate 0: exact match -> IoU = 1.0
    assert np.isclose(iou[0, 0], 1.0)
    # target 0 vs candidate 1: [0, 2] vs [1, 2] -> inter=1, union=2 -> IoU = 0.5
    assert np.isclose(iou[0, 1], 0.5)
    # target 0 vs candidate 2: no overlap -> 0.0
    assert np.isclose(iou[0, 2], 0.0)


def test_compute_average_precision():
    gt = {
        "vid1": [(1.0, 3.0)],
        "vid2": [(2.0, 4.0)],
    }
    # Perfect predictions
    preds_perfect = {
        "vid1": [(1.0, 3.0, 0.95)],
        "vid2": [(2.0, 4.0, 0.90)],
    }
    ap50 = compute_average_precision(gt, preds_perfect, iou_threshold=0.50)
    assert np.isclose(ap50, 1.0)

    # Disjoint predictions -> AP = 0.0
    preds_wrong = {
        "vid1": [(5.0, 7.0, 0.95)],
        "vid2": [(8.0, 10.0, 0.90)],
    }
    ap_wrong = compute_average_precision(gt, preds_wrong, iou_threshold=0.50)
    assert np.isclose(ap_wrong, 0.0)


def test_compute_average_recall():
    gt = {
        "vid1": [(1.0, 3.0), (4.0, 6.0)],
    }
    preds = {
        "vid1": [(1.0, 3.0, 0.9), (4.0, 6.0, 0.8)],
    }
    ar = compute_average_recall(gt, preds, top_k=5, iou_thresholds=np.array([0.5, 0.75]))
    assert np.isclose(ar, 1.0)


def test_evaluator_suite():
    evaluator = TemporalLocalizationEvaluator()
    gt = {"v1": [(0.0, 2.0)]}
    preds = {"v1": [(0.0, 2.0, 0.9)]}
    metrics = evaluator.evaluate_stream(gt, preds)
    assert "AP@0.50" in metrics
    assert "AP@0.75" in metrics
    assert "mAP@[0.50:0.95]" in metrics
    assert "AR@50" in metrics
    assert np.isclose(metrics["AP@0.50"], 1.0)


def test_check_dataset_status():
    status = check_dataset_status()
    assert "AV-Deepfake1M" in status
    assert "LAV-DF" in status
    assert status["AV-Deepfake1M"]["status"] == "pending_dataset_acquisition"
