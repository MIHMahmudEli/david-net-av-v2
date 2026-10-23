"""Regression test for dataset split integrity and evaluation CLI.

Verifies:
1. Manifests exist for train, val, and test splits.
2. All 4 quadrants (RVRA, RVFA, FVRA, FVFA) are present in each split.
3. Strict zero-leakage identity disjointness between train, val, and test splits.
4. Evaluator runs successfully on held-out test split.
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest
import torch

from src.data.datasets import load_manifest
from src.eval.evaluate import evaluate
from src.utils.config import load_config


def test_splits_exist_and_disjoint():
    splits_dir = Path("src/data/splits")
    train_path = splits_dir / "train.jsonl"
    val_path = splits_dir / "val.jsonl"
    test_path = splits_dir / "test.jsonl"

    assert train_path.exists(), "src/data/splits/train.jsonl must exist"
    assert val_path.exists(), "src/data/splits/val.jsonl must exist"
    assert test_path.exists(), "src/data/splits/test.jsonl must exist"

    train_recs = load_manifest(str(train_path))
    val_recs = load_manifest(str(val_path))
    test_recs = load_manifest(str(test_path))

    train_ids = {r["clip_id"] for r in train_recs}
    val_ids = {r["clip_id"] for r in val_recs}
    test_ids = {r["clip_id"] for r in test_recs}

    # Strict disjointness
    assert len(train_ids & val_ids) == 0, f"Train and Val share IDs: {train_ids & val_ids}"
    assert len(train_ids & test_ids) == 0, f"Train and Test share IDs: {train_ids & test_ids}"
    assert len(val_ids & test_ids) == 0, f"Val and Test share IDs: {val_ids & test_ids}"

    # All 4 quadrants represented
    expected_quads = {"RVRA", "RVFA", "FVRA", "FVFA"}
    for name, recs in [("train", train_recs), ("val", val_recs), ("test", test_recs)]:
        quads = {r["quadrant"] for r in recs}
        assert expected_quads.issubset(quads), f"{name} split missing quadrants: {expected_quads - quads}"


def test_evaluate_on_test_manifest():
    cfg = load_config("configs/david_net.yaml")
    cfg.batch_size = 4
    cfg.num_workers = 0

    report = evaluate(cfg, checkpoint="", manifest="src/data/splits/test.jsonl")

    assert report["test_manifest"] == "src/data/splits/test.jsonl"
    assert "video" in report and "auc" in report["video"]
    assert "audio" in report and "auc" in report["audio"]
    assert "quadrant" in report and "macro_f1" in report["quadrant"]
    assert "calibration" in report and "video_ece" in report["calibration"]
    assert report["n"] == 8
