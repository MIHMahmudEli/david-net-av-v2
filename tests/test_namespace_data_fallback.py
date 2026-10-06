"""Test namespace fallback for data splits and feature identity in Session driver."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
from pathlib import Path
import pytest

from src.pipeline.driver import Session


def test_session_namespace_fallback(tmp_path):
    user_config = {
        "mode": "full",
        "seeds": [42],
        "project": {"hf_repo": "test/repo", "hf_repo_type": "model", "hf_private": True},
        "session": {"worker_name": "test-worker", "safety_minutes": 5, "lease_minutes": 45, "heartbeat_minutes": 5},
    }

    session = Session(user_config, repo_dir=str(tmp_path), namespace="repro_check/")
    assert session.ns == "repro_check/"

    # Mock store
    mock_store = MagicMock()
    session.store = mock_store

    # 1. Test feature_identity fallback to root
    def mock_read_json(path, revision=None):
        if path == "repro_check/data/feature_set.json":
            return None
        if path == "data/feature_set.json":
            return {
                "created_at": "2026-09-26T23:31:02Z",
                "feature_set_id": "fs-a3b70d8acd",
                "revisions": {"video": "v1", "audio": "a1"},
            }
        return None

    mock_store.read_json.side_effect = mock_read_json

    fsid = session.feature_identity()
    assert fsid == "fs-a3b70d8acd"
    assert session.fsid == "fs-a3b70d8acd"
    assert session.revisions == {"video": "v1", "audio": "a1"}
