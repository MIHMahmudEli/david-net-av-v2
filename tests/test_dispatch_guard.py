"""Unit tests for pre-dispatch guards (A1).

Tests:
1. Second dispatch is refused while the first kernel is running/queued.
2. Dispatch is refused if the target namespace is non-empty on the Hub.
3. Dispatch succeeds when all kernels are stopped and namespace is empty.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from src.orchestrator.accounts import KaggleAccount
from src.orchestrator.dispatch_guard import (
    DispatchGuardError,
    assert_pre_dispatch_guards,
    check_namespace_is_empty,
    check_running_kernels_for_namespace,
)


@pytest.fixture
def mock_accounts():
    return [
        KaggleAccount(worker_name="kaggle-worker-7", username="mock-user-7", key="fake-key-7"),
        KaggleAccount(worker_name="kaggle-worker-9", username="mock-user-9", key="fake-key-9"),
    ]


def test_dispatch_refused_when_kernel_running(mock_accounts):
    """Test 1: Second dispatch refused while first is running."""
    # Mock KaggleApi returning "running" for worker 7
    mock_api_7 = MagicMock()
    mock_api_7.kernels_status.return_value = {"status": "running"}

    mock_api_9 = MagicMock()
    mock_api_9.kernels_status.return_value = {"status": "complete"}

    def api_factory(acc):
        return mock_api_7 if acc.worker_name == "kaggle-worker-7" else mock_api_9

    ok, reason = check_running_kernels_for_namespace(
        target_namespace="repro_isolated_s42/",
        target_repo="MIHMahmudEli/davidnet-experiments",
        accounts=mock_accounts,
        kaggle_api_factory=api_factory,
    )
    assert not ok
    assert "RUNNING" in reason
    assert "mock-user-7" in reason


def test_dispatch_refused_when_kernel_queued(mock_accounts):
    """Test 1b: Second dispatch refused while first is queued."""
    mock_api_7 = MagicMock()
    mock_api_7.kernels_status.return_value = {"status": "queued"}

    def api_factory(acc):
        return mock_api_7

    ok, reason = check_running_kernels_for_namespace(
        target_namespace="repro_isolated_s42/",
        target_repo="MIHMahmudEli/davidnet-experiments",
        accounts=[mock_accounts[0]],
        kaggle_api_factory=api_factory,
    )
    assert not ok
    assert "QUEUED" in reason


def test_dispatch_refused_when_namespace_non_empty():
    """Test 2: Dispatch refused if target namespace contains existing files."""
    mock_hf = MagicMock()
    mock_hf.list_repo_files.return_value = [
        "data/SPLITS_SHA256.json",
        "repro_check/experiments/EXP_001_video-probe_s42/metrics/summary.json",
        "repro_check/registry/experiments.json",
    ]

    # Non-empty namespace without allow_resume -> refused
    ok, reason = check_namespace_is_empty(
        target_namespace="repro_check/",
        target_repo="MIHMahmudEli/davidnet-experiments",
        allow_resume=False,
        hf_api=mock_hf,
    )
    assert not ok
    assert "Refused dispatch" in reason
    assert "non-empty" in reason

    # With allow_resume=True -> allowed
    ok_resume, _ = check_namespace_is_empty(
        target_namespace="repro_check/",
        target_repo="MIHMahmudEli/davidnet-experiments",
        allow_resume=True,
        hf_api=mock_hf,
    )
    assert ok_resume


def test_dispatch_allowed_when_clean(mock_accounts):
    """Test 3: Dispatch succeeds when all kernels stopped and namespace is clean."""
    mock_api = MagicMock()
    mock_api.kernels_status.return_value = {"status": "complete"}

    mock_hf = MagicMock()
    mock_hf.list_repo_files.return_value = [
        "data/SPLITS_SHA256.json",
        "experiments/EXP_001_video-probe_s42/metrics/summary.json",
    ]

    # Clean fresh namespace
    ok_empty, _ = check_namespace_is_empty(
        target_namespace="repro_clean_s42/",
        target_repo="MIHMahmudEli/davidnet-experiments",
        allow_resume=False,
        hf_api=mock_hf,
    )
    assert ok_empty

    ok_running, _ = check_running_kernels_for_namespace(
        target_namespace="repro_clean_s42/",
        target_repo="MIHMahmudEli/davidnet-experiments",
        accounts=mock_accounts,
        kaggle_api_factory=lambda acc: mock_api,
    )
    assert ok_running

    # Full assert function passes without exception
    assert_pre_dispatch_guards(
        target_namespace="repro_clean_s42/",
        target_repo="MIHMahmudEli/davidnet-experiments",
        accounts=mock_accounts,
        allow_resume=False,
        kaggle_api_factory=lambda acc: mock_api,
        hf_api=mock_hf,
    )
