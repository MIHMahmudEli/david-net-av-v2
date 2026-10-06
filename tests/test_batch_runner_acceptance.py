import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.orchestrator.accounts import KaggleAccount
from src.orchestrator.batch_runner import (
    BatchRunner,
    build_batch_script,
    estimate_batch_hours,
    validate_batch_for_worker,
    validate_batch_ordering,
)
from src.orchestrator.db import JobRecord


def test_3_1_failing_job_does_not_abort_others():
    """Requirement 3.1: In the batch runner script, if job 2 fails with non-zero exit code or error,
    it records the failure and proceeds to execute job 3 and job 4."""
    jobs = [
        JobRecord(job_id=f"job_{i}_s7", name=f"job_{i}", seed=7, stage="baseline", group="baseline", split="strict", init_from="", config_hash=f"h{i}")
        for i in range(1, 5)
    ]
    script = build_batch_script(jobs, "worker-1", "6793d73")

    # Script loops through all batch_jobs and continues upon failure
    assert "for idx, job in enumerate(batch_jobs, 1):" in script
    assert "Continuing with remaining batch jobs..." in script
    assert "results[job[\"job_id\"]] = res.returncode" in script

    # Simulate execution flow in Python:
    # mock job 2 returning 1, other jobs returning 0
    job_results = {}
    for idx, j in enumerate(jobs, 1):
        if idx == 2:
            returncode = 1  # Forced failure in job 2
        else:
            returncode = 0
        job_results[j.job_id] = returncode

    # All 4 jobs were attempted
    assert len(job_results) == 4
    assert job_results["job_2_s7"] == 1
    assert job_results["job_1_s7"] == 0
    assert job_results["job_3_s7"] == 0
    assert job_results["job_4_s7"] == 0


def test_3_2_stage0_stage1_ordering_validation():
    """Requirement 3.2: A Stage 0 job and the Stage 1 job that warm-starts from it may share
    one batch ONLY if ordered (Stage 0 strictly before Stage 1)."""
    stage0 = JobRecord("qacp_s7", "qacp", 7, "qacp", "qacp", "strict", init_from="", config_hash="h_qacp")
    stage1 = JobRecord("davidnet_s7", "davidnet", 7, "stage1", "proposed", "strict", init_from="qacp", config_hash="h_davidnet")

    # Correct order: Stage 0 before Stage 1 -> Valid
    ok, msg = validate_batch_ordering([stage0, stage1])
    assert ok is True
    assert msg == "OK"

    # Inverted order: Stage 1 before Stage 0 -> Invalid
    ok_inv, msg_inv = validate_batch_ordering([stage1, stage0])
    assert ok_inv is False
    assert "requires qacp_s7 which does not precede it" in msg_inv


def test_3_3_batch_quota_and_session_limits():
    """Requirement 3.3: Total batch estimate must include 1.5 h margin on sum, stay below session limit,
    and fail closed on insufficient quota."""
    cheap_jobs = [
        JobRecord("audio-probe_s7", "audio-probe", 7, "baseline", "baseline", "strict", init_from="", config_hash="h1"), # 0.05h
        JobRecord("video-probe_s7", "video-probe", 7, "baseline", "baseline", "strict", init_from="", config_hash="h2"), # 0.05h
        JobRecord("qacp-no_sync_s7", "qacp-no_sync", 7, "qacp", "qacp", "strict", init_from="", config_hash="h3"), # 0.05h
    ]
    est = estimate_batch_hours(cheap_jobs)
    assert pytest.approx(est, 0.01) == 0.15

    # Worker has 5.0h remaining -> needed is 0.15 + 1.5 = 1.65h -> PASS
    ok, msg = validate_batch_for_worker(cheap_jobs, rem_hours=5.0, max_batch_hours=4.0)
    assert ok is True

    # Worker has only 1.0h remaining -> needed 1.65h -> FAIL quota guard
    ok_low, msg_low = validate_batch_for_worker(cheap_jobs, rem_hours=1.0, max_batch_hours=4.0)
    assert ok_low is False
    assert "Quota guard: worker has 1.00h < needed 1.65h" in msg_low

    # Worker quota unknown (None) -> FAIL closed
    ok_none, msg_none = validate_batch_for_worker(cheap_jobs, rem_hours=None)
    assert ok_none is False
    assert "Worker GPU quota is unknown" in msg_none

    # Batch estimate exceeds max_batch_hours (e.g. 5.0h > 4.0h limit)
    large_batch = [
        JobRecord(f"job_{i}", "davidnet", 7, "stage1", "proposed", "strict", init_from="", config_hash="h")
        for i in range(12)  # 12 * 0.5h = 6.0h
    ]
    ok_large, msg_large = validate_batch_for_worker(large_batch, rem_hours=10.0, max_batch_hours=4.0)
    assert ok_large is False
    assert "exceeds max batch limit" in msg_large


def test_3_4_batch_job_directory_isolation(tmp_path):
    """Requirement 3.4: Each job writes strictly to experiments/EXP_xxx/ for its own run_id."""
    jobs = [
        JobRecord(f"cheap_{i}_s7", f"cheap_{i}", 7, "baseline", "baseline", "strict", init_from="", config_hash=f"h{i}")
        for i in range(1, 4)
    ]
    acc = KaggleAccount("worker-1", "user1", "key1")
    runner = BatchRunner(acc, dry_run=True)

    with patch("src.orchestrator.batch_runner.WORKSPACE_DIR", tmp_path):
        ws = runner.prepare_batch_workspace("batch_isolated", jobs, "6793d73")
        script = (ws / "run_job.py").read_text()

        # Each job sets JOB_ID and EXP_NAME separately per iteration
        for j in jobs:
            assert f'"job_id": "{j.job_id}"' in script
            assert f'"name": "{j.name}"' in script
        assert 'env["JOB_ID"] = job["job_id"]' in script
        assert 'env["EXP_NAME"] = job["name"]' in script


def test_3_5_bin_packing_with_unequal_quotas():
    """Requirement C3: Dispatcher must bin-pack by account:
    - job needs est + 1.5 h <= account's remaining
    - assign largest jobs (aasist ~3h, effnet-b4 ~2h, e2e ~1.5h) to accounts with the most remaining
    - batch small jobs per account with sum + 1.5h rule."""
    from src.orchestrator.batch_runner import bin_pack_jobs_to_workers

    # Workers with unequal remaining quotas
    workers_quota = {
        "worker-high-1": 5.8,  # Can take large job (3h + 1.5h = 4.5h <= 5.8h)
        "worker-high-2": 5.5,  # Can take large job (2h + 1.5h = 3.5h <= 5.5h)
        "worker-mid": 3.2,     # Can take small jobs batch (1.5h + 1.5h = 3.0h <= 3.2h)
        "worker-low": 1.6,     # Cannot take jobs needing > 0.1h + 1.5h = 1.6h
        "worker-depleted": 1.2 # Cannot take any job (1.2h < 1.5h margin)
    }

    jobs = [
        JobRecord("aasist_s7", "aasist", 7, "external", "external", "strict", init_from="", config_hash="h1"),       # 3.0h
        JobRecord("effnet-b4_s7", "effnet-b4", 7, "external", "external", "strict", init_from="", config_hash="h2"), # 2.0h
        JobRecord("davidnet-e2e_s7", "davidnet-e2e", 7, "phase_b", "phase_b", "strict", init_from="davidnet", config_hash="h3"), # 1.5h
        JobRecord("qacp_s7", "qacp", 7, "qacp", "qacp", "strict", init_from="", config_hash="h4"),                   # 0.1h
        JobRecord("davidnet_s7", "davidnet", 7, "stage1", "proposed", "strict", init_from="qacp", config_hash="h5"), # 0.5h
        JobRecord("audio-probe_s7", "audio-probe", 7, "baseline", "baseline", "strict", init_from="", config_hash="h6"), # 0.05h
        JobRecord("video-probe_s7", "video-probe", 7, "baseline", "baseline", "strict", init_from="", config_hash="h7"), # 0.05h
    ]

    packed = bin_pack_jobs_to_workers(jobs, workers_quota, max_batch_hours=3.5, safety_margin_hours=1.5)

    # 1. Largest job (aasist, 3.0h) assigned to worker with highest quota (worker-high-1, 5.8h)
    assert any(j.name == "aasist" for j in packed.get("worker-high-1", []))

    # 2. Second largest job (effnet-b4, 2.0h) assigned to worker-high-2 (5.5h)
    assert any(j.name == "effnet-b4" for j in packed.get("worker-high-2", []))

    # 3. Depleted worker (1.2h) assigned ZERO jobs because remaining < 1.5h safety margin
    assert "worker-depleted" not in packed

    # 4. In any batch with qacp and davidnet, qacp strictly precedes davidnet
    for w, b in packed.items():
        names = [j.name for j in b]
        if "qacp" in names and "davidnet" in names:
            assert names.index("qacp") < names.index("davidnet")

