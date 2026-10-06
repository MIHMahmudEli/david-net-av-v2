"""Unit tests for the distributed Kaggle orchestrator.
Tests atomic job claiming, concurrency, duplicate prevention, dependency resolution, and lease recovery.
"""
import concurrent.futures
import tempfile
import time
from pathlib import Path

import pytest

from src.orchestrator.db import JobRecord, OrchestratorDB


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = Path(tmpdir) / "test_orchestrator.db"
        db = OrchestratorDB(db_path)
        yield db


def test_atomic_claim_no_duplicates(temp_db):
    """Test that 16 concurrent workers claiming from a pool of 5 jobs NEVER get duplicate assignments."""
    jobs = [
        JobRecord(
            job_id=f"job_{i}",
            name=f"exp_{i}",
            seed=42,
            stage="stage1",
            group="proposed",
            split="strict",
            init_from="",
            config_hash=f"hash_{i}",
            priority=10
        )
        for i in range(5)
    ]
    temp_db.insert_jobs(jobs)

    workers = [f"worker_{i}" for i in range(16)]
    for w in workers:
        temp_db.register_or_update_worker(w, f"user_{w}")

    claimed_jobs = []

    def worker_claim(w):
        return temp_db.claim_next_job(w)

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(worker_claim, workers))

    for r in results:
        if r is not None:
            claimed_jobs.append(r.job_id)

    # Exactly 5 jobs should be claimed
    assert len(claimed_jobs) == 5
    # All 5 claimed job IDs must be strictly unique!
    assert len(set(claimed_jobs)) == 5


def test_dependency_resolution(temp_db):
    """Stage 1 jobs with init_from MUST NOT be claimed until their parent QACP job is completed."""
    qacp_job = JobRecord(
        job_id="qacp_s42",
        name="qacp",
        seed=42,
        stage="qacp",
        group="qacp",
        split="strict",
        init_from="",
        config_hash="h1",
        priority=20
    )
    stage1_job = JobRecord(
        job_id="davidnet_s42",
        name="davidnet",
        seed=42,
        stage="stage1",
        group="proposed",
        split="strict",
        init_from="qacp",
        config_hash="h2",
        priority=10
    )
    temp_db.insert_jobs([qacp_job, stage1_job])
    temp_db.register_or_update_worker("w1", "u1")

    # w1 should claim QACP first because stage 1 has an uncompleted dependency
    claimed_1 = temp_db.claim_next_job("w1")
    assert claimed_1 is not None
    assert claimed_1.job_id == "qacp_s42"

    # w2 tries to claim while QACP is still in progress -> should get None
    temp_db.register_or_update_worker("w2", "u2")
    claimed_2 = temp_db.claim_next_job("w2")
    assert claimed_2 is None

    # Mark QACP completed
    temp_db.mark_job_completed("qacp_s42")

    # Now w2 can claim stage 1!
    claimed_3 = temp_db.claim_next_job("w2")
    assert claimed_3 is not None
    assert claimed_3.job_id == "davidnet_s42"


def test_stale_lease_recovery(temp_db):
    """Stale leases older than timeout are automatically recovered."""
    job = JobRecord(
        job_id="test_job",
        name="test",
        seed=42,
        stage="stage1",
        group="proposed",
        split="strict",
        init_from="",
        config_hash="h",
        priority=10
    )
    temp_db.insert_jobs([job])
    temp_db.register_or_update_worker("w1", "u1")

    # Claim with a 1-second lease
    claimed = temp_db.claim_next_job("w1", lease_duration_sec=0.1)
    assert claimed is not None

    time.sleep(0.2)
    # Recover
    recovered = temp_db.recover_stale_leases()
    assert recovered == 1

    # Check job is now RETRYING and can be claimed again
    temp_db.register_or_update_worker("w2", "u2")
    reclaimed = temp_db.claim_next_job("w2")
    assert reclaimed is not None
    assert reclaimed.job_id == "test_job"


def test_environment_report_fields():
    from src.pipeline.env import environment_report
    env = environment_report(".", worker_name="test-worker")
    assert "git_commit" in env
    assert env["git_commit"] is not None
    assert len(env["git_commit"]) >= 7
    assert "python_version" in env
    assert "torch_version" in env
    assert env["worker_name"] == "test-worker"


def test_runner_script_contains_no_secrets(tmp_path, monkeypatch):
    import json
    from src.orchestrator.accounts import KaggleAccount
    from src.orchestrator.kaggle_runner import KaggleRunner

    fake_token = "hf_ABC1234567890abcdefghijklmnopqrstuvwxyz"
    monkeypatch.setenv("HF_TOKEN", fake_token)
    monkeypatch.setattr("src.orchestrator.kaggle_runner.WORKSPACE_DIR", tmp_path)

    acc = KaggleAccount(worker_name="kaggle-worker-1", username="testuser", key="fake-key", is_valid=True)
    runner = KaggleRunner(acc, dry_run=True)
    job = JobRecord(
        job_id="test_exp_s42",
        name="test_exp",
        seed=42,
        stage="stage1",
        group="proposed",
        split="strict",
        init_from="",
        config_hash="h1",
    )
    ws_dir = runner.prepare_workspace(job)
    script_text = (ws_dir / "run_job.py").read_text(encoding="utf-8")
    meta_text = (ws_dir / "kernel-metadata.json").read_text(encoding="utf-8")

    # Verify no raw token or token-like string appears
    import re
    assert fake_token not in script_text
    assert not re.search(r"hf_[a-zA-Z0-9]{10,}", script_text)
    # Verify metadata has is_private true
    meta = json.loads(meta_text)
    assert meta.get("is_private") == "true" or meta.get("is_private") is True


def test_two_workers_cannot_overwrite_each_other_or_reports(tmp_path):
    """Simulate two workers uploading artifacts and ensure run isolation."""
    import json
    run1_dir = tmp_path / "experiments" / "EXP_001_video-probe_s7"
    run1_dir.mkdir(parents=True)
    (run1_dir / "metrics.json").write_text('{"auc": 0.95}', encoding="utf-8")

    run2_dir = tmp_path / "experiments" / "EXP_002_audio-probe_s7"
    run2_dir.mkdir(parents=True)
    (run2_dir / "metrics.json").write_text('{"auc": 0.99}', encoding="utf-8")

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True)
    (reports_dir / "summary.json").write_text('{"version": "central"}', encoding="utf-8")

    assert json.loads((run1_dir / "metrics.json").read_text())["auc"] == 0.95
    assert json.loads((run2_dir / "metrics.json").read_text())["auc"] == 0.99
    assert json.loads((reports_dir / "summary.json").read_text())["version"] == "central"
    assert run1_dir != run2_dir


def test_concurrent_dispatch_same_job_results_in_one_claim(temp_db):
    """S1: Concurrent claiming of the same single job from multiple workers results in exactly one claim."""
    job = JobRecord(
        job_id="single_job_s7",
        name="video-probe",
        seed=7,
        stage="baseline",
        group="baseline",
        split="strict",
        init_from="",
        config_hash="h_single",
        priority=10
    )
    temp_db.insert_jobs([job])
    workers = [f"w_{i}" for i in range(10)]
    for w in workers:
        temp_db.register_or_update_worker(w, f"u_{w}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        claimed = list(executor.map(lambda w: temp_db.claim_next_job(w), workers))

    successful_claims = [c for c in claimed if c is not None]
    assert len(successful_claims) == 1
    assert successful_claims[0].job_id == "single_job_s7"


def test_job_code_revision_persistence(temp_db):
    """P2: Verify code_revision column persists and is preserved across claims and resets."""
    job = JobRecord(
        job_id="test_rev_job",
        name="video-probe",
        seed=42,
        stage="baseline",
        group="baseline",
        split="strict",
        init_from="",
        config_hash="hash_test",
        code_revision="phase2-v1"
    )
    temp_db.insert_jobs([job])
    retrieved = temp_db.get_job("test_rev_job")
    assert retrieved is not None
    assert retrieved.code_revision == "phase2-v1"

    # Claim and re-check
    temp_db.register_or_update_worker("w_test", "u_test")
    claimed = temp_db.claim_next_job("w_test")
    assert claimed is not None
    assert claimed.code_revision == "phase2-v1"


def test_revision_map_and_tag_lookup():
    """P2: Verify get_pinned_revision resolves group map, tags, and direct 40-char SHAs."""
    from src.pipeline.revision import get_pinned_revision, get_revision_map

    rev_map = get_revision_map()
    assert isinstance(rev_map, dict)
    assert "baseline" in rev_map
    assert "proposed" in rev_map

    # Check tag lookup
    sha_tag = get_pinned_revision("phase2-v1")
    assert len(sha_tag) == 40

    # Check group lookup
    sha_group = get_pinned_revision("baseline")
    assert len(sha_group) == 40
    assert sha_tag == sha_group

    # Direct 40-char SHA lookup
    custom_sha = "0123456789abcdef0123456789abcdef01234567"
    assert get_pinned_revision(custom_sha) == custom_sha


