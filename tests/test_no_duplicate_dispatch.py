"""Regression test: verify that jobs already running, claimed, or completed cannot be dispatched twice."""
import sqlite3
import pytest
from pathlib import Path
from src.orchestrator.db import JobRecord, OrchestratorDB


def test_no_duplicate_dispatch_when_claimed_or_completed(tmp_path):
    db_file = tmp_path / "test_orchestrator.db"
    db = OrchestratorDB(db_path=db_file)

    # Register test worker
    db.register_or_update_worker("worker-1", "user1")
    db.register_or_update_worker("worker-2", "user2")

    # Add jobs
    jobs = [
        JobRecord(job_id="job_a", name="test-a", seed=42, stage="baseline", group="core", split="strict", init_from="", config_hash="h1"),
        JobRecord(job_id="job_b", name="test-b", seed=42, stage="stage1", group="core", split="strict", init_from="test-a", config_hash="h2"),
    ]
    db.insert_jobs(jobs)

    # w1 claims job_a
    claimed_1 = db.claim_next_job("worker-1", lease_duration_sec=3600)
    assert claimed_1 is not None
    assert claimed_1.job_id == "job_a"

    # w2 attempts to claim - must NOT get job_a (already claimed) and NOT job_b (parent not completed)
    claimed_2 = db.claim_next_job("worker-2", lease_duration_sec=3600)
    assert claimed_2 is None, "w2 must not be able to claim job_a (already active) or job_b (dependency unmet)"

    # Mark job_a as COMPLETED
    db.mark_job_completed("job_a")

    # w1 attempts to claim again - must NOT get job_a (completed)
    claimed_3 = db.claim_next_job("worker-1", lease_duration_sec=3600)
    assert claimed_3 is not None
    assert claimed_3.job_id == "job_b", "w1 should now claim dependent job_b"

    # Neither worker can claim job_a or job_b now
    db.mark_job_completed("job_b")
    assert db.claim_next_job("worker-1") is None
    assert db.claim_next_job("worker-2") is None


def test_sync_hf_prevents_duplicate_dispatch(tmp_path):
    db_file = tmp_path / "test_orchestrator.db"
    db = OrchestratorDB(db_path=db_file)
    db.register_or_update_worker("worker-1", "user1")

    # Jobs that match completed runs in HF registry
    jobs = [
        JobRecord(job_id="late-fusion_s2024", name="late-fusion", seed=2024, stage="baseline", group="core", split="strict", init_from="", config_hash="h1"),
        JobRecord(job_id="davidnet-no_qacp_s2024", name="davidnet-no_qacp", seed=2024, stage="stage1", group="core", split="strict", init_from="", config_hash="h2"),
        JobRecord(job_id="new-arm_s2024", name="new-arm", seed=2024, stage="baseline", group="core", split="strict", init_from="", config_hash="h3"),
    ]
    db.insert_jobs(jobs)

    # Simulate HF sync marking completed IDs
    completed_hf_ids = ["late-fusion_s2024", "davidnet-no_qacp_s2024"]
    with db.get_connection() as conn:
        for jid in completed_hf_ids:
            conn.execute("UPDATE jobs SET status = 'COMPLETED', completed_at = 1.0 WHERE job_id = ?", (jid,))

    # Only new-arm_s2024 should be claimable
    claimed = db.claim_next_job("worker-1")
    assert claimed is not None
    assert claimed.job_id == "new-arm_s2024"

    # No more jobs claimable
    assert db.claim_next_job("worker-1") is None
