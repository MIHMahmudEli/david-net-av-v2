import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.orchestrator.accounts import KaggleAccount
from src.orchestrator.batch_runner import BatchRunner, build_batch_script
from src.orchestrator.db import JobRecord


def test_build_batch_script():
    jobs = [
        JobRecord(job_id="audio-probe_s7", name="audio-probe", seed=7, stage="baseline", group="baseline", split="strict", init_from=None, config_hash="h1", priority=10),
        JobRecord(job_id="qacp-no_sync_s7", name="qacp-no_sync", seed=7, stage="qacp", group="qacp", split="strict", init_from=None, config_hash="h2", priority=20),
    ]
    script = build_batch_script(jobs, "worker-1", "6793d730aa734433bc3c7bc9ff4f89e0b5dc40d1")
    assert "audio-probe_s7" in script
    assert "qacp-no_sync_s7" in script
    assert "6793d730aa734433bc3c7bc9ff4f89e0b5dc40d1" in script
    import re
    assert "for idx, job in enumerate(batch_jobs, 1):" in script
    # Verify no secret token strings
    assert re.search(r"hf_[A-Za-z0-9]{15,}", script) is None


def test_batch_runner_prepare_workspace(tmp_path):
    acc = KaggleAccount("worker-1", "user1", "key1")
    runner = BatchRunner(acc, dry_run=True)

    with patch("src.orchestrator.batch_runner.WORKSPACE_DIR", tmp_path):
        jobs = [
            JobRecord(job_id="job_a_s7", name="job_a", seed=7, stage="baseline", group="baseline", split="strict", init_from=None, config_hash="h1", priority=10),
            JobRecord(job_id="job_b_s7", name="job_b", seed=7, stage="baseline", group="baseline", split="strict", init_from=None, config_hash="h2", priority=10),
        ]
        ws = runner.prepare_batch_workspace("batch_001", jobs, "6793d73")

        assert ws.exists()
        meta = json.loads((ws / "kernel-metadata.json").read_text())
        assert meta["id"] == "user1/davidnet-batch-batch-001"
        assert meta["is_private"] == "true"
        assert "user1/davidnet-hf-token" in meta["dataset_sources"]

        script = (ws / "run_job.py").read_text()
        assert "job_a_s7" in script
        assert "job_b_s7" in script
