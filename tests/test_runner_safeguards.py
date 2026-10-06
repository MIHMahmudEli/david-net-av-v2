import re
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from src.orchestrator.accounts import KaggleAccount
from src.orchestrator.db import JobRecord
from src.orchestrator.kaggle_runner import KaggleRunner


def test_runner_generates_hardened_script(tmp_path, monkeypatch):
    """Verify in-kernel timeout, unbuffered output, and watchdog heartbeat in generated runner script."""
    monkeypatch.setattr("src.orchestrator.kaggle_runner.WORKSPACE_DIR", tmp_path / "workspaces")
    
    acc = KaggleAccount("test-worker", "testuser", "testkey")
    runner = KaggleRunner(acc, dry_run=True)

    job = JobRecord(
        job_id="test_probe_s42",
        name="video-probe",
        seed=42,
        stage="baseline",
        group="baseline",
        split="strict",
        init_from=None,
        config_hash="abc123hash",
        priority=10
    )

    ws_dir = runner.prepare_workspace(job)
    script_path = ws_dir / "run_job.py"
    assert script_path.exists()
    content = script_path.read_text(encoding="utf-8")

    # 1. Unbuffered output
    assert 'os.environ["PYTHONUNBUFFERED"] = "1"' in content
    assert "line_buffering=True" in content

    # 2. Watchdog heartbeat thread every 60s
    assert "def _watchdog_heartbeat():" in content
    assert "[WATCHDOG HEARTBEAT]" in content
    assert "time.sleep(60.0)" in content
    assert "threading.Thread(target=_watchdog_heartbeat, daemon=True)" in content

    # 3. In-kernel timeout (3x estimate) and TimeoutExpired handling
    assert "timeout=" in content
    assert "subprocess.TimeoutExpired" in content
    assert "exit_code = 124" in content
