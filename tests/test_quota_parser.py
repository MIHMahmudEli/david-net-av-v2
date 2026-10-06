import pytest
from unittest.mock import MagicMock, patch

from src.orchestrator.accounts import KaggleAccount
from src.orchestrator.scheduler import (
    MultiWorkerScheduler,
    get_remaining_gpu_hours,
    parse_gpu_quota_response,
)


class DummyQuotaView:
    def __init__(self, gpu_quota=None, refresh_time="2026-10-10T00:00:00Z"):
        self._gpu_quota = gpu_quota
        self._quota_refresh_time = refresh_time


def test_quota_parser_valid_dict():
    q = DummyQuotaView({
        "timeUsed": "3600s",
        "timeReserved": "0s",
        "totalTimeAllowed": "21600s"
    })
    rem = parse_gpu_quota_response(q)
    assert rem is not None
    # (21600 - 3600) / 3600 = 5.0 hours
    assert pytest.approx(rem, 0.01) == 5.0


def test_quota_parser_with_reserved_time():
    # 21600 total, 3600 used, 7200 reserved by active kernel
    q = DummyQuotaView({
        "timeUsed": "3600.5s",
        "timeReserved": "7200.0s",
        "totalTimeAllowed": "21600s"
    })
    rem = parse_gpu_quota_response(q)
    assert rem is not None
    # (21600 - 3600.5 - 7200) / 3600 = 3.0 hours (approx 2.9998h)
    assert pytest.approx(rem, 0.01) == 2.9998


def test_quota_parser_string_format():
    # JSON-encoded string in gpuQuota
    q = DummyQuotaView('{"timeUsed": "1800s", "timeReserved": "0s", "totalTimeAllowed": "21600s"}')
    rem = parse_gpu_quota_response(q)
    assert rem is not None
    assert pytest.approx(rem, 0.01) == 5.5


def test_quota_parser_missing_fields_returns_none():
    # Missing totalTimeAllowed
    q1 = DummyQuotaView({"timeUsed": "100s"})
    assert parse_gpu_quota_response(q1) is None

    # Missing timeUsed
    q2 = DummyQuotaView({"totalTimeAllowed": "21600s"})
    assert parse_gpu_quota_response(q2) is None

    # Missing gpu_quota entirely
    q3 = DummyQuotaView(None)
    assert parse_gpu_quota_response(q3) is None


def test_quota_parser_corrupted_format_returns_none():
    q1 = DummyQuotaView("invalid-json{not_a_dict}")
    assert parse_gpu_quota_response(q1) is None

    q2 = DummyQuotaView({"timeUsed": "unparseable_seconds", "totalTimeAllowed": "21600s"})
    assert parse_gpu_quota_response(q2) is None


def test_get_remaining_gpu_hours_api_exception_returns_none():
    acc = KaggleAccount("worker-1", "user1", "key1")
    with patch("kaggle.api.kaggle_api_extended.KaggleApi.authenticate", side_effect=RuntimeError("API Network Down")):
        rem = get_remaining_gpu_hours(acc)
        assert rem is None


def test_scheduler_fails_closed_when_quota_unknown(tmp_path):
    from src.orchestrator.db import JobRecord, OrchestratorDB
    db = OrchestratorDB(tmp_path / "test.db")
    db.register_or_update_worker("worker-1", "user1")
    db.insert_jobs([
        JobRecord("test_job_s7", "audio-probe", 7, "baseline", "baseline", "strict", None, "hash1", 10)
    ])

    sched = MultiWorkerScheduler(db=db, dry_run=False)
    acc = KaggleAccount("worker-1", "user1", "key1")

    # Mock get_remaining_gpu_hours to return None (unknown quota)
    with patch("src.orchestrator.scheduler.get_remaining_gpu_hours", return_value=None):
        msg = sched.run_worker_tick(acc)
        assert "Quota guard ALERT" in msg
        assert "GPU quota is unknown -> do not dispatch" in msg
        # Job must NOT have been claimed or running
        job = db.get_job("test_job_s7")
        assert job.status == "PENDING"


def test_quota_parser_malformed_seconds_handles_workers_12_to_14():
    """Verify that Kaggle's malformed seconds strings like '1496.377.0s' correctly parse remaining hours."""
    raw_response = {
        "timeUsed": "1496.377.0s",
        "timeReserved": "0s",
        "totalTimeAllowed": "21600s",
        "minimumTimeAllowed": "21600s",
        "hasEverRun": True
    }
    rem = parse_gpu_quota_response(raw_response)
    assert rem is not None
    # 21600 - 1496.377 = 20103.623s = 5.58434h (session remaining)
    assert pytest.approx(rem, 0.001) == 5.584

    # True weekly quota (out of 30h): 30.0 - (1496.377 / 3600) = 29.584h (29h 35m)
    rem_weekly = parse_gpu_quota_response(raw_response, as_weekly=True)
    assert rem_weekly is not None
    assert pytest.approx(rem_weekly, 0.001) == 29.584


def test_quota_fixtures_from_quota_doc():
    """Verify raw API samples from docs/quota.md for workers 7, 4, and 3."""
    # Fixture A: Worker 7 (kaggle-worker-7) - idle
    fix_a = {
        "timeUsed": "1428.729.0s",
        "timeReserved": "0s",
        "totalTimeAllowed": "21600s",
        "minimumTimeAllowed": "21600s",
        "hasEverRun": True
    }
    rem_a_sess = parse_gpu_quota_response(fix_a, as_weekly=False)
    rem_a_week = parse_gpu_quota_response(fix_a, as_weekly=True)
    assert pytest.approx(rem_a_sess, 0.001) == (21600 - 1428.729) / 3600.0
    assert pytest.approx(rem_a_week, 0.001) == 30.0 - (1428.729 / 3600.0)

    # Fixture B: Worker 4 (kaggle-worker-4) - heavily used
    fix_b = {
        "timeUsed": "29342.908206.0s",
        "timeReserved": "0s",
        "totalTimeAllowed": "21600s",
        "minimumTimeAllowed": "21600s",
        "hasEverRun": True
    }
    rem_b_week = parse_gpu_quota_response(fix_b, as_weekly=True)
    assert pytest.approx(rem_b_week, 0.001) == 30.0 - (29342.908206 / 3600.0)

    # Fixture C: Worker 3 (kaggle-worker-3) - over 6h used
    fix_c = {
        "timeUsed": "27842.729312.0s",
        "timeReserved": "0s",
        "totalTimeAllowed": "21600s",
        "minimumTimeAllowed": "21600s",
        "hasEverRun": True
    }
    rem_c_week = parse_gpu_quota_response(fix_c, as_weekly=True)
    assert pytest.approx(rem_c_week, 0.001) == 30.0 - (27842.729312 / 3600.0)

