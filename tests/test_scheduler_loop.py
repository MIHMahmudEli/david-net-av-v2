"""Session-scheduler loop tests against a fake Kaggle client and a fake registry.

Covers: dispatch budget, one-session-per-worker, quota cooldown, completion,
requeue-until-max-attempts, stale-kernel timeout, dry-run safety.
"""
from __future__ import annotations

import time

import pytest

from src.scheduler.config import SETTINGS, Settings, WorkerCredential
from src.scheduler.kaggle_client import KaggleError, Quota
from src.scheduler.scheduler import Scheduler


# ------------------------------------------------------------------- fakes
class FakeRegistry:
    def __init__(self, rows):
        self.rows = rows
        self.fetches = 0

    def fetch(self):
        self.fetches += 1
        return self.rows


def exp_row(key="full:probe:s42", status="pending", name="probe", seed=42):
    return {"key": key, "exp_id": "EXP_900", "name": name, "seed": seed,
            "grp": "baseline", "status": status, "claimed_by": None,
            "test_clip_auc": None, "train_hours": None, "has_checkpoint": 0}


class FakeClient:
    """Shared state: which kernel statuses each ref reports, who pushed, quota."""

    def __init__(self, cred, state):
        self.worker = cred
        self.state = state

    def quota(self) -> Quota:
        return self.state.get("quota") or Quota(0.0, 30.0, 30.0, "2099-01-01T00:00:00")

    def push(self, folder, accelerator=None):
        self.state.setdefault("pushes", []).append(self.worker.name)
        return "pushed"

    def kernel_status(self, ref):
        return self.state.setdefault("status", {}).get(ref, ("QUEUED", "KernelWorkerStatus.QUEUEING"))


@pytest.fixture
def state():
    return {"pushes": [], "status": {}}


@pytest.fixture
def workers():
    return [WorkerCredential(name=f"w{i}", username=f"user{i}", token="KGAT_x", env_index=i)
            for i in (1, 2, 3)]


def make_sched(tmp_path, state, workers, rows, **overrides):
    defaults = dict(db_path=tmp_path / "state.db", work_dir=tmp_path / "work",
                    max_attempts=2, cooldown_fail_s=0.0, quota_min_hours=1.0,
                    quota_refresh_s=3600.0, lease_s=60.0, stale_after_s=60.0)
    defaults.update(overrides)
    settings = Settings(**defaults)
    sched = Scheduler(settings=settings, workers=workers,
                      client_factory=lambda w: FakeClient(w, state),
                      registry=FakeRegistry(rows))
    # secret dataset upload must not hit the network in tests
    sched._ensure_secret_dataset = lambda client: None
    return sched


# ------------------------------------------------------------------- dispatch
def test_dispatch_starts_one_session_per_idle_worker(tmp_path, state, workers):
    rows = [exp_row("full:a:s42", "pending"), exp_row("full:b:s42", "pending"),
            exp_row("full:c:s42", "pending")]
    sched = make_sched(tmp_path, state, workers, rows)
    sched.reconcile()
    sched.sync_registry(force=True)
    tick = sched.tick()
    assert len(tick.dispatched) == 3
    jobs = sched.store.jobs()
    assert {j["status"] for j in jobs} == {"RUNNING"}
    assert all(j["kernel_ref"].endswith("/davidnet-q1-pipeline") for j in jobs)
    assert {w["status"] for w in sched.store.workers()} == {"BUSY"}
    assert sorted(state["pushes"]) == ["w1", "w2", "w3"]


def test_budget_never_exceeds_remaining_experiments(tmp_path, state, workers):
    """1 unfinished experiment -> exactly 1 session, even with 3 idle accounts."""
    sched = make_sched(tmp_path, state, workers, [exp_row("full:a:s42", "pending")])
    sched.reconcile(); sched.sync_registry(force=True)
    tick = sched.tick()
    assert len(tick.dispatched) == 1
    assert len(sched.store.jobs()) == 1
    # a second tick must not pile on: the session is active, budget is 0
    tick2 = sched.tick()
    assert tick2.dispatched == []


def test_no_dispatch_when_campaign_complete(tmp_path, state, workers):
    rows = [exp_row("full:a:s42", "completed"), exp_row("full:b:s42", "completed")]
    sched = make_sched(tmp_path, state, workers, rows)
    sched.reconcile(); sched.sync_registry(force=True)
    tick = sched.tick()
    assert tick.dispatched == [] and sched.store.jobs() == []
    kinds = [e["kind"] for e in sched.store.events(10)]
    assert "campaign_complete" in kinds


def test_bootstrap_allows_one_session_when_registry_is_empty(tmp_path, state, workers):
    """First run of a mode: no registry rows yet -> exactly one session may start."""
    sched = make_sched(tmp_path, state, workers, [])      # registry has no rows
    sched.reconcile(); sched.sync_registry(force=True)
    tick = sched.tick()
    assert len(tick.dispatched) == 1
    assert len(sched.store.jobs()) == 1
    kinds = [e["kind"] for e in sched.store.events(10)]
    assert "campaign_complete" not in kinds


def test_dry_run_touches_no_state(tmp_path, state, workers):
    rows = [exp_row("full:a:s42"), exp_row("full:b:s42"), exp_row("full:c:s42")]
    sched = make_sched(tmp_path, state, workers, rows)
    sched.reconcile(); sched.sync_registry(force=True)
    tick = sched.tick(dry_run=True)
    assert len(tick.dispatched) == 3
    assert sched.store.jobs() == []
    assert state["pushes"] == []


def test_quota_exhaustion_cools_the_worker_down(tmp_path, state, workers):
    state["quota"] = Quota(29.5, 0.5, 30.0, "2099-01-01T00:00:00")
    sched = make_sched(tmp_path, state, workers, [exp_row()])
    sched.reconcile(); sched.sync_registry(force=True)
    tick = sched.tick()
    assert tick.dispatched == []
    now = time.time()
    for w in sched.store.workers():
        assert w["cooldown_until"] is not None and w["cooldown_until"] > now
    kinds = [e["kind"] for e in sched.store.events(10)]
    assert "quota_cooldown" in kinds


def test_push_failure_requeues_with_cooldown(tmp_path, state, workers):
    class BoomClient(FakeClient):
        def push(self, folder, accelerator=None):
            raise KaggleError("push exploded")

    sched = make_sched(tmp_path, state, workers,
                       [exp_row(), exp_row("full:b:s42"), exp_row("full:c:s42")])
    sched._clients = {w.name: BoomClient(w, state) for w in workers}
    sched.reconcile(); sched.sync_registry(force=True)
    tick = sched.tick()
    assert len(tick.dispatched) == 3                      # all three were attempted
    assert {j["status"] for j in sched.store.jobs()} == {"PENDING"}
    assert {w["status"] for w in sched.store.workers()} == {"IDLE"}
    kinds = [e["kind"] for e in sched.store.events(20)]
    assert kinds.count("job_requeued") == 3


# ------------------------------------------------------------------- monitor
def test_completed_kernel_frees_the_worker(tmp_path, state, workers):
    sched = make_sched(tmp_path, state, workers, [exp_row(), exp_row("full:b:s42")])
    sched.reconcile(); sched.sync_registry(force=True)
    sched.tick()
    ref = sched.store.jobs()[0]["kernel_ref"]
    state["status"][ref] = ("COMPLETED", "KernelWorkerStatus.COMPLETE")
    tick = sched.tick()
    assert tick.completed
    job = sched.store.job(sched.store.jobs()[0]["id"])
    assert job["status"] == "COMPLETED"
    finished = [w for w in sched.store.workers() if w["status"] == "IDLE"]
    assert len(finished) == 1
    assert finished[0]["total_completed"] == 1
    # completed session frees budget for another dispatch (2 experiments, 1 done session)
    assert len(sched.store.jobs()) >= 2


def test_running_kernel_renews_lease(tmp_path, state, workers):
    sched = make_sched(tmp_path, state, workers, [exp_row(), exp_row("full:b:s42")])
    sched.reconcile(); sched.sync_registry(force=True)
    sched.tick()
    job = sched.store.jobs()[0]
    ref = job["kernel_ref"]
    state["status"][ref] = ("RUNNING", "KernelWorkerStatus.RUNNING")
    before = sched.store.job(job["id"])["heartbeat_at"]
    time.sleep(0.01)
    sched.tick()
    after = sched.store.job(job["id"])["heartbeat_at"]
    assert after > before
    assert sched.store.job(job["id"])["status"] == "RUNNING"


def test_failed_kernel_requeues_then_fails_at_max_attempts(tmp_path, state, workers):
    sched = make_sched(tmp_path, state, workers, [exp_row(), exp_row("full:b:s42")])
    sched.reconcile(); sched.sync_registry(force=True)
    sched.tick()
    job_id = sched.store.jobs()[0]["id"]
    ref = sched.store.job(job_id)["kernel_ref"]
    state["status"][ref] = ("FAILED", "KernelWorkerStatus.FAILED")

    sched.tick()                                   # attempt 1 fails -> requeued -> re-claimed
    job = sched.store.job(job_id)
    assert job["attempts"] == 2 and job["status"] == "RUNNING"
    assert job["last_error"] and "failed" in job["last_error"]

    sched.tick()                                   # attempt 2 fails -> terminal
    job = sched.store.job(job_id)
    assert job["status"] == "FAILED"
    failed = [w for w in sched.store.workers() if w["total_failed"] == 1]
    assert len(failed) == 1


def test_unknown_kernel_times_out_after_stale_window(tmp_path, state, workers):
    sched = make_sched(tmp_path, state, workers, [exp_row(), exp_row("full:b:s42")])
    sched.reconcile(); sched.sync_registry(force=True)
    sched.tick()
    job_id = sched.store.jobs()[0]["id"]
    ref = sched.store.job(job_id)["kernel_ref"]
    state["status"][ref] = ("UNKNOWN", "no such kernel")
    # inside the stale window -> keep waiting
    sched.tick()
    assert sched.store.job(job_id)["status"] == "RUNNING"
    # beyond the stale window -> requeue as a timeout (max_attempts=1 makes it terminal)
    old = time.time() - 120
    sched.store.set_job(job_id, heartbeat_at=old, started_at=old)
    tick = sched.tick()
    assert job_id in tick.requeued
    assert sched.store.job(job_id)["status"] == "TIMEOUT"


def test_reconcile_recovers_busy_flag_after_restart(tmp_path, state, workers):
    sched = make_sched(tmp_path, state, workers, [exp_row(), exp_row("full:b:s42")])
    sched.reconcile(); sched.sync_registry(force=True)
    sched.tick()
    assert {w["status"] for w in sched.store.workers()} == {"BUSY", "IDLE"}
    # simulate a crash: worker row says BUSY but its job is no longer active
    active_ids = {j["id"] for j in sched.store.jobs()
                  if j["status"] in ("CLAIMED", "STARTING", "RUNNING")}
    orphan = next(j for j in sched.store.jobs() if j["status"] == "RUNNING")
    sched.store.set_job(orphan["id"], status="FAILED", worker_id=None)
    sched.__dict__.pop("_id_to_name", None)
    sched.reconcile()
    active_ids = {j["id"] for j in sched.store.jobs()
                  if j["status"] in ("CLAIMED", "STARTING", "RUNNING")}
    for w in sched.store.workers():
        if w["status"] == "BUSY":
            job = sched.store.job(w["current_job_id"])
            assert job is not None and job["id"] in active_ids
        else:
            assert w["current_job_id"] is None


def test_snapshot_reports_campaign_and_jobs(tmp_path, state, workers):
    rows = [exp_row("full:a:s42", "completed"), exp_row("full:b:s42", "pending")]
    sched = make_sched(tmp_path, state, workers, rows)
    sched.reconcile(); sched.sync_registry(force=True)
    sched.tick()
    snap = sched.snapshot()
    assert snap["campaign"]["total"] == 2
    assert snap["campaign"]["completed"] == 1
    assert snap["campaign"]["remaining"] == 1
    assert snap["job_counts"].get("RUNNING") == 1
