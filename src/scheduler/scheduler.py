"""Session scheduler: assign one kernel run per idle Kaggle account.

Design (spec sections 8-11)
---------------------------
* The **Hugging Face registry is the campaign's source of truth**: which
  experiments exist and which are finished. The scheduler never picks an
  experiment itself - the notebook claims the next pending one through
  ``src/pipeline`` (claim/lease, checkpoint-aware resume). A scheduler *job* is
  therefore one **session**: "run this worker's kernel until it stops".
* One job per worker at a time (SQLite ``BEGIN IMMEDIATE`` claim, guarded again
  by ``status='PENDING'`` in the WHERE clause), so two accounts never share a
  kernel and one account never runs two kernels.
* The loop per tick: sync registry (throttled) -> monitor running kernels
  (terminal / stale handling) -> dispatch new jobs for idle workers that still
  have GPU quota.
* Recovery: a session whose kernel died without a terminal status is requeued
  until ``max_attempts``; the notebook resumes from the HF checkpoint. Kernels
  still RUNNING after a scheduler restart simply keep their job (lease renewed
  by the monitor).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from .config import SETTINGS, Settings, WorkerCredential, load_workers
from .db import Store
from .kaggle_client import KaggleClient, KaggleError, Quota, RateLimited
from .registry import RegistryError, RegistrySync

ACTIVE = ("CLAIMED", "STARTING", "RUNNING")


@dataclass
class Tick:
    """What one loop pass did (returned to the CLI / tests)."""

    synced: bool = False
    dispatched: list[str] = field(default_factory=list)
    completed: list[str] = field(default_factory=list)
    requeued: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "synced": self.synced,
            "dispatched": self.dispatched,
            "completed": self.completed,
            "requeued": self.requeued,
            "errors": self.errors,
        }


class Scheduler:
    def __init__(self, settings: Settings = SETTINGS,
                 store: Optional[Store] = None,
                 client_factory: Callable[[WorkerCredential], KaggleClient] = KaggleClient,
                 workers: Optional[list[WorkerCredential]] = None,
                 registry: Optional[RegistrySync] = None):
        self.settings = settings
        self.workers = workers if workers is not None else load_workers()
        self.store = store or Store(settings.db_path)
        self.store.upsert_workers(self.workers)
        self._cred = {w.name: w for w in self.workers}
        self._clients = {w.name: client_factory(w) for w in self.workers}
        self._registry = registry
        self._registry_due = 0.0
        self._campaign_announced = False

    # ------------------------------------------------------------------ helpers
    def _client(self, name: str) -> KaggleClient:
        return self._clients[name]

    def _remaining_work(self) -> int:
        counts = self.store.experiment_counts()
        return sum(n for st, n in counts.items() if st != "completed")

    def _budget_work(self) -> int:
        """Unfinished experiments, with a bootstrap rule.

        An empty mirror means either "fresh campaign / first run of this mode"
        (the notebook seeds the registry on its first session) or "registry
        unreachable so far". Both are handled the same way: allow ONE session so
        the campaign can start; afterwards the mirror decides.
        """
        counts = self.store.experiment_counts()
        if not counts:
            return 1
        return sum(n for st, n in counts.items() if st != "completed")

    def _registry_instance(self) -> RegistrySync:
        if self._registry is None:
            self._registry = RegistrySync(mode=self.settings.mode)
        return self._registry

    def sync_registry(self, force: bool = False) -> bool:
        """Mirror registry/experiments.json into the local DB (throttled)."""
        now = time.time()
        if not force and now < self._registry_due:
            return False
        try:
            rows = self._registry_instance().fetch()
        except RegistryError as e:
            self.store.event("registry_error", detail=str(e)[:300])
            self._registry_due = now + self.settings.quota_refresh_s  # retry later
            return False
        self.store.sync_experiments(rows, replace=True)
        self._registry_due = now + self.settings.quota_refresh_s
        self.store.event("registry_sync", detail=f"{len(rows)} experiments")
        return True

    def _quota(self, worker_name: str, now: float, force: bool = False) -> Optional[Quota]:
        row = self.store.worker(worker_name)
        if row is None:
            return None
        checked_at = row["quota_checked_at"]
        if (not force and checked_at and now - float(checked_at) < self.settings.quota_refresh_s
                and row["quota_gpu_remaining"] is not None):
            # used-hours are not cached; dispatch only needs `remaining`
            return Quota(0.0, float(row["quota_gpu_remaining"]),
                         float(row["quota_total"] or 0.0), row["quota_refresh_at"])
        try:
            q = self._client(worker_name).quota()
        except RateLimited as e:
            self.store.event("rate_limited", worker_name, str(e)[:200])
            return None
        except KaggleError as e:
            self.store.event("quota_error", worker_name, str(e)[:200])
            return None
        self.store.set_worker(worker_name, quota_gpu_remaining=q.gpu_remaining_h,
                              quota_total=q.gpu_total_h, quota_refresh_at=q.refresh_at,
                              quota_checked_at=now)
        return q

    # ---------------------------------------------------------------- reconcile
    def reconcile(self) -> None:
        """Startup pass: worker flags vs actual jobs (idempotent)."""
        active_by_worker = {j["worker_id"]: j for j in self.store.jobs() if j["status"] in ACTIVE}
        for w in self.store.workers():
            job = active_by_worker.get(w["id"])
            if job:
                if w["status"] != "BUSY" or w["current_job_id"] != job["id"]:
                    self.store.set_worker(w["name"], status="BUSY", current_job_id=job["id"])
            elif w["status"] in ("BUSY", "OFFLINE"):
                self.store.set_worker(w["name"], status="IDLE", current_job_id=None)
        # lease extension for kernels still alive after a scheduler restart
        for job in self.store.expired_leases():
            self.store.heartbeat(job["id"])
            self.store.event("lease_renewed", job["id"], job["kernel_ref"])

    # ---------------------------------------------------------------- dispatch
    def dispatch(self, dry_run: bool = False, now: float | None = None) -> list[str]:
        """Start jobs for idle workers while the campaign has remaining work."""
        now = now if now is not None else time.time()
        dispatched: list[str] = []
        remaining = self._budget_work()
        if remaining <= 0:
            if not self._campaign_announced:
                self.store.event("campaign_complete", detail="no pending/running/failed experiments")
                self._campaign_announced = True
            return dispatched
        self._campaign_announced = False
        # never launch more sessions than there are unfinished experiments
        budget = remaining - len([j for j in self.store.jobs() if j["status"] in ACTIVE])
        # make sure enough PENDING sessions exist to claim (retries already count)
        if not dry_run:
            pending = len(self.store.jobs("PENDING"))
            for _ in range(max(0, budget - pending)):
                self.store.create_job(reason="session", now=now)

        for wrow in self.store.idle_workers(now):
            if budget <= 0:
                break
            name = wrow["name"]
            quota = self._quota(name, now)
            if quota is not None and quota.gpu_remaining_h < self.settings.quota_min_hours:
                until = _cooldown_until(quota.refresh_at, now, self.settings)
                self.store.set_worker(name, cooldown_until=until)
                self.store.event("quota_cooldown", name,
                                 f"{quota.gpu_remaining_h:.1f}h left; idle until refresh")
                continue

            if dry_run:
                dispatched.append(f"would dispatch {name} -> {self.settings.kernel_slug}")
                budget -= 1
                continue

            jid = self.store.claim_job(wrow["id"], self.settings.lease_s, now)
            if not jid:
                continue
            try:
                self._start_job(jid, name, now)
            except RateLimited as e:
                self._requeue(jid, name, reason=f"rate limited: {e}", now=now, cooldown=True)
            except KaggleError as e:
                self._requeue(jid, name, reason=str(e)[:300], now=now, cooldown=True)
            dispatched.append(jid)
            budget -= 1
        return dispatched

    def _start_job(self, jid: str, worker_name: str, now: float) -> None:
        cred = self._cred[worker_name]
        client = self._client(worker_name)
        self.store.set_job(jid, status="STARTING", started_at=now, heartbeat_at=now,
                           lease_expires_at=now + self.settings.lease_s)

        folder = _build_push_folder(cred, self.settings)
        self._ensure_secret_dataset(client)

        client.push(str(folder))                    # creates/updates + starts the run
        ref = f"{cred.username}/{self.settings.kernel_slug}"
        self.store.set_job(jid, status="RUNNING", kernel_ref=ref, kaggle_status="QUEUED",
                           lease_expires_at=now + self.settings.lease_s)
        self.store.set_worker(worker_name, status="BUSY", current_job_id=jid,
                              last_heartbeat=now)
        self.store.event("dispatched", jid, ref)

    def _ensure_secret_dataset(self, client: KaggleClient) -> None:
        """Upload hf_token.txt once per worker (marker file in the gitignored work dir)."""
        from .kernel import ensure_secret_dataset

        marker = self.settings.work_dir / "secret" / client.worker.name / ".uploaded"
        if marker.exists():
            return
        ensure_secret_dataset(client)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), encoding="utf-8")

    # ----------------------------------------------------------------- monitor
    def monitor(self, now: float | None = None) -> Tick:
        """Poll every active job's kernel; finish / requeue as appropriate."""
        now = now if now is not None else time.time()
        tick = Tick()
        for job in self.store.jobs():
            if job["status"] not in ACTIVE:
                continue
            name = self._worker_name(job["worker_id"])
            if name is None:
                self._requeue(job["id"], None, reason="job has no worker", now=now)
                tick.requeued.append(job["id"])
                continue
            if job["status"] in ("CLAIMED", "STARTING") and job["lease_expires_at"] and job["lease_expires_at"] < now:
                self._requeue(job["id"], name, reason="job never reached RUNNING before lease expiry", now=now)
                tick.requeued.append(job["id"])
                continue
            if not job["kernel_ref"]:
                continue                                # not yet pushed; dispatch owns it

            client = self._client(name)
            try:
                status, raw = client.kernel_status(job["kernel_ref"])
            except RateLimited as e:
                tick.errors.append(f"{job['id']}: {e}")
                continue
            except KaggleError as e:
                tick.errors.append(f"{job['id']}: {e}")
                continue

            if status in ("RUNNING", "QUEUED"):
                self.store.heartbeat(job["id"], now)
                self.store.set_job(job["id"], kaggle_status=status)
                self.store.set_worker(name, last_heartbeat=now)
            elif status == "COMPLETED":
                self._complete(job, name, now)
                tick.completed.append(job["id"])
            elif status in ("FAILED", "CANCELLED"):
                self._requeue(job["id"], name, reason=f"kernel {status.lower()}: {raw}", now=now)
                tick.requeued.append(job["id"])
            else:                                       # UNKNOWN: kernel gone / never ran
                last = job["heartbeat_at"] or job["started_at"] or 0
                if now - float(last) > self.settings.stale_after_s:
                    self._requeue(job["id"], name, reason=f"kernel status unknown ({raw})",
                                  now=now, timeout=True)
                    tick.requeued.append(job["id"])
        return tick

    def _complete(self, job, worker_name: str, now: float) -> None:
        self.store.set_job(job["id"], status="COMPLETED", finished_at=now,
                           kaggle_status="COMPLETED", lease_expires_at=None)
        self.store.set_worker(worker_name, status="IDLE", current_job_id=None,
                              last_heartbeat=now)
        with self.store.tx() as c:
            c.execute("UPDATE workers SET total_completed=total_completed+1 WHERE name=?",
                      (worker_name,))
        self.store.event("job_completed", job["id"], job["kernel_ref"])
        self._registry_due = 0.0                        # re-read registry next tick

    def _requeue(self, job_id: str, worker_name: Optional[str], reason: str,
                 now: float, cooldown: bool = True, timeout: bool = False) -> None:
        job = self.store.job(job_id)
        if job is None:
            return
        terminal = timeout or int(job["attempts"]) >= self.settings.max_attempts
        if terminal:
            self.store.set_job(job_id, status="TIMEOUT" if timeout else "FAILED",
                               finished_at=now, last_error=reason, lease_expires_at=None)
            event = "job_timeout" if timeout else "job_failed"
        else:
            # back to PENDING: another attempt, possibly by a different account
            self.store.set_job(job_id, status="PENDING", worker_id=None, last_error=reason,
                               lease_expires_at=None, kaggle_status=None)
            event = "job_requeued"
        if worker_name:
            fields: dict = {"status": "IDLE", "current_job_id": None, "last_error": reason}
            if cooldown:
                fields["cooldown_until"] = now + self.settings.cooldown_fail_s
            self.store.set_worker(worker_name, **fields)
            if terminal:
                with self.store.tx() as c:
                    c.execute("UPDATE workers SET total_failed=total_failed+1 WHERE name=?",
                              (worker_name,))
        self.store.event(event, job_id, reason[:300])

    def _worker_name(self, worker_id) -> Optional[str]:
        if worker_id is None:
            return None
        if not hasattr(self, "_id_to_name"):
            self._id_to_name = {w["id"]: w["name"] for w in self.store.workers()}
        return self._id_to_name.get(worker_id)

    # -------------------------------------------------------------------- loop
    def tick(self, dry_run: bool = False) -> Tick:
        out = Tick()
        out.synced = self.sync_registry()
        mon = self.monitor()
        out.completed, out.requeued, out.errors = mon.completed, mon.requeued, mon.errors
        out.dispatched = self.dispatch(dry_run=dry_run)
        return out

    def run(self, once: bool = False, dry_run: bool = False,
            max_ticks: Optional[int] = None) -> Tick:
        """Blocking loop (poll_interval_s between ticks). ``once`` runs one tick."""
        self.reconcile()
        self.sync_registry(force=True)
        tick = self.tick(dry_run=dry_run)
        ticks = 1
        while not once and (max_ticks is None or ticks < max_ticks):
            time.sleep(self.settings.poll_interval_s)
            tick = self.tick(dry_run=dry_run)
            ticks += 1
        return tick

    # --------------------------------------------------------------- reporting
    def snapshot(self) -> dict:
        workers = [
            {"name": w["name"], "username": w["username"], "status": w["status"],
             "job": w["current_job_id"], "gpu_left": w["quota_gpu_remaining"],
             "completed": w["total_completed"], "failed": w["total_failed"],
             "cooldown_until": w["cooldown_until"], "last_error": w["last_error"]}
            for w in self.store.workers()
        ]
        jobs = [
            {"id": j["id"], "status": j["status"], "worker": self._worker_name(j["worker_id"]),
             "kernel": j["kernel_ref"], "kaggle": j["kaggle_status"],
             "attempts": j["attempts"], "error": j["last_error"]}
            for j in self.store.jobs()
        ]
        counts = self.store.experiment_counts()
        total = sum(counts.values())
        done = counts.get("completed", 0)
        return {
            "campaign": {"total": total, "completed": done,
                         "progress": round(100.0 * done / total, 1) if total else 0.0,
                         "counts": counts, "remaining": self._remaining_work()},
            "workers": workers,
            "jobs": jobs,
            "job_counts": self.store.job_counts(),
        }


# --------------------------------------------------------------------------- util
def _cooldown_until(refresh_at: Optional[str], now: float, settings: Settings) -> float:
    """When the worker may be tried again after quota exhaustion."""
    if refresh_at:
        import datetime as _dt
        try:
            ts = _dt.datetime.fromisoformat(str(refresh_at).replace("Z", "+00:00")).timestamp()
            return ts + 30.0
        except ValueError:
            pass
    return now + settings.quota_refresh_s


def _build_push_folder(cred: WorkerCredential, settings: Settings):
    from .kernel import build_push_folder
    return build_push_folder(cred, settings=settings)
