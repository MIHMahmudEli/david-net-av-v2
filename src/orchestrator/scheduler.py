"""Job Scheduler: manages worker threads, automatic job assignment, and progress reporting."""
from __future__ import annotations

import concurrent.futures
import time
from typing import Optional

from src.orchestrator.accounts import KaggleAccount, load_accounts_from_env
from src.orchestrator.db import JobRecord, OrchestratorDB
from src.orchestrator.kaggle_runner import KaggleRunner


RUNTIME_ESTIMATES_HOURS = {
    "video-probe": 0.05,
    "audio-probe": 0.05,
    "qacp": 0.1,
    "qacp-no_copysynth": 0.1,
    "qacp-no_mismatch": 0.1,
    "qacp-no_selfblend": 0.1,
    "qacp-no_sync": 0.05,
    "late-fusion": 0.3,
    "davidnet-no_qacp": 0.5,
    "davidnet": 0.5,
    "davidnet-qacp_no_copysynth": 0.4,
    "davidnet-qacp_no_mismatch": 0.4,
    "davidnet-qacp_no_selfblend": 0.5,
    "davidnet-no_disentangle": 0.5,
    "davidnet-no_loc": 0.5,
    "davidnet-no_sync": 0.3,
    "davidnet-no_moddrop": 0.5,
    "davidnet-single_task": 0.4,
    "davidnet-compose_quad": 0.4,
    "davidnet-e2e": 1.5,
    "effnet-b4": 2.0,
    "aasist": 3.0,
    "default": 1.0,
}

WEEKLY_GPU_QUOTA_HOURS: float = 30.0  # Official Kaggle weekly GPU quota per account
FIXED_KERNEL_OVERHEAD_HOURS: float = 0.25  # Measured: ~4m container boot/pip + ~5.5m 169-shard HF download + ~5m eval/upload (evidence: EXP_082/086)

# Shared accounts with external projects and default do-not-touch reserve
SHARED_PROJECT_ACCOUNTS: set[str] = {
    "kaggle-worker-8",
    "kaggle-worker-12",
    "kaggle-worker-13",
    "kaggle-worker-14",
}
DEFAULT_SHARED_PROJECT_RESERVE_HOURS: float = 0.5


def parse_gpu_quota_response(raw_q, as_weekly: bool = False, weekly_total_hours: float = WEEKLY_GPU_QUOTA_HOURS) -> Optional[float]:
    """Strictly parses remaining GPU hours from quota_view response without fallbacks.

    - If as_weekly is True: computes remaining weekly account quota out of weekly_total_hours (default 30.0h).
    - If as_weekly is False: computes remaining time in the current single session (capped by totalTimeAllowed, e.g. 6.0h).
    """
    import json
    import re

    if raw_q is None:
        return None

    if isinstance(raw_q, dict) and "totalTimeAllowed" in raw_q and "timeUsed" in raw_q:
        gpu_q = raw_q
    else:
        gpu_q = getattr(raw_q, "_gpu_quota", None) or getattr(raw_q, "gpuQuota", None)
        if gpu_q is None and isinstance(raw_q, dict):
            gpu_q = raw_q.get("_gpu_quota") or raw_q.get("gpuQuota")

    if gpu_q is None:
        return None

    if isinstance(gpu_q, str):
        try:
            gpu_q = json.loads(gpu_q)
        except Exception:
            return None
    elif hasattr(gpu_q, "to_dict"):
        gpu_q = gpu_q.to_dict()
    elif hasattr(gpu_q, "__dict__"):
        gpu_q = {k: v for k, v in gpu_q.__dict__.items() if not k.startswith("_")}

    if not isinstance(gpu_q, dict):
        return None

    if "totalTimeAllowed" not in gpu_q or "timeUsed" not in gpu_q:
        return None

    def _parse_s(v) -> Optional[float]:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        m = re.search(r"^([0-9]+(?:\.[0-9]+)?)", str(v).strip())
        if m:
            return float(m.group(1))
        return None

    tot_s = _parse_s(gpu_q.get("totalTimeAllowed"))
    used_s = _parse_s(gpu_q.get("timeUsed"))
    res_s = _parse_s(gpu_q.get("timeReserved", "0s"))

    if tot_s is None or used_s is None or res_s is None:
        return None

    if as_weekly:
        used_h = used_s / 3600.0
        res_h = res_s / 3600.0
        return max(0.0, weekly_total_hours - used_h - res_h)

    rem_s = max(0.0, tot_s - used_s - res_s)
    return rem_s / 3600.0


def parse_weekly_gpu_quota(raw_q, weekly_total_hours: float = WEEKLY_GPU_QUOTA_HOURS) -> Optional[float]:
    """Computes true remaining weekly GPU quota (out of 30.0 hours)."""
    return parse_gpu_quota_response(raw_q, as_weekly=True, weekly_total_hours=weekly_total_hours)


def get_remaining_gpu_hours(account: KaggleAccount, as_weekly: bool = True) -> Optional[float]:
    """Returns remaining GPU hours for this account from Kaggle CLI, or None if unknown."""
    import subprocess
    import re
    try:
        res = subprocess.run(
            ["kaggle", "quota"],
            env=account.env_dict(),
            capture_output=True,
            text=True,
            timeout=15
        )
        if res.returncode == 0:
            for line in res.stdout.splitlines():
                if line.strip().startswith("GPU"):
                    parts = line.split()
                    if len(parts) >= 3:
                        rem_str = parts[2].rstrip("h")
                        return float(rem_str)
        return None
    except Exception:
        return None



class MultiWorkerScheduler:
    def __init__(self, db: Optional[OrchestratorDB] = None, dry_run: bool = False):
        self.db = db or OrchestratorDB()
        self.dry_run = dry_run
        self.accounts = load_accounts_from_env()
        # Register workers
        for a in self.accounts:
            self.db.register_or_update_worker(a.worker_name, a.username)

    def print_dashboard(self):
        """Displays human-readable dashboard of jobs and workers."""
        summary = self.db.get_queue_summary()
        workers = self.db.get_all_workers()

        print("\n" + "=" * 65, flush=True)
        print("          KAGGLE DISTRIBUTED WORKER DASHBOARD", flush=True)
        print("=" * 65, flush=True)
        print(f"Total Jobs:      {summary['total_jobs']}", flush=True)
        print(f"Completed:       {summary['completed']}  ({summary['progress_pct']}%)", flush=True)
        print(f"Running/Claimed: {summary['running']}", flush=True)
        print(f"Pending/Retry:   {summary['pending']}", flush=True)
        print(f"Failed:          {summary['failed']}", flush=True)
        print("-" * 65, flush=True)
        print("WORKERS:", flush=True)
        for w in workers:
            job_info = f"-> {w.current_job_id}" if w.current_job_id else ""
            status_col = f"{w.status:<10}"
            print(f"  {w.worker_name:<18} ({w.kaggle_username:<18}) {status_col} {job_info}", flush=True)
        print("=" * 65 + "\n", flush=True)

    def run_worker_tick(self, account: KaggleAccount) -> Optional[str]:
        """A single execution cycle for one worker with Quota Guard."""
        worker = self.db.get_worker(account.worker_name)
        if not worker:
            return None

        # Check cooldown
        now = time.time()
        if worker.cooldown_until > now:
            return f"in cooldown for {int(worker.cooldown_until - now)}s"

        # Check if already handling a job
        if worker.current_job_id:
            job = self.db.get_job(worker.current_job_id)
            if job and job.status in ("CLAIMED", "RUNNING"):
                # Poll status
                runner = KaggleRunner(account, dry_run=self.dry_run)
                raw_id = job.job_id[9:] if job.job_id.startswith("davidnet-") else job.job_id
                safe_id = raw_id.lower().replace('_', '-').replace("faceswap", "fs")
                default_slug = f"davidnet-{safe_id}"
                slug = job.kaggle_kernel_slug or default_slug
                status, msg = runner.get_kernel_status(slug)

                if status in ("complete", "completed") or status.endswith("complete"):
                    self.db.mark_job_completed(job.job_id, checkpoint_url=f"kaggle://{account.username}/{slug}")
                    return f"Job {job.job_id} completed successfully"
                elif status in ("error", "failed", "cancel_acknowledged", "canceled") or any(status.endswith(s) for s in ("error", "failed", "cancel_acknowledged", "canceled")):
                    self.db.mark_job_failed(job.job_id, f"Kaggle status {status}: {msg}")
                    return f"Job {job.job_id} {status} on Kaggle"
                else:
                    self.db.update_worker_heartbeat(account.worker_name)
                    return f"Job {job.job_id} is {status}"

        # Claim a new job
        job = self.db.claim_next_job(account.worker_name)
        if not job:
            self.db.update_worker_heartbeat(account.worker_name, status="IDLE")
            return "No eligible jobs to claim"

        # Check Quota Guard before launching the claimed job
        rem_hours = get_remaining_gpu_hours(account) if not self.dry_run else 30.0
        if rem_hours is None:
            with self.db.get_connection() as conn:
                conn.execute("UPDATE jobs SET status = 'PENDING', worker_name = NULL WHERE job_id = ?", (job.job_id,))
            return f"Quota guard ALERT: {account.worker_name} GPU quota is unknown -> do not dispatch. Job {job.job_id} released."

        shared_res = DEFAULT_SHARED_PROJECT_RESERVE_HOURS if account.worker_name in SHARED_PROJECT_ACCOUNTS else 0.0
        effective_rem = max(0.0, rem_hours - shared_res)

        est_hours = RUNTIME_ESTIMATES_HOURS.get(job.name, 1.0)
        # Measured small jobs (<1.0h: probes, late-fusion, qacp, davidnet) use 0.75h margin; large/unmeasured use 1.5h
        margin = 0.75 if est_hours < 1.0 else 1.5
        needed_hours = est_hours + margin
        if effective_rem < needed_hours:
            # Quota violation guard: release job and cool down worker
            with self.db.get_connection() as conn:
                conn.execute("UPDATE jobs SET status = 'PENDING', worker_name = NULL WHERE job_id = ?", (job.job_id,))
                conn.execute("UPDATE workers SET status = 'IDLE', current_job_id = NULL, cooldown_until = ? WHERE worker_name = ?",
                             (now + 3600, account.worker_name))
            return f"Quota guard: {account.worker_name} has {effective_rem:.2f}h usable (after {shared_res:.2f}h reserve) < needed {needed_hours:.2f}h (est {est_hours:.2f}h + margin {margin:.2f}h). Job {job.job_id} released."

        # Start execution
        runner = KaggleRunner(account, dry_run=self.dry_run)
        ws_dir = runner.prepare_workspace(job)
        slug = getattr(runner, "last_slug", None) or (job.job_id[9:] if job.job_id.startswith("davidnet-") else job.job_id)
        success, out = runner.push_kernel(ws_dir)

        if success:
            self.db.mark_job_running(job.job_id, slug)
            return f"Claimed and launched job {job.job_id}"
        else:
            self.db.mark_job_failed(job.job_id, f"Failed to push kernel: {out}")
            return f"Failed to launch job {job.job_id}: {out}"

    def step_all_workers(self):
        """Tick all available workers once."""
        self.db.recover_stale_leases()
        results = {}
        for a in self.accounts:
            res = self.run_worker_tick(a)
            results[a.worker_name] = res
        return results
