"""Batch Kernel Runner: executes multiple queued cheap jobs sequentially within a single Kaggle container.

This avoids paying the ~15-20 min VM spin-up and dataset download penalty multiple times,
while strictly guaranteeing:
1. Each job retains its own unique experiment ID (e.g. EXP_082, EXP_083).
2. Each job writes strictly to its own isolated folder (experiments/EXP_xxx_<name>_s<seed>/).
3. Pinned commit is verified before running any job.
4. Total batch runtime estimate is capped (default: <= 4.0 hours) to avoid Kaggle timeouts.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Sequence

from src.orchestrator.accounts import KaggleAccount
from src.orchestrator.db import JobRecord, OrchestratorDB
from src.orchestrator.kaggle_runner import DATASET_SLUGS, KaggleRunner, WORKSPACE_DIR
from src.pipeline.revision import get_pinned_revision


def estimate_batch_hours(jobs: Sequence[JobRecord]) -> float:
    """Calculates total estimated runtime in hours for a batch of jobs."""
    from src.orchestrator.scheduler import RUNTIME_ESTIMATES_HOURS
    return sum(RUNTIME_ESTIMATES_HOURS.get(j.name, 0.5) for j in jobs)


def validate_batch_ordering(jobs: Sequence[JobRecord]) -> tuple[bool, str]:
    """Ensures Stage 0 prerequisites strictly precede Stage 1 dependents in the batch."""
    seen = set()
    for j in jobs:
        if j.init_from:
            dep_id = f"{j.init_from}_s{j.seed}"
            # If the dependent is in the batch, it must come before j
            # If not in the batch, caller must have verified it's completed on HF
            batch_has_dep = any(other.name == j.init_from and other.seed == j.seed for other in jobs)
            if batch_has_dep and dep_id not in seen:
                return False, f"Job {j.job_id} requires {dep_id} which does not precede it in the batch"
        seen.add(j.job_id)
    return True, "OK"


def validate_batch_for_worker(jobs: Sequence[JobRecord], rem_hours: Optional[float], max_batch_hours: float = 4.0) -> tuple[bool, str]:
    """Validates that batch meets session time limit and worker quota guard with 1.5h margin."""
    if rem_hours is None:
        return False, "Worker GPU quota is unknown"
    if not jobs:
        return False, "Batch is empty"
    
    ok_order, order_msg = validate_batch_ordering(jobs)
    if not ok_order:
        return False, order_msg

    est = estimate_batch_hours(jobs)
    if est > max_batch_hours:
        return False, f"Batch estimate {est:.2f}h exceeds max batch limit {max_batch_hours:.2f}h"

    needed = est + 1.5
    if rem_hours < needed:
        return False, f"Quota guard: worker has {rem_hours:.2f}h < needed {needed:.2f}h (est {est:.2f}h + 1.5h margin)"

    return True, "OK"


def bin_pack_jobs_to_workers(
    jobs: Sequence[JobRecord],
    workers_quota: dict[str, float],
    max_batch_hours: float = 3.5,
    safety_margin_hours: float = 1.5,
) -> dict[str, list[JobRecord]]:
    """Bin-packs jobs across workers based on remaining quota and runtime estimates.

    Rules (C3):
    1. A job/batch requires est + 1.5 h <= that account's remaining quota.
    2. Assign the largest jobs (est >= 1.0h, e.g. aasist ~3h, effnet-b4 ~2h, davidnet-e2e ~1.5h)
       first to the accounts with the most remaining quota.
    3. Batch small jobs per account with the sum(est) + 1.5 h <= remaining rule,
       up to max_batch_hours per batch.
    4. Topological ordering: Stage 0 prerequisites must precede Stage 1 dependents.
    """
    from src.orchestrator.scheduler import RUNTIME_ESTIMATES_HOURS

    available_quotas = dict(workers_quota)
    assignments: dict[str, list[JobRecord]] = {w: [] for w in workers_quota}

    large_jobs = []
    small_jobs = []
    for j in jobs:
        est = RUNTIME_ESTIMATES_HOURS.get(j.name, 0.5)
        if est >= 1.0:
            large_jobs.append(j)
        else:
            small_jobs.append(j)

    # Sort large jobs descending by estimate
    large_jobs.sort(key=lambda j: RUNTIME_ESTIMATES_HOURS.get(j.name, 0.5), reverse=True)

    # Step 1: Assign largest jobs to workers with the most remaining quota
    for j in large_jobs:
        est = RUNTIME_ESTIMATES_HOURS.get(j.name, 0.5)
        candidates = sorted(
            [w for w, q in available_quotas.items() if q >= est + safety_margin_hours],
            key=lambda w: available_quotas[w],
            reverse=True,
        )
        if candidates:
            chosen = candidates[0]
            assignments[chosen].append(j)
            available_quotas[chosen] -= est

    # Step 2: Batch small jobs per account with sum + 1.5h rule
    for j in small_jobs:
        est = RUNTIME_ESTIMATES_HOURS.get(j.name, 0.5)
        candidates = []
        for w, q in available_quotas.items():
            curr_batch = assignments[w]
            curr_est = sum(RUNTIME_ESTIMATES_HOURS.get(other.name, 0.5) for other in curr_batch)
            if curr_est + est <= max_batch_hours and q >= est + safety_margin_hours:
                candidates.append((w, q))

        candidates.sort(key=lambda item: item[1], reverse=True)
        if candidates:
            chosen = candidates[0][0]
            assignments[chosen].append(j)
            available_quotas[chosen] -= est

    # Ensure Stage 0 strictly precedes Stage 1 in each assigned batch
    stage_prio = {"qacp": 0, "baseline": 1, "stage1": 2, "phase_b": 3, "external": 4}
    for w, batch in assignments.items():
        if batch:
            batch.sort(key=lambda j: (j.seed, stage_prio.get(j.stage, 9)))
            assignments[w] = batch

    return {w: b for w, b in assignments.items() if b}


def build_batch_script(jobs: Sequence[JobRecord], worker_name: str, pinned_sha: str) -> str:
    """Generates run_job.py for sequential execution of multiple jobs."""
    jobs_json = json.dumps([
        {
            "job_id": j.job_id,
            "name": j.name,
            "seed": j.seed,
            "stage": j.stage,
            "init_from": j.init_from or ""
        }
        for j in jobs
    ], indent=2)

    script = f'''# Auto-generated by Kaggle Distributed Batch Runner
import os, sys, subprocess, json, threading, time
from datetime import datetime, timezone

# Ensure unbuffered standard output for Kaggle log capture
os.environ["PYTHONUNBUFFERED"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

# 60s Watchdog heartbeat daemon thread
def _watchdog_heartbeat():
    t0 = time.time()
    while True:
        time.sleep(60.0)
        elapsed = time.time() - t0
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"[WATCHDOG HEARTBEAT] {{now_utc}} | Elapsed: {{elapsed:.1f}}s ({{elapsed/60.0:.1f}}m)", flush=True)

_hb_thread = threading.Thread(target=_watchdog_heartbeat, daemon=True)
_hb_thread.start()

print("=== [BATCH RUNNER] Initializing batch execution for {len(jobs)} jobs on {worker_name} ===", flush=True)

# Resolve Hugging Face token securely via Kaggle Secrets or attached dataset file
try:
    from kaggle_secrets import UserSecretsClient
    sec = UserSecretsClient()
    sec_token = sec.get_secret("HF_TOKEN")
    if sec_token:
        os.environ["HF_TOKEN"] = sec_token
        print("=== [WORKER] HF token resolved from Kaggle Secrets ===", flush=True)
except Exception:
    pass

if not os.environ.get("HF_TOKEN"):
    import glob
    print("=== [WORKER] Searching for attached token file in /kaggle/input ===", flush=True)
    token_candidates = sorted(set(
        glob.glob("/kaggle/input/**/hf_token.txt", recursive=True) +
        glob.glob("/kaggle/input/**/*token*.txt", recursive=True)
    ))
    for p in token_candidates:
        try:
            tok = open(p, encoding="utf-8").read().strip()
            if tok and not tok.startswith("#"):
                os.environ["HF_TOKEN"] = tok
                print(f"=== [WORKER] HF token resolved from attached private dataset: {{p}} ===", flush=True)
                break
        except Exception as e:
            print(f"Failed to read token candidate {{p}}: {{e}}", flush=True)

# Clone and verify pinned code revision
pinned_sha = "{pinned_sha}"
if not os.path.exists("Thesis"):
    subprocess.run(["git", "clone", "https://github.com/MIHMahmudEli/Thesis.git"], check=True)

os.chdir("Thesis")
subprocess.run(["git", "fetch", "--all"], check=True)
subprocess.run(["git", "checkout", pinned_sha], check=True)

actual_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
print(f"=== [BATCH WORKER] Checked out commit: {{actual_sha}} ===", flush=True)

batch_jobs = {jobs_json}

results = {{}}
for idx, job in enumerate(batch_jobs, 1):
    print(f"\\n=======================================================", flush=True)
    print(f"=== [BATCH RUNNER] ({{idx}}/{{len(batch_jobs)}}) Starting {{job['job_id']}} ===", flush=True)
    print(f"=======================================================\\n", flush=True)

    env = os.environ.copy()
    env["JOB_ID"] = job["job_id"]
    env["EXP_NAME"] = job["name"]
    env["EXP_SEED"] = str(job["seed"])
    env["EXP_STAGE"] = job["stage"]
    env["WORKER_NAME"] = "{worker_name}"

    cmd = [
        sys.executable, "-m", "src.pipeline.main",
        "--mode", "full",
        "--only", job["name"],
        "--seed", str(job["seed"]),
        "--worker", "{worker_name}"
    ]
    print("Running:", " ".join(cmd), flush=True)
    try:
        res = subprocess.run(cmd, env=env, timeout=5400)
    except subprocess.TimeoutExpired:
        print(f"FATAL: Job {{job['job_id']}} exceeded 5400s timeout!", file=sys.stderr, flush=True)
        class _TimedOut:
            returncode = 124
        res = _TimedOut()
    except Exception as ex:
        print(f"FATAL: Job {{job['job_id']}} failed with exception: {{ex}}", file=sys.stderr, flush=True)
        class _Err:
            returncode = 1
        res = _Err()

    print(f"=== [BATCH RUNNER] {{job['job_id']}} exited with code {{res.returncode}} ===", flush=True)
    results[job["job_id"]] = res.returncode
    if res.returncode != 0:
        print(f"WARNING: Job {{job['job_id']}} failed. Continuing with remaining batch jobs...", flush=True)

print("\\n=== [BATCH RUNNER] All batch jobs finished ===", flush=True)
for jid, code in results.items():
    print(f"  {{jid}}: code {{code}}", flush=True)

if any(c != 0 for c in results.values()):
    sys.exit(1)
sys.exit(0)
'''
    return script


class BatchRunner:
    """Manages creation and dispatch of batch kernels for cheap jobs."""

    def __init__(self, account: KaggleAccount, dry_run: bool = False):
        self.account = account
        self.dry_run = dry_run
        self.runner = KaggleRunner(account, dry_run=dry_run)

    def prepare_batch_workspace(self, batch_id: str, jobs: Sequence[JobRecord], pinned_sha: str) -> Path:
        """Prepares workspace and metadata for batch execution."""
        if not self.dry_run:
            self.runner.ensure_secret_dataset()

        slug = f"davidnet-batch-{batch_id.lower().replace('_', '-')}"
        ws_dir = WORKSPACE_DIR / self.account.worker_name / f"batch_{batch_id}"
        ws_dir.mkdir(parents=True, exist_ok=True)

        user_datasets = list(DATASET_SLUGS)
        secret_ds = f"{self.account.username}/davidnet-hf-token"
        if secret_ds not in user_datasets:
            user_datasets.append(secret_ds)

        meta = {
            "id": f"{self.account.username}/{slug}",
            "title": slug.replace("-", " "),
            "code_file": "run_job.py",
            "language": "python",
            "kernel_type": "script",
            "is_gpu": "true",
            "enable_gpu": True,
            "enable_internet": True,
            "is_private": "true",
            "dataset_sources": user_datasets,
            "kernel_sources": []
        }
        (ws_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        script_content = build_batch_script(jobs, self.account.worker_name, pinned_sha)
        (ws_dir / "run_job.py").write_text(script_content, encoding="utf-8")
        return ws_dir

    def push_batch(self, ws_dir: Path) -> tuple[bool, str]:
        """Pushes the batch kernel using Kaggle CLI."""
        return self.runner.push_kernel(ws_dir)
