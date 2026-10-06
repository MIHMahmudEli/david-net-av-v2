"""Command Line Interface for managing the distributed Kaggle worker system."""
from __future__ import annotations

import argparse
import sys
import time

from src.orchestrator.accounts import load_accounts_from_env, verify_account
from src.orchestrator.db import JobRecord, OrchestratorDB
from src.orchestrator.scheduler import MultiWorkerScheduler
from src.pipeline.config import DEFAULT_CONFIG, build_config
from src.pipeline.plan import full_plan, plan_table, select


def cmd_init_plan(args):
    db = OrchestratorDB()
    plan = full_plan()
    if getattr(args, "only", None):
        plan = select(plan, only=args.only)
    seeds = getattr(args, "seeds", None) or [42, 123, 456]
    base_cfg = build_config()

    rows = plan_table(plan, seeds, base_cfg)
    jobs: list[JobRecord] = []
    for r in rows:
        job_id = f"{r['name']}_s{r['seed']}"
        # Priority: QACP stage 0 higher than stage 1
        priority = 20 if r["stage"] == "qacp" else (15 if r["stage"] == "baseline" else 10)
        jobs.append(JobRecord(
            job_id=job_id,
            name=r["name"],
            seed=r["seed"],
            stage=r["stage"],
            group=r["group"],
            split=r["split"],
            init_from=r["init_from"],
            config_hash=r["config_hash"],
            priority=priority,
            status="PENDING"
        ))

    db.insert_jobs(jobs)
    print(f"Successfully populated database with {len(jobs)} jobs from pipeline plan.")


def cmd_status(args):
    scheduler = MultiWorkerScheduler()
    scheduler.print_dashboard()


def cmd_reset(args):
    db = OrchestratorDB()
    with db.get_connection() as conn:
        conn.execute("UPDATE workers SET status = 'IDLE', current_job_id = NULL, cooldown_until = 0.0;")
        conn.execute("UPDATE jobs SET status = 'PENDING', worker_name = NULL, lease_expires_at = 0.0 WHERE status != 'COMPLETED';")
    print("Reset all workers to IDLE and pending jobs to PENDING.")


def cmd_sync_hf(args):
    import json
    import os
    from pathlib import Path
    from huggingface_hub import hf_hub_download
    from src.pipeline.hub import resolve_hf_token

    db = OrchestratorDB()
    reg_path = Path("report/.cache/audit/registry__experiments.json")
    reg_path.parent.mkdir(parents=True, exist_ok=True)

    token = None
    try:
        token = resolve_hf_token()
    except Exception:
        token = os.getenv("HF_TOKEN")

    try:
        downloaded = hf_hub_download(
            repo_id="MIHMahmudEli/davidnet-experiments",
            repo_type="model",
            filename="registry/experiments.json",
            token=token,
            force_download=True
        )
        content = Path(downloaded).read_text(encoding="utf-8")
        reg_path.write_text(content, encoding="utf-8")
    except Exception as e:
        print(f"Warning: could not download latest registry from HF ({e}), using local cache if present.")

    if not reg_path.exists():
        print("HF registry cache not found.")
        return

    reg = json.loads(reg_path.read_text(encoding="utf-8"))
    experiments = reg.get("experiments", {})
    completed_ids = [
        f"{e['name']}_s{e['seed']}"
        for e in experiments.values()
        if (e.get("status") or "").lower() == "completed"
    ]
    with db.get_connection() as conn:
        updated = 0
        for jid in completed_ids:
            cur = conn.execute("UPDATE jobs SET status = 'COMPLETED', completed_at = 1.0 WHERE job_id = ? AND status != 'COMPLETED'", (jid,))
            updated += cur.rowcount
        conn.execute("UPDATE workers SET status = 'IDLE', current_job_id = NULL WHERE current_job_id IN (SELECT job_id FROM jobs WHERE status = 'COMPLETED')")
    print(f"Synced with Hugging Face registry: {updated} runs updated to COMPLETED ({len(completed_ids)} total in registry).")



def cmd_verify_accounts(args):
    print("Verifying all Kaggle accounts from .env...")
    accounts = load_accounts_from_env()
    for a in accounts:
        print(f"Checking {a.worker_name} ({a.username})...", end=" ", flush=True)
        ok = verify_account(a)
        if ok:
            print("OK!")
        else:
            print(f"FAILED ({a.last_error})")


def cmd_run(args):
    scheduler = MultiWorkerScheduler(dry_run=args.dry_run)
    print(f"Starting Kaggle Distributed Scheduler loop (Dry run: {args.dry_run})...", flush=True)
    try:
        while True:
            scheduler.print_dashboard()
            results = scheduler.step_all_workers()
            for w, msg in results.items():
                if msg and "No eligible jobs" not in msg:
                    print(f"[{w}] {msg}", flush=True)
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nScheduler stopped by user.", flush=True)


def cmd_reports(args):
    """Centralized report generation: downloads completed run results and builds tables/figures."""
    import os
    from src.scheduler.config import hf_token
    if not os.environ.get("HF_TOKEN"):
        tok = hf_token()
        if tok:
            os.environ["HF_TOKEN"] = tok
    from src.pipeline.driver import Session
    print("=== [ORCHESTRATOR] Running Central Report Generation ===")
    cfg = build_config()
    cfg["mode"] = getattr(args, "mode", "full")
    s = Session(cfg, repo_dir=".")
    s.connect()
    s.build_reports()
    s.final_report()
    print("=== [ORCHESTRATOR] Central Reports Built Successfully ===")


def main():
    parser = argparse.ArgumentParser(description="Kaggle Distributed Worker Orchestrator")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init-plan", help="Populate queue from pipeline plan")
    p_init.add_argument("--seeds", type=int, nargs="+", default=None, help="Seeds to run (default: 42 123 456)")
    p_init.add_argument("--only", nargs="+", default=None, help="Only schedule specific experiment names")
    p_init.set_defaults(func=cmd_init_plan)

    p_status = sub.add_parser("status", help="Show worker and queue dashboard")
    p_status.set_defaults(func=cmd_status)

    p_verify = sub.add_parser("verify-accounts", help="Authenticate all 16 accounts")
    p_verify.set_defaults(func=cmd_verify_accounts)

    p_reset = sub.add_parser("reset", help="Reset all workers to IDLE and clear cooldowns")
    p_reset.set_defaults(func=cmd_reset)

    p_sync = sub.add_parser("sync-hf", help="Sync completed experiments from Hugging Face registry")
    p_sync.set_defaults(func=cmd_sync_hf)

    p_reports = sub.add_parser("reports", help="Centrally build reports and figures from completed runs")
    p_reports.add_argument("--mode", default="full", help="Pipeline mode (default: full)")
    p_reports.set_defaults(func=cmd_reports)

    p_run = sub.add_parser("run", help="Start the worker orchestration loop")
    p_run.add_argument("--dry-run", action="store_true", help="Simulate without pushing Kaggle kernels")
    p_run.add_argument("--once", action="store_true", help="Execute one tick and exit")
    p_run.add_argument("--interval", type=int, default=30, help="Seconds between loop ticks")
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
