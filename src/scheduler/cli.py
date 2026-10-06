"""Command line interface: ``python -m src.scheduler <command>``.

Commands
--------
status   dashboard from the local state DB (registry-synced progress, workers, jobs)
plan     dry-run: what would be dispatched right now (no claims, no pushes)
sync     force a registry sync from Hugging Face
validate credentials / CLI / quota sanity check (never prints a token)
run      the loop: sync -> monitor -> dispatch -> sleep   (``--once`` for one pass)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Optional

from .config import SETTINGS, load_workers, hf_token
from .kaggle_client import KaggleError, RateLimited
from .scheduler import Scheduler

PROG = "python -m src.scheduler"


def _fmt_row(cols: list, widths: list[int]) -> str:
    return "  ".join(str(c).ljust(w) for c, w in zip(cols, widths)).rstrip()


def _table(rows: list[list], headers: list[str]) -> str:
    if not rows:
        return _fmt_row(headers, [max(len(h), 8) for h in headers])
    widths = [max(len(str(r[i])) for r in [headers, *rows]) for i in range(len(headers))]
    lines = [_fmt_row(headers, widths), _fmt_row(["-" * w for w in widths], widths)]
    lines += [_fmt_row([str(c) for c in r], widths) for r in rows]
    return "\n".join(lines)


# ------------------------------------------------------------------------ status
def cmd_status(sched: Scheduler, args: argparse.Namespace) -> int:
    if getattr(args, "sync", False):
        sched.sync_registry(force=True)
    snap = sched.snapshot()
    if getattr(args, "json", False):
        print(json.dumps(snap, indent=2, default=str))
        return 0
    c = snap["campaign"]
    print(f"campaign  {c['total']} experiments | {c['completed']} completed "
          f"({c['progress']}%) | {c['remaining']} remaining")
    if c["counts"]:
        print("          " + "  ".join(f"{k}={v}" for k, v in sorted(c["counts"].items())))
    print("jobs      " + ("  ".join(f"{k}={v}" for k, v in sorted(snap["job_counts"].items()))
                          or "none"))
    print()
    rows = []
    for w in snap["workers"]:
        cd = ""
        if w["cooldown_until"] and w["cooldown_until"] > time.time():
            cd = time.strftime("%H:%M", time.localtime(w["cooldown_until"]))
        rows.append([w["name"], w["username"], w["status"], w["job"] or "-",
                     f"{w['gpu_left']:.1f}h" if w["gpu_left"] is not None else "?",
                     w["completed"], w["failed"], cd,
                     (w["last_error"] or "")[:40]])
    print(_table(rows, ["worker", "account", "state", "job", "gpu", "ok", "fail",
                        "cooldown", "last_error"]))
    print()
    rows = [[j["id"], j["status"], j["worker"] or "-", j["kaggle"] or "-", j["attempts"],
             (j["kernel"] or "-"), (j["error"] or "")[:48]] for j in snap["jobs"][-15:]]
    print(_table(rows, ["job", "state", "worker", "kaggle", "try", "kernel", "error"]))
    if getattr(args, "events", False):
        print()
        for e in sched.store.events(15):
            print(f"  {time.strftime('%m-%d %H:%M:%S', time.localtime(e['ts']))} "
                  f"{e['kind']:<18} {e['subject'] or ''} {(e['detail'] or '')[:90]}")
    return 0


# -------------------------------------------------------------------------- plan
def cmd_plan(sched: Scheduler, args: argparse.Namespace) -> int:
    sched.sync_registry(force=True)
    sched.reconcile()
    snap = sched.snapshot()
    total, remaining = snap["campaign"]["total"], snap["campaign"]["remaining"]
    if total == 0:
        print(f"no registry rows for mode {SETTINGS.mode} yet - bootstrap allows one session")
    else:
        print(f"remaining work: {remaining} experiments "
              f"({snap['campaign']['completed']}/{total} done)")
        if remaining == 0:
            print("campaign complete - nothing to dispatch")
            return 0
    would = sched.dispatch(dry_run=True)
    if not would:
        print("no idle workers right now (all busy or cooling down)")
    for line in would:
        print("  " + line)
    print(f"kernel: {SETTINGS.kernel_slug} | accelerator: {SETTINGS.accelerator} | "
          f"mode: {SETTINGS.mode} | data inputs: {len(SETTINGS.data_datasets)}")
    return 0


# -------------------------------------------------------------------------- sync
def cmd_sync(sched: Scheduler, args: argparse.Namespace) -> int:
    ok = sched.sync_registry(force=True)
    snap = sched.snapshot()
    print(("synced" if ok else "sync failed (see events)")
          + f": {snap['campaign']['completed']}/{snap['campaign']['total']} completed, "
            f"{snap['campaign']['remaining']} remaining")
    return 0 if ok else 1


# ---------------------------------------------------------------------- validate
def cmd_validate(sched: Scheduler, args: argparse.Namespace) -> int:
    failures: list[str] = []
    workers = load_workers()
    print(f".env workers: {len(workers)}")
    if not workers:
        failures.append("no Kaggle credentials parsed from .env")

    import shutil
    if not shutil.which("kaggle"):
        failures.append("`kaggle` CLI not found on PATH")
    else:
        import subprocess
        try:
            ver = subprocess.run(["kaggle", "--help"], capture_output=True, text=True, timeout=60)
            print(f"kaggle CLI : rc={ver.returncode}")
            if ver.returncode != 0:
                failures.append("kaggle CLI not working")
        except (OSError, subprocess.TimeoutExpired) as e:
            failures.append(f"kaggle CLI error: {e}")

    print(f"HF token   : {'present' if hf_token() else 'MISSING (needed for registry sync)'}")
    if not hf_token():
        failures.append("no HF token in .env")

    # per-account quota (a real authenticated call per worker)
    limit = getattr(args, "accounts", None)
    sample = workers[:limit] if limit else workers
    rows = []
    for w in sample:
        try:
            q = sched._client(w.name).quota()
            rows.append([w.name, w.username, f"{q.gpu_used_h:.1f}h", f"{q.gpu_remaining_h:.1f}h",
                         f"{q.gpu_total_h:.1f}h", q.refresh_at or "-"])
        except RateLimited as e:
            rows.append([w.name, w.username, "RATE LIMITED", "-", "-", str(e)[:40]])
            failures.append(f"{w.name}: rate limited")
        except KaggleError as e:
            rows.append([w.name, w.username, "ERROR", "-", "-", str(e)[:60]])
            failures.append(f"{w.name}: {e}")
    if rows:
        print()
        print(_table(rows, ["worker", "account", "used", "left", "total", "refresh"]))

    # push folder builds locally (no network)
    if not getattr(args, "skip_build", False) and workers:
        from .kernel import build_push_folder
        try:
            folder = build_push_folder(workers[0])
            meta = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
            print(f"\npush folder : {folder}")
            print(f"kernel id   : {meta['id']} | inputs: {len(meta['dataset_sources'])} datasets")
        except Exception as e:  # noqa: BLE001
            failures.append(f"push folder build failed: {e}")

    print()
    if failures:
        print("FAILED:")
        for f in failures:
            print("  - " + f)
        return 1
    print("all checks passed")
    return 0


# --------------------------------------------------------------------------- run
def cmd_run(sched: Scheduler, args: argparse.Namespace) -> int:
    dry = bool(args.dry_run)
    try:
        tick = sched.run(once=args.once, dry_run=dry)
    except KeyboardInterrupt:
        print("\nstopped (state kept in " + str(SETTINGS.db_path) + ")")
        return 130
    d = tick.as_dict()
    print(json.dumps(d, indent=2))
    if d["errors"]:
        for e in d["errors"]:
            print("  ! " + str(e))
    return 0


# ----------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=PROG, description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("status", help="dashboard from the local state DB")
    sp.add_argument("--sync", action="store_true", help="registry-sync first")
    sp.add_argument("--events", action="store_true", help="also print recent events")
    sp.add_argument("--json", action="store_true", help="full JSON snapshot")
    sp.set_defaults(fn=cmd_status)

    sp = sub.add_parser("plan", help="dry-run: what would be dispatched now")
    sp.set_defaults(fn=cmd_plan)

    sp = sub.add_parser("sync", help="force registry sync from Hugging Face")
    sp.set_defaults(fn=cmd_sync)

    sp = sub.add_parser("validate", help="credentials, CLI and quota sanity check")
    sp.add_argument("--accounts", type=int, default=None,
                    help="check only the first N accounts (default: all)")
    sp.add_argument("--skip-build", action="store_true", help="skip the local push-folder build")
    sp.set_defaults(fn=cmd_validate)

    sp = sub.add_parser("run", help="scheduler loop")
    sp.add_argument("--once", action="store_true", help="single tick then exit")
    sp.add_argument("--dry-run", action="store_true",
                    help="tick without claiming or pushing anything")
    sp.set_defaults(fn=cmd_run)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    sched = Scheduler()
    return args.fn(sched, args)


if __name__ == "__main__":       # pragma: no cover
    sys.exit(main())
