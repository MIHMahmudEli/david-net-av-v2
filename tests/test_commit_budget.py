"""Commit Budget Verification Test (B3).

Simulates a 5-job batch execution using a FakeStore tracking every commit and update_json call.
Asserts that:
1. Intermediate step checkpoints are written locally and NOT uploaded as commits.
2. Heartbeats do not trigger commits.
3. Total commits for the 5-job batch stay strictly below the safety budget (<= 15 commits for 5 jobs).
4. Fails if the commit count exceeds the budget.
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest
import torch

from src.pipeline.checkpoint import CheckpointManager
from src.pipeline.registry import Registry


class FakeHubStore:
    def __init__(self):
        self.commits: list[dict] = []
        self.files: dict[str, str] = {}

    def commit(self, additions: dict, message: str = "", delete_folders=None):
        self.commits.append({
            "type": "commit",
            "message": message,
            "file_count": len(additions),
            "files": list(additions.keys())
        })

    def update_json(self, path: str, fn, message: str = ""):
        current = json.loads(self.files.get(path, "{}")) if path in self.files else None
        updated = fn(current)
        self.files[path] = json.dumps(updated)
        self.commits.append({
            "type": "update_json",
            "path": path,
            "message": message
        })

    def read_json(self, path: str):
        if path in self.files:
            return json.loads(self.files[path])
        return None

    def exists(self, path: str) -> bool:
        return path in self.files


def test_5_job_batch_commit_budget(tmp_path: Path):
    """Verifies that a 5-job batch produces <= 15 commits total on the HubStore."""
    store = FakeHubStore()
    registry = Registry(store, namespace="batch_test/")
    
    # Target budget: <= 3 commits per job on average (max 15 commits for 5 jobs)
    MAX_BUDGET_5_JOBS = 15

    for job_idx in range(1, 6):
        job_name = f"model_{job_idx}"
        exp_dir = f"batch_test/experiments/EXP_{job_idx:03d}_{job_name}_s42"
        local_dir = tmp_path / f"EXP_{job_idx:03d}"
        local_dir.mkdir(parents=True, exist_ok=True)

        # 1. Register & Claim atomically in a single commit (B3)
        entry = registry.register(
            mode="full", name=job_name, seed=42, config_hash=f"hash_{job_idx}",
            meta={"stage": "baseline", "split": "strict"}, claim=True
        )

        # 2. CheckpointManager with upload_steps=False (local only)
        ckpt_mgr = CheckpointManager(
            store=store, exp_dir=exp_dir, local_root=local_dir, upload_steps=False
        )

        # Simulate 10 training steps with checkpoints saved
        dummy_state = {"weights": torch.tensor([1.0, 2.0, 3.0])}
        for step in [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]:
            ckpt_mgr.save(dummy_state, step=step, epoch=step // 500, reason="interval")

        # Simulate multiple heartbeats (must NOT create commits)
        for hb in range(5):
            registry.heartbeat(entry["exp_id"], commit=False, progress_step=hb * 500)

        # Simulate best model
        ckpt_mgr.mark_best(dummy_state, {"step": 4000}, {"config": "val"})

        # 3. Simulate completion & results upload (1 commit)
        results_adds = {
            f"{exp_dir}/metrics/summary.json": str(local_dir / "summary.json"),
            f"{exp_dir}/predictions/test.csv": str(local_dir / "test.csv"),
            f"{exp_dir}/best_model/model.safetensors": str(local_dir / "model.safetensors"),
        }
        store.commit(results_adds, message=f"{entry['exp_id']}: results")

        # 4. Set status completed (1 commit)
        registry.set_status(entry["exp_id"], "completed", train_hours=0.5)

    total_commits = len(store.commits)
    print(f"\nTotal store commits for 5-job batch: {total_commits}")
    for idx, c in enumerate(store.commits, 1):
        print(f"  Commit {idx}: [{c['type']}] {c.get('message', '')}")

    # Assertions
    # 1. Step checkpoints must be 0 in the remote store
    step_commits = [c for c in store.commits if "checkpoint step" in c.get("message", "")]
    assert len(step_commits) == 0, f"Expected 0 remote step commits, got {len(step_commits)}"

    # 2. Heartbeat commits must be 0 in the remote store
    hb_commits = [c for c in store.commits if "heartbeat" in c.get("message", "")]
    assert len(hb_commits) == 0, f"Expected 0 heartbeat commits, got {len(hb_commits)}"

    # 3. Total commits must be <= 15 for 5 jobs
    assert total_commits <= MAX_BUDGET_5_JOBS, (
        f"Commit count {total_commits} exceeded maximum allowed budget of {MAX_BUDGET_5_JOBS} for 5 jobs!"
    )
