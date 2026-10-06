"""Read-only sync with the Hugging Face experiment registry (campaign truth).

The notebook (src/pipeline) owns experiment-level claims and checkpoints; this
module only *reads*:

* ``registry/experiments.json``   - every planned/completed experiment
* ``experiments/<dir>/checkpoints/LATEST.json`` - only checked for
  non-terminal experiments (a handful of HEAD-sized requests per sync)

Local DB mirrors the result so the dashboard keeps working offline; a failed
sync keeps the previous snapshot and records an event.
"""
from __future__ import annotations

import json
import time
from typing import Any

from huggingface_hub import HfApi

from .config import SETTINGS


class RegistryError(RuntimeError):
    pass


def _dir_name(e: dict) -> str:
    return f"{e['exp_id']}_{e['name']}_s{e['seed']}"


def _status(e: dict) -> str:
    s = (e.get("status") or "").lower()
    if s == "completed":
        return "completed"
    if s in ("running", "paused"):
        return "running"
    if s == "failed":
        return "failed"
    if s in ("claimed", "pending", ""):
        return "pending"
    return s


class RegistrySync:
    def __init__(self, repo_id: str | None = None, token: str | None = None,
                 mode: str | None = None):
        from .config import hf_token

        self.repo_id = repo_id or SETTINGS.hf_repo
        self.token = token or hf_token()
        self.mode = mode or SETTINGS.mode      # registry rows can mix smoke/full
        self.api = HfApi(token=self.token)
        self.last_sync_at: float | None = None

    def fetch(self) -> list[dict]:
        """Download the registry and return mirrored experiment rows."""
        try:
            path = self.api.hf_hub_download(
                self.repo_id, "registry/experiments.json", repo_type="model"
            )
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:  # noqa: BLE001 - network / repo / json
            raise RegistryError(f"registry download failed: {type(e).__name__}: {e}") from e

        experiments: dict[str, dict] = data.get("experiments", data) if isinstance(data, dict) else {}
        rows: list[dict] = []
        checkpoint_probes: list[tuple[dict, str]] = []
        for e in experiments.values():
            if self.mode and e.get("mode") and e.get("mode") != self.mode:
                continue
            st = _status(e)
            dirname = _dir_name(e)
            row = {
                "key": e.get("key") or dirname,      # canonical registry key
                "exp_id": e.get("exp_id"),
                "name": e.get("name"),
                "seed": e.get("seed"),
                "grp": e.get("group"),
                "status": st,
                "claimed_by": e.get("claimed_by"),
                "test_clip_auc": e.get("test_clip_auc"),
                "train_hours": e.get("train_hours"),
                "has_checkpoint": 0,
            }
            # checkpoint presence only matters where a session can resume
            if st in ("running", "failed") or (st == "pending" and e.get("claimed_by")):
                checkpoint_probes.append((row, dirname))
            rows.append(row)

        for row, dirname in checkpoint_probes:
            if self._checkpoint_exists(dirname):
                row["has_checkpoint"] = 1

        self.last_sync_at = time.time()
        return rows

    def _checkpoint_exists(self, dirname: str) -> bool:
        try:
            self.api.hf_hub_download(
                self.repo_id, f"experiments/{dirname}/checkpoints/LATEST.json", repo_type="model"
            )
            return True
        except Exception:  # noqa: BLE001
            return False

    # ---------------------------------------------------------------- util
    @staticmethod
    def summarize(rows: list[dict]) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for r in rows:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        total = len(rows)
        done = counts.get("completed", 0)
        return {
            "total": total,
            "completed": done,
            "running": counts.get("running", 0),
            "pending": counts.get("pending", 0),
            "failed": counts.get("failed", 0),
            "progress": (100.0 * done / total) if total else 0.0,
        }
