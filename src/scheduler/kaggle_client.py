"""Thin, safe wrapper around the official ``kaggle`` CLI (v2.x).

Verified behaviour (2026-10-04, kaggle CLI 2.2.4, KGAT_ settings-UI API tokens):

* auth          ``KAGGLE_API_TOKEN`` environment variable, set per invocation -
                never written to disk by us, never printed
* quota         ``kaggle quota``            -> GPU used/remaining/refresh
* list          ``kaggle kernels list -m``  -> own kernels
* push + run    ``kaggle kernels push -p DIR [--accelerator A]`` (creates or
                updates the kernel and starts a new version = Save & Run All)
* status        ``kaggle kernels status OWNER/SLUG``
* logs          ``kaggle kernels logs OWNER/SLUG``
* datasets      ``kaggle datasets create|version -p DIR``

NOT supported by the CLI and therefore handled elsewhere:
* secrets on push  (push detaches secrets) -> HF token is delivered via a
  private per-worker dataset, see kernel.py / hub.resolve_hf_token
* interrupt/stop   (no CLI command)        -> sessions self-terminate at the
  notebook time budget; the scheduler never relies on cancellation
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from typing import Optional

from .config import SETTINGS, WorkerCredential


class KaggleError(RuntimeError):
    pass


class RateLimited(KaggleError):
    pass


_RATE_MARKERS = ("429", "rate limit", "too many requests", "quota exceeded", "throttl")


# status normalization (spec section 10)
_STATUS_MAP = {
    "complete": "COMPLETED", "completed": "COMPLETED", "success": "COMPLETED",
    "running": "RUNNING", "in_progress": "RUNNING",
    "pending": "QUEUED", "queued": "QUEUED", "waiting": "QUEUED", "starting": "QUEUED",
    "failed": "FAILED", "error": "FAILED",
    "cancelled": "CANCELLED", "canceled": "CANCELLED",
}


def normalize_status(raw: str) -> str:
    token = raw.strip().strip('"').split(".")[-1].lower()
    return _STATUS_MAP.get(token, "UNKNOWN")


@dataclass
class Quota:
    gpu_used_h: float
    gpu_remaining_h: float
    gpu_total_h: float
    refresh_at: str | None


class KaggleClient:
    """One client per worker account; every call runs a fresh CLI process."""

    def __init__(self, worker: WorkerCredential, executable: str = "kaggle"):
        self.worker = worker
        self.executable = executable

    # ---------------------------------------------------------------- core
    def _run(self, args: list[str], timeout: float = 180) -> tuple[int, str, str]:
        import os

        env = {**os.environ, **self.worker.kaggle_env()}
        env.pop("KAGGLE_USERNAME", None)
        env.pop("KAGGLE_KEY", None)
        try:
            proc = subprocess.run(
                [self.executable, *args],
                capture_output=True, text=True, timeout=timeout, env=env,
            )
        except subprocess.TimeoutExpired as e:
            raise KaggleError(f"kaggle {' '.join(args)} timed out after {timeout}s") from e
        out = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode != 0:
            low = out.lower()
            if any(m in low for m in _RATE_MARKERS):
                raise RateLimited(f"kaggle {' '.join(args)}: {_redact(out)}")
            raise KaggleError(f"kaggle {' '.join(args)} rc={proc.returncode}: {_redact(out)}")
        return proc.returncode, proc.stdout or "", proc.stderr or ""

    # ------------------------------------------------------------ endpoints
    def quota(self) -> Quota:
        _, out, _ = self._run(["quota"], timeout=60)
        m = re.search(r"GPU\s+([\d.]+)h\s+([\d.]+)h\s+([\d.]+)h\s+(\S+)", out)
        if not m:
            raise KaggleError(f"cannot parse quota output: {_redact(out)}")
        return Quota(float(m.group(1)), float(m.group(2)), float(m.group(3)), m.group(4))

    def kernel_status(self, kernel_ref: str) -> tuple[str, str]:
        """Return (normalized_status, raw_status_text)."""
        try:
            _, out, _ = self._run(["kernels", "status", kernel_ref], timeout=60)
        except KaggleError as e:
            msg = str(e)
            if "404" in msg or "not found" in msg.lower():
                return "UNKNOWN", "kernel not found"
            raise
        m = re.search(r'status\s+"([^"]+)"', out)
        raw = m.group(1) if m else out.strip()
        return normalize_status(raw), raw

    def push(self, folder: str, accelerator: str | None = None) -> str:
        args = ["kernels", "push", "-p", folder]
        acc = accelerator or SETTINGS.accelerator
        if acc:
            args += ["--accelerator", acc]
        _, out, _ = self._run(args, timeout=600)
        return out

    def list_mine(self) -> str:
        _, out, _ = self._run(["kernels", "list", "-m", "--page-size", "50"], timeout=90)
        return out

    def logs(self, kernel_ref: str) -> str:
        _, out, _ = self._run(["kernels", "logs", kernel_ref], timeout=90)
        return out

    def datasets_create(self, folder: str) -> str:
        _, out, _ = self._run(["datasets", "create", "-p", folder, "-q"], timeout=600)
        return out

    def datasets_update(self, folder: str, message: str = "update") -> str:
        # `kaggle datasets version` requires -m
        _, out, _ = self._run(["datasets", "version", "-p", folder, "-q", "-r", "skip",
                               "-m", message], timeout=600)
        return out


def _redact(text: str) -> str:
    """Strip anything token-shaped before it can reach a log or exception message."""
    text = re.sub(r"KGAT_[A-Za-z0-9]+", "KGAT_***", text)
    text = re.sub(r"hf_[A-Za-z0-9]{10,}", "hf_***", text)
    text = re.sub(r"ghp_[A-Za-z0-9]{10,}", "ghp_***", text)
    return text.strip()
