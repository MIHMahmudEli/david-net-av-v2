"""Worker/account configuration for the Kaggle scheduler.

Credentials live in the repo's ``.env`` (gitignored). This module parses it
tolerantly (the file contains hand-typed keys such as ``Kaggel_Tokhon=``) and
exposes tokens only through objects that redact themselves in reprs/logs.

Nothing in this module ever prints a token.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from src.pipeline.revision import get_pinned_revision
# Dataset slugs are defined once in src/pipeline/manifests.py::KAGGLE_SLUGS (canonical config).
from src.pipeline.manifests import KAGGLE_SLUGS as _KAGGLE_SLUGS

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = Path(os.environ.get("SCHED_ENV_FILE", REPO_ROOT / ".env"))

# .env has been edited by hand: tolerate Kaggel_Token / Kaggel_Tokhon / kaggle_token
# spellings (and the odd trailing space before '='). Verified against the real file:
# 15 pairs, keys Kaggel_Token, Kaggel_Tokhon and "Kaggel_Token ".
_TOKEN_KEYS = re.compile(r"^kagg(?:le|el)[ _-]*tok(?:en|hon)\s*=\s*(.*)$", re.I)
_USER_KEYS = re.compile(r"^username\s*=\s*(.*)$", re.I)

_TOKEN_STRIP = "\"' \t\r\n"


def _unquote(v: str) -> str:
    v = v.strip().strip("\"'")
    return v.strip()


@dataclass(frozen=True)
class WorkerCredential:
    """One Kaggle account. ``token`` is never included in repr/str/logs."""

    name: str            # kaggle-worker-1 ...
    username: str        # Kaggle account (kernel owner)
    token: str = field(repr=False)
    env_index: int       # position of the credential pair in .env

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"WorkerCredential(name={self.name!r}, username={self.username!r}, token='***')"

    def kaggle_env(self) -> dict[str, str]:
        """Environment overlay for one ``kaggle`` CLI invocation."""
        return {"KAGGLE_API_TOKEN": self.token}


def parse_env_credentials(path: Path = ENV_PATH) -> list[tuple[str, str]]:
    """Return ordered (username, token) pairs from .env. Tolerates key-name typos."""
    if not path.exists():
        return []
    pairs: list[tuple[str, str]] = []
    pending_user: str | None = None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), _unquote(val)
        if _USER_KEYS.match(key + "="):
            pending_user = val
        elif _TOKEN_KEYS.match(line) and pending_user:
            pairs.append((pending_user, val))
            pending_user = None
    return pairs


def worker_names(count: int) -> list[str]:
    """Worker names are configurable; default is kaggle-worker-1..N (N free to grow)."""
    override = os.environ.get("SCHED_WORKER_NAMES", "").strip()
    if override:
        names = [n.strip() for n in override.split(",") if n.strip()]
        if len(names) < count:
            names += [f"kaggle-worker-{i}" for i in range(len(names) + 1, count + 1)]
        return names[: max(count, len(names))]
    return [f"kaggle-worker-{i}" for i in range(1, count + 1)]


def load_workers(path: Path = ENV_PATH) -> list[WorkerCredential]:
    pairs = parse_env_credentials(path)
    names = worker_names(len(pairs))
    return [
        WorkerCredential(name=names[i], username=user, token=tok, env_index=i)
        for i, (user, tok) in enumerate(pairs)
    ]


def hf_token(path: Path = ENV_PATH) -> str | None:
    """The HF write token from .env (used only to create the per-worker secret dataset)."""
    if not path.exists():
        return None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if raw.strip().lower().startswith("hf_token="):
            return _unquote(raw.partition("=")[2]) or None
    return None


# --------------------------------------------------------------------------- #
# tunables (env-overridable, all safe to log)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Settings:
    poll_interval_s: float = float(os.environ.get("SCHED_POLL_INTERVAL", "90"))
    quota_min_hours: float = float(os.environ.get("SCHED_QUOTA_MIN_HOURS", "1.0"))
    quota_refresh_s: float = float(os.environ.get("SCHED_QUOTA_REFRESH", "300"))
    max_attempts: int = int(os.environ.get("SCHED_MAX_ATTEMPTS", "3"))
    cooldown_fail_s: float = float(os.environ.get("SCHED_COOLDOWN_FAIL", "600"))
    lease_s: float = float(os.environ.get("SCHED_LEASE_S", "3600"))
    stale_after_s: float = float(os.environ.get("SCHED_STALE_AFTER_S", "2700"))
    kernel_slug: str = os.environ.get("SCHED_KERNEL_SLUG", "davidnet-q1-pipeline")
    secret_dataset_slug: str = os.environ.get("SCHED_SECRET_SLUG", "davidnet-hf-token")
    accelerator: str = os.environ.get("SCHED_ACCELERATOR", "NvidiaTeslaT4")
    # Dataset slugs are sourced from src/pipeline/manifests.py::KAGGLE_SLUGS (canonical config).
    data_datasets: tuple[str, ...] = tuple(_KAGGLE_SLUGS.values())
    hf_repo: str = os.environ.get("SCHED_HF_REPO", "MIHMahmudEli/davidnet-experiments")
    mode: str = os.environ.get("SCHED_MODE", "full")
    code_revision: str = os.environ.get("SCHED_CODE_REVISION") or get_pinned_revision()
    db_path: Path = Path(os.environ.get("SCHED_DB", REPO_ROOT / "scheduler" / "state.db"))
    work_dir: Path = Path(os.environ.get("SCHED_WORK_DIR", REPO_ROOT / "scheduler" / "work"))


SETTINGS = Settings()
