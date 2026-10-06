"""[DEPRECATED] Distributed Kaggle worker pool for the DAVID-Net Q1 experiment campaign.
NOTE: This package is deprecated in favor of `src.orchestrator` as the canonical scheduler.
Do not use for new dispatches. Use `python -m src.orchestrator.cli` instead.

Layers:
    config.py        worker/account registry parsed from the repo's .env (never prints secrets)
    db.py            persistent SQLite job/worker state with atomic claiming
    kaggle_client.py thin wrapper over the official ``kaggle`` CLI (token via env per call)
    registry.py      read-only sync with the Hugging Face experiment registry (source of truth)
    kernel.py        builds the push folder (notebook + kernel-metadata.json) per worker
    scheduler.py     reconcile -> dispatch -> monitor -> recover loop
    cli.py           status / plan / sync / run / validate commands

Design notes
------------
* Experiment-level dedup is owned by the *existing* HF claim/lease system in
  ``src/pipeline`` (the notebook claims the next pending experiment atomically).
  The scheduler therefore assigns *sessions* (one kernel run on one idle account)
  and never runs two sessions on the same account at once.
* Secrets: the ``kaggle kernels push`` API detaches notebook secrets, so the HF
  token is delivered through a small private Kaggle dataset attached to each
  worker's kernel and read by ``src.pipeline.hub.resolve_hf_token`` as a fallback.
"""
from __future__ import annotations

import warnings

warnings.warn("src.scheduler is deprecated; use src.orchestrator as the canonical scheduler.",
              DeprecationWarning, stacklevel=2)
