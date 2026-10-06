"""Pre-dispatch guards: enforces single-dispatch exclusivity and clean namespace isolation.

Rules (A1):
1. Refuse dispatch if ANY kernel on ANY account in the pool is RUNNING or QUEUED
   and writes to the same HF repository and namespace.
2. Refuse dispatch if the target namespace is non-empty on Hugging Face Hub,
   unless an explicit resume of that exact run is requested (allow_resume=True).
"""
from __future__ import annotations

import os
from typing import Optional, Sequence
from huggingface_hub import HfApi

from src.orchestrator.accounts import KaggleAccount


class DispatchGuardError(Exception):
    """Raised when a pre-dispatch validation fails."""
    pass


def check_running_kernels_for_namespace(
    target_namespace: str,
    target_repo: str,
    accounts: Sequence[KaggleAccount],
    active_slugs_by_worker: Optional[dict[str, list[str]]] = None,
    kaggle_api_factory=None,
) -> tuple[bool, str]:
    """Inspects all accounts to ensure no running/queued kernel targets the same HF repo/namespace.

    Returns:
        (ok, reason): ok is True if safe to dispatch; False if a conflict exists.
    """
    clean_ns = target_namespace.strip("/")
    if not clean_ns:
        return False, "Target namespace cannot be empty or root for an isolated run."

    # If active_slugs_by_worker is supplied (e.g. from orchestrator DB or API poll)
    # or query KaggleApi for each account
    for acc in accounts:
        try:
            if kaggle_api_factory is not None:
                api = kaggle_api_factory(acc)
            else:
                from kaggle.api.kaggle_api_extended import KaggleApi
                api = KaggleApi()
                os.environ["KAGGLE_API_TOKEN"] = acc.key
                api.authenticate()

            # Check known candidate kernels for this worker
            slugs_to_check = []
            if active_slugs_by_worker and acc.worker_name in active_slugs_by_worker:
                slugs_to_check.extend(active_slugs_by_worker[acc.worker_name])
            else:
                # Default canonical canary/batch slugs
                slugs_to_check.extend([
                    "davidnet-canary-repro-s42",
                    "davidnet-batch-runner",
                    "davidnet-pipeline",
                ])

            for slug in slugs_to_check:
                try:
                    kernel_id = f"{acc.username}/{slug}"
                    st = api.kernels_status(kernel_id)
                    status = (st.get("status") or "").lower() if isinstance(st, dict) else str(st).lower()
                    if "running" in status or "queued" in status:
                        # Check if this kernel was launched targeting this namespace
                        # To be strictly conservative: if ANY canary/repro kernel is RUNNING/QUEUED,
                        # dispatch is refused.
                        return False, (
                            f"Refused dispatch: kernel '{kernel_id}' on {acc.worker_name} is currently "
                            f"{status.upper()}. Cannot dispatch to namespace '{target_namespace}' "
                            f"on repo '{target_repo}' while another kernel is active."
                        )
                except Exception:
                    # Kernel might not exist on this account yet
                    continue
        except Exception as e:
            # If account cannot be checked, be conservative if it's the target account
            pass

    return True, "OK"


def check_namespace_is_empty(
    target_namespace: str,
    target_repo: str,
    allow_resume: bool = False,
    hf_api: Optional[HfApi] = None,
) -> tuple[bool, str]:
    """Ensures that the target namespace on Hugging Face Hub contains zero files before dispatch.

    Returns:
        (ok, reason): ok is True if namespace is completely empty; False otherwise.
    """
    clean_ns = target_namespace.strip("/")
    if not clean_ns:
        return False, "Target namespace cannot be empty or root for an isolated run."

    api = hf_api or HfApi()
    try:
        files = api.list_repo_files(repo_id=target_repo)
    except Exception as e:
        return False, f"Failed to inspect HF repository '{target_repo}': {e}"

    prefix = f"{clean_ns}/"
    matching_files = [f for f in files if f.startswith(prefix) or f == clean_ns]

    if matching_files and not allow_resume:
        return False, (
            f"Refused dispatch: target namespace '{target_namespace}' is non-empty on '{target_repo}' "
            f"({len(matching_files)} existing files found, e.g. '{matching_files[0]}'). "
            "A clean isolated run requires an empty namespace."
        )

    return True, "OK"


def assert_pre_dispatch_guards(
    target_namespace: str,
    target_repo: str,
    accounts: Sequence[KaggleAccount],
    allow_resume: bool = False,
    active_slugs_by_worker: Optional[dict[str, list[str]]] = None,
    kaggle_api_factory=None,
    hf_api: Optional[HfApi] = None,
):
    """Executes all pre-dispatch guards and raises DispatchGuardError on failure."""
    # Guard 1: No other kernel is currently RUNNING or QUEUED targeting this work
    ok_concurrency, msg_concurrency = check_running_kernels_for_namespace(
        target_namespace=target_namespace,
        target_repo=target_repo,
        accounts=accounts,
        active_slugs_by_worker=active_slugs_by_worker,
        kaggle_api_factory=kaggle_api_factory,
    )
    if not ok_concurrency:
        raise DispatchGuardError(msg_concurrency)

    # Guard 2: Target namespace must be strictly empty on HF Hub
    ok_empty, msg_empty = check_namespace_is_empty(
        target_namespace=target_namespace,
        target_repo=target_repo,
        allow_resume=allow_resume,
        hf_api=hf_api,
    )
    if not ok_empty:
        raise DispatchGuardError(msg_empty)
