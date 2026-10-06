"""Kaggle account manager: extracts credentials from .env safely and verifies connectivity."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class KaggleAccount:
    worker_name: str
    username: str
    key: str
    is_valid: bool = False
    last_error: Optional[str] = None

    def env_dict(self) -> dict[str, str]:
        """Returns standard environment variables required by Kaggle CLI/API (v2.x)."""
        env = os.environ.copy()
        env["KAGGLE_API_TOKEN"] = self.key
        env.pop("KAGGLE_USERNAME", None)
        env.pop("KAGGLE_KEY", None)
        return env


def load_accounts_from_env(env_path: str | Path = "E:\\Thesis\\.env") -> list[KaggleAccount]:
    """Parse all worker accounts from the specified .env file."""
    env_file = Path(env_path)
    if not env_file.exists():
        raise FileNotFoundError(f".env file not found at {env_file}")

    lines = env_file.read_text(encoding="utf-8").splitlines()
    accounts: list[KaggleAccount] = []

    current_worker: Optional[str] = None
    current_username: Optional[str] = None
    current_key: Optional[str] = None

    def flush():
        nonlocal current_worker, current_username, current_key
        if current_worker and current_username and current_key:
            accounts.append(KaggleAccount(
                worker_name=current_worker,
                username=current_username.strip(),
                key=current_key.strip()
            ))
        current_worker = None
        current_username = None
        current_key = None

    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue

        worker_match = re.match(r"#\s*(worker\d+)\s*:", line_clean, re.IGNORECASE)
        if worker_match:
            flush()
            # Standardize naming: e.g. "kaggle-worker-1"
            w_num = re.search(r"\d+", worker_match.group(1)).group()
            current_worker = f"kaggle-worker-{w_num}"
            continue

        if current_worker:
            if re.match(r"^Username\s*=", line_clean, re.IGNORECASE):
                current_username = line_clean.split("=", 1)[1].strip()
            elif re.match(r"^Kaggel_Tok[a-z]*\s*=", line_clean, re.IGNORECASE):
                current_key = line_clean.split("=", 1)[1].strip()

    flush()
    return accounts


def verify_account(account: KaggleAccount) -> bool:
    """Verify credentials against Kaggle API via kaggle quota."""
    import subprocess
    try:
        res = subprocess.run(["kaggle", "quota"], env=account.env_dict(),
                             capture_output=True, text=True, timeout=30)
        if res.returncode == 0 and "GPU" in res.stdout:
            account.is_valid = True
            account.last_error = None
            return True
        else:
            account.is_valid = False
            account.last_error = res.stderr.strip() or res.stdout.strip()
            return False
    except Exception as e:
        account.is_valid = False
        account.last_error = str(e)
        return False


if __name__ == "__main__":
    accts = load_accounts_from_env()
    print(f"Loaded {len(accts)} worker accounts from .env:")
    for a in accts:
        print(f"  - {a.worker_name}: {a.username} (Key length: {len(a.key)})")
