"""Sanitization Regression Test.

Enforces that no collaborator usernames or external project names appear in any
tracked file, and that the dataset slug owner appears in exactly ONE canonical
configuration file (src/pipeline/dataset_config.py).

To prevent trivially decodable strings or plain lists from living in public code,
forbidden terms are checked via salted SHA-256 hashes of lower-cased tokens.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

from src.pipeline.dataset_config import FAKEAVCELEB_DATASET_SLUG

REPO_ROOT = Path(__file__).resolve().parents[1]

# Salt used to compute one-way SHA-256 hashes of lower-cased target strings
_SALT = "davidnet_sanitization_salt_2026"

# Salted SHA-256 hashes of lower-cased collaborator usernames
FORBIDDEN_USER_HASHES: set[str] = {
    "9cc7df7cb8bc1fc95d533af9a1b46b91046c90390ef81be6f7edbfba20718a6e",
    "83b81843cd990610f87a777b6225861a7213699c11cf134eed032b941681a830",
    "4596011d62f262a082f4085d741575dbabc11d80e79c3768122da88d0cff2e0a",
    "79de61a2c1f3ad460a974058f886d36e3f4d357877e473f1de9ee96b726df410",
    "41f11eeda61adb6cead58a85519e607a5edc85d3c49f72cc0605301c59a0f2ee",
    "9caf8beb050cda6fbd086ee8797d8d95e1fb02c499a93115669bd2c148f543b7",
    "4717ec802988e3f8ffaa5808eef981e44f23911d33ae20aa7f7c8aafa5b33943",
    "feb7d66c179eafeaf5f5b6bf7e6bb666cfc5dd2c73f0425d0beee06b4e4cb76f",
    "b9dfb1ada45ab38b11ab9e0a7eb0e95180b71d29da6d0e8ac6e3fc142c32f660",
    "defa0ae0ab9fb3648cde583d2989d789cfd4e835621c763bfff5d06565d4b9d6",
    "20443b24deaa3ffa791f72e7b8dab50b5996e12cc8b62539c7759b9846c6e334",
    "1ef2bdfbc4821fa2ed18cc5f1387dd2b0edec59e9bdaf068521368fc568b34a5",
    "5d084260b03ea1618cf91aa839cb1d1f321e3f1efa3648af6ffc7aac2f4b4e18",
    "55f45a3195d7afa618e49e326547555d9021cfd94160bc985e50ed1343c16192",
    "4548806e1c7e7c6f8f3e2c9fe2fca1c75d32fc695787a9744971d9468cab9a4f",
}

# Salted SHA-256 hashes of lower-cased external project names
FORBIDDEN_PROJECT_HASHES: set[str] = {
    "57459a2d7500057a976f33d2e655603fb1974e75e8400e427b5b5e879d88c885",
    "81c02616c691a103fe741f583d484d86182ff7c58738c6dca1398f67972bde79",
    "cc1642cd9ee46b29346dcfd4bd36b80c63f78b9dbe69e3d38e16b5c870d98c39",
    "a6a4f5fa4f0a86b5525a63415dc07464358bd5535b5214d1245c7a6bf28da069",
}

# The ONE allowed config file for the dataset owner slug
ALLOWED_DATASET_CONFIG = Path("src/pipeline/dataset_config.py")
ALLOWED_DATASET_SLUG_OWNER = FAKEAVCELEB_DATASET_SLUG.split("/")[0]


def _hash_token(tok: str) -> str:
    return hashlib.sha256((_SALT + tok.lower()).encode("utf-8")).hexdigest()


def _get_tracked_files() -> list[Path]:
    """Get list of files tracked by git in the repository."""
    res = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [REPO_ROOT / p for p in res.stdout.splitlines() if p.strip()]


def _extract_candidate_tokens(line: str) -> set[str]:
    raw = set(re.findall(r"[A-Za-z0-9_\-]+", line))
    tokens = set()
    for t in raw:
        tokens.add(t)
        for part in re.split(r"[-_]", t):
            if part:
                tokens.add(part)
    return tokens


def test_no_collaborator_usernames_in_repo():
    """Verify that no collaborator usernames appear in any tracked repo file."""
    violations: list[str] = []
    for path in _get_tracked_files():
        if not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for idx, line in enumerate(content.splitlines(), start=1):
            tokens = _extract_candidate_tokens(line)
            for tok in tokens:
                if _hash_token(tok) in FORBIDDEN_USER_HASHES:
                    rel = path.relative_to(REPO_ROOT).as_posix()
                    violations.append(f"{rel}:{idx} (matched token hash {_hash_token(tok)[:8]}): {line.strip()}")

    assert not violations, (
        f"Found {len(violations)} forbidden collaborator username occurrences:\n"
        + "\n".join(violations)
    )


def test_no_external_project_names_in_repo():
    """Verify that no external project names appear in any tracked repo file."""
    violations: list[str] = []
    for path in _get_tracked_files():
        if not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for idx, line in enumerate(content.splitlines(), start=1):
            tokens = _extract_candidate_tokens(line)
            for tok in tokens:
                if _hash_token(tok) in FORBIDDEN_PROJECT_HASHES:
                    rel = path.relative_to(REPO_ROOT).as_posix()
                    violations.append(f"{rel}:{idx} (matched token hash {_hash_token(tok)[:8]}): {line.strip()}")

    assert not violations, (
        f"Found {len(violations)} forbidden external project name occurrences:\n"
        + "\n".join(violations)
    )


def test_dataset_owner_slug_only_in_allowed_config():
    """Verify that dataset owner slug appears ONLY in the allowed config file."""
    pattern = re.compile(re.escape(ALLOWED_DATASET_SLUG_OWNER), re.IGNORECASE)
    allowed_rel = ALLOWED_DATASET_CONFIG.as_posix()
    violations: list[str] = []
    found_in_allowed = False

    for path in _get_tracked_files():
        if not path.is_file():
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        for idx, line in enumerate(content.splitlines(), start=1):
            if pattern.search(line):
                if rel == allowed_rel:
                    found_in_allowed = True
                else:
                    violations.append(f"{rel}:{idx}: {line.strip()}")

    assert not violations, (
        f"Found {len(violations)} unauthorized '{ALLOWED_DATASET_SLUG_OWNER}' occurrences "
        f"outside {allowed_rel}:\n" + "\n".join(violations)
    )
    assert found_in_allowed, f"Expected '{ALLOWED_DATASET_SLUG_OWNER}' to be defined in {allowed_rel}"
