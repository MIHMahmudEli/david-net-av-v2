"""Scheduler unit tests: .env parsing, SQLite claiming, push-folder generation,
and the HF-token input-file fallback. No network anywhere.
"""
from __future__ import annotations

import json
import threading

import pytest

from src.scheduler.config import (
    ENV_PATH,
    WorkerCredential,
    load_workers,
    parse_env_credentials,
    worker_names,
)
from src.scheduler.db import Store
from src.scheduler.kernel import build_push_folder, build_secret_folder


# ------------------------------------------------------------------ .env parsing
def test_real_env_yields_sixteen_workers_without_leaking_tokens():
    workers = load_workers()
    assert len(workers) == 16
    assert len({w.name for w in workers}) == 16
    assert len({w.username for w in workers}) == 16
    for w in workers:
        assert "***" in repr(w)                       # redacted
        assert w.token and w.token not in repr(w)
        assert w.token not in str(w)


@pytest.mark.parametrize("key", [
    "Kaggel_Token", "Kaggel_Tokhon", "KAGGLE_TOKEN", "kaggle_token",
    "Kaggel_Token ", "kaggel tokhon",
])
def test_typo_tolerant_token_keys(tmp_path, key):
    env = tmp_path / ".env"
    env.write_text(f"Username = alice\n{key}=secret-token-value\n", encoding="utf-8")
    assert parse_env_credentials(env) == [("alice", "secret-token-value")]


def test_pairs_stay_paired_across_messy_file(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "HF_UserName = bob\nHF_Token = hf_secret\n\n"
        "FakeAVCeleb = /data/fakeav\n\n"
        "Username = alice\nKaggel_Token = aaa\n\n"
        "Username = carl\nKaggel_Tokhon = ccc\n",
        encoding="utf-8")
    assert parse_env_credentials(env) == [("alice", "aaa"), ("carl", "ccc")]


def test_worker_names_env_override(monkeypatch):
    monkeypatch.setenv("SCHED_WORKER_NAMES", "alpha,beta")
    assert worker_names(2) == ["alpha", "beta"]
    assert worker_names(4) == ["alpha", "beta", "kaggle-worker-3", "kaggle-worker-4"]


def test_real_env_file_exists():
    assert ENV_PATH.exists(), ".env must exist for the scheduler to be useful"


# ----------------------------------------------------------------------- claiming
def _store(path) -> Store:
    return Store(path)


def test_claim_is_atomic_across_connections(tmp_path):
    """Two schedulers (two DB connections) must never claim the same job."""
    db = tmp_path / "state.db"
    s1, s2 = _store(db), _store(db)
    ids = [s1.create_job(priority=100, reason=f"j{i}") for i in range(6)]
    claimed: list[str] = []
    lock = threading.Lock()

    def worker(store: Store, wid: int):
        while True:
            jid = store.claim_job(wid, lease_s=60)
            if jid is None:
                return
            with lock:
                claimed.append(jid)
            store.set_job(jid, status="COMPLETED", worker_id=wid)  # release the slot

    t1 = threading.Thread(target=worker, args=(s1, 1))
    t2 = threading.Thread(target=worker, args=(s2, 2))
    t1.start(); t2.start(); t1.join(); t2.join()
    assert sorted(claimed) == sorted(ids), "each job must be claimed exactly once"
    s1.close(); s2.close()


def test_worker_with_active_job_cannot_claim_another(tmp_path):
    store = _store(tmp_path / "s.db")
    store.create_job()
    store.create_job()
    assert store.claim_job(1, lease_s=60) == "job-001"
    assert store.claim_job(1, lease_s=60) is None, "one job per worker at a time"
    assert store.claim_job(2, lease_s=60) == "job-002"
    store.close()


def test_requeued_job_is_claimable_again(tmp_path):
    store = _store(tmp_path / "s.db")
    store.create_job()
    assert store.claim_job(1, lease_s=60) == "job-001"
    store.set_job("job-001", status="PENDING", worker_id=None, attempts=1)
    assert store.claim_job(2, lease_s=60) == "job-001"
    store.close()


def test_dependencies_gate_claiming(tmp_path):
    store = _store(tmp_path / "s.db")
    a = store.create_job()
    b = store.create_job(depends_on=a)
    assert store.claim_job(1, lease_s=60) == a
    store.set_job(a, status="COMPLETED")
    assert store.claim_job(1, lease_s=60) == b
    store.close()


def _exp(key, status):
    return {"key": key, "exp_id": "EXP_1", "name": "n", "seed": 42, "grp": "g",
            "status": status, "claimed_by": None, "test_clip_auc": None,
            "train_hours": None, "has_checkpoint": 0}


def test_sync_experiments_replace_drops_rows_from_other_modes(tmp_path):
    store = _store(tmp_path / "s.db")
    store.sync_experiments([_exp("full:a:s42", "completed")], replace=True)
    # a successful mode-filtered fetch must fully replace the mirror
    store.sync_experiments([_exp("recovery_test:b:s42", "pending")], replace=True)
    assert [r["key"] for r in store.experiments()] == ["recovery_test:b:s42"]
    # a failed sync keeps the previous snapshot (no replace call) and merges
    store.sync_experiments([_exp("recovery_test:c:s42", "pending")])
    assert sorted(r["key"] for r in store.experiments()) == ["recovery_test:b:s42",
                                                             "recovery_test:c:s42"]
    store.close()


# --------------------------------------------------------------- push folder
def _cred(name="kaggle-worker-7", username="someuser"):
    return WorkerCredential(name=name, username=username, token="KGAT_fake", env_index=0)


def test_push_folder_contains_notebook_and_metadata(tmp_path):
    from src.scheduler.config import Settings

    settings = Settings(work_dir=tmp_path / "work", db_path=tmp_path / "s.db")
    folder = build_push_folder(_cred(), settings=settings, out=tmp_path / "push")
    meta = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert meta["id"] == "someuser/davidnet-q1-pipeline"
    assert meta["enable_gpu"] == "true" and meta["enable_internet"] == "true"
    assert meta["machine_shape"] == "NvidiaTeslaT4"
    assert meta["title"].replace(" ", "-") == "davidnet-q1-pipeline"
    assert meta["dataset_sources"][:7] == list(settings.data_datasets)
    assert meta["dataset_sources"][-1] == "someuser/davidnet-hf-token"
    nb = json.loads((folder / meta["code_file"]).read_text(encoding="utf-8"))
    assert nb["cells"], "notebook must have cells"


def test_notebook_is_parameterized_per_worker(tmp_path):
    from src.scheduler.config import Settings

    settings = Settings(work_dir=tmp_path / "work", db_path=tmp_path / "s.db",
                        mode="smoke", code_revision="abc123")
    folder = build_push_folder(_cred(name="kaggle-worker-9"), settings=settings,
                               out=tmp_path / "push")
    meta = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    src = "".join("".join(c["source"]) for c in
                  json.loads((folder / meta["code_file"]).read_text(encoding="utf-8"))["cells"])
    assert '"kaggle-worker-9"' in src          # unique session.worker_name
    assert "MODE = \"smoke\"" in src
    assert "\"abc123\"" in src
    assert "MIHMahmudEli/davidnet-experiments" in src


def test_secret_folder_writes_token_and_metadata(tmp_path):
    from src.scheduler.config import Settings

    settings = Settings(work_dir=tmp_path / "work", db_path=tmp_path / "s.db")
    folder = build_secret_folder(_cred(), token="hf_abc123", settings=settings,
                                 out=tmp_path / "secret")
    assert (folder / "hf_token.txt").read_text(encoding="utf-8").strip() == "hf_abc123"
    meta = json.loads((folder / "dataset-metadata.json").read_text(encoding="utf-8"))
    assert meta["id"] == "someuser/davidnet-hf-token"
    assert 6 <= len(meta["title"]) <= 50       # Kaggle title rule


def test_build_notebook_defaults_match_legacy(tmp_path):
    """Default invocation must reproduce the committed notebook's cells."""
    from importlib.util import module_from_spec, spec_from_file_location
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    spec = spec_from_file_location("davidnet_build_notebook", repo / "kaggle" / "build_notebook.py")
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)

    out = mod.build_notebook(out=tmp_path / "nb.ipynb")
    new = json.loads(out.read_text(encoding="utf-8"))
    committed = json.loads((repo / "kaggle" / "davidnet_q1_pipeline.ipynb").read_text(encoding="utf-8"))
    assert len(new["cells"]) == len(committed["cells"])

    def sources(nb):
        return ["".join(c["source"]) if isinstance(c["source"], list) else c["source"]
                for c in nb["cells"]]

    # config cell is byte-identical to the committed notebook
    assert sources(new)[2] == sources(committed)[2]


# ----------------------------------------------------- HF token input-file fallback
def test_resolve_hf_token_reads_attached_input_file(tmp_path, monkeypatch):
    from src.pipeline import hub

    tok = tmp_path / "hf_token.txt"
    tok.write_text("hf_from_dataset_file\n", encoding="utf-8")
    monkeypatch.setenv("DAVIDNET_HF_TOKEN_FILE", str(tok))
    monkeypatch.setenv("HF_TOKEN", "")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "")
    assert hub.resolve_hf_token() == "hf_from_dataset_file"


def test_resolve_hf_token_prefers_env_over_file(tmp_path, monkeypatch):
    from src.pipeline import hub

    tok = tmp_path / "hf_token.txt"
    tok.write_text("hf_file\n", encoding="utf-8")
    monkeypatch.setenv("DAVIDNET_HF_TOKEN_FILE", str(tok))
    monkeypatch.setenv("HF_TOKEN", "hf_env")
    assert hub.resolve_hf_token() == "hf_env"
