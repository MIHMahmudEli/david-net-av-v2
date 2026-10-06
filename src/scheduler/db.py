"""Persistent scheduler state (SQLite, WAL) with atomic job claiming.

Tables
------
workers        one row per Kaggle account (tokens are NOT stored here - only the
               .env index needed to resolve them at call time)
jobs           one row per kernel session ("job-001" ...). PENDING -> CLAIMED ->
               RUNNING -> COMPLETED/FAILED/TIMEOUT. Claiming is a single
               ``BEGIN IMMEDIATE`` transaction so two workers can never take
               the same job (spec section 8).
experiments    read-only mirror of the Hugging Face registry (the campaign's
               source of truth for which experiments exist / finished)
events         append-only audit trail
"""
from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS workers (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    name               TEXT NOT NULL UNIQUE,
    username           TEXT NOT NULL,
    env_index          INTEGER NOT NULL,
    status             TEXT NOT NULL DEFAULT 'OFFLINE',
    current_job_id     TEXT,
    last_heartbeat     REAL,
    quota_gpu_remaining REAL,
    quota_total        REAL,
    quota_refresh_at   TEXT,
    quota_checked_at   REAL,
    cooldown_until     REAL,
    total_completed    INTEGER NOT NULL DEFAULT 0,
    total_failed       INTEGER NOT NULL DEFAULT 0,
    last_error         TEXT,
    created_at         REAL NOT NULL,
    updated_at         REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    id             TEXT PRIMARY KEY,
    worker_id      INTEGER,
    status         TEXT NOT NULL DEFAULT 'PENDING',
    priority       INTEGER NOT NULL DEFAULT 100,
    reason         TEXT,
    depends_on     TEXT,
    attempts       INTEGER NOT NULL DEFAULT 0,
    kernel_ref     TEXT,
    lease_expires_at REAL,
    heartbeat_at   REAL,
    started_at     REAL,
    finished_at    REAL,
    kaggle_status  TEXT,
    experiments_done TEXT,
    last_error     TEXT,
    created_at     REAL NOT NULL,
    updated_at     REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS experiments (
    key            TEXT PRIMARY KEY,
    exp_id         TEXT,
    name           TEXT,
    seed           INTEGER,
    grp            TEXT,
    status         TEXT NOT NULL DEFAULT 'pending',
    claimed_by     TEXT,
    test_clip_auc  REAL,
    train_hours    REAL,
    has_checkpoint INTEGER NOT NULL DEFAULT 0,
    synced_at      REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    ts       REAL NOT NULL,
    kind     TEXT NOT NULL,
    subject  TEXT,
    detail   TEXT
);

CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status, priority, created_at);
CREATE INDEX IF NOT EXISTS events_ts ON events(ts);
"""

TERMINAL = ("COMPLETED", "FAILED", "TIMEOUT")


class Store:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), isolation_level=None, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=10000")
        self._conn.executescript(SCHEMA)

    # ------------------------------------------------------------------ tx
    @contextmanager
    def tx(self):
        """Serialized write transaction (BEGIN IMMEDIATE = first-writer-wins)."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    def close(self):
        self._conn.close()

    # ------------------------------------------------------------- workers
    def upsert_workers(self, workers: Iterable[Any]) -> None:
        now = time.time()
        with self.tx() as c:
            for w in workers:
                c.execute(
                    """INSERT INTO workers(name, username, env_index, created_at, updated_at)
                       VALUES (?,?,?,?,?)
                       ON CONFLICT(name) DO UPDATE SET username=excluded.username,
                         env_index=excluded.env_index, updated_at=excluded.updated_at""",
                    (w.name, w.username, w.env_index, now, now),
                )

    def set_worker(self, name: str, **fields: Any) -> None:
        if not fields:
            return
        fields["updated_at"] = time.time()
        cols = ", ".join(f"{k}=?" for k in fields)
        with self.tx() as c:
            c.execute(f"UPDATE workers SET {cols} WHERE name=?", (*fields.values(), name))

    def worker(self, name: str) -> Optional[sqlite3.Row]:
        return self._conn.execute("SELECT * FROM workers WHERE name=?", (name,)).fetchone()

    def workers(self) -> list[sqlite3.Row]:
        return self._conn.execute("SELECT * FROM workers ORDER BY id").fetchall()

    def idle_workers(self, now: float | None = None) -> list[sqlite3.Row]:
        now = now if now is not None else time.time()
        return self._conn.execute(
            """SELECT * FROM workers
               WHERE status = 'IDLE'
                 AND (cooldown_until IS NULL OR cooldown_until <= ?)
               ORDER BY total_completed ASC, id""",
            (now,),
        ).fetchall()

    # ---------------------------------------------------------------- jobs
    def _next_job_id(self, c: sqlite3.Connection) -> str:
        row = c.execute("SELECT COUNT(*) n FROM jobs").fetchone()
        return f"job-{int(row['n']) + 1:03d}"

    def create_job(self, priority: int = 100, reason: str = "", depends_on: str | None = None,
                   now: float | None = None) -> str:
        now = now if now is not None else time.time()
        with self.tx() as c:
            jid = self._next_job_id(c)
            c.execute(
                """INSERT INTO jobs(id, status, priority, reason, depends_on, created_at, updated_at)
                   VALUES (?, 'PENDING', ?, ?, ?, ?, ?)""",
                (jid, priority, reason, depends_on, now, now),
            )
            return jid

    def claim_job(self, worker_id: int, lease_s: float, now: float | None = None) -> Optional[str]:
        """Atomically claim the highest-priority PENDING job for ``worker_id``.

        One transaction, guarded again by ``status='PENDING'`` in the WHERE
        clause: a second claimer in a concurrent transaction finds 0 rows.
        Fresh jobs (attempts=0) are preferred over requeued retries.
        """
        now = now if now is not None else time.time()
        with self.tx() as c:
            busy = c.execute(
                "SELECT id FROM jobs WHERE worker_id=? AND status IN ('CLAIMED','STARTING','RUNNING')",
                (worker_id,),
            ).fetchone()
            if busy:
                return None
            dep_ok = "AND (depends_on IS NULL OR depends_on IN (SELECT id FROM jobs WHERE status='COMPLETED'))"
            row = c.execute(
                f"""UPDATE jobs
                       SET status='CLAIMED', worker_id=?, lease_expires_at=?,
                           heartbeat_at=?, updated_at=?, attempts=attempts+1
                     WHERE id = (SELECT id FROM jobs
                                  WHERE status='PENDING' {dep_ok}
                                  ORDER BY priority ASC, attempts ASC, created_at ASC LIMIT 1)
                       AND status='PENDING'
                 RETURNING id""",
                (worker_id, now + lease_s, now, now),
            ).fetchone()
            return row["id"] if row else None

    def job(self, job_id: str) -> Optional[sqlite3.Row]:
        return self._conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()

    def jobs(self, status: str | None = None) -> list[sqlite3.Row]:
        if status:
            return self._conn.execute(
                "SELECT * FROM jobs WHERE status=? ORDER BY created_at", (status,)).fetchall()
        return self._conn.execute("SELECT * FROM jobs ORDER BY created_at").fetchall()

    def active_job_for(self, worker_id: int) -> Optional[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM jobs WHERE worker_id=? AND status IN ('CLAIMED','STARTING','RUNNING')",
            (worker_id,),
        ).fetchone()

    def set_job(self, job_id: str, **fields: Any) -> None:
        if not fields:
            return
        fields["updated_at"] = time.time()
        cols = ", ".join(f"{k}=?" for k in fields)
        with self.tx() as c:
            c.execute(f"UPDATE jobs SET {cols} WHERE id=?", (*fields.values(), job_id))

    def heartbeat(self, job_id: str, now: float | None = None) -> None:
        now = now if now is not None else time.time()
        with self.tx() as c:
            c.execute("UPDATE jobs SET heartbeat_at=?, lease_expires_at=?, updated_at=? WHERE id=?",
                      (now, now + 3600, now, job_id))

    def expired_leases(self, now: float | None = None) -> list[sqlite3.Row]:
        now = now if now is not None else time.time()
        return self._conn.execute(
            """SELECT * FROM jobs
               WHERE status IN ('CLAIMED','STARTING','RUNNING')
                 AND lease_expires_at IS NOT NULL AND lease_expires_at < ?""",
            (now,),
        ).fetchall()

    def job_counts(self) -> dict[str, int]:
        rows = self._conn.execute("SELECT status, COUNT(*) n FROM jobs GROUP BY status").fetchall()
        return {r["status"]: r["n"] for r in rows}

    # -------------------------------------------------------- experiments
    def sync_experiments(self, rows: list[dict], now: float | None = None,
                         replace: bool = False) -> None:
        """Mirror registry rows. ``replace=True`` after a successful fetch: the
        registry is filtered by mode, so rows from another mode must disappear
        (a FAILED sync never replaces - the previous snapshot is kept)."""
        now = now if now is not None else time.time()
        with self.tx() as c:
            if replace:
                c.execute("DELETE FROM experiments")
            for r in rows:
                c.execute(
                    """INSERT INTO experiments(key, exp_id, name, seed, grp, status, claimed_by,
                                               test_clip_auc, train_hours, has_checkpoint, synced_at)
                       VALUES (:key,:exp_id,:name,:seed,:grp,:status,:claimed_by,
                               :test_clip_auc,:train_hours,:has_checkpoint,:synced_at)
                       ON CONFLICT(key) DO UPDATE SET
                         status=excluded.status, claimed_by=excluded.claimed_by,
                         test_clip_auc=excluded.test_clip_auc, train_hours=excluded.train_hours,
                         has_checkpoint=excluded.has_checkpoint, synced_at=excluded.synced_at""",
                    {**r, "synced_at": now},
                )

    def experiment_counts(self) -> dict[str, int]:
        rows = self._conn.execute("SELECT status, COUNT(*) n FROM experiments GROUP BY status").fetchall()
        return {r["status"]: r["n"] for r in rows}

    def experiments(self, status: str | None = None) -> list[sqlite3.Row]:
        if status:
            return self._conn.execute(
                "SELECT * FROM experiments WHERE status=? ORDER BY key", (status,)).fetchall()
        return self._conn.execute("SELECT * FROM experiments ORDER BY key").fetchall()

    # -------------------------------------------------------------- events
    def event(self, kind: str, subject: str | None = None, detail: str | None = None,
              now: float | None = None) -> None:
        now = now if now is not None else time.time()
        with self.tx() as c:
            c.execute("INSERT INTO events(ts, kind, subject, detail) VALUES (?,?,?,?)",
                      (now, kind, subject, detail))

    def events(self, limit: int = 20) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM events ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
