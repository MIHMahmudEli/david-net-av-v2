"""SQLite state store for distributed Kaggle workers and jobs.
Uses atomic transaction locks (BEGIN IMMEDIATE) to guarantee race-free claiming.
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

DB_PATH = Path("E:\\Thesis\\.kaggle_orchestrator\\orchestrator.db")


@dataclass
class JobRecord:
    job_id: str
    name: str
    seed: int
    stage: str
    group: str
    split: str
    init_from: str
    config_hash: str
    priority: int = 10  # higher = run sooner
    status: str = "PENDING"  # PENDING, CLAIMED, RUNNING, COMPLETED, FAILED, RETRYING, CANCELLED
    worker_id: Optional[str] = None
    worker_name: Optional[str] = None
    kaggle_kernel_slug: Optional[str] = None
    attempt_count: int = 0
    max_attempts: int = 3
    lease_expires_at: float = 0.0
    started_at: float = 0.0
    last_heartbeat: float = 0.0
    completed_at: float = 0.0
    error_message: Optional[str] = None
    checkpoint_url: Optional[str] = None
    metrics_summary: Optional[str] = None
    code_revision: str = "phase2-v1"


@dataclass
class WorkerRecord:
    worker_id: str
    worker_name: str
    kaggle_username: str
    status: str = "IDLE"  # IDLE, BUSY, COOLDOWN, OFFLINE, ERROR
    current_job_id: Optional[str] = None
    last_heartbeat: float = 0.0
    total_completed: int = 0
    total_failed: int = 0
    cooldown_until: float = 0.0
    last_error: Optional[str] = None


class OrchestratorDB:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=60.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self):
        with self.get_connection() as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                seed INTEGER NOT NULL,
                stage TEXT NOT NULL,
                group_name TEXT NOT NULL,
                split TEXT NOT NULL,
                init_from TEXT DEFAULT '',
                config_hash TEXT NOT NULL,
                priority INTEGER NOT NULL DEFAULT 10,
                status TEXT NOT NULL DEFAULT 'PENDING',
                worker_id TEXT,
                worker_name TEXT,
                kaggle_kernel_slug TEXT,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                max_attempts INTEGER NOT NULL DEFAULT 3,
                lease_expires_at REAL NOT NULL DEFAULT 0.0,
                started_at REAL NOT NULL DEFAULT 0.0,
                last_heartbeat REAL NOT NULL DEFAULT 0.0,
                completed_at REAL NOT NULL DEFAULT 0.0,
                error_message TEXT,
                checkpoint_url TEXT,
                metrics_summary TEXT,
                code_revision TEXT DEFAULT 'phase2-v1'
            );
            """)

            # Schema migration: ensure code_revision column exists
            cols = [r["name"] for r in conn.execute("PRAGMA table_info(jobs)").fetchall()]
            if "code_revision" not in cols:
                conn.execute("ALTER TABLE jobs ADD COLUMN code_revision TEXT DEFAULT 'phase2-v1';")

            conn.execute("""
            CREATE TABLE IF NOT EXISTS workers (
                worker_id TEXT PRIMARY KEY,
                worker_name TEXT UNIQUE NOT NULL,
                kaggle_username TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'IDLE',
                current_job_id TEXT,
                last_heartbeat REAL NOT NULL DEFAULT 0.0,
                total_completed INTEGER NOT NULL DEFAULT 0,
                total_failed INTEGER NOT NULL DEFAULT 0,
                cooldown_until REAL NOT NULL DEFAULT 0.0,
                last_error TEXT
            );
            """)

            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_priority ON jobs(priority DESC, job_id ASC);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_init_from ON jobs(init_from);")

    # ---------------------------------------------------------------- Worker ops
    def register_or_update_worker(self, worker_name: str, username: str) -> WorkerRecord:
        now = time.time()
        with self.get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            row = conn.execute("SELECT * FROM workers WHERE worker_name = ?", (worker_name,)).fetchone()
            if row is None:
                worker_id = f"w_{worker_name}"
                conn.execute("""
                INSERT INTO workers (worker_id, worker_name, kaggle_username, status, last_heartbeat)
                VALUES (?, ?, ?, 'IDLE', ?)
                """, (worker_id, worker_name, username, now))
            else:
                worker_id = row["worker_id"]
                conn.execute("""
                UPDATE workers SET kaggle_username = ?, last_heartbeat = ? WHERE worker_id = ?
                """, (username, now, worker_id))
            conn.execute("COMMIT;")
            return self.get_worker(worker_name)

    def get_worker(self, worker_name: str) -> Optional[WorkerRecord]:
        with self.get_connection() as conn:
            row = conn.execute("SELECT * FROM workers WHERE worker_name = ?", (worker_name,)).fetchone()
            if not row:
                return None
            return WorkerRecord(
                worker_id=row["worker_id"],
                worker_name=row["worker_name"],
                kaggle_username=row["kaggle_username"],
                status=row["status"],
                current_job_id=row["current_job_id"],
                last_heartbeat=row["last_heartbeat"],
                total_completed=row["total_completed"],
                total_failed=row["total_failed"],
                cooldown_until=row["cooldown_until"],
                last_error=row["last_error"],
            )

    def get_all_workers(self) -> list[WorkerRecord]:
        with self.get_connection() as conn:
            rows = conn.execute("SELECT * FROM workers ORDER BY worker_name ASC").fetchall()
            return [WorkerRecord(
                worker_id=r["worker_id"],
                worker_name=r["worker_name"],
                kaggle_username=r["kaggle_username"],
                status=r["status"],
                current_job_id=r["current_job_id"],
                last_heartbeat=r["last_heartbeat"],
                total_completed=r["total_completed"],
                total_failed=r["total_failed"],
                cooldown_until=r["cooldown_until"],
                last_error=r["last_error"],
            ) for r in rows]

    def update_worker_heartbeat(self, worker_name: str, status: Optional[str] = None):
        now = time.time()
        with self.get_connection() as conn:
            if status:
                conn.execute("UPDATE workers SET last_heartbeat = ?, status = ? WHERE worker_name = ?",
                             (now, status, worker_name))
            else:
                conn.execute("UPDATE workers SET last_heartbeat = ? WHERE worker_name = ?",
                             (now, worker_name))
            # Automatically extend lease for any active job claimed/running by this worker
            conn.execute("""
            UPDATE jobs
            SET last_heartbeat = ?,
                lease_expires_at = ? + 3600.0
            WHERE worker_name = ? AND (status = 'CLAIMED' OR status = 'RUNNING');
            """, (now, now, worker_name))

    # ---------------------------------------------------------------- Job ops
    def insert_jobs(self, jobs: list[JobRecord]):
        with self.get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            for j in jobs:
                conn.execute("""
                INSERT OR IGNORE INTO jobs (
                    job_id, name, seed, stage, group_name, split, init_from, config_hash,
                    priority, status, max_attempts, code_revision
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    j.job_id, j.name, j.seed, j.stage, j.group, j.split, j.init_from or "",
                    j.config_hash, j.priority, j.status, j.max_attempts, getattr(j, "code_revision", "phase2-v1")
                ))
            conn.execute("COMMIT;")

    def claim_next_job(self, worker_name: str, lease_duration_sec: float = 3600.0) -> Optional[JobRecord]:
        """Atomically claims the highest priority eligible pending job with met dependencies.
        Guarantees that no two workers can claim the same job simultaneously.
        """
        now = time.time()
        lease_expires = now + lease_duration_sec
        with self.get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            # A job is eligible if:
            # 1. status is 'PENDING' or ('RETRYING' and lease_expires_at < now)
            # 2. If it has init_from != '', the parent job for the same seed must be 'COMPLETED'
            query = """
            SELECT j.* FROM jobs j
            WHERE (j.status = 'PENDING' OR (j.status = 'RETRYING' AND j.lease_expires_at < ?))
              AND (
                  j.init_from = '' OR j.init_from IS NULL OR
                  EXISTS (
                      SELECT 1 FROM jobs parent
                      WHERE parent.name = j.init_from
                        AND parent.seed = j.seed
                        AND parent.status = 'COMPLETED'
                  )
              )
            ORDER BY j.priority DESC, j.attempt_count ASC, j.job_id ASC
            LIMIT 1;
            """
            row = conn.execute(query, (now,)).fetchone()
            if not row:
                conn.execute("COMMIT;")
                return None

            job_id = row["job_id"]
            attempt_count = row["attempt_count"] + 1

            # Perform atomic state transition
            conn.execute("""
            UPDATE jobs
            SET status = 'CLAIMED',
                worker_name = ?,
                attempt_count = ?,
                lease_expires_at = ?,
                started_at = ?,
                last_heartbeat = ?
            WHERE job_id = ? AND (status = 'PENDING' OR status = 'RETRYING');
            """, (worker_name, attempt_count, lease_expires, now, now, job_id))

            conn.execute("""
            UPDATE workers
            SET status = 'BUSY',
                current_job_id = ?,
                last_heartbeat = ?
            WHERE worker_name = ?;
            """, (job_id, now, worker_name))

            conn.execute("COMMIT;")
            return self.get_job(job_id)

    def mark_job_running(self, job_id: str, kernel_slug: str):
        now = time.time()
        with self.get_connection() as conn:
            conn.execute("""
            UPDATE jobs
            SET status = 'RUNNING',
                kaggle_kernel_slug = ?,
                lease_expires_at = ? + 7200.0,
                last_heartbeat = ?
            WHERE job_id = ?;
            """, (kernel_slug, now, now, job_id))

    def mark_job_completed(self, job_id: str, checkpoint_url: str = "", metrics_summary: str = ""):
        now = time.time()
        with self.get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            job = conn.execute("SELECT worker_name FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            worker_name = job["worker_name"] if job else None

            conn.execute("""
            UPDATE jobs
            SET status = 'COMPLETED',
                completed_at = ?,
                checkpoint_url = ?,
                metrics_summary = ?
            WHERE job_id = ?;
            """, (now, checkpoint_url, metrics_summary, job_id))

            if worker_name:
                conn.execute("""
                UPDATE workers
                SET status = 'IDLE',
                    current_job_id = NULL,
                    total_completed = total_completed + 1,
                    last_heartbeat = ?
                WHERE worker_name = ?;
                """, (now, worker_name))
            conn.execute("COMMIT;")

    def mark_job_failed(self, job_id: str, error_message: str):
        now = time.time()
        with self.get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            job = conn.execute("SELECT worker_name, attempt_count, max_attempts FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if not job:
                conn.execute("COMMIT;")
                return

            worker_name = job["worker_name"]
            attempts = job["attempt_count"]
            max_attempts = job["max_attempts"]

            new_status = "RETRYING" if attempts < max_attempts else "FAILED"

            conn.execute("""
            UPDATE jobs
            SET status = ?,
                error_message = ?,
                last_heartbeat = ?,
                lease_expires_at = 0.0
            WHERE job_id = ?;
            """, (new_status, error_message, now, job_id))

            if worker_name:
                conn.execute("""
                UPDATE workers
                SET status = 'COOLDOWN',
                    current_job_id = NULL,
                    total_failed = total_failed + 1,
                    cooldown_until = ?,
                    last_error = ?,
                    last_heartbeat = ?
                WHERE worker_name = ?;
                """, (now + 120.0, error_message[:200], now, worker_name))
            conn.execute("COMMIT;")

    def recover_stale_leases(self, timeout_sec: float = 3600.0) -> int:
        """Finds running/claimed jobs whose workers went silent past the lease timeout and resets them."""
        now = time.time()
        with self.get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            rows = conn.execute("""
            SELECT job_id, worker_name FROM jobs
            WHERE (status = 'CLAIMED' OR status = 'RUNNING') AND lease_expires_at < ?;
            """, (now,)).fetchall()

            for r in rows:
                conn.execute("""
                UPDATE jobs
                SET status = 'RETRYING',
                    worker_name = NULL,
                    lease_expires_at = 0.0,
                    error_message = 'Lease expired / worker timed out'
                WHERE job_id = ?;
                """, (r["job_id"],))

                if r["worker_name"]:
                    conn.execute("""
                    UPDATE workers
                    SET status = 'OFFLINE',
                        current_job_id = NULL,
                        last_error = 'Session lease timed out'
                    WHERE worker_name = ?;
                    """, (r["worker_name"],))

            conn.execute("COMMIT;")
            return len(rows)

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        with self.get_connection() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if not row:
                return None
            return JobRecord(
                job_id=row["job_id"],
                name=row["name"],
                seed=row["seed"],
                stage=row["stage"],
                group=row["group_name"],
                split=row["split"],
                init_from=row["init_from"],
                config_hash=row["config_hash"],
                priority=row["priority"],
                status=row["status"],
                worker_id=row["worker_id"],
                worker_name=row["worker_name"],
                kaggle_kernel_slug=row["kaggle_kernel_slug"],
                attempt_count=row["attempt_count"],
                max_attempts=row["max_attempts"],
                lease_expires_at=row["lease_expires_at"],
                started_at=row["started_at"],
                last_heartbeat=row["last_heartbeat"],
                completed_at=row["completed_at"],
                error_message=row["error_message"],
                checkpoint_url=row["checkpoint_url"],
                metrics_summary=row["metrics_summary"],
                code_revision=row["code_revision"] if "code_revision" in row.keys() else "phase2-v1",
            )

    def get_queue_summary(self) -> dict[str, Any]:
        with self.get_connection() as conn:
            rows = conn.execute("SELECT status, COUNT(*) as cnt FROM jobs GROUP BY status").fetchall()
            status_counts = {r["status"]: r["cnt"] for r in rows}
            total = sum(status_counts.values())
            completed = status_counts.get("COMPLETED", 0)
            running = status_counts.get("RUNNING", 0) + status_counts.get("CLAIMED", 0)
            pending = status_counts.get("PENDING", 0) + status_counts.get("RETRYING", 0)
            failed = status_counts.get("FAILED", 0)

            pct = (completed / total * 100.0) if total > 0 else 0.0

            workers = conn.execute("SELECT status, COUNT(*) as cnt FROM workers GROUP BY status").fetchall()
            worker_counts = {w["status"]: w["cnt"] for w in workers}

            return {
                "total_jobs": total,
                "completed": completed,
                "running": running,
                "pending": pending,
                "failed": failed,
                "progress_pct": round(pct, 1),
                "worker_summary": worker_counts,
                "job_status_breakdown": status_counts,
            }
