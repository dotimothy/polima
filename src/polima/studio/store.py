from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


class StudioStore:
    def __init__(self, path: Path, max_runs: int = 100) -> None:
        self.path = path
        self.max_runs = max_runs
        self._lock = threading.RLock()
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  started_at REAL NOT NULL,
                  ended_at REAL,
                  status TEXT NOT NULL,
                  bundle TEXT NOT NULL,
                  task TEXT NOT NULL,
                  config TEXT NOT NULL,
                  error TEXT,
                  log_path TEXT
                );
                CREATE TABLE IF NOT EXISTS settings (
                  key TEXT PRIMARY KEY,
                  value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS benchmarks (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  started_at REAL NOT NULL,
                  ended_at REAL,
                  status TEXT NOT NULL,
                  bundle TEXT NOT NULL,
                  iterations INTEGER NOT NULL,
                  warmup INTEGER NOT NULL,
                  result TEXT,
                  error TEXT,
                  log_path TEXT
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        return db

    def begin_run(self, config: dict[str, Any], log_path: str) -> int:
        with self._lock, self._connect() as db:
            cursor = db.execute(
                "INSERT INTO runs(started_at,status,bundle,task,config,log_path) VALUES(?,?,?,?,?,?)",
                (time.time(), "running", config["bundle"], config["task"], json.dumps(config), log_path),
            )
            run_id = int(cursor.lastrowid)
            self._prune(db)
            return run_id

    def finish_run(self, run_id: int, status: str, error: str | None = None) -> None:
        with self._lock, self._connect() as db:
            db.execute(
                "UPDATE runs SET ended_at=?,status=?,error=? WHERE id=?",
                (time.time(), status, error, run_id),
            )

    def list_runs(self, limit: int = 100) -> list[dict[str, Any]]:
        limit = max(1, min(limit, self.max_runs))
        with self._lock, self._connect() as db:
            rows = db.execute("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["config"] = json.loads(item["config"])
            result.append(item)
        return result

    def set(self, key: str, value: Any) -> None:
        with self._lock, self._connect() as db:
            db.execute(
                "INSERT INTO settings(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, json.dumps(value)),
            )

    def begin_benchmark(
        self, bundle: str, iterations: int, warmup: int, log_path: str
    ) -> int:
        with self._lock, self._connect() as db:
            cursor = db.execute(
                "INSERT INTO benchmarks(started_at,status,bundle,iterations,warmup,log_path) "
                "VALUES(?,?,?,?,?,?)",
                (time.time(), "running", bundle, iterations, warmup, log_path),
            )
            benchmark_id = int(cursor.lastrowid)
            db.execute(
                "DELETE FROM benchmarks WHERE id NOT IN "
                "(SELECT id FROM benchmarks ORDER BY id DESC LIMIT ?)",
                (self.max_runs,),
            )
            return benchmark_id

    def finish_benchmark(
        self, benchmark_id: int, status: str, result: dict[str, Any] | None, error: str | None
    ) -> None:
        with self._lock, self._connect() as db:
            db.execute(
                "UPDATE benchmarks SET ended_at=?,status=?,result=?,error=? WHERE id=?",
                (time.time(), status, json.dumps(result) if result else None, error, benchmark_id),
            )

    def list_benchmarks(self, limit: int = 100) -> list[dict[str, Any]]:
        limit = max(1, min(limit, self.max_runs))
        with self._lock, self._connect() as db:
            rows = db.execute(
                "SELECT * FROM benchmarks ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["result"] = json.loads(item["result"]) if item["result"] else None
            result.append(item)
        return result

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock, self._connect() as db:
            row = db.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return default if row is None else json.loads(row["value"])

    def _prune(self, db: sqlite3.Connection) -> None:
        db.execute(
            "DELETE FROM runs WHERE id NOT IN (SELECT id FROM runs ORDER BY id DESC LIMIT ?)",
            (self.max_runs,),
        )
