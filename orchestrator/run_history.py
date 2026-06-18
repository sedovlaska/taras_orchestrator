from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from shared.config import settings


PROJECT_ROOT = Path(__file__).parent.parent.resolve()


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def resolve_history_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


@dataclass(frozen=True)
class RunHistoryStore:
    db_path: Path

    @classmethod
    def from_settings(cls) -> "RunHistoryStore":
        return cls(resolve_history_path(settings.run_history_db_path))

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema(conn)
        return conn

    def _init_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                status TEXT NOT NULL,
                message TEXT NOT NULL,
                answer TEXT,
                intent TEXT NOT NULL,
                agents_json TEXT NOT NULL,
                route_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS run_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                created_at TEXT NOT NULL,
                event TEXT NOT NULL,
                data_json TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_run_events_run_id_id ON run_events(run_id, id);
            """
        )

    def create_run(self, message: str, route: dict) -> dict:
        run_id = str(uuid4())
        now = utc_now()
        agents = route.get("agents", [])
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs
                    (id, created_at, updated_at, status, message, answer, intent, agents_json, route_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    now,
                    now,
                    "running",
                    message,
                    None,
                    route.get("intent", "orchestrator"),
                    json.dumps(agents, ensure_ascii=False),
                    json.dumps(route, ensure_ascii=False),
                ),
            )
        return {
            "id": run_id,
            "created_at": now,
            "updated_at": now,
            "status": "running",
            "message": message,
            "answer": None,
            "intent": route.get("intent", "orchestrator"),
            "agents": agents,
            "route": route,
        }

    def append_event(self, run_id: str, event: str, data: dict) -> dict:
        now = utc_now()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO run_events (run_id, created_at, event, data_json) VALUES (?, ?, ?, ?)",
                (run_id, now, event, json.dumps(data, ensure_ascii=False)),
            )
            conn.execute("UPDATE runs SET updated_at = ? WHERE id = ?", (now, run_id))
            row = conn.execute(
                "SELECT id, run_id, created_at, event, data_json FROM run_events WHERE rowid = last_insert_rowid()"
            ).fetchone()
        return self._event_from_row(row)

    def complete_run(self, run_id: str, answer: str, status: str = "completed") -> None:
        now = utc_now()
        with self._connect() as conn:
            conn.execute(
                "UPDATE runs SET updated_at = ?, status = ?, answer = ? WHERE id = ?",
                (now, status, answer, run_id),
            )

    def list_runs(self, limit: int = 50) -> list[dict]:
        bounded_limit = min(max(limit, 1), 200)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, created_at, updated_at, status, message, answer, intent, agents_json, route_json
                FROM runs
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (bounded_limit,),
            ).fetchall()
        return [self._run_from_row(row, include_route=False) for row in rows]

    def get_run(self, run_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, created_at, updated_at, status, message, answer, intent, agents_json, route_json
                FROM runs
                WHERE id = ?
                """,
                (run_id,),
            ).fetchone()
        return self._run_from_row(row, include_route=True) if row else None

    def list_events(self, run_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, run_id, created_at, event, data_json
                FROM run_events
                WHERE run_id = ?
                ORDER BY id ASC
                """,
                (run_id,),
            ).fetchall()
        return [self._event_from_row(row) for row in rows]

    def _run_from_row(self, row: sqlite3.Row, include_route: bool) -> dict:
        data = {
            "id": row["id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "status": row["status"],
            "message": row["message"],
            "answer": row["answer"],
            "intent": row["intent"],
            "agents": json.loads(row["agents_json"]),
        }
        if include_route:
            data["route"] = json.loads(row["route_json"])
        return data

    def _event_from_row(self, row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "run_id": row["run_id"],
            "created_at": row["created_at"],
            "event": row["event"],
            "data": json.loads(row["data_json"]),
        }
