from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
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
                route_json TEXT NOT NULL,
                conversation_id TEXT,
                model TEXT
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

            CREATE TABLE IF NOT EXISTS approvals (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                status TEXT NOT NULL,
                tool_id TEXT NOT NULL,
                agent TEXT NOT NULL,
                risk TEXT NOT NULL,
                reason TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_approvals_run_id ON approvals(run_id);
            CREATE INDEX IF NOT EXISTS idx_approvals_status ON approvals(status);
            """
        )
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(runs)")}
        if "conversation_id" not in columns:
            conn.execute("ALTER TABLE runs ADD COLUMN conversation_id TEXT")
        if "model" not in columns:
            conn.execute("ALTER TABLE runs ADD COLUMN model TEXT")

    def create_run(
        self,
        message: str,
        route: dict,
        conversation_id: str | None = None,
        model: str | None = None,
    ) -> dict:
        run_id = str(uuid4())
        now = utc_now()
        agents = route.get("agents", [])
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs
                    (id, created_at, updated_at, status, message, answer, intent, agents_json, route_json, conversation_id, model)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    conversation_id,
                    model,
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
            "conversation_id": conversation_id,
            "model": model,
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

    def create_approval(self, run_id: str, decision: dict, ttl_seconds: int) -> dict:
        approval_id = str(uuid4())
        now_dt = datetime.now(UTC)
        now = now_dt.isoformat().replace("+00:00", "Z")
        expires_at = (now_dt + timedelta(seconds=ttl_seconds)).isoformat().replace("+00:00", "Z")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO approvals
                    (id, run_id, created_at, updated_at, expires_at, status, tool_id, agent, risk, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    approval_id,
                    run_id,
                    now,
                    now,
                    expires_at,
                    "pending",
                    decision["tool_id"],
                    decision["agent"],
                    decision["risk"],
                    decision["reason"],
                ),
            )
            row = conn.execute(
                """
                SELECT id, run_id, created_at, updated_at, expires_at, status, tool_id, agent, risk, reason
                FROM approvals
                WHERE id = ?
                """,
                (approval_id,),
            ).fetchone()
        return self._approval_from_row(row)

    def get_approval(self, approval_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, run_id, created_at, updated_at, expires_at, status, tool_id, agent, risk, reason
                FROM approvals
                WHERE id = ?
                """,
                (approval_id,),
            ).fetchone()
        return self._approval_from_row(row) if row else None

    def list_approvals(self, run_id: str | None = None, status: str | None = None) -> list[dict]:
        clauses = []
        values = []
        if run_id is not None:
            clauses.append("run_id = ?")
            values.append(run_id)
        if status is not None:
            clauses.append("status = ?")
            values.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT id, run_id, created_at, updated_at, expires_at, status, tool_id, agent, risk, reason
                FROM approvals
                {where}
                ORDER BY created_at DESC
                """,
                values,
            ).fetchall()
        return [self._approval_from_row(row) for row in rows]

    def resolve_approval(self, approval_id: str, status: str) -> dict | None:
        now = utc_now()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, run_id, created_at, updated_at, expires_at, status, tool_id, agent, risk, reason
                FROM approvals
                WHERE id = ?
                """,
                (approval_id,),
            ).fetchone()
            if row is None:
                return None
            current = self._approval_from_row(row)
            if current["status"] != "pending":
                return current
            if datetime.fromisoformat(current["expires_at"].replace("Z", "+00:00")) < datetime.now(UTC):
                status = "expired"
            conn.execute(
                "UPDATE approvals SET updated_at = ?, status = ? WHERE id = ?",
                (now, status, approval_id),
            )
            updated = conn.execute(
                """
                SELECT id, run_id, created_at, updated_at, expires_at, status, tool_id, agent, risk, reason
                FROM approvals
                WHERE id = ?
                """,
                (approval_id,),
            ).fetchone()
        return self._approval_from_row(updated)

    @staticmethod
    def _is_expired(approval: dict, now: datetime | None = None) -> bool:
        now = now or datetime.now(UTC)
        return datetime.fromisoformat(approval["expires_at"].replace("Z", "+00:00")) < now

    def approvals_ready(self, run_id: str) -> bool:
        approvals = self.list_approvals(run_id=run_id)
        now = datetime.now(UTC)
        return bool(approvals) and all(
            approval["status"] == "approved" and not self._is_expired(approval, now)
            for approval in approvals
        )

    def granted_tool_ids(self, run_id: str) -> list[str]:
        """Tool ids whose approval is still both approved and non-expired.

        Re-validates ``expires_at`` at read time so a stale approved grant never
        outlives ``TOOL_APPROVAL_TTL_SECONDS`` and silently re-enables a tool.
        """
        now = datetime.now(UTC)
        return [
            approval["tool_id"]
            for approval in self.list_approvals(run_id=run_id, status="approved")
            if not self._is_expired(approval, now)
        ]

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

    def summary(self) -> dict:
        with self._connect() as conn:
            totals = conn.execute("SELECT COUNT(*) AS count FROM runs").fetchone()
            status_rows = conn.execute(
                """
                SELECT status, COUNT(*) AS count
                FROM runs
                GROUP BY status
                ORDER BY count DESC, status ASC
                """
            ).fetchall()
            intent_rows = conn.execute(
                """
                SELECT intent, COUNT(*) AS count
                FROM runs
                GROUP BY intent
                ORDER BY count DESC, intent ASC
                LIMIT 12
                """
            ).fetchall()
            event_rows = conn.execute(
                """
                SELECT event, COUNT(*) AS count
                FROM run_events
                GROUP BY event
                ORDER BY count DESC, event ASC
                LIMIT 12
                """
            ).fetchall()
            recent_errors = conn.execute(
                """
                SELECT run_id, created_at, event, data_json
                FROM run_events
                WHERE event IN ('runner_error', 'error')
                ORDER BY id DESC
                LIMIT 10
                """
            ).fetchall()
            approval_rows = conn.execute(
                """
                SELECT status, COUNT(*) AS count
                FROM approvals
                GROUP BY status
                ORDER BY count DESC, status ASC
                """
            ).fetchall()
            runs = conn.execute(
                """
                SELECT agents_json
                FROM runs
                ORDER BY created_at DESC
                LIMIT 500
                """
            ).fetchall()

        agent_counts: dict[str, int] = {}
        for row in runs:
            for agent in json.loads(row["agents_json"]):
                agent_counts[agent] = agent_counts.get(agent, 0) + 1

        return {
            "total_runs": totals["count"],
            "by_status": [{"status": row["status"], "count": row["count"]} for row in status_rows],
            "by_intent": [{"intent": row["intent"], "count": row["count"]} for row in intent_rows],
            "by_agent": [
                {"agent": agent, "count": count}
                for agent, count in sorted(agent_counts.items(), key=lambda item: (-item[1], item[0]))
            ],
            "by_event": [{"event": row["event"], "count": row["count"]} for row in event_rows],
            "approvals": [
                {"status": row["status"], "count": row["count"]} for row in approval_rows
            ],
            "recent_errors": [
                {
                    "run_id": row["run_id"],
                    "created_at": row["created_at"],
                    "event": row["event"],
                    "data": json.loads(row["data_json"]),
                }
                for row in recent_errors
            ],
        }

    def get_run(self, run_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, created_at, updated_at, status, message, answer, intent, agents_json, route_json, model
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
            data["model"] = row["model"]
        return data

    def _event_from_row(self, row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "run_id": row["run_id"],
            "created_at": row["created_at"],
            "event": row["event"],
            "data": json.loads(row["data_json"]),
        }

    def _approval_from_row(self, row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "run_id": row["run_id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "expires_at": row["expires_at"],
            "status": row["status"],
            "tool_id": row["tool_id"],
            "agent": row["agent"],
            "risk": row["risk"],
            "reason": row["reason"],
        }
