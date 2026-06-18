from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from orchestrator.context_bundles import build_context_bundle
from orchestrator.run_history import PROJECT_ROOT, utc_now
from shared.config import settings


def resolve_context_pack_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


@dataclass(frozen=True)
class ContextPackStore:
    db_path: Path

    @classmethod
    def from_settings(cls) -> "ContextPackStore":
        return cls(resolve_context_pack_path(settings.context_pack_db_path))

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        self._init_schema(conn)
        return conn

    def _init_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS context_packs (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                paths_json TEXT NOT NULL,
                query TEXT,
                search_limit INTEGER NOT NULL,
                max_chars INTEGER
            );

            CREATE INDEX IF NOT EXISTS idx_context_packs_updated_at
                ON context_packs(updated_at DESC);
            """
        )

    def create_pack(
        self,
        name: str,
        description: str = "",
        paths: list[str] | None = None,
        query: str | None = None,
        search_limit: int = 20,
        max_chars: int | None = None,
    ) -> dict:
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("context pack name is required")
        pack_id = str(uuid4())
        now = utc_now()
        normalized_paths = _normalize_paths(paths)
        normalized_query = (query or "").strip() or None
        if not normalized_paths and not normalized_query:
            raise ValueError("at least one path or query is required")

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO context_packs
                    (id, created_at, updated_at, name, description, paths_json, query, search_limit, max_chars)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pack_id,
                    now,
                    now,
                    normalized_name,
                    description.strip(),
                    json.dumps(normalized_paths, ensure_ascii=False),
                    normalized_query,
                    _bounded_search_limit(search_limit),
                    max_chars,
                ),
            )
            row = self._get_row(conn, pack_id)
        return self._pack_from_row(row)

    def update_pack(
        self,
        pack_id: str,
        name: str,
        description: str = "",
        paths: list[str] | None = None,
        query: str | None = None,
        search_limit: int = 20,
        max_chars: int | None = None,
    ) -> dict | None:
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("context pack name is required")
        normalized_paths = _normalize_paths(paths)
        normalized_query = (query or "").strip() or None
        if not normalized_paths and not normalized_query:
            raise ValueError("at least one path or query is required")

        now = utc_now()
        with self._connect() as conn:
            if self._get_row(conn, pack_id) is None:
                return None
            conn.execute(
                """
                UPDATE context_packs
                SET updated_at = ?, name = ?, description = ?, paths_json = ?, query = ?,
                    search_limit = ?, max_chars = ?
                WHERE id = ?
                """,
                (
                    now,
                    normalized_name,
                    description.strip(),
                    json.dumps(normalized_paths, ensure_ascii=False),
                    normalized_query,
                    _bounded_search_limit(search_limit),
                    max_chars,
                    pack_id,
                ),
            )
            row = self._get_row(conn, pack_id)
        return self._pack_from_row(row)

    def list_packs(self, limit: int = 50) -> list[dict]:
        bounded_limit = min(max(limit, 1), 200)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, created_at, updated_at, name, description, paths_json, query,
                    search_limit, max_chars
                FROM context_packs
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (bounded_limit,),
            ).fetchall()
        return [self._pack_from_row(row) for row in rows]

    def get_pack(self, pack_id: str) -> dict | None:
        with self._connect() as conn:
            row = self._get_row(conn, pack_id)
        return self._pack_from_row(row) if row else None

    def delete_pack(self, pack_id: str) -> bool:
        with self._connect() as conn:
            result = conn.execute("DELETE FROM context_packs WHERE id = ?", (pack_id,))
        return result.rowcount > 0

    def build_bundle(self, pack_id: str) -> dict | None:
        pack = self.get_pack(pack_id)
        if pack is None:
            return None
        bundle = build_context_bundle(
            paths=pack["paths"],
            query=pack["query"],
            search_limit=pack["search_limit"],
            max_chars=pack["max_chars"],
        )
        return {"pack": pack, "bundle": bundle}

    def _get_row(self, conn: sqlite3.Connection, pack_id: str) -> sqlite3.Row | None:
        return conn.execute(
            """
            SELECT id, created_at, updated_at, name, description, paths_json, query,
                search_limit, max_chars
            FROM context_packs
            WHERE id = ?
            """,
            (pack_id,),
        ).fetchone()

    def _pack_from_row(self, row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "name": row["name"],
            "description": row["description"],
            "paths": json.loads(row["paths_json"]),
            "query": row["query"],
            "search_limit": row["search_limit"],
            "max_chars": row["max_chars"],
        }


def _normalize_paths(paths: list[str] | None) -> list[str]:
    return list(dict.fromkeys(path.strip() for path in (paths or []) if path.strip()))


def _bounded_search_limit(search_limit: int) -> int:
    return min(max(search_limit, 1), settings.workspace_max_search_results)
