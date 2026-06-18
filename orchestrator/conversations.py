from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from orchestrator.run_history import resolve_history_path, utc_now
from shared.config import settings


@dataclass(frozen=True)
class ConversationStore:
    db_path: Path

    @classmethod
    def from_settings(cls) -> "ConversationStore":
        return cls(resolve_history_path(settings.conversations_db_path))

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
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                title TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS conversation_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                created_at TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_conversations_updated_at ON conversations(updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_conversation_messages_conv_id_id
                ON conversation_messages(conversation_id, id);
            """
        )

    def create_conversation(self, title: str | None = None) -> dict:
        conversation_id = str(uuid4())
        now = utc_now()
        clean_title = (title or "").strip() or "New conversation"
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO conversations (id, created_at, updated_at, title) VALUES (?, ?, ?, ?)",
                (conversation_id, now, now, clean_title),
            )
        return {
            "id": conversation_id,
            "created_at": now,
            "updated_at": now,
            "title": clean_title,
        }

    def get_conversation(self, conversation_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, created_at, updated_at, title FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
        return self._conversation_from_row(row) if row else None

    def list_conversations(self, limit: int = 50) -> list[dict]:
        bounded_limit = min(max(limit, 1), 200)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, created_at, updated_at, title
                FROM conversations
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (bounded_limit,),
            ).fetchall()
        return [self._conversation_from_row(row) for row in rows]

    def append_message(self, conversation_id: str, role: str, content: str) -> dict:
        now = utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversation_messages (conversation_id, created_at, role, content)
                VALUES (?, ?, ?, ?)
                """,
                (conversation_id, now, role, content),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )
            row = conn.execute(
                """
                SELECT id, conversation_id, created_at, role, content
                FROM conversation_messages
                WHERE rowid = last_insert_rowid()
                """
            ).fetchone()
        return self._message_from_row(row)

    def list_messages(self, conversation_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, conversation_id, created_at, role, content
                FROM conversation_messages
                WHERE conversation_id = ?
                ORDER BY id ASC
                """,
                (conversation_id,),
            ).fetchall()
        return [self._message_from_row(row) for row in rows]

    def delete_conversation(self, conversation_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM conversations WHERE id = ?", (conversation_id,)
            )
        return cursor.rowcount > 0

    def _conversation_from_row(self, row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "title": row["title"],
        }

    def _message_from_row(self, row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "conversation_id": row["conversation_id"],
            "created_at": row["created_at"],
            "role": row["role"],
            "content": row["content"],
        }
