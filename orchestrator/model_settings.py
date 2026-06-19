from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from orchestrator.run_history import resolve_history_path, utc_now
from shared.config import settings


MODEL_SETTINGS_DB_PATH = ".data/model_settings.sqlite3"
PROVIDERS = {"ollama", "openai"}


@dataclass(frozen=True)
class EffectiveModelSettings:
    provider: str
    model: str
    ollama_host: str
    openai_base_url: str
    openai_api_key: str


def mask_secret(value: str) -> str | None:
    if not value:
        return None
    if len(value) <= 4:
        return "****"
    return f"****{value[-4:]}"


def default_model_settings() -> EffectiveModelSettings:
    return EffectiveModelSettings(
        provider=settings.llm_provider,
        model=settings.llm_model,
        ollama_host=settings.ollama_host,
        openai_base_url=settings.openai_base_url,
        openai_api_key=settings.openai_api_key,
    )


@dataclass(frozen=True)
class ModelSettingsStore:
    db_path: Path

    @classmethod
    def from_settings(cls) -> "ModelSettingsStore":
        return cls(resolve_history_path(MODEL_SETTINGS_DB_PATH))

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        self._init_schema(conn)
        return conn

    def _init_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS model_settings (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                updated_at TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                openai_base_url TEXT NOT NULL,
                openai_api_key TEXT NOT NULL
            );
            """
        )

    def get(self) -> EffectiveModelSettings:
        base = default_model_settings()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT provider, model, openai_base_url, openai_api_key
                FROM model_settings
                WHERE id = 1
                """
            ).fetchone()
        if row is None:
            return base
        return EffectiveModelSettings(
            provider=row["provider"],
            model=row["model"],
            ollama_host=base.ollama_host,
            openai_base_url=row["openai_base_url"],
            openai_api_key=row["openai_api_key"],
        )

    def public(self) -> dict:
        effective = self.get()
        return {
            "provider": effective.provider,
            "model": effective.model,
            "base_url": effective.openai_base_url,
            "api_key_set": bool(effective.openai_api_key),
            "api_key_masked": mask_secret(effective.openai_api_key),
        }

    def update(
        self,
        *,
        provider: str,
        model: str,
        base_url: str = "",
        api_key: str | None = None,
        keep_existing_api_key: bool = False,
    ) -> dict:
        current = self.get()
        provider = provider.strip().lower()
        model = model.strip()
        base_url = base_url.strip()
        if provider not in PROVIDERS:
            raise ValueError("provider must be one of: ollama, openai")
        if not model:
            raise ValueError("model must be a non-empty string")
        if api_key is None and keep_existing_api_key:
            resolved_api_key = current.openai_api_key
        else:
            resolved_api_key = (api_key or "").strip()
        if provider == "openai":
            if not base_url:
                raise ValueError("base_url is required when provider is openai")
            if not resolved_api_key:
                raise ValueError("api_key is required when provider is openai")
            if not base_url.startswith(("http://", "https://")):
                raise ValueError("base_url must start with http:// or https://")
        else:
            base_url = ""
            resolved_api_key = ""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO model_settings
                    (id, updated_at, provider, model, openai_base_url, openai_api_key)
                VALUES (1, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    provider = excluded.provider,
                    model = excluded.model,
                    openai_base_url = excluded.openai_base_url,
                    openai_api_key = excluded.openai_api_key
                """,
                (utc_now(), provider, model, base_url, resolved_api_key),
            )
        return self.public()


model_settings_store = ModelSettingsStore.from_settings()


def effective_model_settings() -> EffectiveModelSettings:
    return model_settings_store.get()
