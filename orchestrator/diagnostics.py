from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from orchestrator.context_packs import ContextPackStore
from orchestrator.run_history import RunHistoryStore
from orchestrator.tool_registry import tool_inventory
from orchestrator.workspace import list_workspace_files
from shared.config import settings


@dataclass(frozen=True)
class DiagnosticCheck:
    id: str
    status: str
    title: str
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "status": self.status,
            "title": self.title,
            "detail": self.detail,
        }


def build_diagnostics(
    run_history: RunHistoryStore | None = None,
    context_packs: ContextPackStore | None = None,
) -> dict[str, object]:
    run_history = run_history or RunHistoryStore.from_settings()
    context_packs = context_packs or ContextPackStore.from_settings()
    checks = [
        _settings_check(),
        _tool_inventory_check(),
        _workspace_check(),
        _store_check("run_history", "Run history database", run_history.db_path),
        _store_check("context_packs", "Context pack database", context_packs.db_path),
        _model_config_check(),
    ]
    status = _overall_status(checks)
    return {
        "status": status,
        "checks": [check.as_dict() for check in checks],
        "counts": {
            "ok": sum(1 for check in checks if check.status == "ok"),
            "warn": sum(1 for check in checks if check.status == "warn"),
            "error": sum(1 for check in checks if check.status == "error"),
        },
    }


def _settings_check() -> DiagnosticCheck:
    if settings.orchestrator_port <= 0:
        return DiagnosticCheck("settings", "error", "Settings", "ORCHESTRATOR_PORT must be positive.")
    if settings.context_bundle_max_chars <= 0:
        return DiagnosticCheck("settings", "error", "Settings", "CONTEXT_BUNDLE_MAX_CHARS must be positive.")
    return DiagnosticCheck("settings", "ok", "Settings", "Core settings are loaded.")


def _tool_inventory_check() -> DiagnosticCheck:
    inventory = tool_inventory()
    if not inventory:
        return DiagnosticCheck("tools", "error", "Tool inventory", "No tools are registered.")
    high_risk = sum(1 for tool in inventory if tool["risk"] == "high")
    return DiagnosticCheck(
        "tools",
        "ok",
        "Tool inventory",
        f"{len(inventory)} tools registered; {high_risk} high-risk tools.",
    )


def _workspace_check() -> DiagnosticCheck:
    try:
        files = list_workspace_files(limit=5)
    except Exception as exc:
        return DiagnosticCheck("workspace", "error", "Workspace", str(exc))
    if not files:
        return DiagnosticCheck("workspace", "warn", "Workspace", "No readable text files found.")
    return DiagnosticCheck("workspace", "ok", "Workspace", f"{len(files)} sample files readable.")


def _store_check(check_id: str, title: str, path: Path) -> DiagnosticCheck:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("ab"):
            pass
    except OSError as exc:
        return DiagnosticCheck(check_id, "error", title, str(exc))
    return DiagnosticCheck(check_id, "ok", title, f"Writable at {path}.")


def _model_config_check() -> DiagnosticCheck:
    if not settings.llm_model.strip():
        return DiagnosticCheck("model", "error", "Model config", "LLM_MODEL is empty.")
    if not settings.ollama_host.startswith(("http://", "https://")):
        return DiagnosticCheck("model", "warn", "Model config", "OLLAMA_HOST should be an HTTP URL.")
    return DiagnosticCheck(
        "model",
        "ok",
        "Model config",
        f"{settings.llm_model} via {settings.ollama_host}.",
    )


def _overall_status(checks: list[DiagnosticCheck]) -> str:
    if any(check.status == "error" for check in checks):
        return "error"
    if any(check.status == "warn" for check in checks):
        return "warn"
    return "ok"
