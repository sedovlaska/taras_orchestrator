from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from orchestrator.policy import PROJECT_ROOT
from shared.config import settings

IGNORED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".data",
    "build",
    "dist",
    "node_modules",
    "static",
    "a2a_agents.egg-info",
}
IGNORED_SUFFIXES = {".pyc", ".pyo", ".sqlite3", ".db"}
BINARY_PROBE_BYTES = 4096


@dataclass(frozen=True)
class WorkspaceFile:
    path: str
    size_bytes: int
    modified_at: str

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "modified_at": self.modified_at,
        }


@dataclass(frozen=True)
class WorkspaceRead:
    path: str
    size_bytes: int
    truncated: bool
    content: str

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "truncated": self.truncated,
            "content": self.content,
        }


@dataclass(frozen=True)
class WorkspaceSearchResult:
    path: str
    line: int
    preview: str

    def as_dict(self) -> dict[str, object]:
        return {"path": self.path, "line": self.line, "preview": self.preview}


def _positive_limit(value: int | None, default: int) -> int:
    if value is None:
        return default
    return max(1, min(value, default))


def workspace_path(raw_path: str = ".") -> Path:
    normalized = raw_path.strip() or "."
    candidate = (PROJECT_ROOT / normalized).resolve()
    if candidate != PROJECT_ROOT and PROJECT_ROOT not in candidate.parents:
        raise PermissionError(f"path escapes project root: {raw_path}")
    return candidate


def relative_workspace_path(path: Path) -> str:
    if path == PROJECT_ROOT:
        return "."
    return path.relative_to(PROJECT_ROOT).as_posix()


def _is_ignored(path: Path) -> bool:
    relative = path.relative_to(PROJECT_ROOT)
    if any(part in IGNORED_DIRS for part in relative.parts):
        return True
    return path.suffix.lower() in IGNORED_SUFFIXES


def _is_probably_binary(path: Path) -> bool:
    with path.open("rb") as handle:
        sample = handle.read(BINARY_PROBE_BYTES)
    return b"\0" in sample


def _is_readable_text_file(path: Path) -> bool:
    try:
        return path.stat().st_size <= settings.workspace_max_file_bytes and not _is_probably_binary(path)
    except OSError:
        return False


def _file_meta(path: Path) -> WorkspaceFile:
    stat = path.stat()
    return WorkspaceFile(
        path=relative_workspace_path(path),
        size_bytes=stat.st_size,
        modified_at=datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
    )


def iter_workspace_files() -> Iterator[Path]:
    for root, dirs, filenames in os.walk(PROJECT_ROOT):
        root_path = Path(root)
        dirs[:] = sorted(
            dirname
            for dirname in dirs
            if dirname not in IGNORED_DIRS and not _is_ignored(root_path / dirname)
        )
        for filename in sorted(filenames):
            path = root_path / filename
            if _is_ignored(path) or not path.is_file():
                continue
            yield path


def list_workspace_files(limit: int | None = None) -> list[dict[str, object]]:
    max_files = _positive_limit(limit, settings.workspace_file_list_limit)
    files = []
    for path in iter_workspace_files():
        if not _is_readable_text_file(path):
            continue
        files.append(_file_meta(path).as_dict())
        if len(files) >= max_files:
            break
    return files


def read_workspace_file(raw_path: str) -> dict[str, object]:
    path = workspace_path(raw_path)
    if not path.is_file():
        raise FileNotFoundError(f"workspace file not found: {raw_path}")
    if _is_ignored(path):
        raise PermissionError(f"workspace file is ignored: {raw_path}")
    size_bytes = path.stat().st_size
    if size_bytes > settings.workspace_max_file_bytes:
        raise ValueError(f"workspace file exceeds {settings.workspace_max_file_bytes} bytes")
    if _is_probably_binary(path):
        raise ValueError("workspace file appears to be binary")

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        content = handle.read(settings.workspace_max_file_chars + 1)
    truncated = len(content) > settings.workspace_max_file_chars
    if truncated:
        content = content[: settings.workspace_max_file_chars]

    return WorkspaceRead(
        path=relative_workspace_path(path),
        size_bytes=size_bytes,
        truncated=truncated,
        content=content,
    ).as_dict()


def search_workspace(query: str, limit: int | None = None) -> list[dict[str, object]]:
    needle = query.strip()
    if not needle:
        raise ValueError("search query is required")

    max_results = _positive_limit(limit, settings.workspace_max_search_results)
    results: list[WorkspaceSearchResult] = []
    lowered = needle.lower()
    for path in iter_workspace_files():
        if path.stat().st_size > settings.workspace_max_file_bytes:
            continue
        try:
            if _is_probably_binary(path):
                continue
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if lowered not in line.lower():
                        continue
                    results.append(
                        WorkspaceSearchResult(
                            path=relative_workspace_path(path),
                            line=line_number,
                            preview=line.strip()[:240],
                        )
                    )
                    if len(results) >= max_results:
                        return [result.as_dict() for result in results]
        except OSError:
            continue
    return [result.as_dict() for result in results]
