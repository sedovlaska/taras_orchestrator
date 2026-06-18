from __future__ import annotations

from dataclasses import dataclass

from orchestrator.workspace import read_workspace_file, search_workspace
from shared.config import settings


@dataclass(frozen=True)
class ContextSection:
    kind: str
    title: str
    content: str
    truncated: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "title": self.title,
            "content": self.content,
            "truncated": self.truncated,
        }


def _bounded_limit(max_chars: int | None) -> int:
    if max_chars is None:
        return settings.context_bundle_max_chars
    return max(1, min(max_chars, settings.context_bundle_max_chars))


def _take_content(content: str, remaining: int) -> tuple[str, bool]:
    if len(content) <= remaining:
        return content, False
    return content[:remaining], True


def _render_sections(sections: list[ContextSection], errors: list[dict[str, str]]) -> str:
    lines = ["## Workspace context"]
    for section in sections:
        lines.append("")
        lines.append(f"### {section.title}")
        if section.kind == "file":
            suffix = "\n[section truncated]" if section.truncated else ""
            lines.append(f"```text\n{section.content}{suffix}\n```")
        else:
            lines.append(section.content)
    if errors:
        lines.append("")
        lines.append("### Context errors")
        for error in errors:
            lines.append(f"- {error['source']}: {error['message']}")
    return "\n".join(lines).strip()


def build_context_bundle(
    paths: list[str] | None = None,
    query: str | None = None,
    search_limit: int = 20,
    max_chars: int | None = None,
) -> dict[str, object]:
    paths = [path.strip() for path in (paths or []) if path.strip()]
    query = (query or "").strip()
    if not paths and not query:
        raise ValueError("at least one path or query is required")

    limit = _bounded_limit(max_chars)
    remaining = limit
    sections: list[ContextSection] = []
    errors: list[dict[str, str]] = []

    for path in dict.fromkeys(paths):
        if remaining <= 0:
            break
        try:
            file = read_workspace_file(path)
            content, truncated = _take_content(str(file["content"]), remaining)
            sections.append(
                ContextSection(
                    kind="file",
                    title=f"File: {file['path']}",
                    content=content,
                    truncated=bool(file["truncated"]) or truncated,
                )
            )
            remaining -= len(content)
        except (FileNotFoundError, PermissionError, ValueError, OSError) as exc:
            errors.append({"source": path, "message": str(exc)})

    if query and remaining > 0:
        try:
            results = search_workspace(query, search_limit)
            lines = [f"Search query: {query}"]
            for result in results:
                lines.append(f"- {result['path']}:{result['line']}: {result['preview']}")
            if len(lines) == 1:
                lines.append("- No matches found.")
            content, truncated = _take_content("\n".join(lines), remaining)
            sections.append(
                ContextSection(
                    kind="search",
                    title=f"Search: {query}",
                    content=content,
                    truncated=truncated,
                )
            )
            remaining -= len(content)
        except ValueError as exc:
            errors.append({"source": "search", "message": str(exc)})

    prompt_context = _render_sections(sections, errors)
    return {
        "max_chars": limit,
        "used_chars": limit - remaining,
        "truncated": remaining <= 0 or any(section.truncated for section in sections),
        "sections": [section.as_dict() for section in sections],
        "errors": errors,
        "prompt_context": prompt_context,
    }
