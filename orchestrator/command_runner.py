from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from shared.config import settings


PROJECT_ROOT = Path(__file__).parent.parent.resolve()


def _csv(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    cwd: str
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    @property
    def output(self) -> str:
        return self.stdout or self.stderr

    def as_dict(self) -> dict:
        return {
            "command": self.command,
            "cwd": self.cwd,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "timed_out": self.timed_out,
        }


def require_project_cwd(raw_cwd: str | Path | None = None) -> Path:
    cwd = PROJECT_ROOT if raw_cwd is None else (PROJECT_ROOT / raw_cwd if not Path(raw_cwd).is_absolute() else Path(raw_cwd))
    resolved = cwd.resolve()
    if resolved != PROJECT_ROOT and PROJECT_ROOT not in resolved.parents:
        raise PermissionError(f"command cwd escapes project root: {raw_cwd}")
    return resolved


def _truncate(text: str, max_chars: int | None = None) -> str:
    limit = max_chars if max_chars is not None else settings.command_output_max_chars
    if len(text) <= limit:
        return text
    suffix = f"\n...[truncated {len(text) - limit} chars]"
    keep = max(0, limit - len(suffix))
    return text[:keep] + suffix


def _command_name(command: Sequence[str]) -> str:
    if not command:
        raise ValueError("command cannot be empty")
    return Path(command[0]).name.lower()


def _assert_allowed(command: Sequence[str]) -> None:
    allowed = {name.lower() for name in _csv(settings.command_allowed_executables)}
    name = _command_name(command)
    stem = Path(name).stem
    if name not in allowed and stem not in allowed:
        raise PermissionError(f"command executable is not allowed: {command[0]}")


def command_env(overrides: dict[str, str] | None = None) -> dict[str, str]:
    allowlist = _csv(settings.command_env_allowlist)
    env = {key: value for key, value in os.environ.items() if key in allowlist}
    for key, value in (overrides or {}).items():
        if key not in allowlist:
            raise PermissionError(f"environment variable is not allowed for commands: {key}")
        env[key] = value
    return env


def run_command(
    command: Sequence[str],
    *,
    cwd: str | Path | None = None,
    timeout_seconds: int,
    env_overrides: dict[str, str] | None = None,
    max_output_chars: int | None = None,
) -> CommandResult:
    _assert_allowed(command)
    safe_cwd = require_project_cwd(cwd)
    env = command_env(env_overrides)
    try:
        result = subprocess.run(
            list(command),
            cwd=safe_cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            env=env,
        )
        return CommandResult(
            command=list(command),
            cwd=str(safe_cwd),
            returncode=result.returncode,
            stdout=_truncate(result.stdout or "", max_output_chars),
            stderr=_truncate(result.stderr or "", max_output_chars),
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return CommandResult(
            command=list(command),
            cwd=str(safe_cwd),
            returncode=None,
            stdout=_truncate(stdout, max_output_chars),
            stderr=_truncate(stderr or f"Command timed out after {timeout_seconds}s", max_output_chars),
            timed_out=True,
        )
