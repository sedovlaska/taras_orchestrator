from __future__ import annotations

import asyncio
import json
import re
import sys
import urllib.request
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import psutil
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from orchestrator import ai_sdk_stream
from orchestrator.command_runner import command_env, require_project_cwd, run_command
from orchestrator.context_bundles import build_context_bundle
from orchestrator.context_packs import ContextPackStore
from orchestrator.conversations import ConversationStore
from orchestrator.diagnostics import build_diagnostics
from orchestrator.evals import diff_against_baseline, list_eval_cases, run_eval_suite
from orchestrator.agno_agents import TOOL_GATED_MARKER
from orchestrator.model_settings import effective_model_settings, model_settings_store
from orchestrator.ollama import list_ollama_models
from orchestrator.policy import APPROVAL_GRANTS_ENV, current_policy, warn_if_gate_disabled
from orchestrator.routing import RoutingResult, route_request, should_use_local_system_status
from orchestrator.runbooks import get_runbook, list_runbooks, render_runbook
from orchestrator.run_history import RunHistoryStore
from orchestrator.tool_registry import AGNO_MEMBER_NAMES, get_tool, tool_inventory
from orchestrator.trace_export import build_run_trace
from orchestrator.workspace import list_workspace_files, read_workspace_file, search_workspace
from shared.config import settings

app = FastAPI(title="AGNO Team Orchestrator")

STATIC_DIR = Path(__file__).parent.parent / "static"
PROJECT_ROOT = Path(__file__).parent.parent

RunEvent = dict[str, object]
run_history = RunHistoryStore.from_settings()
context_pack_store = ContextPackStore.from_settings()
conversation_store = ConversationStore.from_settings()


@app.on_event("startup")
async def _warn_on_disabled_gate() -> None:
    # Surface a misconfigured (effectively off) approval gate loudly, so it is
    # not mistaken for an active gate.
    warn_if_gate_disabled()

class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None
    model: str | None = None


class ConversationCreateRequest(BaseModel):
    title: str | None = None


class ModelSettingsRequest(BaseModel):
    provider: str
    base_url: str = ""
    api_key: str | None = None
    keep_existing_api_key: bool = False
    model: str


class ChatResponse(BaseModel):
    answer: str
    agents_used: list[str]
    route: dict
    events: list[dict]
    run_id: str


class WorkspaceSearchRequest(BaseModel):
    query: str
    limit: int | None = None


class RunbookRenderRequest(BaseModel):
    values: dict[str, str] = Field(default_factory=dict)


class ContextBundleRequest(BaseModel):
    paths: list[str] = Field(default_factory=list)
    query: str | None = None
    search_limit: int = 20
    max_chars: int | None = None


class ContextPackRequest(BaseModel):
    name: str
    description: str = ""
    paths: list[str] = Field(default_factory=list)
    query: str | None = None
    search_limit: int = 20
    max_chars: int | None = None


def route_agents(message: str) -> list[str]:
    return route_request(message).agents


_MODEL_TAG_RE = re.compile(r"[A-Za-z0-9._:/-]+")


def resolve_model(model: str | None) -> str | None:
    """Validate an optional per-request model override.

    Returns the cleaned model string, or None to use the configured default.
    Raises ValueError when a model is supplied but is not a non-empty string,
    starts with a dash (which ``ollama run`` would parse as a CLI flag), or
    contains characters outside the ollama tag charset.
    """
    if model is None:
        return None
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty string")
    cleaned = model.strip()
    if cleaned.startswith("-") or not _MODEL_TAG_RE.fullmatch(cleaned):
        raise ValueError("model contains invalid characters")
    return cleaned


def conversation_context(conversation_id: str | None, message: str) -> str:
    """Prepend prior conversation turns as context to the current message."""
    if not conversation_id:
        return message
    history = conversation_store.list_messages(conversation_id)
    if not history:
        return message
    lines = [f"{turn['role']}: {turn['content']}" for turn in history]
    # Bound the prior-turns transcript, dropping oldest turns first so the most
    # recent context survives. The current user message is always included in full.
    limit = settings.conversation_context_max_chars
    while lines and len("\n".join(lines)) > limit:
        lines.pop(0)
    transcript = "\n".join(lines)
    if not transcript:
        return message
    return (
        "Prior conversation:\n"
        f"{transcript}\n\n"
        f"Current user message: {message}"
    )


def initial_run_events(route: RoutingResult) -> list[RunEvent]:
    events: list[RunEvent] = [
        {"event": "route", "data": route.as_dict()},
        {
            "event": "classify",
            "data": {
                "intent": route.intent,
                "agents": route.agents,
                "confidence": route.confidence,
                "reason": route.reason,
            },
        },
    ]
    events.extend({"event": "policy_decision", "data": decision} for decision in route.policy_decisions)
    events.extend({"event": "agent_start", "data": {"agent": agent}} for agent in route.agents)
    return events


def record_events(run_id: str, events: list[RunEvent]) -> None:
    for event in events:
        run_history.append_event(run_id, str(event["event"]), dict(event["data"]))


def approval_required_risks() -> set[str]:
    return {
        risk.strip().lower()
        for risk in settings.tool_approval_required_risks.split(",")
        if risk.strip()
    }


def create_required_approvals(run_id: str, route: RoutingResult) -> list[dict]:
    approvals = []
    for decision in route.policy_decisions:
        if decision["allowed"] and decision["risk"] in approval_required_risks():
            approval = run_history.create_approval(
                run_id,
                decision,
                ttl_seconds=settings.tool_approval_ttl_seconds,
            )
            approvals.append(approval)
            run_history.append_event(
                run_id,
                "approval_required",
                {
                    "approval_id": approval["id"],
                    "run_id": run_id,
                    "tool_id": approval["tool_id"],
                    "agent": approval["agent"],
                    "risk": approval["risk"],
                    "expires_at": approval["expires_at"],
                },
            )
    return approvals


def approval_required_event(approval: dict) -> RunEvent:
    return {
        "event": "approval_required",
        "data": {
            "approval_id": approval["id"],
            "run_id": approval["run_id"],
            "tool_id": approval["tool_id"],
            "agent": approval["agent"],
            "risk": approval["risk"],
            "expires_at": approval["expires_at"],
        },
    }


def approval_waiting_answer(approvals: list[dict]) -> str:
    tools = ", ".join(approval["tool_id"] for approval in approvals)
    return f"Tool approval required before execution: {tools}"


def system_status(events: list[RunEvent] | None = None) -> str:
    policy = current_policy()
    for tool_id in ("system.get_system_info", "system.get_disk_usage"):
        policy.require(tool_id)
        if events is not None:
            events.append({"event": "tool_start", "data": {"tool_id": tool_id, "agent": "system"}})

    cpu = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disks = []

    for part in psutil.disk_partitions():
        try:
            usage = psutil.disk_usage(part.mountpoint)
        except Exception:
            continue
        used_gb = usage.used // (1024**3)
        total_gb = usage.total // (1024**3)
        disks.append(f"{part.mountpoint}: {usage.percent}% ({used_gb}GB / {total_gb}GB)")

    disk_block = "\n".join(disks[:8]) or "Disk information is unavailable."
    used_mb = memory.used // (1024**2)
    total_mb = memory.total // (1024**2)

    if events is not None:
        for tool_id in ("system.get_system_info", "system.get_disk_usage"):
            events.append(
                {
                    "event": "tool_result",
                    "data": {"tool_id": tool_id, "agent": "system", "status": "ok"},
                }
            )

    return (
        "\u0421\u043e\u0441\u0442\u043e\u044f\u043d\u0438\u0435 \u0441\u0438\u0441\u0442\u0435\u043c\u044b:\n"
        f"CPU: {cpu}%\n"
        f"Memory: {memory.percent}% ({used_mb}MB / {total_mb}MB)\n"
        f"Disk:\n{disk_block}"
    )


def _ollama_env_overrides() -> dict[str, str]:
    model_settings = effective_model_settings()
    allowlist = {
        key.strip()
        for key in settings.command_env_allowlist.split(",")
        if key.strip()
    }
    env = {
        "NO_PROXY": "localhost,127.0.0.1,::1",
        "no_proxy": "localhost,127.0.0.1,::1",
        "OLLAMA_HOST": model_settings.ollama_host,
        "PYTHONIOENCODING": "utf-8",
    }
    optional_env = {
        # The AGNO Team runs in a subprocess; the child re-loads settings from env,
        # so provider selection is passed through when the command env policy permits it.
        "LLM_PROVIDER": model_settings.provider,
        "LLM_MODEL": model_settings.model,
    }
    if model_settings.provider == "openai":
        # Thread the OpenAI-compatible credentials to the child. The api key is
        # never logged (run_command does not echo env values).
        optional_env["OPENAI_BASE_URL"] = model_settings.openai_base_url
        optional_env["OPENAI_API_KEY"] = model_settings.openai_api_key
    env.update({key: value for key, value in optional_env.items() if key in allowlist})
    return env


def _record_gated_tools(stdout: str, events: list[RunEvent] | None) -> None:
    """Turn TOOL_GATED markers printed by the subprocess into trace events.

    The fail-closed gate runs inside the AGNO subprocess, which cannot reach the
    run-history store. It prints one marker line per blocked tool; here we parse
    them so each gated tool is recorded as a `tool_gated` trace event.
    """
    if events is None or TOOL_GATED_MARKER not in stdout:
        return
    for line in stdout.splitlines():
        # Honour the marker only at start-of-line. The marker is high-entropy
        # and gate_tool always emits it on its own line, so tool/model output
        # that echoes the literal mid-line cannot forge a tool_gated event.
        if not line.startswith(TOOL_GATED_MARKER):
            continue
        try:
            payload = json.loads(line[len(TOOL_GATED_MARKER):])
        except json.JSONDecodeError:
            continue
        events.append({"event": "tool_gated", "data": payload})


def ask_agno_team(
    message: str,
    agents: list[str],
    model: str | None = None,
    granted_tools: list[str] | None = None,
    events: list[RunEvent] | None = None,
) -> str:
    print(f"AGNO_TEAM_START agents={agents!r} model={model!r} message={message!r}", flush=True)
    code = r'''
import asyncio
import json
import sys
from orchestrator.agno_agents import create_orchestrator

message = sys.argv[1]
target_agents = json.loads(sys.argv[2])
model = sys.argv[3] or None
if target_agents and target_agents != ["orchestrator"]:
    message = f"Route this request to these AGNO team members: {target_agents}. User request: {message}"
resp = create_orchestrator(model).run(message)
print("__AGNO_JSON__" + json.dumps({"content": str(resp.content)}, ensure_ascii=False), flush=True)
'''
    env_overrides = _ollama_env_overrides()
    if granted_tools:
        env_overrides[APPROVAL_GRANTS_ENV] = ",".join(granted_tools)
    result = run_command(
        [sys.executable, "-c", code, message, json.dumps(agents, ensure_ascii=False), model or ""],
        cwd=PROJECT_ROOT,
        timeout_seconds=180,
        env_overrides=env_overrides,
    )
    _record_gated_tools(result.stdout, events)
    marker = "__AGNO_JSON__"
    marker_pos = result.stdout.rfind(marker)
    if result.ok and marker_pos != -1:
        payload = json.loads(result.stdout[marker_pos + len(marker):].strip())
        content = payload.get("content", "")
        if content and not _looks_like_model_error(content):
            print(f"AGNO_TEAM_SUCCESS agents={agents!r}", flush=True)
            return content

    details = (result.stderr or result.stdout or f"exit code {result.returncode}").strip()
    print(f"AGNO_TEAM_FAILED agents={agents!r} error={details!r}", flush=True)
    raise RuntimeError(details or "AGNO Team returned an empty response")


def ask_ollama_direct(message: str, model: str | None = None) -> str:
    model_settings = effective_model_settings()
    model = model or model_settings.model
    print(f"OLLAMA_DIRECT_START model={model!r} message={message!r}", flush=True)
    url = f"{model_settings.ollama_host.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a concise helpful assistant. Reply in the user's language.",
            },
            {"role": "user", "content": message},
        ],
        "stream": False,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=120) as response:
        body = json.loads(response.read().decode("utf-8"))
    print("OLLAMA_DIRECT_SUCCESS", flush=True)
    return body.get("message", {}).get("content") or body.get("response") or "Ollama returned an empty response."


def _iter_ollama_direct(message: str, model: str | None = None):
    """Stream Ollama /api/chat with ``stream:true``, yielding content deltas.

    This is the synchronous NDJSON reader; ``stream_orchestrator`` drives it in a
    worker thread so the event loop is never blocked. Each yielded value is a
    non-empty content fragment as the model produces it.
    """
    model_settings = effective_model_settings()
    model = model or model_settings.model
    print(f"OLLAMA_DIRECT_STREAM_START model={model!r} message={message!r}", flush=True)
    url = f"{model_settings.ollama_host.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a concise helpful assistant. Reply in the user's language.",
            },
            {"role": "user", "content": message},
        ],
        "stream": True,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    token_usage: dict[str, int] = {}
    with opener.open(request, timeout=120) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line:
                continue
            chunk = json.loads(line)
            delta = chunk.get("message", {}).get("content") or chunk.get("response") or ""
            if delta:
                yield delta
            if chunk.get("done"):
                prompt_tokens = chunk.get("prompt_eval_count")
                completion_tokens = chunk.get("eval_count")
                if isinstance(prompt_tokens, int):
                    token_usage["prompt_tokens"] = prompt_tokens
                if isinstance(completion_tokens, int):
                    token_usage["completion_tokens"] = completion_tokens
                if token_usage:
                    token_usage["total_tokens"] = token_usage.get("prompt_tokens", 0) + token_usage.get(
                        "completion_tokens", 0
                    )
    if token_usage:
        yield ("usage", {"token_usage": token_usage})
    print("OLLAMA_DIRECT_STREAM_SUCCESS", flush=True)


# Inner subprocess script for streaming the AGNO Team. It mirrors the buffered
# ask_agno_team child, but uses arun(stream=True, stream_events=True) and prints
# one NDJSON line per RunContent delta / tool event so the parent can forward
# tokens live. The final answer is also re-emitted under __AGNO_JSON__ so the
# parent can persist a complete answer even when the stream is consumed lazily.
_AGNO_STREAM_CHILD = r'''
import asyncio
import json
import sys
from orchestrator.agno_agents import create_orchestrator

message = sys.argv[1]
target_agents = json.loads(sys.argv[2])
model = sys.argv[3] or None
if target_agents and target_agents != ["orchestrator"]:
    message = f"Route this request to these AGNO team members: {target_agents}. User request: {message}"


def emit(obj):
    sys.stdout.write("__AGNO_EV__" + json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


async def main():
    parts = []
    async for ev in create_orchestrator(model).arun(message, stream=True, stream_events=True):
        name = getattr(ev, "event", "") or ""
        if name == "RunContent":
            delta = getattr(ev, "content", None)
            if isinstance(delta, str) and delta:
                parts.append(delta)
                emit({"kind": "text", "delta": delta})
        elif name == "ToolCallStarted":
            tool = getattr(ev, "tool", None)
            tool_name = getattr(tool, "tool_name", None) if tool else None
            emit({"kind": "tool_start", "tool_id": tool_name or "tool"})
        elif name == "ToolCallCompleted":
            tool = getattr(ev, "tool", None)
            tool_name = getattr(tool, "tool_name", None) if tool else None
            emit({"kind": "tool_result", "tool_id": tool_name or "tool", "status": "ok"})
    sys.stdout.write("__AGNO_JSON__" + json.dumps({"content": "".join(parts)}, ensure_ascii=False) + "\n")
    sys.stdout.flush()


asyncio.run(main())
'''


# Wall-clock bound for the streaming subprocess, matching the buffered
# ``run_command(..., timeout_seconds=180)`` path in ``ask_ollama_cli``.
_AGNO_STREAM_TIMEOUT_SECONDS = 180


async def _iter_agno_team_stream(
    message: str,
    agents: list[str],
    model: str | None = None,
    granted_tools: list[str] | None = None,
) -> AsyncIterator[tuple[str, dict]]:
    """Stream the AGNO Team subprocess line-by-line.

    Yields ``("text", {"delta": ...})`` for RunContent deltas and
    ``("trace", {...})`` for tool events.

    This re-applies the full ``run_command`` security envelope, because the live
    stream needs ``create_subprocess_exec`` rather than the buffered
    ``subprocess.run`` that ``run_command`` uses. All five guarantees are
    enforced here: (1) executable allowlist via ``_assert_allowed``,
    (2) project-scoped cwd via ``require_project_cwd``, (3) env allowlist via
    ``command_env``, (4) a 180s wall-clock timeout (``_AGNO_STREAM_TIMEOUT_SECONDS``,
    matching the buffered path) that kills and reaps the child on expiry, and
    (5) an output cap (``settings.command_output_max_chars``) on both the
    cumulative forwarded stdout and the stderr read on a non-zero exit.
    """
    from orchestrator.command_runner import _assert_allowed, _truncate  # local: security helpers

    args = [sys.executable, "-c", _AGNO_STREAM_CHILD, message, json.dumps(agents, ensure_ascii=False), model or ""]
    _assert_allowed(args)
    safe_cwd = require_project_cwd(PROJECT_ROOT)
    env_overrides = _ollama_env_overrides()
    if granted_tools:
        env_overrides[APPROVAL_GRANTS_ENV] = ",".join(granted_tools)
    env = command_env(env_overrides)
    max_chars = settings.command_output_max_chars

    print(f"AGNO_STREAM_START agents={agents!r} model={model!r}", flush=True)
    proc = await asyncio.create_subprocess_exec(
        *args,
        cwd=str(safe_cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _AGNO_STREAM_TIMEOUT_SECONDS
    gated: list[dict] = []
    forwarded_chars = 0  # cumulative text forwarded to the client (output cap)
    truncated = False

    def _remaining() -> float:
        left = deadline - loop.time()
        if left <= 0:
            raise asyncio.TimeoutError
        return left

    try:
        assert proc.stdout is not None
        while True:
            try:
                raw = await asyncio.wait_for(proc.stdout.readline(), timeout=_remaining())
            except asyncio.TimeoutError:
                print(
                    f"AGNO_STREAM_TIMEOUT agents={agents!r} after={_AGNO_STREAM_TIMEOUT_SECONDS}s",
                    flush=True,
                )
                raise RuntimeError(
                    f"AGNO stream timed out after {_AGNO_STREAM_TIMEOUT_SECONDS}s"
                ) from None
            if not raw:  # EOF
                break
            line = raw.decode("utf-8", errors="replace").rstrip("\n")
            idx = line.find("__AGNO_EV__")
            if idx != -1:
                try:
                    ev = json.loads(line[idx + len("__AGNO_EV__"):])
                except json.JSONDecodeError:
                    continue
                kind = ev.get("kind")
                if kind == "text":
                    delta = ev.get("delta", "")
                    if truncated or forwarded_chars >= max_chars:
                        truncated = True
                        continue
                    remaining = max_chars - forwarded_chars
                    if len(delta) > remaining:
                        delta = delta[:remaining]
                        truncated = True
                    forwarded_chars += len(delta)
                    if delta:
                        yield ("text", {"delta": delta})
                elif kind == "tool_start":
                    yield ("trace", {"event": "tool_start", "data": {"tool_id": ev.get("tool_id"), "agent": "agno"}})
                elif kind == "tool_result":
                    yield (
                        "trace",
                        {"event": "tool_result", "data": {"tool_id": ev.get("tool_id"), "agent": "agno", "status": ev.get("status", "ok")}},
                    )
                continue
            # Start-of-line only: the high-entropy marker is unforgeable from
            # mid-line tool output (see _record_gated_tools).
            if line.startswith(TOOL_GATED_MARKER):
                try:
                    gated.append(json.loads(line[len(TOOL_GATED_MARKER):]))
                except json.JSONDecodeError:
                    pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=_remaining())
        except asyncio.TimeoutError:
            print(
                f"AGNO_STREAM_TIMEOUT agents={agents!r} after={_AGNO_STREAM_TIMEOUT_SECONDS}s",
                flush=True,
            )
            raise RuntimeError(
                f"AGNO stream timed out after {_AGNO_STREAM_TIMEOUT_SECONDS}s"
            ) from None
        if truncated:
            print(f"AGNO_STREAM_TRUNCATED agents={agents!r} cap={max_chars}", flush=True)
            yield ("text", {"delta": f"\n...[truncated at {max_chars} chars]"})
        for payload in gated:
            yield ("trace", {"event": "tool_gated", "data": payload})
        if proc.returncode != 0:
            raw_err = (await proc.stderr.read()).decode("utf-8", errors="replace") if proc.stderr else ""
            err = _truncate(raw_err, max_chars)
            print(f"AGNO_STREAM_FAILED code={proc.returncode} err={err!r}", flush=True)
            raise RuntimeError(err.strip() or f"AGNO stream exited with code {proc.returncode}")
        print(f"AGNO_STREAM_SUCCESS agents={agents!r}", flush=True)
    finally:
        # Kill regardless of returncode (a None returncode means the child is
        # still running, e.g. on timeout/cancellation), then await so it is reaped.
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            try:
                await proc.wait()
            except Exception:  # noqa: BLE001 - best-effort reap
                pass


def _clean_cli_output(text: str) -> str:
    text = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", text)
    text = re.sub(r"\r", "", text)
    text = text.replace("Thinking...", "").replace("...done thinking.", "")
    return text.strip()


def ask_ollama_cli(message: str, model: str | None = None) -> str:
    model = model or effective_model_settings().model
    print(f"OLLAMA_CLI_START command='ollama run {model} --hidethinking --think=false --nowordwrap <message>'", flush=True)
    result = run_command(
        [
            "ollama",
            "run",
            model,
            "--hidethinking",
            "--think=false",
            "--nowordwrap",
            message,
        ],
        timeout_seconds=180,
        env_overrides=_ollama_env_overrides(),
    )
    output = _clean_cli_output(result.stdout or result.stderr)
    if not result.ok:
        print(f"OLLAMA_CLI_FAILED code={result.returncode} output={output!r}", flush=True)
        raise RuntimeError(output or f"ollama CLI failed with code {result.returncode}")
    print("OLLAMA_CLI_SUCCESS", flush=True)
    return output or "Ollama returned an empty response."


def _friendly_model_error(error_text: str) -> str:
    lowered = error_text.lower()
    if "winerror 10054" in lowered or "forcibly closed" in lowered or "10054" in lowered:
        return (
            "Ollama closed the connection while generating. "
            "The local Ollama service is running, but the AGNO/Ollama request failed."
        )
    if "status code: 502" in lowered or "bad gateway" in lowered or "http error 502" in lowered:
        return (
            "Ollama returned 502 Bad Gateway while generating. "
            "Try the request again or restart Ollama if this repeats."
        )
    return error_text


def _looks_like_model_error(content: str) -> bool:
    lowered = content.lower()
    markers = ("winerror 10054", "status code: 502", "bad gateway", "http error 502", "10054")
    return any(marker in lowered for marker in markers)


def _skip_non_system() -> str:
    raise RuntimeError("not a system request")


def run_orchestrator(
    message: str,
    route: RoutingResult | list[str],
    events: list[RunEvent] | None = None,
    model: str | None = None,
    granted_tools: list[str] | None = None,
) -> str:
    errors = []
    agents = route.agents if isinstance(route, RoutingResult) else route
    runners = [
        ("AGNO Team", lambda: ask_agno_team(message, agents, model, granted_tools, events)),
        (
            "Local system",
            lambda: system_status(events)
            if should_use_local_system_status(message)
            else _skip_non_system(),
        ),
    ]
    # The direct/CLI fallbacks are Ollama-specific; they are irrelevant (and
    # would fail) when running against an OpenAI-compatible provider, where the
    # AGNO Team path is the one that works.
    if effective_model_settings().provider != "openai":
        runners.append(("Ollama direct", lambda: ask_ollama_direct(message, model)))
        runners.append(("Ollama CLI", lambda: ask_ollama_cli(message, model)))
    for label, runner in runners:
        if events is not None:
            events.append({"event": "runner_start", "data": {"runner": label}})
        try:
            answer = runner()
            if events is not None:
                events.append({"event": "runner_result", "data": {"runner": label, "status": "ok"}})
            return answer
        except Exception as exc:
            errors.append(f"{label}: {exc}")
            if events is not None:
                events.append(
                    {
                        "event": "runner_error",
                        "data": {"runner": label, "message": _friendly_model_error(str(exc))},
                    }
                )
            print(f"{label.upper().replace(' ', '_')}_FALLBACK error={str(exc)!r}", flush=True)

    return _friendly_model_error("; ".join(errors))


async def _aiter_in_thread(sync_gen_factory):
    """Drive a blocking generator in a worker thread, yielding items as they arrive."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    _DONE = object()

    def pump():
        try:
            for item in sync_gen_factory():
                loop.call_soon_threadsafe(queue.put_nowait, ("item", item))
        except Exception as exc:  # noqa: BLE001 - propagated to the consumer
            loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, ("done", _DONE))

    task = asyncio.create_task(asyncio.to_thread(pump))
    try:
        while True:
            kind, payload = await queue.get()
            if kind == "item":
                yield payload
            elif kind == "error":
                raise payload
            else:
                break
    finally:
        await task


async def stream_orchestrator(
    message: str,
    route: RoutingResult | list[str],
    model: str | None = None,
    granted_tools: list[str] | None = None,
) -> AsyncIterator[tuple[str, dict]]:
    """Stream tokens through the runner chain, yielding typed chunks.

    Yields ``("text", {"delta": ...})`` token fragments, ``("trace", {event,
    data})`` governance/tool events, and a final ``("done", {"answer": ...})``
    carrying the assembled answer for persistence.

    Honours the no-stream-then-fallback rule: it buffers events from a runner
    until that runner emits its first token, only then commits to it. A runner
    that fails before its first token falls through to the next one.
    """
    agents = route.agents if isinstance(route, RoutingResult) else route

    runners: list[tuple[str, object]] = [
        ("AGNO Team", lambda: _iter_agno_team_stream(message, agents, model, granted_tools)),
    ]
    if effective_model_settings().provider != "openai":
        runners.append(
            ("Ollama direct", lambda: _aiter_in_thread(lambda: _iter_ollama_direct(message, model))),
        )

    errors: list[str] = []
    for label, factory in runners:
        yield ("trace", {"event": "runner_start", "data": {"runner": label}})
        committed = False
        text_parts: list[str] = []
        pending_traces: list[tuple[str, dict]] = []
        token_usage: dict | None = None
        source = factory()
        try:
            # AGNO yields (kind, payload); Ollama yields plain delta strings.
            async for produced in source:
                if isinstance(produced, tuple):
                    kind, payload = produced
                else:
                    kind, payload = "text", {"delta": produced}
                if kind == "usage":
                    token_usage = payload.get("token_usage") or payload
                    continue
                if kind == "trace":
                    if committed:
                        yield ("trace", payload)
                    else:
                        pending_traces.append(("trace", payload))
                    continue
                delta = payload.get("delta", "")
                if not delta:
                    continue
                if not committed:
                    committed = True
                    for held in pending_traces:
                        yield held
                text_parts.append(delta)
                yield ("text", {"delta": delta})
            answer = "".join(text_parts).strip()
            if committed and answer and not _looks_like_model_error(answer):
                yield ("trace", {"event": "runner_result", "data": {"runner": label, "status": "ok"}})
                done_payload = {"answer": answer}
                if token_usage:
                    done_payload["token_usage"] = token_usage
                yield ("done", done_payload)
                return
            raise RuntimeError(answer or "runner produced no tokens")
        except Exception as exc:  # noqa: BLE001 - try the next runner pre-first-token
            errors.append(f"{label}: {exc}")
            yield (
                "trace",
                {"event": "runner_error", "data": {"runner": label, "message": _friendly_model_error(str(exc))}},
            )
            print(f"{label.upper().replace(' ', '_')}_STREAM_FALLBACK error={str(exc)!r}", flush=True)
            if committed:
                # Bytes already sent for this runner; cannot fall back further.
                yield ("done", {"answer": "".join(text_parts).strip()})
                return

    # No runner streamed; fall back to the buffered runner chain for an answer.
    fallback_events: list[RunEvent] = []
    answer = await asyncio.to_thread(run_orchestrator, message, route, fallback_events, model, granted_tools)
    for event in fallback_events:
        yield ("trace", {"event": str(event["event"]), "data": dict(event["data"])})
    yield ("done", {"answer": answer})


@app.get("/models")
async def models():
    return list_ollama_models()


@app.get("/settings/model")
async def get_model_settings():
    return {"settings": model_settings_store.public()}


@app.put("/settings/model")
async def update_model_settings(request: ModelSettingsRequest):
    try:
        return {
            "settings": model_settings_store.update(
                provider=request.provider,
                base_url=request.base_url,
                api_key=request.api_key,
                keep_existing_api_key=request.keep_existing_api_key,
                model=request.model,
            )
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/health")
async def health():
    model_settings = effective_model_settings()
    return {
        "status": "ok",
        "framework": "agno",
        "provider": model_settings.provider,
        "model": model_settings.model,
        "ollama_host": model_settings.ollama_host,
        "members": AGNO_MEMBER_NAMES,
        "policy": {
            "mode": settings.tool_policy_mode,
            "allowed_risks": settings.tool_allowed_risks,
            "denied": settings.tool_denied,
        },
    }


@app.get("/diagnostics")
async def diagnostics():
    return {"diagnostics": build_diagnostics(run_history, context_pack_store)}


@app.get("/evals")
async def evals():
    return {"cases": list_eval_cases()}


@app.post("/evals/run")
async def run_evals(compare: str | None = None):
    # `?compare=baseline` runs the suite and diffs it against the committed
    # baseline snapshot, reporting regressions (pass->fail or routing/tool drift)
    # and new cases. Without it, the raw suite result is returned as before.
    if compare == "baseline":
        return {"diff": diff_against_baseline()}
    return {"suite": run_eval_suite()}


@app.get("/evals/baseline")
async def evals_baseline():
    return {"diff": diff_against_baseline()}


@app.get("/tools")
async def tools():
    policy = current_policy()
    inventory = []
    for tool in tool_inventory():
        decision = policy.evaluate(get_tool(tool["id"]))
        inventory.append({**tool, "policy": decision.as_dict()})
    return {"tools": inventory}


@app.get("/workspace/files")
async def workspace_files(limit: int | None = None):
    return {"files": list_workspace_files(limit)}


@app.get("/workspace/file")
async def workspace_file(path: str):
    try:
        return {"file": read_workspace_file(path)}
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/workspace/search")
async def workspace_search(request: WorkspaceSearchRequest):
    try:
        return {"results": search_workspace(request.query, request.limit)}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/runbooks")
async def runbooks():
    return {"runbooks": list_runbooks()}


@app.get("/runbooks/{runbook_id}")
async def runbook_detail(runbook_id: str):
    try:
        return {"runbook": get_runbook(runbook_id).as_dict(include_template=True)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Runbook not found") from exc


@app.post("/runbooks/{runbook_id}/render")
async def render_runbook_prompt(runbook_id: str, request: RunbookRenderRequest):
    try:
        return render_runbook(runbook_id, request.values)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Runbook not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/context/bundle")
async def context_bundle(request: ContextBundleRequest):
    try:
        return build_context_bundle(
            paths=request.paths,
            query=request.query,
            search_limit=request.search_limit,
            max_chars=request.max_chars,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/context/packs")
async def context_packs(limit: int = 50):
    return {"packs": context_pack_store.list_packs(limit)}


@app.post("/context/packs")
async def create_context_pack(request: ContextPackRequest):
    try:
        return {
            "pack": context_pack_store.create_pack(
                name=request.name,
                description=request.description,
                paths=request.paths,
                query=request.query,
                search_limit=request.search_limit,
                max_chars=request.max_chars,
            )
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/context/packs/{pack_id}")
async def context_pack_detail(pack_id: str):
    pack = context_pack_store.get_pack(pack_id)
    if pack is None:
        raise HTTPException(status_code=404, detail="Context pack not found")
    return {"pack": pack}


@app.put("/context/packs/{pack_id}")
async def update_context_pack(pack_id: str, request: ContextPackRequest):
    try:
        pack = context_pack_store.update_pack(
            pack_id,
            name=request.name,
            description=request.description,
            paths=request.paths,
            query=request.query,
            search_limit=request.search_limit,
            max_chars=request.max_chars,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if pack is None:
        raise HTTPException(status_code=404, detail="Context pack not found")
    return {"pack": pack}


@app.post("/context/packs/{pack_id}/bundle")
async def context_pack_bundle(pack_id: str):
    result = context_pack_store.build_bundle(pack_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Context pack not found")
    return result


@app.delete("/context/packs/{pack_id}")
async def delete_context_pack(pack_id: str):
    if not context_pack_store.delete_pack(pack_id):
        raise HTTPException(status_code=404, detail="Context pack not found")
    return {"deleted": True}


@app.get("/runs")
async def runs(limit: int = 50):
    return {"runs": run_history.list_runs(limit)}


@app.get("/runs/summary")
async def runs_summary():
    return {"summary": run_history.summary()}


@app.get("/runs/{run_id}/trace")
async def run_trace(run_id: str):
    trace = build_run_trace(run_history, run_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"trace": trace}


@app.get("/runs/{run_id}")
async def run_detail(run_id: str):
    run = run_history.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"run": run}


@app.get("/runs/{run_id}/events")
async def run_events(run_id: str):
    if run_history.get_run(run_id) is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"events": run_history.list_events(run_id)}


@app.get("/approvals")
async def approvals(status: str | None = None, run_id: str | None = None):
    return {"approvals": run_history.list_approvals(run_id=run_id, status=status)}


@app.post("/approvals/{approval_id}/approve")
async def approve(approval_id: str):
    approval = run_history.resolve_approval(approval_id, "approved")
    if approval is None:
        raise HTTPException(status_code=404, detail="Approval not found")
    run_history.append_event(
        approval["run_id"],
        "approval_resolved",
        {
            "approval_id": approval["id"],
            "tool_id": approval["tool_id"],
            "status": approval["status"],
        },
    )
    return {"approval": approval}


@app.post("/approvals/{approval_id}/deny")
async def deny(approval_id: str):
    approval = run_history.resolve_approval(approval_id, "denied")
    if approval is None:
        raise HTTPException(status_code=404, detail="Approval not found")
    run_history.append_event(
        approval["run_id"],
        "approval_resolved",
        {
            "approval_id": approval["id"],
            "tool_id": approval["tool_id"],
            "status": approval["status"],
        },
    )
    if approval["status"] == "denied":
        run_history.complete_run(approval["run_id"], "Tool approval denied.", status="denied")
    return {"approval": approval}


@app.post("/runs/{run_id}/resume", response_model=ChatResponse)
async def resume_run(run_id: str):
    run = run_history.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if run["status"] != "waiting_approval":
        raise HTTPException(status_code=409, detail="Run is not waiting for approval")
    if not run_history.approvals_ready(run_id):
        raise HTTPException(status_code=409, detail="Run still has unresolved approvals")

    resume_event = {"event": "resume", "data": {"run_id": run_id}}
    run_history.append_event(run_id, "resume", resume_event["data"])
    # TTL is re-validated here: only approvals still within their expires_at
    # window are threaded into the grant set, so a stale approval cannot
    # silently defeat TOOL_APPROVAL_TTL_SECONDS at resume time.
    granted_tools = run_history.granted_tool_ids(run_id)
    events: list[RunEvent] = []
    answer = run_orchestrator(
        run["message"], run["agents"], events, model=run["model"], granted_tools=granted_tools
    )
    record_events(run_id, events)
    done_data = {
        "answer": answer,
        "intent": run["intent"],
        "agents": run["agents"],
        "route": run["route"],
        "run_id": run_id,
        "status": "completed",
    }
    run_history.append_event(run_id, "done", done_data)
    run_history.complete_run(run_id, answer)
    return ChatResponse(
        answer=answer,
        agents_used=run["agents"],
        route=run["route"],
        events=[resume_event] + events + [{"event": "done", "data": done_data}],
        run_id=run_id,
    )


@app.post("/conversations")
async def create_conversation(request: ConversationCreateRequest):
    return {"conversation": conversation_store.create_conversation(request.title)}


@app.get("/conversations")
async def conversations(limit: int = 50):
    return {"conversations": conversation_store.list_conversations(limit)}


@app.get("/conversations/{conversation_id}")
async def conversation_detail(conversation_id: str):
    conversation = conversation_store.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {
        "conversation": conversation,
        "messages": conversation_store.list_messages(conversation_id),
    }


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    if not conversation_store.delete_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"deleted": True}


@app.get("/", response_class=HTMLResponse)
async def index():
    html_file = STATIC_DIR / "index.html"
    return HTMLResponse(content=html_file.read_text(encoding="utf-8"))


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    conversation_id = request.conversation_id
    if conversation_id and conversation_store.get_conversation(conversation_id) is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    try:
        model = resolve_model(request.model)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    team_input = conversation_context(conversation_id, request.message)
    route = route_request(request.message)
    run = run_history.create_run(
        request.message, route.as_dict(), conversation_id=conversation_id, model=model
    )
    if conversation_id:
        conversation_store.append_message(conversation_id, "user", request.message)
    initial_events = initial_run_events(route)
    record_events(run["id"], initial_events)
    approvals = create_required_approvals(run["id"], route)
    if approvals:
        answer = approval_waiting_answer(approvals)
        run_history.complete_run(run["id"], answer, status="waiting_approval")
        approval_events = [approval_required_event(approval) for approval in approvals]
        return ChatResponse(
            answer=answer,
            agents_used=route.agents,
            route=route.as_dict(),
            events=initial_events + approval_events,
            run_id=run["id"],
        )
    events: list[RunEvent] = []
    answer = run_orchestrator(team_input, route, events, model)
    record_events(run["id"], events)
    if conversation_id:
        conversation_store.append_message(conversation_id, "assistant", answer)
    done_data = {"answer": answer, "intent": route.intent, "agents": route.agents, "route": route.as_dict()}
    run_history.append_event(run["id"], "done", done_data)
    run_history.complete_run(run["id"], answer)
    return ChatResponse(
        answer=answer,
        agents_used=route.agents,
        route=route.as_dict(),
        events=initial_events + events + [{"event": "done", "data": done_data}],
        run_id=run["id"],
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _ai_sdk_event_generator(request: ChatRequest, model: str | None, conversation_id: str | None):
    """Emit the run as an AI SDK v5 UI-message-stream (selected by ?protocol=ai-sdk).

    Governance events become TRANSIENT ``data-trace`` parts; ``approval_required``
    becomes a PERSISTENT ``data-approval`` part keyed by approval id; model output
    streams as ``text-delta`` frames between ``text-start`` / ``text-end``.
    """
    message_id = uuid.uuid4().hex
    text_id = uuid.uuid4().hex
    yield ai_sdk_stream.start_frame(message_id)
    try:
        team_input = conversation_context(conversation_id, request.message)
        route = route_request(request.message)
        run = run_history.create_run(
            request.message, route.as_dict(), conversation_id=conversation_id, model=model
        )
        if conversation_id:
            conversation_store.append_message(conversation_id, "user", request.message)
        for event in initial_run_events(route):
            data = dict(event["data"])
            run_history.append_event(run["id"], str(event["event"]), data)
            if event["event"] == "route":
                data = {**data, "run_id": run["id"]}
            yield ai_sdk_stream.trace_frame(str(event["event"]), data)

        approvals = create_required_approvals(run["id"], route)
        if approvals:
            answer = approval_waiting_answer(approvals)
            run_history.complete_run(run["id"], answer, status="waiting_approval")
            for approval in approvals:
                yield ai_sdk_stream.approval_frame(approval["id"], approval_required_event(approval)["data"])
            yield ai_sdk_stream.finish_frame()
            yield ai_sdk_stream.DONE_FRAME
            return

        text_open = False
        answer = ""
        token_usage: dict | None = None
        async for kind, payload in stream_orchestrator(team_input, route, model):
            if kind == "trace":
                run_history.append_event(run["id"], str(payload["event"]), dict(payload["data"]))
                yield ai_sdk_stream.trace_frame(str(payload["event"]), dict(payload["data"]))
            elif kind == "text":
                if not text_open:
                    text_open = True
                    yield ai_sdk_stream.text_start_frame(text_id)
                yield ai_sdk_stream.text_delta_frame(text_id, payload["delta"])
            elif kind == "done":
                answer = payload.get("answer", "")
                token_usage = payload.get("token_usage")
        if not text_open and answer:
            yield ai_sdk_stream.text_start_frame(text_id)
            yield ai_sdk_stream.text_delta_frame(text_id, answer)
            text_open = True
        if text_open:
            yield ai_sdk_stream.text_end_frame(text_id)
        if conversation_id:
            conversation_store.append_message(conversation_id, "assistant", answer)
        done_data = {
            "answer": answer,
            "intent": route.intent,
            "agents": route.agents,
            "route": route.as_dict(),
            "run_id": run["id"],
        }
        if isinstance(token_usage, dict):
            done_data["token_usage"] = token_usage
        run_history.append_event(run["id"], "done", done_data)
        run_history.complete_run(
            run["id"],
            answer,
            token_usage=token_usage if isinstance(token_usage, dict) else None,
        )
        yield ai_sdk_stream.trace_frame("done", done_data)
        yield ai_sdk_stream.finish_frame()
        yield ai_sdk_stream.DONE_FRAME
    except Exception as exc:  # noqa: BLE001 - surface as an AI SDK error part
        yield ai_sdk_stream.error_frame(_friendly_model_error(str(exc)))
        yield ai_sdk_stream.DONE_FRAME


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest, http_request: Request):
    conversation_id = request.conversation_id
    if conversation_id and conversation_store.get_conversation(conversation_id) is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    try:
        model = resolve_model(request.model)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if http_request.query_params.get("protocol") == "ai-sdk":
        return StreamingResponse(
            _ai_sdk_event_generator(request, model, conversation_id),
            media_type="text/event-stream",
            headers={
                ai_sdk_stream.UI_MESSAGE_STREAM_HEADER: ai_sdk_stream.UI_MESSAGE_STREAM_VERSION,
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    async def event_generator():
        team_input = conversation_context(conversation_id, request.message)
        route = route_request(request.message)
        run = run_history.create_run(
            request.message, route.as_dict(), conversation_id=conversation_id, model=model
        )
        if conversation_id:
            conversation_store.append_message(conversation_id, "user", request.message)
        initial_events = initial_run_events(route)
        for event in initial_events:
            run_history.append_event(run["id"], str(event["event"]), dict(event["data"]))
            data = dict(event["data"])
            if event["event"] == "route":
                data["run_id"] = run["id"]
            yield _sse(str(event["event"]), data)
        approvals = create_required_approvals(run["id"], route)
        if approvals:
            answer = approval_waiting_answer(approvals)
            run_history.complete_run(run["id"], answer, status="waiting_approval")
            for approval in approvals:
                yield _sse("approval_required", approval_required_event(approval)["data"])
            yield _sse(
                "done",
                {
                    "answer": answer,
                    "intent": route.intent,
                    "agents": route.agents,
                    "route": route.as_dict(),
                    "run_id": run["id"],
                    "status": "waiting_approval",
                },
            )
            return
        await asyncio.sleep(0.05)
        events: list[RunEvent] = []
        answer = await asyncio.to_thread(run_orchestrator, team_input, route, events, model)
        if conversation_id:
            conversation_store.append_message(conversation_id, "assistant", answer)
        for event in events:
            run_history.append_event(run["id"], str(event["event"]), dict(event["data"]))
            yield _sse(str(event["event"]), dict(event["data"]))
        done_data = {
            "answer": answer,
            "intent": route.intent,
            "agents": route.agents,
            "route": route.as_dict(),
            "run_id": run["id"],
        }
        run_history.append_event(run["id"], "done", done_data)
        run_history.complete_run(run["id"], answer)
        yield _sse(
            "done",
            done_data,
        )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/agents/status")
async def agents_status():
    agents = [{"name": "orchestrator", "online": True, "url": "agno-team"}]
    agents.extend({"name": name, "online": True, "url": "agno-member"} for name in AGNO_MEMBER_NAMES)
    return {"agents": agents}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.orchestrator_port)
