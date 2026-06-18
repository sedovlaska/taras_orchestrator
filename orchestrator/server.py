from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

import psutil
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from orchestrator.policy import current_policy
from orchestrator.routing import RoutingResult, route_request, should_use_local_system_status
from orchestrator.tool_registry import AGNO_MEMBER_NAMES, get_tool, tool_inventory
from shared.config import settings

app = FastAPI(title="AGNO Team Orchestrator")

STATIC_DIR = Path(__file__).parent.parent / "static"
PROJECT_ROOT = Path(__file__).parent.parent

RunEvent = dict[str, object]

class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    agents_used: list[str]
    route: dict
    events: list[dict]

def route_agents(message: str) -> list[str]:
    return route_request(message).agents

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


def _ollama_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env.pop(key, None)
    env["NO_PROXY"] = "localhost,127.0.0.1,::1"
    env["no_proxy"] = "localhost,127.0.0.1,::1"
    env["OLLAMA_HOST"] = settings.ollama_host
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def ask_agno_team(message: str, agents: list[str]) -> str:
    print(f"AGNO_TEAM_START agents={agents!r} message={message!r}", flush=True)
    code = r'''
import asyncio
import json
import sys
from orchestrator.agno_agents import create_orchestrator

message = sys.argv[1]
target_agents = json.loads(sys.argv[2])
if target_agents and target_agents != ["orchestrator"]:
    message = f"Route this request to these AGNO team members: {target_agents}. User request: {message}"
resp = create_orchestrator().run(message)
print("__AGNO_JSON__" + json.dumps({"content": str(resp.content)}, ensure_ascii=False), flush=True)
'''
    env = _ollama_env()
    result = subprocess.run(
        [sys.executable, "-c", code, message, json.dumps(agents, ensure_ascii=False)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
        env=env,
    )
    marker = "__AGNO_JSON__"
    marker_pos = result.stdout.rfind(marker)
    if result.returncode == 0 and marker_pos != -1:
        payload = json.loads(result.stdout[marker_pos + len(marker):].strip())
        content = payload.get("content", "")
        if content and not _looks_like_model_error(content):
            print(f"AGNO_TEAM_SUCCESS agents={agents!r}", flush=True)
            return content

    details = (result.stderr or result.stdout or f"exit code {result.returncode}").strip()
    print(f"AGNO_TEAM_FAILED agents={agents!r} error={details!r}", flush=True)
    raise RuntimeError(details or "AGNO Team returned an empty response")


def ask_ollama_direct(message: str) -> str:
    print(f"OLLAMA_DIRECT_START message={message!r}", flush=True)
    url = f"{settings.ollama_host.rstrip('/')}/api/chat"
    payload = {
        "model": settings.llm_model,
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


def _clean_cli_output(text: str) -> str:
    text = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", text)
    text = re.sub(r"\r", "", text)
    text = text.replace("Thinking...", "").replace("...done thinking.", "")
    return text.strip()


def ask_ollama_cli(message: str) -> str:
    print(f"OLLAMA_CLI_START command='ollama run {settings.llm_model} --hidethinking --think=false --nowordwrap <message>'", flush=True)
    env = _ollama_env()
    result = subprocess.run(
        [
            "ollama",
            "run",
            settings.llm_model,
            "--hidethinking",
            "--think=false",
            "--nowordwrap",
            message,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
        env=env,
    )
    output = _clean_cli_output(result.stdout or result.stderr)
    if result.returncode != 0:
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
) -> str:
    errors = []
    agents = route.agents if isinstance(route, RoutingResult) else route
    for label, runner in (
        ("AGNO Team", lambda: ask_agno_team(message, agents)),
        (
            "Local system",
            lambda: system_status(events)
            if should_use_local_system_status(message)
            else _skip_non_system(),
        ),
        ("Ollama direct", lambda: ask_ollama_direct(message)),
        ("Ollama CLI", lambda: ask_ollama_cli(message)),
    ):
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

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "framework": "agno",
        "model": settings.llm_model,
        "ollama_host": settings.ollama_host,
        "members": AGNO_MEMBER_NAMES,
        "policy": {
            "mode": settings.tool_policy_mode,
            "allowed_risks": settings.tool_allowed_risks,
            "denied": settings.tool_denied,
        },
    }


@app.get("/tools")
async def tools():
    policy = current_policy()
    inventory = []
    for tool in tool_inventory():
        decision = policy.evaluate(get_tool(tool["id"]))
        inventory.append({**tool, "policy": decision.as_dict()})
    return {"tools": inventory}


@app.get("/", response_class=HTMLResponse)
async def index():
    html_file = STATIC_DIR / "index.html"
    return HTMLResponse(content=html_file.read_text(encoding="utf-8"))


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    route = route_request(request.message)
    events: list[RunEvent] = []
    answer = run_orchestrator(request.message, route, events)
    return ChatResponse(
        answer=answer,
        agents_used=route.agents,
        route=route.as_dict(),
        events=events,
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    async def event_generator():
        route = route_request(request.message)
        yield _sse("route", route.as_dict())
        yield _sse(
            "classify",
            {
                "intent": route.intent,
                "agents": route.agents,
                "confidence": route.confidence,
                "reason": route.reason,
            },
        )
        for decision in route.policy_decisions:
            yield _sse("policy_decision", decision)
        for agent in route.agents:
            yield _sse("agent_start", {"agent": agent})
        await asyncio.sleep(0.05)
        events: list[RunEvent] = []
        answer = await asyncio.to_thread(run_orchestrator, request.message, route, events)
        for event in events:
            yield _sse(str(event["event"]), dict(event["data"]))
        yield _sse(
            "done",
            {
                "answer": answer,
                "intent": route.intent,
                "agents": route.agents,
                "route": route.as_dict(),
            },
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
