from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import psutil
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from orchestrator.agno_agents import AGNO_MEMBER_NAMES
from shared.config import settings

app = FastAPI(title="AGNO Team Orchestrator")

STATIC_DIR = Path(__file__).parent.parent / "static"
PROJECT_ROOT = Path(__file__).parent.parent
SYSTEM_KEYWORDS = (
    "\u0441\u043e\u0441\u0442\u043e\u044f\u043d\u0438\u0435 \u0441\u0438\u0441\u0442\u0435\u043c\u044b",
    "\u0441\u0438\u0441\u0442\u0435\u043c",
    "cpu",
    "memory",
    "\u043f\u0430\u043c\u044f\u0442",
    "\u0434\u0438\u0441\u043a",
    "disk",
)
ROUTE_KEYWORDS = {
    "docs": ("doc", "docs", "readme", "documentation", "\u0434\u043e\u043a\u0443\u043c\u0435\u043d", "\u0440\u0438\u0434\u043c\u0438"),
    "code": ("code", "lint", "refactor", "review", "\u043a\u043e\u0434", "\u0440\u0435\u0444\u0430\u043a\u0442", "\u043b\u0438\u043d\u0442"),
    "devops": ("test", "build", "ci", "deploy", "\u0442\u0435\u0441\u0442", "\u0441\u0431\u043e\u0440"),
    "system": SYSTEM_KEYWORDS,
    "docker": ("docker", "container", "image", "\u043a\u043e\u043d\u0442\u0435\u0439\u043d\u0435\u0440"),
    "db": ("sql", "database", "schema", "migration", "\u0431\u0430\u0437", "\u0441\u0445\u0435\u043c"),
}

class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    agents_used: list[str]

def route_agents(message: str) -> list[str]:
    lowered = message.lower()
    agents = []
    for name, keywords in ROUTE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in ("doc", "docs"):
                if re.search(r"\bdocs?\b", lowered):
                    agents.append(name)
                    break
            elif keyword in lowered:
                agents.append(name)
                break
    return agents or ["orchestrator"]

def system_status() -> str:
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

    return (
        "\u0421\u043e\u0441\u0442\u043e\u044f\u043d\u0438\u0435 \u0441\u0438\u0441\u0442\u0435\u043c\u044b:\n"
        f"CPU: {cpu}%\n"
        f"Memory: {memory.percent}% ({used_mb}MB / {total_mb}MB)\n"
        f"Disk:\n{disk_block}"
    )


def should_use_local_system_status(message: str) -> bool:
    lowered = message.lower()
    return any(keyword in lowered for keyword in SYSTEM_KEYWORDS)


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


def run_orchestrator(message: str, agents: list[str]) -> str:
    errors = []
    for label, runner in (
        ("AGNO Team", lambda: ask_agno_team(message, agents)),
        ("Local system", lambda: system_status() if should_use_local_system_status(message) else _skip_non_system()),
        ("Ollama direct", lambda: ask_ollama_direct(message)),
        ("Ollama CLI", lambda: ask_ollama_cli(message)),
    ):
        try:
            return runner()
        except Exception as exc:
            errors.append(f"{label}: {exc}")
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
    }


@app.get("/", response_class=HTMLResponse)
async def index():
    html_file = STATIC_DIR / "index.html"
    return HTMLResponse(content=html_file.read_text(encoding="utf-8"))


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    agents = route_agents(request.message)
    return ChatResponse(answer=run_orchestrator(request.message, agents), agents_used=agents)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    async def event_generator():
        agents = route_agents(request.message)
        intent = "multi" if len(agents) > 1 else agents[0]
        yield _sse("classify", {"intent": intent, "agents": agents})
        for agent in agents:
            yield _sse("agent_start", {"agent": agent})
        await asyncio.sleep(0.05)
        answer = await asyncio.to_thread(run_orchestrator, request.message, agents)
        yield _sse("done", {"answer": answer, "intent": intent})

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