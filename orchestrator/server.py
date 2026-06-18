from __future__ import annotations

import asyncio
import json
import re
import sys
import urllib.request
from pathlib import Path

import psutil
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from orchestrator.command_runner import run_command
from orchestrator.policy import current_policy
from orchestrator.routing import RoutingResult, route_request, should_use_local_system_status
from orchestrator.run_history import RunHistoryStore
from orchestrator.tool_registry import AGNO_MEMBER_NAMES, get_tool, tool_inventory
from orchestrator.workspace import list_workspace_files, read_workspace_file, search_workspace
from shared.config import settings

app = FastAPI(title="AGNO Team Orchestrator")

STATIC_DIR = Path(__file__).parent.parent / "static"
PROJECT_ROOT = Path(__file__).parent.parent

RunEvent = dict[str, object]
run_history = RunHistoryStore.from_settings()

class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    agents_used: list[str]
    route: dict
    events: list[dict]
    run_id: str


class WorkspaceSearchRequest(BaseModel):
    query: str
    limit: int | None = None

def route_agents(message: str) -> list[str]:
    return route_request(message).agents


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
    return {
        "NO_PROXY": "localhost,127.0.0.1,::1",
        "no_proxy": "localhost,127.0.0.1,::1",
        "OLLAMA_HOST": settings.ollama_host,
        "PYTHONIOENCODING": "utf-8",
    }


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
    result = run_command(
        [sys.executable, "-c", code, message, json.dumps(agents, ensure_ascii=False)],
        cwd=PROJECT_ROOT,
        timeout_seconds=180,
        env_overrides=_ollama_env_overrides(),
    )
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
    result = run_command(
        [
            "ollama",
            "run",
            settings.llm_model,
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


@app.get("/runs")
async def runs(limit: int = 50):
    return {"runs": run_history.list_runs(limit)}


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
    events: list[RunEvent] = []
    answer = run_orchestrator(run["message"], run["agents"], events)
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


@app.get("/", response_class=HTMLResponse)
async def index():
    html_file = STATIC_DIR / "index.html"
    return HTMLResponse(content=html_file.read_text(encoding="utf-8"))


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    route = route_request(request.message)
    run = run_history.create_run(request.message, route.as_dict())
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
    answer = run_orchestrator(request.message, route, events)
    record_events(run["id"], events)
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


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    async def event_generator():
        route = route_request(request.message)
        run = run_history.create_run(request.message, route.as_dict())
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
        answer = await asyncio.to_thread(run_orchestrator, request.message, route, events)
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
