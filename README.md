# AGNO Team Orchestrator

A small multi-agent assistant built around a single AGNO Team orchestrator. The FastAPI server exposes a chat API and a lightweight browser UI, while AGNO routes each request to specialized in-process members: code, database, DevOps, documentation, system, and Docker.

## Stack

- Python 3.11+
- FastAPI + Uvicorn
- AGNO Team
- Ollama model backend
- Pydantic settings

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e .[dev]
copy .env.example .env
```

Edit `.env` if your Ollama model or host differs.

## Run

Start Ollama separately, then run:

```bash
python -m orchestrator.server
```

Open `http://localhost:8000`.

## API

- `GET /health` returns runtime status, AGNO framework name, model, and member list.
- `GET /tools` returns the typed tool inventory with current policy decisions.
- `POST /chat` returns a complete answer.
- `POST /chat/stream` streams route, policy, runner, tool, and final answer events for the browser UI.
- `GET /agents/status` returns the in-process AGNO team member status.

## Tool Policy

Tool execution is guarded by a local policy layer. Each tool has typed metadata for owner agent,
risk level, timeout, arguments, commands, and local system surfaces it may touch. Defaults allow
low and medium risk tools, while explicitly denying process-listing and Docker inspection tools.

Configure policy in `.env`:

```bash
TOOL_POLICY_MODE=safe
TOOL_ALLOWED_RISKS=low,medium
TOOL_ALLOWED=
TOOL_DENIED=system.list_processes,docker.list_containers,docker.list_images
```

Set `TOOL_POLICY_MODE=off` only in a trusted local environment. You can explicitly allow or deny
individual tools by id, for example `docker.list_images`.

## Development Notes

This project intentionally uses AGNO Team as the single orchestration runtime. The earlier LangGraph/A2A client path was removed to keep the architecture focused and easier to maintain.

Some agent tools are still prototype-level placeholders. Commands that touch the local system,
Docker, tests, or linting are now policy-gated, but this should still be treated as a trusted
local tool until authentication, explicit approvals, and deeper sandboxing are added.
