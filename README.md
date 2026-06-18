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
- `POST /chat` returns a complete answer.
- `POST /chat/stream` streams Server-Sent Events for the browser UI.
- `GET /agents/status` returns the in-process AGNO team member status.

## Development Notes

This project intentionally uses AGNO Team as the single orchestration runtime. The earlier LangGraph/A2A client path was removed to keep the architecture focused and easier to maintain.

Some agent tools are still prototype-level placeholders. Commands that touch the local system, Docker, tests, or linting should be constrained before exposing this beyond a trusted local environment.
