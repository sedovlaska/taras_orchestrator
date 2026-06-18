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
- `GET /diagnostics` returns local readiness checks for config, storage, workspace, tools, and model settings.
- `GET /tools` returns the typed tool inventory with current policy decisions.
- `GET /runs` returns recent chat runs from the local audit log.
- `GET /runs/summary` returns aggregate run analytics for the browser dashboard.
- `GET /runs/{run_id}/trace` exports a run as structured trace data and Markdown.
- `GET /runs/{run_id}` returns one stored run with its route metadata.
- `GET /runs/{run_id}/events` returns the persisted event timeline for a run.
- `GET /approvals` returns pending or historical tool approvals.
- `POST /approvals/{approval_id}/approve` approves one pending tool approval.
- `POST /approvals/{approval_id}/deny` denies one pending tool approval.
- `POST /runs/{run_id}/resume` resumes a run after all required approvals are approved.
- `GET /workspace/files` returns project-scoped text file metadata.
- `GET /workspace/file?path=README.md` returns one project-scoped text file with truncation metadata.
- `POST /workspace/search` searches project-scoped text files for a query.
- `GET /runbooks` returns reusable prompt runbooks for common repo workflows.
- `GET /runbooks/{runbook_id}` returns one runbook and its template.
- `POST /runbooks/{runbook_id}/render` renders a runbook prompt from provided field values.
- `POST /context/bundle` builds a bounded prompt context from workspace files and search results.
- `GET /context/packs` returns saved reusable context pack definitions.
- `POST /context/packs` saves a context pack made from workspace paths and/or search query.
- `POST /context/packs/{pack_id}/bundle` renders a saved context pack into prompt context.
- `DELETE /context/packs/{pack_id}` removes a saved context pack.
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
TOOL_APPROVAL_REQUIRED_RISKS=medium,high
TOOL_APPROVAL_TTL_SECONDS=600
RUN_HISTORY_DB_PATH=.data/run_history.sqlite3
COMMAND_ALLOWED_EXECUTABLES=python,ollama,ruff,pytest,docker
COMMAND_OUTPUT_MAX_CHARS=12000
COMMAND_ENV_ALLOWLIST=PATH,Path,PATHEXT,SYSTEMROOT,SystemRoot,WINDIR,COMSPEC,ComSpec,TEMP,TMP,HOME,USERPROFILE,LOCALAPPDATA,APPDATA,PYTHONPATH,OLLAMA_HOST,NO_PROXY,no_proxy,PYTHONIOENCODING
WORKSPACE_FILE_LIST_LIMIT=500
WORKSPACE_MAX_FILE_BYTES=1048576
WORKSPACE_MAX_FILE_CHARS=40000
WORKSPACE_MAX_SEARCH_RESULTS=100
CONTEXT_BUNDLE_MAX_CHARS=12000
CONTEXT_PACK_DB_PATH=.data/context_packs.sqlite3
```

Set `TOOL_POLICY_MODE=off` only in a trusted local environment. You can explicitly allow or deny
individual tools by id, for example `docker.list_images`. Allowed tools with risk levels listed in
`TOOL_APPROVAL_REQUIRED_RISKS` are paused until approved through the local approval API or UI.
All local subprocess execution goes through a central command runner that enforces executable
allowlists, project-root working directories, environment allowlists, timeouts, and output limits.

## Diagnostics

The `/diagnostics` endpoint performs fast local checks without calling the model: settings sanity,
tool inventory, readable workspace samples, run-history storage, context-pack storage, and model
configuration. The browser UI shows the overall status and individual check results.

## Workspace File Tools

The code agent and browser UI can list, preview, and search files under the project root. These
tools are read-only, reject path escapes, skip ignored runtime directories such as `.git`, `.data`,
and caches, reject binary files, and apply byte, character, and result limits from `.env`.

## Runbooks

The browser UI includes reusable runbooks for repo onboarding, feature planning, code review, test
failure triage, and docs updates. Each runbook declares its fields and renders to a normal chat
prompt through the `/runbooks/{runbook_id}/render` API, so users can inspect or edit the generated
prompt before running it.

## Context Bundles

The `/context/bundle` API packages selected workspace files and search results into a bounded
Markdown context block for chat prompts. The browser UI exposes this through `Attach file` and
`Attach search` actions in the Workspace panel. Bundles reuse the workspace path, binary, ignore,
and size safeguards, and then apply `CONTEXT_BUNDLE_MAX_CHARS` to keep prompts manageable.

Saved context packs persist named path/search bundles in a local SQLite database at
`CONTEXT_PACK_DB_PATH`. The UI can save the current selected file/search, apply a saved pack into
the chat prompt, or delete stale packs.

## Run History

Every chat request is recorded in a local SQLite audit log. The log stores the original message,
selected route, policy decisions, runner/tool events, final answer, and status. By default this
database is written under `.data/run_history.sqlite3`, which is ignored by Git.

The browser UI also shows summary analytics derived from the audit log: total runs, status counts,
agent usage, approval status counts, event counts, and recent runner errors.

Each run can also be exported as a Markdown trace with route details, policy decisions, event
timeline, final answer, and compact event payloads. This is intended for issue reports and PR
debugging without granting direct database access.

## Development Notes

This project intentionally uses AGNO Team as the single orchestration runtime. The earlier LangGraph/A2A client path was removed to keep the architecture focused and easier to maintain.

Some agent tools are still prototype-level placeholders. Commands that touch the local system,
Docker, tests, or linting are now policy-gated, but this should still be treated as a trusted
local tool until authentication, explicit approvals, and deeper sandboxing are added.
