# AGENTS.md

## Build, test, lint

Backend (Python 3.11+):

```bash
python -m pip install -e ".[dev]"   # install + dev deps
ruff check .                         # lint
pytest -q                            # tests
python -m orchestrator.server        # run server on :8000 (needs Ollama running)
```

Frontend (Node 20+):

```bash
npm ci                               # install (uses package-lock.json)
npm run build                        # tsc -b && vite build -> static/dist
npm run dev                          # vite dev server on :5173, proxies API to :8000
```

CI (`.github/workflows/ci.yml`) runs `ruff check .` + `pytest -q` and `npm ci` + `npm run build` on every PR. Make these green locally before opening a PR — that is the verification gate an agent uses to confirm its own work.

## Agent skills

### Issue tracker

Issues and PRDs live in this repo's **GitHub Issues**, managed via the `gh` CLI. External PRs are **not** a triage surface. See `docs/agents/issue-tracker.md`.

### Triage labels

Five canonical triage roles, each mapped to a label string of the same name (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout: one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
