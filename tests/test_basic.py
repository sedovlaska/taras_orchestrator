import json
import sys

import pytest
from fastapi.testclient import TestClient

from orchestrator.command_runner import command_env, run_command
from orchestrator.policy import ToolPolicy
from orchestrator.run_history import RunHistoryStore
from orchestrator.routing import route_request
from orchestrator.tool_registry import ToolRisk, get_tool, tools_for_agent
from shared.config import settings
from shared.streaming import EventQueue, StreamEvent


def test_settings_defaults():
    assert settings.llm_model
    assert settings.ollama_host.startswith("http")
    assert settings.orchestrator_port == 8000
    assert settings.tool_policy_mode == "safe"
    assert settings.run_history_db_path.endswith(".sqlite3")
    assert settings.workspace_max_file_chars > 0
    assert settings.context_bundle_max_chars > 0
    assert settings.context_pack_db_path.endswith(".sqlite3")
    assert settings.conversations_db_path.endswith(".sqlite3")


def test_stream_event_sse():
    event = StreamEvent(event="classify", data={"intent": "agno", "agents": ["orchestrator"]})
    sse = event.to_sse()
    assert "event: classify" in sse
    assert '"intent": "agno"' in sse
    assert sse.endswith("\n\n")


def test_server_sse_helper_outputs_json():
    from orchestrator.server import _sse

    sse = _sse("done", {"answer": "Privet", "intent": "agno"})
    payload = sse.split("data: ", 1)[1].strip()
    assert json.loads(payload) == {"answer": "Privet", "intent": "agno"}


def test_tool_registry_exposes_typed_metadata():
    lint = get_tool("code.lint_code")
    search = get_tool("code.search_workspace")

    assert lint.agent == "code"
    assert lint.risk == ToolRisk.MEDIUM
    assert lint.touches_filesystem is True
    assert lint.touches_processes is True
    assert lint.timeout_seconds == 30
    assert lint in tools_for_agent("code")
    assert search.risk == ToolRisk.LOW
    assert search.touches_filesystem is True


def test_policy_defaults_allow_medium_and_deny_explicit_tools():
    policy = ToolPolicy(
        allowed_risks={"low", "medium"},
        denied_tools={"docker.list_images"},
    )

    assert policy.evaluate(get_tool("devops.run_tests")).allowed is True
    denied = policy.evaluate(get_tool("docker.list_images"))
    assert denied.allowed is False
    assert denied.reason == "tool explicitly denied"


def test_policy_blocks_project_path_escape():
    policy = ToolPolicy()

    with pytest.raises(PermissionError):
        policy.require_project_path("../outside")


def test_command_runner_executes_allowed_command_and_truncates_output():
    result = run_command(
        [sys.executable, "-c", "print('x' * 100)"],
        timeout_seconds=10,
        max_output_chars=40,
    )

    assert result.ok is True
    assert result.returncode == 0
    assert "[truncated" in result.stdout
    assert result.cwd.endswith("taras_orchestrator")


def test_command_runner_blocks_unapproved_executables(monkeypatch):
    monkeypatch.setattr(settings, "command_allowed_executables", "python")

    with pytest.raises(PermissionError):
        run_command(["definitely-not-python"], timeout_seconds=1)


def test_command_runner_blocks_cwd_escape(tmp_path):
    with pytest.raises(PermissionError):
        run_command([sys.executable, "-c", "print('no')"], cwd=tmp_path, timeout_seconds=1)


def test_command_runner_blocks_unapproved_env_vars():
    with pytest.raises(PermissionError):
        command_env({"SECRET_TOKEN": "not-allowed"})


def test_route_request_returns_structured_policy_context():
    route = route_request("Run tests and check docker images")

    assert route.agents == ["devops", "docker"]
    assert route.intent == "multi"
    assert route.confidence > 0.5
    assert "keyword matches" in route.reason
    assert any(tool.id == "devops.run_tests" for tool in route.required_tools)
    assert any(decision["tool_id"] == "docker.list_images" for decision in route.policy_decisions)


def test_route_request_uses_read_only_workspace_tools_for_file_search():
    route = route_request("search files for FastAPI")

    assert route.agents == ["code"]
    assert [tool.id for tool in route.required_tools] == [
        "code.list_workspace_files",
        "code.read_workspace_file",
        "code.search_workspace",
    ]
    assert all(decision["risk"] == "low" for decision in route.policy_decisions)


def test_server_route_agents_keeps_legacy_list_contract():
    from orchestrator.server import route_agents

    assert route_agents("please lint this code") == ["code"]


def test_workspace_tools_are_project_scoped_and_truncate(tmp_path, monkeypatch):
    import orchestrator.workspace as workspace

    (tmp_path / "README.md").write_text("AGNO Team Orchestrator\nneedle\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("secret", encoding="utf-8")
    monkeypatch.setattr(workspace, "PROJECT_ROOT", tmp_path.resolve())
    monkeypatch.setattr(settings, "workspace_file_list_limit", 20)
    monkeypatch.setattr(settings, "workspace_max_file_bytes", 1024)
    monkeypatch.setattr(settings, "workspace_max_file_chars", 10)

    files = workspace.list_workspace_files()
    read = workspace.read_workspace_file("README.md")

    assert [file["path"] for file in files] == ["README.md"]
    assert read["path"] == "README.md"
    assert read["content"] == "AGNO Team "
    assert read["truncated"] is True
    with pytest.raises(PermissionError):
        workspace.read_workspace_file("../outside.txt")


def test_workspace_search_skips_binary_files(tmp_path, monkeypatch):
    import orchestrator.workspace as workspace

    (tmp_path / "notes.txt").write_text("first line\nneedle match\n", encoding="utf-8")
    (tmp_path / "blob.bin").write_bytes(b"needle\0binary")
    monkeypatch.setattr(workspace, "PROJECT_ROOT", tmp_path.resolve())
    monkeypatch.setattr(settings, "workspace_max_file_bytes", 1024)
    monkeypatch.setattr(settings, "workspace_max_search_results", 20)

    files = workspace.list_workspace_files()
    results = workspace.search_workspace("needle")

    assert [file["path"] for file in files] == ["notes.txt"]
    assert results == [{"path": "notes.txt", "line": 2, "preview": "needle match"}]
    with pytest.raises(ValueError):
        workspace.read_workspace_file("blob.bin")


def test_workspace_api_lists_reads_and_searches(tmp_path, monkeypatch):
    import orchestrator.server as server
    import orchestrator.workspace as workspace

    (tmp_path / "README.md").write_text("FastAPI workspace endpoint\n", encoding="utf-8")
    monkeypatch.setattr(workspace, "PROJECT_ROOT", tmp_path.resolve())
    monkeypatch.setattr(settings, "workspace_file_list_limit", 20)
    monkeypatch.setattr(settings, "workspace_max_file_bytes", 1024)
    monkeypatch.setattr(settings, "workspace_max_file_chars", 100)
    monkeypatch.setattr(settings, "workspace_max_search_results", 20)
    client = TestClient(server.app)

    files = client.get("/workspace/files").json()["files"]
    read = client.get("/workspace/file", params={"path": "README.md"}).json()["file"]
    search = client.post("/workspace/search", json={"query": "workspace"}).json()["results"]
    escape = client.get("/workspace/file", params={"path": "../outside.txt"})

    assert files[0]["path"] == "README.md"
    assert read["content"] == "FastAPI workspace endpoint\n"
    assert search[0]["path"] == "README.md"
    assert escape.status_code == 403


def test_runbook_registry_renders_defaults_and_validates_required_fields():
    from orchestrator.runbooks import get_runbook, list_runbooks, render_runbook

    runbooks = list_runbooks()
    onboarding = render_runbook("repo_onboarding", {})

    assert len(runbooks) >= 5
    assert get_runbook("feature_plan").suggested_agent == "code"
    assert "Onboard me to this repository" in onboarding["prompt"]
    with pytest.raises(ValueError):
        render_runbook("feature_plan", {})
    with pytest.raises(KeyError):
        get_runbook("missing")


def test_runbook_api_lists_details_and_renders_prompt():
    import orchestrator.server as server

    client = TestClient(server.app)

    listing = client.get("/runbooks").json()["runbooks"]
    detail = client.get("/runbooks/feature_plan").json()["runbook"]
    rendered = client.post(
        "/runbooks/feature_plan/render",
        json={"values": {"feature": "workspace snapshots", "files": "orchestrator/"}},
    ).json()
    invalid = client.post("/runbooks/feature_plan/render", json={"values": {}})
    missing = client.get("/runbooks/not-real")

    assert listing[0]["id"] == "repo_onboarding"
    assert "{{feature}}" in detail["template"]
    assert "workspace snapshots" in rendered["prompt"]
    assert invalid.status_code == 422
    assert missing.status_code == 404


def test_context_bundle_builds_bounded_file_and_search_context(tmp_path, monkeypatch):
    import orchestrator.workspace as workspace
    from orchestrator.context_bundles import build_context_bundle

    (tmp_path / "README.md").write_text("alpha beta gamma\nsecond beta line\n", encoding="utf-8")
    monkeypatch.setattr(workspace, "PROJECT_ROOT", tmp_path.resolve())
    monkeypatch.setattr(settings, "workspace_max_file_bytes", 1024)
    monkeypatch.setattr(settings, "workspace_max_file_chars", 100)
    monkeypatch.setattr(settings, "workspace_max_search_results", 20)
    monkeypatch.setattr(settings, "context_bundle_max_chars", 40)

    bundle = build_context_bundle(paths=["README.md"], query="beta", max_chars=30)
    partial = build_context_bundle(paths=["missing.md"], query="nomatch", max_chars=200)

    assert bundle["max_chars"] == 30
    assert bundle["truncated"] is True
    assert "File: README.md" in bundle["prompt_context"]
    assert partial["errors"][0]["source"] == "missing.md"
    assert "No matches found" in partial["prompt_context"]
    with pytest.raises(ValueError):
        build_context_bundle()


def test_context_bundle_api_builds_prompt_context(tmp_path, monkeypatch):
    import orchestrator.server as server
    import orchestrator.workspace as workspace

    (tmp_path / "notes.txt").write_text("context bundle api\n", encoding="utf-8")
    monkeypatch.setattr(workspace, "PROJECT_ROOT", tmp_path.resolve())
    monkeypatch.setattr(settings, "workspace_max_file_bytes", 1024)
    monkeypatch.setattr(settings, "workspace_max_file_chars", 100)
    monkeypatch.setattr(settings, "context_bundle_max_chars", 500)
    client = TestClient(server.app)

    response = client.post("/context/bundle", json={"paths": ["notes.txt"], "query": "bundle"})
    invalid = client.post("/context/bundle", json={})

    payload = response.json()
    assert response.status_code == 200
    assert payload["sections"][0]["title"] == "File: notes.txt"
    assert "context bundle api" in payload["prompt_context"]
    assert invalid.status_code == 422


def test_context_pack_store_persists_updates_bundles_and_deletes(tmp_path, monkeypatch):
    import orchestrator.workspace as workspace
    from orchestrator.context_packs import ContextPackStore

    (tmp_path / "notes.txt").write_text("saved context pack\n", encoding="utf-8")
    monkeypatch.setattr(workspace, "PROJECT_ROOT", tmp_path.resolve())
    monkeypatch.setattr(settings, "workspace_max_file_bytes", 1024)
    monkeypatch.setattr(settings, "workspace_max_file_chars", 100)
    monkeypatch.setattr(settings, "context_bundle_max_chars", 500)
    store = ContextPackStore(tmp_path / "packs.sqlite3")

    pack = store.create_pack(name="Review pack", paths=["notes.txt"], query="saved")
    updated = store.update_pack(pack["id"], name="Updated pack", paths=["notes.txt"])
    bundle = store.build_bundle(pack["id"])

    assert store.list_packs()[0]["id"] == pack["id"]
    assert updated["name"] == "Updated pack"
    assert bundle["pack"]["id"] == pack["id"]
    assert "saved context pack" in bundle["bundle"]["prompt_context"]
    assert store.delete_pack(pack["id"]) is True
    assert store.get_pack(pack["id"]) is None
    with pytest.raises(ValueError):
        store.create_pack(name="", paths=["notes.txt"])


def test_context_pack_api_crud_and_bundle(tmp_path, monkeypatch):
    import orchestrator.server as server
    import orchestrator.workspace as workspace
    from orchestrator.context_packs import ContextPackStore

    (tmp_path / "README.md").write_text("api context pack\n", encoding="utf-8")
    monkeypatch.setattr(workspace, "PROJECT_ROOT", tmp_path.resolve())
    monkeypatch.setattr(settings, "workspace_max_file_bytes", 1024)
    monkeypatch.setattr(settings, "workspace_max_file_chars", 100)
    monkeypatch.setattr(settings, "context_bundle_max_chars", 500)
    monkeypatch.setattr(server, "context_pack_store", ContextPackStore(tmp_path / "packs.sqlite3"))
    client = TestClient(server.app)

    created = client.post(
        "/context/packs",
        json={"name": "API pack", "paths": ["README.md"], "query": "context"},
    )
    pack_id = created.json()["pack"]["id"]
    listing = client.get("/context/packs").json()["packs"]
    rendered = client.post(f"/context/packs/{pack_id}/bundle").json()
    updated = client.put(
        f"/context/packs/{pack_id}",
        json={"name": "Renamed pack", "paths": ["README.md"]},
    ).json()["pack"]
    deleted = client.delete(f"/context/packs/{pack_id}")
    missing = client.get(f"/context/packs/{pack_id}")
    invalid = client.post("/context/packs", json={"name": "Invalid"})

    assert created.status_code == 200
    assert listing[0]["name"] == "API pack"
    assert "api context pack" in rendered["bundle"]["prompt_context"]
    assert updated["name"] == "Renamed pack"
    assert deleted.status_code == 200
    assert missing.status_code == 404
    assert invalid.status_code == 422


def test_diagnostics_report_ok_for_local_runtime(tmp_path, monkeypatch):
    import orchestrator.diagnostics as diagnostics
    import orchestrator.workspace as workspace
    from orchestrator.context_packs import ContextPackStore

    (tmp_path / "README.md").write_text("diagnostics file\n", encoding="utf-8")
    monkeypatch.setattr(workspace, "PROJECT_ROOT", tmp_path.resolve())
    monkeypatch.setattr(settings, "workspace_max_file_bytes", 1024)
    report = diagnostics.build_diagnostics(
        run_history=RunHistoryStore(tmp_path / "runs.sqlite3"),
        context_packs=ContextPackStore(tmp_path / "packs.sqlite3"),
    )

    assert report["status"] == "ok"
    assert report["counts"]["error"] == 0
    assert {check["id"] for check in report["checks"]} >= {
        "settings",
        "tools",
        "workspace",
        "run_history",
        "context_packs",
        "model",
    }


def test_diagnostics_api_uses_current_stores(tmp_path, monkeypatch):
    import orchestrator.server as server
    import orchestrator.workspace as workspace
    from orchestrator.context_packs import ContextPackStore

    (tmp_path / "README.md").write_text("diagnostics api\n", encoding="utf-8")
    monkeypatch.setattr(workspace, "PROJECT_ROOT", tmp_path.resolve())
    monkeypatch.setattr(server, "run_history", RunHistoryStore(tmp_path / "runs.sqlite3"))
    monkeypatch.setattr(server, "context_pack_store", ContextPackStore(tmp_path / "packs.sqlite3"))
    client = TestClient(server.app)

    response = client.get("/diagnostics")
    payload = response.json()["diagnostics"]

    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert payload["counts"]["ok"] >= 5


def test_eval_suite_passes_current_routing_expectations():
    from orchestrator.evals import list_eval_cases, run_eval_suite

    cases = list_eval_cases()
    suite = run_eval_suite()

    assert len(cases) >= 5
    assert suite["total"] == len(cases)
    assert suite["failed"] == 0
    assert suite["pass_rate"] == 1
    assert any(result["id"] == "route_workspace_search_to_code" for result in suite["results"])


def test_eval_api_lists_and_runs_suite():
    import orchestrator.server as server

    client = TestClient(server.app)
    listing = client.get("/evals")
    run = client.post("/evals/run")

    assert listing.status_code == 200
    assert run.status_code == 200
    assert run.json()["suite"]["failed"] == 0


def test_index_exposes_runbook_controls():
    import orchestrator.server as server

    client = TestClient(server.app)
    html = client.get("/").text

    assert 'id="runbook-select"' in html
    assert "loadRunbooks()" in html
    assert 'id="context-file-btn"' in html
    assert 'id="context-pack-save-btn"' in html
    assert 'id="diagnostics-refresh-btn"' in html
    assert "exportRunTrace" in html
    assert 'id="eval-run-btn"' in html


def test_run_history_store_persists_runs_and_events(tmp_path):
    store = RunHistoryStore(tmp_path / "runs.sqlite3")
    route = route_request("please lint this code").as_dict()
    run = store.create_run("please lint this code", route)

    stored_event = store.append_event(run["id"], "route", route)
    store.complete_run(run["id"], "done")

    runs = store.list_runs()
    detail = store.get_run(run["id"])
    events = store.list_events(run["id"])

    assert runs[0]["id"] == run["id"]
    assert runs[0]["answer"] == "done"
    assert detail["route"]["agents"] == ["code"]
    assert events == [stored_event]
    assert events[0]["data"]["intent"] == "code"


def test_run_history_summary_counts_status_agents_events_and_errors(tmp_path):
    store = RunHistoryStore(tmp_path / "runs.sqlite3")
    route = route_request("please lint this code").as_dict()
    run = store.create_run("please lint this code", route)
    store.append_event(run["id"], "route", route)
    store.append_event(run["id"], "runner_error", {"runner": "fake", "message": "boom"})
    store.complete_run(run["id"], "failed", status="failed")

    summary = store.summary()

    assert summary["total_runs"] == 1
    assert summary["by_status"] == [{"status": "failed", "count": 1}]
    assert summary["by_agent"] == [{"agent": "code", "count": 1}]
    assert any(row["event"] == "runner_error" for row in summary["by_event"])
    assert summary["recent_errors"][0]["data"]["message"] == "boom"


def test_chat_api_records_run_history(tmp_path, monkeypatch):
    import orchestrator.server as server

    store = RunHistoryStore(tmp_path / "runs.sqlite3")
    monkeypatch.setattr(server, "run_history", store)

    def fake_runner(message, route, events, model=None):
        events.append({"event": "runner_start", "data": {"runner": "fake"}})
        events.append({"event": "runner_result", "data": {"runner": "fake", "status": "ok"}})
        return f"answer: {message}"

    monkeypatch.setattr(server, "run_orchestrator", fake_runner)
    client = TestClient(server.app)

    response = client.post("/chat", json={"message": "cpu status"})
    payload = response.json()
    run_id = payload["run_id"]
    runs = client.get("/runs").json()["runs"]
    events = client.get(f"/runs/{run_id}/events").json()["events"]

    assert response.status_code == 200
    assert payload["answer"] == "answer: cpu status"
    assert runs[0]["id"] == run_id
    assert any(event["event"] == "route" for event in events)
    assert any(event["event"] == "runner_result" for event in events)
    assert events[-1]["event"] == "done"


def test_run_summary_api_uses_current_history_store(tmp_path, monkeypatch):
    import orchestrator.server as server

    store = RunHistoryStore(tmp_path / "runs.sqlite3")
    monkeypatch.setattr(server, "run_history", store)
    run = store.create_run("cpu status", route_request("cpu status").as_dict())
    store.complete_run(run["id"], "done")
    client = TestClient(server.app)

    response = client.get("/runs/summary")
    payload = response.json()["summary"]

    assert response.status_code == 200
    assert payload["total_runs"] == 1
    assert payload["by_status"] == [{"status": "completed", "count": 1}]


def test_run_trace_export_includes_route_policy_and_timeline(tmp_path):
    from orchestrator.trace_export import build_run_trace

    store = RunHistoryStore(tmp_path / "runs.sqlite3")
    route = route_request("please lint this code").as_dict()
    run = store.create_run("please lint this code", route)
    store.append_event(run["id"], "route", route)
    store.append_event(
        run["id"],
        "policy_decision",
        {"tool_id": "code.lint_code", "allowed": True, "risk": "medium", "reason": "test"},
    )
    store.complete_run(run["id"], "done")

    trace = build_run_trace(store, run["id"])

    assert trace["run"]["id"] == run["id"]
    assert trace["timeline"][0]["title"] == "Route to code"
    assert "## Policy Decisions" in trace["markdown"]
    assert "code.lint_code" in trace["markdown"]
    assert build_run_trace(store, "missing") is None


def test_run_trace_api_uses_current_history_store(tmp_path, monkeypatch):
    import orchestrator.server as server

    store = RunHistoryStore(tmp_path / "runs.sqlite3")
    monkeypatch.setattr(server, "run_history", store)
    route = route_request("cpu status").as_dict()
    run = store.create_run("cpu status", route)
    store.append_event(run["id"], "route", route)
    store.complete_run(run["id"], "done")
    client = TestClient(server.app)

    response = client.get(f"/runs/{run['id']}/trace")
    missing = client.get("/runs/missing/trace")

    assert response.status_code == 200
    assert response.json()["trace"]["run"]["id"] == run["id"]
    assert "# Run Trace:" in response.json()["trace"]["markdown"]
    assert missing.status_code == 404


def test_chat_api_requires_approval_before_medium_risk_tools(tmp_path, monkeypatch):
    import orchestrator.server as server

    store = RunHistoryStore(tmp_path / "runs.sqlite3")
    monkeypatch.setattr(server, "run_history", store)
    called = {"runner": False}

    def fake_runner(message, route, events):
        called["runner"] = True
        events.append({"event": "runner_result", "data": {"runner": "fake", "status": "ok"}})
        return f"answer: {message}"

    monkeypatch.setattr(server, "run_orchestrator", fake_runner)
    client = TestClient(server.app)

    response = client.post("/chat", json={"message": "please lint this code"})
    payload = response.json()
    approvals = client.get("/approvals", params={"run_id": payload["run_id"]}).json()["approvals"]
    events = client.get(f"/runs/{payload['run_id']}/events").json()["events"]
    run = client.get(f"/runs/{payload['run_id']}").json()["run"]

    assert response.status_code == 200
    assert called["runner"] is False
    assert run["status"] == "waiting_approval"
    assert payload["answer"].startswith("Tool approval required")
    assert [approval["tool_id"] for approval in approvals] == ["code.lint_code"]
    assert any(event["event"] == "approval_required" for event in events)

    approval_id = approvals[0]["id"]
    approved = client.post(f"/approvals/{approval_id}/approve").json()["approval"]
    resumed = client.post(f"/runs/{payload['run_id']}/resume").json()

    assert approved["status"] == "approved"
    assert called["runner"] is True
    assert resumed["answer"] == "answer: please lint this code"
    assert client.get(f"/runs/{payload['run_id']}").json()["run"]["status"] == "completed"


def test_deny_approval_marks_run_denied(tmp_path, monkeypatch):
    import orchestrator.server as server

    store = RunHistoryStore(tmp_path / "runs.sqlite3")
    monkeypatch.setattr(server, "run_history", store)
    monkeypatch.setattr(server, "run_orchestrator", lambda message, route, events: "unexpected")
    client = TestClient(server.app)

    response = client.post("/chat", json={"message": "please lint this code"})
    run_id = response.json()["run_id"]
    approval_id = client.get("/approvals", params={"run_id": run_id}).json()["approvals"][0]["id"]

    denied = client.post(f"/approvals/{approval_id}/deny").json()["approval"]
    resume = client.post(f"/runs/{run_id}/resume")
    run = client.get(f"/runs/{run_id}").json()["run"]

    assert denied["status"] == "denied"
    assert resume.status_code == 409
    assert run["status"] == "denied"


def test_agno_agents_import_with_model_dependencies():
    from orchestrator.agno_agents import AGNO_MEMBER_NAMES, create_orchestrator

    assert "code" in AGNO_MEMBER_NAMES
    assert callable(create_orchestrator)


def test_models_api_parses_ollama_tags(monkeypatch):
    import orchestrator.server as server

    # Exercise the real parser by faking only the HTTP call.
    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def read(self):
            return json.dumps(self._payload).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class FakeOpener:
        def open(self, request, timeout=None):
            return FakeResponse(
                {"models": [{"name": "qwen3:1.7b"}, {"name": "llama3:8b"}, {"size": 1}]}
            )

    monkeypatch.setattr(server.urllib.request, "build_opener", lambda *a, **k: FakeOpener())
    client = TestClient(server.app)

    payload = client.get("/models").json()

    assert payload["reachable"] is True
    assert payload["models"] == ["qwen3:1.7b", "llama3:8b"]
    assert payload["default_model"] == settings.llm_model


def test_models_api_degrades_when_ollama_unreachable(monkeypatch):
    import orchestrator.server as server

    class FailingOpener:
        def open(self, request, timeout=None):
            raise OSError("connection refused")

    monkeypatch.setattr(server.urllib.request, "build_opener", lambda *a, **k: FailingOpener())
    client = TestClient(server.app)

    payload = client.get("/models").json()

    assert payload["reachable"] is False
    assert payload["models"] == []
    assert payload["default_model"] == settings.llm_model


def test_chat_threads_model_override_to_runner(tmp_path, monkeypatch):
    import orchestrator.server as server

    monkeypatch.setattr(server, "run_history", RunHistoryStore(tmp_path / "runs.sqlite3"))
    seen = {}

    def fake_runner(message, route, events, model=None):
        seen["model"] = model
        events.append({"event": "runner_result", "data": {"runner": "fake", "status": "ok"}})
        return "ok"

    monkeypatch.setattr(server, "run_orchestrator", fake_runner)
    client = TestClient(server.app)

    response = client.post("/chat", json={"message": "cpu status", "model": "llama3:8b"})
    invalid = client.post("/chat", json={"message": "cpu status", "model": "   "})

    assert response.status_code == 200
    assert seen["model"] == "llama3:8b"
    assert invalid.status_code == 422


def test_chat_defaults_model_to_none_when_omitted(tmp_path, monkeypatch):
    import orchestrator.server as server

    monkeypatch.setattr(server, "run_history", RunHistoryStore(tmp_path / "runs.sqlite3"))
    seen = {}

    def fake_runner(message, route, events, model=None):
        seen["model"] = model
        events.append({"event": "runner_result", "data": {"runner": "fake", "status": "ok"}})
        return "ok"

    monkeypatch.setattr(server, "run_orchestrator", fake_runner)
    client = TestClient(server.app)

    response = client.post("/chat", json={"message": "cpu status"})

    assert response.status_code == 200
    assert seen["model"] is None


def test_conversation_store_creates_appends_lists_and_deletes(tmp_path):
    from orchestrator.conversations import ConversationStore

    store = ConversationStore(tmp_path / "conversations.sqlite3")
    convo = store.create_conversation("My thread")
    store.append_message(convo["id"], "user", "hello")
    store.append_message(convo["id"], "assistant", "hi there")

    messages = store.list_messages(convo["id"])
    listing = store.list_conversations()
    detail = store.get_conversation(convo["id"])

    assert convo["title"] == "My thread"
    assert listing[0]["id"] == convo["id"]
    assert detail["id"] == convo["id"]
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert [m["content"] for m in messages] == ["hello", "hi there"]
    assert store.delete_conversation(convo["id"]) is True
    assert store.get_conversation(convo["id"]) is None
    assert store.list_messages(convo["id"]) == []
    assert store.delete_conversation(convo["id"]) is False


def test_conversation_api_crud_and_multi_turn_context(tmp_path, monkeypatch):
    import orchestrator.server as server
    from orchestrator.conversations import ConversationStore

    monkeypatch.setattr(server, "run_history", RunHistoryStore(tmp_path / "runs.sqlite3"))
    monkeypatch.setattr(
        server, "conversation_store", ConversationStore(tmp_path / "conversations.sqlite3")
    )
    seen = {}

    def fake_runner(message, route, events, model=None):
        seen["message"] = message
        events.append({"event": "runner_result", "data": {"runner": "fake", "status": "ok"}})
        return f"answer: {message}"

    monkeypatch.setattr(server, "run_orchestrator", fake_runner)
    client = TestClient(server.app)

    created = client.post("/conversations", json={"title": "Session"})
    conversation_id = created.json()["conversation"]["id"]

    first = client.post(
        "/chat", json={"message": "cpu status", "conversation_id": conversation_id}
    )
    assert first.status_code == 200
    assert seen["message"] == "cpu status"  # no prior turns yet

    second = client.post(
        "/chat", json={"message": "disk usage", "conversation_id": conversation_id}
    )
    assert second.status_code == 200
    # prior turns are fed as context to the second turn
    assert "cpu status" in seen["message"]
    assert "Current user message: disk usage" in seen["message"]

    listing = client.get("/conversations").json()["conversations"]
    detail = client.get(f"/conversations/{conversation_id}").json()
    messages = detail["messages"]

    assert listing[0]["id"] == conversation_id
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]
    assert messages[0]["content"] == "cpu status"
    assert messages[1]["content"] == "answer: cpu status"

    deleted = client.delete(f"/conversations/{conversation_id}")
    missing_detail = client.get(f"/conversations/{conversation_id}")
    missing_chat = client.post(
        "/chat", json={"message": "cpu status", "conversation_id": conversation_id}
    )

    assert deleted.status_code == 200
    assert missing_detail.status_code == 404
    assert missing_chat.status_code == 404


async def test_event_queue():
    queue = EventQueue()
    await queue.emit(StreamEvent(event="test", data={"key": "value"}))
    await queue.emit(StreamEvent(event="done", data={"result": "ok"}))
    await queue.finish()

    events = []
    async for event in queue.stream():
        events.append(event)

    assert len(events) == 2
    assert events[0].event == "test"
    assert events[0].data == {"key": "value"}
    assert events[1].event == "done"
