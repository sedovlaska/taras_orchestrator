import json

import pytest

from orchestrator.policy import ToolPolicy
from orchestrator.routing import route_request
from orchestrator.tool_registry import ToolRisk, get_tool, tools_for_agent
from shared.config import settings
from shared.streaming import EventQueue, StreamEvent


def test_settings_defaults():
    assert settings.llm_model
    assert settings.ollama_host.startswith("http")
    assert settings.orchestrator_port == 8000
    assert settings.tool_policy_mode == "safe"


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

    assert lint.agent == "code"
    assert lint.risk == ToolRisk.MEDIUM
    assert lint.touches_filesystem is True
    assert lint.touches_processes is True
    assert lint.timeout_seconds == 30
    assert lint in tools_for_agent("code")


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


def test_route_request_returns_structured_policy_context():
    route = route_request("Run tests and check docker images")

    assert route.agents == ["devops", "docker"]
    assert route.intent == "multi"
    assert route.confidence > 0.5
    assert "keyword matches" in route.reason
    assert any(tool.id == "devops.run_tests" for tool in route.required_tools)
    assert any(decision["tool_id"] == "docker.list_images" for decision in route.policy_decisions)


def test_server_route_agents_keeps_legacy_list_contract():
    from orchestrator.server import route_agents

    assert route_agents("please lint this code") == ["code"]


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
