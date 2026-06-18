from __future__ import annotations

from dataclasses import dataclass, field

from orchestrator.routing import route_request


@dataclass(frozen=True)
class EvalCase:
    id: str
    message: str
    expected_agents: tuple[str, ...]
    expected_tools: tuple[str, ...] = field(default_factory=tuple)
    forbidden_tools: tuple[str, ...] = field(default_factory=tuple)
    min_confidence: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "message": self.message,
            "expected_agents": list(self.expected_agents),
            "expected_tools": list(self.expected_tools),
            "forbidden_tools": list(self.forbidden_tools),
            "min_confidence": self.min_confidence,
        }


EVAL_CASES: tuple[EvalCase, ...] = (
    EvalCase(
        id="route_workspace_search_to_code",
        message="search files for FastAPI",
        expected_agents=("code",),
        expected_tools=(
            "code.list_workspace_files",
            "code.read_workspace_file",
            "code.search_workspace",
        ),
        forbidden_tools=("code.lint_code",),
        min_confidence=0.6,
    ),
    EvalCase(
        id="route_lint_to_code_lint",
        message="please lint this code",
        expected_agents=("code",),
        expected_tools=("code.lint_code",),
        min_confidence=0.6,
    ),
    EvalCase(
        id="route_tests_and_docker_to_multi",
        message="Run tests and check docker images",
        expected_agents=("devops", "docker"),
        expected_tools=("devops.run_tests", "docker.list_images"),
        min_confidence=0.8,
    ),
    EvalCase(
        id="route_docs_to_docs",
        message="write documentation for the API",
        expected_agents=("docs",),
        min_confidence=0.6,
    ),
    EvalCase(
        id="fallback_unknown_to_orchestrator",
        message="what should we do next?",
        expected_agents=("orchestrator",),
        min_confidence=0.3,
    ),
)


def list_eval_cases() -> list[dict[str, object]]:
    return [case.as_dict() for case in EVAL_CASES]


def run_eval_suite() -> dict[str, object]:
    results = [_run_case(case) for case in EVAL_CASES]
    passed = sum(1 for result in results if result["passed"])
    failed = len(results) - passed
    return {
        "total": len(results),
        "passed": passed,
        "failed": failed,
        "pass_rate": passed / len(results) if results else 0,
        "results": results,
    }


def _run_case(case: EvalCase) -> dict[str, object]:
    route = route_request(case.message)
    tool_ids = [tool.id for tool in route.required_tools]
    failures = []
    if list(case.expected_agents) != route.agents:
        failures.append(f"agents expected {list(case.expected_agents)}, got {route.agents}")
    for tool_id in case.expected_tools:
        if tool_id not in tool_ids:
            failures.append(f"missing expected tool {tool_id}")
    for tool_id in case.forbidden_tools:
        if tool_id in tool_ids:
            failures.append(f"forbidden tool present {tool_id}")
    if route.confidence < case.min_confidence:
        failures.append(f"confidence {route.confidence} below {case.min_confidence}")

    return {
        "id": case.id,
        "message": case.message,
        "passed": not failures,
        "failures": failures,
        "actual_agents": route.agents,
        "actual_tools": tool_ids,
        "confidence": route.confidence,
        "reason": route.reason,
    }
