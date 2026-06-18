from __future__ import annotations

import re
from dataclasses import dataclass

from .policy import ToolPolicy, current_policy
from .tool_registry import AGNO_MEMBER_NAMES, ToolSpec, tools_for_agent


SYSTEM_KEYWORDS = (
    "состояние системы",
    "систем",
    "cpu",
    "memory",
    "памят",
    "диск",
    "disk",
)

ROUTE_KEYWORDS = {
    "docs": ("doc", "docs", "readme", "documentation", "докумен", "ридми"),
    "code": ("code", "lint", "refactor", "review", "код", "рефакт", "линт"),
    "devops": ("test", "build", "ci", "deploy", "тест", "сбор"),
    "system": SYSTEM_KEYWORDS,
    "docker": ("docker", "container", "image", "контейнер"),
    "db": ("sql", "database", "schema", "migration", "баз", "схем"),
}


@dataclass(frozen=True)
class RouteMatch:
    agent: str
    keyword: str

    def as_dict(self) -> dict:
        return {"agent": self.agent, "keyword": self.keyword}


@dataclass(frozen=True)
class RoutingResult:
    agents: list[str]
    confidence: float
    reason: str
    matches: list[RouteMatch]
    required_tools: list[ToolSpec]
    policy_decisions: list[dict]

    @property
    def intent(self) -> str:
        return "multi" if len(self.agents) > 1 else self.agents[0]

    def as_dict(self) -> dict:
        return {
            "intent": self.intent,
            "agents": self.agents,
            "confidence": self.confidence,
            "reason": self.reason,
            "matches": [match.as_dict() for match in self.matches],
            "required_tools": [tool.as_dict() for tool in self.required_tools],
            "policy_decisions": self.policy_decisions,
        }


def _keyword_matches(message: str) -> list[RouteMatch]:
    lowered = message.lower()
    matches = []
    for name, keywords in ROUTE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in ("doc", "docs"):
                if re.search(r"\bdocs?\b", lowered):
                    matches.append(RouteMatch(name, keyword))
                    break
            elif keyword in lowered:
                matches.append(RouteMatch(name, keyword))
                break
    return matches


def route_request(message: str, policy: ToolPolicy | None = None) -> RoutingResult:
    policy = policy or current_policy()
    matches = _keyword_matches(message)
    agents = [match.agent for match in matches] or ["orchestrator"]
    confidence = min(0.95, 0.45 + (0.2 * len(matches))) if matches else 0.35
    if matches:
        match_text = ", ".join(f"{match.agent}:{match.keyword}" for match in matches)
        reason = f"keyword matches: {match_text}"
    else:
        reason = "no specific keyword matched; using orchestrator"

    required_tools = []
    if agents == ["orchestrator"]:
        for agent in AGNO_MEMBER_NAMES:
            required_tools.extend(tools_for_agent(agent))
    else:
        for agent in agents:
            required_tools.extend(tools_for_agent(agent))

    return RoutingResult(
        agents=agents,
        confidence=confidence,
        reason=reason,
        matches=matches,
        required_tools=required_tools,
        policy_decisions=[policy.evaluate(tool).as_dict() for tool in required_tools],
    )


def should_use_local_system_status(message: str) -> bool:
    lowered = message.lower()
    return any(keyword in lowered for keyword in SYSTEM_KEYWORDS)
