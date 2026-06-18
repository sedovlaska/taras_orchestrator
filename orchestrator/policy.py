from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from shared.config import settings

from .tool_registry import ToolRisk, ToolSpec, get_tool


PROJECT_ROOT = Path(__file__).parent.parent.resolve()


def _csv(value: str) -> set[str]:
    return {item.strip().lower() for item in value.split(",") if item.strip()}


@dataclass(frozen=True)
class PolicyDecision:
    tool_id: str
    allowed: bool
    reason: str
    risk: str
    agent: str

    def as_dict(self) -> dict:
        return {
            "tool_id": self.tool_id,
            "allowed": self.allowed,
            "reason": self.reason,
            "risk": self.risk,
            "agent": self.agent,
        }


@dataclass(frozen=True)
class ToolPolicy:
    mode: str = "safe"
    allowed_risks: set[str] | None = None
    allowed_tools: set[str] | None = None
    denied_tools: set[str] | None = None
    project_root: Path = PROJECT_ROOT

    @classmethod
    def from_settings(cls) -> "ToolPolicy":
        return cls(
            mode=settings.tool_policy_mode.lower(),
            allowed_risks=_csv(settings.tool_allowed_risks),
            allowed_tools=_csv(settings.tool_allowed),
            denied_tools=_csv(settings.tool_denied),
        )

    def evaluate(self, tool: ToolSpec) -> PolicyDecision:
        allowed_tools = self.allowed_tools or set()
        denied_tools = self.denied_tools or set()
        allowed_risks = self.allowed_risks or {ToolRisk.LOW.value}

        if self.mode == "off":
            return PolicyDecision(tool.id, True, "policy disabled", tool.risk.value, tool.agent)
        if tool.id.lower() in denied_tools or tool.name.lower() in denied_tools:
            return PolicyDecision(tool.id, False, "tool explicitly denied", tool.risk.value, tool.agent)
        if tool.id.lower() in allowed_tools or tool.name.lower() in allowed_tools:
            return PolicyDecision(tool.id, True, "tool explicitly allowed", tool.risk.value, tool.agent)
        if tool.risk.value in allowed_risks:
            return PolicyDecision(tool.id, True, f"risk '{tool.risk.value}' allowed", tool.risk.value, tool.agent)
        return PolicyDecision(
            tool.id,
            False,
            f"risk '{tool.risk.value}' is not in TOOL_ALLOWED_RISKS",
            tool.risk.value,
            tool.agent,
        )

    def require(self, tool_id: str) -> None:
        decision = self.evaluate(get_tool(tool_id))
        if not decision.allowed:
            raise PermissionError(f"{decision.tool_id} denied by tool policy: {decision.reason}")

    def require_project_path(self, raw_path: str) -> Path:
        candidate = (self.project_root / raw_path).resolve()
        if candidate != self.project_root and self.project_root not in candidate.parents:
            raise PermissionError(f"path escapes project root: {raw_path}")
        return candidate


def current_policy() -> ToolPolicy:
    return ToolPolicy.from_settings()
