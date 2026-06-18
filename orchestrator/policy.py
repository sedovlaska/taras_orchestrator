from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from shared.config import settings

from .tool_registry import ToolRisk, ToolSpec, get_tool


PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Env var carrying the comma-separated tool ids that an operator has explicitly
# approved for the current run. It is the only channel the in-subprocess tool
# wrappers have to learn which approvals were actually granted, so the runtime
# gate (ToolPolicy.require) can fail closed on anything that is not listed.
APPROVAL_GRANTS_ENV = "ORCHESTRATOR_APPROVED_TOOLS"


class ToolApprovalRequired(PermissionError):
    """Raised when a tool that needs approval is invoked without a grant."""


def _csv(value: str) -> set[str]:
    return {item.strip().lower() for item in value.split(",") if item.strip()}


def granted_tool_ids() -> set[str]:
    return _csv(os.environ.get(APPROVAL_GRANTS_ENV, ""))


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
    approval_required_risks: set[str] | None = None
    project_root: Path = PROJECT_ROOT

    @classmethod
    def from_settings(cls) -> "ToolPolicy":
        return cls(
            mode=settings.tool_policy_mode.lower(),
            allowed_risks=_csv(settings.tool_allowed_risks),
            allowed_tools=_csv(settings.tool_allowed),
            denied_tools=_csv(settings.tool_denied),
            approval_required_risks=_csv(settings.tool_approval_required_risks),
        )

    def requires_approval(self, tool: ToolSpec) -> bool:
        """Whether this tool's risk tier must be approved before it may run."""
        if self.mode == "off":
            return False
        required = self.approval_required_risks or set()
        return tool.risk.value in required

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
        """Runtime gate at the point a tool actually executes.

        Fails closed: a disallowed tool is denied, and a tool whose risk tier
        requires approval is blocked unless an explicit grant for this exact
        tool id is present (passed in via ``APPROVAL_GRANTS_ENV``). This is the
        backstop that closes the keyword-prediction bypass — even if the router
        never predicted the tool, it cannot run without a real approval.
        """
        tool = get_tool(tool_id)
        decision = self.evaluate(tool)
        if not decision.allowed:
            raise ToolApprovalRequired(
                f"{decision.tool_id} denied by tool policy: {decision.reason}"
            )
        if self.requires_approval(tool) and tool_id.lower() not in granted_tool_ids():
            raise ToolApprovalRequired(
                f"blocked: approval required for {tool_id} "
                f"(risk '{tool.risk.value}') and no approval was granted"
            )

    def require_project_path(self, raw_path: str) -> Path:
        candidate = (self.project_root / raw_path).resolve()
        if candidate != self.project_root and self.project_root not in candidate.parents:
            raise PermissionError(f"path escapes project root: {raw_path}")
        return candidate


def current_policy() -> ToolPolicy:
    return ToolPolicy.from_settings()
