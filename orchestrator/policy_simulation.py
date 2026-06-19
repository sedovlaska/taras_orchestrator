from __future__ import annotations

from dataclasses import dataclass, field

from .evals import EVAL_CASES
from .policy import ToolPolicy, current_policy
from .routing import route_request
from .run_history import RunHistoryStore
from .tool_registry import TOOLS, get_tool


@dataclass(frozen=True)
class PolicySimulationInput:
    id: str
    source: str
    message: str
    tool_ids: list[str]
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidatePolicyPatch:
    mode: str | None = None
    allowed_risks: set[str] | None = None
    allowed_tools: set[str] | None = None
    denied_tools: set[str] | None = None
    approval_required_risks: set[str] | None = None


def _normalise(values: set[str] | list[str] | tuple[str, ...] | None) -> set[str] | None:
    if values is None:
        return None
    return {value.strip().lower() for value in values if value.strip()}


def build_candidate_policy(
    patch: CandidatePolicyPatch,
    base_policy: ToolPolicy | None = None,
) -> ToolPolicy:
    """Build a candidate policy by applying a sparse patch to the active policy."""
    base_policy = base_policy or current_policy()
    return ToolPolicy(
        mode=(patch.mode or base_policy.mode).lower(),
        allowed_risks=_normalise(patch.allowed_risks) if patch.allowed_risks is not None else base_policy.allowed_risks,
        allowed_tools=_normalise(patch.allowed_tools) if patch.allowed_tools is not None else base_policy.allowed_tools,
        denied_tools=_normalise(patch.denied_tools) if patch.denied_tools is not None else base_policy.denied_tools,
        approval_required_risks=(
            _normalise(patch.approval_required_risks)
            if patch.approval_required_risks is not None
            else base_policy.approval_required_risks
        ),
        project_root=base_policy.project_root,
    )


def _append_unique(target: list[str], tool_id: object) -> None:
    if not isinstance(tool_id, str) or not tool_id:
        return
    if tool_id not in target:
        target.append(tool_id)


def _tool_ids_from_route(route: dict) -> list[str]:
    tool_ids: list[str] = []
    for tool in route.get("required_tools", []):
        if isinstance(tool, dict):
            _append_unique(tool_ids, tool.get("id"))
    for decision in route.get("policy_decisions", []):
        if isinstance(decision, dict):
            _append_unique(tool_ids, decision.get("tool_id"))
    return tool_ids


def _tool_ids_from_events(events: list[dict]) -> list[str]:
    tool_ids: list[str] = []
    for event in events:
        data = event.get("data")
        if isinstance(data, dict):
            _append_unique(tool_ids, data.get("tool_id"))
    return tool_ids


def history_policy_inputs(store: RunHistoryStore, limit: int = 50) -> list[PolicySimulationInput]:
    samples = []
    for run in store.list_runs(limit):
        run_id = run["id"]
        detail = store.get_run(run_id)
        if detail is None:
            continue
        tool_ids = _tool_ids_from_route(detail.get("route", {}))
        for tool_id in _tool_ids_from_events(store.list_events(run_id)):
            _append_unique(tool_ids, tool_id)
        samples.append(
            PolicySimulationInput(
                id=f"run:{run_id}",
                source="history",
                message=detail["message"],
                tool_ids=tool_ids,
                metadata={
                    "run_id": run_id,
                    "created_at": detail["created_at"],
                    "status": detail["status"],
                    "intent": detail["intent"],
                    "agents": detail["agents"],
                },
            )
        )
    return samples


def eval_policy_inputs(base_policy: ToolPolicy | None = None) -> list[PolicySimulationInput]:
    base_policy = base_policy or current_policy()
    samples = []
    for case in EVAL_CASES:
        route = route_request(case.message, policy=base_policy)
        samples.append(
            PolicySimulationInput(
                id=f"eval:{case.id}",
                source="eval",
                message=case.message,
                tool_ids=[tool.id for tool in route.required_tools],
                metadata={
                    "case_id": case.id,
                    "category": case.category,
                    "agents": route.agents,
                    "confidence": route.confidence,
                },
            )
        )
    return samples


def _input_sources(
    store: RunHistoryStore,
    source: str,
    history_limit: int,
    base_policy: ToolPolicy,
) -> list[PolicySimulationInput]:
    if source == "history":
        return history_policy_inputs(store, history_limit)
    if source == "evals":
        return eval_policy_inputs(base_policy)
    if source == "all":
        return history_policy_inputs(store, history_limit) + eval_policy_inputs(base_policy)
    raise ValueError("source must be one of: history, evals, all")


def _policy_snapshot(policy: ToolPolicy) -> dict[str, object]:
    return {
        "mode": policy.mode,
        "allowed_risks": sorted(policy.allowed_risks or set()),
        "allowed_tools": sorted(policy.allowed_tools or set()),
        "denied_tools": sorted(policy.denied_tools or set()),
        "approval_required_risks": sorted(policy.approval_required_risks or set()),
    }


def _tool_diff(tool_id: str, base_policy: ToolPolicy, candidate_policy: ToolPolicy) -> dict:
    tool = get_tool(tool_id)
    current = base_policy.evaluate(tool)
    candidate = candidate_policy.evaluate(tool)
    current_requires_approval = current.allowed and base_policy.requires_approval(tool)
    candidate_requires_approval = candidate.allowed and candidate_policy.requires_approval(tool)
    flip = None
    if current.allowed != candidate.allowed:
        flip = "allow_to_deny" if current.allowed else "deny_to_allow"
    approval_flip = None
    if current_requires_approval != candidate_requires_approval:
        approval_flip = (
            "approval_added" if candidate_requires_approval else "approval_removed"
        )
    return {
        "tool_id": tool_id,
        "tool": tool.as_dict(),
        "current": {
            **current.as_dict(),
            "requires_approval": current_requires_approval,
        },
        "candidate": {
            **candidate.as_dict(),
            "requires_approval": candidate_requires_approval,
        },
        "flip": flip,
        "approval_flip": approval_flip,
    }


def simulate_policy(
    store: RunHistoryStore,
    candidate_policy: ToolPolicy,
    *,
    source: str = "history",
    history_limit: int = 50,
    include_unchanged: bool = False,
    base_policy: ToolPolicy | None = None,
) -> dict[str, object]:
    base_policy = base_policy or current_policy()
    samples = _input_sources(store, source, history_limit, base_policy)
    results = []
    unknown_tool_ids: set[str] = set()
    total_tools = 0
    allow_to_deny = 0
    deny_to_allow = 0
    approval_added = 0
    approval_removed = 0

    for sample in samples:
        diffs = []
        for tool_id in sample.tool_ids:
            if tool_id not in TOOLS:
                unknown_tool_ids.add(tool_id)
                continue
            total_tools += 1
            diff = _tool_diff(tool_id, base_policy, candidate_policy)
            if diff["flip"] == "allow_to_deny":
                allow_to_deny += 1
            elif diff["flip"] == "deny_to_allow":
                deny_to_allow += 1
            if diff["approval_flip"] == "approval_added":
                approval_added += 1
            elif diff["approval_flip"] == "approval_removed":
                approval_removed += 1
            if include_unchanged or diff["flip"] is not None or diff["approval_flip"] is not None:
                diffs.append(diff)
        if include_unchanged or diffs:
            results.append(
                {
                    "id": sample.id,
                    "source": sample.source,
                    "message": sample.message,
                    "metadata": sample.metadata,
                    "tool_diffs": diffs,
                }
            )

    return {
        "source": source,
        "history_limit": history_limit,
        "current_policy": _policy_snapshot(base_policy),
        "candidate_policy": _policy_snapshot(candidate_policy),
        "summary": {
            "inputs": len(samples),
            "inputs_with_changes": len(results),
            "tools_evaluated": total_tools,
            "allow_to_deny": allow_to_deny,
            "deny_to_allow": deny_to_allow,
            "approval_added": approval_added,
            "approval_removed": approval_removed,
            "unknown_tools": len(unknown_tool_ids),
        },
        "unknown_tool_ids": sorted(unknown_tool_ids),
        "results": results,
    }
