from __future__ import annotations

import json

from orchestrator.run_history import RunHistoryStore


def build_run_trace(store: RunHistoryStore, run_id: str) -> dict[str, object] | None:
    run = store.get_run(run_id)
    if run is None:
        return None
    events = store.list_events(run_id)
    return {
        "run": run,
        "events": events,
        "timeline": [_timeline_item(event) for event in events],
        "markdown": render_trace_markdown(run, events),
    }


def render_trace_markdown(run: dict, events: list[dict]) -> str:
    lines = [
        f"# Run Trace: {run['id']}",
        "",
        "## Summary",
        f"- Status: {run['status']}",
        f"- Intent: {run['intent']}",
        f"- Agents: {', '.join(run.get('agents', [])) or 'none'}",
        f"- Created: {run['created_at']}",
        f"- Updated: {run['updated_at']}",
        "",
        "## Message",
        _code_block(run["message"]),
    ]
    if run.get("answer"):
        lines.extend(["", "## Final Answer", _code_block(run["answer"])])

    route = run.get("route") or {}
    if route:
        lines.extend(
            [
                "",
                "## Route",
                f"- Confidence: {route.get('confidence', 'unknown')}",
                f"- Reason: {route.get('reason', 'unknown')}",
            ]
        )

    decisions = [
        event["data"] for event in events if event["event"] == "policy_decision"
    ]
    if decisions:
        lines.extend(["", "## Policy Decisions"])
        for decision in decisions:
            status = "allowed" if decision.get("allowed") else "denied"
            lines.append(
                f"- {decision.get('tool_id', 'tool')}: {status}, "
                f"{decision.get('risk', 'unknown')} risk, {decision.get('reason', '')}"
            )

    lines.extend(["", "## Timeline"])
    if not events:
        lines.append("- No events recorded.")
    for event in events:
        lines.append(
            f"- {event['created_at']} `{event['event']}`: "
            f"{_compact_json(event.get('data', {}))}"
        )
    return "\n".join(lines).strip()


def _timeline_item(event: dict) -> dict[str, object]:
    data = event.get("data", {})
    return {
        "id": event["id"],
        "created_at": event["created_at"],
        "event": event["event"],
        "title": _event_title(event["event"], data),
        "data": data,
    }


def _event_title(event: str, data: dict) -> str:
    if event == "route":
        return f"Route to {', '.join(data.get('agents', [])) or 'orchestrator'}"
    if event == "policy_decision":
        state = "allow" if data.get("allowed") else "deny"
        return f"{state} {data.get('tool_id', 'tool')}"
    if event == "runner_error":
        return f"{data.get('runner', 'runner')} failed"
    if event == "done":
        return f"Completed as {data.get('status', 'completed')}"
    return event.replace("_", " ")


def _compact_json(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))[:500]


def _code_block(text: str) -> str:
    return f"```text\n{text}\n```"
