"""AI SDK v5 UI-message-stream protocol framing.

This translates the orchestrator's native SSE vocabulary (route/classify/
policy_decision/runner_*/tool_*/done and approval_required) into the Vercel AI
SDK v5 UI-message-stream protocol consumed by the shadcn + AI SDK frontend at
``/chat/stream``.

Protocol (verified against https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol and
https://ai-sdk.dev/docs/ai-sdk-ui/streaming-data):

- Response carries header ``x-vercel-ai-ui-message-stream: v1`` and
  ``content-type: text/event-stream``.
- Each frame is ``data: {json}\\n\\n``; the stream ends with ``data: [DONE]\\n\\n``.
- Message framing: ``{"type":"start","messageId":...}`` ... ``{"type":"finish"}``.
- Text: ``text-start`` / ``text-delta`` / ``text-end`` sharing one ``id``.
- Errors: ``{"type":"error","errorText":...}``.
- Custom data parts: TRANSIENT ``data-trace`` (UI telemetry, not persisted) for
  governance events; PERSISTENT ``data-approval`` (stable ``id`` = approval id)
  for ``approval_required`` so it survives a reconnect.
"""

from __future__ import annotations

import json

UI_MESSAGE_STREAM_HEADER = "x-vercel-ai-ui-message-stream"
UI_MESSAGE_STREAM_VERSION = "v1"

# Governance events that map to TRANSIENT data-trace parts (UI telemetry only).
TRACE_EVENTS = frozenset(
    {
        "route",
        "classify",
        "policy_decision",
        "agent_start",
        "runner_start",
        "runner_result",
        "runner_error",
        "tool_start",
        "tool_result",
        "tool_gated",
        "done",
    }
)


def _frame(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def start_frame(message_id: str) -> str:
    return _frame({"type": "start", "messageId": message_id})


def finish_frame() -> str:
    return _frame({"type": "finish"})


def text_start_frame(text_id: str) -> str:
    return _frame({"type": "text-start", "id": text_id})


def text_delta_frame(text_id: str, delta: str) -> str:
    return _frame({"type": "text-delta", "id": text_id, "delta": delta})


def text_end_frame(text_id: str) -> str:
    return _frame({"type": "text-end", "id": text_id})


def error_frame(error_text: str) -> str:
    return _frame({"type": "error", "errorText": error_text})


def trace_frame(event: str, data: dict) -> str:
    """A TRANSIENT data part carrying one governance event for UI telemetry."""
    return _frame(
        {"type": "data-trace", "data": {"event": event, **data}, "transient": True}
    )


def approval_frame(approval_id: str, data: dict) -> str:
    """A PERSISTENT data part for an approval, keyed by the stable approval id."""
    return _frame({"type": "data-approval", "id": approval_id, "data": data})


DONE_FRAME = "data: [DONE]\n\n"
