from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import AsyncIterator


@dataclass
class StreamEvent:
    event: str  # classify | agent_start | agent_result | done | error
    data: dict = field(default_factory=dict)

    def to_sse(self) -> str:
        return f"event: {self.event}\ndata: {json.dumps(self.data, ensure_ascii=False)}\n\n"


class EventQueue:
    def __init__(self):
        self._queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()

    async def emit(self, event: StreamEvent):
        await self._queue.put(event)

    async def finish(self):
        await self._queue.put(None)

    async def stream(self) -> AsyncIterator[StreamEvent]:
        while True:
            item = await self._queue.get()
            if item is None:
                break
            yield item
