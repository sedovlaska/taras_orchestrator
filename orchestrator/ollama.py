from __future__ import annotations

import json
import urllib.request

from shared.config import settings


def list_ollama_models() -> dict:
    """List locally available Ollama models via /api/tags.

    Degrades gracefully: if Ollama is unreachable, returns an empty model list
    with reachable=False plus the configured default model. Uses a short timeout
    so callers (e.g. diagnostics) never hang when Ollama is down.
    """
    url = f"{settings.ollama_host.rstrip('/')}/api/tags"
    request = urllib.request.Request(url, method="GET")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=5) as response:
            body = json.loads(response.read().decode("utf-8"))
        models = [model["name"] for model in body.get("models", []) if model.get("name")]
        return {"models": models, "default_model": settings.llm_model, "reachable": True}
    except Exception as exc:
        print(f"OLLAMA_TAGS_FAILED error={str(exc)!r}", flush=True)
        return {"models": [], "default_model": settings.llm_model, "reachable": False}
