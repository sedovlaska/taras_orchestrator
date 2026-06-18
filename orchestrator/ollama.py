from __future__ import annotations

import json
import urllib.request

from shared.config import settings


def list_ollama_models() -> dict:
    """List available models for the active provider.

    For the OpenAI-compatible provider this does NOT call Ollama: it returns the
    configured model and reports ``reachable`` based on whether a base URL and
    API key are configured (a cheap, network-free auth check). For the Ollama
    provider it queries ``/api/tags`` and degrades gracefully (reachable=False
    with an empty model list) if Ollama is unreachable. Keeps the same response
    shape (``models``, ``default_model``, ``reachable``) for all callers.
    """
    if settings.llm_provider == "openai":
        reachable = bool(settings.openai_base_url and settings.openai_api_key)
        return {
            "models": [settings.llm_model] if settings.llm_model else [],
            "default_model": settings.llm_model,
            "reachable": reachable,
        }
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
