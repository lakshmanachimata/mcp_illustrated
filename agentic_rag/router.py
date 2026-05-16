"""Select which MCP servers to spawn based on the user prompt."""

from __future__ import annotations

import json
import re
from typing import Literal

from agentic_rag.mcp_registry import MCP_SERVERS, all_server_ids, routing_catalog_text

RouterMode = Literal["keyword", "llm", "hybrid", "all"]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def route_servers_keyword(prompt: str, *, min_score: int = 1) -> list[str]:
    """Score each server by keyword / tool-hint overlap; return ids with score >= min_score."""
    text = _normalize(prompt)
    if not text:
        return []

    scores: dict[str, int] = {sid: 0 for sid in MCP_SERVERS}
    for sid, spec in MCP_SERVERS.items():
        for kw in spec.keywords:
            if kw in text:
                scores[sid] += 2 if " " in kw else 1
        for hint in spec.tool_name_hints:
            if hint.replace("_", " ") in text or hint in text:
                scores[sid] += 2

    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    picked = [sid for sid, sc in ranked if sc >= min_score]
    return picked


async def route_servers_llm(
    prompt: str,
    *,
    model: str,
    ollama_base_url: str,
) -> list[str]:
    """Ask Ollama (JSON) which server ids apply."""
    import httpx

    system = (
        "You route user questions to MCP backend servers. "
        "Reply with JSON only: {\"servers\": [\"leave\"|\"users_tasks\"|\"vector\", ...]}. "
        "Pick the smallest set that can answer the question. "
        "Use multiple only if the question clearly needs them.\n\n"
        f"Available backends:\n{routing_catalog_text()}"
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "format": "json",
    }
    base = ollama_base_url.rstrip("/")
    async with httpx.AsyncClient() as client:
        r = await client.post(f"{base}/api/chat", json=payload, timeout=120.0)
        r.raise_for_status()
        content = (r.json().get("message") or {}).get("content") or "{}"

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = {}

    raw = data.get("servers") or data.get("server_ids") or []
    if not isinstance(raw, list):
        raw = []
    valid = {sid for sid in raw if sid in MCP_SERVERS}
    return sorted(valid)


def route_servers(
    prompt: str,
    *,
    mode: RouterMode = "hybrid",
    llm_model: str | None = None,
    ollama_base_url: str = "http://127.0.0.1:11434",
) -> list[str]:
    """Synchronous keyword/all routing (use ``route_servers_async`` for LLM modes)."""
    if mode == "all":
        return all_server_ids()
    if mode == "keyword":
        picked = route_servers_keyword(prompt)
        return picked if picked else all_server_ids()
    raise ValueError("Use route_servers_async for mode llm or hybrid")


async def route_servers_async(
    prompt: str,
    *,
    mode: RouterMode = "hybrid",
    llm_model: str = "qwen2.5:latest",
    ollama_base_url: str = "http://127.0.0.1:11434",
) -> list[str]:
    if mode == "all":
        return all_server_ids()

    keyword_pick = route_servers_keyword(prompt)

    if mode == "keyword":
        return keyword_pick if keyword_pick else all_server_ids()

    if mode == "llm":
        llm_pick = await route_servers_llm(prompt, model=llm_model, ollama_base_url=ollama_base_url)
        return llm_pick if llm_pick else all_server_ids()

    # hybrid: keyword when confident; else LLM; fallback all
    if len(keyword_pick) == 1:
        return keyword_pick
    if len(keyword_pick) >= 2:
        return keyword_pick

    llm_pick = await route_servers_llm(prompt, model=llm_model, ollama_base_url=ollama_base_url)
    return llm_pick if llm_pick else all_server_ids()
