"""Build langchain-mcp-adapters connection dict for a subset of servers."""

from __future__ import annotations

import os
from pathlib import Path

from agentic_rag.mcp_registry import McpServerSpec, get_spec


def build_stdio_connections(
    server_ids: list[str],
    *,
    python_exe: str | Path,
    base_env: dict[str, str] | None = None,
) -> dict[str, dict]:
    """
    Connection configs for ``MultiServerMCPClient`` (stdio transport only).

    Only includes ``server_ids`` — dynamic subset per prompt.
    """
    env_base = {**os.environ, **(base_env or {})}
    out: dict[str, dict] = {}
    py = str(python_exe)
    for sid in server_ids:
        spec = get_spec(sid)
        if not spec.script.is_file():
            raise FileNotFoundError(f"MCP server script missing for {sid}: {spec.script}")
        env = {**env_base, **spec.extra_env}
        out[sid] = {
            "transport": "stdio",
            "command": py,
            "args": [str(spec.script.resolve())],
            "env": env,
        }
    return out


def connected_backends_blurb(server_ids: list[str]) -> str:
    parts = [f"{get_spec(sid).title} ({sid})" for sid in server_ids]
    return "Connected MCP backends for this turn: " + ", ".join(parts) + "."
