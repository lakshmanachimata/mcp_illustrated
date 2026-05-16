"""Catalog of stdio MCP servers in this repo (paths + routing hints)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class McpServerSpec:
    server_id: str
    script: Path
    title: str
    description: str
    keywords: tuple[str, ...] = ()
    tool_name_hints: tuple[str, ...] = ()
    extra_env: dict[str, str] = field(default_factory=dict)


def _vector_env() -> dict[str, str]:
    out: dict[str, str] = {}
    for key in (
        "VECTOR_MCP_DB",
        "VECTOR_MCP_EMBED_MODEL",
        "VECTOR_MCP_OLLAMA_URL",
        "VECTOR_MCP_STARTUP_REFRESH",
        "VECTOR_MCP_STARTUP_INDEX_STATS",
    ):
        val = os.environ.get(key, "").strip()
        if val:
            out[key] = val
    return out


MCP_SERVERS: dict[str, McpServerSpec] = {
    "leave": McpServerSpec(
        server_id="leave",
        script=REPO_ROOT / "mcp_server_0" / "leave_mcp_server.py",
        title="Employee leave",
        description=(
            "Leave records for fixed employees (Alice, Bob, Carol). "
            "Apply, list, approve, or revoke leaves."
        ),
        keywords=(
            "leave",
            "leaves",
            "vacation",
            "pto",
            "time off",
            "sick",
            "alice",
            "bob",
            "carol",
            "emp001",
            "emp002",
            "emp003",
            "approve leave",
            "revoke leave",
            "employee leave",
        ),
        tool_name_hints=(
            "get_leaves",
            "show_leaves",
            "apply_leave",
            "revoke_leave",
            "approve_leave",
        ),
    ),
    "users_tasks": McpServerSpec(
        server_id="users_tasks",
        script=REPO_ROOT / "mcp_server_1" / "server.py",
        title="Users and tasks",
        description="SQLite CRUD for users and tasks (list, create, update, filter by user).",
        keywords=(
            "user",
            "users",
            "task",
            "tasks",
            "todo",
            "database",
            "sqlite",
            "crud",
            "pending",
            "status",
            "user_id",
            "list_users",
            "create_user",
            "update_task",
        ),
        tool_name_hints=(
            "list_users",
            "create_user",
            "get_user",
            "list_tasks",
            "create_task",
            "get_tasks_for_users",
            "update_user",
            "update_task",
        ),
    ),
    "vector": McpServerSpec(
        server_id="vector",
        script=REPO_ROOT / "mcp_server_2" / "server.py",
        title="Vector index (RAG)",
        description=(
            "PDF/text folder embeddings via Ollama; semantic search over indexed chunks. "
            "Tools: refresh_vector_db, query_vectors, vector_index_stats, index_folder."
        ),
        keywords=(
            "vector",
            "embedding",
            "embed",
            "semantic",
            "rag",
            "pdf",
            "pdfs",
            "document",
            "documents",
            "chunk",
            "similar",
            "search index",
            "query_vectors",
            "vector_index",
            "refresh_vector",
            "index_folder",
            "invoice",
            "payslip",
            "deed",
        ),
        tool_name_hints=(
            "refresh_vector_db",
            "query_vectors",
            "vector_index_stats",
            "index_folder",
            "clear_vector_index",
        ),
        extra_env=_vector_env(),
    ),
}


def all_server_ids() -> list[str]:
    return list(MCP_SERVERS.keys())


def get_spec(server_id: str) -> McpServerSpec:
    if server_id not in MCP_SERVERS:
        raise KeyError(f"Unknown MCP server id: {server_id!r}; known: {all_server_ids()}")
    return MCP_SERVERS[server_id]


def routing_catalog_text() -> str:
    lines: list[str] = []
    for spec in MCP_SERVERS.values():
        kw = ", ".join(spec.keywords[:12])
        if len(spec.keywords) > 12:
            kw += ", ..."
        lines.append(f"- {spec.server_id}: {spec.description} (keywords: {kw})")
    return "\n".join(lines)
