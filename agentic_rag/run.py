"""LangGraph ReAct agent over dynamically selected MCP servers."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from agentic_rag.mcp_connections import build_stdio_connections, connected_backends_blurb
from agentic_rag.mcp_registry import get_spec
from agentic_rag.router import RouterMode, route_servers_async

SYSTEM_BASE = """You are a helpful assistant with MCP tools from backend services in this project.
Use only tools to fetch or change data; do not invent records.
After tool results, answer clearly in plain English.
If a tool errors, explain what failed and what the user can try next."""


async def run_agent_query(
    prompt: str,
    *,
    server_ids: list[str] | None = None,
    router_mode: RouterMode = "hybrid",
    model: str | None = None,
    ollama_base_url: str | None = None,
    python_exe: Path | None = None,
    max_iterations: int = 16,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Route (optional), connect only selected MCP servers, run a LangGraph ReAct agent.

    Returns dict with keys: answer, selected_servers, messages (serialized summary).
    """
    llm_model = model or os.environ.get("OLLAMA_MODEL", "qwen2.5:latest")
    base_url = ollama_base_url or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    py = python_exe or Path(sys.executable)

    if server_ids is None:
        selected = await route_servers_async(
            prompt,
            mode=router_mode,
            llm_model=llm_model,
            ollama_base_url=base_url,
        )
    else:
        selected = list(server_ids)

    if verbose:
        print(f"[agentic_rag] selected MCP servers: {selected}", file=sys.stderr)

    connections = build_stdio_connections(selected, python_exe=py)
    client = MultiServerMCPClient(connections, tool_name_prefix=True)
    tools = await client.get_tools()

    if verbose:
        print(f"[agentic_rag] loaded {len(tools)} tools", file=sys.stderr)
        for t in tools:
            print(f"  - {t.name}", file=sys.stderr)

    llm = ChatOllama(model=llm_model, base_url=base_url, temperature=0)
    system = SystemMessage(
        content=SYSTEM_BASE + "\n\n" + connected_backends_blurb(selected)
    )
    agent = create_react_agent(llm, tools)

    result = await agent.ainvoke(
        {"messages": [system, HumanMessage(content=prompt)]},
        config={"recursion_limit": max_iterations},
    )

    messages = result.get("messages") or []
    answer = ""
    if messages:
        last = messages[-1]
        answer = getattr(last, "content", None) or str(last)

    return {
        "ok": True,
        "answer": answer,
        "selected_servers": selected,
        "tool_count": len(tools),
        "servers": [
            {
                "id": sid,
                "title": get_spec(sid).title,
                "script": str(get_spec(sid).script),
            }
            for sid in selected
        ],
    }
