"""
Explicit LangGraph workflow: route → load MCP tools → ReAct loop.

Use ``run.run_agent_query`` for the default path; this module exposes the same
stages as named graph nodes for extension (logging, human-in-the-loop, etc.).
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from agentic_rag.mcp_connections import build_stdio_connections, connected_backends_blurb
from agentic_rag.router import RouterMode, route_servers_async


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    prompt: str
    selected_servers: list[str]
    router_mode: str
    tools_ready: bool


async def build_routed_graph(
    *,
    llm: Any,
    tools: list[Any],
    selected_servers: list[str],
):
    """Compile a graph that assumes tools are already loaded for ``selected_servers``."""

    system_blurb = connected_backends_blurb(selected_servers)
    model = llm.bind_tools(tools)

    async def call_model(state: AgentState) -> dict[str, Any]:
        msgs = state["messages"]
        if not msgs or msgs[0].type != "system":
            from langchain_core.messages import SystemMessage

            msgs = [
                SystemMessage(
                    content=(
                        "You are a helpful assistant with MCP tools. "
                        "Use tools for data; do not invent facts.\n\n"
                        + system_blurb
                    )
                ),
                *msgs,
            ]
        response = await model.ainvoke(msgs)
        return {"messages": [response]}

    builder = StateGraph(AgentState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    return builder.compile()


async def select_servers_node(state: AgentState) -> dict[str, Any]:
    mode: RouterMode = state.get("router_mode", "hybrid")  # type: ignore[assignment]
    selected = await route_servers_async(state["prompt"], mode=mode)
    return {"selected_servers": selected, "tools_ready": False}
