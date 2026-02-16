"""
LlamaIndex MCP-powered agent: discovers tools from MCP server on each query.
The agent parses the user prompt and decides which tools to invoke (no regex).
Stream events (ToolCall, ToolCallResult) are used to log MCP tool invocations.
"""
import logging

from config import MCP_SERVER_1_URL, SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def _tool_name_and_desc(tool) -> tuple[str, str]:
    """Get (name, description) from a LlamaIndex tool for logging."""
    name = getattr(tool, "metadata", None) and getattr(tool.metadata, "name", None)
    if not name:
        name = getattr(tool, "name", None) or str(type(tool).__name__)
    desc = getattr(tool, "metadata", None) and getattr(tool.metadata, "description", None)
    if not desc:
        desc = getattr(tool, "description", None) or ""
    return (str(name), str(desc)[:200] if desc else "")


async def get_mcp_tools():
    """Load tools from MCP server (streamable HTTP at MCP_SERVER_1_URL). Called on each agent query to reflect current MCP server capabilities."""
    try:
        from llama_index.tools.mcp import aget_tools_from_mcp_url
    except ImportError:
        raise RuntimeError(
            "llama-index-tools-mcp not installed. pip install llama-index-tools-mcp"
        )
    tools = await aget_tools_from_mcp_url(MCP_SERVER_1_URL)
    logger.info("Loaded %s MCP tools from %s", len(tools), MCP_SERVER_1_URL)
    # Log capabilities so we can confirm create_table etc. are visible to the agent
    names = []
    for t in tools:
        name, desc = _tool_name_and_desc(t)
        names.append(name)
        logger.info("MCP capability: %s — %s", name, desc or "(no description)")
    print(f"[LLM Service] MCP server capabilities ({len(tools)} tools): {', '.join(names)}", flush=True)
    return tools


def _create_agent(tools, llm, system_prompt: str | None = None):
    """Build ReAct agent with tools, LLM, and optional system prompt (tool calling / database)."""
    from llama_index.core.agent.workflow import ReActAgent
    kwargs = {"tools": tools, "llm": llm}
    if system_prompt is not None:
        kwargs["system_prompt"] = system_prompt
    return ReActAgent(**kwargs)


async def run_agent_query(
    message_content: str,
    model: str,
    system_prompt: str | None = None,
    verbose: bool = True,
) -> str:
    """
    Pass the user message to the agent. The agent parses intent and invokes MCP tools as needed.
    No regex or pattern matching: the LLM decides when to call tools.
    Stream events (ToolCall, ToolCallResult) are logged for visibility.
    """
    try:
        from llama_index.llms.ollama import Ollama
    except ImportError:
        raise RuntimeError(
            "llama-index-llms-ollama not installed. pip install llama-index-llms-ollama"
        )
    from llama_index.core.workflow import Context

    logger.info("run_agent_query message=%s model=%s", message_content[:150], model)
    print(f"[LLM Service] Agent message: {repr(message_content[:300])}", flush=True)

    tools = await get_mcp_tools()
    if not tools:
        return "No tools available from MCP server. Ensure the MCP server is running and exposes tools."

    llm = Ollama(
        model=model,
        request_timeout=360.0,
        context_window=8192,
    )
    prompt_to_use = system_prompt if system_prompt is not None else SYSTEM_PROMPT
    agent = _create_agent(tools, llm, system_prompt=prompt_to_use)
    ctx = Context(agent)

    # Invoke agent with user message (agent decides tool use)
    handler = agent.run(message_content, ctx=ctx)

    # Stream events: log ToolCall and ToolCallResult (MCP tools as they run)
    try:
        from llama_index.core.agent.workflow import ToolCall, ToolCallResult
    except ImportError:
        ToolCall = ToolCallResult = None

    if ToolCall is not None and ToolCallResult is not None:
        async for event in handler.stream_events():
            if verbose and type(event) == ToolCall:
                tool_name = getattr(event, "tool_name", None) or getattr(event, "name", "?")
                tool_kwargs = getattr(event, "tool_kwargs", None) or getattr(event, "args", {})
                logger.info("Agent calling tool: %s with %s", tool_name, tool_kwargs)
                print(f"[LLM Service] Calling tool: {tool_name}", flush=True)
            elif verbose and type(event) == ToolCallResult:
                tool_name = getattr(event, "tool_name", None) or getattr(event, "name", "?")
                tool_output = getattr(event, "tool_output", None) or getattr(event, "output", "")
                out_preview = str(tool_output)[:200] + ("..." if len(str(tool_output)) > 200 else "")
                logger.info("Tool %s returned: %s", tool_name, out_preview)
                print(f"[LLM Service] Tool {tool_name} returned: {out_preview}", flush=True)

    response = await handler
    return str(response)
