"""Call MCP server (mcp_server_1) tools. Used when user asks to add/store data in the DB."""
import asyncio
import json
import re

from config import MCP_SERVER_URL


# Phrases that indicate the user wants to add/store something in the MCP server DB
ADD_TO_DB_PATTERNS = [
    r"add\s+(?:these?\s+)?(?:survey\s+)?questions?\s*(?::|to\s+(?:the\s+)?(?:db|database))?",
    r"add\s+(?:to\s+(?:the\s+)?(?:db|database)|in\s+(?:mcp\s+)?(?:server\s+)?db)",
    r"store\s+(?:these?\s+)?(?:survey\s+)?questions?\s*(?::|in\s+(?:the\s+)?(?:db|database))?",
    r"store\s+(?:in\s+(?:the\s+)?(?:db|database)|in\s+mcp)",
    r"save\s+(?:these?\s+)?(?:survey\s+)?questions?\s*(?::|to\s+(?:the\s+)?(?:db|database))?",
    r"save\s+(?:to\s+(?:the\s+)?(?:db|database)|in\s+mcp)",
    r"add\s+to\s+mcp_server_1",
    r"add\s+to\s+mcp\s+server\s+db",
]


def should_use_mcp_db(prompt: str) -> bool:
    """True if the prompt asks to add/store/save something in the database (use MCP server)."""
    if not prompt or not prompt.strip():
        return False
    lower = prompt.strip().lower()
    for pat in ADD_TO_DB_PATTERNS:
        if re.search(pat, lower, re.IGNORECASE):
            return True
    return False


def _call_mcp_sync(instruction: str) -> dict:
    """Run MCP client in anyio (blocking). Used from async FastAPI via run_in_executor."""
    try:
        import anyio
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client
    except ImportError:
        return {"success": False, "error": "MCP client not installed (pip install mcp[cli])"}

    async def _do():
        async with streamable_http_client(MCP_SERVER_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool("execute_instruction", arguments={"instruction": instruction})
                if not result or not getattr(result, "content", None):
                    return {"success": False, "error": "No response from MCP server"}
                text_parts = []
                for block in result.content:
                    if getattr(block, "type", None) == "text" and getattr(block, "text", None):
                        text_parts.append(block.text)
                text = "\n".join(text_parts)
                if not text:
                    return {"success": False, "error": "Empty tool response"}
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return {"success": True, "raw": text}

    return anyio.run(_do)


async def call_mcp_execute_instruction(instruction: str) -> dict:
    """Call MCP server's execute_instruction tool. Returns result dict or error."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _call_mcp_sync, instruction)
