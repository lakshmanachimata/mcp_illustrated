"""Call MCP server (mcp_server_1) tools. Agent parses user prompt and invokes tools; no regex routing here."""
import asyncio
import json

from config import MCP_SERVER_1_URL


def _call_mcp_sync(instruction: str) -> dict:
    """Run MCP client in anyio (blocking). Used from async FastAPI via run_in_executor."""
    try:
        import anyio
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client
    except ImportError:
        return {"success": False, "error": "MCP client not installed (pip install mcp[cli])"}

    async def _do():
        async with streamable_http_client(MCP_SERVER_1_URL) as (read_stream, write_stream, _):
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
