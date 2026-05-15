"""
MCP client: drives the local users/tasks MCP server (``server.py``) over stdio and uses Ollama
(e.g. qwen2.5:latest) for natural-language questions about users and tasks in SQLite.

Requires: Ollama running locally (default http://127.0.0.1:11434) with a tool-capable model.

Examples:
  .venv/bin/python db_mcp_ollama_client.py
  .venv/bin/python db_mcp_ollama_client.py -q "list all users"
  .venv/bin/python db_mcp_ollama_client.py --demo
  OLLAMA_MODEL=qwen2.5:latest .venv/bin/python db_mcp_ollama_client.py -q "add user Jane age 28 gender f"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, TextContent

try:
    import readline  # noqa: F401 — line editing in interactive mode
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent
DEFAULT_SERVER = ROOT / "server.py"

SYSTEM_PROMPT = """You are an assistant for a users-and-tasks database accessed only via the provided MCP tools.

Data model:
- Users have integer id, name (text), age (integer), gender (text).
- Tasks belong to a user (user_id). Tasks have integer id, name, description (text), status (text, e.g. pending, open, done).

Rules:
- Always use tools to read or change data; do not invent users or tasks.
- Tool responses are JSON strings; parse them to know ids and errors.
- For get_tasks_for_users, pass user_ids as a comma-separated list of integer user ids (e.g. "1,2"). Pass status as empty string to include all statuses, or a specific status to filter.
- For list_tasks, omit user_id only when listing all tasks; pass user_id as an integer to filter.
- For update_user and update_task, omit optional fields you are not changing (do not send null unless the schema allows it).

After tools return, answer concisely in plain English."""


def mcp_tools_to_ollama(mcp_tools: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for t in mcp_tools:
        out.append(
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": (t.description or "").strip() or f"MCP tool {t.name}",
                    "parameters": t.inputSchema,
                },
            }
        )
    return out


def call_tool_result_to_text(result: CallToolResult) -> str:
    parts: list[str] = []
    for block in result.content:
        if isinstance(block, TextContent):
            parts.append(block.text)
        else:
            parts.append(block.model_dump_json())
    if result.isError:
        return "TOOL_ERROR:\n" + "\n".join(parts)
    return "\n".join(parts) if parts else "(empty tool result)"


def parse_tool_arguments(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return {}
        return json.loads(raw)
    raise TypeError(f"Unexpected tool arguments type: {type(raw)}")


async def ollama_chat(
    client: httpx.AsyncClient,
    ollama_url: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    base = ollama_url.rstrip("/")
    payload: dict[str, Any] = {"model": model, "messages": messages, "stream": False}
    if tools is not None:
        payload["tools"] = tools
    r = await client.post(f"{base}/api/chat", json=payload, timeout=300.0)
    r.raise_for_status()
    return r.json()


async def answer_with_tools(
    session: ClientSession,
    http: httpx.AsyncClient,
    *,
    ollama_url: str,
    model: str,
    ollama_tools: list[dict[str, Any]],
    query: str,
    max_rounds: int,
    verbose: bool,
) -> str:
    """One user question: system + query, then Ollama tool loop until a text reply."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    for round_i in range(max_rounds):
        data = await ollama_chat(http, ollama_url, model, messages, ollama_tools)
        msg = data.get("message") or {}
        if verbose:
            print(f"[ollama round {round_i + 1}]", file=sys.stderr)

        messages.append(msg)

        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            return (msg.get("content") or "").strip() or "(no text content from model)"

        for tc in tool_calls:
            fn = tc.get("function") or {}
            name = fn.get("name")
            if not name:
                continue
            args = parse_tool_arguments(fn.get("arguments"))
            if verbose:
                print(f"[mcp] tools/call {name} {args}", file=sys.stderr)
            result = await session.call_tool(name, args)
            text = call_tool_result_to_text(result)
            if verbose:
                print(f"[mcp] tools/call {name} result: {text[:500]}", file=sys.stderr)
            messages.append({"role": "tool", "tool_name": name, "content": text})

    return f"(stopped after {max_rounds} tool rounds; last assistant message may be incomplete)"


async def _input_line(prompt: str) -> str | None:
    def _read() -> str | None:
        try:
            return input(prompt)
        except EOFError:
            return None

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _read)


async def run_interactive(
    *,
    ollama_url: str,
    model: str,
    python_exe: Path,
    server_script: Path,
    max_rounds: int,
    verbose: bool,
) -> None:
    env = {**os.environ}
    params = StdioServerParameters(
        command=str(python_exe),
        args=[str(server_script)],
        env=env,
    )
    print(
        "Users/Tasks MCP + Ollama — type questions, Enter to send. "
        "Commands: quit | exit | q  (or Ctrl-D) to stop.\n",
        flush=True,
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools()
            ollama_tools = mcp_tools_to_ollama(listed.tools)
            async with httpx.AsyncClient() as http:
                while True:
                    line = await _input_line("> ")
                    if line is None:
                        print("\n(exit)", flush=True)
                        break
                    query = line.strip()
                    if not query:
                        continue
                    if query.lower() in ("quit", "exit", "q"):
                        break
                    try:
                        out = await answer_with_tools(
                            session,
                            http,
                            ollama_url=ollama_url,
                            model=model,
                            ollama_tools=ollama_tools,
                            query=query,
                            max_rounds=max_rounds,
                            verbose=verbose,
                        )
                    except (httpx.HTTPError, OSError, RuntimeError) as exc:
                        out = f"(error: {exc})"
                    print(out, "\n", flush=True)


async def run_query(
    query: str,
    *,
    ollama_url: str,
    model: str,
    python_exe: Path,
    server_script: Path,
    max_rounds: int,
    verbose: bool,
) -> str:
    env = {**os.environ}
    params = StdioServerParameters(
        command=str(python_exe),
        args=[str(server_script)],
        env=env,
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools()
            ollama_tools = mcp_tools_to_ollama(listed.tools)
            async with httpx.AsyncClient() as http:
                return await answer_with_tools(
                    session,
                    http,
                    ollama_url=ollama_url,
                    model=model,
                    ollama_tools=ollama_tools,
                    query=query,
                    max_rounds=max_rounds,
                    verbose=verbose,
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ollama + stdio MCP client for users/tasks server.py.")
    parser.add_argument(
        "-q",
        "--query",
        help="Natural-language question (e.g. 'list all users', 'show tasks for user 1').",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a few sample queries end-to-end (requires Ollama).",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OLLAMA_MODEL", "qwen2.5:latest"),
        help="Ollama model name (default: qwen2.5:latest or OLLAMA_MODEL).",
    )
    parser.add_argument(
        "--ollama-url",
        default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
        help="Ollama base URL (default: OLLAMA_HOST or http://127.0.0.1:11434).",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python interpreter used to spawn server.py (default: current interpreter).",
    )
    parser.add_argument(
        "--server-script",
        type=Path,
        default=DEFAULT_SERVER,
        help="Path to server.py.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=12,
        help="Max Ollama↔tool iterations per query.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Log tool calls to stderr.")
    args = parser.parse_args()

    if not args.server_script.is_file():
        print(f"Server script not found: {args.server_script}", file=sys.stderr)
        sys.exit(1)

    if args.demo:
        queries = [
            "list all users",
            "list all tasks",
            "show me tasks for user ids 1 and 2 with status pending, or say if there are none",
        ]
        for q in queries:
            print(f"\n=== Q: {q}\n", flush=True)
            out = asyncio.run(
                run_query(
                    q,
                    ollama_url=args.ollama_url,
                    model=args.model,
                    python_exe=args.python,
                    server_script=args.server_script,
                    max_rounds=args.max_rounds,
                    verbose=args.verbose,
                )
            )
            print(out, flush=True)
        return

    if args.query:
        out = asyncio.run(
            run_query(
                args.query,
                ollama_url=args.ollama_url,
                model=args.model,
                python_exe=args.python,
                server_script=args.server_script,
                max_rounds=args.max_rounds,
                verbose=args.verbose,
            )
        )
        print(out)
        return

    asyncio.run(
        run_interactive(
            ollama_url=args.ollama_url,
            model=args.model,
            python_exe=args.python,
            server_script=args.server_script,
            max_rounds=args.max_rounds,
            verbose=args.verbose,
        )
    )


if __name__ == "__main__":
    main()
