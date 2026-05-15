"""
Multi–MCP Ollama client: connects to two stdio MCP servers at once.

Default servers (repo layout):
  - mcp_server_0/leave_mcp_server.py — employee leave records (JSON store)
  - mcp_server_1/server.py — users & tasks (SQLite)

Requires: pip install mcp httpx (see project requirements or mcp_server_1/requirements.txt)
          Ollama running (default http://127.0.0.1:11434) with a tool-capable model.

Examples:
  python multi_mcp_ollama_client.py
  python multi_mcp_ollama_client.py -q "List all users, then list leaves for alice"
  OLLAMA_MODEL=qwen2.5:latest python multi_mcp_ollama_client.py --demo
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Awaitable, Callable
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
DEFAULT_LEAVE_SERVER = ROOT / "mcp_server_0" / "leave_mcp_server.py"
DEFAULT_USERS_TASKS_SERVER = ROOT / "mcp_server_1" / "server.py"

SYSTEM_PROMPT = """You are a single assistant with access to two separate systems via tools. Use only the tools; do not invent data.

## A) Employee leave (leave tools)
Three fixed employees: EMP001 Alice Johnson (Engineering), EMP002 Bob Smith (Product), EMP003 Carol Williams (Operations).
Users may say emp1, alice, EMP001, etc. Leave records use ISO dates YYYY-MM-DD.
Use the leave-* tools for applying, listing, revoking, or approving leaves.

## B) Users and tasks (users/tasks database tools)
Users have integer id, name, age, gender. Tasks belong to a user (user_id) with id, name, description, status (e.g. pending, open, done).
- get_tasks_for_users: pass user_ids as a comma-separated string of integer ids (e.g. "1,2"). Use status="" for all statuses, or a specific status to filter.
- list_tasks: omit user_id for all tasks; pass user_id to filter one user.
- update_user / update_task: omit optional fields you are not changing.

If a question applies to only one system, use only those tools. If it spans both, call tools from both as needed.
After tool results, answer concisely in plain English."""


def mcp_tools_to_ollama(mcp_tools: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for t in mcp_tools:
        out.append(
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": ((t.description or "").strip() or f"MCP tool {t.name}"),
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


def build_tool_router(
    *pairs: tuple[str, ClientSession, list[Any]],
) -> tuple[list[dict[str, Any]], Callable[[str, dict[str, Any]], Awaitable[str]]]:
    """
    pairs: (label, session, mcp_tools_list)
    Returns merged Ollama tool defs and async router(name, args) -> text.
    """
    merged_ollama_tools: list[dict[str, Any]] = []
    name_to_session: dict[str, ClientSession] = {}

    for _label, session, tools in pairs:
        for t in tools:
            if t.name in name_to_session:
                raise ValueError(
                    f"Duplicate MCP tool name across servers: {t.name!r}. "
                    "Rename tools on one server or add namespacing."
                )
            name_to_session[t.name] = session
        merged_ollama_tools.extend(mcp_tools_to_ollama(tools))

    async def call_tool(name: str, args: dict[str, Any]) -> str:
        sess = name_to_session.get(name)
        if sess is None:
            return f"TOOL_ERROR:\nUnknown tool {name!r}; known: {sorted(name_to_session)}"
        result = await sess.call_tool(name, args)
        return call_tool_result_to_text(result)

    return merged_ollama_tools, call_tool


async def answer_with_tools(
    call_tool: Callable[[str, dict[str, Any]], Awaitable[str]],
    http: httpx.AsyncClient,
    *,
    ollama_url: str,
    model: str,
    ollama_tools: list[dict[str, Any]],
    query: str,
    max_rounds: int,
    verbose: bool,
) -> str:
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
            text = await call_tool(name, args)
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


async def run_dual_sessions(
    *,
    ollama_url: str,
    model: str,
    python_exe: Path,
    leave_script: Path,
    users_tasks_script: Path,
    max_rounds: int,
    verbose: bool,
    query: str | None,
    interactive: bool,
    demo: bool,
) -> None:
    env = {**os.environ}
    params_leave = StdioServerParameters(
        command=str(python_exe),
        args=[str(leave_script)],
        env=env,
    )
    params_ut = StdioServerParameters(
        command=str(python_exe),
        args=[str(users_tasks_script)],
        env=env,
    )

    async with stdio_client(params_leave) as (read_l, write_l):
        async with ClientSession(read_l, write_l) as session_leave:
            await session_leave.initialize()
            listed_l = await session_leave.list_tools()
            async with stdio_client(params_ut) as (read_u, write_u):
                async with ClientSession(read_u, write_u) as session_ut:
                    await session_ut.initialize()
                    listed_u = await session_ut.list_tools()

                    ollama_tools, call_tool = build_tool_router(
                        ("leave", session_leave, listed_l.tools),
                        ("users_tasks", session_ut, listed_u.tools),
                    )

                    async with httpx.AsyncClient() as http:
                        if demo:
                            queries = [
                                "Using only user/task tools: list all users.",
                                "Using only leave tools: show human-readable leaves for alice.",
                                "Briefly summarize what both systems are for (no tool calls needed if you already know from context).",
                            ]
                            for q in queries:
                                print(f"\n=== Q: {q}\n", flush=True)
                                out = await answer_with_tools(
                                    call_tool,
                                    http,
                                    ollama_url=ollama_url,
                                    model=model,
                                    ollama_tools=ollama_tools,
                                    query=q,
                                    max_rounds=max_rounds,
                                    verbose=verbose,
                                )
                                print(out, flush=True)
                            return

                        if query is not None:
                            out = await answer_with_tools(
                                call_tool,
                                http,
                                ollama_url=ollama_url,
                                model=model,
                                ollama_tools=ollama_tools,
                                query=query,
                                max_rounds=max_rounds,
                                verbose=verbose,
                            )
                            print(out)
                            return

                        if interactive:
                            print(
                                "Multi MCP (leave + users/tasks) + Ollama — type a question. "
                                "Commands: quit | exit | q  (or Ctrl-D).\n",
                                flush=True,
                            )
                            while True:
                                line = await _input_line("> ")
                                if line is None:
                                    print("\n(exit)", flush=True)
                                    break
                                q = line.strip()
                                if not q:
                                    continue
                                if q.lower() in ("quit", "exit", "q"):
                                    break
                                try:
                                    out = await answer_with_tools(
                                        call_tool,
                                        http,
                                        ollama_url=ollama_url,
                                        model=model,
                                        ollama_tools=ollama_tools,
                                        query=q,
                                        max_rounds=max_rounds,
                                        verbose=verbose,
                                    )
                                except (httpx.HTTPError, OSError, RuntimeError, ValueError) as exc:
                                    out = f"(error: {exc})"
                                print(out, "\n", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ollama client for leave MCP + users/tasks MCP (two stdio servers)."
    )
    parser.add_argument(
        "-q",
        "--query",
        help="Single natural-language question (may use both backends).",
    )
    parser.add_argument("--demo", action="store_true", help="Run sample queries (needs Ollama).")
    parser.add_argument(
        "--model",
        default=os.environ.get("OLLAMA_MODEL", "qwen2.5:latest"),
        help="Ollama model (default: qwen2.5:latest or OLLAMA_MODEL).",
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
        help="Python used to spawn both MCP servers (default: current interpreter).",
    )
    parser.add_argument(
        "--leave-server",
        type=Path,
        default=DEFAULT_LEAVE_SERVER,
        help="Path to leave_mcp_server.py",
    )
    parser.add_argument(
        "--users-tasks-server",
        type=Path,
        default=DEFAULT_USERS_TASKS_SERVER,
        help="Path to mcp_server_1/server.py",
    )
    parser.add_argument("--max-rounds", type=int, default=16, help="Max tool rounds per question.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Log rounds and tool I/O to stderr.")
    args = parser.parse_args()

    for label, p in (
        ("leave server", args.leave_server),
        ("users/tasks server", args.users_tasks_server),
    ):
        if not p.is_file():
            print(f"{label} not found: {p}", file=sys.stderr)
            sys.exit(1)

    if args.demo:
        asyncio.run(
            run_dual_sessions(
                ollama_url=args.ollama_url,
                model=args.model,
                python_exe=args.python,
                leave_script=args.leave_server,
                users_tasks_script=args.users_tasks_server,
                max_rounds=args.max_rounds,
                verbose=args.verbose,
                query=None,
                interactive=False,
                demo=True,
            )
        )
        return

    if args.query:
        asyncio.run(
            run_dual_sessions(
                ollama_url=args.ollama_url,
                model=args.model,
                python_exe=args.python,
                leave_script=args.leave_server,
                users_tasks_script=args.users_tasks_server,
                max_rounds=args.max_rounds,
                verbose=args.verbose,
                query=args.query,
                interactive=False,
                demo=False,
            )
        )
        return

    asyncio.run(
        run_dual_sessions(
            ollama_url=args.ollama_url,
            model=args.model,
            python_exe=args.python,
            leave_script=args.leave_server,
            users_tasks_script=args.users_tasks_server,
            max_rounds=args.max_rounds,
            verbose=args.verbose,
            query=None,
            interactive=True,
            demo=False,
        )
    )


if __name__ == "__main__":
    main()
