"""
Multi–MCP Ollama client: connects to three stdio MCP servers at once.

Default servers (repo layout):
  - mcp_server_0/leave_mcp_server.py — employee leave records (JSON store)
  - mcp_server_1/server.py — users & tasks (SQLite)
  - mcp_server_2/server.py — folder vector index (SQLite + Ollama embeddings)

Requires: pip install -r requirements-mcp-client.txt
          Ollama running (default http://127.0.0.1:11434) with a tool-capable chat model;
          for vector index/embeddings, pull an embed model on Ollama (e.g. ``embeddinggemma``).

Examples (vector data is never read here; MCP tools speak to ``mcp_server_2/server.py``, which owns the DB):
  python multi_mcp_ollama_client.py
  python multi_mcp_ollama_client.py -q "List users, show alice's leaves, and vector_index_stats"
  # Optional: choose where ``mcp_server_2`` stores SQLite (passed into that server subprocess only).
  VECTOR_MCP_DB=/path/to/vector_mcp.db python multi_mcp_ollama_client.py -q "vector_index_stats"
  python multi_mcp_ollama_client.py --vector-mcp-db mcp_server_2/vector_mcp.db -q "query_vectors about invoices"
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
DEFAULT_VECTOR_SERVER = ROOT / "mcp_server_2" / "server.py"

SYSTEM_PROMPT = """You are a single assistant with access to three separate systems via tools. Use only the tools; do not invent data.

## A) Employee leave (leave tools)
Three fixed employees: EMP001 Alice Johnson (Engineering), EMP002 Bob Smith (Product), EMP003 Carol Williams (Operations).
Users may say emp1, alice, EMP001, etc. Leave records use ISO dates YYYY-MM-DD.
Use the leave-* tools for applying, listing, revoking, or approving leaves.

## B) Users and tasks (users/tasks database tools)
Users have integer id, name, age, gender. Tasks belong to a user (user_id) with id, name, description, status (e.g. pending, open, done).
- get_tasks_for_users: pass user_ids as a comma-separated string of integer ids (e.g. "1,2"). Use status="" for all statuses, or a specific status to filter.
- list_tasks: omit user_id for all tasks; pass user_id to filter one user.
- update_user / update_task: omit optional fields you are not changing.

## C) Folder vector index (via **mcp_server_2** MCP tools only)
This assistant never opens the vector SQLite database file. **`mcp_server_2/server.py`** is spawned by the MCP host; it owns storage and embeddings. You only interact through named tools routed to that server (e.g. ``vector_index_stats``, ``refresh_vector_db``, ``query_vectors``).

On the server process only: SQLite path ``VECTOR_MCP_DB`` (default ``mcp_server_2/vector_mcp.db``), embed model ``VECTOR_MCP_EMBED_MODEL``, Ollama base ``VECTOR_MCP_OLLAMA_URL`` or ``OLLAMA_HOST``. Default PDF library folder: ``mcp_server_2/pdfs``.

- **No automatic rebuild on server start.** Call ``refresh_vector_db`` when the user wants to re-embed all PDFs under ``mcp_server_2/pdfs`` (can take minutes). Hosts may start the server with ``VECTOR_MCP_STARTUP_REFRESH=1`` to rebuild once at process start (avoid two server processes sharing one DB).
- **Optional startup stats:** ``VECTOR_MCP_STARTUP_INDEX_STATS=1`` on the server logs chunk/file counts at boot (otherwise it skips DB reads for fast start).
- **CLI bulk refresh (runs server code standalone, not this client):** ``python mcp_server_2/manual_refresh_pdfs.py`` with the server's env — not an MCP conversation.
- **Tools (all MCP calls into mcp_server_2):** ``refresh_vector_db``, ``index_folder``, ``query_vectors``, ``vector_index_stats`` (``chunk_count``, ``file_count``, ``source_paths``, ``sample_paths``), ``clear_vector_index``.

If a question applies to only one system, use only those tools. If it spans several, call tools from each as needed.
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


async def run_multi_sessions(
    *,
    ollama_url: str,
    model: str,
    python_exe: Path,
    leave_script: Path,
    users_tasks_script: Path,
    vector_script: Path,
    vector_env_extra: dict[str, str] | None,
    max_rounds: int,
    verbose: bool,
    query: str | None,
    interactive: bool,
    demo: bool,
) -> None:
    env = {**os.environ}
    # Spawn env for mcp_server_2 only; this script never imports vector_store or opens vector_mcp.db.
    env_vec = {**env, **(vector_env_extra or {})}
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
    params_vec = StdioServerParameters(
        command=str(python_exe),
        args=[str(vector_script)],
        env=env_vec,
    )

    async with stdio_client(params_leave) as (read_l, write_l):
        async with ClientSession(read_l, write_l) as session_leave:
            await session_leave.initialize()
            listed_l = await session_leave.list_tools()
            async with stdio_client(params_ut) as (read_u, write_u):
                async with ClientSession(read_u, write_u) as session_ut:
                    await session_ut.initialize()
                    listed_u = await session_ut.list_tools()
                    async with stdio_client(params_vec) as (read_v, write_v):
                        async with ClientSession(read_v, write_v) as session_vec:
                            await session_vec.initialize()
                            listed_v = await session_vec.list_tools()

                            ollama_tools, call_tool = build_tool_router(
                                ("leave", session_leave, listed_l.tools),
                                ("users_tasks", session_ut, listed_u.tools),
                                ("vector", session_vec, listed_v.tools),
                            )

                            async with httpx.AsyncClient() as http:
                                if demo:
                                    queries = [
                                        "Using only user/task tools: list all users.",
                                        "Using only leave tools: show human-readable leaves for alice.",
                                        "Using only vector tools: call vector_index_stats and report chunk_count, file_count, and source_paths length.",
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
                                        "Multi MCP (leave + users/tasks + vector) + Ollama — type a question. "
                                        "Commands: quit | exit | q  (or Ctrl-D).\n"
                                        "Vector index: MCP tools → mcp_server_2/server.py only (no direct SQLite from "
                                        "this client). Extra --vector-* flags set env for that spawned server.\n",
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
        description=(
            "Ollama client for leave + users/tasks + vector via **three stdio MCP servers** (no direct DB access)."
            " Vector index data is accessed only through MCP tools to mcp_server_2/server.py; optional --vector-* "
            "arguments set environment variables for that spawned server process."
        )
    )
    parser.add_argument(
        "-q",
        "--query",
        help="Single natural-language question (may use any of the three backends).",
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
        help="Python used to spawn all MCP servers (default: current interpreter).",
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
    parser.add_argument(
        "--vector-server",
        type=Path,
        default=DEFAULT_VECTOR_SERVER,
        help="Path to mcp_server_2/server.py",
    )
    parser.add_argument(
        "--vector-mcp-db",
        type=str,
        default="",
        metavar="PATH",
        help=(
            "Pass VECTOR_MCP_DB into the **mcp_server_2** child process only. This client never opens SQLite; "
            "the MCP server chooses the DB file."
        ),
    )
    parser.add_argument(
        "--vector-startup-refresh",
        action="store_true",
        help="Forwarded to **mcp_server_2** only: VECTOR_MCP_STARTUP_REFRESH=1 (server rebuilds pdf index once at startup).",
    )
    parser.add_argument(
        "--vector-startup-index-stats",
        action="store_true",
        help="Forwarded to **mcp_server_2** only: VECTOR_MCP_STARTUP_INDEX_STATS=1 (server logs chunk/file counts at boot).",
    )
    parser.add_argument(
        "--vector-ollama-url",
        type=str,
        default="",
        metavar="URL",
        help="Forwarded to **mcp_server_2** only: VECTOR_MCP_OLLAMA_URL (embeddings; this client talks to its own URL for chat).",
    )
    parser.add_argument(
        "--vector-embed-model",
        type=str,
        default="",
        metavar="NAME",
        help="Forwarded to **mcp_server_2** only: VECTOR_MCP_EMBED_MODEL (embed model name on Ollama).",
    )
    parser.add_argument("--max-rounds", type=int, default=16, help="Max tool rounds per question.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Log rounds and tool I/O to stderr.")
    args = parser.parse_args()

    if not args.leave_server.is_file():
        print(f"leave server not found: {args.leave_server}", file=sys.stderr)
        sys.exit(1)
    if not args.users_tasks_server.is_file():
        print(f"users/tasks server not found: {args.users_tasks_server}", file=sys.stderr)
        sys.exit(1)
    if not args.vector_server.is_file():
        print(f"vector server not found: {args.vector_server}", file=sys.stderr)
        sys.exit(1)

    vec_extra: dict[str, str] = {}
    if args.vector_mcp_db.strip():
        vec_extra["VECTOR_MCP_DB"] = str(Path(args.vector_mcp_db).expanduser().resolve())
    if args.vector_startup_refresh:
        vec_extra["VECTOR_MCP_STARTUP_REFRESH"] = "1"
    if args.vector_startup_index_stats:
        vec_extra["VECTOR_MCP_STARTUP_INDEX_STATS"] = "1"
    if args.vector_ollama_url.strip():
        vec_extra["VECTOR_MCP_OLLAMA_URL"] = args.vector_ollama_url.strip().rstrip("/")
    if args.vector_embed_model.strip():
        vec_extra["VECTOR_MCP_EMBED_MODEL"] = args.vector_embed_model.strip()
    vector_env_extra = vec_extra if vec_extra else None
    if args.demo:
        asyncio.run(
            run_multi_sessions(
                ollama_url=args.ollama_url,
                model=args.model,
                python_exe=args.python,
                leave_script=args.leave_server,
                users_tasks_script=args.users_tasks_server,
                vector_script=args.vector_server,
                vector_env_extra=vector_env_extra,
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
            run_multi_sessions(
                ollama_url=args.ollama_url,
                model=args.model,
                python_exe=args.python,
                leave_script=args.leave_server,
                users_tasks_script=args.users_tasks_server,
                vector_script=args.vector_server,
                vector_env_extra=vector_env_extra,
                max_rounds=args.max_rounds,
                verbose=args.verbose,
                query=args.query,
                interactive=False,
                demo=False,
            )
        )
        return

    asyncio.run(
        run_multi_sessions(
            ollama_url=args.ollama_url,
            model=args.model,
            python_exe=args.python,
            leave_script=args.leave_server,
            users_tasks_script=args.users_tasks_server,
            vector_script=args.vector_server,
            vector_env_extra=vector_env_extra,
            max_rounds=args.max_rounds,
            verbose=args.verbose,
            query=None,
            interactive=True,
            demo=False,
        )
    )


if __name__ == "__main__":
    main()
