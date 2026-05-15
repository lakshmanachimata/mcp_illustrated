"""
MCP client: folder vector index (``mcp_server_2/server.py``) + Ollama natural language.

Embeddings use Ollama (``embeddinggemma`` by default on the server side); chat uses your
``--model`` for planning tool calls.

Examples:
  .venv/bin/python vector_mcp_ollama_client.py
  .venv/bin/python vector_mcp_ollama_client.py -q "What does vector_index_stats say?"
  VECTOR_MCP_EMBED_MODEL=embeddinggemma .venv/bin/python vector_mcp_ollama_client.py --demo
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
    import readline  # noqa: F401
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent
DEFAULT_SERVER = ROOT / "server.py"

SYSTEM_PROMPT = """You are an assistant for a local folder vector index exposed only via MCP tools.

Tools:
- refresh_vector_db: rebuild embeddings for every PDF under the server’s ``pdfs`` folder (next to ``server.py``). On the server side this runs only when you call this tool (or optionally at process start via ``VECTOR_MCP_STARTUP_REFRESH=1`` on the server).
- index_folder: index any folder (text + PDF), with glob and extension filters.
- query_vectors: semantic search over indexed chunks (query text, top_k).
- vector_index_stats: chunk/file counts and every indexed path as ``source_paths``.
- clear_vector_index: wipe the index.

Always use tools for index operations; do not invent file contents. After tools return, answer concisely."""


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
    params = StdioServerParameters(command=str(python_exe), args=[str(server_script)], env=env)
    print(
        "Vector MCP + Ollama — natural language against index_folder / query_vectors.\n"
        "quit | exit | q  or Ctrl-D to stop.\n",
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
    params = StdioServerParameters(command=str(python_exe), args=[str(server_script)], env=env)
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
    parser = argparse.ArgumentParser(description="Ollama + stdio MCP client for mcp_server_2 vector index.")
    parser.add_argument("-q", "--query", help="Natural-language instruction for the vector tools.")
    parser.add_argument("--demo", action="store_true", help="Sample questions (needs Ollama chat + embed).")
    parser.add_argument(
        "--model",
        default=os.environ.get("OLLAMA_MODEL", "qwen2.5:latest"),
        help="Ollama chat model (default: qwen2.5:latest or OLLAMA_MODEL).",
    )
    parser.add_argument(
        "--ollama-url",
        default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
        help="Ollama base URL.",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable), help="Python for server.py.")
    parser.add_argument("--server-script", type=Path, default=DEFAULT_SERVER, help="Path to server.py.")
    parser.add_argument("--max-rounds", type=int, default=12, help="Max tool rounds per query.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if not args.server_script.is_file():
        print(f"Server script not found: {args.server_script}", file=sys.stderr)
        sys.exit(1)

    if args.demo:
        repo = Path(__file__).resolve().parents[1]
        docs_dir = repo / "docs"
        folder = str(docs_dir) if docs_dir.is_dir() else str(repo)
        queries = [
            "Call vector_index_stats and report the numbers.",
            f"If the index is empty, call index_folder with folder_path {folder!r} and glob_pattern **/*.md with extensions .md only. Then call vector_index_stats again.",
            "Call query_vectors with something relevant to this repo (one short phrase) and top_k 3, then summarize the top match.",
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
