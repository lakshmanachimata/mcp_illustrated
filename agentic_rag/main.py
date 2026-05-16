#!/usr/bin/env python3
"""
Agentic RAG CLI: LangGraph agent + dynamic MCP routing.

Connects only to MCP servers needed for each prompt (leave / users_tasks / vector).
All data access goes through MCP subprocesses — this script never opens vector SQLite.

Examples:
  cd /path/to/mcp_illustrated
  pip install -r agentic_rag/requirements.txt

  python -m agentic_rag -q "Show alice's leaves"
  python -m agentic_rag -q "vector_index_stats and file count" --router keyword
  python -m agentic_rag -q "List users and their open tasks" --servers users_tasks
  python -m agentic_rag -q "..." --router all   # connect every server (like multi_mcp)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from agentic_rag.router import RouterMode
from agentic_rag.run import run_agent_query


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LangGraph agent with prompt-based dynamic MCP server connections."
    )
    parser.add_argument("-q", "--query", required=True, help="User question.")
    parser.add_argument(
        "--router",
        choices=("keyword", "llm", "hybrid", "all"),
        default="hybrid",
        help="How to pick MCP servers from the prompt (default: hybrid).",
    )
    parser.add_argument(
        "--servers",
        default="",
        metavar="IDS",
        help="Comma-separated server ids (leave,users_tasks,vector). Skips auto-routing.",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Ollama chat model (default: OLLAMA_MODEL or qwen2.5:latest).",
    )
    parser.add_argument(
        "--ollama-url",
        default="",
        help="Ollama base URL (default: OLLAMA_HOST or http://127.0.0.1:11434).",
    )
    parser.add_argument("--max-iterations", type=int, default=16)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    server_ids: list[str] | None = None
    if args.servers.strip():
        server_ids = [s.strip() for s in args.servers.split(",") if s.strip()]

    try:
        out = asyncio.run(
            run_agent_query(
                args.query,
                server_ids=server_ids,
                router_mode=args.router,  # type: ignore[arg-type]
                model=args.model or None,
                ollama_base_url=args.ollama_url or None,
                max_iterations=args.max_iterations,
                verbose=args.verbose,
            )
        )
    except KeyboardInterrupt:
        print("\n(interrupted)", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        sys.exit(1)

    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    if out.get("answer"):
        print("\n--- answer ---\n")
        print(out["answer"])


if __name__ == "__main__":
    main()
