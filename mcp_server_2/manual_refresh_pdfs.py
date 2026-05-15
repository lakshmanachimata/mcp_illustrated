#!/usr/bin/env python3
"""Rebuild ``pdfs/*.pdf`` into ``vector_mcp.db`` (same work as MCP ``refresh_vector_db``).

Run from this directory while Ollama is up. Prefer **only one** writer to the DB
(stop Cursor’s vector MCP server or any other ``server.py`` first).

Examples::

    cd mcp_server_2 && ../.venv/bin/python manual_refresh_pdfs.py

    VECTOR_MCP_EMBED_MODEL=embeddinggemma OLLAMA_HOST=http://127.0.0.1:11434 \\
      python manual_refresh_pdfs.py

Summary JSON prints to stdout; ``vector_store`` logs to logger ``vector_mcp`` — same embedding path as MCP ``refresh_vector_db`` / manual ``vs.refresh_vector_db()``.
"""

from __future__ import annotations

import json
import logging
import sys

import vector_store as vs


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [vector-mcp-cli] %(message)s", stream=sys.stderr)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    try:
        out = vs.refresh_vector_db()
    except KeyboardInterrupt:
        print("\ninterrupt", file=sys.stderr)
        return 130
    print(json.dumps(out, indent=2, ensure_ascii=False))
    ok = bool(out.get("ok", True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
