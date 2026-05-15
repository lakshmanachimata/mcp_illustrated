"""
MCP server: index text files and PDFs under folders with Ollama embeddings; query by similarity (SQLite).

Does **not** rebuild the vector index on startup. Call the ``refresh_vector_db`` tool from the MCP client when you want to re-embed all PDFs under ``pdfs/`` next to this script. Optional opt-in: ``VECTOR_MCP_STARTUP_REFRESH=1`` runs that same rebuild once at server process start (e.g. for scripts; avoid with multiple server instances).

Run: python server.py

Stores vectors in SQLite (see env ``VECTOR_MCP_DB``). Embeddings via Ollama ``/api/embed`` (fallback ``/api/embeddings``).
By default startup does **not** query SQLite (instant boot). Set ``VECTOR_MCP_STARTUP_INDEX_STATS=1`` to log chunk/file counts before serving (uses ``connect(lite=True)`` via ``VECTOR_MCP_SQLITE_DELETE_CONNECT_TIMEOUT``, ``VECTOR_MCP_SQL_LITE_CONNECT_BUSY_MS``). Heavy tools use full ``connect()`` with optional ``VECTOR_MCP_SQLITE_TIMEOUT`` (seconds, default ``300``) and ``VECTOR_MCP_SQLITE_BUSY_MS`` (default ``300000``). Deletes during refresh use ``connect(lite=True)``. Per-delete pragmas ``VECTOR_MCP_SQLITE_DELETE_BUSY_MS`` (default ``6000`` ms) and ``VECTOR_MCP_SQLITE_BUSY_RETRIES`` (default ``8``). ``VECTOR_MCP_SQLITE_LSOF_ON_BUSY=1`` runs ``lsof`` on the **first** busy row (**without** needing ``VECTOR_MCP_SQLITE_LSOF``); ``VECTOR_MCP_SQLITE_LSOF=1`` at exhaustion. ``VECTOR_MCP_SQLITE_LOCK_FILE``, ``VECTOR_MCP_SQLITE_LOCK_BACKOFF``. Debugging: ``VECTOR_MCP_SQLITE_DIAG``, ``VECTOR_MCP_SQLITE_LSOF``, ``VECTOR_MCP_SQLITE_LSOF_TIMEOUT``.
Env ``VECTOR_MCP_EMBED_MODEL`` (default ``embeddinggemma``). Stdio MCP reserves **stdout** for JSON-RPC; this server logs to **stderr**. Do not type into this process’s stdin when it is a terminal (keystrokes corrupt JSON-RPC).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

import vector_store as vs

mcp = FastMCP("Folder vector index")

_LOG = logging.getLogger("vector_mcp")
_DEFAULT_LOG_PATH = Path(__file__).resolve().parent / "vector-mcp.log"


def _ensure_logging() -> None:
    if _LOG.handlers:
        return
    fmt = logging.Formatter("%(asctime)s %(levelname)s [vector-mcp] %(message)s")
    err = logging.StreamHandler(sys.stderr)
    err.setFormatter(fmt)
    _LOG.addHandler(err)

    no_file = os.environ.get("VECTOR_MCP_NO_FILE_LOG", "").strip().lower() in ("1", "true", "yes")
    path_str = os.environ.get("VECTOR_MCP_LOG_FILE", "").strip()
    log_path = Path(path_str) if path_str else _DEFAULT_LOG_PATH
    if not no_file:
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
            fh.setFormatter(fmt)
            _LOG.addHandler(fh)
        except OSError as exc:
            sys.stderr.write(f"[vector-mcp] WARNING: could not open log file {log_path}: {exc}\n")
    _LOG.setLevel(logging.INFO)
    _LOG.propagate = False


def _flush_log_handlers() -> None:
    for h in _LOG.handlers:
        try:
            h.flush()
        except OSError:
            pass


def _log_tool(tool_name: str, **params: Any) -> None:
    _ensure_logging()
    _LOG.info("tool_call %s %s", tool_name, json.dumps(params, default=str, sort_keys=True))
    _flush_log_handlers()


@mcp.tool()
def index_folder(
    folder_path: str,
    glob_pattern: str = "**/*",
    extensions: str = ".txt,.md,.py,.json,.yaml,.yml,.rst,.pdf",
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> str:
    """
    Read text/PDF files under ``folder_path``, chunk them, compute embeddings via Ollama,
    and store rows in SQLite. Replaces existing chunks per file path when re-indexing the same file.
    ``extensions``: comma-separated list (e.g. ".txt,.md"). ``glob_pattern`` is relative to the folder.
    """
    _log_tool(
        "index_folder",
        folder_path=folder_path,
        glob_pattern=glob_pattern,
        extensions=extensions,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    try:
        with redirect_stdout(StringIO()):
            out = vs.index_folder(
                folder_path,
                glob_pattern=glob_pattern,
                extensions_csv=extensions,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        return json.dumps(out, indent=2)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"ok": False, "error": str(exc)}, indent=2)


@mcp.tool()
def refresh_vector_db(chunk_size: int = 1200, chunk_overlap: int = 200) -> str:
    """
    Rebuild the vector index for all PDFs under the bundled ``pdfs`` folder (inside mcp_server_2).
    Removes prior chunks from that folder, then re-embeds every ``*.pdf`` found there.
    """
    _log_tool("refresh_vector_db", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    try:
        # Same entry as manual_refresh_pdfs.py (avoid extra stdout wrapper; MCP JSON-RPC stays on stdout only).
        out = vs.refresh_vector_db(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return json.dumps(out, indent=2)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"ok": False, "error": str(exc)}, indent=2)


@mcp.tool()
def clear_vector_index() -> str:
    """Remove all chunks and metadata from the vector index."""
    _log_tool("clear_vector_index")
    try:
        return json.dumps(vs.clear_index(), indent=2)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"ok": False, "error": str(exc)}, indent=2)


@mcp.tool()
def query_vectors(query: str, top_k: int = 5) -> str:
    """Embed ``query`` and return the top_k most similar chunks (text excerpts + scores)."""
    _log_tool("query_vectors", query=query[:200], top_k=top_k)
    try:
        with redirect_stdout(StringIO()):
            out = vs.query_similar(query, top_k=top_k)
        return json.dumps(out, indent=2)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"ok": False, "error": str(exc)}, indent=2)


@mcp.tool()
def vector_index_stats() -> str:
    """Counts, ``file_count``, and ``source_paths`` (every indexed file path, sorted)."""
    _log_tool("vector_index_stats")
    try:
        return json.dumps(vs.index_stats(), indent=2)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"ok": False, "error": str(exc)}, indent=2)


def _maybe_startup_refresh() -> None:
    """Rebuild from ``pdfs/`` only when ``VECTOR_MCP_STARTUP_REFRESH=1`` (default: off; use ``refresh_vector_db`` tool)."""
    _ensure_logging()
    enabled = os.environ.get("VECTOR_MCP_STARTUP_REFRESH", "").strip().lower() in ("1", "true", "yes")
    if not enabled:
        _LOG.info("vector index: startup refresh disabled — use refresh_vector_db from the MCP client when needed")
        _flush_log_handlers()
        return
    try:
        out = vs.refresh_vector_db()
        summary = {
            k: out.get(k)
            for k in ("ok", "files_indexed", "chunks_written", "pdfs_dir", "chunks_removed_before_refresh")
        }
        summary["errors"] = out.get("errors", [])[:5]
        if out.get("errors") and len(out["errors"]) > 5:
            summary["errors_truncated"] = True
        _LOG.info("startup vector refresh (VECTOR_MCP_STARTUP_REFRESH): %s", json.dumps(summary, default=str))
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("startup vector refresh failed (MCP server still starts): %s", exc)
        estr = str(exc).lower()
        if "locked" in estr or "busy" in estr:
            _LOG.warning(
                "SQLite busy — close other processes using %s, then call refresh_vector_db from the client "
                "or retry with a single server instance.",
                vs.db_path(),
            )
    _flush_log_handlers()


def main() -> None:
    _ensure_logging()
    _maybe_startup_refresh()
    startup_stats = os.environ.get("VECTOR_MCP_STARTUP_INDEX_STATS", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if startup_stats:
        try:
            st = vs.index_stats()
            chunks_preview = st.get("chunk_count")
            files_preview = st.get("file_count")
        except sqlite3.OperationalError as exc:
            chunks_preview = f"unavailable ({exc})"
            files_preview = chunks_preview
        _LOG.info(
            "vector_mcp_server starting (stdio MCP) db=%s chunks=%s files=%s embed_model_env=%s ollama=%s",
            vs.db_path(),
            chunks_preview,
            files_preview,
            vs.embed_model(),
            vs.ollama_base(),
        )
    else:
        _LOG.info(
            "vector_mcp_server starting (stdio MCP) db=%s embed_model_env=%s ollama=%s "
            "(no startup DB query; call vector_index_stats or set VECTOR_MCP_STARTUP_INDEX_STATS=1)",
            vs.db_path(),
            vs.embed_model(),
            vs.ollama_base(),
        )
    _flush_log_handlers()
    if sys.stdin.isatty():
        _LOG.warning(
            "stdio MCP: stdin is an interactive TTY — do not press keys here; stray input breaks JSON-RPC. "
            "Run this server only from an MCP host, or use a dedicated terminal without typing into it."
        )
        _LOG.info(
            "stdio MCP: this process now blocks waiting for MCP JSON-RPC on stdin — it is idle (no embedding work) "
            "until a host connects. The shell prompt will not return until you press Ctrl+C."
        )
        _flush_log_handlers()
    mcp.run()


if __name__ == "__main__":
    main()
