"""SQLite + Ollama embeddings for a simple local vector index."""

from __future__ import annotations

import contextlib
import io
import json
import logging
import math
import os
import sqlite3
import subprocess
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None  # type: ignore[misc, assignment]

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "vector_mcp.db"
_PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_PDFS_DIR = _PACKAGE_ROOT / "pdfs"

_LOG = logging.getLogger("vector_mcp")

# Serialize all SQLite use in this process (MCP may call tools from different threads).
_SQLITE_THREAD_LOCK = threading.RLock()


def _flush_vector_mcp_handlers() -> None:
    """So file+log handlers push lines promptly."""
    for h in _LOG.handlers:
        try:
            h.flush()
        except OSError:
            pass


def db_path() -> Path:
    raw = os.environ.get("VECTOR_MCP_DB", "").strip()
    return Path(raw).expanduser() if raw else DEFAULT_DB_PATH


def _sqlite_diag_enabled() -> bool:
    return os.environ.get("VECTOR_MCP_SQLITE_DIAG", "").strip().lower() in ("1", "true", "yes")


def _sqlite_lsof_enabled() -> bool:
    return os.environ.get("VECTOR_MCP_SQLITE_LSOF", "").strip().lower() in ("1", "true", "yes")


def _thread_debug_label() -> str:
    tid = threading.get_native_id()
    return (
        f"pid={os.getpid()} native_tid={tid} thread={threading.current_thread().name!r} "
        f"active_threads={threading.active_count()}"
    )


def _run_lsof_on_db(where: str) -> None:
    """Run ``lsof`` on the DB path and log a short preview (no env gate)."""
    path = db_path()
    try:
        r = subprocess.run(
            ["lsof", str(path)],
            capture_output=True,
            text=True,
            timeout=float(os.environ.get("VECTOR_MCP_SQLITE_LSOF_TIMEOUT", "4").strip() or "4"),
        )
        out = (r.stdout or "").strip()
        lines = out.splitlines()
        preview = "\n".join(lines[:25])
        suffix = "\n...(truncated)" if len(lines) > 25 else ""
        _LOG.warning("sqlite_lsof_diag where=%s path=%s exit=%s\n%s%s", where, path, r.returncode, preview, suffix)
        _flush_vector_mcp_handlers()
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        _LOG.warning("sqlite_lsof_diag_failed where=%s err=%s", where, exc)


def _diag_lsof_db(where: str) -> None:
    if not _sqlite_lsof_enabled():
        return
    _run_lsof_on_db(where)


@contextlib.contextmanager
def sqlite_mutex(where: str) -> Iterator[None]:
    """
    In-process mutex for every sqlite3 connection (``check_same_thread=False`` + this lock).
    If this waits, another **thread** in the same process held the DB.
    """
    t0 = time.perf_counter()
    _SQLITE_THREAD_LOCK.acquire()
    waited = time.perf_counter() - t0
    if waited >= 0.001:
        _LOG.info(
            "sqlite_inprocess_mutex_wait waited_s=%.4f where=%s %s",
            waited,
            where,
            _thread_debug_label(),
        )
        _flush_vector_mcp_handlers()
    elif _sqlite_diag_enabled():
        _LOG.info("sqlite_inprocess_mutex_ok where=%s %s", where, _thread_debug_label())
    try:
        yield
    finally:
        _SQLITE_THREAD_LOCK.release()


def _silence_noisy_loggers() -> None:
    """Keep httpx/httpcore urllib3 chatter off the default handlers (stdout risk for stdio MCP)."""
    for name in (
        "httpx",
        "httpcore",
        "httpcore.http11",
        "httpcore.connection",
        "urllib3",
        "urllib3.connectionpool",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)


_silence_noisy_loggers()


try:
    import fcntl

    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False


_write_lock_depth = threading.local()


def _sqlite_write_lock_path() -> Path:
    raw = os.environ.get("VECTOR_MCP_SQLITE_LOCK_FILE", "").strip()
    if raw:
        return Path(raw).expanduser()
    return db_path().parent / ".vector_mcp.sqlite.write.lock"


@contextlib.contextmanager
def serialized_sqlite_writes() -> Iterator[None]:
    """
    Serialize SQLite-mutating sections across processes (e.g. two MCP server instances).

    Same-process nested calls only take the flock once (``refresh_vector_db`` → ``index_folder``).
    Without ``fcntl`` (Windows), this is a no-op.
    """
    with sqlite_mutex("serialized_sqlite_writes"):
        d = getattr(_write_lock_depth, "depth", 0)
        lf = None
        try:
            if d == 0 and _HAS_FCNTL:
                p = _sqlite_write_lock_path()
                p.parent.mkdir(parents=True, exist_ok=True)
                lf = open(p, "a+b")  # noqa: SIM115 — closed in finally
                try:
                    fcntl.flock(lf.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    if _sqlite_diag_enabled():
                        _LOG.info(
                            "sqlite_flock_ok_nonblocking path=%s %s",
                            p,
                            _thread_debug_label(),
                        )
                except BlockingIOError:
                    _LOG.warning(
                        "sqlite_flock_wait path=%s %s hint=another_process_holds_interprocess_lock",
                        p,
                        _thread_debug_label(),
                    )
                    _flush_vector_mcp_handlers()
                    tw = time.perf_counter()
                    fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
                    _LOG.info(
                        "sqlite_flock_acquired_after_wait wait_s=%.3f path=%s %s",
                        time.perf_counter() - tw,
                        p,
                        _thread_debug_label(),
                    )
                    _flush_vector_mcp_handlers()
            _write_lock_depth.depth = d + 1
            yield
        finally:
            _write_lock_depth.depth = d
            if lf is not None:
                try:
                    fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
                try:
                    lf.close()
                except OSError:
                    pass


def ollama_base() -> str:
    return os.environ.get("VECTOR_MCP_OLLAMA_URL", os.environ.get("OLLAMA_HOST", "")).strip() or "http://127.0.0.1:11434"


def embed_model() -> str:
    return os.environ.get("VECTOR_MCP_EMBED_MODEL", "embeddinggemma").strip() or "embeddinggemma"


def connect(*, lite: bool = False) -> sqlite3.Connection:
    """
    Normal connections use generous timeouts for long indexing jobs.
    ``lite=True`` uses shorter connection wait / ``busy_timeout`` (same knobs as DELETE:
    ``VECTOR_MCP_SQLITE_DELETE_CONNECT_TIMEOUT``, ``VECTOR_MCP_SQL_LITE_CONNECT_BUSY_MS``) for fast
    progress when another process holds the DB — including ``index_stats()`` and server startup probes.
    """
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if lite:
        timeout = float(os.environ.get("VECTOR_MCP_SQLITE_DELETE_CONNECT_TIMEOUT", "12").strip() or "12")
        busy_ms_raw = os.environ.get("VECTOR_MCP_SQL_LITE_CONNECT_BUSY_MS", "").strip()
        busy_ms = int(busy_ms_raw) if busy_ms_raw else 5_000
    else:
        timeout = float(os.environ.get("VECTOR_MCP_SQLITE_TIMEOUT", "300").strip() or "300")
        busy_ms_raw = os.environ.get("VECTOR_MCP_SQLITE_BUSY_MS", "").strip()
        busy_ms = int(busy_ms_raw) if busy_ms_raw else 300_000
    if busy_ms < 0:
        busy_ms = 300_000
    conn = sqlite3.connect(path, timeout=timeout, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(f"PRAGMA busy_timeout = {busy_ms}")
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.OperationalError:
        pass
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_path TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            embedding_json TEXT NOT NULL,
            UNIQUE (source_path, chunk_index)
        );

        CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(source_path);
        """
    )


def _set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def _get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def fetch_embedding(client: httpx.Client, text: str) -> list[float]:
    """Call Ollama. Newer servers use ``POST /api/embed``; older use ``/api/embeddings``."""
    base = ollama_base().rstrip("/")
    model = embed_model()

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = client.post(
            f"{base}/api/embed",
            json={"model": model, "input": text},
            timeout=120.0,
        )
        if r.status_code == 404:
            r = client.post(
                f"{base}/api/embeddings",
                json={"model": model, "prompt": text},
                timeout=120.0,
            )
        r.raise_for_status()
        data = r.json()

    embs = data.get("embeddings")
    if isinstance(embs, list) and embs:
        vec = embs[0]
        if isinstance(vec, list):
            return [float(x) for x in vec]

    emb = data.get("embedding")
    if isinstance(emb, list):
        return [float(x) for x in emb]

    raise RuntimeError(f"Unexpected embedding response: {data!r}")


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return -1.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_overlap must be in [0, chunk_size)")
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end >= n:
            break
        start = end - overlap
    return chunks


def _parse_extensions(raw: str) -> set[str]:
    parts = [p.strip().lower() for p in raw.replace(";", ",").split(",")]
    out: set[str] = set()
    for p in parts:
        if not p:
            continue
        if not p.startswith("."):
            p = "." + p
        out.add(p)
    return out


def _iter_files(root: Path, glob_pattern: str, extensions: set[str]) -> list[Path]:
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")
    out: set[Path] = set()
    for p in root.glob(glob_pattern):
        if not p.is_file():
            continue
        if extensions and p.suffix.lower() not in extensions:
            continue
        out.add(p.resolve())
    return sorted(out)


_MAX_FILE_BYTES = 5 * 1024 * 1024
_MAX_PDF_BYTES = 25 * 1024 * 1024


def default_pdfs_dir() -> Path:
    """Bundled PDF library directory for this MCP server."""
    return DEFAULT_PDFS_DIR


def extract_pdf_text(path: Path) -> str:
    if PdfReader is None:
        raise RuntimeError("pypdf is not installed; pip install pypdf")

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        reader = PdfReader(str(path), strict=False)
        parts: list[str] = []
        for page in reader.pages:
            t = page.extract_text()
            if t and t.strip():
                parts.append(t.strip())
    return "\n\n".join(parts)


def read_document(fp: Path) -> tuple[str | None, str | None]:
    """
    Load file body as UTF-8 text or PDF extract. Returns (text, error_message).
    """
    try:
        size = fp.stat().st_size
    except OSError as exc:
        return None, f"stat_error: {fp}: {exc}"
    suffix = fp.suffix.lower()
    max_bytes = _MAX_PDF_BYTES if suffix == ".pdf" else _MAX_FILE_BYTES
    if size > max_bytes:
        return None, f"skip_too_large: {fp} ({size} bytes)"
    if suffix == ".pdf":
        try:
            body = extract_pdf_text(fp)
            return body, None
        except Exception as exc:  # noqa: BLE001
            return None, f"pdf_error: {fp}: {exc}"
    try:
        return fp.read_text(encoding="utf-8", errors="replace"), None
    except OSError as exc:
        return None, f"read_error: {fp}: {exc}"


def delete_chunks_under_directory(conn: sqlite3.Connection, folder: Path) -> int:
    """Remove chunks under ``folder`` (resolved paths). Uses ``BEGIN IMMEDIATE`` and retries briefly on SQLITE_BUSY."""
    root = folder.resolve()
    prefix = str(root)
    like = prefix + os.sep + "%"

    del_busy_ms = int(os.environ.get("VECTOR_MCP_SQLITE_DELETE_BUSY_MS", "6000").strip() or "6000")
    if del_busy_ms < 500:
        del_busy_ms = 500
    conn.execute(f"PRAGMA busy_timeout = {del_busy_ms}")

    backoff = float(os.environ.get("VECTOR_MCP_SQLITE_LOCK_BACKOFF", "0.25").strip() or "0.25")
    attempts = max(1, min(25, int(os.environ.get("VECTOR_MCP_SQLITE_BUSY_RETRIES", "8").strip() or "8")))

    lsof_on_busy = os.environ.get("VECTOR_MCP_SQLITE_LSOF_ON_BUSY", "").strip().lower() in ("1", "true", "yes")
    did_lsof = False

    last_exc: sqlite3.OperationalError | None = None
    for attempt in range(attempts):
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                cur = conn.execute(
                    "DELETE FROM chunks WHERE source_path = ? OR source_path LIKE ?",
                    (prefix, like),
                )
                rowcount = cur.rowcount
                conn.commit()
                return rowcount
            except Exception:
                conn.rollback()
                raise
        except sqlite3.OperationalError as exc:
            last_exc = exc
            if "locked" not in str(exc).lower() and "busy" not in str(exc).lower():
                raise
            _LOG.warning(
                "sqlite_delete_busy_retry attempt=%s/%s db=%s %s err=%s",
                attempt + 1,
                attempts,
                db_path(),
                _thread_debug_label(),
                exc,
            )
            if attempt == 0:
                nt = threading.active_count()
                if nt <= 2:
                    _LOG.warning(
                        "sqlite_busy_hint nt=%s (main thread only → lock is usually **another OS process**/GUI "
                        "with %s open, not MCP multi-thread here). VECTOR_MCP_SQLITE_LSOF_ON_BUSY=1 for lsof snippet "
                        "(no need for VECTOR_MCP_SQLITE_LSOF); VECTOR_MCP_SQLITE_LSOF=1 also runs lsof when retries "
                        "are exhausted.",
                        nt,
                        db_path(),
                    )
                else:
                    _LOG.warning(
                        "sqlite_busy_hint nt=%s (multiple threads) — serialization may still overlap; capture logs.",
                        nt,
                    )
                if lsof_on_busy and not did_lsof:
                    did_lsof = True
                    _run_lsof_on_db("sqlite_delete_first_busy")
            _flush_vector_mcp_handlers()
            time.sleep(min(2.0, backoff * (attempt + 1)))
    _diag_lsof_db("delete_chunks_exhausted")
    raise last_exc if last_exc else sqlite3.OperationalError("database is locked")


def index_folder(
    folder_path: str,
    *,
    glob_pattern: str = "**/*",
    extensions_csv: str = ".txt,.md,.py,.json,.yaml,.yml,.rst,.pdf",
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> dict[str, Any]:
    root = Path(folder_path).expanduser().resolve()
    exts = _parse_extensions(extensions_csv)
    files = _iter_files(root, glob_pattern, exts)

    with serialized_sqlite_writes():
        conn = connect()
        init_schema(conn)
        _set_meta(conn, "embed_model", embed_model())
        _set_meta(conn, "ollama_base", ollama_base())
        conn.commit()

        indexed_files = 0
        total_chunks = 0
        errors: list[str] = []

        with httpx.Client() as client:
            for fp in files:
                rel = str(fp.resolve())
                body, err = read_document(fp)
                if err:
                    errors.append(err)
                    continue
                if body is None:
                    continue

                conn.execute("DELETE FROM chunks WHERE source_path = ?", (rel,))
                pieces = chunk_text(body, chunk_size, chunk_overlap) if body.strip() else []
                if not pieces:
                    conn.commit()
                    indexed_files += 1
                    continue

                try:
                    for i, piece in enumerate(pieces):
                        emb = fetch_embedding(client, piece)
                        conn.execute(
                            "INSERT INTO chunks(source_path, chunk_index, text, embedding_json) VALUES (?,?,?,?)",
                            (rel, i, piece, json.dumps(emb)),
                        )
                        total_chunks += 1
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
                indexed_files += 1

        conn.close()
    return {
        "ok": True,
        "folder": str(root),
        "files_indexed": indexed_files,
        "chunks_written": total_chunks,
        "embed_model": embed_model(),
        "errors": errors,
    }


def refresh_vector_db(
    *,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> dict[str, Any]:
    """
    Rebuild the index for PDFs under the bundled ``pdfs`` directory (next to this package).
    Removes existing chunks for those paths, then re-embeds all ``*.pdf`` files found.
    Logs ``vector_db_refresh_started`` / ``vector_db_refresh_finished`` (and failure) on logger ``vector_mcp``.
    """
    pdf_root = default_pdfs_dir()
    resolved = pdf_root.resolve()
    pdf_root.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    _LOG.info(
        "vector_db_refresh_started pdfs_dir=%s db_path=%s ollama=%s embed_model=%s chunk_size=%s chunk_overlap=%s",
        resolved,
        db_path(),
        ollama_base(),
        embed_model(),
        chunk_size,
        chunk_overlap,
    )
    _flush_vector_mcp_handlers()

    removed = 0
    try:
        with serialized_sqlite_writes():
            conn = connect(lite=True)
            init_schema(conn)
            removed = delete_chunks_under_directory(conn, pdf_root)
            conn.close()

            out = index_folder(
                str(pdf_root),
                glob_pattern="**/*.pdf",
                extensions_csv=".pdf",
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        out["refresh"] = True
        out["pdfs_dir"] = str(resolved)
        out["chunks_removed_before_refresh"] = removed

        elapsed = time.perf_counter() - t0
        err_n = len(out.get("errors") or [])
        _LOG.info(
            "vector_db_refresh_finished ok=%s files_indexed=%s chunks_written=%s chunks_removed_before=%s "
            "elapsed_s=%.2f embed_errors=%s",
            out.get("ok"),
            out.get("files_indexed"),
            out.get("chunks_written"),
            removed,
            elapsed,
            err_n,
        )
        return out
    except Exception:
        _LOG.exception(
            "vector_db_refresh_failed pdfs_dir=%s elapsed_s=%.2f",
            resolved,
            time.perf_counter() - t0,
        )
        raise
    finally:
        _flush_vector_mcp_handlers()


def clear_index() -> dict[str, Any]:
    with serialized_sqlite_writes():
        conn = connect()
        init_schema(conn)
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM meta")
        conn.commit()
        conn.close()
    return {"ok": True, "message": "All chunks and meta cleared."}


def index_stats(*, lite: bool = True) -> dict[str, Any]:
    """Metadata + all indexed paths. Uses ``connect(lite=True)`` by default so MCP startup does not block long."""
    with sqlite_mutex("index_stats"):
        conn = connect(lite=lite)
        init_schema(conn)
        n_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        n_files = conn.execute("SELECT COUNT(DISTINCT source_path) FROM chunks").fetchone()[0]
        model = _get_meta(conn, "embed_model")
        base = _get_meta(conn, "ollama_base")
        paths = conn.execute(
            "SELECT DISTINCT source_path FROM chunks ORDER BY source_path"
        ).fetchall()
        conn.close()
    sorted_paths = [r[0] for r in paths]
    return {
        "ok": True,
        "chunk_count": n_chunks,
        "file_count": n_files,
        "embed_model": model,
        "ollama_base_recorded": base,
        "source_paths": sorted_paths,
        # Back-compat: historically only 8 rows; callers should use ``source_paths``.
        "sample_paths": sorted_paths,
    }


def query_similar(query: str, top_k: int = 5) -> dict[str, Any]:
    k = max(1, min(int(top_k), 50))
    with sqlite_mutex("query_similar_load"):
        conn = connect()
        init_schema(conn)
        rows = conn.execute(
            "SELECT id, source_path, chunk_index, text, embedding_json FROM chunks"
        ).fetchall()
        conn.close()

    if not rows:
        return {"ok": True, "matches": [], "note": "Index is empty; run refresh_vector_db or index_folder first."}

    with httpx.Client() as client:
        q_emb = fetch_embedding(client, query)

    scored: list[tuple[float, dict[str, Any]]] = []
    for _id, path, idx, text, emb_json in rows:
        try:
            emb = json.loads(emb_json)
            if not isinstance(emb, list):
                continue
            vec = [float(x) for x in emb]
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        sim = cosine_similarity(q_emb, vec)
        scored.append(
            (
                sim,
                {
                    "source_path": path,
                    "chunk_index": idx,
                    "score": round(sim, 6),
                    "text": text[:2000] + ("…" if len(text) > 2000 else ""),
                },
            )
        )

    scored.sort(key=lambda x: x[0], reverse=True)
    matches = [item[1] for item in scored[:k]]
    return {"ok": True, "query": query, "top_k": k, "matches": matches}
