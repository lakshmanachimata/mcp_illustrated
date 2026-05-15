"""
MCP server: users and tasks in SQLite (one user, many tasks).

Run: python server.py
Configure MCP with command ``python`` and args pointing to this file.

Database path: ``users_tasks.db`` next to this script, or override with env ``USERS_TASKS_MCP_DB``.

Logging: stderr and ``users-tasks-mcp.log`` beside this script (override ``USERS_TASKS_MCP_LOG_FILE``;
set ``USERS_TASKS_MCP_NO_FILE_LOG=1`` to disable file logging).
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

import db as dbmod

mcp = FastMCP("Users and Tasks")

_LOG = logging.getLogger("users_tasks_mcp")
_DEFAULT_LOG_PATH = Path(__file__).resolve().parent / "users-tasks-mcp.log"


def _ensure_logging() -> None:
    if _LOG.handlers:
        return
    fmt = logging.Formatter("%(asctime)s %(levelname)s [users-tasks-mcp] %(message)s")

    err = logging.StreamHandler(sys.stderr)
    err.setFormatter(fmt)
    _LOG.addHandler(err)

    no_file = os.environ.get("USERS_TASKS_MCP_NO_FILE_LOG", "").strip().lower() in ("1", "true", "yes")
    path_str = os.environ.get("USERS_TASKS_MCP_LOG_FILE", "").strip()
    log_path = Path(path_str) if path_str else _DEFAULT_LOG_PATH

    if not no_file:
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
            fh.setFormatter(fmt)
            _LOG.addHandler(fh)
        except OSError as exc:
            sys.stderr.write(f"[users-tasks-mcp] WARNING: could not open log file {log_path}: {exc}\n")

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


def _with_conn() -> Any:
    conn = dbmod.connect()
    dbmod.init_schema(conn)
    return conn


def _parse_int_ids(csv: str) -> tuple[list[int], list[str]]:
    """Parse comma/semicolon-separated integers; non-integers collected as unknown."""
    raw = csv.strip()
    if not raw:
        return [], []
    chunks = [c.strip() for c in re.split(r"[,;]+", raw) if c.strip()]
    ids: list[int] = []
    bad: list[str] = []
    for c in chunks:
        try:
            ids.append(int(c))
        except ValueError:
            bad.append(c)
    return ids, bad


@mcp.tool()
def create_user(name: str, age: int, gender: str) -> str:
    """Create a user. Returns JSON with the new row including ``id``."""
    _log_tool("create_user", name=name, age=age, gender=gender)
    conn = _with_conn()
    try:
        conn.execute("BEGIN")
        row = dbmod.user_create(conn, name, age, gender)
        conn.execute("COMMIT")
        return json.dumps({"ok": True, "user": row}, indent=2)
    except Exception as exc:  # noqa: BLE001
        conn.execute("ROLLBACK")
        return json.dumps({"ok": False, "error": str(exc)}, indent=2)


@mcp.tool()
def get_user(user_id: int) -> str:
    """Read one user by integer ``user_id``."""
    _log_tool("get_user", user_id=user_id)
    conn = _with_conn()
    row = dbmod.user_get(conn, user_id)
    if not row:
        return json.dumps({"ok": False, "error": "not_found", "user_id": user_id}, indent=2)
    return json.dumps({"ok": True, "user": row}, indent=2)


@mcp.tool()
def list_users() -> str:
    """List all users."""
    _log_tool("list_users")
    conn = _with_conn()
    rows = dbmod.user_list(conn)
    return json.dumps({"ok": True, "users": rows}, indent=2)


@mcp.tool()
def update_user(
    user_id: int,
    name: str | None = None,
    age: int | None = None,
    gender: str | None = None,
) -> str:
    """Update a user. Omit a field (leave default) to keep its current value."""
    _log_tool("update_user", user_id=user_id, name=name, age=age, gender=gender)
    conn = _with_conn()
    try:
        conn.execute("BEGIN")
        row = dbmod.user_update(conn, user_id, name=name, age=age, gender=gender)
        conn.execute("COMMIT")
        if not row:
            return json.dumps({"ok": False, "error": "not_found", "user_id": user_id}, indent=2)
        return json.dumps({"ok": True, "user": row}, indent=2)
    except Exception as exc:  # noqa: BLE001
        conn.execute("ROLLBACK")
        return json.dumps({"ok": False, "error": str(exc)}, indent=2)


@mcp.tool()
def delete_user(user_id: int) -> str:
    """Delete a user and their tasks (cascade)."""
    _log_tool("delete_user", user_id=user_id)
    conn = _with_conn()
    try:
        conn.execute("BEGIN")
        ok = dbmod.user_delete(conn, user_id)
        conn.execute("COMMIT")
        if not ok:
            return json.dumps({"ok": False, "error": "not_found", "user_id": user_id}, indent=2)
        return json.dumps({"ok": True, "deleted_user_id": user_id}, indent=2)
    except Exception as exc:  # noqa: BLE001
        conn.execute("ROLLBACK")
        return json.dumps({"ok": False, "error": str(exc)}, indent=2)


@mcp.tool()
def create_task(user_id: int, name: str, description: str = "", status: str = "pending") -> str:
    """Create a task for ``user_id``. Default ``status`` is ``pending``."""
    _log_tool("create_task", user_id=user_id, name=name, description=description, status=status)
    conn = _with_conn()
    try:
        conn.execute("BEGIN")
        row = dbmod.task_create(conn, user_id, name, description, status)
        conn.execute("COMMIT")
        if not row:
            return json.dumps({"ok": False, "error": "user_not_found", "user_id": user_id}, indent=2)
        return json.dumps({"ok": True, "task": row}, indent=2)
    except Exception as exc:  # noqa: BLE001
        conn.execute("ROLLBACK")
        return json.dumps({"ok": False, "error": str(exc)}, indent=2)


@mcp.tool()
def get_task(task_id: int) -> str:
    """Read one task by ``task_id``."""
    _log_tool("get_task", task_id=task_id)
    conn = _with_conn()
    row = dbmod.task_get(conn, task_id)
    if not row:
        return json.dumps({"ok": False, "error": "not_found", "task_id": task_id}, indent=2)
    return json.dumps({"ok": True, "task": row}, indent=2)


@mcp.tool()
def list_tasks(user_id: int | None = None) -> str:
    """List tasks. Pass ``user_id`` to restrict to one user; omit for all tasks."""
    _log_tool("list_tasks", user_id=user_id)
    conn = _with_conn()
    rows = dbmod.task_list(conn, user_id)
    return json.dumps({"ok": True, "tasks": rows}, indent=2)


@mcp.tool()
def update_task(
    task_id: int,
    user_id: int | None = None,
    name: str | None = None,
    description: str | None = None,
    status: str | None = None,
) -> str:
    """Update a task. Omit fields to keep current values. ``user_id`` reassigns the owner."""
    _log_tool(
        "update_task",
        task_id=task_id,
        user_id=user_id,
        name=name,
        description=description,
        status=status,
    )
    conn = _with_conn()
    try:
        conn.execute("BEGIN")
        if not dbmod.task_get(conn, task_id):
            conn.execute("ROLLBACK")
            return json.dumps({"ok": False, "error": "task_not_found", "task_id": task_id}, indent=2)
        if user_id is not None and not dbmod.user_get(conn, user_id):
            conn.execute("ROLLBACK")
            return json.dumps({"ok": False, "error": "user_not_found", "user_id": user_id}, indent=2)
        row = dbmod.task_update(
            conn,
            task_id,
            name=name,
            description=description,
            status=status,
            user_id=user_id,
        )
        conn.execute("COMMIT")
        assert row is not None
        return json.dumps({"ok": True, "task": row}, indent=2)
    except Exception as exc:  # noqa: BLE001
        conn.execute("ROLLBACK")
        return json.dumps({"ok": False, "error": str(exc)}, indent=2)


@mcp.tool()
def delete_task(task_id: int) -> str:
    """Delete a task by ``task_id``."""
    _log_tool("delete_task", task_id=task_id)
    conn = _with_conn()
    try:
        conn.execute("BEGIN")
        ok = dbmod.task_delete(conn, task_id)
        conn.execute("COMMIT")
        if not ok:
            return json.dumps({"ok": False, "error": "not_found", "task_id": task_id}, indent=2)
        return json.dumps({"ok": True, "deleted_task_id": task_id}, indent=2)
    except Exception as exc:  # noqa: BLE001
        conn.execute("ROLLBACK")
        return json.dumps({"ok": False, "error": str(exc)}, indent=2)


@mcp.tool()
def get_tasks_for_users(user_ids: str, status: str = "") -> str:
    """
    Tasks for one or more users. ``user_ids`` is comma- or semicolon-separated integer ids.
    If ``status`` is non-empty, only tasks with that status are returned; otherwise all statuses.
    """
    _log_tool("get_tasks_for_users", user_ids=user_ids, status=status)
    conn = _with_conn()
    ids, bad = _parse_int_ids(user_ids)
    if bad and not ids:
        return json.dumps(
            {"ok": False, "error": "no_valid_user_ids", "unknown_tokens": bad},
            indent=2,
        )
    st = status.strip() or None
    by_user = dbmod.tasks_for_users(conn, ids, status=st)
    payload = {
        "ok": True,
        "tasks_by_user_id": {str(k): v for k, v in by_user.items()},
    }
    if bad:
        payload["skipped_tokens"] = bad
    return json.dumps(payload, indent=2)


def main() -> None:
    _ensure_logging()
    conn = _with_conn()
    n_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    n_tasks = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
    conn.close()
    _LOG.info(
        "users_tasks_mcp starting (stdio MCP) db=%s users=%s tasks=%s",
        dbmod.db_path(),
        n_users,
        n_tasks,
    )
    _flush_log_handlers()
    mcp.run()


if __name__ == "__main__":
    main()
