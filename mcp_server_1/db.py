"""SQLite persistence for users and tasks (one user, many tasks)."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "users_tasks.db"


def db_path() -> Path:
    raw = os.environ.get("USERS_TASKS_MCP_DB", "").strip()
    return Path(raw).expanduser() if raw else DEFAULT_DB_PATH


def connect() -> sqlite3.Connection:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER NOT NULL,
            gender TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'pending'
        );

        CREATE INDEX IF NOT EXISTS idx_tasks_user_id ON tasks(user_id);
        CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
        """
    )


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return dict(row)


def user_create(conn: sqlite3.Connection, name: str, age: int, gender: str) -> dict[str, Any]:
    cur = conn.execute(
        "INSERT INTO users (name, age, gender) VALUES (?, ?, ?)",
        (name.strip(), age, gender.strip()),
    )
    uid = int(cur.lastrowid)
    r = conn.execute("SELECT * FROM users WHERE id = ?", (uid,)).fetchone()
    assert r is not None
    return row_to_dict(r)  # type: ignore[return-value]


def user_get(conn: sqlite3.Connection, user_id: int) -> dict[str, Any] | None:
    r = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return row_to_dict(r)


def user_list(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT * FROM users ORDER BY id").fetchall()
    return [dict(x) for x in rows]


def user_update(
    conn: sqlite3.Connection,
    user_id: int,
    *,
    name: str | None = None,
    age: int | None = None,
    gender: str | None = None,
) -> dict[str, Any] | None:
    existing = user_get(conn, user_id)
    if not existing:
        return None
    new_name = name.strip() if name is not None else existing["name"]
    new_age = age if age is not None else existing["age"]
    new_gender = gender.strip() if gender is not None else existing["gender"]
    conn.execute(
        "UPDATE users SET name = ?, age = ?, gender = ? WHERE id = ?",
        (new_name, new_age, new_gender, user_id),
    )
    return user_get(conn, user_id)


def user_delete(conn: sqlite3.Connection, user_id: int) -> bool:
    cur = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    return cur.rowcount > 0


def task_create(
    conn: sqlite3.Connection,
    user_id: int,
    name: str,
    description: str = "",
    status: str = "pending",
) -> dict[str, Any] | None:
    if user_get(conn, user_id) is None:
        return None
    cur = conn.execute(
        "INSERT INTO tasks (user_id, name, description, status) VALUES (?, ?, ?, ?)",
        (user_id, name.strip(), description.strip(), status.strip() or "pending"),
    )
    tid = int(cur.lastrowid)
    r = conn.execute("SELECT * FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert r is not None
    return row_to_dict(r)  # type: ignore[return-value]


def task_get(conn: sqlite3.Connection, task_id: int) -> dict[str, Any] | None:
    r = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    return row_to_dict(r)


def task_list(conn: sqlite3.Connection, user_id: int | None = None) -> list[dict[str, Any]]:
    if user_id is None:
        rows = conn.execute("SELECT * FROM tasks ORDER BY id").fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM tasks WHERE user_id = ? ORDER BY id",
            (user_id,),
        ).fetchall()
    return [dict(x) for x in rows]


def task_update(
    conn: sqlite3.Connection,
    task_id: int,
    *,
    name: str | None = None,
    description: str | None = None,
    status: str | None = None,
    user_id: int | None = None,
) -> dict[str, Any] | None:
    existing = task_get(conn, task_id)
    if not existing:
        return None
    new_user = user_id if user_id is not None else existing["user_id"]
    if user_get(conn, int(new_user)) is None:
        return None
    new_name = name.strip() if name is not None else existing["name"]
    new_desc = description.strip() if description is not None else existing["description"]
    new_status = status.strip() if status is not None else existing["status"]
    conn.execute(
        "UPDATE tasks SET user_id = ?, name = ?, description = ?, status = ? WHERE id = ?",
        (new_user, new_name, new_desc, new_status or "pending", task_id),
    )
    return task_get(conn, task_id)


def task_delete(conn: sqlite3.Connection, task_id: int) -> bool:
    cur = conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
    return cur.rowcount > 0


def tasks_for_users(
    conn: sqlite3.Connection,
    user_ids: list[int],
    status: str | None = None,
) -> dict[int, list[dict[str, Any]]]:
    if not user_ids:
        return {}
    placeholders = ",".join("?" * len(user_ids))
    params: list[Any] = list(user_ids)
    sql = f"SELECT * FROM tasks WHERE user_id IN ({placeholders})"
    if status is not None and status.strip():
        sql += " AND status = ?"
        params.append(status.strip())
    sql += " ORDER BY user_id, id"
    rows = conn.execute(sql, params).fetchall()
    out: dict[int, list[dict[str, Any]]] = {uid: [] for uid in user_ids}
    for row in rows:
        d = dict(row)
        uid = int(d["user_id"])
        if uid in out:
            out[uid].append(d)
    return out
