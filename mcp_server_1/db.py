"""Local SQLite database layer for CRUD operations."""
import json
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "mcp_server_1.db"


def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = _get_conn()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            table_name TEXT NOT NULL,
            data TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_records_table ON records(table_name)"
    )
    conn.commit()
    conn.close()


def create_record(table_name: str, data: dict) -> dict:
    """Insert a record. data is stored as JSON."""
    now = datetime.utcnow().isoformat() + "Z"
    conn = _get_conn()
    conn.execute(
        "INSERT INTO records (table_name, data, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (table_name, json.dumps(data), now, now),
    )
    row_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.commit()
    conn.close()
    return get_record(table_name, row_id)


def get_record(table_name: str, record_id: int) -> dict | None:
    """Get a single record by id."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT id, table_name, data, created_at, updated_at FROM records WHERE table_name = ? AND id = ?",
        (table_name, record_id),
    ).fetchone()
    conn.close()
    if not row:
        return None
    out = dict(row)
    out["data"] = json.loads(out["data"])
    return out


def list_records(table_name: str, limit: int = 100) -> list[dict]:
    """List records for a table, newest first."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, table_name, data, created_at, updated_at FROM records WHERE table_name = ? ORDER BY id DESC LIMIT ?",
        (table_name, limit),
    ).fetchall()
    conn.close()
    result = []
    for row in rows:
        r = dict(row)
        r["data"] = json.loads(r["data"])
        result.append(r)
    return result


def update_record(table_name: str, record_id: int, data: dict) -> dict | None:
    """Update a record. Merges with existing data (shallow merge)."""
    existing = get_record(table_name, record_id)
    if not existing:
        return None
    merged = {**existing["data"], **data}
    now = datetime.utcnow().isoformat() + "Z"
    conn = _get_conn()
    conn.execute(
        "UPDATE records SET data = ?, updated_at = ? WHERE table_name = ? AND id = ?",
        (json.dumps(merged), now, table_name, record_id),
    )
    conn.commit()
    conn.close()
    return get_record(table_name, record_id)


def delete_record(table_name: str, record_id: int) -> bool:
    """Delete a record. Returns True if deleted."""
    conn = _get_conn()
    cur = conn.execute(
        "DELETE FROM records WHERE table_name = ? AND id = ?",
        (table_name, record_id),
    )
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted


def list_tables() -> list[str]:
    """Return distinct table names."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT DISTINCT table_name FROM records ORDER BY table_name"
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]
