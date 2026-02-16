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
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS table_schemas (
            table_name TEXT PRIMARY KEY,
            fields_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT
        )
        """
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
    """Return distinct table names (from records and from table_schemas)."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT DISTINCT table_name FROM records ORDER BY table_name"
    ).fetchall()
    schema_rows = conn.execute(
        "SELECT table_name FROM table_schemas ORDER BY table_name"
    ).fetchall()
    conn.close()
    names = {r[0] for r in rows}
    names.update(r[0] for r in schema_rows)
    return sorted(names)


def create_table_schema(table_name: str, fields: list[dict]) -> dict:
    """
    Register a new table with the given fields. fields is a list of {"name": str, "type": str} (type defaults to "text").
    Does not create a real SQL table; records for this table_name still go into the records table.
    """
    now = datetime.utcnow().isoformat() + "Z"
    normalized = []
    for f in fields:
        if isinstance(f, dict):
            normalized.append({"name": str(f.get("name", f.get("field", ""))).strip(), "type": str(f.get("type", "text")).lower() or "text"})
        else:
            normalized.append({"name": str(f).strip(), "type": "text"})
    normalized = [x for x in normalized if x["name"]]
    if not normalized:
        raise ValueError("At least one field name is required")
    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO table_schemas (table_name, fields_json, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (table_name.strip(), json.dumps(normalized), now, now),
    )
    conn.commit()
    conn.close()
    return {"table_name": table_name, "fields": normalized, "created_at": now}


def get_table_schema(table_name: str) -> list[dict] | None:
    """Return the schema (list of {name, type}) for a table, or None if not defined."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT fields_json FROM table_schemas WHERE table_name = ?",
        (table_name,),
    ).fetchone()
    conn.close()
    if not row:
        return None
    return json.loads(row[0])


def alter_table_schema(table_name: str, fields: list[dict]) -> dict:
    """Update the table schema (replace with new field list). Table must exist."""
    existing = get_table_schema(table_name)
    if not existing:
        raise ValueError(f"Table '{table_name}' does not exist. Create it first with create_table.")
    now = datetime.utcnow().isoformat() + "Z"
    normalized = []
    for f in fields:
        if isinstance(f, dict):
            normalized.append({"name": str(f.get("name", f.get("field", ""))).strip(), "type": str(f.get("type", "text")).lower() or "text"})
        else:
            normalized.append({"name": str(f).strip(), "type": "text"})
    normalized = [x for x in normalized if x["name"]]
    if not normalized:
        raise ValueError("At least one field name is required")
    conn = _get_conn()
    conn.execute(
        "UPDATE table_schemas SET fields_json = ?, updated_at = ? WHERE table_name = ?",
        (json.dumps(normalized), now, table_name),
    )
    conn.commit()
    conn.close()
    return {"table_name": table_name, "fields": normalized, "updated_at": now}


def drop_table(table_name: str) -> dict:
    """Delete all records for this table and remove its schema. Returns count of records deleted."""
    conn = _get_conn()
    cur = conn.execute("DELETE FROM records WHERE table_name = ?", (table_name,))
    records_deleted = cur.rowcount
    conn.execute("DELETE FROM table_schemas WHERE table_name = ?", (table_name,))
    conn.commit()
    conn.close()
    return {"table_name": table_name, "records_deleted": records_deleted, "dropped": True}
