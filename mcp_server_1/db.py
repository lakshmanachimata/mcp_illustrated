"""Local SQLite database layer for CRUD operations."""
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "mcp_server_1.db"

# Internal tables we never treat as user "real" tables
_INTERNAL_TABLES = {"records", "table_schemas", "sqlite_sequence"}

# Map schema type to SQLite type
_SQLITE_TYPE = {"text": "TEXT", "integer": "INTEGER", "int": "INTEGER", "real": "REAL", "blob": "BLOB"}


def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _real_table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    """Return True if a SQLite table with this name exists (and is not internal)."""
    if table_name in _INTERNAL_TABLES:
        return False
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _safe_identifier(name: str) -> str:
    """Return name safe for use as SQL identifier (no quotes if alphanumeric/underscore)."""
    if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name):
        return name
    return f'"{name}"'


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


def _uses_real_table(conn: sqlite3.Connection, table_name: str) -> bool:
    """True if this table has a real SQLite table with separate columns."""
    return _real_table_exists(conn, table_name)


def create_record(table_name: str, data: dict) -> dict:
    """Insert a record. For schema tables: separate columns; else JSON in records table."""
    now = datetime.utcnow().isoformat() + "Z"
    conn = _get_conn()
    if _uses_real_table(conn, table_name):
        schema = get_table_schema(table_name)
        if not schema:
            conn.close()
            raise ValueError(f"Table '{table_name}' has no schema")
        col_names = [f["name"] for f in schema]
        placeholders = []
        values = []
        for k in col_names:
            if k in data:
                placeholders.append("?")
                values.append(data[k])
            else:
                placeholders.append("?")
                values.append(None)
        values.append(now)
        values.append(now)
        cols = [_safe_identifier(c) for c in col_names] + ["created_at", "updated_at"]
        conn.execute(
            f"INSERT INTO {_safe_identifier(table_name)} ({', '.join(cols)}) VALUES ({', '.join(placeholders)}, ?, ?)",
            values,
        )
        row_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.commit()
        conn.close()
        return get_record(table_name, row_id)
    conn.execute(
        "INSERT INTO records (table_name, data, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (table_name, json.dumps(data), now, now),
    )
    row_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.commit()
    conn.close()
    return get_record(table_name, row_id)


def _row_to_record(table_name: str, row: sqlite3.Row, schema: list[dict] | None) -> dict:
    """Build {id, table_name, data, created_at, updated_at} from a real-table row."""
    row_dict = dict(row)
    data = {}
    for k in row_dict:
        if k not in ("id", "created_at", "updated_at"):
            data[k] = row_dict[k]
    return {
        "id": row_dict["id"],
        "table_name": table_name,
        "data": data,
        "created_at": row_dict.get("created_at"),
        "updated_at": row_dict.get("updated_at"),
    }


def get_record(table_name: str, record_id: int) -> dict | None:
    """Get a single record by id."""
    conn = _get_conn()
    if _uses_real_table(conn, table_name):
        row = conn.execute(
            f"SELECT * FROM {_safe_identifier(table_name)} WHERE id = ?",
            (record_id,),
        ).fetchone()
        conn.close()
        if not row:
            return None
        return _row_to_record(table_name, row, None)
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
    if _uses_real_table(conn, table_name):
        rows = conn.execute(
            f"SELECT * FROM {_safe_identifier(table_name)} ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()
        return [_row_to_record(table_name, row, None) for row in rows]
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


def find_records_by_field(table_name: str, field_name: str, field_value: str | int | float | bool) -> list[dict]:
    """Return records in the table whose data has field_name equal to field_value (e.g. find user by name)."""
    records = list_records(table_name, limit=500)
    out = []
    for r in records:
        data = r.get("data") or {}
        val = data.get(field_name)
        if val == field_value or (isinstance(field_value, str) and str(val).strip().lower() == str(field_value).strip().lower()):
            out.append(r)
    return out


def find_update_and_get(table_name: str, field_name: str, field_value: str | int | float | bool, update_data: dict) -> dict | None:
    """Find first record where data[field_name]==field_value, merge update_data into it, return the updated record. Record data accepts any fields (schema is informational)."""
    matches = find_records_by_field(table_name, field_name, field_value)
    if not matches:
        return None
    rec = matches[0]
    rid = rec["id"]
    updated = update_record(table_name, rid, update_data)
    return updated


def update_record(table_name: str, record_id: int, data: dict) -> dict | None:
    """Update a record. Merges with existing data (shallow merge)."""
    existing = get_record(table_name, record_id)
    if not existing:
        return None
    merged = {**existing["data"], **data}
    now = datetime.utcnow().isoformat() + "Z"
    conn = _get_conn()
    if _uses_real_table(conn, table_name):
        schema = get_table_schema(table_name)
        if not schema:
            conn.close()
            return None
        set_parts = []
        values = []
        for f in schema:
            name = f["name"]
            if name in merged:
                set_parts.append(f"{_safe_identifier(name)} = ?")
                values.append(merged[name])
        set_parts.append("updated_at = ?")
        values.append(now)
        values.append(record_id)
        conn.execute(
            f"UPDATE {_safe_identifier(table_name)} SET {', '.join(set_parts)} WHERE id = ?",
            values,
        )
        conn.commit()
        conn.close()
        return get_record(table_name, record_id)
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
    if _uses_real_table(conn, table_name):
        cur = conn.execute(
            f"DELETE FROM {_safe_identifier(table_name)} WHERE id = ?",
            (record_id,),
        )
        conn.commit()
        deleted = cur.rowcount > 0
        conn.close()
        return deleted
    cur = conn.execute(
        "DELETE FROM records WHERE table_name = ? AND id = ?",
        (table_name, record_id),
    )
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted


def list_tables() -> list[str]:
    """Return distinct table names (from records, table_schemas, and real SQLite tables)."""
    conn = _get_conn()
    names = set()
    for row in conn.execute("SELECT DISTINCT table_name FROM records").fetchall():
        names.add(row[0])
    for row in conn.execute("SELECT table_name FROM table_schemas").fetchall():
        names.add(row[0])
    for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    ).fetchall():
        if row[0] not in _INTERNAL_TABLES:
            names.add(row[0])
    conn.close()
    return sorted(names)


def create_table_schema(table_name: str, fields: list[dict]) -> dict:
    """
    Create a real SQLite table with one column per field (no fields_json).
    fields is a list of {"name": str, "type": str} (e.g. [{"name": "Name", "type": "text"}, ...]).
    All CRUD queries use these columns.
    """
    table_name = table_name.strip()
    if not table_name or table_name in _INTERNAL_TABLES:
        raise ValueError("Invalid or reserved table name")
    now = datetime.utcnow().isoformat() + "Z"
    normalized = []
    for f in fields:
        if isinstance(f, dict):
            name = str(f.get("name", f.get("field", ""))).strip()
            typ = (str(f.get("type", "text")).lower() or "text").strip()
            if name:
                normalized.append({"name": name, "type": typ})
        else:
            s = str(f).strip()
            if s:
                normalized.append({"name": s, "type": "text"})
    if not normalized:
        raise ValueError("At least one field name is required")
    conn = _get_conn()
    # Replace existing table if present so we always have correct column layout (one column per field)
    if _real_table_exists(conn, table_name):
        conn.execute(f"DROP TABLE {_safe_identifier(table_name)}")
    # Build columns: id, then one column per schema field (Name, Email, DOB, etc.), then created_at, updated_at
    cols = ["id INTEGER PRIMARY KEY AUTOINCREMENT"]
    for f in normalized:
        sql_type = _SQLITE_TYPE.get(f["type"], "TEXT")
        cols.append(f"{_safe_identifier(f['name'])} {sql_type}")
    cols.append("created_at TEXT NOT NULL")
    cols.append("updated_at TEXT")
    sql = f"CREATE TABLE {_safe_identifier(table_name)} ({', '.join(cols)})"
    conn.execute(sql)
    # Store schema in metadata table only (table_schemas.fields_json); data table has no fields_json column
    conn.execute(
        "INSERT OR REPLACE INTO table_schemas (table_name, fields_json, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (table_name, json.dumps([{"name": f["name"], "type": f["type"]} for f in normalized]), now, now),
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
    """Update the table schema (replace with new field list). Recreates real table with new columns."""
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
    if _uses_real_table(conn, table_name):
        temp_name = f"_alter_{table_name}"
        cols = ["id INTEGER PRIMARY KEY AUTOINCREMENT"]
        for f in normalized:
            sql_type = _SQLITE_TYPE.get(f["type"], "TEXT")
            cols.append(f"{_safe_identifier(f['name'])} {sql_type}")
        cols.append("created_at TEXT NOT NULL")
        cols.append("updated_at TEXT")
        conn.execute(f"CREATE TABLE {_safe_identifier(temp_name)} ({', '.join(cols)})")
        old_cols = [f["name"] for f in existing]
        new_cols = [f["name"] for f in normalized]
        common = [c for c in new_cols if c in old_cols]
        common_sql = ", ".join(_safe_identifier(c) for c in common)
        if common_sql:
            conn.execute(
                f"INSERT INTO {_safe_identifier(temp_name)} (id, {common_sql}, created_at, updated_at) "
                f"SELECT id, {common_sql}, created_at, updated_at FROM {_safe_identifier(table_name)}"
            )
        else:
            conn.execute(
                f"INSERT INTO {_safe_identifier(temp_name)} (id, created_at, updated_at) "
                f"SELECT id, created_at, updated_at FROM {_safe_identifier(table_name)}"
            )
        conn.execute(f"DROP TABLE {_safe_identifier(table_name)}")
        conn.execute(f"ALTER TABLE {_safe_identifier(temp_name)} RENAME TO {_safe_identifier(table_name)}")
    conn.execute(
        "UPDATE table_schemas SET fields_json = ?, updated_at = ? WHERE table_name = ?",
        (json.dumps(normalized), now, table_name),
    )
    conn.commit()
    conn.close()
    return {"table_name": table_name, "fields": normalized, "updated_at": now}


def drop_table(table_name: str) -> dict:
    """Drop the table (real or legacy), remove schema. Returns count of records deleted for legacy tables."""
    conn = _get_conn()
    records_deleted = 0
    if _uses_real_table(conn, table_name):
        cur = conn.execute(f"SELECT COUNT(*) FROM {_safe_identifier(table_name)}")
        records_deleted = cur.fetchone()[0]
        conn.execute(f"DROP TABLE {_safe_identifier(table_name)}")
    else:
        cur = conn.execute("DELETE FROM records WHERE table_name = ?", (table_name,))
        records_deleted = cur.rowcount
    conn.execute("DELETE FROM table_schemas WHERE table_name = ?", (table_name,))
    conn.commit()
    conn.close()
    return {"table_name": table_name, "records_deleted": records_deleted, "dropped": True}
