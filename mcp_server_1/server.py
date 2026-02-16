"""
MCP server exposing tools as capabilities for the LLM service.
Tools are annotated so the agent can discover and run queries against the local database.
"""
import json
import os
import re
from typing import Any

from mcp.server.fastmcp import FastMCP

import db

# Initialize DB on import
db.init_db()

# Port 8001 by default so backend (LLM service) can use 8000
_port = int(os.environ.get("MCP_PORT") or os.environ.get("FASTMCP_PORT") or "8001")
mcp = FastMCP(
    "Local DB",
    instructions="Database tools exposed as capabilities for the LLM: run queries to create, alter, drop tables (with fields from prompt); create, read, update, delete records; list tables and get table schema.",
    json_response=True,
    port=_port,
)


@mcp.tool(description="Capability: Run a query to create a new record in a table. Args: table_name, data (JSON object of fields).")
def create_record(table_name: str, data: dict[str, Any]) -> dict:
    """Create a new record in the given table. data is a JSON object of fields (e.g. {"name": "Alice", "email": "alice@example.com"})."""
    return db.create_record(table_name, data)


@mcp.tool(description="Capability: Run a query to get a single record by table name and id. Returns null if not found.")
def get_record(table_name: str, record_id: int) -> dict | None:
    """Get a single record by table name and id. Returns null if not found."""
    return db.get_record(table_name, record_id)


@mcp.tool(description="Capability: Run a query to list records in a table (newest first). Use to list, show, or get all records.")
def list_records(table_name: str, limit: int = 100) -> list[dict]:
    """List records in a table, newest first. Use when the user asks to list, show, or get all records."""
    return db.list_records(table_name, limit=limit)


@mcp.tool(description="Capability: Find records in a table where a field equals a value (e.g. find user by name: table_name='users', field_name='name', field_value='lakshmana'). Use when the user asks to find/update/get a user or record by name or other field.")
def find_records_by_field(table_name: str, field_name: str, field_value: str | int | float | bool) -> list[dict]:
    """Find records where data[field_name] == field_value. E.g. find user with name 'lakshmana': table_name='users', field_name='name', field_value='lakshmana'. Returns list of full records (id, data, etc.)."""
    return db.find_records_by_field(table_name, field_name, field_value)


@mcp.tool(
    description="Capability: Find one record by a field value, update it with given data, and return the updated record. Use for 'update user X and get user details' (e.g. table_name='users', field_name='name', field_value='lakshmana', update_data={'status': 'inactive'}). Record data can contain any fields; table schema is informational only."
)
def find_update_and_get_record(
    table_name: str, field_name: str, field_value: str | int | float | bool, update_data: dict[str, Any]
) -> dict | None:
    """Find the first record where data[field_name]==field_value, merge update_data into it, return the updated record. E.g. set status inactive and get user: table_name='users', field_name='name', field_value='lakshmana', update_data={'status': 'inactive'}."""
    return db.find_update_and_get(table_name, field_name, field_value, update_data)


@mcp.tool(description="Capability: Run a query to update an existing record. data is merged with existing fields.")
def update_record(table_name: str, record_id: int, data: dict[str, Any]) -> dict | None:
    """Update an existing record. data is merged with existing fields. Returns null if record not found."""
    return db.update_record(table_name, record_id, data)


@mcp.tool(description="Capability: Run a query to delete a record by table name and id. Returns success status.")
def delete_record(table_name: str, record_id: int) -> dict:
    """Delete a record by table name and id. Returns success status."""
    deleted = db.delete_record(table_name, record_id)
    return {"deleted": deleted, "table_name": table_name, "record_id": record_id}


@mcp.tool(description="Capability: Run a query to list all table names that have at least one record.")
def list_tables() -> list[str]:
    """List all table names that have at least one record. Use when the user asks what tables exist or to list tables."""
    return db.list_tables()


def _parse_fields(fields: str | list[str] | list[dict[str, Any]]) -> list[dict[str, str]]:
    """Parse fields: JSON array string, or 'name, email, age', or ['name','email'] or [{'name':'Name','type':'text'}]."""
    # If string looks like JSON array of {name, type}, parse it so we get one column per field
    if isinstance(fields, str):
        s = fields.strip()
        if s.startswith("[") and (s.endswith("]") or "]" in s):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return _parse_fields(parsed)
            except json.JSONDecodeError:
                pass
    if isinstance(fields, list):
        if not fields:
            return []
        if isinstance(fields[0], dict):
            return [{"name": str(f.get("name", f.get("field", ""))).strip(), "type": str(f.get("type", "text")).lower() or "text"} for f in fields]
        return [{"name": str(f).strip(), "type": "text"} for f in fields if str(f).strip()]
    s = (fields or "").strip()
    if not s:
        return []
    out = []
    for part in re.split(r"[,;]", s):
        part = part.strip()
        if not part:
            continue
        # "name text" or "name" or "name: text"
        if re.match(r"^\w+\s+(?:text|integer|int|real|blob)$", part, re.I):
            name, _, typ = part.partition(" ")
            out.append({"name": name.strip(), "type": typ.strip().lower()})
        elif ":" in part:
            name, _, typ = part.partition(":")
            out.append({"name": name.strip(), "type": (typ.strip() or "text").lower()})
        else:
            out.append({"name": part, "type": "text"})
    return out


@mcp.tool(description="Capability: Run a query to create a new table with the given fields (e.g. fields='name, email, age' or list of field names). Fields can be supplied from the user prompt.")
def create_table(table_name: str, fields: str | list[str] | list[dict[str, Any]]) -> dict:
    """Create a new table with the given fields. fields can be comma-separated string (e.g. 'name, email, age'), list of names, or list of {name, type}. Types default to text."""
    table_name = table_name.strip()
    if not table_name:
        return {"success": False, "error": "table_name is required"}
    parsed = _parse_fields(fields)
    if not parsed:
        return {"success": False, "error": "At least one field is required (e.g. 'name, email, age')"}
    try:
        result = db.create_table_schema(table_name, parsed)
        return {"success": True, **result}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Capability: Run a query to alter an existing table's schema (replace with new fields). Supply new field list from prompt.")
def alter_table(table_name: str, fields: str | list[str] | list[dict[str, Any]]) -> dict:
    """Alter a table's schema: replace with the new list of fields. fields can be comma-separated string or list (e.g. 'name, email, phone' to add phone)."""
    table_name = table_name.strip()
    if not table_name:
        return {"success": False, "error": "table_name is required"}
    parsed = _parse_fields(fields)
    if not parsed:
        return {"success": False, "error": "At least one field is required"}
    try:
        result = db.alter_table_schema(table_name, parsed)
        return {"success": True, **result}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Capability: Run a query to delete (drop) a table and all its records. Use when user asks to drop or delete a table.")
def drop_table(table_name: str) -> dict:
    """Drop a table: remove its schema and delete all records in that table."""
    table_name = table_name.strip()
    if not table_name:
        return {"success": False, "error": "table_name is required"}
    try:
        result = db.drop_table(table_name)
        return {"success": True, **result}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool(description="Capability: Run a query to get the schema (field names and types) of a table, if defined.")
def get_table_schema(table_name: str) -> dict | None:
    """Get the schema (list of fields with names and types) for a table. Returns null if table has no defined schema."""
    schema = db.get_table_schema(table_name.strip())
    if schema is None:
        return None
    return {"table_name": table_name, "fields": schema}


@mcp.tool(description="Capability: Run a query to add survey questions to the database. Pass newline-separated or numbered list.")
def add_survey_questions(questions_text: str) -> dict:
    """Add one or more survey questions to the database. Pass questions as newline-separated or numbered list (e.g. '1. How old are you? 2. What is your gender?'). Each line/item is stored as one record in table survey_questions."""
    questions = _split_survey_questions(questions_text)
    if not questions:
        return {"success": False, "error": "No questions found in text"}
    created = []
    for q in questions:
        rec = db.create_record("survey_questions", {"text": q})
        created.append(rec)
    return {"success": True, "created": len(created), "records": created}


def _split_survey_questions(text: str) -> list[str]:
    """Split text into individual questions (by newlines or numbered items)."""
    if not text or not text.strip():
        return []
    text = text.strip()
    # Split by newlines first
    parts = re.split(r"\n+", text)
    questions = []
    for p in parts:
        p = p.strip()
        # Remove leading numbering: "1. ", "1) ", "2. ", etc.
        p = re.sub(r"^\s*\d+[.)]\s*", "", p).strip()
        if p:
            questions.append(p)
    return questions if questions else [text]


def _parse_instruction(instruction: str) -> dict | None:
    """
    Parse a simple natural-language instruction and return an action dict.
    Used when the user explicitly states what to do in a prompt.
    """
    raw = instruction.strip()
    instruction_lower = raw.lower()
    # add survey questions / add questions to db / add to database (capture rest as content)
    m = re.search(r"(?:add|store|save)\s+(?:these?\s+)?(?:survey\s+)?questions?\s*(?::|to\s+(?:the\s+)?(?:db|database))?\s*(.*)", instruction_lower, re.DOTALL | re.I)
    if m:
        content = (m.group(1) or raw).strip()
        if not content and ":" in raw:
            content = raw.split(":", 1)[-1].strip()
        if not content:
            content = raw
        if content:
            return {"action": "add_survey_questions", "content": content}
    m = re.search(r"(?:add|store|save)\s+(?:to\s+(?:the\s+)?(?:db|database)|in\s+(?:mcp\s+)?(?:server\s+)?db)\s*(?::)?\s*(.*)", instruction_lower, re.DOTALL | re.I)
    if m:
        content = (m.group(1) or raw).strip()
        if not content and ":" in raw:
            content = raw.split(":", 1)[-1].strip()
        if content:
            return {"action": "add_survey_questions", "content": content}
    # create / add / insert
    m = re.search(r"(?:add|create|insert)\s+(?:a\s+)?(?:record\s+)?(?:in\s+)?(?:table\s+)?['\"]?(\w+)['\"]?\s*(?:with\s+)?(.*)", instruction_lower, re.DOTALL | re.I)
    if m:
        table = m.group(1)
        rest = m.group(2).strip()
        data = {}
        if rest:
            for part in re.split(r",\s*(?=(?:[^\"']*[\"'][^\"']*[\"'])*[^\"']*$)", rest):
                kv = re.match(r"(\w+)\s*[:=]\s*['\"]?([^'\"]+)['\"]?", part.strip())
                if kv:
                    data[kv.group(1)] = kv.group(2).strip()
        return {"action": "create", "table_name": table, "data": data}

    # list / show / get all
    m = re.search(r"(?:list|show|get\s+all|fetch\s+all)\s+(?:records?\s+)?(?:from\s+)?(?:table\s+)?['\"]?(\w+)['\"]?", instruction)
    if m:
        return {"action": "list", "table_name": m.group(1)}

    # get one
    m = re.search(r"(?:get|fetch|read|show)\s+(?:record\s+)?(?:id\s+)?(\d+)\s+(?:from\s+)?(?:table\s+)?['\"]?(\w+)['\"]?", instruction)
    if m:
        return {"action": "get", "table_name": m.group(2), "record_id": int(m.group(1))}

    # update
    m = re.search(r"(?:update|change|edit)\s+(?:record\s+)?(?:id\s+)?(\d+)\s+(?:in\s+)?(?:table\s+)?['\"]?(\w+)['\"]?\s*(?:set\s+)?(.*)", instruction, re.DOTALL | re.I)
    if m:
        table = m.group(2)
        rid = int(m.group(1))
        rest = m.group(3).strip()
        data = {}
        if rest:
            for part in re.split(r",\s*", rest):
                kv = re.match(r"(\w+)\s*[:=]\s*['\"]?([^'\"]+)['\"]?", part.strip())
                if kv:
                    data[kv.group(1)] = kv.group(2).strip()
        return {"action": "update", "table_name": table, "record_id": rid, "data": data}

    # delete / remove
    m = re.search(r"(?:delete|remove)\s+(?:record\s+)?(?:id\s+)?(\d+)\s+(?:from\s+)?(?:table\s+)?['\"]?(\w+)['\"]?", instruction)
    if m:
        return {"action": "delete", "table_name": m.group(2), "record_id": int(m.group(1))}

    # create table ... with fields ...
    m = re.search(r"(?:create|add)\s+(?:a\s+)?table\s+['\"]?(\w+)['\"]?\s+(?:with\s+)?(?:fields?\s+)?(.*)", instruction_lower, re.DOTALL | re.I)
    if m:
        table = m.group(1)
        rest = (m.group(2) or "").strip()
        if rest:
            return {"action": "create_table", "table_name": table, "fields": rest}
        return {"action": "create_table", "table_name": table, "fields": ""}

    # alter table ... (add/drop columns or set new fields)
    m = re.search(r"alter\s+table\s+['\"]?(\w+)['\"]?\s+(?:set\s+)?(?:fields?\s+)?(.*)", instruction_lower, re.DOTALL | re.I)
    if m:
        return {"action": "alter_table", "table_name": m.group(1), "fields": (m.group(2) or "").strip()}

    # drop / delete table
    m = re.search(r"(?:drop|delete|remove)\s+(?:the\s+)?table\s+['\"]?(\w+)['\"]?", instruction_lower)
    if m:
        return {"action": "drop_table", "table_name": m.group(1)}

    return None


@mcp.tool(description="Capability: Run a natural-language query to perform one database operation (add, list, get, update, delete). Use when the user states what to do in one sentence.")
def execute_instruction(instruction: str) -> dict:
    """
    Execute a database operation from a short natural-language instruction.
    Use this when the user explicitly states what to do in their prompt (e.g. 'add a user named John', 'list all items', 'delete record 5 from users').
    Supported patterns: add/create in <table> [with key: value, ...]; list/show from <table>; get record <id> from <table>; update record <id> in <table> set key: value; delete record <id> from <table>.
    """
    parsed = _parse_instruction(instruction)
    if not parsed:
        return {"success": False, "error": "Could not parse instruction. Use create_record, list_records, get_record, update_record, or delete_record with explicit parameters instead."}

    action = parsed["action"]
    table_name = parsed.get("table_name", "default")

    try:
        if action == "add_survey_questions":
            content = parsed.get("content", "")
            questions = _split_survey_questions(content)
            created = []
            for q in questions:
                rec = db.create_record("survey_questions", {"text": q})
                created.append(rec)
            return {"success": True, "action": "add_survey_questions", "result": {"created": len(created), "records": created}}
        if action == "create":
            out = db.create_record(table_name, parsed.get("data", {}))
            return {"success": True, "action": "create", "result": out}
        if action == "list":
            out = db.list_records(table_name)
            return {"success": True, "action": "list", "result": out}
        if action == "get":
            out = db.get_record(table_name, parsed["record_id"])
            return {"success": True, "action": "get", "result": out}
        if action == "update":
            out = db.update_record(table_name, parsed["record_id"], parsed.get("data", {}))
            return {"success": True, "action": "update", "result": out}
        if action == "delete":
            deleted = db.delete_record(table_name, parsed["record_id"])
            return {"success": True, "action": "delete", "result": {"deleted": deleted}}
        if action == "create_table":
            fields_str = parsed.get("fields", "")
            parsed_fields = _parse_fields(fields_str)
            if not parsed_fields and fields_str:
                parsed_fields = _parse_fields([x.strip() for x in re.split(r"[,;]", fields_str) if x.strip()])
            if not parsed_fields:
                return {"success": False, "error": "create_table requires fields (e.g. 'name, email, age')"}
            out = db.create_table_schema(table_name, parsed_fields)
            return {"success": True, "action": "create_table", "result": out}
        if action == "alter_table":
            fields_str = parsed.get("fields", "")
            parsed_fields = _parse_fields(fields_str)
            if not parsed_fields:
                return {"success": False, "error": "alter_table requires new field list"}
            out = db.alter_table_schema(table_name, parsed_fields)
            return {"success": True, "action": "alter_table", "result": out}
        if action == "drop_table":
            out = db.drop_table(table_name)
            return {"success": True, "action": "drop_table", "result": out}
    except Exception as e:
        return {"success": False, "error": str(e)}

    return {"success": False, "error": "Unknown action"}


@mcp.prompt()
def db_instructions() -> str:
    """Use when the user asks to manage data or schema: create/alter/drop tables (with fields from prompt), or create/read/update/delete records."""
    return (
        "When the user asks to manage the database, use the right tool. For tables: create_table (with fields from the prompt, e.g. 'name, email, age'), "
        "alter_table (new field list), drop_table. For records: create_record, get_record, list_records, update_record, delete_record. "
        "Or use execute_instruction with a short natural-language sentence. Act as specified in the user's prompt."
    )


if __name__ == "__main__":
    # Port: set MCP_PORT or FASTMCP_PORT (e.g. 8001) to avoid conflict with backend on 8000
    mcp.run(transport="streamable-http")
