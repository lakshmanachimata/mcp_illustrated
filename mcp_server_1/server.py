"""
MCP server with local DB CRUD. When the user specifies an action explicitly in a prompt,
use the appropriate tool (or execute_instruction) to act as per the prompt.
"""
import os
import re
from typing import Any

from mcp.server.fastmcp import FastMCP

import db

# Initialize DB on import
db.init_db()

# Port 8001 by default so backend (LLM service) can use 8000
_port = int(os.environ.get("MCP_PORT") or os.environ.get("FASTMCP_PORT") or "8001")
mcp = FastMCP("Local DB", json_response=True, port=_port)


@mcp.tool()
def create_record(table_name: str, data: dict[str, Any]) -> dict:
    """Create a new record in the given table. data is a JSON object of fields (e.g. {"name": "Alice", "email": "alice@example.com"})."""
    return db.create_record(table_name, data)


@mcp.tool()
def get_record(table_name: str, record_id: int) -> dict | None:
    """Get a single record by table name and id. Returns null if not found."""
    return db.get_record(table_name, record_id)


@mcp.tool()
def list_records(table_name: str, limit: int = 100) -> list[dict]:
    """List records in a table, newest first. Use when the user asks to list, show, or get all records."""
    return db.list_records(table_name, limit=limit)


@mcp.tool()
def update_record(table_name: str, record_id: int, data: dict[str, Any]) -> dict | None:
    """Update an existing record. data is merged with existing fields. Returns null if record not found."""
    return db.update_record(table_name, record_id, data)


@mcp.tool()
def delete_record(table_name: str, record_id: int) -> dict:
    """Delete a record by table name and id. Returns success status."""
    deleted = db.delete_record(table_name, record_id)
    return {"deleted": deleted, "table_name": table_name, "record_id": record_id}


@mcp.tool()
def list_tables() -> list[str]:
    """List all table names that have at least one record. Use when the user asks what tables exist or to list tables."""
    return db.list_tables()


def _parse_instruction(instruction: str) -> dict | None:
    """
    Parse a simple natural-language instruction and return an action dict.
    Used when the user explicitly states what to do in a prompt.
    """
    instruction = instruction.strip().lower()
    # create / add / insert
    m = re.search(r"(?:add|create|insert)\s+(?:a\s+)?(?:record\s+)?(?:in\s+)?(?:table\s+)?['\"]?(\w+)['\"]?\s*(?:with\s+)?(.*)", instruction, re.DOTALL | re.I)
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

    return None


@mcp.tool()
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
    except Exception as e:
        return {"success": False, "error": str(e)}

    return {"success": False, "error": "Unknown action"}


@mcp.prompt()
def db_instructions() -> str:
    """Use this when the user explicitly asks to create, read, update, or delete data. Call the appropriate tool: create_record, get_record, list_records, update_record, delete_record, or execute_instruction with their exact request."""
    return (
        "When the user explicitly asks to create, read, update, or delete data in the database, "
        "use the available tools: create_record (to add), get_record or list_records (to read), "
        "update_record (to update), delete_record (to remove), or execute_instruction (to perform "
        "a single action from a short natural-language instruction). Act exactly as specified in the user's prompt."
    )


if __name__ == "__main__":
    # Port: set MCP_PORT or FASTMCP_PORT (e.g. 8001) to avoid conflict with backend on 8000
    mcp.run(transport="streamable-http")
