# MCP Server 1 — Local DB CRUD

MCP (Model Context Protocol) server in Python that exposes **local SQLite CRUD** as tools. When the user specifies an action explicitly in a prompt, the server acts as per the prompt (via the tools or `execute_instruction`).

## Setup

```bash
cd mcp_server_1
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
# or: uv add "mcp[cli]"
```

## Run

**Streamable HTTP (for MCP Inspector / Claude Code):**

```bash
.venv/bin/python server.py
# or: uv run --with mcp server.py
```

Then connect at `http://localhost:8000/mcp` (e.g. with [MCP Inspector](https://github.com/modelcontextprotocol/inspector): `npx -y @modelcontextprotocol/inspector`).

**Stdio (for Claude Desktop / other stdio clients):**

```bash
uv run mcp run server.py --transport stdio
```

## Tools

| Tool | Description |
|------|-------------|
| `create_record` | Insert a record: `table_name`, `data` (dict) |
| `get_record` | Get one record: `table_name`, `record_id` |
| `list_records` | List records in a table: `table_name`, optional `limit` |
| `update_record` | Update a record: `table_name`, `record_id`, `data` (merged) |
| `delete_record` | Delete a record: `table_name`, `record_id` |
| `list_tables` | List all table names that have data |
| `execute_instruction` | Run one operation from a short natural-language instruction (e.g. "add a record in users with name: John", "list all from items", "delete record 5 from users") |

## Prompt-driven behavior

When the user **explicitly** asks to create, read, update, or delete data:

- The client/LLM can call the specific CRUD tools with the right arguments, or
- Call `execute_instruction` with a short instruction; the server parses it and performs the operation.

The server also exposes a **prompt** (`db_instructions`) that tells the LLM to use these tools when the user asks for database operations and to act as specified in the prompt.

## Database

- SQLite file: `mcp_server_1/mcp_server_1.db`
- Single table `records`: `id`, `table_name`, `data` (JSON), `created_at`, `updated_at`
- Logical “tables” are distinguished by `table_name` (e.g. `users`, `items`).
