# Backend Architecture: LLM Service and MCP Server

This document describes the backend LLM service, the agent service, the MCP client, and the MCP server (Python classes and modules in `mcp_server_1`).

---

## 1. Overview

```
┌─────────────────┐     HTTP      ┌──────────────────┐     MCP (streamable-http)     ┌─────────────────┐
│   Web UI / API   │ ◄──────────► │  LLM Service     │ ◄──────────────────────────► │  MCP Server     │
│   (frontend)     │   :8000       │  (FastAPI)       │   :8001/mcp                   │  (mcp_server_1) │
└─────────────────┘               │                  │                               │                 │
                                  │  • main.py        │                               │  • server.py    │
                                  │  • agent_service  │                               │  • db.py        │
                                  │  • mcp_client     │                               │  • SQLite       │
                                  │  • Ollama         │                               └─────────────────┘
                                  └──────────────────┘
```

- **LLM Service** (`backend/llm_service/`): FastAPI app on port 8000. Handles models, prompts, and **agent queries**. The agent uses tools provided by the MCP server.
- **MCP Server** (`mcp_server_1/`): FastMCP app on port 8001. Exposes database tools (tables, records, schema) over the MCP protocol at `/mcp`.
- **Agent**: Runs inside the LLM service. On each user message it fetches tools from the MCP server, uses Ollama as the LLM, and decides which tools to call (no regex routing).

---

## 2. LLM Service (Backend)

**Location:** `backend/llm_service/`

### 2.1 Entry Point: `main.py`

- **FastAPI app** with CORS, lifespan, and routes.
- **Lifespan** (`lifespan`): On startup, initializes the settings DB and calls `get_mcp_tools()` to load and log MCP capabilities. On shutdown, logs shutdown.
- **Key routes:**
  - **Models:** `GET/POST /api/models`, `POST /api/models/load`, `GET/POST /api/models/active`, `DELETE /api/models/{model_name}`, `GET /api/models/active/capabilities`
  - **Library:** `GET /api/library/search`, `POST /api/library/pull`
  - **Agent:** `POST /api/agent/query` — runs the MCP-powered agent with the active model.
  - **Prompt:** `POST /api/prompt` — sends the user message to the **agent** first; if the agent/MCP fails, falls back to direct Ollama (no tool calls).
  - **Context:** `GET/POST /api/context` — get/set system context stored in DB.
  - **Health:** `GET /health`
- **Active model:** Resolved from `database.get_setting("active_model")` or, if unset, from the first running Ollama model. Used for both agent and fallback prompt.
- **OpenAPI:** Served at `/openapi.json`, docs at `/docs`, ReDoc at `/redoc`.

### 2.2 Config: `config.py`

- **OLLAMA_HOST** — Ollama API base URL (default `http://localhost:11434`).
- **OLLAMA_LIBRARY_URL** — Library/tags URL for search.
- **DB_PATH** — SQLite path for LLM service settings (default `llm_service.db`).
- **MCP_SERVER_1_URL** — MCP server URL (default `http://127.0.0.1:8001/mcp`). Used by the agent and the MCP client.
- **SYSTEM_PROMPT** — Default system prompt for the agent (tool-calling / database instructions). Can be overridden with env `AGENT_SYSTEM_PROMPT`.

### 2.3 Database: `database.py`

- **Purpose:** SQLite storage for LLM service state (active model, context prompt).
- **Table:** `settings (key TEXT PRIMARY KEY, value TEXT)`.
- **API:** `init_db()`, `get_setting(key)`, `set_setting(key, value)`.
- **Usage:** Active model and context prompt are read/written here; no MCP or agent logic.

---

## 3. Agent Service

**Location:** `backend/llm_service/services/agent_service.py`

The agent is the component that interprets the user message and calls MCP tools when needed. It uses **LlamaIndex** (ReAct agent + Ollama) and discovers tools from the MCP server on each query.

### 3.1 Dependencies

- `llama-index-tools-mcp` — to load tools from an MCP URL.
- `llama-index-llms-ollama` — Ollama LLM for the agent.
- `llama-index-core` — ReAct agent and workflow (e.g. `ReActAgent`, `Context`, `ToolCall`, `ToolCallResult`).

### 3.2 Main Functions

**`get_mcp_tools()` (async)**  
- Loads tools from the MCP server at `MCP_SERVER_1_URL` via `aget_tools_from_mcp_url(MCP_SERVER_1_URL)`.
- Called at **startup** (lifespan) and on **every agent query** so tool list is always current.
- Logs and prints tool names and short descriptions.
- Returns a list of LlamaIndex tools that the agent can invoke.

**`_create_agent(tools, llm, system_prompt)`**  
- Builds a **ReActAgent** with the given tools, Ollama LLM, and optional system prompt.
- Used internally by `run_agent_query`.

**`run_agent_query(message_content, model, system_prompt=None, verbose=True)` (async)**  
- **Flow:**
  1. Load tools with `get_mcp_tools()`.
  2. Create Ollama LLM for the given `model` (timeout 360s, context window 8192).
  3. Use `system_prompt` or `SYSTEM_PROMPT` from config.
  4. Create ReAct agent and run `agent.run(message_content, ctx=ctx)`.
  5. Optionally stream events to log **ToolCall** and **ToolCallResult** (tool name, arguments, and result preview).
  6. Return the final response as a string.
- **No regex or pattern routing:** the LLM decides when and which tools to call from the user message and tool descriptions.
- **Return value:** Final text response from the agent (including tool results summarized in natural language).

### 3.3 System Prompt (config)

The system prompt tells the agent to:

- Use tools to answer; don’t give generic or theoretical answers or ask “would you like me to proceed?”
- Treat record data as flexible (e.g. any field like `status`); schema is informational.
- For “update user X and get details”: e.g. `list_tables`, then `find_update_and_get_record(table_name, "name", "X", {"status": "inactive"})`, and return the tool result.
- For tables/schema: call `list_tables` and `get_table_schema` and return actual results.
- For create/alter/drop table: use `create_table`, `alter_table`, `drop_table`.
- Always return real data from the database; never generic SQL or hypothetical answers.

---

## 4. MCP Client

**Location:** `backend/llm_service/services/mcp_client.py`

Thin client used to call the MCP server **directly** (one specific tool), without going through the agent. The main prompt flow uses the **agent**, which in turn uses MCP tools via LlamaIndex; this client is for optional direct calls.

### 4.1 Implementation

- **`_call_mcp_sync(instruction)`:** Synchronous helper. Uses `anyio` and the official `mcp` package (`ClientSession`, `streamable_http_client`). Connects to `MCP_SERVER_1_URL`, initializes the session, and calls the **`execute_instruction`** tool with `arguments={"instruction": instruction}`. Parses the tool response (text blocks), tries to parse as JSON, and returns a dict.
- **`call_mcp_execute_instruction(instruction)` (async):** Wraps `_call_mcp_sync` in `run_in_executor` so it can be awaited from FastAPI. Returns the same dict or error structure.

### 4.2 Usage

- **Not used** by the main `/api/prompt` or `/api/agent/query` flow (those use the agent, which gets all MCP tools via LlamaIndex).
- Available for any endpoint or script that wants to call the MCP server’s `execute_instruction` tool directly (e.g. natural-language one-shot DB operations).

### 4.3 Dependencies

- `mcp` (with streamable HTTP client) and `anyio` for running the sync MCP call.

---

## 5. MCP Server (`mcp_server_1`)

**Location:** `mcp_server_1/`

The MCP server exposes database operations as MCP tools. It uses **FastMCP** and a local **SQLite** layer. The LLM service connects to it at `MCP_SERVER_1_URL` (e.g. `http://127.0.0.1:8001/mcp`).

### 5.1 Server Entry: `server.py`

**FastMCP app**

- **Instance:** `mcp = FastMCP("Local DB", instructions="...", json_response=True, port=...)`
- **Port:** From env `MCP_PORT` or `FASTMCP_PORT`, default **8001**.
- **Transport:** Streamable HTTP; run with `mcp.run(transport="streamable-http")` when `__name__ == "__main__"`.
- **DB:** Imports `db` and calls `db.init_db()` on load.

**Tools (each registered with `@mcp.tool(description="...")`)**

| Tool | Purpose |
|------|--------|
| `create_record(table_name, data)` | Insert a record; for schema tables uses separate columns, else JSON in `records` table. |
| `get_record(table_name, record_id)` | Get one record by id. |
| `list_records(table_name, limit=100)` | List records, newest first. |
| `find_records_by_field(table_name, field_name, field_value)` | Find records where `data[field_name] == field_value` (e.g. user by name). |
| `find_update_and_get_record(table_name, field_name, field_value, update_data)` | Find by field, update with `update_data`, return updated record. |
| `update_record(table_name, record_id, data)` | Merge `data` into existing record. |
| `delete_record(table_name, record_id)` | Delete one record. |
| `list_tables()` | List table names (from records, table_schemas, and real SQLite tables). |
| `create_table(table_name, fields)` | Create a real SQLite table with one column per field; `fields` can be string, list of names, or list of `{name, type}` (or JSON string of that list). |
| `alter_table(table_name, fields)` | Replace table schema; recreates real table with new columns and copies data. |
| `drop_table(table_name)` | Drop table and its schema. |
| `get_table_schema(table_name)` | Return schema (field names and types) or null. |
| `add_survey_questions(questions_text)` | Parse questions (newline or numbered) and insert into `survey_questions` table. |
| `execute_instruction(instruction)` | Parse natural-language instruction and run one DB operation (add/list/get/update/delete/create_table/alter_table/drop_table/add_survey_questions). |

**Helpers in server.py**

- **`_parse_fields(fields)`:** Normalizes `fields` for create/alter: accepts comma-separated string, list of names, list of `{name, type}`, or JSON string of that list. Returns `[{"name": str, "type": str}, ...]`.
- **`_split_survey_questions(text)`:** Splits text into individual questions (newlines, numbered lines).
- **`_parse_instruction(instruction)`:** Regex-based parser for `execute_instruction` (e.g. “add record in X with name: Y”, “list from X”, “update record id in table set k: v”, “create table X with fields a, b”, “drop table X”). Returns an action dict or None.

**Prompt**

- **`@mcp.prompt() db_instructions()`:** Returns short instructions for the LLM on when to use which tool (create/alter/drop tables, CRUD, or `execute_instruction`).

### 5.2 Database Layer: `db.py`

**Purpose:** Single SQLite file (`mcp_server_1.db`) with two modes:

1. **Schema-backed tables:** Tables created with `create_table_schema` get a **real SQLite table** with one column per field (plus `id`, `created_at`, `updated_at`). CRUD uses these columns.
2. **Legacy / no schema:** Data stored in the generic **`records`** table as JSON in a `data` column, keyed by `table_name`.

**Internal structures**

- **`records`** — Generic store: `(id, table_name, data TEXT, created_at, updated_at)`. Used when there is no real table for that `table_name`.
- **`table_schemas`** — Metadata: `(table_name, fields_json, created_at, updated_at)`. `fields_json` is the list of `{name, type}` used for schema-aware CRUD and for creating/altering real tables.
- **Real tables** — Created per logical table (e.g. `users`, `file`) with columns: `id`, then one column per schema field, then `created_at`, `updated_at`.

**Key functions**

- **`init_db()`** — Creates `records` and `table_schemas` if not present.
- **`_real_table_exists(conn, table_name)`** — True if a SQLite table with that name exists (excluding internal names).
- **`_uses_real_table(conn, table_name)`** — True if CRUD should use the real table instead of `records`.
- **`_safe_identifier(name)`** — Sanitizes table/column names for SQL.
- **`create_table_schema(table_name, fields)`** — Optionally drops existing real table, then creates a new SQLite table with one column per field and updates `table_schemas`. No `fields_json` column in the data table.
- **`get_table_schema(table_name)`** — Reads `table_schemas.fields_json` and returns list of `{name, type}`.
- **`create_record(table_name, data)`** — If real table exists, inserts into it using schema columns; otherwise inserts into `records` with `json.dumps(data)`.
- **`get_record` / `list_records`** — If real table exists, SELECT from it and shape as `{id, table_name, data, created_at, updated_at}`; otherwise read from `records` and parse JSON.
- **`find_records_by_field`** — Uses `list_records` and filters in memory by `data[field_name] == field_value` (case-insensitive for strings).
- **`find_update_and_get`** — Uses `find_records_by_field`, then `update_record`, then `get_record`; returns updated record or None.
- **`update_record`** — Merges `data` into existing; for real tables builds `UPDATE ... SET col1=?, ...` from schema.
- **`delete_record`** — Deletes from real table or from `records`.
- **`list_tables()`** — Union of distinct `table_name` from `records`, names from `table_schemas`, and table names from `sqlite_master` (excluding internal tables).
- **`alter_table_schema(table_name, fields)`** — For real tables: create temp table with new schema, copy overlapping columns, drop old table, rename temp; then update `table_schemas`.
- **`drop_table(table_name)`** — Drops real table if present, deletes rows in `records` for that table_name, and removes from `table_schemas`.

**Constants**

- **`_INTERNAL_TABLES`** — `{"records", "table_schemas", "sqlite_sequence"}` — never treated as user tables.
- **`_SQLITE_TYPE`** — Maps schema type names to SQLite types (e.g. text→TEXT, integer→INTEGER).

---

## 6. Request Flows

### 6.1 User sends a prompt (e.g. “List all tables” or “Make status of user lakshmana inactive and get user details”)

1. **Web UI** → `POST /api/prompt` with `{ "prompt": "..." }`.
2. **main.py** resolves active model, then calls **`run_agent_query(body.prompt, model=active, verbose=True)`**.
3. **agent_service** loads MCP tools from `MCP_SERVER_1_URL`, builds ReAct agent with Ollama and system prompt, runs `agent.run(message)`.
4. **Agent (LLM)** decides to call tools (e.g. `list_tables`, then `find_update_and_get_record`). LlamaIndex sends MCP tool calls to the MCP server.
5. **MCP server** (`server.py`) receives tool invocations, calls the corresponding function (e.g. `db.list_tables()`, `db.find_update_and_get(...)`), returns results.
6. **Agent** gets tool results, may call more tools or form a final answer, and returns text.
7. **main.py** returns `{ "response": response_text, "model": active }`. If the agent raises, it falls back to direct Ollama (no tools).

### 6.2 Direct agent query

- **`POST /api/agent/query`** with `{ "query": "..." }` — same as above but explicitly “agent” endpoint; no fallback to direct Ollama in code (both use `run_agent_query`).

### 6.3 Optional direct MCP call (no agent)

- Any code can call **`call_mcp_execute_instruction(instruction)`** to hit the MCP server’s `execute_instruction` tool only. The main UI flow does not use this; the agent uses the full set of tools instead.

---

## 7. File Reference

| Path | Role |
|------|------|
| `backend/llm_service/main.py` | FastAPI app, routes, lifespan, agent and prompt entry points. |
| `backend/llm_service/config.py` | OLLAMA_*, MCP_SERVER_1_URL, SYSTEM_PROMPT, DB_PATH. |
| `backend/llm_service/database.py` | Settings SQLite (active model, context). |
| `backend/llm_service/services/agent_service.py` | get_mcp_tools, run_agent_query, ReAct agent, tool event logging. |
| `backend/llm_service/services/mcp_client.py` | call_mcp_execute_instruction, _call_mcp_sync (direct MCP). |
| `mcp_server_1/server.py` | FastMCP app, all @mcp.tool definitions, _parse_fields, _parse_instruction, execute_instruction, db_instructions prompt. |
| `mcp_server_1/db.py` | init_db, table_schemas + records + real tables, all CRUD and schema create/alter/drop. |

---

## 8. Environment Summary

**LLM Service**

- `OLLAMA_HOST` — Ollama API URL.
- `MCP_SERVER_1_URL` — MCP server base URL (e.g. `http://127.0.0.1:8001/mcp`).
- `LLM_SERVICE_DB_PATH` — Path to `llm_service.db`.
- `AGENT_SYSTEM_PROMPT` — Optional override for agent system prompt.

**MCP Server**

- `MCP_PORT` or `FASTMCP_PORT` — HTTP port (default 8001).

No separate “registration” step: the LLM service discovers tools from the MCP server at the configured URL on each agent query and at startup.
