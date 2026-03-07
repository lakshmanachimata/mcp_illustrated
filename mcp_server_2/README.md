# MCP Server 2 — Scraper & Vector Search

Tech stack: **Bright Data** (web scraping at scale), **Zvec** (in-process vector DB), **Cursor** (MCP client).

## Workflow

1. User inputs a query through the MCP client (Cursor).
2. Client contacts this MCP server and can call **select_relevant_tool** (or **get_relevant_tools**) with the query.
3. Server returns the most relevant tool(s) from Zvec (semantic match on tool names/descriptions).
4. Client invokes the chosen tool (e.g. **scrape_page**, **search_stored_documents**, **store_document**).
5. Tool output is returned to the client; the client uses it to generate a response.

## Port

Runs on **port 8002** by default (`MCP_PORT` or `FASTMCP_PORT`).

## Tools

| Tool | Description |
|------|-------------|
| `get_relevant_tools(user_query, top_k=3)` | Select tools most relevant to the query (vector search in Zvec). |
| `select_relevant_tool_for_query(user_query, top_k=3)` | Alias for tool selection. |
| `scrape_page(url, timeout=30)` | Scrape a URL (Bright Data proxy if set); returns text and title. |
| `search_stored_documents(query, limit=5)` | Semantic search over documents stored in Zvec. |
| `store_document(text, metadata?, doc_id?)` | Store text in Zvec for later search. |

## Prerequisites

- **Python 3.10+** (Zvec supports Linux x86_64/ARM64 and macOS ARM64)
- **Zvec** — in-process vector DB; no separate server. Data is stored under `ZVEC_BASE_PATH`.
- **Bright Data** (optional): set `BRIGHT_DATA_PROXY` for scraping at scale

## Environment

| Variable | Description | Default |
|----------|-------------|---------|
| `MCP_PORT` / `FASTMCP_PORT` | HTTP port | `8002` |
| `BRIGHT_DATA_PROXY` | Proxy URL (e.g. `http://user:pass@brd.superproxy.io:22225`) | — |
| `ZVEC_BASE_PATH` | Directory for Zvec collections (tools + documents) | `./zvec_data` |
| `EMBEDDING_MODEL` | sentence-transformers model | `all-MiniLM-L6-v2` |

## Run

```bash
cd mcp_server_2
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python server.py
```

Or with env:

```bash
export BRIGHT_DATA_PROXY="http://user:pass@brd.superproxy.io:22225"
export ZVEC_BASE_PATH="./zvec_data"
python server.py
```

## Cursor (MCP client) setup

In Cursor, add this MCP server so the client can call it:

- **Transport:** streamable HTTP
- **URL:** `http://127.0.0.1:8002/mcp` (or your host/port)

Then when the user asks a question, Cursor can:

1. Call `get_relevant_tools(user_query)` to get the best tool(s).
2. Call the suggested tool (e.g. `scrape_page("https://example.com")`).
3. Use the tool output in the generated response.

## Zvec collections

- **tools** — Stored under `{ZVEC_BASE_PATH}/tools`. Tool names and descriptions for semantic tool selection.
- **documents** — Stored under `{ZVEC_BASE_PATH}/documents`. Stored documents for semantic search (e.g. scraped content).

First run creates these collections on disk if they don’t exist and seeds the tools collection.
