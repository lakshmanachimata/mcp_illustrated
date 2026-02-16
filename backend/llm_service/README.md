# LLM Service (Ollama)

FastAPI service that wraps Ollama with persistence (SQLite) for active model and context prompt, plus an **MCP-powered agent** (LlamaIndex + Ollama).

## Tech stack

- **LlamaIndex** – MCP-powered agent: discovers tools from the MCP server and chooses which to invoke per query.
- **Ollama** – Local LLM; the agent uses the **active model** selected in the web UI (not hardcoded).
- **LightningAI** – Optional: use [Lightning Studio](https://lightning.ai/docs/overview/studios/) for development and hosting (expose port 8000, run this service + MCP server).

## Agent flow

1. User submits a query (e.g. via `POST /api/agent/query`).
2. Agent connects to the MCP server and discovers tools (e.g. `list_tables`, `list_records`, `create_record`, `execute_instruction`).
3. Based on the query, the agent invokes the right tool(s) and gets context.
4. Agent returns a context-aware response.

The agent uses whichever model is currently **active** in the web UI (same as the prompt endpoint). On each agent query, tools are re-fetched from the MCP server so capabilities stay in sync.

The agent uses a **system prompt** instructing it to use tool calling and interact with the database before answering (configurable via `AGENT_SYSTEM_PROMPT`). The MCP server exposes its tools with **annotations** (descriptions) so the LLM service sees them as clear capabilities (e.g. "Run a query to list tables", "Run a query to create a record").

## Setup

```bash
cd backend/llm_service
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt  # -r = install from file (not a package named "requirements.txt")
```

Ensure [Ollama](https://ollama.com) is installed and running (`ollama serve` or start the app).

## Run

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- API: http://localhost:8000  

**If port 8000 is already in use:** stop the other process first, or you’ll see `[Errno 48] address already in use`. To find and kill it (macOS/Linux):
```bash
lsof -i :8000 -t | xargs kill
```

## MCP server (registration / configuration)

The LLM service talks to the MCP server by **URL**—there is no separate “register” step. It’s configured once and then used when the user’s prompt looks like “add to database” / “store in MCP server”, etc.

1. **Configure the URL**  
   In `config.py` the default is:
   ```text
   MCP_SERVER_URL = http://127.0.0.1:8001/mcp
   ```
   Override with the env var if your MCP server runs elsewhere:
   ```bash
   export MCP_SERVER_URL="http://127.0.0.1:8001/mcp"
   ```

2. **Run the MCP server**  
   Start the MCP server (e.g. `mcp_server_1`) so it listens on **port 8001** and exposes the `/mcp` streamable HTTP endpoint. In this repo, the launch config sets `MCP_PORT=8001` / `FASTMCP_PORT=8001` for the MCP Server.

3. **Start order**  
   Start **LLM Service** (port 8000) and **MCP Server** (8001). The LLM service does not “register” at startup; it calls the MCP server on demand when handling a prompt that matches the add/store patterns (see `services/mcp_client.py` → `should_use_mcp_db` and `call_mcp_execute_instruction`).

So: **“Register MCP server to LLM service”** = set `MCP_SERVER_URL` (or leave default) and run the MCP server on that host/port. No extra registration API or step is required.

## Debugging (Cursor / VS Code)

- Use launch config **"LLM Service (no reload)"** so the debugger attaches to the process that serves requests (with `--reload`, a child process serves and breakpoints don’t hit).
- **Start the LLM Service first** so it can bind to 8000. If you start MCP Server or another app on 8000, free the port (see above) then start LLM Service again.
- Breakpoints and `[LLM Service]` logs appear in the **Debug Console** when running under the debugger.  
- **Swagger UI**: http://localhost:8000/docs  
- **ReDoc**: http://localhost:8000/redoc  
- **OpenAPI JSON**: http://localhost:8000/openapi.json  

To generate static OpenAPI files in the repo:

```bash
python generate_openapi.py
```

This writes `openapi.json` and `swagger.yaml` in this directory.

## APIs

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/models` | List local LLM models |
| POST | `/api/models/load` | Load/switch model (body: `{"model": "name"}`) |
| GET | `/api/models/active` | Get active model |
| POST | `/api/models/active` | Set active model in DB |
| GET | `/api/library/search?q=` | Search Ollama library |
| POST | `/api/library/pull` | Pull model from library (streams progress, body: `{"model": "name"}`) |
| GET | `/api/models/active/capabilities` | Show capabilities of active model |
| POST | `/api/agent/query` | MCP-powered agent: body `{"query": "..."}` → agent discovers tools, invokes the right one(s), returns context-aware response |
| POST | `/api/prompt` | Send prompt to active model (body: `{"prompt": "...", "stream": false}`) |
| POST | `/api/context` | Set context/system prompt (body: `{"context": "..."}`) |
| GET | `/api/context` | Get current context prompt |

DB file: `llm_service.db` (path via `LLM_SERVICE_DB_PATH`).  
Ollama host: `OLLAMA_HOST` (default `http://localhost:11434`).  
Agent uses the active model from the web UI (no separate env).

## LightningAI (development and hosting)

To run this service on [Lightning Studio](https://lightning.ai/docs/overview/studios/):

1. Create a Studio and open it in the browser.
2. Clone this repo or upload the `backend/llm_service` (and optionally `mcp_server_1`) code.
3. Install deps: `pip install -r backend/llm_service/requirements.txt`. For the agent you need Ollama (or a remote Ollama URL) and the MCP server reachable at `MCP_SERVER_URL`.
4. Expose a public port (e.g. 8000) in the Studio UI and run: `uvicorn main:app --host 0.0.0.0 --port 8000` from `backend/llm_service`. If the MCP server runs in the same Studio, start it on 8001 first so the agent can discover tools.
