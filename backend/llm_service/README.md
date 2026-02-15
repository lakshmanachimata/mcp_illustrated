# LLM Service (Ollama)

FastAPI service that wraps Ollama with persistence (SQLite) for active model and context prompt.

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
| POST | `/api/prompt` | Send prompt to active model (body: `{"prompt": "...", "stream": false}`) |
| POST | `/api/context` | Set context/system prompt (body: `{"context": "..."}`) |
| GET | `/api/context` | Get current context prompt |

DB file: `llm_service.db` (path via `LLM_SERVICE_DB_PATH`).  
Ollama host: `OLLAMA_HOST` (default `http://localhost:11434`).
