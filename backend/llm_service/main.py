"""FastAPI app for LLM service (Ollama)."""
import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import database
from config import OLLAMA_HOST
from services.mcp_client import call_mcp_execute_instruction, should_use_mcp_db
from services.ollama_client import (
    get_running_models,
    generate_response,
    list_models,
    load_model as ollama_load_model,
    delete_model as ollama_delete_model,
    pull_model_stream_sync,
    search_library,
    show_model,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    database.init_db()
    yield


app = FastAPI(
    title="LLM Service",
    description="Ollama-backed API: list/load models, search library, pull with progress, prompt with context.",
    version="1.0.0",
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LoadModelRequest(BaseModel):
    model: str

    model_config = {"json_schema_extra": {"examples": [{"model": "llama3.2"}]}}


class PromptRequest(BaseModel):
    prompt: str
    stream: bool = False

    model_config = {"json_schema_extra": {"examples": [{"prompt": "What is 2+2?", "stream": False}]}}


class ContextRequest(BaseModel):
    context: str

    model_config = {"json_schema_extra": {"examples": [{"context": "You are a helpful coding assistant."}]}}


class PullRequest(BaseModel):
    model: str

    model_config = {"json_schema_extra": {"examples": [{"model": "llama3.2"}]}}


def _to_dict(obj):
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(mode="json")  # ensures datetime, etc. are JSON-serializable
        except TypeError:
            return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    return obj if isinstance(obj, dict) else {}


def _normalize_model(m):
    """Ensure each model has both 'name' and 'model' for the frontend."""
    d = _to_dict(m) if not isinstance(m, dict) else m
    if not isinstance(d, dict):
        return {"name": str(m), "model": str(m)}
    name = d.get("name") or d.get("model") or ""
    d["name"] = name
    d["model"] = name
    return d


@app.get("/api/models", tags=["Models"])
def api_list_models():
    """Returns list of locally available LLM models."""
    try:
        resp = list_models()
        models = None
        if hasattr(resp, "models") and resp.models is not None:
            models = [_normalize_model(m) for m in resp.models]
        elif isinstance(resp, dict) and "models" in resp:
            raw = resp["models"] or []
            models = [_normalize_model(m) for m in raw]
        if models is None:
            models = []
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/api/models/load", tags=["Models"])
def api_load_model(body: LoadModelRequest):
    """Load or switch the active LLM model. May take time."""
    try:
        return ollama_load_model(body.model)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/models/active", tags=["Models"])
def api_active_model():
    """Returns the active (selected) LLM model name from DB, or currently running from Ollama."""
    saved = database.get_setting("active_model")
    if saved:
        return {"active_model": saved}
    try:
        ps = get_running_models()
        models = getattr(ps, "models", None) or []
        if models:
            m = models[0]
            name = m.get("name") if isinstance(m, dict) else getattr(m, "name", None) or getattr(m, "model", None)
            if name:
                return {"active_model": name}
    except Exception:
        pass
    return {"active_model": None}


@app.post("/api/models/active", tags=["Models"])
def api_set_active_model(body: LoadModelRequest):
    """Set the active model in DB (does not load it; use /api/models/load to load)."""
    database.set_setting("active_model", body.model)
    return {"active_model": body.model}


@app.delete("/api/models/{model_name:path}", tags=["Models"])
def api_delete_model(model_name: str):
    """Delete a model from local Ollama. Clears active model if it was the deleted one."""
    try:
        result = ollama_delete_model(model_name)
        if database.get_setting("active_model") == model_name:
            database.set_setting("active_model", "")
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/library/search", tags=["Library"])
async def api_library_search(q: str = ""):
    """Search Ollama library (ollama.com) for models. Optional query filters by name."""
    try:
        models = await search_library(query=q)
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/api/library/pull", tags=["Library"])
async def api_library_pull(body: PullRequest):
    """Pull model from Ollama library; streams progress as NDJSON."""
    queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def run_pull():
        try:
            for chunk in pull_model_stream_sync(body.model):
                asyncio.run_coroutine_threadsafe(queue.put(("chunk", chunk)), loop).result()
        except Exception as e:
            asyncio.run_coroutine_threadsafe(queue.put(("error", str(e))), loop).result()
        asyncio.run_coroutine_threadsafe(queue.put(("done", None)), loop).result()

    async def gen():
        task = loop.run_in_executor(None, run_pull)
        while True:
            kind, payload = await queue.get()
            if kind == "error":
                yield json.dumps({"error": payload}) + "\n"
                break
            if kind == "done":
                break
            yield json.dumps(payload) + "\n"
        await task

    return StreamingResponse(
        gen(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/models/active/capabilities", tags=["Models"])
def api_active_model_capabilities():
    """Returns capabilities/info of the active model from Ollama show."""
    saved = database.get_setting("active_model")
    if not saved:
        try:
            ps = get_running_models()
            models = getattr(ps, "models", None) or []
            if models:
                m = models[0]
                saved = m.get("name") or m.get("model") if isinstance(m, dict) else getattr(m, "name", None) or getattr(m, "model", None)
        except Exception:
            pass
    if not saved:
        raise HTTPException(status_code=404, detail="No active model selected or loaded")
    try:
        return show_model(saved)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/prompt", tags=["Prompt"])
async def api_prompt(body: PromptRequest):
    """Send prompt to the active model and return response. When user asks to add/store in DB, uses MCP server (mcp_server_1)."""
    if should_use_mcp_db(body.prompt):
        try:
            mcp_result = await call_mcp_execute_instruction(body.prompt)
            if mcp_result.get("success"):
                result = mcp_result.get("result", mcp_result)
                if isinstance(result, dict) and "created" in result:
                    msg = f"Added {result.get('created', 0)} item(s) to the database (MCP server)."
                else:
                    msg = f"Done. Database result: {json.dumps(result, indent=2)}"
                return {"response": msg, "model": "mcp_server_1", "mcp_result": mcp_result}
            return {"response": f"Database action failed: {mcp_result.get('error', mcp_result)}", "model": "mcp_server_1", "mcp_result": mcp_result}
        except Exception as e:
            return {"response": f"Could not reach MCP server (is it running on port 8001?): {e}", "model": "mcp_server_1"}

    active = database.get_setting("active_model")
    if not active:
        try:
            ps = get_running_models()
            models = getattr(ps, "models", None) or []
            if models:
                active = models[0].get("name") or models[0].get("model") if isinstance(models[0], dict) else getattr(models[0], "name", None) or getattr(models[0], "model", None)
        except Exception:
            pass
    if not active:
        raise HTTPException(status_code=400, detail="No active model selected or loaded")
    context = database.get_setting("context_prompt") or ""
    try:
        if body.stream:
            def stream_gen():
                for chunk in generate_response(active, body.prompt, system=context or None, stream=True):
                    yield json.dumps({"content": getattr(chunk.message, "content", "") or (chunk.get("message", {}).get("content", "") if isinstance(chunk, dict) else "")}) + "\n"
            return StreamingResponse(
                stream_gen(),
                media_type="application/x-ndjson",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        resp = generate_response(active, body.prompt, system=context or None, stream=False)
        content = getattr(resp.message, "content", None) or (resp.get("message", {}).get("content", "") if isinstance(resp, dict) else "")
        return {"response": content, "model": active}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/context", tags=["Context"])
def api_set_context(body: ContextRequest):
    """Set context/system prompt. Remains until a new context is set."""
    database.set_setting("context_prompt", body.context)
    return {"context_prompt": body.context}


@app.get("/api/context", tags=["Context"])
def api_get_context():
    """Get current context prompt."""
    value = database.get_setting("context_prompt")
    return {"context_prompt": value or ""}


@app.get("/health", tags=["Health"])
def health():
    """Service and Ollama host status."""
    return {"status": "ok", "ollama_host": OLLAMA_HOST}
