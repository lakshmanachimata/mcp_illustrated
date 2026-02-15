"""Ollama client wrapper and library fetch."""
import httpx
from ollama import Client

from config import OLLAMA_HOST, OLLAMA_LIBRARY_URL


def get_client() -> Client:
    return Client(host=OLLAMA_HOST)


def _fetch_tags_http():
    """GET /api/tags from Ollama (direct HTTP). Returns list of model dicts."""
    with httpx.Client(timeout=10.0) as client:
        base = OLLAMA_HOST.rstrip("/")
        r = client.get(f"{base}/api/tags")
        r.raise_for_status()
        data = r.json()
    return data.get("models") or []


def list_models():
    """List locally available models. Uses Python client; falls back to direct HTTP if needed."""
    try:
        resp = get_client().list()
        if hasattr(resp, "models") and resp.models is not None:
            return resp
    except Exception:
        pass
    # Fallback: direct HTTP (e.g. client failed or returned empty)
    models = _fetch_tags_http()
    class _ListLike:
        models = models
    return _ListLike()


def load_model(model: str):
    """Load/switch model (keeps it in memory). Runs a no-op generate to load."""
    client = get_client()
    client.generate(model=model, prompt="", stream=False)
    return {"model": model, "status": "loaded"}


def get_running_models():
    """Currently loaded model(s) from Ollama."""
    return get_client().ps()


def delete_model(model: str):
    """Remove a model from local Ollama."""
    get_client().delete(model=model)
    return {"model": model, "status": "deleted"}


def pull_model_stream_sync(model: str):
    """Sync generator: stream pull progress. Yields dicts with status, completed, total, etc."""
    client = get_client()
    for progress in client.pull(model=model, stream=True):
        if hasattr(progress, "__dict__"):
            yield {"status": getattr(progress, "status", ""), "completed": getattr(progress, "completed", 0), "total": getattr(progress, "total", 0)}
        else:
            yield progress


def show_model(model: str) -> dict:
    """Get model info and capabilities from ollama show."""
    info = get_client().show(model)
    if hasattr(info, "model_dump"):
        return info.model_dump()
    if hasattr(info, "__dict__"):
        return {k: v for k, v in info.__dict__.items() if not k.startswith("_")}
    return dict(info) if info else {}


def generate_response(model: str, prompt: str, system: str | None = None, stream: bool = False):
    """Generate response from model. If stream=True, returns generator."""
    client = get_client()
    if system:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
    else:
        messages = [{"role": "user", "content": prompt}]
    if stream:
        return client.chat(model=model, messages=messages, stream=True)
    return client.chat(model=model, messages=messages, stream=False)


async def search_library(query: str = "") -> list[dict]:
    """Search Ollama library (ollama.com). Optional query filters by name."""
    async with httpx.AsyncClient() as client:
        r = await client.get(OLLAMA_LIBRARY_URL)
        r.raise_for_status()
        data = r.json()
    models = data.get("models") or []
    if query:
        q = query.lower()
        models = [m for m in models if q in (m.get("name") or "").lower()]
    return models
