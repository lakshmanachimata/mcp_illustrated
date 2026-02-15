import os

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_LIBRARY_URL = os.getenv("OLLAMA_LIBRARY_URL", "https://ollama.com/api/tags")
DB_PATH = os.getenv("LLM_SERVICE_DB_PATH", "llm_service.db")
