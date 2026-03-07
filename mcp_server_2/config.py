"""Configuration for MCP Server 2: Bright Data scraping + Zvec vector DB."""
import os
from pathlib import Path

# Port for this MCP server (Cursor connects here)
PORT = int(os.environ.get("MCP_PORT", os.environ.get("FASTMCP_PORT", "8002")))

# Bright Data: proxy URL for web scraping (e.g. http://user:pass@brd.superproxy.io:22225)
# Or leave empty to use direct HTTP (no proxy).
BRIGHT_DATA_PROXY = os.environ.get("BRIGHT_DATA_PROXY", "").strip()

# Zvec: in-process vector DB (no separate server). Base directory for collection storage.
ZVEC_BASE_PATH = os.environ.get("ZVEC_BASE_PATH", str(Path(__file__).resolve().parent / "zvec_data"))
TOOLS_COLLECTION_PATH = os.path.join(ZVEC_BASE_PATH, "tools")
DOCUMENTS_COLLECTION_PATH = os.path.join(ZVEC_BASE_PATH, "documents")

# Embedding model for semantic search (sentence-transformers)
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
VECTOR_SIZE = 384  # all-MiniLM-L6-v2 output size

# Collection names (for logging / API)
TOOLS_COLLECTION = "mcp_server_2_tools"
DOCUMENTS_COLLECTION = "mcp_server_2_documents"
