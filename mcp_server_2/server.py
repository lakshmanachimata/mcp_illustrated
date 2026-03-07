"""
MCP Server 2: Bright Data scraping + Zvec vector DB.
Cursor (MCP client) connects to this server. User query → select relevant tool → tool output → client response.
"""
import logging
import os

from mcp.server.fastmcp import FastMCP

from config import PORT
from scraper import scrape_url
from vector_store import (
    add_document,
    ensure_collections,
    init_tools_registry,
    search_documents,
    select_relevant_tool,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP(
    "Scraper & Vector Search",
    instructions="Tools for web scraping (Bright Data) and semantic search (Zvec). Use select_relevant_tool to pick the right tool for the user's query; then run that tool and return its output.",
    json_response=True,
    port=PORT,
)

# Tool definitions registered in Zvec for semantic tool selection
TOOL_DEFINITIONS = [
    {
        "name": "select_relevant_tool",
        "description": "Given a user query, select the most relevant tool(s) to use. Call this first to decide which tool to invoke (e.g. scrape a URL, search stored documents).",
    },
    {
        "name": "scrape_url",
        "description": "Scrape a web page by URL using Bright Data (or direct HTTP). Returns extracted text and title. Use when the user wants to fetch or read content from a URL.",
    },
    {
        "name": "search_documents",
        "description": "Semantic search over documents stored in Qdrant. Use when the user asks to search, find, or query previously stored or scraped content.",
    },
    {
        "name": "store_document",
        "description": "Store a piece of text in the vector database (Qdrant) for later semantic search. Use when the user wants to save or index content (e.g. after scraping).",
    },
]


def _init():
    try:
        ensure_collections()
        init_tools_registry(TOOL_DEFINITIONS)
        logger.info("MCP Server 2: tools registry initialized in Zvec")
    except Exception as e:
        logger.warning("Could not init Qdrant/tools registry (Qdrant may be down): %s", e)


@mcp.tool(
    description="Select the most relevant tool(s) for the user's query. Returns tool names and descriptions with relevance scores. Call this first to decide which tool to run."
)
def get_relevant_tools(user_query: str, top_k: int = 3) -> list[dict]:
    """
    Given the user's query, return the most relevant tools (name, description, score).
    The client can then invoke the top tool and return its output.
    """
    return select_relevant_tool(user_query, top_k=top_k)


@mcp.tool(
    description="Scrape a web page at the given URL. Uses Bright Data proxy if configured. Returns extracted text and page title."
)
def scrape_page(url: str, timeout: float = 30.0) -> dict:
    """Fetch URL and extract text. Set BRIGHT_DATA_PROXY for Bright Data at scale."""
    return scrape_url(url, timeout=timeout)


@mcp.tool(description="Semantic search over documents stored in Qdrant. Returns matching text snippets and metadata.")
def search_stored_documents(query: str, limit: int = 5) -> list[dict]:
    """Search previously stored documents by meaning (vector search)."""
    return search_documents(query, limit=limit)


@mcp.tool(description="Store a document (text) in the vector database for later semantic search.")
def store_document(text: str, metadata: dict | None = None, doc_id: str | None = None) -> dict:
    """Add text to Qdrant. Optional metadata dict and doc_id for reference."""
    return add_document(text, metadata=metadata or None, doc_id=doc_id)


# Alias for workflow: "select a relevant tool" → same as get_relevant_tools
@mcp.tool(
    description="Select which tool is relevant for the user's query. Returns the best-matching tool(s) with scores."
)
def select_relevant_tool_for_query(user_query: str, top_k: int = 3) -> list[dict]:
    """Select relevant tool(s) for the given user query. Use before invoking a specific tool."""
    return select_relevant_tool(user_query, top_k=top_k)


_init()

if __name__ == "__main__":
    print(f"MCP Server 2 starting on port {PORT} (Bright Data + Qdrant)", flush=True)
    mcp.run(transport="streamable-http")
