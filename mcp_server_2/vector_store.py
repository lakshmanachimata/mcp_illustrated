"""Zvec vector store: tool selection and document search (in-process, no separate server)."""
import logging
import os
import uuid
from pathlib import Path
from typing import Any

import zvec
from config import (
    DOCUMENTS_COLLECTION_PATH,
    EMBEDDING_MODEL,
    TOOLS_COLLECTION_PATH,
    VECTOR_SIZE,
)

logger = logging.getLogger(__name__)

_embedder = None
_tools_coll = None
_docs_coll = None

VECTOR_FIELD = "embedding"


def _get_embedder():
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedder = SentenceTransformer(EMBEDDING_MODEL)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load embedding model '{EMBEDDING_MODEL}'. pip install sentence-transformers"
            ) from e
    return _embedder


def _embed(text: str | list[str]):
    model = _get_embedder()
    if isinstance(text, str):
        return model.encode(text, normalize_embeddings=True).tolist()
    return model.encode(text, normalize_embeddings=True).tolist()


def _tools_schema():
    try:
        from zvec.model.param import FlatIndexParam
        from zvec.typing import MetricType
        index_param = FlatIndexParam(metric_type=MetricType.COSINE)
    except (ImportError, AttributeError):
        index_param = None
    return zvec.CollectionSchema(
        name="tools",
        fields=[
            zvec.FieldSchema("name", zvec.DataType.STRING, nullable=True),
            zvec.FieldSchema("description", zvec.DataType.STRING, nullable=True),
        ],
        vectors=zvec.VectorSchema(
            VECTOR_FIELD,
            zvec.DataType.VECTOR_FP32,
            dimension=VECTOR_SIZE,
            index_param=index_param,
        ),
    )


def _documents_schema():
    try:
        from zvec.model.param import FlatIndexParam
        from zvec.typing import MetricType
        index_param = FlatIndexParam(metric_type=MetricType.COSINE)
    except (ImportError, AttributeError):
        index_param = None
    return zvec.CollectionSchema(
        name="documents",
        fields=[
            zvec.FieldSchema("text", zvec.DataType.STRING, nullable=True),
            zvec.FieldSchema("doc_id", zvec.DataType.STRING, nullable=True),
        ],
        vectors=zvec.VectorSchema(
            VECTOR_FIELD,
            zvec.DataType.VECTOR_FP32,
            dimension=VECTOR_SIZE,
            index_param=index_param,
        ),
    )


def _get_tools_collection():
    global _tools_coll
    if _tools_coll is None:
        Path(TOOLS_COLLECTION_PATH).parent.mkdir(parents=True, exist_ok=True)
        if os.path.exists(TOOLS_COLLECTION_PATH):
            _tools_coll = zvec.open(TOOLS_COLLECTION_PATH)
        else:
            _tools_coll = zvec.create_and_open(TOOLS_COLLECTION_PATH, _tools_schema())
        logger.info("Opened Zvec tools collection at %s", TOOLS_COLLECTION_PATH)
    return _tools_coll


def _get_documents_collection():
    global _docs_coll
    if _docs_coll is None:
        Path(DOCUMENTS_COLLECTION_PATH).parent.mkdir(parents=True, exist_ok=True)
        if os.path.exists(DOCUMENTS_COLLECTION_PATH):
            _docs_coll = zvec.open(DOCUMENTS_COLLECTION_PATH)
        else:
            _docs_coll = zvec.create_and_open(DOCUMENTS_COLLECTION_PATH, _documents_schema())
        logger.info("Opened Zvec documents collection at %s", DOCUMENTS_COLLECTION_PATH)
    return _docs_coll


def ensure_collections():
    """Create or open tools and documents collections."""
    _get_tools_collection()
    _get_documents_collection()


def init_tools_registry(tools: list[dict[str, str]]):
    """
    Register tool definitions in Zvec for semantic tool selection.
    Each tool: {"name": str, "description": str}.
    """
    if not tools:
        return
    coll = _get_tools_collection()
    docs = []
    for t in tools:
        name = t.get("name", "").strip()
        desc = t.get("description", "").strip()
        if not name:
            continue
        text = f"{name}: {desc}"
        vector = _embed(text)
        doc_id = f"tool_{name}"
        docs.append(
            zvec.Doc(
                id=doc_id,
                vectors={VECTOR_FIELD: vector},
                fields={"name": name, "description": desc},
            )
        )
    if docs:
        coll.upsert(docs)
        logger.info("Upserted %s tool definitions into Zvec tools collection", len(docs))


def select_relevant_tool(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    """Given a user query, return the most relevant tools (name, description, score)."""
    query = (query or "").strip()
    if not query:
        return []
    try:
        ensure_collections()
        coll = _get_tools_collection()
        vector = _embed(query)
        results = coll.query(
            vectors=zvec.VectorQuery(VECTOR_FIELD, vector=vector),
            topk=top_k,
        )
        return [
            {
                "name": d.field("name") or "",
                "description": d.field("description") or "",
                "score": float(d.score) if d.score is not None else 0.0,
            }
            for d in results
        ]
    except Exception as e:
        logger.exception("select_relevant_tool failed")
        return [{"error": str(e)}]


def add_document(text: str, metadata: dict[str, Any] | None = None, doc_id: str | None = None) -> dict[str, Any]:
    """Embed and store a document in Zvec. Returns id and status."""
    text = (text or "").strip()
    if not text:
        return {"success": False, "error": "text is required"}
    try:
        ensure_collections()
        coll = _get_documents_collection()
        vector = _embed(text)
        id_str = doc_id or str(uuid.uuid4())
        payload_text = text[:10000]
        doc = zvec.Doc(
            id=id_str,
            vectors={VECTOR_FIELD: vector},
            fields={"text": payload_text, "doc_id": id_str},
        )
        coll.upsert(doc)
        return {"success": True, "id": id_str, "collection": "documents"}
    except Exception as e:
        logger.exception("add_document failed")
        return {"success": False, "error": str(e)}]


def search_documents(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """Semantic search over stored documents. Returns list of {text, payload, score}."""
    query = (query or "").strip()
    if not query:
        return []
    try:
        ensure_collections()
        coll = _get_documents_collection()
        vector = _embed(query)
        results = coll.query(
            vectors=zvec.VectorQuery(VECTOR_FIELD, vector=vector),
            topk=limit,
        )
        out = []
        for d in results:
            text_val = d.field("text") or ""
            out.append({
                "text": text_val[:2000],
                "payload": {"text": text_val, "doc_id": d.field("doc_id")},
                "score": float(d.score) if d.score is not None else 0.0,
            })
        return out
    except Exception as e:
        logger.exception("search_documents failed")
        return [{"error": str(e)}]
