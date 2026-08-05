import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import rag_core as core

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading embedding model (first run downloads it once)...")
    core.get_embedder()
    core.get_collection()
    print("Ready.")
    yield


app = FastAPI(title="Local RAG System — Text Input Mode", lifespan=lifespan)

os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ── Request / Response models ─────────────────────────────────────────────────

class IngestTextRequest(BaseModel):
    text: str
    session_id: str
    label: Optional[str] = "Employee Data"


class QueryRequest(BaseModel):
    question: str
    session_id: str
    top_k: int = core.TOP_K_DEFAULT


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def serve_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "lm_studio": "connected" if core.check_lm_studio_health() else "not reachable",
    }


@app.post("/ingest-text")
def ingest_text(req: IngestTextRequest):
    """
    Receive raw text from the frontend, ingest it into ChromaDB under the given
    session_id (clearing any previous session data first), then auto-generate a
    summary using LM Studio and return it.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        ingest_result = core.ingest_text(req.text, req.session_id, req.label)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Auto-generate summary (best-effort — if LM Studio is down, return a fallback)
    try:
        summary = core.generate_auto_summary(req.text)
    except core.LMStudioError as e:
        summary = f"[Auto-summary unavailable — LM Studio not reachable]\n\nError: {e}"

    return {
        "session_id": req.session_id,
        "label": ingest_result["label"],
        "num_chunks": ingest_result["num_chunks"],
        "summary": summary,
    }


@app.post("/query")
def run_query(req: QueryRequest):
    """Query the text loaded under the given session_id."""
    try:
        return core.query(req.question, req.top_k, req.session_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except core.LMStudioError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    """Remove all ChromaDB data associated with a session."""
    deleted = core.clear_session(session_id)
    return {"session_id": session_id, "deleted_chunks": deleted}