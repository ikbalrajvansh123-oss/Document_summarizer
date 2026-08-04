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

# 1. FIX: Move startup logic into the Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading embedding model (first run downloads it once)...")
    core.get_embedder()
    core.get_collection()
    print("Ready.")

    print("Scanning uploads/ folder for new .txt files...")
    try:
        _startup_result = core.scan_folder()
        # Using .get() is safer in case the dictionary structure is missing a key
        if _startup_result.get("ingested"):
            print(f"  Ingested {len(_startup_result['ingested'])} new file(s): {[r['filename'] for r in _startup_result['ingested']]}")
        if _startup_result.get("skipped"):
            print(f"  Already indexed (skipped): {_startup_result['skipped']}")
        if _startup_result.get("errors"):
            print(f"  Errors: {_startup_result['errors']}")
    except Exception as e:
        print(f"Warning: Startup scan failed, but server will still start. Error: {e}")
    
    print("Folder scan complete.")
    
    # Yield hands control back to FastAPI to start accepting requests
    yield 
    
    # (Optional) You can put cleanup/shutdown logic here after the yield

# Pass the lifespan manager to the app
app = FastAPI(title="Local RAG System", lifespan=lifespan)

# 2. FIX: Ensure static directory exists before mounting to prevent RuntimeErrors
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

class QueryRequest(BaseModel):
    question: str
    top_k: int = core.TOP_K_DEFAULT
    file_id: Optional[str] = None

@app.get("/")
def serve_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.get("/health")
def health():
    return {"status": "ok", "lm_studio": "connected" if core.check_lm_studio_health() else "not reachable"}

@app.post("/scan-folder")
def scan_folder():
    """Re-scan the uploads/ folder and ingest any new .txt files found there."""
    return core.scan_folder()

@app.get("/files")
def list_files():
    return core.list_files()

@app.delete("/files/{file_id}")
def delete_file(file_id: str):
    try:
        filename = core.delete_file(file_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"message": f"'{filename}' and all its chunks have been deleted."}

@app.post("/query")
def run_query(req: QueryRequest):
    try:
        return core.query(req.question, req.top_k, req.file_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except core.LMStudioError as e: 
        raise HTTPException(status_code=503, detail=str(e))