
import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import rag_core as core

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="Local RAG System")

print("Loading embedding model (first run downloads it once)...")
core.get_embedder()
core.get_collection()
print("Ready.")

# Auto-scan uploads/ folder on startup
print("Scanning uploads/ folder for new .txt files...")
_startup_result = core.scan_folder()
if _startup_result["ingested"]:
    print(f"  Ingested {len(_startup_result['ingested'])} new file(s): {[r['filename'] for r in _startup_result['ingested']]}")
if _startup_result["skipped"]:
    print(f"  Already indexed (skipped): {_startup_result['skipped']}")
if _startup_result["errors"]:
    print(f"  Errors: {_startup_result['errors']}")
print("Folder scan complete.")


class QueryRequest(BaseModel):
    question: str
    top_k: int = core.TOP_K_DEFAULT
    file_id: Optional[str] = None


@app.get("/")
def serve_index():
    return FileResponse(os.path.join(BASE_DIR, "static", "index.html"))


@app.get("/health")
def health():
    return {"status": "ok", "lm_studio": "connected" if core.check_lm_studio_health() else "not reachable"}


@app.post("/scan-folder")
def scan_folder():
    """Re-scan the uploads/ folder and ingest any new .txt files found there."""
    result = core.scan_folder()
    return result


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


app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
