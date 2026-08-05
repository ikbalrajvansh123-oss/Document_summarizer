
import json
import os
import re
import uuid
from datetime import datetime
from typing import Optional

import chromadb
import requests
from sentence_transformers import SentenceTransformer

# Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
FILES_DB_PATH = os.path.join(BASE_DIR, "files_db.json")

LM_STUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"
LM_STUDIO_MODEL = "qwen3-1.7b"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

CHUNK_SIZE_WORDS = 200
CHUNK_OVERLAP_WORDS = 40
TOP_K_DEFAULT = 4

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)


# Lazy singletons (loaded once, reused across calls)
_embedder = None
_chroma_client = None
_collection = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder


def get_collection():
    global _chroma_client, _collection
    if _collection is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = _chroma_client.get_or_create_collection(name="documents")
    return _collection


# File metadata "database" (simple JSON)
def load_files_db() -> dict:
    if os.path.exists(FILES_DB_PATH):
        with open(FILES_DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_files_db(db: dict) -> None:
    with open(FILES_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


# Chunking
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_into_sentences(paragraph: str) -> list[str]:
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(paragraph.strip()) if s.strip()]
    return sentences


def _word_sliding_window(words: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Last-resort fallback for a single sentence/paragraph with no punctuation to split on.
    Guarantees full coverage: since step <= chunk_size always, no word is ever skipped."""
    chunks = []
    step = max(chunk_size - overlap, 1)
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + chunk_size]))
        if i + chunk_size >= len(words):
            break
        i += step
    return chunks


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_WORDS, overlap: int = CHUNK_OVERLAP_WORDS) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []

    for para in paragraphs:
        words = para.split()
        if not words:
            continue

        if len(words) <= chunk_size:
            chunks.append(para)
            continue

        sentences = _split_into_sentences(para)
        if len(sentences) <= 1:
            # no sentence punctuation to split on - fall back to word-level windowing
            chunks.extend(_word_sliding_window(words, chunk_size, overlap))
            continue

        current_sentences: list[str] = []
        current_word_count = 0

        for sentence in sentences:
            sentence_words = sentence.split()

            if len(sentence_words) > chunk_size:
                # this single sentence alone is bigger than a whole chunk - flush what we have,
                # then fall back to word-level windowing just for this oversized sentence
                if current_sentences:
                    chunks.append(" ".join(current_sentences))
                    current_sentences, current_word_count = [], 0
                chunks.extend(_word_sliding_window(sentence_words, chunk_size, overlap))
                continue

            if current_word_count + len(sentence_words) > chunk_size and current_sentences:
                chunks.append(" ".join(current_sentences))
                # carry trailing sentences into the next chunk as overlap, without exceeding it
                carried: list[str] = []
                carried_word_count = 0
                for s in reversed(current_sentences):
                    w = len(s.split())
                    if carried_word_count + w > overlap:
                        break
                    carried.insert(0, s)
                    carried_word_count += w
                current_sentences = carried
                current_word_count = carried_word_count

            current_sentences.append(sentence)
            current_word_count += len(sentence_words)

        if current_sentences:
            chunks.append(" ".join(current_sentences))

    return chunks


# Queries like these need the WHOLE document, not just the top-K most similar chunks -
# similarity search can never reliably answer "how many" / "list all" style questions,
# since the correct chunks aren't necessarily the ones most similar to the question text.
_AGGREGATE_KEYWORDS = [
    "how many", "list all", "list of", "all employee", "all names", "all the",
    "every ", "count", "total number", "each employee", "names of", "who are all",
    "how much people", "how many people", "how many person", "how many chapter",
    "summarize", "summary of", "overview of the", "all chapters",
]


def is_aggregate_query(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in _AGGREGATE_KEYWORDS)


# LM Studio call
class LMStudioError(Exception):
    pass


def ask_lm_studio(question: str, context: str) -> str:
    system_prompt = (
        "You are a helpful assistant answering questions using ONLY the context provided below.\n"
        "Rules you must follow strictly:\n"
        "1. Read the ENTIRE context carefully before answering, not just the first part.\n"
        "2. If the question asks for a count, a list, or 'all' of something, go through every part of "
        "the context and include every matching item. Never guess a number or stop early.\n"
        "3. Each entry/paragraph in the context may describe a DIFFERENT person or item. Never mix or "
        "borrow a detail (like a location, salary, or role) from one entry and attach it to a different "
        "person just because they appear near each other. Only state a fact about a person if it is "
        "written directly in that person's own entry.\n"
        "4. If the answer is not present in the context, say clearly that you don't know based on the "
        "given documents, rather than guessing.\n"
        "Be concise and accurate."
    )
    payload = {
        "model": LM_STUDIO_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\n---\n\nQuestion: {question}"},
        ],
        "temperature": 0.3,
        "max_tokens": 600,
    }
    try:
        resp = requests.post(LM_STUDIO_URL, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.ConnectionError:
        raise LMStudioError(
            "Could not connect to LM Studio. Please check that LM Studio's local server is running "
            "(http://127.0.0.1:1234) and a model is loaded."
        )
    except Exception as e:
        raise LMStudioError(f"LM Studio error: {e}")


def check_lm_studio_health() -> bool:
    try:
        r = requests.get("http://127.0.0.1:1234/v1/models", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# SESSION-BASED TEXT INGESTION  (new — replaces file upload)
# ─────────────────────────────────────────────────────────────────────────────

def clear_session(session_id: str) -> int:
    """Remove all ChromaDB chunks that belong to `session_id`.
    Returns the number of deleted chunks."""
    col = get_collection()
    try:
        existing = col.get(where={"session_id": session_id}, include=[])
        ids_to_delete = existing.get("ids", [])
        if ids_to_delete:
            col.delete(ids=ids_to_delete)
        return len(ids_to_delete)
    except Exception:
        return 0


def ingest_text(text: str, session_id: str, label: str = "input") -> dict:
    """Chunk, embed and store raw text under a given session_id.
    Any previous data for this session_id is wiped first.
    Returns: { session_id, label, num_chunks }
    """
    # 1. Clear previous data for this session
    clear_session(session_id)

    # 2. Chunk
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("The provided text appears to be empty — no content was found.")

    # 3. Embed & store
    embedder = get_embedder()
    embeddings = embedder.encode(chunks, convert_to_numpy=True).tolist()
    ids = [f"{session_id}_{i}" for i in range(len(chunks))]
    metadatas = [
        {"session_id": session_id, "label": label, "chunk_index": i}
        for i in range(len(chunks))
    ]

    get_collection().add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)

    return {"session_id": session_id, "label": label, "num_chunks": len(chunks)}


def generate_auto_summary(text: str) -> str:
    """Ask LM Studio to produce a simple, plain-language summary of the given text.
    Falls back gracefully if LM Studio is not reachable."""
    system_prompt = (
        "/no_think\n"
        "You are a helpful assistant. Read the text and write a clear, simple summary.\n"
        "Rules you MUST follow:\n"
        "1. Write in very simple, everyday language — like explaining to a friend.\n"
        "2. Do NOT use bullet points, markdown symbols (* # -), or bold/italic text. "
        "Write only in plain normal sentences.\n"
        "3. For each person or employee in the text, write one short paragraph about them. "
        "Example: 'Amandeep Singh works as a Senior Developer in the Engineering department. "
        "He is based in Chandigarh and earns Rs 85,000 per month. He has 6 years of experience.'\n"
        "4. Cover all available details: name, job title, department, location, salary, experience, skills.\n"
        "5. If there are multiple people, separate each person's paragraph with a blank line.\n"
        "6. Do NOT invent or guess any information not present in the text.\n"
        "7. Be concise. Each person's paragraph should be 2-4 sentences.\n"
        "8. Write the summary directly — do not add any introductory phrase like 'Here is a summary'."
    )
    # For summary we send the full text directly (not chunked context)
    # Truncate to ~6000 chars to stay within token limits
    truncated = text[:6000]
    payload = {
        "model": LM_STUDIO_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"/no_think\n\nText to summarize:\n\n{truncated}"},
        ],
        "temperature": 0.1,
        "max_tokens": 1000,
    }
    try:
        resp = requests.post(LM_STUDIO_URL, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.ConnectionError:
        raise LMStudioError(
            "Could not connect to LM Studio. Please check that LM Studio's local server is running."
        )
    except Exception as e:
        raise LMStudioError(f"LM Studio error during summary: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# QUERY (session-scoped)
# ─────────────────────────────────────────────────────────────────────────────

UNANSWERED_LOG_PATH = os.path.join(BASE_DIR, "unanswered.json")
NOT_FOUND_MESSAGE = "This information was not found in the provided text."

_STOPWORDS = {
    "where", "who", "what", "when", "why", "how", "does", "do", "is", "are", "was", "were",
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "his", "her", "their",
    "lives", "live", "living", "tell", "me", "please", "list",
    "all", "has", "have", "had", "works", "work", "working", "here", "many", "much", "count",
    "summarize", "summary", "about", "employee", "employees", "person", "people", "company",
    "team", "you", "give", "show", "find", "get", "can", "could", "would", "its", "into",
}

CONFIDENCE_THRESHOLD = 0.50
MIN_GROUNDING_COVERAGE = CONFIDENCE_THRESHOLD


def get_focus_terms(question: str) -> set[str]:
    words = re.findall(r"[a-zA-Z]+", question.lower())
    return {w for w in words if len(w) >= 3 and w not in _STOPWORDS}


def context_coverage(context: str, focus_terms: set[str]) -> float:
    if not focus_terms:
        return 1.0
    context_lower = context.lower()
    found = sum(1 for term in focus_terms if term in context_lower)
    return found / len(focus_terms)


def log_unanswered(question: str, coverage: float, sources: list[dict]) -> None:
    entries = []
    if os.path.exists(UNANSWERED_LOG_PATH):
        try:
            with open(UNANSWERED_LOG_PATH, "r", encoding="utf-8") as f:
                entries = json.load(f)
        except (json.JSONDecodeError, OSError):
            entries = []

    entries.append(
        {
            "question": question,
            "timestamp": datetime.utcnow().isoformat(),
            "coverage": round(coverage, 2),
            "checked_sources": [s.get("label") for s in sources],
        }
    )

    with open(UNANSWERED_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


_ANSWER_FILLER_WORDS = {
    "the", "this", "that", "these", "those", "it", "if", "based", "according", "she", "he",
    "his", "her", "they", "we", "you", "your", "is", "are", "was", "were", "not", "yes", "no",
    "please", "note", "here", "there", "provided", "context", "document", "documents", "given",
}


def extract_claim_tokens(answer: str, question: str) -> tuple[list[str], list[str]]:
    question_lower = question.lower()
    word_tokens = re.findall(r"\b[A-Za-z][a-zA-Z]{2,}\b", answer)
    claims = []
    for tok in word_tokens:
        if not tok[0].isupper():
            continue
        if tok.lower() in _ANSWER_FILLER_WORDS:
            continue
        if tok.lower() in question_lower:
            continue
        claims.append(tok)

    numbers = re.findall(r"\d[\d,]*", answer)
    return claims, numbers


def answer_confidence(answer: str, context: str, question: str) -> float:
    claims, numbers = extract_claim_tokens(answer, question)
    all_items = claims + numbers
    if not all_items:
        return 1.0
    context_lower = context.lower()
    verified = 0
    for tok in claims:
        if tok.lower() in context_lower:
            verified += 1
    for num in numbers:
        # Normalize: strip commas so "72,000" matches "72000" in context
        num_plain = num.replace(",", "")
        if num in context or num_plain in context:
            verified += 1
    return verified / len(all_items)


def query(question: str, top_k: int = TOP_K_DEFAULT, session_id: Optional[str] = None) -> dict:
    """Query the ChromaDB collection, optionally scoped to a session_id."""
    col = get_collection()

    where_filter = {"session_id": session_id} if session_id else None
    aggregate_mode = is_aggregate_query(question)

    # Check there is data for this session
    check = col.get(where=where_filter, include=[]) if where_filter else col.get(include=[])
    if not check.get("ids"):
        raise ValueError("No text has been loaded yet. Please paste your text in the left panel first.")

    if aggregate_mode:
        all_data = col.get(where=where_filter, include=["documents", "metadatas"])
        documents = all_data.get("documents", [])
        metadatas = all_data.get("metadatas", [])

        paired = sorted(
            zip(documents, metadatas),
            key=lambda dm: (dm[1].get("label", ""), dm[1].get("chunk_index", 0)),
        )
        documents = [d for d, _ in paired]
        metadatas = [m for _, m in paired]
        distances = [None] * len(documents)
    else:
        embedder = get_embedder()
        query_embedding = embedder.encode([question], convert_to_numpy=True).tolist()[0]
        results = col.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, 20),
            where=where_filter,
        )
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

    if not documents:
        return {"answer": "No relevant content was found in the provided text for this question.", "sources": []}

    focus_terms = get_focus_terms(question) if not aggregate_mode else set()

    if focus_terms:
        matching_idx = [i for i, doc in enumerate(documents) if any(t in doc.lower() for t in focus_terms)]
        if matching_idx:
            documents = [documents[i] for i in matching_idx]
            metadatas = [metadatas[i] for i in matching_idx]
            distances = [distances[i] for i in matching_idx]

    context = "\n\n---\n\n".join(documents)

    sources = [
        {
            "label": meta.get("label"),
            "chunk_index": meta.get("chunk_index"),
            "preview": doc[:200],
            "distance": dist,
        }
        for doc, meta, dist in zip(documents, metadatas, distances)
    ]

    if not aggregate_mode:
        pre_confidence = context_coverage(context, focus_terms)
        if pre_confidence < CONFIDENCE_THRESHOLD:
            log_unanswered(question, pre_confidence, sources)
            return {
                "answer": NOT_FOUND_MESSAGE,
                "sources": sources,
                "mode": "not_found_in_document",
                "grounded": False,
                "confidence": round(pre_confidence, 2),
            }

    answer = ask_lm_studio(question, context)

    if not aggregate_mode:
        post_confidence = answer_confidence(answer, context, question)
        if post_confidence < CONFIDENCE_THRESHOLD:
            log_unanswered(question, post_confidence, sources)
            return {
                "answer": NOT_FOUND_MESSAGE,
                "sources": sources,
                "mode": "not_found_in_document",
                "grounded": False,
                "confidence": round(post_confidence, 2),
            }
    else:
        post_confidence = 1.0

    return {
        "answer": answer,
        "sources": sources,
        "mode": "full_document" if aggregate_mode else "similarity_search",
        "grounded": True,
        "confidence": round(post_confidence, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Legacy file-based helpers (kept for backward compat, not used by new UI)
# ─────────────────────────────────────────────────────────────────────────────

def ingest_file(filepath: str, filename: str) -> dict:
    """Chunk, embed, and store a text file from disk. Returns metadata dict."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    chunks = chunk_text(text)
    if not chunks:
        raise ValueError(f"'{filename}' appears to be empty — no text content was found.")

    file_id = uuid.uuid4().hex
    embedder = get_embedder()
    embeddings = embedder.encode(chunks, convert_to_numpy=True).tolist()
    ids = [f"{file_id}_{i}" for i in range(len(chunks))]
    metadatas = [{"file_id": file_id, "filename": filename, "chunk_index": i} for i in range(len(chunks))]

    get_collection().add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)

    db = load_files_db()
    db[file_id] = {
        "filename": filename,
        "path": filepath,
        "num_chunks": len(chunks),
        "uploaded_at": datetime.utcnow().isoformat(),
    }
    save_files_db(db)

    return {"file_id": file_id, "filename": filename, "num_chunks": len(chunks)}


def scan_folder() -> dict:
    db = load_files_db()
    already_known = {meta["filename"] for meta in db.values()}

    ingested = []
    skipped = []
    errors = []

    for entry in os.scandir(UPLOAD_DIR):
        if not entry.is_file():
            continue
        if not entry.name.lower().endswith(".txt"):
            continue
        if entry.name in already_known:
            skipped.append(entry.name)
            continue
        try:
            result = ingest_file(entry.path, entry.name)
            ingested.append(result)
        except Exception as exc:
            errors.append({"filename": entry.name, "error": str(exc)})

    return {"ingested": ingested, "skipped": skipped, "errors": errors}


def list_files() -> list[dict]:
    db = load_files_db()
    return [{"file_id": fid, **meta} for fid, meta in db.items()]


def delete_file(file_id: str) -> str:
    db = load_files_db()
    if file_id not in db:
        raise KeyError("File ID not found.")

    get_collection().delete(where={"file_id": file_id})

    meta = db.pop(file_id)
    if os.path.exists(meta["path"]):
        os.remove(meta["path"])

    save_files_db(db)
    return meta["filename"]
