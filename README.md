# Local RAG System (LM Studio + lfm2.5-350m)

Ye ek poora local RAG (Retrieval-Augmented Generation) system hai:
- `.txt` file upload karo → automatically chunk + embed + save ho jaandi hai (disk te, persistent — restart baad vi rehnda hai)
- File delete karo → uske saare chunks/embeddings vector store cho v delete ho jaande hain
- Koi v sawaal pucho → sabse relevant chunks dhoond ke (semantic search, sirf keyword match nahi) LM Studio nu context de ke answer generate karda hai

## Architecture (kive kaam karda hai)

1. **Chunking**: Har file 200-word chunks vich divide hundi hai (40-word overlap, taaki context na tuttay).
2. **Embeddings**: Har chunk da vector `sentence-transformers` (model: `all-MiniLM-L6-v2`) naal locally banda hai — ye small, fast, aur free model hai, koi API key nahi chahida.
3. **Storage**: `ChromaDB` (local persistent vector database) — `chroma_db/` folder vich save hunda hai. File metadata `files_db.json` vich.
4. **Delete**: File delete karan te `collection.delete(where={"file_id": ...})` call hundi hai — sirf ohi file de chunks hatde hain, baaki safe rehnde han.
5. **Answering**: Query embed karke top-K similar chunks retrieve hunde han, fir wo context LM Studio (`http://127.0.0.1:1234`) nu bheja janda hai final answer generate karan layi.

## Setup

### 1. LM Studio
- LM Studio khol ke `lfm2.5-350m` model load karo.
- **Local Server** tab vich jaa ke server start karo (default: `http://127.0.0.1:1234`).
- Confirm karo ki server ON hai.

### 2. Python environment
```bash
cd rag-project
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Pehli baar chalaan te `sentence-transformers` apna embedding model (~90MB) download karega — internet chahida hoga sirf ek baar.

### 3a. FastAPI (custom HTML UI) chalao
```bash
uvicorn app:app --reload --port 8000
```
Browser vich kholo: `http://127.0.0.1:8000`

### 3b. YA Streamlit interface chalao (testing ke liye simpler)
```bash
streamlit run streamlit_app.py
```
Ye automatically browser vich khul jaega (default: `http://localhost:8501`).

Dono interfaces same `rag_core.py` use karde han — same uploaded files, same chunks, same vector store. Koi v ik chalao, ya dono (alag terminals vich) — data shared rahega.

Upar status indicator dasse ga ki LM Studio connected hai ki nahi.

## Use kive karna hai

1. Left panel cho `.txt` file upload karo.
2. Chat box vich sawaal likho — Enter dabao ya "Ask" button.
3. Answer de heth "source chunks" expand karke dekh sakde ho ki kehda chunk use hoya.
4. File delete karni ho ta list cho ✕ button dabao.

## Customize (agar chaheda)

- `app.py` vich `CHUNK_SIZE_WORDS` / `CHUNK_OVERLAP_WORDS` badal ke chunking tune kar sakde ho.
- `TOP_K_DEFAULT` badal ke kinne chunks context vich jaan control kar sakde ho (zyada chunks = zyada context, par slower + LM Studio ton bade prompt).
- `.pdf` ya `.docx` support add karna hove ta `upload_file()` vich text-extraction logic add karna padu (abhi sirf `.txt`).

## Grounding check & unanswered.json

Small local models can sometimes answer confidently even when the retrieved chunks don't
actually contain the subject being asked about (e.g. mixing up two different people). To guard
against this, before calling LM Studio the system checks whether the key terms from your question
(like a person's name) actually appear in the retrieved context:

- If they **do** → the question is answered normally.
- If they **don't** → the system skips the LLM call, tells you clearly that *"This information
  was not found in the uploaded document(s)"*, and logs the question to `unanswered.json` in the
  project folder (with a timestamp and which files were checked).

This does not apply to "how many / list all / summarize" style questions, since those already use
the full document as context (see below), so there's nothing to miss.

You can review `unanswered.json` any time to see what questions your documents don't currently
cover, so you know what to add to them later.

## Sentence-aware chunking + 75% confidence gate

- **Chunking**: paragraphs are grouped by whole sentences up to the chunk size, carrying a few
  trailing sentences into the next chunk as overlap. A sentence is never cut in half, and every
  path guarantees full coverage - verified with 0 words missing on `details.txt` (475/475 words)
  and on a 2,400-word synthetic paragraph (60 chunks, 0 missing).
- **75% confidence gate**: every answer (aggregate/full-document questions excluded) is scored
  from 0-1 in two places - before calling the LLM (does the context mention the subject at all?)
  and after (does every specific name/place/number in the generated answer actually appear in
  the context?). Anything scoring below **0.75** is withheld and replaced with
  *"This information was not found in the uploaded document(s)"*, and logged to
  `unanswered.json` along with its confidence score. This trades a small amount of extra
  processing time for much higher answer reliability.

## Full data coverage guarantee

Two guarantees are built in, on purpose trading some speed for completeness:

1. **Chunking never drops a word.** Chunking is paragraph-based with a sliding-window fallback
   for long paragraphs, and the window always overlaps the next one - so every word from the
   original `.txt` file ends up in at least one chunk, verified against `details.txt` (475/475
   words present across all chunks, 0 missing).
2. **Aggregate queries ("how many", "list all", "summarize") always use every chunk of the
   document**, with no cap. This can make those specific questions slower on very large files
   (bigger prompt to LM Studio), but nothing gets silently truncated or left out.

One caveat worth knowing: `lfm2.5-350m` (like any local model) has a limited context window. On
a very large document, an aggregate query might hit that limit even though every chunk was
included in the request. If that happens, consider a model with a larger context window in LM
Studio, or ask about one file at a time using the `file_id` filter.

## Notes

- Ye system pura **local** hai — koi cloud API nahi, koi data bahar nahi jaanda.
- Agar "LM Studio not reachable" dikhe, check karo LM Studio da local server ON hai te port `1234` match karda hai.
- Bade `.txt` files (jaise poori book) v chalengi — bas upload time thoda zyada lagega embedding banaan vaaste.
