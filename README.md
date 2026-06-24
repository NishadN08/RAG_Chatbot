# RAG Chatbot 🤖👨‍💻 — FSU Department of Scientific Computing

An interactive **Retrieval-Augmented Generation (RAG)** chatbot that answers questions about the **FSU Department of Scientific Computing** (`sc.fsu.edu`) by combining document retrieval with local LLM reasoning.

- Built with **LangChain 1.x**, **LangGraph**, and **Ollama** (`gpt-oss:20b`), with full real-time, token-by-token streaming.
- Ingests crawled website content stored as JSONL, splits it into a **"new" (current)** and **"old" (archived)** corpus, embeds it with a HuggingFace sentence-transformer, and reranks retrieved chunks with a cross-encoder.
- Automatically rewrites follow-up questions into standalone questions (using chat history, the current date/semester, and a keyword dictionary for spelling correction) before retrieval.
- Served over **FastAPI** as a Server-Sent-Events (SSE) streaming API (`api.py`).

> ⚠️ This project requires a **local Ollama installation**, a **Chrome browser** (for crawling), and a **valid FSU CAS login** (for crawling protected pages). It is designed to run on a machine with enough RAM/VRAM to host the `gpt-oss:20b` model.

---

## Table of Contents

- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Prerequisites](#prerequisites)
- [Setup Guide](#setup-guide)
- [Data Pipeline (Run in This Order)](#data-pipeline-run-in-this-order)
  - [Step 1 — Crawl `sc.fsu.edu` with `main_crawler.py`](#step-1--crawl-scfsuedu-with-main_crawlerpy)
  - [Step 2 — Manually split the crawl into `new.jsonl` and `old.jsonl`](#step-2--manually-split-the-crawl-into-newjsonl-and-oldjsonl)
  - [Step 3 — Generate the spelling/keyword dictionary with `keywords_gen.py`](#step-3--generate-the-spellingkeyword-dictionary-with-keywords_genpy)
- [Running the Chatbot API (`api.py`)](#running-the-chatbot-api-apipy)
- [API Reference](#api-reference)
- [Configuration Reference](#configuration-reference)
- [Notes, Caveats & Known Issues](#notes-caveats--known-issues)
- [Tech Stack](#tech-stack)

---

## Project Structure

| File / Folder | Purpose |
|---|---|
| `main_crawler.py` | Selenium-based BFS crawler that logs into FSU CAS and scrapes `sc.fsu.edu`, producing a raw `.jsonl` crawl file. |
| `new.jsonl` | **(You create this)** Subset of the crawl containing current/active content. Loaded by `rag.py` as the primary corpus. |
| `old.jsonl` | **(You create this)** Subset of the crawl containing outdated/archived content. Used as a fallback corpus. |
| `keywords_gen.py` | Uses a local Ollama model to extract searchable keywords (faculty names, course codes, research areas, etc.) from `new.jsonl`/`old.jsonl` into a flat keyword list. |
| `clean_keywords.txt` | Output of `keywords_gen.py` — keyword dictionary used for spelling correction during question reformulation. |
| `rag.py` | Core RAG logic: data cleaning, Chroma vector stores, cross-encoder reranking, prompts, and the LangGraph pipeline (`app`). |
| `api.py` | FastAPI app that wraps `rag.py` and exposes a streaming chat endpoint. |
| `requirements.txt` | Python dependencies. |
| `__pycache__/` | Compiled Python bytecode (auto-generated, safe to ignore/delete). |

---

## How It Works

`rag.py` builds a **LangGraph** state machine with the following flow:

```
reformulate → retrieve_new → answer_new ──(no answer found)──► retrieve_old → answer_old ──(no answer found)──► fallback_llm → END
      │                            │                                              │
      └─(answer found)──────────► END                          └─(answer found)──► END
```

1. **`reformulate`** — Rewrites the user's latest question into a standalone question using the last 3 turns of chat history, today's date (to resolve "this semester" → e.g. "Fall 2026"), and fuzzy keyword matches from the keyword vector store (spelling correction).
2. **`retrieve_new` / `answer_new`** — Retrieves the top chunks from the **new** corpus (Chroma + cross-encoder reranking) and answers strictly from that context.
3. If the model responds with *"I could not find any relevant information..."*, the graph falls through to **`retrieve_old` / `answer_old`** (the archived corpus).
4. If that also fails, it falls through to **`fallback_llm`**, which answers using the LLM's general knowledge (scoped to FSU SC by prompt) and points the user to `www.sc.fsu.edu`.

`api.py` exposes this graph over a streaming `/chat/stream` endpoint and emits intermediate debug events (the reformulation prompt, retrieved chunks, fallback prompt) alongside the final answer.

---

## Prerequisites

- **Python 3.10 or 3.11** (recommended, for compatibility with `torch`, `chromadb`, and `transformers`)
- **[Ollama](https://ollama.com/)** installed and running locally
- **Google Chrome** installed (only needed for `main_crawler.py` — `webdriver-manager` auto-downloads a matching ChromeDriver)
- A valid **FSU CAS** account if you need to crawl login-gated pages on `sc.fsu.edu`
- Enough local compute/VRAM to run `gpt-oss:20b` via Ollama (the main QA + reformulation model) and `llama3.1:8b` (used only by `keywords_gen.py`)

---

## Setup Guide

1. **Clone the repository**

   ```bash
   git clone https://github.com/NishadN08/RAG_Chatbot.git
   cd RAG_Chatbot
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate      # macOS/Linux
   venv\Scripts\activate         # Windows
   ```

3. **Install Python dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Install and start Ollama, then pull the required models**

   ```bash
   ollama pull gpt-oss:20b      # used by rag.py (QA + question reformulation + fallback)
   ollama pull llama3.1:8b      # used by keywords_gen.py (keyword extraction)
   ollama serve                 # if Ollama isn't already running as a service
   ```

5. **(Optional) LangSmith tracing** — `rag.py` has LangSmith tracing code commented out at the top. If you want call-by-call tracing, install `python-dotenv`/`langsmith` (already in `requirements.txt`), create a `.env` file with `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, and `LANGCHAIN_PROJECT`, and uncomment the relevant lines in `rag.py`.

With dependencies installed, you're ready to build the data pipeline below **before** the chatbot can actually answer anything — `rag.py` will fail to start if `new.jsonl`, `old.jsonl`, or `clean_keywords.txt` are missing.

---

## Data Pipeline (Run in This Order)

The chatbot is only as good as its data. Follow these three steps **in order** before running `api.py`.

### Step 1 — Crawl `sc.fsu.edu` with `main_crawler.py`

```bash
python main_crawler.py
```

What happens:

- A visible Chrome window opens (`HEADLESS = False`) and navigates to the FSU CAS login page.
- Because `WAIT_FOR_MANUAL_LOGIN = True`, **you must manually complete the FSU CAS / Duo 2FA login** in that browser window. The script waits up to `MANUAL_LOGIN_MAX_SECONDS` (default 300s) and then prompts you in the terminal to press **Enter** once you're logged in.
- The crawler then performs a breadth-first crawl of `sc.fsu.edu`, restricted to `ALLOWED_NETLOCS` / `ALLOWED_PATH_PREFIXES`, up to `MAX_PAGES` (5000) and `MAX_DEPTH` (20), respecting `robots.txt`.
- For each page it extracts the main content (stripping nav/footer/ads), titles, mailto emails, anchor texts, internal links, LinkedIn/Google Scholar links, and text from any linked PDFs.
- Results are written **one JSON object per line** to the file named in `OUTPUT_JSONL` (default: `crawl_3-19-26(2).jsonl`). You can edit this constant at the top of `main_crawler.py` before running if you want a different output filename.

> Tip: Crawling can take a long time on a large site — `REQUEST_DELAY = 1.0` adds a 1‑second delay between pages to be polite to the server.

### Step 2 — Manually split the crawl into `new.jsonl` and `old.jsonl`

`rag.py` expects **two** files in the project root:

- **`new.jsonl`** — current/active content (current faculty, current course listings, current semester announcements, current colloquium/newsletter entries, etc.)
- **`old.jsonl`** — outdated/archived content (past newsletters, past colloquium pages, stale semester pages, etc.)

This split is **manual**: open the raw crawl output from Step 1, review the records, and copy each JSON line into whichever of the two files it belongs in. Both files must live in the project root and use the same one‑JSON‑object‑per‑line format produced by the crawler (each record should at least contain `url`, `title`, and `text`).

These two files back **two separate Chroma vector stores** (`chroma_new` / `chroma_old`). At query time, the chatbot searches `new.jsonl` first and only falls back to `old.jsonl` if no answer is found there.

### Step 3 — Generate the spelling/keyword dictionary with `keywords_gen.py`

```bash
python keywords_gen.py
```

What happens:

- The script loads documents from the files listed in `JSON_FILES` and uses a local Ollama model (`OLLAMA_MODEL = "llama3.1:8b"`) to extract searchable keywords (faculty names, research areas, course codes, software, labs, organizations, etc.) in batches of `DOCS_PER_BATCH = 5` documents at a time.
- Extracted keywords are cleaned (numbering/markdown stripped, sentences discarded, deduplicated) and written one per line to `OUTPUT_FILE`.

> ⚠️ **Important — check the filenames before running:** as shipped, `keywords_gen.py` defaults to:
> ```python
> JSON_FILES = ["new1.jsonl", "old1.jsonl"]
> OUTPUT_FILE = "clean_keywords1.txt"
> ```
> but `rag.py` expects the keyword file to be named **`clean_keywords.txt`** and the corpora to be **`new.jsonl`** / **`old.jsonl`** (see `KEYWORDS_FILE`, `NEW_FILE`, `OLD_FILE` in `rag.py`). Before running this script, either:
> - edit `JSON_FILES = ["new.jsonl", "old.jsonl"]` and `OUTPUT_FILE = "clean_keywords.txt"` at the top of `keywords_gen.py`, **or**
> - run it as-is and then rename the output file to `clean_keywords.txt` (and point `JSON_FILES` at your real `new.jsonl`/`old.jsonl` from Step 2).

The resulting `clean_keywords.txt` is loaded into its own small Chroma vector store inside `rag.py` and is used during the **question reformulation** step to fuzzy-match / spell-correct names and terms the user typed (e.g. correcting a misspelled faculty name or course code before retrieval runs).

---

## Running the Chatbot API (`api.py`)

Once `new.jsonl`, `old.jsonl`, and `clean_keywords.txt` exist in the project root, and Ollama is running with `gpt-oss:20b` pulled:

```bash
python api.py
```

This will:

1. Import `rag.py`, which:
   - Downloads/loads the embedding model (`sentence-transformers/all-MiniLM-L6-v2`) and cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) from HuggingFace on first run.
   - Cleans and chunks `new.jsonl` and `old.jsonl` (`CHUNK_SIZE = 1700`, no overlap) and **rebuilds** the `chroma_new` and `chroma_old` Chroma directories from scratch every time it's imported.
   - Builds the keyword vector store from `clean_keywords.txt` (rebuilding `chroma_keywords` each run).
   - Sets up a SQLite LLM cache (`langchain_cache.db`).
   - Compiles the LangGraph pipeline.
2. Start a **uvicorn** server on `0.0.0.0:8000`.

You should see `[DEBUG]` log lines confirming how many documents/chunks were loaded for each corpus, followed by uvicorn's startup banner.

Alternatively, you can run it directly with uvicorn:

```bash
uvicorn api:api --host 0.0.0.0 --port 8000
```

> Avoid `--reload` for normal use — every reload re-imports `rag.py`, which wipes and rebuilds all three Chroma stores from disk, which can be slow for larger corpora.

---

## API Reference

### `GET /`

Health check.

```bash
curl http://localhost:8000/
```

```json
{ "message": "FSU-SC RAG Chatbot API is running!" }
```

### `POST /chat/stream`

Streams a Server-Sent-Events (SSE) response for a user question.

**Headers**

| Header | Required | Description |
|---|---|---|
| `X-Session-Id` | ✅ Yes | Any string identifying the conversation/session; used to keep a short rolling chat history server-side. |

**Body**

```json
{
  "question": "Who teaches the computational biology course this semester?"
}
```

**Example request**

```bash
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -H "X-Session-Id: demo-session-1" \
  -d '{"question": "What are the PhD admission requirements?"}'
```

**Streamed event types** (each line is `data: {...}\n\n`):

| `type` | Contents |
|---|---|
| `standalone_question` | The reformulated, standalone version of the user's question. |
| `token` | A streamed word/token of the final answer (simulated word-by-word streaming with a small delay). |
| `answer_complete` | The full final answer text. |
| `memory` | Debug info: the filled-in question-reformulation prompt, the chat history used, and the matched keywords. |
| `sources` | Debug info: the filled-in QA prompt template plus the retrieved/reranked chunks (`title`, `url`, `text`). |
| `fallback` | The filled-in fallback prompt (for debugging the no-RAG-context path). |
| `chat_history` | The session's updated rolling chat history. |
| `done` | Signals the stream is finished; contains the final answer text. |
| `error` | Returned instead of the above if an exception occurred server-side. |

---

## Configuration Reference

Key constants you may want to tune, all near the top of their respective files:

**`rag.py`**

| Constant | Default | Description |
|---|---|---|
| `NEW_FILE` / `OLD_FILE` | `new.jsonl` / `old.jsonl` | Source corpora. |
| `KEYWORDS_FILE` | `clean_keywords.txt` | Spelling-correction keyword dictionary. |
| `EMBED_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model for Chroma. |
| `CROSS_ENCODER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model. |
| `OLLAMA_MODEL` | `gpt-oss:20b` | Main QA / reformulation / fallback LLM. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1700` / `0` | Text splitter settings. |

**`main_crawler.py`**

| Constant | Default | Description |
|---|---|---|
| `START_URL` | `https://www.sc.fsu.edu` | Crawl entry point. |
| `OUTPUT_JSONL` | `crawl_3-19-26(2).jsonl` | Raw crawl output (input to Step 2's manual split). |
| `ALLOWED_PATH_PREFIXES` / `MUST_CRAWL` | see file | Scope of the crawl. |
| `MAX_PAGES` / `MAX_DEPTH` | `5000` / `20` | Crawl limits. |
| `HEADLESS` / `WAIT_FOR_MANUAL_LOGIN` | `False` / `True` | Manual CAS/Duo login support. |

**`keywords_gen.py`**

| Constant | Default | Description |
|---|---|---|
| `JSON_FILES` | `["new1.jsonl", "old1.jsonl"]` | ⚠️ Update to match `new.jsonl`/`old.jsonl` from Step 2. |
| `OUTPUT_FILE` | `clean_keywords1.txt` | ⚠️ Update to `clean_keywords.txt` to match `rag.py`. |
| `OLLAMA_MODEL` | `llama3.1:8b` | Keyword-extraction model. |
| `DOCS_PER_BATCH` | `5` | Documents per LLM call. |

---

## Notes, Caveats & Known Issues

- **Vector stores rebuild on every startup.** `rag.py` deletes and recreates `./chroma_new`, `./chroma_old`, and `./chroma_keywords` each time it's imported, so the first request after starting `api.py` will be slower while embeddings are computed.
- **`api.py` uses a module-level `chat_histories` dict** that is read/written inside `chat_stream()` but isn't defined or imported anywhere in `api.py` itself. If you hit a `NameError: chat_histories` when calling `/chat/stream`, add `chat_histories: dict = {}` near the top of `api.py`.
- **Filename mismatch between `keywords_gen.py` and `rag.py`.** As shipped, `keywords_gen.py` reads `new1.jsonl`/`old1.jsonl` and writes `clean_keywords1.txt`, while `rag.py` expects `new.jsonl`/`old.jsonl`/`clean_keywords.txt`. See [Step 3](#step-3--generate-the-spellingkeyword-dictionary-with-keywords_genpy) above.
- **CORS is wide open** (`allow_origins` includes `"*"`). Tighten this in `api.py` before deploying publicly.
- **Crawling requires a real FSU CAS login** and manual interaction (Duo 2FA); it cannot be fully automated as shipped.
- **Pinned dependency versions matter** — `rag.py` relies on the `langchain-classic` package split (`LLMChain`, `StuffDocumentsChain`, `ContextualCompressionRetriever`, etc.), so install exactly what's in `requirements.txt` rather than the latest LangChain release.
- No `LICENSE` file is currently included in this repository.

---

## Tech Stack

- **API / Serving:** FastAPI, Uvicorn, SSE streaming
- **Orchestration:** LangChain 1.x, `langchain-classic`, LangGraph
- **LLM runtime:** Ollama (`gpt-oss:20b`, `llama3.1:8b`)
- **Embeddings / Retrieval:** HuggingFace `sentence-transformers`, ChromaDB, cross-encoder reranking
- **Crawling:** Selenium, `webdriver-manager`, BeautifulSoup4, `pdfplumber`, `requests`
- **Caching:** SQLite (via `langchain-community` `SQLiteCache`)
