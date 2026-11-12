# Mini RAG Service

A local-first RAG API. Ask questions about a PDF and get answers with page-cited sources.

**Stack:** FastAPI, ChromaDB, sentence-transformers, Ollama or any OpenAI-compatible LLM, Redis, PostgreSQL, Docker.

## Architecture

```
POST /ask
   |
   v
cache lookup (memory or Redis) --hit--> response
   | miss
   v
retrieve top-k chunks (Chroma, cosine distance)
   v
generate answer (Ollama or OpenAI-compatible API)
   v
write to cache -> respond -> log to PostgreSQL (background task)
```

## Features

- Async FastAPI endpoints; blocking work (embedding search, LLM calls) runs in worker threads via `asyncio.to_thread`
- Swappable LLM backend through one env var (`LLM_BACKEND=ollama|openai`); the OpenAI backend also works with Gemini and other OpenAI-compatible providers via `OPENAI_BASE_URL`
- Two-tier cache: in-memory with TTL by default, Redis when `REDIS_URL` is set; cache keys are normalized (case and whitespace)
- The index rebuilds automatically when the PDF, chunk settings or embedding model change (collection name is a content hash)
- Every request logged to PostgreSQL in a background task (question, answer, sources, cache hit, latency), with analytics queries in `queries.sql`
- Per-stage timing logs (retrieve vs generate) for finding bottlenecks
- Tests that run without Ollama, a GPU or a database (LLM and index are mocked)
- Docker Compose setup: API + Redis + PostgreSQL

## Quickstart (local)

```bash
pip install -r requirements-dev.txt
ollama pull llama3.2
uvicorn app:app --reload
```

Open http://localhost:8000/docs, or:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How many tokens does the GPT-2 vocabulary have?"}'
```

Use your own document: `PDF_PATH=my.pdf uvicorn app:app`

## Docker

```bash
docker compose up --build
```

Ollama runs on the host (the container reaches it at `host.docker.internal:11434`). To use a hosted model instead:

```bash
LLM_BACKEND=openai OPENAI_API_KEY=your-key docker compose up --build
```

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `PDF_PATH` | `notes.pdf` | Document to index |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `150` | Chunking (characters) |
| `TOP_K` | `4` | Chunks retrieved per question |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Embedding model |
| `LLM_BACKEND` | `ollama` | `ollama` or `openai` |
| `OLLAMA_MODEL`, `OLLAMA_HOST` | `llama3.2`, `http://localhost:11434` | Ollama settings |
| `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_BASE_URL` | | OpenAI-compatible settings |
| `REDIS_URL` | unset (in-memory cache) | Enables Redis cache |
| `CACHE_TTL` | `3600` | Cache lifetime in seconds |
| `DATABASE_URL` | unset (logging off) | Enables PostgreSQL logging |

## Tests

```bash
pytest
```

## Experiments

Add your own measurements here (these tables are intentionally empty until you run them).

**Chunk size** (`TOP_K=4`, same questions each run)

| CHUNK_SIZE | CHUNK_OVERLAP | Chunks | Fact question: top distance | Broad question: observation |
|---|---|---|---|---|
| 800 | 150 | 23 | 0.3893 (p.3) | Top distance 0.6472 (p.2); complete context with self-attention and architecture in single chunk |
| 300 | 50 | 53 | 0.2553 (p.3) | Top distance 0.6310 (p.2); lower distance for fact chunk, but broad concepts split across chunks |

**Caching** (same question twice: *"How many tokens does the GPT-2 vocabulary have?"*)

| Call | latency_ms |
|---|---|
| First (miss) | 2793.3 |
| Repeat (hit) | 0.3 |

## Limitations and next steps

- One PDF per instance, no authentication
- A new database connection per logged request (a connection pool would be better under load)
- Dependencies are not pinned; run `pip freeze` once you have a known-good environment
- Next: sentence-window retrieval, hybrid search (BM25 + vectors), reranking, and RAG Triad evaluation (context relevance, groundedness, answer relevance)
