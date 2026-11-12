import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

import db
import llm
import rag
from cache import get_cache

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")

state = {}


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


class Source(BaseModel):
    page: int
    distance: float
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    cached: bool
    latency_ms: float


@asynccontextmanager
async def lifespan(app):
    pdf_path = os.getenv("PDF_PATH", "notes.pdf")
    t0 = time.perf_counter()
    state["col"] = await asyncio.to_thread(rag.build_index, pdf_path)
    state["cache"] = get_cache()
    await asyncio.to_thread(db.init_db)
    log.info("index ready in %.1fs (%s)", time.perf_counter() - t0, pdf_path)
    yield
    state.clear()


app = FastAPI(title="Mini RAG Service", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "llm_backend": os.getenv("LLM_BACKEND", "ollama")}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest, background: BackgroundTasks):
    start = time.perf_counter()
    cache = state["cache"]

    result = await asyncio.to_thread(cache.get, req.question)
    cached = result is not None

    if not cached:
        t0 = time.perf_counter()
        hits = await asyncio.to_thread(rag.retrieve, state["col"], req.question)
        t1 = time.perf_counter()
        try:
            answer = await asyncio.to_thread(llm.generate, req.question, hits)
        except Exception:
            log.exception("llm backend failed")
            raise HTTPException(status_code=502, detail="LLM backend failed")
        t2 = time.perf_counter()
        log.info("retrieve=%.0fms generate=%.0fms", (t1 - t0) * 1000, (t2 - t1) * 1000)
        result = {"answer": answer, "sources": hits}
        await asyncio.to_thread(cache.set, req.question, result)

    latency_ms = round((time.perf_counter() - start) * 1000, 1)
    log.info("ask cached=%s latency=%.1fms", cached, latency_ms)
    background.add_task(
        db.log_request, req.question, result["answer"], result["sources"], cached, latency_ms
    )
    return AskResponse(
        answer=result["answer"], sources=result["sources"], cached=cached, latency_ms=latency_ms
    )
