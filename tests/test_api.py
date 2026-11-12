import pytest
from fastapi.testclient import TestClient

import app as app_module
import llm
import rag

HITS = [{"text": "GPT-2 vocab is 50,257", "page": 3, "distance": 0.3}]


@pytest.fixture
def client(monkeypatch):
    calls = {"generate": 0}
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(rag, "build_index", lambda path: object())
    monkeypatch.setattr(rag, "retrieve", lambda col, q, k=4: HITS)

    def fake_generate(question, hits):
        calls["generate"] += 1
        return "50,257 (p.3)"

    monkeypatch.setattr(llm, "generate", fake_generate)
    with TestClient(app_module.app) as c:
        yield c, calls


def test_health(client):
    c, _ = client
    assert c.get("/health").json()["status"] == "ok"


def test_ask_returns_answer_and_sources(client):
    c, _ = client
    r = c.post("/ask", json={"question": "How many tokens in GPT-2 vocab?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "50,257 (p.3)"
    assert body["sources"][0]["page"] == 3
    assert body["cached"] is False


def test_repeat_question_hits_cache(client):
    c, calls = client
    c.post("/ask", json={"question": "How many tokens in GPT-2 vocab?"})
    r = c.post("/ask", json={"question": "  how many TOKENS in gpt-2 vocab? "})
    assert r.json()["cached"] is True
    assert calls["generate"] == 1


def test_short_question_rejected(client):
    c, _ = client
    assert c.post("/ask", json={"question": "hi"}).status_code == 422


def test_llm_failure_returns_502(client, monkeypatch):
    c, _ = client

    def boom(question, hits):
        raise RuntimeError("backend down")

    monkeypatch.setattr(llm, "generate", boom)
    r = c.post("/ask", json={"question": "something that fails"})
    assert r.status_code == 502
