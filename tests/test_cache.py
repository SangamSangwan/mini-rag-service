import time

from cache import MemoryCache, make_key


def test_key_normalizes_case_and_whitespace():
    assert make_key("What is  RAG?") == make_key("  what is rag? ")


def test_set_and_get():
    c = MemoryCache()
    c.set("q one", {"answer": "x"})
    assert c.get("Q  ONE") == {"answer": "x"}
    assert c.get("other") is None


def test_ttl_expiry():
    c = MemoryCache(ttl=0.05)
    c.set("q", 1)
    time.sleep(0.1)
    assert c.get("q") is None


def test_max_items_evicts_oldest():
    c = MemoryCache(max_items=2)
    c.set("a", 1)
    c.set("b", 2)
    c.set("c", 3)
    assert c.get("a") is None
    assert c.get("c") == 3
