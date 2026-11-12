import pytest

from rag import chunk_text


def test_chunk_size_and_overlap():
    text = "a" * 2000
    chunks = chunk_text(text, size=800, overlap=150)
    assert all(len(c) <= 800 for c in chunks)
    assert chunks[0][-150:] == chunks[1][:150]


def test_chunk_covers_all_text():
    text = " ".join(f"w{i}" for i in range(500))
    chunks = chunk_text(text, size=100, overlap=20)
    assert chunks[0].startswith("w0")
    assert chunks[-1].endswith("w499")


def test_empty_text_gives_no_chunks():
    assert chunk_text("   ") == []


def test_bad_overlap_raises():
    with pytest.raises(ValueError):
        chunk_text("hello world", size=10, overlap=10)
