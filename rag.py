import hashlib
import os

from pypdf import PdfReader

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))
TOP_K = int(os.getenv("TOP_K", 4))
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")


def load_pdf(path):
    reader = PdfReader(path)
    return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def collection_name(pdf_path, size, overlap, model):
    with open(pdf_path, "rb") as f:
        digest = hashlib.sha256(f.read() + f"{size}-{overlap}-{model}".encode()).hexdigest()
    return f"docs_{digest[:16]}"


def build_index(pdf_path):
    import chromadb
    from chromadb.utils import embedding_functions

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    name = collection_name(pdf_path, CHUNK_SIZE, CHUNK_OVERLAP, EMBED_MODEL)
    col = client.get_or_create_collection(
        name, embedding_function=embed_fn, metadata={"hnsw:space": "cosine"}
    )
    if col.count() > 0:
        return col

    ids, docs, metas = [], [], []
    for page_no, text in load_pdf(pdf_path):
        for j, chunk in enumerate(chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)):
            ids.append(f"p{page_no}_c{j}")
            docs.append(chunk)
            metas.append({"page": page_no})
    for i in range(0, len(docs), 500):
        col.add(ids=ids[i:i + 500], documents=docs[i:i + 500], metadatas=metas[i:i + 500])
    return col


def retrieve(col, question, k=TOP_K):
    res = col.query(query_texts=[question], n_results=k)
    return [
        {"text": d, "page": m["page"], "distance": round(dist, 4)}
        for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]
