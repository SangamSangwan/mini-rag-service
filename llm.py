import os

SYSTEM_PROMPT = (
    "Answer using ONLY the provided context. If the answer is not in the context, "
    "say \"I don't know based on the notes.\" Cite page numbers like (p.3)."
)


def build_messages(question, hits):
    context = "\n\n".join(f"[page {h['page']}] {h['text']}" for h in hits)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]


def _ollama(messages):
    import ollama

    client = ollama.Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    resp = client.chat(
        model=os.getenv("OLLAMA_MODEL", "llama3.2"),
        messages=messages,
        options={"temperature": 0},
    )
    return resp["message"]["content"]


def _openai(messages):
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    client = OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL") or None)
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=messages,
        temperature=0,
    )
    return resp.choices[0].message.content


BACKENDS = {"ollama": _ollama, "openai": _openai}


def generate(question, hits):
    name = os.getenv("LLM_BACKEND", "ollama")
    if name not in BACKENDS:
        raise ValueError(f"Unknown LLM_BACKEND '{name}'. Options: {list(BACKENDS)}")
    return BACKENDS[name](build_messages(question, hits))
