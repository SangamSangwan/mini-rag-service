import logging
import os

log = logging.getLogger("db")

DDL = """
CREATE TABLE IF NOT EXISTS qa_log (
    id SERIAL PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    sources JSONB NOT NULL,
    cached BOOLEAN NOT NULL,
    latency_ms DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""


def _connect():
    import psycopg

    return psycopg.connect(os.environ["DATABASE_URL"])


def init_db():
    if not os.getenv("DATABASE_URL"):
        log.info("DATABASE_URL not set, request logging disabled")
        return
    try:
        with _connect() as conn:
            conn.execute(DDL)
    except Exception:
        log.exception("init_db failed")


def log_request(question, answer, sources, cached, latency_ms):
    if not os.getenv("DATABASE_URL"):
        return
    try:
        from psycopg.types.json import Jsonb

        with _connect() as conn:
            conn.execute(
                "INSERT INTO qa_log (question, answer, sources, cached, latency_ms) "
                "VALUES (%s, %s, %s, %s, %s)",
                (question, answer, Jsonb(sources), cached, latency_ms),
            )
    except Exception:
        log.exception("log_request failed")
