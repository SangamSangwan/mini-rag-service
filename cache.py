import hashlib
import json
import logging
import os
import time

log = logging.getLogger("cache")


def make_key(question):
    normalized = " ".join(question.lower().split())
    return "rag:" + hashlib.sha256(normalized.encode()).hexdigest()


class MemoryCache:
    def __init__(self, ttl=3600, max_items=1000):
        self.ttl = ttl
        self.max_items = max_items
        self._data = {}

    def get(self, question):
        key = make_key(question)
        item = self._data.get(key)
        if item is None:
            return None
        expires_at, value = item
        if expires_at < time.monotonic():
            self._data.pop(key, None)
            return None
        return value

    def set(self, question, value):
        if len(self._data) >= self.max_items:
            self._data.pop(next(iter(self._data)), None)
        self._data[make_key(question)] = (time.monotonic() + self.ttl, value)


class RedisCache:
    def __init__(self, url, ttl=3600):
        import redis

        self._redis = redis
        self.client = redis.Redis.from_url(url, decode_responses=True)
        self.ttl = ttl

    def get(self, question):
        try:
            raw = self.client.get(make_key(question))
            return json.loads(raw) if raw else None
        except self._redis.RedisError:
            log.warning("redis get failed, treating as miss")
            return None

    def set(self, question, value):
        try:
            self.client.setex(make_key(question), self.ttl, json.dumps(value))
        except self._redis.RedisError:
            log.warning("redis set failed, skipping cache write")


def get_cache():
    ttl = int(os.getenv("CACHE_TTL", 3600))
    url = os.getenv("REDIS_URL")
    return RedisCache(url, ttl) if url else MemoryCache(ttl)
