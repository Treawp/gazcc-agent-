"""
agent/memory.py
Long-term memory with vector similarity search.
Backends: file (local) | redis (Vercel/Upstash)
"""

import json
import math
import os
import re
import time
import uuid
import asyncio
import aiofiles
from collections import Counter
from typing import Any, Optional
from pathlib import Path


# ── Privacy Filter ────────────────────────────────────────────────────────────

_PRIVATE_RE = re.compile(r"<private>.*?</private>", re.IGNORECASE | re.DOTALL)

def strip_private(text: str) -> str:
    """Remove <private>...</private> blocks before storing to memory."""
    return _PRIVATE_RE.sub("[REDACTED]", text).strip()


# ── helpers ──────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    import re
    return re.findall(r"\b\w+\b", text.lower())


def _tfidf_vector(tokens: list[str], vocab: dict[str, int]) -> list[float]:
    tf = Counter(tokens)
    total = max(len(tokens), 1)
    vec = [0.0] * len(vocab)
    for word, idx in vocab.items():
        vec[idx] = tf.get(word, 0) / total
    return vec


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ── MemoryEntry ───────────────────────────────────────────────────────────────

class MemoryEntry:
    __slots__ = ("key", "content", "metadata", "timestamp", "tokens", "obs_id", "token_cost")

    def __init__(self, key: str, content: str, metadata: dict | None = None):
        self.key = key
        self.content = strip_private(content)
        self.metadata = metadata or {}
        self.timestamp = time.time()
        self.tokens = _tokenize(self.content)
        self.obs_id: str = str(uuid.uuid4())
        # Rough token cost: ~4 chars per token
        self.token_cost: int = max(1, len(self.content) // 4)

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "obs_id": self.obs_id,
            "token_cost": self.token_cost,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryEntry":
        e = cls.__new__(cls)
        e.key = d["key"]
        e.content = d["content"]
        e.metadata = d.get("metadata", {})
        e.timestamp = d.get("timestamp", time.time())
        e.tokens = _tokenize(e.content)
        e.obs_id = d.get("obs_id", str(uuid.uuid4()))
        e.token_cost = d.get("token_cost", max(1, len(e.content) // 4))
        return e


# ── FileMemory ────────────────────────────────────────────────────────────────

class FileMemory:
    """Persistent JSON-file memory for local runs."""

    def __init__(self, path: str, max_entries: int = 500):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.index_file = self.path / "index.json"
        self.max_entries = max_entries
        self._entries: dict[str, MemoryEntry] = {}
        self._lock = asyncio.Lock()
        self._loaded = False

    async def _load(self):
        if self._loaded:
            return
        if self.index_file.exists():
            async with aiofiles.open(self.index_file, "r") as f:
                raw = json.loads(await f.read())
            self._entries = {k: MemoryEntry.from_dict(v) for k, v in raw.items()}
        self._loaded = True

    async def _save(self):
        data = {k: v.to_dict() for k, v in self._entries.items()}
        async with aiofiles.open(self.index_file, "w") as f:
            await f.write(json.dumps(data, indent=2))

    async def store(self, key: str, content: str, metadata: dict | None = None):
        async with self._lock:
            await self._load()
            self._entries[key] = MemoryEntry(key, content, metadata)
            # evict oldest if over limit
            if len(self._entries) > self.max_entries:
                oldest = sorted(self._entries.values(), key=lambda e: e.timestamp)
                for e in oldest[: len(self._entries) - self.max_entries]:
                    del self._entries[e.key]
            await self._save()

    async def retrieve(self, key: str) -> Optional[MemoryEntry]:
        async with self._lock:
            await self._load()
            return self._entries.get(key)

    async def search(self, query: str, top_k: int = 5, threshold: float = 0.2) -> list[MemoryEntry]:
        async with self._lock:
            await self._load()
            if not self._entries:
                return []
            # build vocab
            all_tokens: list[str] = []
            for e in self._entries.values():
                all_tokens.extend(e.tokens)
            vocab = {w: i for i, w in enumerate(set(all_tokens))}
            q_tokens = _tokenize(query)
            q_vec = _tfidf_vector(q_tokens, vocab)
            results = []
            for entry in self._entries.values():
                e_vec = _tfidf_vector(entry.tokens, vocab)
                score = _cosine(q_vec, e_vec)
                if score >= threshold:
                    results.append((score, entry))
            results.sort(key=lambda x: -x[0])
            return [e for _, e in results[:top_k]]

    async def delete(self, key: str):
        async with self._lock:
            await self._load()
            self._entries.pop(key, None)
            await self._save()

    async def all_keys(self) -> list[str]:
        async with self._lock:
            await self._load()
            return list(self._entries.keys())

    async def all_entries(self) -> list[MemoryEntry]:
        async with self._lock:
            await self._load()
            return sorted(self._entries.values(), key=lambda e: e.timestamp, reverse=True)

    async def retrieve_by_obs_id(self, obs_id: str) -> Optional[MemoryEntry]:
        async with self._lock:
            await self._load()
            for entry in self._entries.values():
                if entry.obs_id == obs_id:
                    return entry
            return None

    async def search_with_scores(
        self, query: str, top_k: int = 5, threshold: float = 0.2
    ) -> list[tuple[float, MemoryEntry]]:
        """Like search() but returns (score, entry) pairs for progressive disclosure."""
        async with self._lock:
            await self._load()
            if not self._entries:
                return []
            all_tokens: list[str] = []
            for e in self._entries.values():
                all_tokens.extend(e.tokens)
            vocab = {w: i for i, w in enumerate(set(all_tokens))}
            q_vec = _tfidf_vector(_tokenize(query), vocab)
            results = []
            for entry in self._entries.values():
                score = _cosine(q_vec, _tfidf_vector(entry.tokens, vocab))
                if score >= threshold:
                    results.append((score, entry))
            results.sort(key=lambda x: -x[0])
            return results[:top_k]


# ── RedisMemory ───────────────────────────────────────────────────────────────

class RedisMemory:
    """Upstash Redis memory for Vercel serverless."""

    def __init__(self, url: str, token: str, max_entries: int = 500, ttl: int = 86400):
        from upstash_redis import Redis  # type: ignore
        self._r = Redis(url=url, token=token)
        self.max_entries = max_entries
        self.ttl = ttl
        self._prefix = "gazcc:mem:"

    def _k(self, key: str) -> str:
        return f"{self._prefix}{key}"

    async def store(self, key: str, content: str, metadata: dict | None = None):
        entry = MemoryEntry(key, content, metadata)
        rk = self._k(key)
        self._r.set(rk, json.dumps(entry.to_dict()), ex=self.ttl)
        self._r.zadd("gazcc:mem:index", {rk: entry.timestamp})

    async def retrieve(self, key: str) -> Optional[MemoryEntry]:
        raw = self._r.get(self._k(key))
        if raw is None:
            return None
        return MemoryEntry.from_dict(json.loads(raw))

    async def search(self, query: str, top_k: int = 5, threshold: float = 0.2) -> list[MemoryEntry]:
        keys = self._r.zrange("gazcc:mem:index", 0, -1)
        entries = []
        for rk in keys:
            raw = self._r.get(rk)
            if raw:
                entries.append(MemoryEntry.from_dict(json.loads(raw)))
        if not entries:
            return []
        all_tokens: list[str] = []
        for e in entries:
            all_tokens.extend(e.tokens)
        vocab = {w: i for i, w in enumerate(set(all_tokens))}
        q_vec = _tfidf_vector(_tokenize(query), vocab)
        results = []
        for entry in entries:
            e_vec = _tfidf_vector(entry.tokens, vocab)
            score = _cosine(q_vec, e_vec)
            if score >= threshold:
                results.append((score, entry))
        results.sort(key=lambda x: -x[0])
        return [e for _, e in results[:top_k]]

    async def delete(self, key: str):
        rk = self._k(key)
        self._r.delete(rk)
        self._r.zrem("gazcc:mem:index", rk)

    async def all_keys(self) -> list[str]:
        keys = self._r.zrange("gazcc:mem:index", 0, -1)
        return [k.replace(self._prefix, "") for k in keys]


# ── TaskState ─────────────────────────────────────────────────────────────────

class TaskState:
    """Stores running task state (steps, checkpoints) in Redis or file."""

    def __init__(self, backend: Any, task_id: str, ttl: int = 86400):
        self._backend = backend
        self._task_id = task_id
        self._ttl = ttl
        self._key = f"gazcc:task:{task_id}"

    async def save(self, state: dict):
        if isinstance(self._backend, RedisMemory):
            self._backend._r.set(self._key, json.dumps(state), ex=self._ttl)
        else:
            p = Path(self._backend.path) / f"task_{self._task_id}.json"
            async with aiofiles.open(p, "w") as f:
                await f.write(json.dumps(state, indent=2))

    async def load(self) -> dict | None:
        if isinstance(self._backend, RedisMemory):
            raw = self._backend._r.get(self._key)
            return json.loads(raw) if raw else None
        p = Path(self._backend.path) / f"task_{self._task_id}.json"
        if not p.exists():
            return None
        async with aiofiles.open(p, "r") as f:
            return json.loads(await f.read())

    async def delete(self):
        if isinstance(self._backend, RedisMemory):
            self._backend._r.delete(self._key)
        else:
            p = Path(self._backend.path) / f"task_{self._task_id}.json"
            if p.exists():
                p.unlink()


# ── factory ───────────────────────────────────────────────────────────────────

def build_memory(cfg: dict) -> FileMemory | RedisMemory:
    backend = cfg.get("backend", "file")
    if backend == "redis":
        url = os.environ.get("UPSTASH_REDIS_REST_URL", "")
        token = os.environ.get("UPSTASH_REDIS_REST_TOKEN", "")
        if not url or not token:
            raise ValueError("UPSTASH_REDIS_REST_URL / TOKEN env vars not set")
        return RedisMemory(url, token, max_entries=cfg.get("max_entries", 500))
    return FileMemory(
        cfg.get("file_path", "/tmp/gazcc_memory"),
        max_entries=cfg.get("max_entries", 500),
    )
