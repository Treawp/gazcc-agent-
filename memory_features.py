"""
agent/memory_features.py
═══════════════════════════════════════════════════════════════════════════════
  GAZCC — MEMORY FEATURE EXTENSIONS
  ════════════════════════════════════════════════════════════════════════════

  Features:
    ProgressiveDisclosure — Layered memory retrieval with token cost tracking
    ContextInjector       — Fine-grained control over what context gets injected
    MemoryEventBus        — Pub/sub for real-time memory viewer streaming (SSE)

  Usage in agent/core.py:
    from .memory_features import ProgressiveDisclosure, ContextInjector, bus
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .memory import MemoryEntry, FileMemory, RedisMemory


# ── MemoryEventBus ────────────────────────────────────────────────────────────
# Tiny pub/sub used by memory_viewer.py for real-time SSE stream.

@dataclass
class MemoryEvent:
    event_type: str          # "store" | "search" | "delete" | "retrieve"
    key: str
    obs_id: str = ""
    content_preview: str = ""  # first 200 chars
    metadata: dict = field(default_factory=dict)
    token_cost: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_sse(self) -> str:
        payload = json.dumps({
            "type":    self.event_type,
            "key":     self.key,
            "obs_id":  self.obs_id,
            "preview": self.content_preview,
            "meta":    self.metadata,
            "tokens":  self.token_cost,
            "ts":      self.timestamp,
        })
        return f"data: {payload}\n\n"


class MemoryEventBus:
    """Global pub/sub bus for memory events. Subscribers get async queues."""

    def __init__(self):
        self._subscribers: list[asyncio.Queue] = []

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        try:
            self._subscribers.remove(q)
        except ValueError:
            pass

    async def publish(self, event: MemoryEvent):
        dead = []
        for q in self._subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            self.unsubscribe(q)

    async def stream(self, q: asyncio.Queue) -> AsyncIterator[str]:
        try:
            while True:
                event: MemoryEvent = await asyncio.wait_for(q.get(), timeout=30)
                yield event.to_sse()
        except asyncio.TimeoutError:
            yield ": heartbeat\n\n"


# Singleton bus — imported by both skill_tools and memory_viewer
bus = MemoryEventBus()


# ── ProgressiveDisclosure ─────────────────────────────────────────────────────

@dataclass
class DisclosureLayer:
    """One retrieval layer with its token cost and relevance tier."""
    tier: str           # "core" | "extended" | "background"
    score: float
    entry: "MemoryEntry"
    token_cost: int


class ProgressiveDisclosure:
    """
    Retrieves memory in layers, respecting a token budget.

    Tiers:
      core       score ≥ 0.6  — injected first, highest priority
      extended   score ≥ 0.35 — injected if budget allows
      background score ≥ 0.15 — brief summary only

    Returns a formatted context string + cost report.
    """

    TIER_THRESHOLDS = {
        "core":       0.60,
        "extended":   0.35,
        "background": 0.15,
    }

    def __init__(self, memory_backend, token_budget: int = 2000):
        self._mem = memory_backend
        self.token_budget = token_budget

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        include_tiers: list[str] | None = None,
    ) -> tuple[str, dict]:
        """
        Returns (context_str, cost_report).

        cost_report = {
          "total_tokens": int,
          "budget": int,
          "layers": [ {"tier": ..., "key": ..., "score": ..., "tokens": ...} ]
        }
        """
        include_tiers = include_tiers or ["core", "extended", "background"]

        if hasattr(self._mem, "search_with_scores"):
            scored = await self._mem.search_with_scores(query, top_k=top_k, threshold=0.10)
        else:
            # fallback for RedisMemory
            entries = await self._mem.search(query, top_k=top_k, threshold=0.10)
            scored = [(0.5, e) for e in entries]

        layers: list[DisclosureLayer] = []
        for score, entry in scored:
            if score >= self.TIER_THRESHOLDS["core"]:
                tier = "core"
            elif score >= self.TIER_THRESHOLDS["extended"]:
                tier = "extended"
            elif score >= self.TIER_THRESHOLDS["background"]:
                tier = "background"
            else:
                continue
            if tier not in include_tiers:
                continue
            layers.append(DisclosureLayer(tier=tier, score=score, entry=entry, token_cost=entry.token_cost))

        # Fill budget layer by layer
        budget_left = self.token_budget
        context_parts: list[str] = []
        cost_layers: list[dict] = []

        for layer in layers:
            if budget_left <= 0:
                break
            if layer.tier == "background" and layer.token_cost > budget_left:
                # Add trimmed summary instead
                summary = layer.entry.content[:budget_left * 4]  # ~4 chars/token
                snippet = f"[background|{layer.entry.key}|id:{layer.entry.obs_id[:8]}] {summary}…"
                context_parts.append(snippet)
                cost_layers.append({
                    "tier": "background(trimmed)",
                    "key": layer.entry.key,
                    "score": round(layer.score, 3),
                    "tokens": budget_left,
                    "obs_id": layer.entry.obs_id,
                })
                budget_left = 0
                break

            label = f"[{layer.tier}|{layer.entry.key}|id:{layer.entry.obs_id[:8]}]"
            context_parts.append(f"{label}\n{layer.entry.content}")
            cost_layers.append({
                "tier":    layer.tier,
                "key":     layer.entry.key,
                "score":   round(layer.score, 3),
                "tokens":  layer.token_cost,
                "obs_id":  layer.entry.obs_id,
            })
            budget_left -= layer.token_cost

        context_str = "\n\n".join(context_parts)
        total_used = self.token_budget - max(0, budget_left)
        cost_report = {
            "total_tokens": total_used,
            "budget":       self.token_budget,
            "utilization":  f"{100 * total_used / max(1, self.token_budget):.1f}%",
            "layers":       cost_layers,
        }
        return context_str, cost_report


# ── ContextInjector ───────────────────────────────────────────────────────────

@dataclass
class ContextConfig:
    """Fine-grained control over what context is injected into agent prompts."""
    inject_memory:      bool = True
    inject_task_state:  bool = True
    inject_tool_list:   bool = True
    inject_system_time: bool = True
    inject_file_tree:   bool = False  # can be noisy
    max_memory_tokens:  int  = 1500
    memory_tiers:       list = field(default_factory=lambda: ["core", "extended"])
    strip_private:      bool = True   # always strip <private> before injection


class ContextInjector:
    """
    Assembles the context block injected at the start of each agent turn.
    Reads config from config.yaml → context section.
    """

    def __init__(self, memory_backend, cfg: dict | None = None):
        self._mem = memory_backend
        raw = (cfg or {}).get("context", {})
        self._conf = ContextConfig(
            inject_memory      = raw.get("inject_memory",      True),
            inject_task_state  = raw.get("inject_task_state",  True),
            inject_tool_list   = raw.get("inject_tool_list",   True),
            inject_system_time = raw.get("inject_system_time", True),
            inject_file_tree   = raw.get("inject_file_tree",   False),
            max_memory_tokens  = raw.get("max_memory_tokens",  1500),
            memory_tiers       = raw.get("memory_tiers",       ["core", "extended"]),
            strip_private      = raw.get("strip_private",      True),
        )
        self._disclosure = ProgressiveDisclosure(
            memory_backend, token_budget=self._conf.max_memory_tokens
        )

    @property
    def config(self) -> ContextConfig:
        return self._conf

    async def build(
        self,
        query: str,
        task_state: dict | None = None,
        tool_names: list[str] | None = None,
    ) -> tuple[str, dict]:
        """
        Build context string + metadata dict.
        Returns (context_str, meta) where meta includes token cost breakdown.
        """
        parts: list[str] = []
        meta: dict = {}

        if self._conf.inject_system_time:
            parts.append(f"[system_time] {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")

        if self._conf.inject_memory and query:
            ctx, cost = await self._disclosure.retrieve(
                query, include_tiers=self._conf.memory_tiers
            )
            if ctx:
                parts.append(f"[memory]\n{ctx}")
            meta["memory_cost"] = cost

        if self._conf.inject_task_state and task_state:
            import json as _j
            parts.append(f"[task_state]\n{_j.dumps(task_state, indent=2)}")

        if self._conf.inject_tool_list and tool_names:
            parts.append(f"[available_tools] {', '.join(tool_names)}")

        return "\n\n".join(parts), meta
