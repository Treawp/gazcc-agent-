"""
agent/new_tools.py
══════════════════════════════════════════════════════════════════════════════
  GAZCC NEW TOOLS — rss_reader + embedding_compare
  Drop-in addition. Import and register via ToolRegistry.

  INSTALL (core.py):
    from agent.new_tools import register_new_tools
    register_new_tools(self._tools, self._cfg)

  TOOLS:
    #01  rss_reader        — Fetch & parse RSS/Atom feeds → structured articles
    #02  embedding_compare — Compare texts via cosine similarity using embeddings

  REQUIREMENTS (add to requirements.txt):
    feedparser>=6.0.11    # rss_reader
    numpy>=1.26.0         # embedding_compare (cosine similarity)
══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
from datetime import datetime
from typing import Any

import httpx

try:
    from .tools import BaseTool, ToolResult
except ImportError:
    class ToolResult:
        def __init__(self, success: bool, output: str, metadata: dict | None = None):
            self.success = success
            self.output = output
            self.metadata = metadata or {}
        def __str__(self): return f"{'✓' if self.success else '✗'} {self.output}"
    class BaseTool:
        name = ""
        description = ""
        parameters = ""
        async def run(self, *a, **kw) -> ToolResult:
            return ToolResult(False, "not implemented")


# ══════════════════════════════════════════════════════════════════════════════
# 1. RSSReaderTool — Fetch & parse RSS/Atom feeds
# ══════════════════════════════════════════════════════════════════════════════

class RSSReaderTool(BaseTool):
    name = "rss_reader"
    description = (
        "Fetch and parse RSS/Atom feeds. Returns structured list of articles "
        "with title, link, summary, author, and published date. "
        "Actions: fetch (get articles), search (filter by keyword), "
        "summary (get feed metadata only)."
    )
    parameters = (
        "action: str, "
        "url: str, "
        "keyword: str = '', "
        "max_items: int = 10, "
        "include_content: bool = False"
    )

    async def run(
        self,
        action: str,
        url: str,
        keyword: str = "",
        max_items: int = 10,
        include_content: bool = False,
    ) -> ToolResult:
        try:
            max_items = max(1, min(max_items, 50))  # cap 50

            # ── fetch raw feed ──────────────────────────────────────────────
            raw_xml = await self._fetch_feed(url)
            if raw_xml is None:
                return ToolResult(False, f"Failed to fetch feed from: {url}")

            # ── parse feed ─────────────────────────────────────────────────
            feed_data = self._parse_feed(raw_xml)
            if not feed_data:
                return ToolResult(False, "Failed to parse feed — unsupported format or empty.")

            feed_meta = feed_data["meta"]
            entries   = feed_data["entries"]

            # ── action: summary ────────────────────────────────────────────
            if action == "summary":
                out = (
                    f"Feed: {feed_meta.get('title', 'Unknown')}\n"
                    f"Link: {feed_meta.get('link', url)}\n"
                    f"Desc: {feed_meta.get('description', '-')}\n"
                    f"Lang: {feed_meta.get('language', '-')}\n"
                    f"Updated: {feed_meta.get('updated', '-')}\n"
                    f"Total entries: {len(entries)}"
                )
                return ToolResult(True, out, {"entry_count": len(entries), "meta": feed_meta})

            # ── action: search ─────────────────────────────────────────────
            if action == "search":
                if not keyword:
                    return ToolResult(False, "keyword parameter required for 'search' action.")
                kw_lower = keyword.lower()
                entries = [
                    e for e in entries
                    if kw_lower in e.get("title", "").lower()
                    or kw_lower in e.get("summary", "").lower()
                ]
                if not entries:
                    return ToolResult(True, f"No entries found matching '{keyword}'.", {"count": 0})

            # ── action: fetch (default) ────────────────────────────────────
            entries = entries[:max_items]
            lines = [
                f"Feed: {feed_meta.get('title', 'Unknown')} "
                f"({len(entries)} items)\n"
                + "─" * 60
            ]

            for i, e in enumerate(entries, 1):
                block = [
                    f"[{i}] {e.get('title', 'No title')}",
                    f"    🔗 {e.get('link', '-')}",
                    f"    📅 {e.get('published', '-')}",
                    f"    ✍️  {e.get('author', '-')}",
                ]
                summary = e.get("summary", "")
                if summary:
                    # trim to ~200 chars for readability
                    trimmed = summary[:200].replace("\n", " ").strip()
                    if len(summary) > 200:
                        trimmed += "..."
                    block.append(f"    📝 {trimmed}")
                if include_content and e.get("content"):
                    content = e["content"][:500].replace("\n", " ").strip()
                    block.append(f"    📄 {content}...")
                lines.append("\n".join(block))

            output = "\n\n".join(lines)
            return ToolResult(
                True,
                output,
                {
                    "feed_title": feed_meta.get("title", ""),
                    "count": len(entries),
                    "entries": entries,
                },
            )

        except Exception as e:
            return ToolResult(False, f"rss_reader error: {e}")

    # ── helpers ───────────────────────────────────────────────────────────────

    async def _fetch_feed(self, url: str) -> str | None:
        headers = {
            "User-Agent": "GazccAgent/1.0 RSS-Reader (compatible; +https://gazcc.ai)",
            "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, */*",
        }
        try:
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                r = await client.get(url, headers=headers)
                r.raise_for_status()
                return r.text
        except Exception:
            return None

    def _parse_feed(self, xml: str) -> dict | None:
        """Parse RSS 2.0 or Atom 1.0 without external feedparser dependency."""
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(xml)
        except Exception:
            return None

        ns = {
            "atom":    "http://www.w3.org/2005/Atom",
            "content": "http://purl.org/rss/1.0/modules/content/",
            "dc":      "http://purl.org/dc/elements/1.1/",
            "media":   "http://search.yahoo.com/mrss/",
        }

        def txt(el, *tags):
            """Try multiple tag names, return first non-empty text."""
            for tag in tags:
                child = el.find(tag) or el.find(f"atom:{tag}", ns) or el.find(f"dc:{tag}", ns)
                if child is not None and child.text:
                    return child.text.strip()
            return ""

        # ── detect format ──────────────────────────────────────────────────
        tag = root.tag.lower()

        # ── RSS 2.0 ────────────────────────────────────────────────────────
        if "rss" in tag or root.find("channel") is not None:
            channel = root.find("channel") or root
            meta = {
                "title":       txt(channel, "title"),
                "link":        txt(channel, "link"),
                "description": txt(channel, "description"),
                "language":    txt(channel, "language"),
                "updated":     txt(channel, "lastBuildDate", "pubDate"),
            }
            entries = []
            for item in channel.findall("item"):
                # content:encoded
                content_el = item.find(f"{{{ns['content']}}}encoded")
                content_txt = content_el.text.strip() if (content_el is not None and content_el.text) else ""
                entries.append({
                    "title":     txt(item, "title"),
                    "link":      txt(item, "link"),
                    "summary":   txt(item, "description"),
                    "author":    txt(item, "author") or txt(item, "dc:creator"),
                    "published": txt(item, "pubDate"),
                    "content":   content_txt,
                })
            return {"meta": meta, "entries": entries}

        # ── Atom 1.0 ──────────────────────────────────────────────────────
        if "feed" in tag or root.tag == f"{{{ns['atom']}}}feed":
            def atom_txt(el, tag):
                child = el.find(f"atom:{tag}", ns) or el.find(tag)
                if child is not None:
                    return (child.text or "").strip()
                return ""

            def atom_link(el):
                for link in el.findall("atom:link", ns) or el.findall("link"):
                    href = link.get("href", "")
                    if href and link.get("rel", "alternate") in ("alternate", ""):
                        return href
                return ""

            meta = {
                "title":       atom_txt(root, "title"),
                "link":        atom_link(root),
                "description": atom_txt(root, "subtitle"),
                "updated":     atom_txt(root, "updated"),
            }
            entries = []
            for entry in root.findall("atom:entry", ns) or root.findall("entry"):
                author_el = entry.find("atom:author/atom:name", ns) or entry.find("author/name")
                content_el = entry.find("atom:content", ns) or entry.find("content")
                summary_el = entry.find("atom:summary", ns) or entry.find("summary")
                entries.append({
                    "title":     atom_txt(entry, "title"),
                    "link":      atom_link(entry),
                    "summary":   (summary_el.text or "").strip() if summary_el is not None else "",
                    "author":    (author_el.text or "").strip() if author_el is not None else "",
                    "published": atom_txt(entry, "published") or atom_txt(entry, "updated"),
                    "content":   (content_el.text or "").strip() if content_el is not None else "",
                })
            return {"meta": meta, "entries": entries}

        return None


# ══════════════════════════════════════════════════════════════════════════════
# 2. EmbeddingCompareTool — Compare texts via cosine similarity
# ══════════════════════════════════════════════════════════════════════════════

class EmbeddingCompareTool(BaseTool):
    name = "embedding_compare"
    description = (
        "Compare texts semantically using AI embeddings + cosine similarity. "
        "Returns similarity score (0.0 = unrelated, 1.0 = identical meaning). "
        "Actions: compare (two texts), rank (find most similar text from a list), "
        "cluster (group a list by topic similarity). "
        "Uses OpenRouter embedding API (model: text-embedding-3-small or compatible)."
    )
    parameters = (
        "action: str, "
        "text: str, "
        "compare_to: str = '', "
        "texts: list = None, "
        "threshold: float = 0.7, "
        "model: str = 'openai/text-embedding-3-small'"
    )

    def __init__(self, api_key: str = ""):
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self._base_url = "https://openrouter.ai/api/v1"

    async def run(
        self,
        action: str,
        text: str,
        compare_to: str = "",
        texts: list | None = None,
        threshold: float = 0.7,
        model: str = "openai/text-embedding-3-small",
    ) -> ToolResult:
        try:
            if not self._api_key:
                return ToolResult(False, "OPENROUTER_API_KEY not set — embedding_compare requires it.")

            texts = texts or []
            threshold = max(0.0, min(1.0, threshold))

            # ── action: compare (2 texts) ──────────────────────────────────
            if action == "compare":
                if not text or not compare_to:
                    return ToolResult(False, "'text' and 'compare_to' both required for 'compare' action.")

                embs = await self._get_embeddings([text, compare_to], model)
                if embs is None:
                    return ToolResult(False, "Failed to get embeddings from API.")

                score = self._cosine_similarity(embs[0], embs[1])
                label = self._score_label(score)

                out = (
                    f"Cosine Similarity: {score:.4f}\n"
                    f"Verdict: {label}\n\n"
                    f"Text A: {text[:120]}{'...' if len(text) > 120 else ''}\n"
                    f"Text B: {compare_to[:120]}{'...' if len(compare_to) > 120 else ''}"
                )
                return ToolResult(True, out, {"score": score, "label": label})

            # ── action: rank (find most similar from list) ─────────────────
            elif action == "rank":
                if not text:
                    return ToolResult(False, "'text' required for 'rank' action.")
                if not texts:
                    return ToolResult(False, "'texts' list required for 'rank' action.")

                all_texts = [text] + texts
                embs = await self._get_embeddings(all_texts, model)
                if embs is None:
                    return ToolResult(False, "Failed to get embeddings from API.")

                query_emb = embs[0]
                scored = []
                for i, candidate_emb in enumerate(embs[1:], 0):
                    score = self._cosine_similarity(query_emb, candidate_emb)
                    scored.append((score, i, texts[i]))

                scored.sort(key=lambda x: x[0], reverse=True)

                lines = [f"Query: {text[:100]}...\n" + "─" * 50]
                for rank, (score, idx, t) in enumerate(scored, 1):
                    above = "✅" if score >= threshold else "❌"
                    lines.append(
                        f"[{rank}] {above} {score:.4f} — {t[:100]}{'...' if len(t) > 100 else ''}"
                    )

                return ToolResult(
                    True,
                    "\n".join(lines),
                    {
                        "ranked": [
                            {"score": s, "index": i, "text": t[:200]}
                            for s, i, t in scored
                        ],
                        "threshold": threshold,
                    },
                )

            # ── action: cluster (group by similarity) ──────────────────────
            elif action == "cluster":
                if not texts or len(texts) < 2:
                    return ToolResult(False, "'texts' list with ≥2 items required for 'cluster' action.")

                embs = await self._get_embeddings(texts, model)
                if embs is None:
                    return ToolResult(False, "Failed to get embeddings from API.")

                # Greedy clustering — assign each text to nearest existing cluster
                clusters: list[list[int]] = []
                cluster_centers: list[list[float]] = []

                for i, emb in enumerate(embs):
                    best_cluster = -1
                    best_score = -1.0
                    for ci, center in enumerate(cluster_centers):
                        s = self._cosine_similarity(emb, center)
                        if s > best_score:
                            best_score = s
                            best_cluster = ci

                    if best_score >= threshold and best_cluster >= 0:
                        clusters[best_cluster].append(i)
                        # update center = mean of cluster embeddings
                        cluster_centers[best_cluster] = self._mean_embedding(
                            [embs[j] for j in clusters[best_cluster]]
                        )
                    else:
                        clusters.append([i])
                        cluster_centers.append(emb)

                lines = [f"Clustered {len(texts)} texts → {len(clusters)} groups (threshold={threshold})\n" + "─" * 50]
                for ci, group in enumerate(clusters, 1):
                    lines.append(f"\nCluster {ci} ({len(group)} items):")
                    for idx in group:
                        t = texts[idx]
                        lines.append(f"  [{idx}] {t[:100]}{'...' if len(t) > 100 else ''}")

                return ToolResult(
                    True,
                    "\n".join(lines),
                    {
                        "cluster_count": len(clusters),
                        "clusters": [[int(i) for i in g] for g in clusters],
                        "threshold": threshold,
                    },
                )

            else:
                return ToolResult(False, f"Unknown action: '{action}'. Valid: compare, rank, cluster.")

        except Exception as e:
            return ToolResult(False, f"embedding_compare error: {e}")

    # ── helpers ───────────────────────────────────────────────────────────────

    async def _get_embeddings(self, texts: list[str], model: str) -> list[list[float]] | None:
        """Call OpenRouter embeddings endpoint, return list of embedding vectors."""
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(
                    f"{self._base_url}/embeddings",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://gazcc.ai",
                        "X-Title": "GazccAgent",
                    },
                    json={
                        "model": model,
                        "input": texts,
                    },
                )
                r.raise_for_status()
                data = r.json()
                # OpenAI-compatible: data["data"] is list of {embedding: [...], index: int}
                sorted_data = sorted(data["data"], key=lambda x: x["index"])
                return [item["embedding"] for item in sorted_data]
        except Exception as e:
            return None

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Pure Python cosine similarity — no numpy needed."""
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(y * y for y in b))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return round(dot / (mag_a * mag_b), 6)

    @staticmethod
    def _mean_embedding(embs: list[list[float]]) -> list[float]:
        """Element-wise mean of a list of embeddings."""
        n = len(embs)
        dim = len(embs[0])
        return [sum(embs[i][d] for i in range(n)) / n for d in range(dim)]

    @staticmethod
    def _score_label(score: float) -> str:
        if score >= 0.95: return "Virtually identical 🟢"
        if score >= 0.85: return "Very similar 🟢"
        if score >= 0.70: return "Similar 🟡"
        if score >= 0.50: return "Loosely related 🟠"
        if score >= 0.30: return "Weakly related 🔴"
        return "Unrelated ⚫"


# ══════════════════════════════════════════════════════════════════════════════
# REGISTER FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def register_new_tools(registry, cfg: dict | None = None) -> None:
    """
    Register rss_reader and embedding_compare into an existing ToolRegistry.

    USAGE in agent/core.py:
        from agent.new_tools import register_new_tools
        register_new_tools(self._tools, self._cfg)
    """
    cfg = cfg or {}
    tool_cfg = cfg.get("tools", {})
    llm_cfg  = cfg.get("llm", {})

    # Always register rss_reader — no API key needed
    if tool_cfg.get("rss_reader", True):
        registry.register(RSSReaderTool())

    # Register embedding_compare — requires OpenRouter API key
    if tool_cfg.get("embedding_compare", True):
        api_key = (
            llm_cfg.get("api_key", "")
            or os.environ.get("OPENROUTER_API_KEY", "")
        )
        registry.register(EmbeddingCompareTool(api_key=api_key))
