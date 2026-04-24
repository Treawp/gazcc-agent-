"""
agent/video_analyzer.py
Analisis video YouTube & TikTok via oEmbed + Transcript API.
Tidak download video — ambil metadata + transcript + analyze.
"""

import re
import json
import asyncio
import httpx
from typing import Optional

try:
    from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
    HAS_TRANSCRIPT_API = True
except ImportError:
    HAS_TRANSCRIPT_API = False

from .tools import BaseTool, ToolResult


# ── helpers ───────────────────────────────────────────────────────────────────

def _detect_platform(url: str) -> str:
    url = url.lower()
    if "tiktok.com" in url or "vm.tiktok.com" in url:
        return "tiktok"
    if "youtube.com" in url or "youtu.be" in url:
        return "youtube"
    return "unknown"


def _extract_youtube_id(url: str) -> Optional[str]:
    patterns = [
        r"youtube\.com/shorts/([a-zA-Z0-9_-]{11})",
        r"youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})",
        r"youtu\.be/([a-zA-Z0-9_-]{11})",
        r"youtube\.com/embed/([a-zA-Z0-9_-]{11})",
        r"youtube\.com/v/([a-zA-Z0-9_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


async def _fetch_oembed(url: str, platform: str) -> dict:
    """Ambil metadata via oEmbed — gratis, no API key."""
    endpoints = {
        "youtube": f"https://www.youtube.com/oembed?url={url}&format=json",
        "tiktok":  f"https://www.tiktok.com/oembed?url={url}",
    }
    fallback = f"https://noembed.com/embed?url={url}"

    headers = {"User-Agent": "Mozilla/5.0 (compatible; GazccBot/1.0)"}

    async with httpx.AsyncClient(timeout=12, follow_redirects=True) as client:
        # Primary
        try:
            r = await client.get(endpoints[platform], headers=headers)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass

        # Fallback noembed
        try:
            r = await client.get(fallback, headers=headers)
            if r.status_code == 200:
                data = r.json()
                if data.get("error"):
                    return {}
                return data
        except Exception:
            pass

    return {}


async def _fetch_youtube_transcript(video_id: str) -> tuple[str, str]:
    """
    Returns (transcript_text, lang_used).
    Coba ID dulu, fallback EN, fallback auto-generated.
    """
    if not HAS_TRANSCRIPT_API:
        return "", "N/A (youtube-transcript-api tidak terinstall)"

    def _sync_fetch():
        priority_langs = ["id", "en", "en-US", "en-GB"]
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=priority_langs)
            lang = priority_langs[0]
            text = " ".join(t["text"] for t in transcript)
            return text, lang
        except NoTranscriptFound:
            pass

        # Coba semua yang tersedia (auto-generated termasuk)
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            for t in transcript_list:
                try:
                    fetched = t.fetch()
                    text = " ".join(seg["text"] for seg in fetched)
                    return text, t.language_code
                except Exception:
                    continue
        except Exception:
            pass

        return "", "Tidak tersedia"

    loop = asyncio.get_event_loop()
    try:
        text, lang = await loop.run_in_executor(None, _sync_fetch)
        return text, lang
    except Exception as e:
        return "", f"Error: {e}"


async def _fetch_tiktok_metadata(url: str) -> dict:
    """
    TikTok tidak punya transcript API publik.
    Ambil metadata tambahan via scrape ringan.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) "
                      "AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148",
        "Accept-Language": "id-ID,id;q=0.9,en;q=0.8",
    }
    extra = {}
    try:
        async with httpx.AsyncClient(timeout=12, follow_redirects=True) as client:
            r = await client.get(url, headers=headers)
            if r.status_code == 200:
                html = r.text
                # Extract og tags
                og_tags = {
                    "og:title":       re.search(r'<meta[^>]+property="og:title"[^>]+content="([^"]+)"', html),
                    "og:description": re.search(r'<meta[^>]+property="og:description"[^>]+content="([^"]+)"', html),
                    "og:video":       re.search(r'<meta[^>]+property="og:video"[^>]+content="([^"]+)"', html),
                }
                for key, match in og_tags.items():
                    if match:
                        extra[key.replace("og:", "")] = match.group(1)
    except Exception:
        pass
    return extra


# ── Tool class ─────────────────────────────────────────────────────────────────

class VideoAnalyzerTool(BaseTool):
    name = "analyze_video"
    description = (
        "Analisis video dari link YouTube atau TikTok. "
        "Ambil judul, author, deskripsi, dan transkrip (YouTube). "
        "Gunakan untuk summarize, cek konten, analisis sentimen, atau extract info dari video. "
        "Input: URL YouTube/TikTok. Optional: focus = 'summary' | 'sentiment' | 'keywords' | 'general'."
    )
    parameters = "url: str, focus: str = 'general'"

    async def run(self, url: str, focus: str = "general") -> ToolResult:
        platform = _detect_platform(url)

        if platform == "unknown":
            return ToolResult(
                False,
                "URL tidak dikenali. Pastikan link dari YouTube atau TikTok.\n"
                "Contoh: https://youtube.com/watch?v=xxx atau https://www.tiktok.com/@user/video/xxx"
            )

        lines = [
            f"{'='*55}",
            f"  ANALISIS VIDEO — {platform.upper()}",
            f"{'='*55}",
            f"URL     : {url}",
            f"Platform: {platform.upper()}",
            f"Focus   : {focus}",
            "",
        ]

        # ── Metadata via oEmbed ────────────────────────────────────────────
        meta = await _fetch_oembed(url, platform)

        if meta:
            lines += [
                "── METADATA ──────────────────────────────────────────",
                f"Judul   : {meta.get('title', 'N/A')}",
                f"Author  : {meta.get('author_name', 'N/A')}",
                f"Provider: {meta.get('provider_name', platform.capitalize())}",
            ]
            if meta.get("thumbnail_url"):
                lines.append(f"Thumb   : {meta.get('thumbnail_url')}")
            if meta.get("width") and meta.get("height"):
                lines.append(f"Dimensi : {meta.get('width')}x{meta.get('height')}")
            lines.append("")
        else:
            lines += [
                "── METADATA ──────────────────────────────────────────",
                "⚠ Metadata tidak bisa diambil via oEmbed.",
                "",
            ]

        # ── Transcript / Extra Content ─────────────────────────────────────
        transcript_text = ""
        extra_content = ""

        if platform == "youtube":
            video_id = _extract_youtube_id(url)
            if video_id:
                lines.append(f"Video ID: {video_id}")
                transcript_text, lang = await _fetch_youtube_transcript(video_id)
                lines += [
                    "",
                    "── TRANSKRIP ─────────────────────────────────────────",
                    f"Bahasa  : {lang}",
                ]
                if transcript_text:
                    char_count = len(transcript_text)
                    word_count = len(transcript_text.split())
                    lines += [
                        f"Panjang : {char_count} chars / {word_count} kata",
                        "",
                        "[ ISI TRANSKRIP ]",
                        transcript_text[:6000] + (" ... [terpotong]" if char_count > 6000 else ""),
                        "",
                    ]
                else:
                    lines += [
                        "Status  : Transkrip tidak tersedia untuk video ini.",
                        "(Video mungkin: tidak ada caption, private, atau CC dimatikan)",
                        "",
                    ]
            else:
                lines.append("⚠ Gagal extract Video ID dari URL.")

        elif platform == "tiktok":
            lines += [
                "",
                "── KONTEN TIKTOK ─────────────────────────────────────",
                "ℹ TikTok tidak menyediakan transcript publik.",
            ]
            extra = await _fetch_tiktok_metadata(url)
            if extra:
                if extra.get("title"):
                    lines.append(f"Judul (OG): {extra.get('title')}")
                if extra.get("description"):
                    lines.append(f"Deskripsi : {extra.get('description')}")
                extra_content = json.dumps(extra, ensure_ascii=False)
            else:
                lines.append("⚠ Tidak bisa scrape metadata tambahan (bot protection).")
            lines.append("")

        # ── Build context for analysis ─────────────────────────────────────
        analysis_content = []
        if meta.get("title"):
            analysis_content.append(f"Judul: {meta.get('title')}")
        if meta.get("author_name"):
            analysis_content.append(f"Author: {meta.get('author_name')}")
        if transcript_text:
            analysis_content.append(f"Transkrip: {transcript_text[:4000]}")
        if extra_content:
            analysis_content.append(f"Extra: {extra_content}")

        lines += [
            "── RINGKASAN DATA ────────────────────────────────────────",
            f"Status  : {'✓ BERHASIL — data cukup untuk analisis' if analysis_content else '⚠ Data minimal — analisis terbatas'}",
            f"Konten  : {len(analysis_content)} komponen tersedia",
            "=" * 55,
        ]

        if not analysis_content:
            lines += [
                "",
                "CATATAN: Tidak ada data yang bisa diambil.",
                "Kemungkinan penyebab:",
                "- Video private/deleted",
                "- Region block",
                "- Bot protection dari platform",
            ]

        return ToolResult(
            success=bool(analysis_content or meta),
            output="\n".join(lines),
            metadata={
                "platform": platform,
                "title": meta.get("title", ""),
                "author": meta.get("author_name", ""),
                "has_transcript": bool(transcript_text),
                "transcript_len": len(transcript_text),
                "focus": focus,
            }
        )


# ── Registration ───────────────────────────────────────────────────────────────

def register_video_tools(registry, cfg: dict):
    registry.register(VideoAnalyzerTool())
