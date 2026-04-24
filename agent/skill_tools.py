"""
agent/skill_tools.py
═══════════════════════════════════════════════════════════════════════════════
  GAZCC — SKILL TOOLS SUITE
  ══════════════════════════════════════════════════════════════════════════════
  Tools (terinspirasi dari Claude Skills marketplace):

    grill_me            — Kritik brutal terhadap ide/desain sampai matang
    frontend_design     — Generate UI/frontend siap produksi, kualitas tinggi
    theme_factory       — Buat & kelola sistem branding otomatis
    browser_use         — Kasih agent akses ke browser (search + fetch + extract)
    trailofbits_security — Audit keamanan kode menggunakan metodologi Trail of Bits
    skill_creator       — Panduan step-by-step bikin skill/tool baru
    superpowers_skills  — Supercharge coding workflow untuk agent

  REGISTER ke ToolRegistry:
    from .skill_tools import register_skill_tools
    register_skill_tools(registry, cfg)

  CONFIG (config.yaml):
    tools:
      skill_tools: true
      skill_tools_model: ""   # override model khusus untuk skill tools (opsional)
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
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
        def __str__(self):
            return f"{'✓' if self.success else '✗'} {self.output}"

    class BaseTool:
        name: str = ""
        description: str = ""
        parameters: str = ""
        async def run(self, *a, **kw) -> ToolResult:
            return ToolResult(False, "not implemented")


# ── LLM Base Helper ───────────────────────────────────────────────────────────

class _LLMToolBase(BaseTool):
    """Base class untuk tools yang butuh LLM sub-call."""

    def __init__(self, cfg: dict):
        llm = cfg.get("llm", {})
        tool_cfg = cfg.get("tools", {})
        self._base_url = llm.get("base_url", "https://openrouter.ai/api/v1").rstrip("/")
        self._api_key = (
            llm.get("api_key", "")
            or os.environ.get("OPENROUTER_API_KEY", "")
            or os.environ.get("ANTHROPIC_API_KEY", "")
        )
        # Bisa override model khusus skill tools via config
        self._model = (
            tool_cfg.get("skill_tools_model", "")
            or llm.get("model", "deepseek/deepseek-v4-flash")
        )

    async def _llm(
        self,
        system: str,
        user: str,
        max_tokens: int = 4000,
        temperature: float = 0.2,
    ) -> str:
        """Call LLM dengan system + user prompt, return content string."""
        if not self._api_key:
            raise RuntimeError(
                "API key tidak ditemukan. Set OPENROUTER_API_KEY di .env atau config.yaml."
            )
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://gazcc.vercel.app",
                    "X-Title": "GazccAgent",
                },
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return content.strip()


# ══════════════════════════════════════════════════════════════════════════════
# 1. GRILL ME — Brutal Idea Critique
# ══════════════════════════════════════════════════════════════════════════════

class GrillMeTool(_LLMToolBase):
    """
    Membantu memikirkan sebuah ide sampai matang dan memastikan kamu
    memilih desain yang paling tepat. Terinspirasi dari /grill-me/ skill.
    """

    name = "grill_me"
    description = (
        "Kritik brutal terhadap sebuah ide, arsitektur, desain, atau rencana. "
        "Expose semua kelemahan, asumsi yang salah, dan masalah scalability. "
        "Output: verdict, critical flaws, missing parts, dan pendekatan yang lebih baik. "
        "Gunakan sebelum commit ke suatu pendekatan teknis."
    )
    parameters = (
        "idea: str (ide/desain/kode yang mau dikritik), "
        "context: str = '' (konteks tambahan: tech stack, constraints, goals)"
    )

    _SYSTEM = """\
Kamu adalah senior engineer dengan 20 tahun pengalaman yang tugasnya adalah GRILL — mengkritik ide secara brutal dan jujur.
Gaya: blunt, langsung, technically sharp. Tidak ada basa-basi. Expose SEMUA kelemahan.

Format output WAJIB:
🔴 VERDICT (1 kalimat, keras dan jujur)

⚡ CRITICAL FLAWS
(masalah teknis spesifik, numbered, reference teknologi yang relevan)

🕳️ WHAT'S MISSING
(security, scalability, edge cases, error handling, business logic gaps)

💡 BETTER APPROACH
(alternatif konkret yang lebih pintar, dengan reasoning)

🛠️ IF YOU INSIST ON THIS APPROACH
(versi minimal yang masih survivable dari ide original)

Jangan sugarcoat. Jadi mentor yang bilang kebenaran pahit."""

    async def run(self, idea: str, context: str = "", **_) -> ToolResult:
        if not idea.strip():
            return ToolResult(False, "Parameter 'idea' tidak boleh kosong.")
        try:
            user_msg = f"Grill ide berikut ini{f' (context: {context})' if context else ''}:\n\n{idea}"
            result = await self._llm(self._SYSTEM, user_msg, max_tokens=2000, temperature=0.3)
            return ToolResult(True, result, {"tool": "grill_me", "idea_len": len(idea)})
        except Exception as e:
            return ToolResult(False, f"grill_me gagal: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. FRONTEND DESIGN — Production UI Generator
# ══════════════════════════════════════════════════════════════════════════════

class FrontendDesignTool(_LLMToolBase):
    """
    Membuat interface frontend yang siap produksi dengan kualitas desain tinggi.
    Terinspirasi dari /frontend-design/ skill.
    """

    name = "frontend_design"
    description = (
        "Generate kode HTML/CSS/JS frontend production-ready dengan desain berkualitas tinggi. "
        "Output: single-file HTML dengan embedded CSS dan JS, siap pakai langsung. "
        "Support: landing pages, dashboards, UI components, forms, modal, navbar, dll. "
        "Style options: dark, cyberpunk, glassmorphism, minimal, brutalist, neomorphism, dll."
    )
    parameters = (
        "description: str (apa yang mau dibuat — detail komponen/halaman), "
        "style: str = 'dark' (style desain: dark|cyberpunk|minimal|glassmorphism|brutalist|neomorphism|colorful), "
        "output_file: str = '' (opsional: path file untuk menyimpan hasil HTML)"
    )

    _SYSTEM = """\
Kamu adalah world-class frontend engineer dan visual designer.
Task: generate kode HTML/CSS/JS yang STRIKING, production-ready, single-file.

Rules wajib:
- Semua dalam satu file HTML (embedded <style> dan <script>)
- Design HARUS visual striking — bukan template generik
- Mobile responsive
- Functional (interaksi bekerja, tidak hanya tampilan)
- Tidak perlu external CDN kecuali benar-benar diperlukan
- Jangan pakai font overused (Arial, Roboto, Inter)
- CSS custom properties untuk theming
- Animasi subtle tapi polished

Output: HANYA kode HTML dimulai dari <!DOCTYPE html. Tidak ada penjelasan, tidak ada markdown fence."""

    async def run(
        self,
        description: str,
        style: str = "dark",
        output_file: str = "",
        **_,
    ) -> ToolResult:
        if not description.strip():
            return ToolResult(False, "Parameter 'description' tidak boleh kosong.")
        try:
            user_msg = f"Create: {description}\nStyle direction: {style or 'dark cyberpunk'}"
            html = await self._llm(self._SYSTEM, user_msg, max_tokens=4000, temperature=0.4)

            # Clean up jika ada markdown fence
            html = re.sub(r"^```(?:html)?\s*", "", html, flags=re.MULTILINE)
            html = re.sub(r"\s*```$", "", html, flags=re.MULTILINE)
            html = html.strip()

            # Simpan ke file jika diminta
            saved_path = ""
            if output_file:
                try:
                    import aiofiles
                    os.makedirs(os.path.dirname(os.path.abspath(output_file)) or ".", exist_ok=True)
                    async with aiofiles.open(output_file, "w", encoding="utf-8") as f:
                        await f.write(html)
                    saved_path = output_file
                except Exception as fe:
                    saved_path = f"(gagal simpan: {fe})"

            meta = {"tool": "frontend_design", "style": style, "chars": len(html)}
            if saved_path:
                meta["saved_to"] = saved_path

            summary = f"✓ HTML generated ({len(html):,} chars){f' → saved to {saved_path}' if saved_path else ''}\n\n{html[:500]}..."
            return ToolResult(True, html, meta)

        except Exception as e:
            return ToolResult(False, f"frontend_design gagal: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. THEME FACTORY — Brand System Generator
# ══════════════════════════════════════════════════════════════════════════════

class ThemeFactoryTool(_LLMToolBase):
    """
    Mengelola branding secara otomatis — buat sistem branding lengkap.
    Terinspirasi dari /theme-factory/ skill.
    """

    name = "theme_factory"
    description = (
        "Generate sistem branding lengkap untuk sebuah brand/produk secara otomatis. "
        "Output: palet warna (dengan hex codes), tipografi, logo concept, voice & tone, "
        "CSS design tokens siap pakai, dan usage guidelines. "
        "Gunakan untuk project baru yang butuh identitas visual yang kohesif."
    )
    parameters = (
        "brand_name: str (nama brand atau produk), "
        "style_direction: str = '' (kepribadian brand, misal: 'dark tech aggressive', 'friendly colorful startup', 'luxury minimal'), "
        "industry: str = '' (industri/kategori produk, opsional)"
    )

    _SYSTEM = """\
Kamu adalah world-class brand designer dan design systems expert.
Task: buat sistem branding LENGKAP yang cohesive, memorable, dan immediately usable.

Format output:

## 🎯 Brand Positioning
(1 kalimat positioning statement yang sharp)

## 🎨 Color Palette
(5-7 warna dengan: nama, hex code, dan use case spesifik)
```
Primary:   #XXXXXX  — [use case]
Secondary: #XXXXXX  — [use case]
...
```

## ✍️ Typography Stack
(font pairing + size scale, jelaskan karakter masing-masing font)

## 💡 Logo Concept
(deskripsi visual detail — bentuk, simbolisme, potential execution)

## 📢 Voice & Tone
(3 DO examples + 3 DON'T examples)

## 🛠️ CSS Design Tokens
```css
:root {
  --color-primary: #XXXXXX;
  ...
  --font-display: 'FontName', sans-serif;
  --spacing-base: 8px;
  /* semua token yang dibutuhkan */
}
```

## 📐 Usage Examples
(bagaimana mengaplikasikan sistem ini di UI)

Jadilah opinionated dan spesifik. Tidak ada generic placeholder."""

    async def run(
        self,
        brand_name: str,
        style_direction: str = "",
        industry: str = "",
        **_,
    ) -> ToolResult:
        if not brand_name.strip():
            return ToolResult(False, "Parameter 'brand_name' tidak boleh kosong.")
        try:
            parts = [f"Brand: \"{brand_name}\""]
            if style_direction:
                parts.append(f"Style direction: {style_direction}")
            if industry:
                parts.append(f"Industry: {industry}")
            user_msg = "\n".join(parts)

            result = await self._llm(self._SYSTEM, user_msg, max_tokens=3000, temperature=0.5)
            return ToolResult(True, result, {"tool": "theme_factory", "brand": brand_name})
        except Exception as e:
            return ToolResult(False, f"theme_factory gagal: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. BROWSER USE — Agentic Browser Access
# ══════════════════════════════════════════════════════════════════════════════

class BrowserUseTool(BaseTool):
    """
    Memberikan agent akses ke browser: search web, buka URL, extract konten.
    Terinspirasi dari /browser-use/ skill.
    """

    name = "browser_use"
    description = (
        "Berikan agent akses ke browser — bisa search web, buka URL, dan extract konten halaman. "
        "Mode: 'search' untuk cari informasi, 'fetch' untuk buka URL spesifik, "
        "'research' untuk riset topik lengkap (search + fetch + synthesize). "
        "Gunakan untuk: cek dokumentasi, cari contoh kode, validasi informasi terkini."
    )
    parameters = (
        "task: str (apa yang mau dicari/dibuka/diresearch — natural language), "
        "mode: str = 'auto' (auto|search|fetch|research), "
        "url: str = '' (URL spesifik untuk mode fetch), "
        "max_results: int = 5"
    )

    _UA = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
    ]

    def _headers(self) -> dict:
        return {
            "User-Agent": self._UA[int(time.time()) % len(self._UA)],
            "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
            "Accept-Language": "id,en-US;q=0.7,en;q=0.3",
            "DNT": "1",
        }

    async def _search_ddg(self, query: str, max_results: int = 5) -> list[dict]:
        """DuckDuckGo instant answer API + HTML fallback."""
        results = []

        # Try DDG Instant Answer API dulu
        try:
            async with httpx.AsyncClient(timeout=15, headers=self._headers()) as client:
                resp = await client.get(
                    "https://api.duckduckgo.com/",
                    params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
                )
                data = resp.json()
                if data.get("AbstractText"):
                    results.append({
                        "title": data.get("Heading", "DuckDuckGo Abstract"),
                        "url": data.get("AbstractURL", ""),
                        "snippet": data.get("AbstractText", ""),
                        "source": "ddg_instant",
                    })
                for r in data.get("RelatedTopics", [])[:max_results]:
                    if isinstance(r, dict) and r.get("FirstURL"):
                        results.append({
                            "title": r.get("Text", "")[:100],
                            "url": r.get("FirstURL", ""),
                            "snippet": r.get("Text", ""),
                            "source": "ddg_related",
                        })
        except Exception:
            pass

        # DDG HTML search sebagai fallback
        if len(results) < max_results:
            try:
                async with httpx.AsyncClient(
                    timeout=20,
                    headers=self._headers(),
                    follow_redirects=True,
                ) as client:
                    resp = await client.get(
                        "https://html.duckduckgo.com/html/",
                        params={"q": query},
                    )
                    # Simple regex extraction (no BS4 dependency)
                    links = re.findall(
                        r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>([^<]+)</a>',
                        resp.text,
                    )
                    snippets = re.findall(
                        r'<a[^>]+class="result__snippet"[^>]*>([^<]+)</a>',
                        resp.text,
                    )
                    for i, (url, title) in enumerate(links[:max_results]):
                        if not any(r["url"] == url for r in results):
                            results.append({
                                "title": title.strip(),
                                "url": url,
                                "snippet": snippets[i].strip() if i < len(snippets) else "",
                                "source": "ddg_html",
                            })
            except Exception:
                pass

        return results[:max_results]

    async def _fetch_url(self, url: str, max_chars: int = 8000) -> str:
        """Fetch URL dan extract teks penting."""
        try:
            async with httpx.AsyncClient(
                timeout=20,
                headers=self._headers(),
                follow_redirects=True,
            ) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                html = resp.text

                # Strip tags, ambil teks
                text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r"<[^>]+>", " ", text)
                text = re.sub(r"\s{3,}", "\n\n", text)
                text = text.strip()

                return text[:max_chars] + (f"\n\n[... truncated at {max_chars} chars]" if len(text) > max_chars else "")
        except Exception as e:
            return f"Gagal fetch {url}: {e}"

    async def run(
        self,
        task: str,
        mode: str = "auto",
        url: str = "",
        max_results: int = 5,
        **_,
    ) -> ToolResult:
        if not task.strip():
            return ToolResult(False, "Parameter 'task' tidak boleh kosong.")

        # Auto-detect mode
        if mode == "auto":
            if url:
                mode = "fetch"
            elif any(kw in task.lower() for kw in ["riset", "research", "lengkap", "comprehensive", "semua tentang"]):
                mode = "research"
            else:
                mode = "search"

        try:
            output_parts = [f"🌐 Browser Use — Mode: {mode.upper()}\nTask: {task}\n"]

            if mode == "fetch" or (mode == "auto" and url):
                if not url:
                    return ToolResult(False, "Mode 'fetch' butuh parameter 'url'.")
                content = await self._fetch_url(url)
                output_parts.append(f"📄 Content dari {url}:\n{content}")

            elif mode == "search":
                results = await self._search_ddg(task, max_results)
                if not results:
                    return ToolResult(False, f"Tidak ada hasil untuk: {task}")
                output_parts.append(f"🔍 Search Results ({len(results)} hasil):\n")
                for i, r in enumerate(results, 1):
                    output_parts.append(
                        f"{i}. {r['title']}\n   URL: {r['url']}\n   {r['snippet'][:200]}\n"
                    )

            elif mode == "research":
                # Search dulu
                results = await self._search_ddg(task, max_results)
                if not results:
                    return ToolResult(False, f"Tidak ada hasil untuk: {task}")

                output_parts.append(f"🔍 Search Results:\n")
                for i, r in enumerate(results, 1):
                    output_parts.append(f"{i}. {r['title']} — {r['url']}")

                # Fetch top 2 hasil untuk konten lebih dalam
                output_parts.append(f"\n📄 Page Content (top 2 results):\n")
                tasks_fetch = [self._fetch_url(r["url"], 3000) for r in results[:2]]
                contents = await asyncio.gather(*tasks_fetch, return_exceptions=True)
                for r, content in zip(results[:2], contents):
                    if isinstance(content, Exception):
                        content = f"Gagal fetch: {content}"
                    output_parts.append(f"\n--- {r['title']} ---\n{content}\n")

            output = "\n".join(output_parts)
            return ToolResult(True, output, {"tool": "browser_use", "mode": mode})

        except Exception as e:
            return ToolResult(False, f"browser_use gagal: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. TRAILOFBITS SECURITY — Security Audit Tool
# ══════════════════════════════════════════════════════════════════════════════

class TrailOfBitsSecurityTool(_LLMToolBase):
    """
    Berisi praktik keamanan penting untuk website/kode hasil vibe coding.
    Terinspirasi dari /trailofbits/skills/.
    """

    name = "trailofbits_security"
    description = (
        "Audit keamanan kode mengikuti metodologi Trail of Bits — firm security penelitian terkemuka. "
        "Identifikasi: injection vulnerabilities, auth issues, insecure configs, XSS/CSRF, "
        "hardcoded secrets, logic flaws, dan masalah cryptography. "
        "Output: severity classification, attack vectors, dan fix code yang konkret. "
        "Ideal untuk audit kode vibe-coded atau pre-production security review."
    )
    parameters = (
        "code: str (kode yang mau diaudit), "
        "language: str = '' (bahasa pemrograman: python|js|php|lua|go|rust|dll), "
        "context: str = '' (konteks: web app|API|smart contract|mobile|dll)"
    )

    _SYSTEM = """\
Kamu adalah security researcher dari Trail of Bits — firma security research kelas dunia.
Task: lakukan security audit profesional yang thorough.

Format output WAJIB:

## 🔴 CRITICAL (CVSS 9-10) — Fix SEKARANG atau jangan deploy
[Setiap finding: nama vulnerability, line reference jika ada, attack vector, impact]

## 🟠 HIGH (CVSS 7-8.9)
[sama seperti di atas]

## 🟡 MEDIUM (CVSS 4-6.9)
[sama seperti di atas]

## 🟢 LOW / INFORMATIONAL
[sama seperti di atas]

## 🛠️ REMEDIATION CODE
[Untuk setiap finding CRITICAL dan HIGH: berikan kode fix yang siap pakai]

## 📊 SECURITY SCORECARD
- Overall Score: X/10
- Risk Level: CRITICAL|HIGH|MEDIUM|LOW
- Deploy Ready: YES|NO
- Summary (2 kalimat)

Jadilah sangat spesifik: reference line number, nama CVE jika relevan, kode exploit proof-of-concept (untuk edukasi)."""

    async def run(
        self,
        code: str,
        language: str = "",
        context: str = "",
        **_,
    ) -> ToolResult:
        if not code.strip():
            return ToolResult(False, "Parameter 'code' tidak boleh kosong.")
        try:
            lang_hint = f" ({language})" if language else ""
            ctx_hint = f"\nContext: {context}" if context else ""
            user_msg = f"Audit kode berikut{lang_hint}:{ctx_hint}\n\n```{language}\n{code}\n```"

            result = await self._llm(self._SYSTEM, user_msg, max_tokens=3000, temperature=0.1)
            return ToolResult(
                True, result,
                {"tool": "trailofbits_security", "language": language, "code_lines": code.count("\n") + 1}
            )
        except Exception as e:
            return ToolResult(False, f"trailofbits_security gagal: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. SKILL CREATOR — Custom Skill Builder
# ══════════════════════════════════════════════════════════════════════════════

class SkillCreatorTool(_LLMToolBase):
    """
    Memandu kamu langkah demi langkah untuk membuat skill milikmu sendiri.
    Terinspirasi dari /skill-creator/ skill.
    """

    name = "skill_creator"
    description = (
        "Buat spesifikasi skill/tool baru secara terstruktur — step by step. "
        "Output: dokumen SKILL.md lengkap dengan: purpose, triggers, internal process, "
        "parameters, contoh konkret, edge cases, best practices, dan implementation notes. "
        "Gunakan untuk mendokumentasikan tool baru atau workflow yang ingin di-automate."
    )
    parameters = (
        "skill_name: str (nama skill yang mau dibuat), "
        "skill_description: str (apa yang dilakukan skill ini), "
        "use_cases: str = '' (kapan digunakan, contoh use cases), "
        "output_file: str = '' (opsional: path untuk menyimpan SKILL.md)"
    )

    _SYSTEM = """\
Kamu adalah AI skill architect yang expert dalam membuat dokumentasi skill yang actionable.
Task: buat dokumen SKILL.md yang lengkap, terstruktur, dan langsung bisa digunakan.

Format output WAJIB (markdown):

---
name: [skill-name]
description: [1 kalimat deskripsi]
---

# Overview & Purpose
[Apa yang dilakukan skill ini dan kenapa penting]

# When to Use (Triggers)
Trigger phrases dan konteks yang memicu skill ini digunakan.
- Trigger: [contoh kalimat/konteks]
- ...

# When NOT to Use (Anti-Patterns)
Situasi dimana skill ini TIDAK tepat digunakan.
- ...

# Internal Process (Step-by-Step)
Bagaimana skill ini bekerja secara internal:
1. ...
2. ...

# Input Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| ... | ... | Yes/No | ... |

# Output Format
[Deskripsi format output yang dihasilkan]

# Examples

## Example 1: [Use Case]
Input: ...
Output: ...

## Example 2: [Use Case]
Input: ...
Output: ...

## Example 3: Edge Case
Input: ...
Output: ...

# Edge Cases & Error Handling
- ...

# Best Practices
- ...

# Implementation Notes
[Catatan teknis untuk implementasi]

Jadilah sangat spesifik dan concrete — bukan generic placeholder."""

    async def run(
        self,
        skill_name: str,
        skill_description: str,
        use_cases: str = "",
        output_file: str = "",
        **_,
    ) -> ToolResult:
        if not skill_name.strip() or not skill_description.strip():
            return ToolResult(False, "Parameter 'skill_name' dan 'skill_description' wajib diisi.")
        try:
            user_msg = (
                f"Create skill spec:\n"
                f"Name: {skill_name}\n"
                f"Description: {skill_description}\n"
                f"Use cases: {use_cases or 'general purpose'}"
            )
            result = await self._llm(self._SYSTEM, user_msg, max_tokens=3000, temperature=0.3)

            # Simpan ke file jika diminta
            saved_path = ""
            if output_file:
                try:
                    import aiofiles
                    os.makedirs(os.path.dirname(os.path.abspath(output_file)) or ".", exist_ok=True)
                    async with aiofiles.open(output_file, "w", encoding="utf-8") as f:
                        await f.write(result)
                    saved_path = output_file
                except Exception as fe:
                    saved_path = f"(gagal simpan: {fe})"

            meta = {"tool": "skill_creator", "skill": skill_name}
            if saved_path:
                meta["saved_to"] = saved_path

            return ToolResult(True, result, meta)
        except Exception as e:
            return ToolResult(False, f"skill_creator gagal: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. SUPERPOWERS SKILLS — Coding Workflow Supercharger
# ══════════════════════════════════════════════════════════════════════════════

class SuperpowersSkillsTool(_LLMToolBase):
    """
    Melengkapi workflow pengembangan software untuk coding agent.
    Terinspirasi dari /superpowers/skills/.
    """

    name = "superpowers_skills"
    description = (
        "Supercharge coding workflow dengan analisis level senior engineer. "
        "Lakukan: deep code analysis, refactoring roadmap, architecture review, "
        "performance bottleneck detection, tech stack recommendations, dan "
        "implementation plan yang execution-ready. "
        "Gunakan untuk: review codebase, plan fitur baru, solve complex engineering problems."
    )
    parameters = (
        "task: str (coding task, masalah, atau codebase yang mau dianalisis), "
        "codebase_context: str = '' (bahasa, framework, constraints, existing architecture), "
        "focus: str = 'full' (full|architecture|performance|refactor|debug|planning)"
    )

    _SYSTEMS = {
        "full": """\
Kamu adalah world-class software architect dalam SUPERPOWER mode.
Berikan analisis engineering yang ELITE — tidak ada generic advice.

Output format:

## ⚡ QUICK WINS (high impact, low effort — lakukan SEKARANG)
[List konkret, bukan teori]

## 🏗️ ARCHITECTURE ANALYSIS
[Apa yang solid, apa yang perlu diganti, kenapa]

## 🔄 REFACTORING ROADMAP
[Prioritized — P1/P2/P3 dengan reasoning dan estimasi effort]

## 🐌 PERFORMANCE BOTTLENECKS
[Identifikasi + fix spesifik + expected improvement]

## 🛠️ TECH STACK VERDICT
[Keep/Replace/Add untuk setiap komponen — dengan reasoning]

## 📋 IMPLEMENTATION PLAN
[Step-by-step execution-ready plan — bukan high-level fluff]

## ⚠️ CRITICAL PITFALLS
[Hal yang biasanya salah di project tipe ini — be specific]

Brutal honesty. No sugarcoating. Senior engineer yang pernah fix production disasters.""",

        "architecture": """\
Kamu adalah principal architect. Fokus: architecture review mendalam.
Analisa: coupling, cohesion, scalability, maintainability, patterns yang digunakan.
Berikan: architecture diagram (ASCII), masalah yang ada, dan target architecture.""",

        "performance": """\
Kamu adalah performance engineer. Fokus: bottleneck detection dan optimization.
Tools: profiling strategy, benchmarking approach, caching strategy, async patterns.
Output: list bottleneck dengan severity + konkret fix code.""",

        "refactor": """\
Kamu adalah refactoring specialist. Fokus: code quality improvement tanpa breaking changes.
Identifikasi: code smells, duplication, complexity, coupling.
Output: refactoring plan dengan before/after code examples.""",

        "debug": """\
Kamu adalah debugging expert. Fokus: root cause analysis dan fix.
Process: reproduce → isolate → fix → verify → prevent.
Output: probable causes (ranked), debugging steps, fix implementation.""",

        "planning": """\
Kamu adalah technical project manager + architect.
Fokus: breakdown fitur/task menjadi implementasi yang concrete dan deliverable.
Output: epic → stories → tasks → acceptance criteria → tech notes.""",
    }

    async def run(
        self,
        task: str,
        codebase_context: str = "",
        focus: str = "full",
        **_,
    ) -> ToolResult:
        if not task.strip():
            return ToolResult(False, "Parameter 'task' tidak boleh kosong.")
        try:
            system = self._SYSTEMS.get(focus, self._SYSTEMS["full"])
            ctx_part = f"\n\nContext / Codebase:\n{codebase_context}" if codebase_context else ""
            user_msg = f"Task/Problem:\n{task}{ctx_part}"

            result = await self._llm(system, user_msg, max_tokens=4000, temperature=0.2)
            return ToolResult(
                True, result,
                {"tool": "superpowers_skills", "focus": focus, "task_len": len(task)}
            )
        except Exception as e:
            return ToolResult(False, f"superpowers_skills gagal: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. MEM SEARCH — Query Project History / Persistent Memory
# ══════════════════════════════════════════════════════════════════════════════

class MemSearchTool(BaseTool):
    """
    Search agent's persistent memory with progressive disclosure.
    Returns layered results: core → extended → background, each tagged with
    token cost and obs_id for citation.
    """

    name = "mem_search"
    description = (
        "Search agent persistent memory for past observations, project history, "
        "and stored context. Supports progressive disclosure: core (highly relevant), "
        "extended (related), background (loose match). Each result includes obs_id "
        "for citation and estimated token cost. "
        "Use to recall what was done before, find relevant past context, or verify facts."
    )
    parameters = (
        "query: str (what to search for), "
        "top_k: int = 10 (max results), "
        "tiers: str = 'core,extended,background' (comma-separated tiers to include), "
        "token_budget: int = 2000 (max tokens to return)"
    )

    def __init__(self, memory_backend):
        self._mem = memory_backend

    async def run(
        self,
        query: str,
        top_k: int = 10,
        tiers: str = "core,extended,background",
        token_budget: int = 2000,
        **_,
    ) -> "ToolResult":
        if not query.strip():
            return ToolResult(False, "Parameter 'query' tidak boleh kosong.")
        try:
            from .memory_features import ProgressiveDisclosure
            tier_list = [t.strip() for t in tiers.split(",") if t.strip()]
            pd = ProgressiveDisclosure(self._mem, token_budget=token_budget)
            ctx, cost = await pd.retrieve(query, top_k=top_k, include_tiers=tier_list)

            if not ctx:
                return ToolResult(True, "No relevant memories found.", {"query": query, "cost": cost})

            summary = (
                f"📊 Token cost: {cost['total_tokens']}/{cost['budget']} "
                f"({cost['utilization']} budget used) | "
                f"{len(cost['layers'])} layers retrieved\n\n"
            )
            layer_list = "\n".join(
                f"  [{l['tier']}] key={l['key']} score={l['score']} "
                f"tokens={l['tokens']} id={l['obs_id']}"
                for l in cost["layers"]
            )
            output = summary + layer_list + "\n\n---\n\n" + ctx
            return ToolResult(True, output, {"query": query, "cost": cost})
        except Exception as e:
            return ToolResult(False, f"mem_search gagal: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 9. MEM STORE — Manually Store Observation to Persistent Memory
# ══════════════════════════════════════════════════════════════════════════════

class MemStoreTool(BaseTool):
    """
    Manually store a key/value observation to persistent memory.
    Strips <private>...</private> blocks automatically before storage.
    Publishes event to memory viewer SSE stream.
    """

    name = "mem_store"
    description = (
        "Store an observation or piece of information into persistent memory. "
        "Content inside <private>...</private> tags is automatically redacted before storage. "
        "Returns an obs_id that can be used as a citation reference. "
        "Use to save important findings, task checkpoints, or project-specific context "
        "that should survive across sessions."
    )
    parameters = (
        "key: str (unique identifier, e.g. 'task:plan', 'finding:vuln_1'), "
        "content: str (content to store — wrap sensitive data in <private> tags), "
        "metadata: str = '{}' (JSON string of extra metadata tags)"
    )

    def __init__(self, memory_backend):
        self._mem = memory_backend

    async def run(
        self,
        key: str,
        content: str,
        metadata: str = "{}",
        **_,
    ) -> "ToolResult":
        if not key.strip() or not content.strip():
            return ToolResult(False, "Parameter 'key' dan 'content' wajib diisi.")
        try:
            meta = json.loads(metadata) if metadata.strip() != "{}" else {}
        except Exception:
            meta = {}

        try:
            from .memory_features import bus, MemoryEvent
            from .memory import strip_private
            clean_content = strip_private(content)
            await self._mem.store(key, clean_content, meta)
            entry = await self._mem.retrieve(key)
            obs_id = entry.obs_id if entry else "unknown"

            # Publish to memory viewer
            event = MemoryEvent(
                event_type="store",
                key=key,
                obs_id=obs_id,
                content_preview=clean_content[:200],
                metadata=meta,
                token_cost=entry.token_cost if entry else 0,
            )
            await bus.publish(event)

            return ToolResult(
                True,
                f"✓ Stored → key='{key}' | obs_id={obs_id} | "
                f"~{entry.token_cost if entry else 0} tokens",
                {"key": key, "obs_id": obs_id},
            )
        except Exception as e:
            return ToolResult(False, f"mem_store gagal: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 10. MEM STATS — Memory Statistics with Token Cost Visibility
# ══════════════════════════════════════════════════════════════════════════════

class MemStatsTool(BaseTool):
    """
    Show memory statistics: total observations, token cost breakdown,
    oldest/newest entries, and top keys by token usage.
    """

    name = "mem_stats"
    description = (
        "Display persistent memory statistics: observation count, total token cost, "
        "token budget utilization, oldest/newest entries, and heaviest keys. "
        "Use to audit memory health, identify bloat, or verify storage is working. "
        "Also shows the memory viewer URL (http://localhost:37777)."
    )
    parameters = "top_n: int = 10 (show top N keys by token cost)"

    def __init__(self, memory_backend):
        self._mem = memory_backend

    async def run(self, top_n: int = 10, **_) -> "ToolResult":
        try:
            if hasattr(self._mem, "all_entries"):
                entries = await self._mem.all_entries()
            else:
                keys = await self._mem.all_keys()
                entries = []
                for k in keys:
                    e = await self._mem.retrieve(k)
                    if e:
                        entries.append(e)

            if not entries:
                return ToolResult(True, "Memory is empty. No observations stored yet.")

            import time as _t
            total_tokens = sum(getattr(e, "token_cost", 0) for e in entries)
            oldest = min(entries, key=lambda e: e.timestamp)
            newest = max(entries, key=lambda e: e.timestamp)
            top = sorted(entries, key=lambda e: getattr(e, "token_cost", 0), reverse=True)[:top_n]

            lines = [
                "📊 MEMORY STATS",
                f"  Observations : {len(entries)}",
                f"  Total tokens : ~{total_tokens:,}",
                f"  Oldest entry : {_t.strftime('%Y-%m-%d %H:%M', _t.localtime(oldest.timestamp))} — {oldest.key}",
                f"  Newest entry : {_t.strftime('%Y-%m-%d %H:%M', _t.localtime(newest.timestamp))} — {newest.key}",
                f"  Viewer URL   : http://localhost:37777",
                "",
                f"🔝 TOP {top_n} BY TOKEN COST:",
            ]
            for e in top:
                tok = getattr(e, "token_cost", 0)
                oid = getattr(e, "obs_id", "?")[:8]
                lines.append(f"  {tok:>5} tok | {e.key:<40} | id:{oid}")

            return ToolResult(True, "\n".join(lines), {"total_tokens": total_tokens, "count": len(entries)})
        except Exception as e:
            return ToolResult(False, f"mem_stats gagal: {e}")




def register_skill_tools(registry, cfg: dict | None = None, memory_backend=None) -> None:
    """
    Register semua skill tools ke ToolRegistry yang sudah ada.

    USAGE di agent/core.py:
        from .skill_tools import register_skill_tools
        register_skill_tools(self._tools, self._cfg, memory_backend=self._memory)

    CONFIG (config.yaml):
        tools:
          skill_tools: true
          skill_tools_model: ""   # opsional override model
    """
    cfg = cfg or {}
    tool_cfg = cfg.get("tools", {})

    if not tool_cfg.get("skill_tools", True):
        return

    tools = [
        GrillMeTool(cfg),
        FrontendDesignTool(cfg),
        ThemeFactoryTool(cfg),
        BrowserUseTool(),          # tidak butuh LLM config
        TrailOfBitsSecurityTool(cfg),
        SkillCreatorTool(cfg),
        SuperpowersSkillsTool(cfg),
    ]

    # Memory skills — hanya didaftarkan jika memory backend tersedia
    if memory_backend is not None:
        tools += [
            MemSearchTool(memory_backend),
            MemStoreTool(memory_backend),
            MemStatsTool(memory_backend),
        ]

    for tool in tools:
        if hasattr(registry, "register"):
            registry.register(tool)
        elif hasattr(registry, "_register"):
            registry._register(tool)
        else:
            raise RuntimeError(
                f"ToolRegistry tidak punya method register() atau _register(). "
                f"Pastikan versi registry kompatibel."
            )

    tool_names = [t.name for t in tools]
    import logging
    logging.getLogger("gazcc.agent").info(
        f"[skill_tools] ✓ {len(tools)} tools registered: {', '.join(tool_names)}"
    )
