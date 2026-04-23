"""
agent/tools.py
Real tool implementations: file ops, web fetch, web search, code execution.
All async. All safe-guarded.
"""

import asyncio
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import uuid
import zipfile
import shutil

import aiofiles
import httpx
from bs4 import BeautifulSoup


# ── base ──────────────────────────────────────────────────────────────────────

class ToolResult:
    def __init__(self, success: bool, output: str, metadata: dict | None = None):
        self.success = success
        self.output = output
        self.metadata = metadata or {}

    def __str__(self) -> str:
        prefix = "✓" if self.success else "✗"
        return f"{prefix} {self.output}"


class BaseTool:
    name: str = ""
    description: str = ""
    parameters: str = ""

    async def run(self, *args, **kwargs) -> ToolResult:
        raise NotImplementedError


# ── file tools ────────────────────────────────────────────────────────────────

class ReadFileTool(BaseTool):
    name = "read_file"
    description = "Read content of a file. Returns file content as string."
    parameters = "path: str"

    async def run(self, path: str) -> ToolResult:
        try:
            p = Path(path)
            if not p.exists():
                return ToolResult(False, f"File not found: {path}")
            if p.stat().st_size > 2 * 1024 * 1024:
                return ToolResult(False, f"File too large (>2MB): {path}")
            async with aiofiles.open(p, "r", errors="replace") as f:
                content = await f.read()
            return ToolResult(True, content, {"path": str(p), "size": len(content)})
        except Exception as e:
            return ToolResult(False, f"read_file error: {e}")


class WriteFileTool(BaseTool):
    name = "write_file"
    description = (
        "Write content to a file. Creates parent dirs if needed. "
        "After writing, call export_file_base64 with the same path to generate a downloadable [FILE_EXPORT:...] marker for the user."
    )
    parameters = "path: str, content: str"

    async def run(self, path: str, content: str) -> ToolResult:
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(p, "w") as f:
                await f.write(content)
            return ToolResult(True, f"Written {len(content)} chars to {path}", {"path": str(p)})
        except Exception as e:
            return ToolResult(False, f"write_file error: {e}")


class AppendFileTool(BaseTool):
    name = "append_file"
    description = "Append content to an existing file (or create it)."
    parameters = "path: str, content: str"

    async def run(self, path: str, content: str) -> ToolResult:
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(p, "a") as f:
                await f.write(content)
            return ToolResult(True, f"Appended {len(content)} chars to {path}")
        except Exception as e:
            return ToolResult(False, f"append_file error: {e}")


class ListDirTool(BaseTool):
    name = "list_dir"
    description = "List files and directories at given path."
    parameters = "path: str"

    async def run(self, path: str = ".") -> ToolResult:
        try:
            p = Path(path)
            if not p.exists():
                return ToolResult(False, f"Path not found: {path}")
            items = []
            for item in sorted(p.iterdir()):
                kind = "DIR" if item.is_dir() else "FILE"
                size = item.stat().st_size if item.is_file() else 0
                items.append(f"[{kind}] {item.name}  ({size} bytes)")
            return ToolResult(True, "\n".join(items) if items else "(empty)")
        except Exception as e:
            return ToolResult(False, f"list_dir error: {e}")


class DeleteFileTool(BaseTool):
    name = "delete_file"
    description = "Delete a file."
    parameters = "path: str"

    async def run(self, path: str) -> ToolResult:
        try:
            p = Path(path)
            if not p.exists():
                return ToolResult(False, f"File not found: {path}")
            p.unlink()
            return ToolResult(True, f"Deleted: {path}")
        except Exception as e:
            return ToolResult(False, f"delete_file error: {e}")


# ── web tools ─────────────────────────────────────────────────────────────────

class FetchURLTool(BaseTool):
    name = "fetch_url"
    description = (
        "Fetch content from a URL. Returns cleaned text (HTML stripped). "
        "Auto-retry with backoff, rotates User-Agent, handles redirects, "
        "separate connect/read timeouts to prevent hanging."
    )
    parameters = "url: str, max_chars: int = 6000, retries: int = 3, timeout: int = 12"

    # Rotate UA biar gak keblok
    _USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    ]

    async def run(self, url: str, max_chars: int = 6000, retries: int = 3, timeout: int = 12) -> ToolResult:
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        last_err = ""
        for attempt in range(max(1, retries)):
            ua = self._USER_AGENTS[attempt % len(self._USER_AGENTS)]
            headers = {
                "User-Agent": ua,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "DNT": "1",
            }
            try:
                # Pisah connect timeout vs read timeout — kunci anti-hang
                timeout_cfg = httpx.Timeout(connect=6.0, read=timeout, write=5.0, pool=4.0)
                async with httpx.AsyncClient(
                    timeout=timeout_cfg,
                    follow_redirects=True,
                    headers=headers,
                    http2=True,
                ) as client:
                    resp = await client.get(url)
                    resp.raise_for_status()

                ct = resp.headers.get("content-type", "")

                # JSON — return raw
                if "json" in ct:
                    return ToolResult(True, resp.text[:max_chars], {"url": url, "content_type": ct, "attempt": attempt + 1})

                # HTML — strip noise
                soup = BeautifulSoup(resp.text, "lxml")
                for tag in soup(["script", "style", "nav", "footer", "header",
                                  "aside", "form", "noscript", "iframe", "svg"]):
                    tag.decompose()

                # Prioritas: article > main > body
                main = soup.find("article") or soup.find("main") or soup.find("body")
                raw_text = main.get_text(separator=" ") if main else soup.get_text(separator=" ")
                text = " ".join(raw_text.split())

                return ToolResult(True, text[:max_chars], {
                    "url": url,
                    "chars": len(text),
                    "attempt": attempt + 1,
                    "status": resp.status_code,
                })

            except httpx.HTTPStatusError as e:
                last_err = f"HTTP {e.response.status_code}"
                if e.response.status_code in (403, 404, 410, 451):
                    break  # Gak perlu retry kalau memang diblok/gak ada
                await asyncio.sleep(1.5 * (attempt + 1))

            except (httpx.ConnectTimeout, httpx.ReadTimeout) as e:
                last_err = f"Timeout (attempt {attempt + 1}): {type(e).__name__}"
                await asyncio.sleep(1.0 * (attempt + 1))

            except httpx.ConnectError as e:
                last_err = f"ConnectError: {e}"
                await asyncio.sleep(1.0 * (attempt + 1))

            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                await asyncio.sleep(0.5)

        return ToolResult(False, f"fetch_url gagal setelah {retries} attempt. Error terakhir: {last_err}", {"url": url})


class WebSearchTool(BaseTool):
    name = "web_search"
    description = (
        "Search the web. Auto-cascade melalui 4 engine: "
        "DDG API → DDG HTML → Brave → SearXNG. "
        "Retry otomatis per engine, concurrent fallback kalau semua lambat."
    )
    parameters = "query: str, max_results: int = 5, engine: str = 'auto'"

    # ── User-Agent pool ────────────────────────────────────────────────────────
    _UA = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 Version/17.4 Safari/605.1.15",
    ]

    # SearXNG public instances — fallback terakhir
    _SEARXNG_INSTANCES = [
        "https://searx.be",
        "https://search.inetol.net",
        "https://searxng.site",
        "https://searx.tiekoetter.com",
    ]

    def _headers(self, idx: int = 0) -> dict:
        return {
            "User-Agent": self._UA[idx % len(self._UA)],
            "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "DNT": "1",
        }

    def _timeout(self, connect: float = 6.0, read: float = 12.0) -> httpx.Timeout:
        return httpx.Timeout(connect=connect, read=read, write=5.0, pool=4.0)

    # ── Engine 1: DDG Instant Answer API ──────────────────────────────────────
    async def _ddg_api(self, query: str, max_results: int) -> list[dict] | None:
        try:
            params = {"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"}
            async with httpx.AsyncClient(timeout=self._timeout(5, 8), headers=self._headers(0)) as c:
                resp = await c.get("https://api.duckduckgo.com/", params=params)
                data = resp.json()

            results = []
            if data.get("Answer"):
                results.append({"title": "Instant Answer", "snippet": data["Answer"], "url": ""})
            if data.get("AbstractText"):
                results.append({"title": data.get("Heading", "Summary"), "snippet": data["AbstractText"], "url": data.get("AbstractURL", "")})

            for t in data.get("RelatedTopics", []):
                if len(results) >= max_results:
                    break
                if isinstance(t, dict) and t.get("Text"):
                    results.append({"title": t.get("Text", "")[:80], "snippet": t["Text"], "url": t.get("FirstURL", "")})
                elif isinstance(t, dict) and t.get("Topics"):
                    for sub in t["Topics"]:
                        if len(results) >= max_results:
                            break
                        if sub.get("Text"):
                            results.append({"title": sub["Text"][:80], "snippet": sub["Text"], "url": sub.get("FirstURL", "")})

            return results if results else None
        except Exception:
            return None

    # ── Engine 2: DDG HTML scrape ─────────────────────────────────────────────
    async def _ddg_html(self, query: str, max_results: int) -> list[dict] | None:
        try:
            q = query.replace(" ", "+")
            url = f"https://html.duckduckgo.com/html/?q={q}"
            async with httpx.AsyncClient(timeout=self._timeout(6, 12), headers=self._headers(1), follow_redirects=True) as c:
                resp = await c.get(url)

            soup = BeautifulSoup(resp.text, "lxml")
            results = []
            for r in soup.select(".result")[:max_results + 2]:
                title_el = r.select_one(".result__title")
                snip_el  = r.select_one(".result__snippet")
                url_el   = r.select_one(".result__url")
                href_el  = r.select_one("a.result__a")
                title  = title_el.get_text(strip=True) if title_el else ""
                snip   = snip_el.get_text(strip=True) if snip_el else ""
                link   = ""
                if href_el and href_el.get("href"):
                    raw = href_el["href"]
                    # DDG wraps link di redirect — extract uddg param
                    if "uddg=" in raw:
                        import urllib.parse
                        qs = urllib.parse.parse_qs(urllib.parse.urlparse(raw).query)
                        link = qs.get("uddg", [""])[0]
                    else:
                        link = raw
                elif url_el:
                    link = url_el.get_text(strip=True)
                if title:
                    results.append({"title": title, "snippet": snip, "url": link})
                if len(results) >= max_results:
                    break

            return results if results else None
        except Exception:
            return None

    # ── Engine 3: Brave Search scrape ─────────────────────────────────────────
    async def _brave_html(self, query: str, max_results: int) -> list[dict] | None:
        try:
            import urllib.parse
            q = urllib.parse.quote_plus(query)
            url = f"https://search.brave.com/search?q={q}&source=web"
            headers = self._headers(2)
            headers["Sec-Fetch-Site"] = "none"
            headers["Sec-Fetch-Mode"] = "navigate"

            async with httpx.AsyncClient(timeout=self._timeout(6, 14), headers=headers, follow_redirects=True) as c:
                resp = await c.get(url)

            soup = BeautifulSoup(resp.text, "lxml")
            results = []

            # Brave result cards
            for card in soup.select(".snippet, [data-type='web']")[:max_results + 3]:
                title_el = card.select_one(".title, h3, .snippet-title")
                snip_el  = card.select_one(".snippet-description, p, .description")
                url_el   = card.select_one("cite, .netloc, .url")
                href_el  = card.select_one("a[href]")

                title = title_el.get_text(strip=True) if title_el else ""
                snip  = snip_el.get_text(strip=True) if snip_el else ""
                link  = href_el["href"] if href_el and href_el.get("href") else (url_el.get_text(strip=True) if url_el else "")

                if title and len(title) > 3:
                    results.append({"title": title, "snippet": snip, "url": link})
                if len(results) >= max_results:
                    break

            return results if results else None
        except Exception:
            return None

    # ── Engine 4: SearXNG public instances ────────────────────────────────────
    async def _searxng(self, query: str, max_results: int) -> list[dict] | None:
        import urllib.parse
        q = urllib.parse.quote_plus(query)

        for instance in self._SEARXNG_INSTANCES:
            try:
                url = f"{instance}/search?q={q}&format=json&language=en"
                async with httpx.AsyncClient(timeout=self._timeout(5, 10), headers=self._headers(0)) as c:
                    resp = await c.get(url)
                    data = resp.json()

                results_raw = data.get("results", [])
                if not results_raw:
                    continue

                results = []
                for r in results_raw[:max_results]:
                    results.append({
                        "title":   r.get("title", ""),
                        "snippet": r.get("content", ""),
                        "url":     r.get("url", ""),
                    })
                if results:
                    return results
            except Exception:
                continue

        return None

    # ── Format output ─────────────────────────────────────────────────────────
    @staticmethod
    def _format(results: list[dict], engine: str, query: str) -> ToolResult:
        lines = [f"🔍 Query: \"{query}\" — via {engine}\n"]
        for i, r in enumerate(results, 1):
            title   = r.get("title", "").strip()
            snippet = r.get("snippet", "").strip()
            url     = r.get("url", "").strip()
            lines.append(f"{i}. {title}")
            if snippet:
                lines.append(f"   {snippet[:200]}")
            if url:
                lines.append(f"   → {url}")
        return ToolResult(True, "\n".join(lines), {"engine": engine, "query": query, "count": len(results)})

    # ── Main run — cascade fallback ───────────────────────────────────────────
    async def run(self, query: str, max_results: int = 5, engine: str = "auto") -> ToolResult:
        engine = engine.lower().strip()

        # Manual engine select
        if engine == "ddg_api":
            r = await self._ddg_api(query, max_results)
            return self._format(r, "DDG API", query) if r else ToolResult(False, "DDG API returned no results.")
        if engine == "ddg_html":
            r = await self._ddg_html(query, max_results)
            return self._format(r, "DDG HTML", query) if r else ToolResult(False, "DDG HTML scrape failed.")
        if engine == "brave":
            r = await self._brave_html(query, max_results)
            return self._format(r, "Brave", query) if r else ToolResult(False, "Brave search failed.")
        if engine == "searxng":
            r = await self._searxng(query, max_results)
            return self._format(r, "SearXNG", query) if r else ToolResult(False, "All SearXNG instances failed.")

        # AUTO — cascade semua engine, pakai yang pertama berhasil
        engines = [
            ("DDG API",  self._ddg_api),
            ("DDG HTML", self._ddg_html),
            ("Brave",    self._brave_html),
            ("SearXNG",  self._searxng),
        ]

        errors = []
        for eng_name, eng_fn in engines:
            try:
                result = await eng_fn(query, max_results)
                if result:
                    return self._format(result, eng_name, query)
                errors.append(f"{eng_name}: no results")
            except Exception as e:
                errors.append(f"{eng_name}: {type(e).__name__}")
                continue

        return ToolResult(
            False,
            f"Semua search engine gagal untuk query '{query}'.\n" + "\n".join(errors),
            {"query": query, "engines_tried": [e[0] for e in engines]},
        )


# ── code execution ────────────────────────────────────────────────────────────

class ExecuteCodeTool(BaseTool):
    name = "execute_code"
    description = (
        "Execute Python code in a sandboxed subprocess. "
        "Returns stdout/stderr. Timeout enforced."
    )
    parameters = "code: str, timeout: int = 15"

    # Only allow when explicitly enabled in config
    ENABLED: bool = True
    TIMEOUT: int = 15

    async def run(self, code: str, timeout: int = None) -> ToolResult:
        if not self.ENABLED:
            return ToolResult(False, "Code execution disabled in this environment.")
        timeout = timeout if timeout is not None else self.TIMEOUT
        try:
            with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as tmp:
                tmp.write(code)
                tmp_path = tmp.name

            proc = await asyncio.create_subprocess_exec(
                "python3", tmp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                return ToolResult(False, f"Code execution timed out after {timeout}s")
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

            out = stdout.decode(errors="replace").strip()
            err = stderr.decode(errors="replace").strip()

            if proc.returncode == 0:
                return ToolResult(True, out or "(no output)", {"returncode": 0})
            else:
                combined = "\n".join(filter(None, [out, err]))
                return ToolResult(False, f"Exit {proc.returncode}:\n{combined}", {"returncode": proc.returncode})
        except Exception as e:
            return ToolResult(False, f"execute_code error: {e}")


# ── utility tools ─────────────────────────────────────────────────────────────

class GetTimeTool(BaseTool):
    name = "get_time"
    description = "Returns the current UTC date and time."
    parameters = "(none)"

    async def run(self) -> ToolResult:
        import datetime
        now = datetime.datetime.utcnow()
        return ToolResult(True, now.strftime("UTC %Y-%m-%d %H:%M:%S"))


class CalculateTool(BaseTool):
    name = "calculate"
    description = "Evaluate a safe math expression. E.g. '2 ** 32', 'sqrt(144)'."
    parameters = "expression: str"

    async def run(self, expression: str) -> ToolResult:
        import math as _math
        safe_globals = {k: getattr(_math, k) for k in dir(_math) if not k.startswith("_")}
        safe_globals["__builtins__"] = {}
        expr = expression.replace("^", "**")
        try:
            result = eval(expr, {"__builtins__": {}}, safe_globals)  # noqa: S307
            return ToolResult(True, str(result))
        except Exception as e:
            return ToolResult(False, f"calculate error: {e}")


class SummarizeTextTool(BaseTool):
    name = "summarize_text"
    description = "Truncate and summarize long text to first N chars with line count."
    parameters = "text: str, max_chars: int = 2000"

    async def run(self, text: str, max_chars: int = 2000) -> ToolResult:
        lines = text.split("\n")
        truncated = text[:max_chars]
        if len(text) > max_chars:
            truncated += f"\n... [{len(text) - max_chars} more chars, {len(lines)} total lines]"
        return ToolResult(True, truncated)




# ── file export tools ─────────────────────────────────────────────────────────
# Jembatan antara sandbox Agent dan User Interface.
# Agent bisa "kirim" file ke user lewat Base64 atau Download Link.

import base64 as _base64
import mimetypes as _mimetypes


class ExportFileBase64Tool(BaseTool):
    """
    Option 1 — The Base64 Bridge.
    Baca file dari sandbox, encode ke Base64, kembalikan sebagai JSON.
    Frontend akan render tombol download otomatis dari data ini.

    Agent harus panggil tool ini dan tulis output-nya di Final Answer
    dengan format:  [FILE_EXPORT:filename:base64data:mimetype]
    Frontend mendeteksi marker ini dan membuat tombol download.
    """
    name = "export_file_base64"
    description = (
        "Export a file from the sandbox so the user can download it. "
        "Reads the file, encodes it as Base64, and returns a download marker "
        "that the UI renders as a clickable download button. "
        "Use this after creating any file the user should receive."
    )
    parameters = "path: str"

    async def run(self, path: str) -> ToolResult:
        try:
            p = Path(path)
            if not p.exists():
                return ToolResult(False, f"File not found: {path}")

            size = p.stat().st_size
            if size > 10 * 1024 * 1024:  # 10 MB limit
                return ToolResult(False, f"File too large to export via Base64 (>{size // 1024 // 1024}MB). Use export_file_link instead.")

            # Detect MIME type
            mime, _ = _mimetypes.guess_type(str(p))
            if not mime:
                mime = "application/octet-stream"

            # Read and encode
            with open(p, "rb") as f:
                raw = f.read()
            b64 = _base64.b64encode(raw).decode("utf-8")

            filename = p.name
            # Format: marker yang dikenali frontend untuk render tombol download
            marker = f"[FILE_EXPORT:{filename}:{b64}:{mime}]"

            return ToolResult(
                True,
                marker,
                {
                    "filename": filename,
                    "size_bytes": size,
                    "mime_type": mime,
                    "encoding": "base64",
                    "instruction": (
                        f"File '{filename}' siap didownload. "
                        f"Sertakan marker ini persis di Final Answer kamu: {marker}"
                    ),
                },
            )
        except Exception as e:
            return ToolResult(False, f"export_file_base64 error: {e}")


class ExportFileLinkTool(BaseTool):
    """
    Option 2 — The Download Link Bridge.
    Copy file ke /tmp/gazcc_exports/, simpan ke Redis (jika tersedia),
    dan kembalikan URL download yang bisa diklik user.

    Jika Redis tidak tersedia, otomatis fallback ke Base64 bridge.
    URL format: /api/download/{token}
    """
    name = "export_file_link"
    description = (
        "Generate a download link for a file so the user can download it directly. "
        "Returns a URL the user can click. Falls back to Base64 if storage unavailable. "
        "Prefer this over export_file_base64 for large files or better UX."
    )
    parameters = "path: str"

    EXPORT_DIR = Path("/tmp/gazcc_exports")

    async def run(self, path: str) -> ToolResult:
        try:
            p = Path(path)
            if not p.exists():
                return ToolResult(False, f"File not found: {path}")

            size = p.stat().st_size
            mime, _ = _mimetypes.guess_type(str(p))
            if not mime:
                mime = "application/octet-stream"

            filename = p.name
            token = str(uuid.uuid4())[:12]  # short unique token

            # ── Coba simpan ke Redis ──────────────────────────────────────────
            redis_url   = os.environ.get("UPSTASH_REDIS_REST_URL", "")
            redis_token = os.environ.get("UPSTASH_REDIS_REST_TOKEN", "")

            if redis_url and redis_token:
                with open(p, "rb") as f:
                    raw = f.read()
                b64 = _base64.b64encode(raw).decode("utf-8")

                # Simpan ke Redis dengan TTL 1 jam
                redis_key = f"gazcc_export:{token}"
                payload = json.dumps({"filename": filename, "mime": mime, "data": b64})

                async with httpx.AsyncClient(timeout=10) as client:
                    r = await client.post(
                        f"{redis_url}/set/{redis_key}",
                        headers={"Authorization": f"Bearer {redis_token}"},
                        json=[payload, "EX", 3600],
                    )
                    r.raise_for_status()

                # Ambil base URL dari env atau gunakan relative path
                base_url = os.environ.get("VERCEL_URL", "")
                if base_url and not base_url.startswith("http"):
                    base_url = f"https://{base_url}"
                download_url = f"{base_url}/api/download/{token}" if base_url else f"/api/download/{token}"

                marker = f"[FILE_LINK:{filename}:{download_url}]"
                return ToolResult(
                    True,
                    marker,
                    {
                        "filename": filename,
                        "download_url": download_url,
                        "token": token,
                        "expires_in": "1 hour",
                        "size_bytes": size,
                        "instruction": (
                            f"File '{filename}' siap didownload via link. "
                            f"Sertakan marker ini di Final Answer: {marker}"
                        ),
                    },
                )

            # ── Fallback ke Base64 jika Redis tidak ada ───────────────────────
            with open(p, "rb") as f:
                raw = f.read()
            b64 = _base64.b64encode(raw).decode("utf-8")
            marker = f"[FILE_EXPORT:{filename}:{b64}:{mime}]"
            return ToolResult(
                True,
                marker,
                {
                    "filename": filename,
                    "fallback": "base64 (Redis not configured)",
                    "instruction": (
                        f"Redis tidak tersedia, fallback ke Base64. "
                        f"Sertakan marker ini di Final Answer: {marker}"
                    ),
                },
            )

        except Exception as e:
            return ToolResult(False, f"export_file_link error: {e}")


# ── zip management tool ───────────────────────────────────────────────────────

class ZipManageTool(BaseTool):
    name = "manage_zip"
    description = (
        "ZIP file management: list contents, extract, or create new ZIP. "
        "action='list': show ZIP contents | action='extract': unzip files | "
        "action='create': compress files into new ZIP."
    )
    parameters = "action: str, filename: str, files: list = None, internal_file: str = None, dest_dir: str = None"

    async def run(self, action: str, filename: str, files: list = None,
                  internal_file: str = None, dest_dir: str = None) -> ToolResult:
        import json as _json
        action = action.lower().strip()
        p = Path(filename)

        if action == "list":
            if not p.exists():
                return ToolResult(False, f"ZIP tidak ditemukan: {filename}")
            if not zipfile.is_zipfile(p):
                return ToolResult(False, f"Bukan file ZIP valid: {filename}")
            try:
                entries = []
                with zipfile.ZipFile(p, "r") as zf:
                    for info in zf.infolist():
                        entries.append({"name": info.filename, "size_bytes": info.file_size,
                                        "compressed_bytes": info.compress_size, "is_dir": info.filename.endswith("/")})
                return ToolResult(True, _json.dumps({"zip_file": str(p), "total": len(entries), "entries": entries}, indent=2))
            except Exception as e:
                return ToolResult(False, f"manage_zip(list) error: {e}")

        elif action == "extract":
            if not p.exists():
                return ToolResult(False, f"ZIP tidak ditemukan: {filename}")
            if not zipfile.is_zipfile(p):
                return ToolResult(False, f"Bukan file ZIP valid: {filename}")
            out_dir = Path(dest_dir) if dest_dir else p.parent / p.stem
            out_dir.mkdir(parents=True, exist_ok=True)
            try:
                extracted = []
                with zipfile.ZipFile(p, "r") as zf:
                    if internal_file:
                        zf.extract(internal_file, path=out_dir)
                        extracted.append(internal_file)
                    else:
                        zf.extractall(path=out_dir)
                        extracted = [m.filename for m in zf.infolist() if not m.filename.endswith("/")]
                return ToolResult(True, _json.dumps({"extracted_to": str(out_dir), "files": extracted}, indent=2))
            except KeyError:
                return ToolResult(False, f"File '{internal_file}' tidak ada di dalam ZIP.")
            except Exception as e:
                return ToolResult(False, f"manage_zip(extract) error: {e}")

        elif action == "create":
            if not files:
                return ToolResult(False, "Parameter 'files' wajib diisi untuk action='create'.")
            out_zip = Path(filename)
            out_zip.parent.mkdir(parents=True, exist_ok=True)
            added = []
            try:
                with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for f in files:
                        fp = Path(f)
                        if not fp.exists():
                            return ToolResult(False, f"File tidak ditemukan: {f}")
                        zf.write(fp, arcname=fp.name)
                        added.append(fp.name)
                return ToolResult(True, _json.dumps({"zip_created": str(out_zip), "files_added": added,
                                                     "zip_size_bytes": out_zip.stat().st_size}, indent=2))
            except Exception as e:
                return ToolResult(False, f"manage_zip(create) error: {e}")
        else:
            return ToolResult(False, f"Action tidak valid: '{action}'. Gunakan: list, extract, create.")


# ── diff / patch tool ─────────────────────────────────────────────────────────

import difflib as _difflib


class DiffPatchTool(BaseTool):
    """
    Two operations in one tool:
      action='diff'  — generate a unified diff between two texts or file paths.
      action='patch' — apply a unified diff string to a target file, write result back.
    """
    name = "diff_patch"
    description = (
        "Generate a unified diff between two texts/files (action='diff'), "
        "or apply a unified diff patch to a file (action='patch'). "
        "Use 'diff' to compare original vs modified content before committing changes. "
        "Use 'patch' to apply surgical edits without rewriting the entire file."
    )
    parameters = (
        "action: str ('diff'|'patch'), "
        "original: str (text or file path), "
        "modified: str (text or file path, for diff) | patch_str: str (unified diff, for patch), "
        "output_path: str | None"
    )

    # ── helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _load(src: str) -> tuple[str, str]:
        """Return (content, label). If src looks like an existing path, read it; else treat as raw text."""
        p = Path(src)
        if len(src) < 512 and p.exists():
            try:
                return p.read_text(errors="replace"), str(p)
            except Exception:
                pass
        return src, "<text>"

    # ── run ────────────────────────────────────────────────────────────────────

    async def run(
        self,
        action: str,
        original: str = "",
        modified: str = "",
        patch_str: str = "",
        output_path: str | None = None,
    ) -> ToolResult:

        action = action.lower().strip()

        # ── DIFF ──────────────────────────────────────────────────────────────
        if action == "diff":
            if not original or not modified:
                return ToolResult(False, "diff requires both 'original' and 'modified'.")
            orig_text, orig_label = self._load(original)
            mod_text, mod_label = self._load(modified)

            diff_lines = list(
                _difflib.unified_diff(
                    orig_text.splitlines(keepends=True),
                    mod_text.splitlines(keepends=True),
                    fromfile=orig_label,
                    tofile=mod_label,
                    lineterm="",
                )
            )
            if not diff_lines:
                return ToolResult(True, "(no differences)", {"changed": False})

            diff_str = "\n".join(diff_lines)

            if output_path:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                Path(output_path).write_text(diff_str)

            return ToolResult(
                True,
                diff_str,
                {
                    "changed": True,
                    "lines_changed": len([l for l in diff_lines if l.startswith(("+", "-")) and not l.startswith(("+++", "---"))]),
                    "saved_to": output_path,
                },
            )

        # ── PATCH ─────────────────────────────────────────────────────────────
        elif action == "patch":
            if not original:
                return ToolResult(False, "patch requires 'original' (file path or text).")
            if not patch_str:
                return ToolResult(False, "patch requires 'patch_str' (unified diff string).")

            orig_text, orig_label = self._load(original)
            orig_lines = orig_text.splitlines(keepends=True)

            try:
                patched_lines = list(
                    _difflib._patch_files(  # type: ignore[attr-defined]
                        orig_lines, patch_str.splitlines(keepends=True)
                    )
                )
            except AttributeError:
                # _patch_files is private; manual apply fallback
                patched_lines = self._manual_apply(orig_lines, patch_str)

            patched_text = "".join(patched_lines)

            # Write back
            target = output_path or (original if Path(original).exists() else None)
            if target:
                p = Path(target)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(patched_text)
                return ToolResult(True, f"Patch applied and written to {target}", {"output": target, "size": len(patched_text)})
            else:
                return ToolResult(True, patched_text, {"note": "No output_path — returning patched content only"})

        else:
            return ToolResult(False, f"Unknown action '{action}'. Use: diff | patch")

    # ── manual patch fallback (pure stdlib) ───────────────────────────────────
    @staticmethod
    def _manual_apply(orig_lines: list[str], patch_str: str) -> list[str]:
        """Best-effort line-level patch application from a unified diff string."""
        result = list(orig_lines)
        patch_lines = patch_str.splitlines(keepends=True)
        offset = 0
        i = 0
        while i < len(patch_lines):
            line = patch_lines[i]
            if line.startswith("@@"):
                import re
                m = re.search(r"-(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))?", line)
                if not m:
                    i += 1
                    continue
                orig_start = int(m.group(1)) - 1
                i += 1
                pos = orig_start + offset
                removes, adds = [], []
                while i < len(patch_lines) and not patch_lines[i].startswith("@@"):
                    pl = patch_lines[i]
                    if pl.startswith("-"):
                        removes.append(pl[1:])
                    elif pl.startswith("+"):
                        adds.append(pl[1:])
                    i += 1
                # Replace removes with adds at pos
                del result[pos:pos + len(removes)]
                result[pos:pos] = adds
                offset += len(adds) - len(removes)
            else:
                i += 1
        return result


# ══════════════════════════════════════════════════════════════════════════════
# ── BATCH 2 — 20 NEW TOOLS ────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

import difflib as _difflib
import glob as _glob
import gzip as _gzip
import hashlib as _hashlib
import random as _random
import shutil as _shutil
import string as _string
import xml.etree.ElementTree as _ET
import zlib as _zlib
from base64 import b64decode as _b64decode, b64encode as _b64encode
from csv import DictReader as _DictReader, DictWriter as _DictWriter
from io import StringIO as _StringIO


# ── 1. shell_exec ─────────────────────────────────────────────────────────────

class ShellExecTool(BaseTool):
    name = "shell_exec"
    description = (
        "Run a shell (bash) command and return stdout + stderr. "
        "Timeout enforced. Use for git ops, package installs, build commands, "
        "file manipulation — anything Python subprocess can do."
    )
    parameters = "command: str, timeout: int = 20, cwd: str = None"

    BLOCKED = ("rm -rf /", "mkfs", ":(){:|:&};:", "dd if=/dev/zero")

    async def run(self, command: str, timeout: int = 20, cwd: str = None) -> ToolResult:
        for blocked in self.BLOCKED:
            if blocked in command:
                return ToolResult(False, f"Blocked dangerous command: {blocked}")
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd or None,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                return ToolResult(False, f"shell_exec timed out after {timeout}s")
            out = stdout.decode(errors="replace").strip()
            err = stderr.decode(errors="replace").strip()
            combined = "\n".join(filter(None, [out, err]))
            ok = proc.returncode == 0
            return ToolResult(ok, combined or "(no output)", {"returncode": proc.returncode, "command": command})
        except Exception as e:
            return ToolResult(False, f"shell_exec error: {e}")


# ── 2. grep_search ────────────────────────────────────────────────────────────

class GrepSearchTool(BaseTool):
    name = "grep_search"
    description = (
        "Search for a regex pattern across files in a directory (recursive). "
        "Returns matching lines with file path and line number. "
        "Use to locate functions, variables, error strings, or any pattern in a codebase."
    )
    parameters = "pattern: str, path: str = '.', file_glob: str = '*', max_matches: int = 50"

    async def run(self, pattern: str, path: str = ".", file_glob: str = "*", max_matches: int = 50) -> ToolResult:
        import re
        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return ToolResult(False, f"Invalid regex: {e}")
        results = []
        base = Path(path)
        if not base.exists():
            return ToolResult(False, f"Path not found: {path}")
        for fp in sorted(base.rglob(file_glob)):
            if not fp.is_file():
                continue
            try:
                lines = fp.read_text(errors="replace").splitlines()
                for i, line in enumerate(lines, 1):
                    if rx.search(line):
                        results.append(f"{fp}:{i}: {line.strip()}")
                        if len(results) >= max_matches:
                            results.append(f"... (limit {max_matches} reached)")
                            break
            except Exception:
                continue
            if len(results) >= max_matches:
                break
        if not results:
            return ToolResult(False, f"No matches for '{pattern}' in {path}")
        return ToolResult(True, "\n".join(results), {"matches": len(results), "pattern": pattern})


# ── 3. find_files ─────────────────────────────────────────────────────────────

class FindFilesTool(BaseTool):
    name = "find_files"
    description = (
        "Find files matching a glob pattern recursively in a directory. "
        "Returns paths with size and modification time. "
        "Use to locate specific files, list by extension, or explore project structure."
    )
    parameters = "pattern: str, base_path: str = '.', max_results: int = 100"

    async def run(self, pattern: str, base_path: str = ".", max_results: int = 100) -> ToolResult:
        import datetime
        base = Path(base_path)
        if not base.exists():
            return ToolResult(False, f"Path not found: {base_path}")
        try:
            matches = sorted(base.rglob(pattern))[:max_results]
            if not matches:
                return ToolResult(False, f"No files matching '{pattern}' in {base_path}")
            lines = []
            for p in matches:
                stat = p.stat()
                mt = datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
                size = stat.st_size
                lines.append(f"{p}  ({size}B, {mt})")
            return ToolResult(True, "\n".join(lines), {"count": len(matches), "pattern": pattern})
        except Exception as e:
            return ToolResult(False, f"find_files error: {e}")


# ── 4. copy_file ──────────────────────────────────────────────────────────────

class CopyFileTool(BaseTool):
    name = "copy_file"
    description = (
        "Copy a file or directory to a destination path. "
        "Creates parent directories as needed. "
        "Use to duplicate files, create backups before editing, or stage output."
    )
    parameters = "src: str, dst: str, overwrite: bool = True"

    async def run(self, src: str, dst: str, overwrite: bool = True) -> ToolResult:
        try:
            s = Path(src)
            d = Path(dst)
            if not s.exists():
                return ToolResult(False, f"Source not found: {src}")
            d.parent.mkdir(parents=True, exist_ok=True)
            if d.exists() and not overwrite:
                return ToolResult(False, f"Destination exists (overwrite=False): {dst}")
            if s.is_dir():
                _shutil.copytree(str(s), str(d), dirs_exist_ok=True)
            else:
                _shutil.copy2(str(s), str(d))
            size = d.stat().st_size if d.is_file() else sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            return ToolResult(True, f"Copied {src} → {dst}", {"src": src, "dst": dst, "size_bytes": size})
        except Exception as e:
            return ToolResult(False, f"copy_file error: {e}")


# ── 5. move_file ──────────────────────────────────────────────────────────────

class MoveFileTool(BaseTool):
    name = "move_file"
    description = (
        "Move or rename a file or directory. "
        "Creates parent directories at destination as needed. "
        "Use for renaming, reorganizing files, or staging output."
    )
    parameters = "src: str, dst: str"

    async def run(self, src: str, dst: str) -> ToolResult:
        try:
            s = Path(src)
            d = Path(dst)
            if not s.exists():
                return ToolResult(False, f"Source not found: {src}")
            d.parent.mkdir(parents=True, exist_ok=True)
            _shutil.move(str(s), str(d))
            return ToolResult(True, f"Moved {src} → {dst}", {"src": src, "dst": dst})
        except Exception as e:
            return ToolResult(False, f"move_file error: {e}")


# ── 6. file_stat ──────────────────────────────────────────────────────────────

class FileStatTool(BaseTool):
    name = "file_stat"
    description = (
        "Get detailed metadata of a file or directory: size, line count, "
        "modification time, permissions, extension. "
        "Use before reading large files or to audit project contents."
    )
    parameters = "path: str"

    async def run(self, path: str) -> ToolResult:
        import datetime, stat as _stat
        try:
            p = Path(path)
            if not p.exists():
                return ToolResult(False, f"Path not found: {path}")
            s = p.stat()
            info = {
                "path": str(p.resolve()),
                "type": "directory" if p.is_dir() else "file",
                "size_bytes": s.st_size,
                "modified": datetime.datetime.fromtimestamp(s.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "created": datetime.datetime.fromtimestamp(s.st_ctime).strftime("%Y-%m-%d %H:%M:%S"),
                "permissions": oct(s.st_mode & 0o777),
                "extension": p.suffix,
            }
            if p.is_file():
                try:
                    text = p.read_text(errors="replace")
                    info["line_count"] = text.count("\n") + 1
                    info["char_count"] = len(text)
                    info["word_count"] = len(text.split())
                except Exception:
                    info["note"] = "binary file"
            else:
                files = list(p.rglob("*"))
                info["file_count"] = sum(1 for f in files if f.is_file())
                info["dir_count"] = sum(1 for f in files if f.is_dir())
                info["total_size_bytes"] = sum(f.stat().st_size for f in files if f.is_file())
            return ToolResult(True, json.dumps(info, indent=2), info)
        except Exception as e:
            return ToolResult(False, f"file_stat error: {e}")


# ── 7. http_request ───────────────────────────────────────────────────────────

class HttpRequestTool(BaseTool):
    name = "http_request"
    description = (
        "Make a generic HTTP request (GET/POST/PUT/PATCH/DELETE). "
        "Supports custom headers, JSON body, and query params. "
        "Use to call APIs, webhooks, or any external HTTP endpoint."
    )
    parameters = "url: str, method: str = 'GET', headers: dict = None, body: dict = None, params: dict = None, timeout: int = 20"

    async def run(
        self,
        url: str,
        method: str = "GET",
        headers: dict = None,
        body: dict = None,
        params: dict = None,
        timeout: int = 20,
    ) -> ToolResult:
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        method = method.upper()
        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                resp = await client.request(
                    method=method,
                    url=url,
                    headers=headers or {},
                    json=body if body else None,
                    params=params or {},
                )
            ct = resp.headers.get("content-type", "")
            try:
                data = resp.json() if "json" in ct else resp.text
            except Exception:
                data = resp.text
            body_str = json.dumps(data, indent=2) if isinstance(data, (dict, list)) else str(data)
            ok = 200 <= resp.status_code < 300
            return ToolResult(ok, body_str[:8000], {
                "status_code": resp.status_code,
                "url": url,
                "method": method,
                "content_type": ct,
            })
        except Exception as e:
            return ToolResult(False, f"http_request error: {e}")


# ── 8. json_format ────────────────────────────────────────────────────────────

class JsonFormatTool(BaseTool):
    name = "json_format"
    description = (
        "Parse, validate, format (pretty-print), minify, or query JSON data. "
        "action='format': pretty-print | action='minify': compact | "
        "action='validate': check validity | action='query': JMESPath-style key access."
    )
    parameters = "action: str ('format'|'minify'|'validate'|'query'), data: str, key_path: str = None, indent: int = 2"

    async def run(self, action: str, data: str, key_path: str = None, indent: int = 2) -> ToolResult:
        # Load from file if path given
        if len(data) < 512 and Path(data).exists():
            data = Path(data).read_text(errors="replace")
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError as e:
            return ToolResult(False, f"Invalid JSON: {e}")

        action = action.lower().strip()

        if action == "format":
            return ToolResult(True, json.dumps(parsed, indent=indent, ensure_ascii=False))
        elif action == "minify":
            return ToolResult(True, json.dumps(parsed, separators=(",", ":"), ensure_ascii=False))
        elif action == "validate":
            return ToolResult(True, f"Valid JSON. Type: {type(parsed).__name__}, "
                              f"Keys: {list(parsed.keys()) if isinstance(parsed, dict) else len(parsed)} items")
        elif action == "query":
            if not key_path:
                return ToolResult(False, "query requires key_path (e.g. 'data.items.0.name')")
            node = parsed
            for part in key_path.split("."):
                try:
                    node = node[int(part)] if part.isdigit() else node[part]
                except (KeyError, IndexError, TypeError) as e:
                    return ToolResult(False, f"Key '{part}' not found in path '{key_path}': {e}")
            result = json.dumps(node, indent=2, ensure_ascii=False) if isinstance(node, (dict, list)) else str(node)
            return ToolResult(True, result)
        else:
            return ToolResult(False, f"Unknown action '{action}'. Use: format | minify | validate | query")


# ── 9. yaml_parse ─────────────────────────────────────────────────────────────

class YamlParseTool(BaseTool):
    name = "yaml_parse"
    description = (
        "Parse YAML content or file, convert to JSON, or validate structure. "
        "action='parse': parse YAML → JSON | action='validate': check YAML syntax | "
        "action='to_yaml': convert JSON string → YAML."
    )
    parameters = "action: str ('parse'|'validate'|'to_yaml'), data: str"

    async def run(self, action: str, data: str) -> ToolResult:
        try:
            import yaml  # type: ignore
        except ImportError:
            return ToolResult(False, "PyYAML not installed. Run: pip install pyyaml")

        if len(data) < 512 and Path(data).exists():
            data = Path(data).read_text(errors="replace")

        action = action.lower().strip()

        if action == "parse":
            try:
                parsed = yaml.safe_load(data)
                return ToolResult(True, json.dumps(parsed, indent=2, ensure_ascii=False, default=str),
                                  {"type": type(parsed).__name__})
            except yaml.YAMLError as e:
                return ToolResult(False, f"YAML parse error: {e}")
        elif action == "validate":
            try:
                yaml.safe_load(data)
                return ToolResult(True, "Valid YAML")
            except yaml.YAMLError as e:
                return ToolResult(False, f"Invalid YAML: {e}")
        elif action == "to_yaml":
            try:
                obj = json.loads(data)
                return ToolResult(True, yaml.dump(obj, default_flow_style=False, allow_unicode=True))
            except Exception as e:
                return ToolResult(False, f"to_yaml error: {e}")
        else:
            return ToolResult(False, f"Unknown action '{action}'. Use: parse | validate | to_yaml")


# ── 10. csv_query ─────────────────────────────────────────────────────────────

class CsvQueryTool(BaseTool):
    name = "csv_query"
    description = (
        "Read, filter, sort, and transform CSV data. "
        "action='read': load CSV | action='filter': filter rows by column=value | "
        "action='sort': sort by column | action='stats': basic stats per column."
    )
    parameters = "action: str ('read'|'filter'|'sort'|'stats'), path: str, column: str = None, value: str = None, ascending: bool = True, max_rows: int = 50"

    async def run(self, action: str, path: str, column: str = None, value: str = None,
                  ascending: bool = True, max_rows: int = 50) -> ToolResult:
        try:
            p = Path(path)
            if not p.exists():
                return ToolResult(False, f"CSV not found: {path}")
            text = p.read_text(errors="replace")
            reader = list(_DictReader(_StringIO(text)))
            if not reader:
                return ToolResult(False, "CSV is empty or has no rows")
            headers = list(reader[0].keys())

            action = action.lower().strip()

            if action == "read":
                rows = reader[:max_rows]
                lines = [",".join(headers)]
                for r in rows:
                    lines.append(",".join(str(r.get(h, "")) for h in headers))
                note = f"\n(showing {len(rows)}/{len(reader)} rows)" if len(reader) > max_rows else ""
                return ToolResult(True, "\n".join(lines) + note, {"rows": len(reader), "columns": headers})

            elif action == "filter":
                if not column or column not in headers:
                    return ToolResult(False, f"column '{column}' not in headers: {headers}")
                filtered = [r for r in reader if str(r.get(column, "")).lower() == str(value or "").lower()]
                lines = [",".join(headers)]
                for r in filtered[:max_rows]:
                    lines.append(",".join(str(r.get(h, "")) for h in headers))
                return ToolResult(True, "\n".join(lines), {"matches": len(filtered)})

            elif action == "sort":
                if not column or column not in headers:
                    return ToolResult(False, f"column '{column}' not in headers: {headers}")
                try:
                    sorted_rows = sorted(reader, key=lambda r: float(r.get(column, 0)) if r.get(column, "").replace(".", "").replace("-", "").isdigit() else r.get(column, ""), reverse=not ascending)
                except Exception:
                    sorted_rows = sorted(reader, key=lambda r: r.get(column, ""), reverse=not ascending)
                lines = [",".join(headers)]
                for r in sorted_rows[:max_rows]:
                    lines.append(",".join(str(r.get(h, "")) for h in headers))
                return ToolResult(True, "\n".join(lines), {"sorted_by": column, "ascending": ascending})

            elif action == "stats":
                stats = {}
                for h in headers:
                    vals = [r.get(h, "") for r in reader]
                    nums = []
                    for v in vals:
                        try:
                            nums.append(float(v))
                        except Exception:
                            pass
                    if nums:
                        stats[h] = {"count": len(vals), "numeric": len(nums),
                                    "min": min(nums), "max": max(nums),
                                    "avg": round(sum(nums) / len(nums), 4)}
                    else:
                        unique = len(set(vals))
                        stats[h] = {"count": len(vals), "unique": unique, "type": "string"}
                return ToolResult(True, json.dumps(stats, indent=2), {"columns": len(headers), "rows": len(reader)})

            else:
                return ToolResult(False, f"Unknown action '{action}'. Use: read | filter | sort | stats")
        except Exception as e:
            return ToolResult(False, f"csv_query error: {e}")


# ── 11. base64_codec ──────────────────────────────────────────────────────────

class Base64CodecTool(BaseTool):
    name = "base64_codec"
    description = (
        "Encode text/file to Base64, or decode a Base64 string back to text. "
        "action='encode': text or file → base64 | action='decode': base64 → text."
    )
    parameters = "action: str ('encode'|'decode'), data: str, is_file: bool = False"

    async def run(self, action: str, data: str, is_file: bool = False) -> ToolResult:
        action = action.lower().strip()
        try:
            if action == "encode":
                if is_file:
                    p = Path(data)
                    if not p.exists():
                        return ToolResult(False, f"File not found: {data}")
                    raw = p.read_bytes()
                else:
                    raw = data.encode("utf-8")
                encoded = _b64encode(raw).decode("ascii")
                return ToolResult(True, encoded, {"original_bytes": len(raw), "encoded_chars": len(encoded)})
            elif action == "decode":
                try:
                    decoded = _b64decode(data.strip())
                    try:
                        return ToolResult(True, decoded.decode("utf-8"), {"decoded_bytes": len(decoded)})
                    except UnicodeDecodeError:
                        return ToolResult(True, f"(binary data, {len(decoded)} bytes — save to file to use)",
                                          {"decoded_bytes": len(decoded), "is_binary": True})
                except Exception as e:
                    return ToolResult(False, f"Base64 decode error: {e}")
            else:
                return ToolResult(False, f"Unknown action '{action}'. Use: encode | decode")
        except Exception as e:
            return ToolResult(False, f"base64_codec error: {e}")


# ── 12. hash_generate ─────────────────────────────────────────────────────────

class HashGenerateTool(BaseTool):
    name = "hash_generate"
    description = (
        "Generate cryptographic hash of text or a file. "
        "Supports: md5, sha1, sha256, sha512. "
        "Use to verify file integrity, generate checksums, or fingerprint content."
    )
    parameters = "data: str, algorithm: str = 'sha256', is_file: bool = False"

    async def run(self, data: str, algorithm: str = "sha256", is_file: bool = False) -> ToolResult:
        alg = algorithm.lower().strip()
        if alg not in ("md5", "sha1", "sha256", "sha512"):
            return ToolResult(False, f"Unsupported algorithm '{alg}'. Use: md5 | sha1 | sha256 | sha512")
        try:
            h = _hashlib.new(alg)
            if is_file:
                p = Path(data)
                if not p.exists():
                    return ToolResult(False, f"File not found: {data}")
                h.update(p.read_bytes())
                label = str(p)
            else:
                h.update(data.encode("utf-8"))
                label = f"text({len(data)} chars)"
            digest = h.hexdigest()
            return ToolResult(True, digest, {"algorithm": alg, "source": label, "digest": digest})
        except Exception as e:
            return ToolResult(False, f"hash_generate error: {e}")


# ── 13. text_replace ──────────────────────────────────────────────────────────

class TextReplaceTool(BaseTool):
    name = "text_replace"
    description = (
        "Find and replace text in a file (string or regex). "
        "Writes result back to file or returns patched content. "
        "Use for bulk renaming, refactoring, or applying multiple edits at once."
    )
    parameters = "path: str, find: str, replace: str, use_regex: bool = False, count: int = 0, backup: bool = False"

    async def run(self, path: str, find: str, replace: str,
                  use_regex: bool = False, count: int = 0, backup: bool = False) -> ToolResult:
        import re
        try:
            p = Path(path)
            if not p.exists():
                return ToolResult(False, f"File not found: {path}")
            original = p.read_text(errors="replace")

            if backup:
                Path(str(p) + ".bak").write_text(original)

            if use_regex:
                try:
                    if count:
                        result, n = re.subn(find, replace, original, count=count)
                    else:
                        result, n = re.subn(find, replace, original)
                except re.error as e:
                    return ToolResult(False, f"Invalid regex: {e}")
            else:
                n = original.count(find) if count == 0 else min(original.count(find), count)
                result = original.replace(find, replace, count or -1)

            if result == original:
                return ToolResult(True, "No occurrences found — file unchanged.", {"replacements": 0})

            p.write_text(result)
            return ToolResult(True, f"Replaced {n} occurrence(s) in {path}",
                              {"replacements": n, "backup": str(p) + ".bak" if backup else None})
        except Exception as e:
            return ToolResult(False, f"text_replace error: {e}")


# ── 14. extract_urls ──────────────────────────────────────────────────────────

class ExtractUrlsTool(BaseTool):
    name = "extract_urls"
    description = (
        "Extract all URLs, emails, IPs, or domains from a text string or file. "
        "type='urls': http/https links | type='emails': email addresses | "
        "type='ips': IPv4 addresses | type='all': everything."
    )
    parameters = "data: str, extract_type: str = 'all', is_file: bool = False"

    async def run(self, data: str, extract_type: str = "all", is_file: bool = False) -> ToolResult:
        import re
        if is_file:
            p = Path(data)
            if not p.exists():
                return ToolResult(False, f"File not found: {data}")
            data = p.read_text(errors="replace")

        patterns = {
            "urls": r"https?://[^\s\"'<>)\]]+",
            "emails": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
            "ips": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        }
        et = extract_type.lower().strip()
        results = {}
        targets = list(patterns.keys()) if et == "all" else [et]

        for t in targets:
            if t not in patterns:
                return ToolResult(False, f"Unknown type '{t}'. Use: urls | emails | ips | all")
            found = sorted(set(re.findall(patterns[t], data)))
            results[t] = found

        output = []
        for t, items in results.items():
            output.append(f"[{t.upper()}] ({len(items)} found)")
            output.extend(f"  {x}" for x in items)
        return ToolResult(True, "\n".join(output) if output else "Nothing found",
                          {t: len(v) for t, v in results.items()})


# ── 15. string_transform ──────────────────────────────────────────────────────

class StringTransformTool(BaseTool):
    name = "string_transform"
    description = (
        "Apply string transformations: upper, lower, title, strip, reverse, "
        "snake_case, camel_case, count_words, truncate, pad, repeat, slug."
    )
    parameters = "text: str, operation: str, arg: str = None"

    async def run(self, text: str, operation: str, arg: str = None) -> ToolResult:
        import re
        op = operation.lower().strip()
        try:
            result = {
                "upper":       lambda: text.upper(),
                "lower":       lambda: text.lower(),
                "title":       lambda: text.title(),
                "strip":       lambda: text.strip(),
                "reverse":     lambda: text[::-1],
                "count_words": lambda: str(len(text.split())),
                "count_chars": lambda: str(len(text)),
                "count_lines": lambda: str(text.count("\n") + 1),
                "snake_case":  lambda: re.sub(r"[\s\-]+", "_", text).lower(),
                "camel_case":  lambda: "".join(w.title() for w in re.split(r"[\s_\-]+", text)),
                "slug":        lambda: re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-"),
                "truncate":    lambda: text[:int(arg or 100)] + ("..." if len(text) > int(arg or 100) else ""),
                "pad_left":    lambda: text.rjust(int(arg or 20)),
                "pad_right":   lambda: text.ljust(int(arg or 20)),
                "repeat":      lambda: text * int(arg or 2),
                "split_lines": lambda: "\n".join(f"{i+1}: {l}" for i, l in enumerate(text.splitlines())),
            }.get(op)
            if result is None:
                ops = "upper|lower|title|strip|reverse|count_words|count_chars|count_lines|snake_case|camel_case|slug|truncate|pad_left|pad_right|repeat|split_lines"
                return ToolResult(False, f"Unknown operation '{op}'. Available: {ops}")
            return ToolResult(True, result(), {"operation": op, "input_len": len(text)})
        except Exception as e:
            return ToolResult(False, f"string_transform error: {e}")


# ── 16. date_calc ─────────────────────────────────────────────────────────────

class DateCalcTool(BaseTool):
    name = "date_calc"
    description = (
        "Date and time operations: parse dates, add/subtract intervals, "
        "format timestamps, calculate differences, get current time in timezone. "
        "action='now'|'parse'|'add'|'diff'|'format'."
    )
    parameters = "action: str, date: str = None, date2: str = None, amount: int = None, unit: str = 'days', fmt: str = '%Y-%m-%d %H:%M:%S', tz: str = 'UTC'"

    async def run(self, action: str, date: str = None, date2: str = None,
                  amount: int = None, unit: str = "days",
                  fmt: str = "%Y-%m-%d %H:%M:%S", tz: str = "UTC") -> ToolResult:
        import datetime as dt
        action = action.lower().strip()

        def _parse(s):
            for f in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y%m%d"):
                try:
                    return dt.datetime.strptime(s, f)
                except ValueError:
                    pass
            raise ValueError(f"Cannot parse date: {s}")

        try:
            if action == "now":
                now = dt.datetime.utcnow()
                return ToolResult(True, now.strftime(fmt), {"utc": now.isoformat(), "timestamp": now.timestamp()})

            elif action == "parse":
                d = _parse(date)
                return ToolResult(True, d.strftime(fmt), {"parsed": d.isoformat(), "weekday": d.strftime("%A"), "timestamp": d.timestamp()})

            elif action == "add":
                d = _parse(date)
                delta_map = {"days": dt.timedelta(days=amount or 0),
                             "hours": dt.timedelta(hours=amount or 0),
                             "minutes": dt.timedelta(minutes=amount or 0),
                             "weeks": dt.timedelta(weeks=amount or 0)}
                delta = delta_map.get(unit, dt.timedelta(days=amount or 0))
                result = d + delta
                return ToolResult(True, result.strftime(fmt), {"result": result.isoformat()})

            elif action == "diff":
                d1 = _parse(date)
                d2 = _parse(date2)
                diff = abs((d2 - d1))
                return ToolResult(True, f"{diff.days} days, {diff.seconds // 3600} hours, {(diff.seconds % 3600) // 60} minutes",
                                  {"total_seconds": diff.total_seconds(), "days": diff.days})

            elif action == "format":
                d = _parse(date)
                return ToolResult(True, d.strftime(fmt))

            else:
                return ToolResult(False, f"Unknown action '{action}'. Use: now | parse | add | diff | format")
        except Exception as e:
            return ToolResult(False, f"date_calc error: {e}")


# ── 17. random_gen ────────────────────────────────────────────────────────────

class RandomGenTool(BaseTool):
    name = "random_gen"
    description = (
        "Generate random data: UUIDs, integers, floats, strings, passwords, "
        "choices from a list, or random file names. "
        "type='uuid'|'int'|'float'|'string'|'password'|'choice'."
    )
    parameters = "gen_type: str, min_val: int = 0, max_val: int = 100, length: int = 16, charset: str = 'alphanum', choices: str = None, count: int = 1"

    async def run(self, gen_type: str, min_val: int = 0, max_val: int = 100,
                  length: int = 16, charset: str = "alphanum",
                  choices: str = None, count: int = 1) -> ToolResult:
        import uuid as _uuid
        gt = gen_type.lower().strip()
        charsets = {
            "alphanum": _string.ascii_letters + _string.digits,
            "alpha":    _string.ascii_letters,
            "digits":   _string.digits,
            "hex":      _string.hexdigits[:16],
            "password": _string.ascii_letters + _string.digits + "!@#$%^&*",
            "lower":    _string.ascii_lowercase,
            "upper":    _string.ascii_uppercase,
        }
        try:
            results = []
            for _ in range(max(1, min(count, 100))):
                if gt == "uuid":
                    results.append(str(_uuid.uuid4()))
                elif gt == "int":
                    results.append(str(_random.randint(min_val, max_val)))
                elif gt == "float":
                    results.append(str(round(_random.uniform(min_val, max_val), 6)))
                elif gt in ("string", "password"):
                    cs = charsets.get(charset if gt == "string" else "password", charsets["alphanum"])
                    results.append("".join(_random.choices(cs, k=length)))
                elif gt == "choice":
                    if not choices:
                        return ToolResult(False, "choice requires 'choices' (comma-separated)")
                    items = [c.strip() for c in choices.split(",")]
                    results.append(_random.choice(items))
                else:
                    return ToolResult(False, f"Unknown type '{gt}'. Use: uuid | int | float | string | password | choice")
            return ToolResult(True, "\n".join(results), {"count": len(results), "type": gt})
        except Exception as e:
            return ToolResult(False, f"random_gen error: {e}")


# ── 18. compress_text ─────────────────────────────────────────────────────────

class CompressTextTool(BaseTool):
    name = "compress_text"
    description = (
        "Compress or decompress text/file using gzip or zlib. "
        "Returns base64-encoded compressed data, or decompressed text. "
        "action='compress'|'decompress'. algorithm='gzip'|'zlib'."
    )
    parameters = "action: str ('compress'|'decompress'), data: str, algorithm: str = 'gzip', is_file: bool = False"

    async def run(self, action: str, data: str, algorithm: str = "gzip", is_file: bool = False) -> ToolResult:
        action = action.lower().strip()
        alg = algorithm.lower().strip()
        try:
            if is_file:
                p = Path(data)
                if not p.exists():
                    return ToolResult(False, f"File not found: {data}")
                raw = p.read_bytes()
            else:
                raw = data.encode("utf-8")

            if action == "compress":
                if alg == "gzip":
                    compressed = _gzip.compress(raw, compresslevel=9)
                elif alg == "zlib":
                    compressed = _zlib.compress(raw, level=9)
                else:
                    return ToolResult(False, f"Unknown algorithm '{alg}'. Use: gzip | zlib")
                encoded = _b64encode(compressed).decode("ascii")
                ratio = round(len(compressed) / len(raw) * 100, 1)
                return ToolResult(True, encoded, {
                    "original_bytes": len(raw), "compressed_bytes": len(compressed),
                    "ratio_pct": ratio, "algorithm": alg,
                    "note": "Paste this base64 into action='decompress' to restore."
                })

            elif action == "decompress":
                try:
                    compressed = _b64decode(data.strip())
                except Exception as e:
                    return ToolResult(False, f"Base64 decode error: {e}")
                if alg == "gzip":
                    decompressed = _gzip.decompress(compressed)
                elif alg == "zlib":
                    decompressed = _zlib.decompress(compressed)
                else:
                    return ToolResult(False, f"Unknown algorithm '{alg}'. Use: gzip | zlib")
                try:
                    return ToolResult(True, decompressed.decode("utf-8"), {"bytes": len(decompressed)})
                except UnicodeDecodeError:
                    return ToolResult(True, f"(binary, {len(decompressed)} bytes)", {"is_binary": True})
            else:
                return ToolResult(False, f"Unknown action '{action}'. Use: compress | decompress")
        except Exception as e:
            return ToolResult(False, f"compress_text error: {e}")


# ── 19. xml_parse ─────────────────────────────────────────────────────────────

class XmlParseTool(BaseTool):
    name = "xml_parse"
    description = (
        "Parse XML content or file: extract elements, attributes, text, "
        "or convert to JSON dict. "
        "action='parse': full parse → JSON | action='xpath': find elements by tag | "
        "action='validate': check XML syntax."
    )
    parameters = "action: str ('parse'|'xpath'|'validate'), data: str, tag: str = None"

    async def run(self, action: str, data: str, tag: str = None) -> ToolResult:
        if len(data) < 512 and Path(data).exists():
            data = Path(data).read_text(errors="replace")

        action = action.lower().strip()

        def _elem_to_dict(elem):
            d = {"tag": elem.tag, "attrib": elem.attrib, "text": (elem.text or "").strip()}
            children = [_elem_to_dict(c) for c in elem]
            if children:
                d["children"] = children
            return d

        try:
            root = _ET.fromstring(data)
        except _ET.ParseError as e:
            return ToolResult(False, f"XML parse error: {e}")

        if action == "validate":
            return ToolResult(True, f"Valid XML. Root tag: <{root.tag}>, children: {len(list(root))}")

        elif action == "parse":
            d = _elem_to_dict(root)
            return ToolResult(True, json.dumps(d, indent=2, ensure_ascii=False), {"root_tag": root.tag})

        elif action == "xpath":
            if not tag:
                return ToolResult(False, "xpath requires 'tag' parameter")
            found = root.findall(f".//{tag}")
            if not found:
                return ToolResult(False, f"No elements found with tag '{tag}'")
            results = []
            for e in found:
                results.append({"tag": e.tag, "attrib": e.attrib, "text": (e.text or "").strip()})
            return ToolResult(True, json.dumps(results, indent=2, ensure_ascii=False), {"count": len(found)})

        else:
            return ToolResult(False, f"Unknown action '{action}'. Use: parse | xpath | validate")


# ── 20. count_stats ───────────────────────────────────────────────────────────

class CountStatsTool(BaseTool):
    name = "count_stats"
    description = (
        "Count statistics of text or file: words, lines, characters, "
        "unique words, most frequent words, avg line length, blank lines. "
        "Useful before processing large files or measuring code complexity."
    )
    parameters = "data: str, is_file: bool = False, top_words: int = 10"

    async def run(self, data: str, is_file: bool = False, top_words: int = 10) -> ToolResult:
        import collections, re
        try:
            if is_file:
                p = Path(data)
                if not p.exists():
                    return ToolResult(False, f"File not found: {data}")
                data = p.read_text(errors="replace")

            lines = data.splitlines()
            words = re.findall(r"\b[a-zA-Z_]\w*\b", data.lower())
            freq = collections.Counter(words).most_common(top_words)

            stats = {
                "characters":    len(data),
                "characters_no_space": len(data.replace(" ", "").replace("\n", "")),
                "words":         len(words),
                "unique_words":  len(set(words)),
                "lines":         len(lines),
                "blank_lines":   sum(1 for l in lines if not l.strip()),
                "avg_line_len":  round(sum(len(l) for l in lines) / max(len(lines), 1), 1),
                "max_line_len":  max((len(l) for l in lines), default=0),
                "top_words":     dict(freq),
            }
            return ToolResult(True, json.dumps(stats, indent=2), stats)
        except Exception as e:
            return ToolResult(False, f"count_stats error: {e}")


# ── deep_reason & claude_reason tools ────────────────────────────────────────

DEEP_REASON_SYSTEM = """\
Kamu adalah analytical engine yang SANGAT AKURAT dan grounded.
Aturan MUTLAK:
1. HANYA nyatakan hal yang kamu YAKIN benar — kalau ragu, katakan "Aku tidak yakin tentang X"
2. JANGAN mengarang data, angka, statistik, atau fakta teknis
3. Gunakan reasoning step-by-step yang eksplisit — tunjukkan alur pikir
4. Pisahkan dengan jelas: [FACTS] vs [ASSUMPTIONS] vs [INFERENCES]
5. Jika ada ambiguitas, nyatakan dan jawab untuk tiap interpretasi
6. Lebih baik akui ketidaktahuan daripada mengarang jawaban yang terdengar meyakinkan

Format output:
[ANALYSIS] — pemahaman masalah
[FACTS] — hal yang diketahui pasti
[REASONING] — langkah-langkah logika
[CONCLUSION] — jawaban akhir dengan confidence level (Low/Medium/High)
"""


class DeepReasonTool(BaseTool):
    """
    Grounded step-by-step reasoning tool — anti-hallucination.
    Re-calls the configured LLM with a strict analytical system prompt
    at very low temperature to minimize confabulation.
    """
    name = "deep_reason"
    description = (
        "Analisis mendalam & grounded — anti-halusinasi. "
        "Gunakan sebelum menjawab hal teknis kompleks, klaim faktual, debugging, "
        "atau saat confidence rendah. Input: pertanyaan atau klaim yang perlu diverifikasi."
    )
    parameters = "query: str"

    def __init__(self, llm_cfg: dict):
        self._cfg = llm_cfg
        self._base_url = llm_cfg.get("base_url", "https://openrouter.ai/api/v1")
        self._model = llm_cfg.get("model", "minimax/minimax-m2.7")
        self._api_key = llm_cfg.get("api_key", "")

    async def run(self, query: str) -> ToolResult:
        if not query.strip():
            return ToolResult(False, "Error: query kosong")
        if not self._api_key:
            return ToolResult(False, "Error: API key tidak ditemukan di config LLM")
        try:
            url = f"{self._base_url.rstrip('/')}/chat/completions"
            payload = {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": DEEP_REASON_SYSTEM},
                    {"role": "user", "content": query},
                ],
                "max_tokens": 2000,
                "temperature": 0.05,
            }
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/gazcc",
                "X-Title": "GazccDeepReason",
            }
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                result = data["choices"][0]["message"]["content"]
            return ToolResult(True, result, {"model": self._model, "mode": "deep_reason"})
        except httpx.HTTPStatusError as e:
            return ToolResult(False, f"HTTP {e.response.status_code}: {e.response.text[:300]}")
        except Exception as e:
            return ToolResult(False, f"DeepReasonTool error: {e}")


class ClaudeReasonTool(BaseTool):
    """
    High-quality reasoning via Claude Opus (via OpenRouter).
    Useful for complex architectural decisions, nuanced analysis,
    or tasks requiring Claude-level accuracy and depth.
    """
    name = "claude_reason"
    description = (
        "Deep reasoning pakai model gratis tier (DeepSeek R1 / Gemini) via OpenRouter — "
        "kualitas tinggi tanpa biaya. Gunakan untuk task kompleks: arsitektur sistem, "
        "analisis mendalam, keputusan teknis kritis, atau verifikasi output agent. "
        "Input: pertanyaan atau deskripsi task yang butuh analisis tingkat tinggi. "
        "Optional: tentukan model lain dengan parameter model=."
    )
    parameters = "query: str, model: str = 'minimax/minimax-m2.7'"

    def __init__(self, llm_cfg: dict):
        self._cfg = llm_cfg
        self._base_url = llm_cfg.get("base_url", "https://openrouter.ai/api/v1")
        self._api_key = llm_cfg.get("api_key", "")

    async def run(self, query: str, model: str = "minimax/minimax-m2.7") -> ToolResult:
        if not query.strip():
            return ToolResult(False, "Error: query kosong")
        if not self._api_key:
            return ToolResult(False, "Error: API key tidak ditemukan di config LLM")
        try:
            url = f"{self._base_url.rstrip('/')}/chat/completions"
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are Claude, an AI assistant by Anthropic known for exceptional "
                            "reasoning, honesty, and accuracy. Always think step by step. "
                            "Never hallucinate facts. If uncertain, explicitly state your uncertainty. "
                            "Provide thorough, nuanced, and practically actionable analysis."
                        ),
                    },
                    {"role": "user", "content": query},
                ],
                "max_tokens": 3000,
                "temperature": 0.1,
            }
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/gazcc",
                "X-Title": "GazccClaudeReason",
            }
            async with httpx.AsyncClient(timeout=90) as client:
                resp = await client.post(url, json=payload, headers=headers)
                if resp.status_code == 402:
                    return ToolResult(False, "Error 402: Kredit OpenRouter habis. Top up di openrouter.ai/credits")
                if resp.status_code == 401:
                    return ToolResult(False, "Error 401: API key invalid atau tidak punya akses ke model ini")
                resp.raise_for_status()
                data = resp.json()
                result = data["choices"][0]["message"]["content"]
            return ToolResult(True, f"[Claude {model}]\n\n{result}", {"model": model, "mode": "claude_reason"})
        except httpx.HTTPStatusError as e:
            return ToolResult(False, f"HTTP {e.response.status_code}: {e.response.text[:300]}")
        except Exception as e:
            return ToolResult(False, f"ClaudeReasonTool error: {e}")


# ── registry ──────────────────────────────────────────────────────────────────

class ToolRegistry:
    def __init__(self, cfg: dict):
        self._tools: dict[str, BaseTool] = {}
        enabled = cfg.get("tools", {})

        # Always register utility tools
        for tool in [GetTimeTool(), CalculateTool(), SummarizeTextTool(),
                     CountStatsTool(), RandomGenTool(), DateCalcTool(), StringTransformTool()]:
            self._register(tool)

        # ── Reasoning tools — always on
        llm_cfg_resolved = cfg.get("llm", {})
        self._register(DeepReasonTool(llm_cfg_resolved))
        self._register(ClaudeReasonTool(llm_cfg_resolved))

        if enabled.get("file_ops", True):
            for tool in [
                ReadFileTool(), WriteFileTool(), AppendFileTool(), ListDirTool(), DeleteFileTool(),
                ZipManageTool(), ExportFileBase64Tool(), ExportFileLinkTool(), DiffPatchTool(),
                CopyFileTool(), MoveFileTool(), FindFilesTool(), FileStatTool(), TextReplaceTool(),
                GrepSearchTool(),
            ]:
                self._register(tool)

        if enabled.get("fetch_url", True):
            self._register(FetchURLTool())
            self._register(HttpRequestTool())

        if enabled.get("web_search", True):
            self._register(WebSearchTool())
            self._register(ExtractUrlsTool())

        if enabled.get("code_exec", True):
            exec_tool = ExecuteCodeTool()
            exec_tool.ENABLED = True
            exec_tool.TIMEOUT = enabled.get("code_exec_timeout", 15)
            self._register(exec_tool)
            self._register(ShellExecTool())
        else:
            exec_tool = ExecuteCodeTool()
            exec_tool.ENABLED = False
            self._register(exec_tool)

        # Data & encoding tools — always on
        for tool in [
            JsonFormatTool(), YamlParseTool(), CsvQueryTool(),
            Base64CodecTool(), HashGenerateTool(), CompressTextTool(), XmlParseTool(),
        ]:
            self._register(tool)

    def _init_mcp(self):
        """Register MCP tools — called after all classes are defined (bottom of module)."""
        for tool in [
            McpServerConnectTool(), McpListToolsTool(), McpCallToolTool(),
            McpListResourcesTool(), McpReadResourceTool(), McpListPromptsTool(),
            McpGetPromptTool(), McpServerPingTool(), McpListServersTool(),
            McpDisconnectTool(), McpBatchCallTool(),
        ]:
            self._register(tool)

    def _register(self, tool: BaseTool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def list_all(self) -> list[dict]:
        return [
            {"name": t.name, "description": t.description, "parameters": t.parameters}
            for t in self._tools.values()
        ]

    def schema_string(self) -> str:
        lines = []
        for t in self._tools.values():
            lines.append(f"  {t.name}({t.parameters})\n    → {t.description}")
        return "\n".join(lines)

    def slim_schema_string(self) -> str:
        """
        Compressed schema — drops verbose multi-sentence descriptions,
        keeps only tool name + first sentence of description + params.
        ~50-60% fewer tokens than schema_string().
        """
        lines = []
        for t in self._tools.values():
            # Keep only first sentence of description
            short_desc = t.description.split(".")[0].strip()
            # Trim params to first 120 chars if too long
            params = t.parameters[:120] + "…" if len(t.parameters) > 120 else t.parameters
            lines.append(f"  {t.name}({params}) → {short_desc}")
        return "\n".join(lines)

    def step_schema_string(self, tool_hints: list[str], core_tools: list[str] | None = None) -> str:
        """
        Per-step schema — only includes schemas for hinted tools + a small set
        of always-available core tools. Saves ~60-70% tokens vs full schema_string().

        Args:
            tool_hints: tool names relevant to current step (from plan tool_hint)
            core_tools: tools always included regardless of hint
        """
        _core = set(core_tools or ["web_search", "fetch_url", "write_file", "read_file",
                                    "export_file_base64", "run_code", "shell", "mem_search", "mem_store"])
        target = set(tool_hints) | _core
        lines = []
        # First: hinted tools (full desc)
        for name in tool_hints:
            t = self._tools.get(name)
            if t:
                lines.append(f"  {t.name}({t.parameters})\n    → {t.description}")
        # Then: core tools (slim desc)
        for name in _core - set(tool_hints):
            t = self._tools.get(name)
            if t:
                short = t.description.split(".")[0].strip()
                lines.append(f"  {t.name} → {short}")
        # Finally: list remaining tool names only (no desc)
        rest = [n for n in self._tools if n not in target]
        if rest:
            lines.append(f"\n  (other available: {', '.join(rest)})")
        return "\n".join(lines)

    async def run(self, name: str, args: dict) -> ToolResult:
        tool = self.get(name)
        if tool is None:
            return ToolResult(False, f"Unknown tool: '{name}'. Available: {list(self._tools)}")
        try:
            return await tool.run(**args)
        except TypeError as e:
            return ToolResult(False, f"Bad arguments for {name}: {e}")
        except Exception as e:
            return ToolResult(False, f"Tool {name} crashed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# ── MCP (Model Context Protocol) TOOLS ───────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#
#  MCP adalah open protocol dari Anthropic yang nyambungin AI agent ke
#  external tools, data sources, dan services lewat standar JSON-RPC 2.0.
#
#  Tools ini implement MCP CLIENT — agent bisa:
#    • Discover & list tools dari MCP server manapun
#    • Call tools di remote MCP server
#    • Baca resources (files, DB, APIs) lewat MCP
#    • Ambil prompt templates dari MCP
#    • Health check MCP servers
#    • Manage multiple server connections
#    • Batch call multiple tools sekaligus
#
# ══════════════════════════════════════════════════════════════════════════════


# ── mcp_server_connect ────────────────────────────────────────────────────────

class McpServerConnectTool(BaseTool):
    name = "mcp_server_connect"
    description = (
        "Connect to an MCP (Model Context Protocol) server via HTTP/SSE. "
        "Performs handshake, negotiates capabilities, and returns server info. "
        "Must be called before using mcp_call_tool or mcp_list_tools on that server. "
        "Stores connection metadata in session for reuse."
    )
    parameters = "server_url: str, server_name: str = None, api_key: str = None, timeout: int = 15"

    # In-memory session store — shared across all MCP tool instances
    _sessions: dict = {}

    async def run(
        self,
        server_url: str,
        server_name: str = None,
        api_key: str = None,
        timeout: int = 15,
    ) -> ToolResult:
        if not server_url.startswith(("http://", "https://")):
            server_url = "https://" + server_url

        name = server_name or server_url.split("//")[-1].split("/")[0]
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        # MCP initialize handshake (JSON-RPC 2.0)
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "roots": {"listChanged": True},
                    "sampling": {},
                },
                "clientInfo": {"name": "GazccAgent", "version": "2.0"},
            },
        }

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(server_url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()

            if "error" in data:
                return ToolResult(False, f"MCP handshake error: {data['error']}")

            result = data.get("result", {})
            caps = result.get("capabilities", {})
            server_info = result.get("serverInfo", {})

            session = {
                "url": server_url,
                "name": name,
                "api_key": api_key,
                "headers": headers,
                "protocol_version": result.get("protocolVersion", "unknown"),
                "server_info": server_info,
                "capabilities": caps,
                "connected": True,
            }
            McpServerConnectTool._sessions[name] = session

            summary = {
                "server_name": name,
                "url": server_url,
                "protocol_version": session["protocol_version"],
                "server_info": server_info,
                "capabilities": list(caps.keys()),
                "status": "connected",
                "session_key": name,
            }
            return ToolResult(True, json.dumps(summary, indent=2), summary)

        except httpx.HTTPStatusError as e:
            return ToolResult(False, f"HTTP {e.response.status_code} connecting to {server_url}")
        except Exception as e:
            return ToolResult(False, f"mcp_server_connect error: {e}")


# ── mcp_list_tools ────────────────────────────────────────────────────────────

class McpListToolsTool(BaseTool):
    name = "mcp_list_tools"
    description = (
        "List all tools available on a connected MCP server. "
        "Returns tool names, descriptions, and input schemas. "
        "Use after mcp_server_connect to discover what the server can do. "
        "If server_name is omitted, lists tools from all connected servers."
    )
    parameters = "server_name: str = None, server_url: str = None"

    async def run(self, server_name: str = None, server_url: str = None) -> ToolResult:
        sessions = McpServerConnectTool._sessions

        # Auto-connect if url given but not in sessions
        if server_url and not server_name:
            connector = McpServerConnectTool()
            conn_result = await connector.run(server_url=server_url)
            if not conn_result.success:
                return conn_result
            server_name = list(sessions.keys())[-1]

        targets = (
            {server_name: sessions[server_name]}
            if server_name and server_name in sessions
            else sessions
        )

        if not targets:
            return ToolResult(False, "No MCP servers connected. Use mcp_server_connect first.")

        all_tools = []
        for sname, sess in targets.items():
            payload = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    resp = await client.post(sess["url"], json=payload, headers=sess["headers"])
                    data = resp.json()

                tools = data.get("result", {}).get("tools", [])
                for t in tools:
                    all_tools.append({
                        "server": sname,
                        "name": t.get("name"),
                        "description": t.get("description", ""),
                        "input_schema": t.get("inputSchema", {}),
                    })
            except Exception as e:
                all_tools.append({"server": sname, "error": str(e)})

        if not all_tools:
            return ToolResult(False, "No tools found on connected servers.")

        lines = []
        for t in all_tools:
            if "error" in t:
                lines.append(f"[{t['server']}] ERROR: {t['error']}")
            else:
                required = t["input_schema"].get("required", [])
                lines.append(f"[{t['server']}] {t['name']}\n  → {t['description']}\n  required: {required}")

        return ToolResult(True, "\n\n".join(lines), {"total_tools": len(all_tools)})


# ── mcp_call_tool ─────────────────────────────────────────────────────────────

class McpCallToolTool(BaseTool):
    name = "mcp_call_tool"
    description = (
        "Call a specific tool on a connected MCP server and return its result. "
        "Use mcp_list_tools to discover available tool names and their required parameters. "
        "Returns the tool's output content directly."
    )
    parameters = "server_name: str, tool_name: str, arguments: dict = None, timeout: int = 30"

    _call_id: int = 100

    async def run(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict = None,
        timeout: int = 30,
    ) -> ToolResult:
        sessions = McpServerConnectTool._sessions
        if server_name not in sessions:
            return ToolResult(False, f"Server '{server_name}' not connected. Use mcp_server_connect first.")

        sess = sessions[server_name]
        McpCallToolTool._call_id += 1

        payload = {
            "jsonrpc": "2.0",
            "id": McpCallToolTool._call_id,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments or {},
            },
        }

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(sess["url"], json=payload, headers=sess["headers"])
                resp.raise_for_status()
                data = resp.json()

            if "error" in data:
                err = data["error"]
                return ToolResult(False, f"MCP tool error [{err.get('code')}]: {err.get('message')}")

            result = data.get("result", {})
            is_error = result.get("isError", False)
            content = result.get("content", [])

            # Extract text from content blocks
            parts = []
            for block in content:
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "image":
                    parts.append(f"[IMAGE: {block.get('mimeType', 'unknown')}]")
                elif block.get("type") == "resource":
                    parts.append(f"[RESOURCE: {block.get('uri', '')}]")

            output = "\n".join(parts) if parts else json.dumps(result, indent=2)
            return ToolResult(not is_error, output, {
                "server": server_name,
                "tool": tool_name,
                "is_error": is_error,
                "content_blocks": len(content),
            })

        except httpx.HTTPStatusError as e:
            return ToolResult(False, f"HTTP {e.response.status_code}: {e.response.text[:300]}")
        except Exception as e:
            return ToolResult(False, f"mcp_call_tool error: {e}")


# ── mcp_list_resources ────────────────────────────────────────────────────────

class McpListResourcesTool(BaseTool):
    name = "mcp_list_resources"
    description = (
        "List all resources exposed by a connected MCP server. "
        "Resources are data sources like files, database records, API endpoints "
        "that the server makes available for reading. "
        "Returns URI, name, description, and MIME type of each resource."
    )
    parameters = "server_name: str"

    async def run(self, server_name: str) -> ToolResult:
        sessions = McpServerConnectTool._sessions
        if server_name not in sessions:
            return ToolResult(False, f"Server '{server_name}' not connected.")

        sess = sessions[server_name]
        payload = {"jsonrpc": "2.0", "id": 10, "method": "resources/list", "params": {}}

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(sess["url"], json=payload, headers=sess["headers"])
                data = resp.json()

            if "error" in data:
                return ToolResult(False, f"MCP error: {data['error']}")

            resources = data.get("result", {}).get("resources", [])
            if not resources:
                return ToolResult(True, "No resources available on this server.", {"count": 0})

            lines = []
            for r in resources:
                lines.append(
                    f"URI:  {r.get('uri', '')}\n"
                    f"Name: {r.get('name', '')}\n"
                    f"Desc: {r.get('description', '')}\n"
                    f"MIME: {r.get('mimeType', 'unknown')}"
                )

            return ToolResult(True, "\n\n".join(lines), {"count": len(resources)})
        except Exception as e:
            return ToolResult(False, f"mcp_list_resources error: {e}")


# ── mcp_read_resource ─────────────────────────────────────────────────────────

class McpReadResourceTool(BaseTool):
    name = "mcp_read_resource"
    description = (
        "Read the content of a specific resource from a connected MCP server. "
        "Use mcp_list_resources to get available URIs first. "
        "Returns the raw resource content (text, JSON, or binary info)."
    )
    parameters = "server_name: str, uri: str"

    async def run(self, server_name: str, uri: str) -> ToolResult:
        sessions = McpServerConnectTool._sessions
        if server_name not in sessions:
            return ToolResult(False, f"Server '{server_name}' not connected.")

        sess = sessions[server_name]
        payload = {
            "jsonrpc": "2.0",
            "id": 20,
            "method": "resources/read",
            "params": {"uri": uri},
        }

        try:
            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.post(sess["url"], json=payload, headers=sess["headers"])
                data = resp.json()

            if "error" in data:
                return ToolResult(False, f"MCP error: {data['error']}")

            contents = data.get("result", {}).get("contents", [])
            parts = []
            for c in contents:
                if "text" in c:
                    parts.append(c["text"])
                elif "blob" in c:
                    parts.append(f"[BINARY BLOB: {len(c['blob'])} chars base64, MIME: {c.get('mimeType','?')}]")

            return ToolResult(True, "\n".join(parts) if parts else "(empty)", {
                "uri": uri, "server": server_name, "blocks": len(contents)
            })
        except Exception as e:
            return ToolResult(False, f"mcp_read_resource error: {e}")


# ── mcp_list_prompts ──────────────────────────────────────────────────────────

class McpListPromptsTool(BaseTool):
    name = "mcp_list_prompts"
    description = (
        "List prompt templates available on an MCP server. "
        "MCP servers can expose reusable prompt templates with dynamic arguments. "
        "Returns prompt names, descriptions, and required arguments."
    )
    parameters = "server_name: str"

    async def run(self, server_name: str) -> ToolResult:
        sessions = McpServerConnectTool._sessions
        if server_name not in sessions:
            return ToolResult(False, f"Server '{server_name}' not connected.")

        sess = sessions[server_name]
        payload = {"jsonrpc": "2.0", "id": 30, "method": "prompts/list", "params": {}}

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(sess["url"], json=payload, headers=sess["headers"])
                data = resp.json()

            if "error" in data:
                return ToolResult(False, f"MCP error: {data['error']}")

            prompts = data.get("result", {}).get("prompts", [])
            if not prompts:
                return ToolResult(True, "No prompts available on this server.", {"count": 0})

            lines = []
            for p in prompts:
                args = p.get("arguments", [])
                arg_str = ", ".join(
                    f"{a['name']}{'*' if a.get('required') else ''}" for a in args
                ) or "none"
                lines.append(
                    f"Name: {p.get('name')}\n"
                    f"Desc: {p.get('description', '')}\n"
                    f"Args: {arg_str}"
                )

            return ToolResult(True, "\n\n".join(lines), {"count": len(prompts)})
        except Exception as e:
            return ToolResult(False, f"mcp_list_prompts error: {e}")


# ── mcp_get_prompt ────────────────────────────────────────────────────────────

class McpGetPromptTool(BaseTool):
    name = "mcp_get_prompt"
    description = (
        "Fetch and render a specific prompt template from an MCP server. "
        "Pass required arguments to fill in the template. "
        "Returns the rendered prompt messages ready to inject into an LLM call."
    )
    parameters = "server_name: str, prompt_name: str, arguments: dict = None"

    async def run(self, server_name: str, prompt_name: str, arguments: dict = None) -> ToolResult:
        sessions = McpServerConnectTool._sessions
        if server_name not in sessions:
            return ToolResult(False, f"Server '{server_name}' not connected.")

        sess = sessions[server_name]
        payload = {
            "jsonrpc": "2.0",
            "id": 40,
            "method": "prompts/get",
            "params": {"name": prompt_name, "arguments": arguments or {}},
        }

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(sess["url"], json=payload, headers=sess["headers"])
                data = resp.json()

            if "error" in data:
                return ToolResult(False, f"MCP error: {data['error']}")

            result = data.get("result", {})
            messages = result.get("messages", [])
            description = result.get("description", "")

            parts = []
            if description:
                parts.append(f"Description: {description}\n")
            for msg in messages:
                role = msg.get("role", "?")
                content = msg.get("content", {})
                text = content.get("text", "") if isinstance(content, dict) else str(content)
                parts.append(f"[{role.upper()}]\n{text}")

            return ToolResult(True, "\n\n".join(parts), {
                "prompt": prompt_name, "messages": len(messages)
            })
        except Exception as e:
            return ToolResult(False, f"mcp_get_prompt error: {e}")


# ── mcp_server_ping ───────────────────────────────────────────────────────────

class McpServerPingTool(BaseTool):
    name = "mcp_server_ping"
    description = (
        "Check health/connectivity of an MCP server. "
        "Returns latency, server status, and whether the server is reachable. "
        "Use to validate a server before calling tools, or monitor uptime."
    )
    parameters = "server_url: str = None, server_name: str = None"

    async def run(self, server_url: str = None, server_name: str = None) -> ToolResult:
        import time

        sessions = McpServerConnectTool._sessions

        # Resolve URL from session if name given
        if server_name and server_name in sessions:
            server_url = sessions[server_name]["url"]
        elif not server_url:
            if not sessions:
                return ToolResult(False, "No server_url or server_name provided.")
            # Ping all connected
            results = []
            for sname, sess in sessions.items():
                t0 = time.monotonic()
                try:
                    async with httpx.AsyncClient(timeout=5) as client:
                        r = await client.get(sess["url"])
                    ms = round((time.monotonic() - t0) * 1000, 1)
                    results.append(f"✓ {sname} — {ms}ms (HTTP {r.status_code})")
                except Exception as e:
                    results.append(f"✗ {sname} — unreachable: {e}")
            return ToolResult(True, "\n".join(results))

        if not server_url.startswith(("http://", "https://")):
            server_url = "https://" + server_url

        t0 = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=8) as client:
                resp = await client.get(server_url)
            ms = round((time.monotonic() - t0) * 1000, 1)
            return ToolResult(True, f"✓ Reachable — {ms}ms (HTTP {resp.status_code})", {
                "url": server_url, "latency_ms": ms, "status_code": resp.status_code
            })
        except Exception as e:
            ms = round((time.monotonic() - t0) * 1000, 1)
            return ToolResult(False, f"✗ Unreachable after {ms}ms: {e}", {
                "url": server_url, "latency_ms": ms
            })


# ── mcp_list_servers ──────────────────────────────────────────────────────────

class McpListServersTool(BaseTool):
    name = "mcp_list_servers"
    description = (
        "List all currently connected MCP servers in this session. "
        "Shows server name, URL, protocol version, and capabilities. "
        "Use to check which servers are available before calling tools."
    )
    parameters = "(none)"

    async def run(self) -> ToolResult:
        sessions = McpServerConnectTool._sessions
        if not sessions:
            return ToolResult(True, "No MCP servers connected. Use mcp_server_connect to add one.")

        lines = []
        for name, sess in sessions.items():
            caps = list(sess.get("capabilities", {}).keys())
            lines.append(
                f"Name:     {name}\n"
                f"URL:      {sess['url']}\n"
                f"Protocol: {sess.get('protocol_version', '?')}\n"
                f"Caps:     {caps}\n"
                f"Status:   {'✓ connected' if sess.get('connected') else '✗ disconnected'}"
            )

        return ToolResult(True, "\n\n".join(lines), {"count": len(sessions)})


# ── mcp_disconnect ────────────────────────────────────────────────────────────

class McpDisconnectTool(BaseTool):
    name = "mcp_disconnect"
    description = (
        "Disconnect from an MCP server and remove it from the session. "
        "Pass server_name to disconnect one, or 'all' to clear all connections."
    )
    parameters = "server_name: str"

    async def run(self, server_name: str) -> ToolResult:
        sessions = McpServerConnectTool._sessions
        if server_name.lower() == "all":
            count = len(sessions)
            sessions.clear()
            return ToolResult(True, f"Disconnected all {count} MCP server(s).")
        if server_name not in sessions:
            return ToolResult(False, f"Server '{server_name}' not found in session.")
        del sessions[server_name]
        return ToolResult(True, f"Disconnected from '{server_name}'.")


# ── mcp_batch_call ────────────────────────────────────────────────────────────

class McpBatchCallTool(BaseTool):
    name = "mcp_batch_call"
    description = (
        "Call multiple tools on an MCP server in parallel (concurrent requests). "
        "Pass a list of {tool_name, arguments} dicts. "
        "All calls fire simultaneously — much faster than sequential mcp_call_tool. "
        "Returns results indexed by tool name."
    )
    parameters = "server_name: str, calls: list, timeout: int = 30"

    async def run(self, server_name: str, calls: list, timeout: int = 30) -> ToolResult:
        sessions = McpServerConnectTool._sessions
        if server_name not in sessions:
            return ToolResult(False, f"Server '{server_name}' not connected.")
        if not calls:
            return ToolResult(False, "calls list is empty.")

        caller = McpCallToolTool()

        async def _call_one(i, c):
            tool_name = c.get("tool_name") or c.get("name", "")
            args = c.get("arguments") or c.get("args") or {}
            result = await caller.run(server_name=server_name, tool_name=tool_name,
                                      arguments=args, timeout=timeout)
            return tool_name, i, result

        tasks = [_call_one(i, c) for i, c in enumerate(calls)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output = {}
        errors = 0
        for item in results:
            if isinstance(item, Exception):
                errors += 1
                output[f"error_{errors}"] = str(item)
            else:
                tool_name, idx, res = item
                key = f"{tool_name}_{idx}" if tool_name in output else tool_name
                output[key] = {"success": res.success, "output": res.output[:2000]}
                if not res.success:
                    errors += 1

        return ToolResult(
            errors == 0,
            json.dumps(output, indent=2),
            {"total": len(calls), "errors": errors, "server": server_name},
        )




# ── Late-bind MCP tools into ToolRegistry ─────────────────────────────────────
# MCP classes defined after ToolRegistry — register them here after module load.

def _bootstrap_mcp(registry_cls):
    """Monkey-patch ToolRegistry to auto-register MCP tools on instantiation."""
    original_init = registry_cls.__init__

    def new_init(self, cfg):
        original_init(self, cfg)
        self._init_mcp()

    registry_cls.__init__ = new_init

_bootstrap_mcp(ToolRegistry)
