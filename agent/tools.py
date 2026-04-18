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
    description = "Write content to a file. Creates parent dirs if needed."
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
    description = "Fetch content from a URL. Returns cleaned text (HTML stripped)."
    parameters = "url: str, max_chars: int = 6000"

    async def run(self, url: str, max_chars: int = 6000) -> ToolResult:
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (compatible; GazccAgent/1.0)"
                ),
            }
            async with httpx.AsyncClient(timeout=20, follow_redirects=True, headers=headers) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                ct = resp.headers.get("content-type", "")
                if "json" in ct:
                    return ToolResult(True, resp.text[:max_chars], {"url": url, "content_type": ct})
                soup = BeautifulSoup(resp.text, "lxml")
                # remove noise
                for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
                    tag.decompose()
                text = " ".join(soup.get_text(separator=" ").split())
                return ToolResult(True, text[:max_chars], {"url": url, "chars": len(text)})
        except httpx.HTTPStatusError as e:
            return ToolResult(False, f"HTTP {e.response.status_code}: {url}")
        except Exception as e:
            return ToolResult(False, f"fetch_url error: {e}")


class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Search the web using DuckDuckGo. Returns top results with titles and snippets."
    parameters = "query: str, max_results: int = 5"

    async def run(self, query: str, max_results: int = 5) -> ToolResult:
        try:
            # DuckDuckGo Instant Answer API
            url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1",
            }
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(url, params=params)
                data = resp.json()

            parts: list[str] = []

            if data.get("Answer"):
                parts.append(f"📌 Instant Answer: {data['Answer']}")

            if data.get("AbstractText"):
                parts.append(f"📖 Summary: {data['AbstractText']}")
                if data.get("AbstractURL"):
                    parts.append(f"   Source: {data['AbstractURL']}")

            topics = data.get("RelatedTopics", [])
            count = 0
            for t in topics:
                if count >= max_results:
                    break
                if isinstance(t, dict) and t.get("Text"):
                    txt = t["Text"]
                    link = t.get("FirstURL", "")
                    parts.append(f"• {txt}" + (f"\n  → {link}" if link else ""))
                    count += 1
                elif isinstance(t, dict) and t.get("Topics"):
                    for sub in t["Topics"]:
                        if count >= max_results:
                            break
                        if sub.get("Text"):
                            parts.append(f"• {sub['Text']}" + (f"\n  → {sub.get('FirstURL','')}" if sub.get("FirstURL") else ""))
                            count += 1

            if not parts:
                # fallback: try HTML scrape of DDG
                return await self._html_fallback(query, max_results)

            return ToolResult(True, "\n\n".join(parts), {"query": query})
        except Exception as e:
            return ToolResult(False, f"web_search error: {e}")

    async def _html_fallback(self, query: str, max_results: int) -> ToolResult:
        try:
            url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
            headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0"}
            async with httpx.AsyncClient(timeout=15, headers=headers) as client:
                resp = await client.get(url)
            soup = BeautifulSoup(resp.text, "lxml")
            results = []
            for r in soup.select(".result")[:max_results]:
                title_el = r.select_one(".result__title")
                snip_el = r.select_one(".result__snippet")
                url_el = r.select_one(".result__url")
                title = title_el.get_text(strip=True) if title_el else ""
                snip = snip_el.get_text(strip=True) if snip_el else ""
                link = url_el.get_text(strip=True) if url_el else ""
                if title:
                    results.append(f"• {title}\n  {snip}\n  → {link}")
            if not results:
                return ToolResult(False, f"No results found for: {query}")
            return ToolResult(True, "\n\n".join(results), {"query": query})
        except Exception as e:
            return ToolResult(False, f"search fallback error: {e}")


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

# ── registry ──────────────────────────────────────────────────────────────────

class ToolRegistry:
    def __init__(self, cfg: dict):
        self._tools: dict[str, BaseTool] = {}
        enabled = cfg.get("tools", {})

        # Always register utility tools
        for tool in [GetTimeTool(), CalculateTool(), SummarizeTextTool()]:
            self._register(tool)

        if enabled.get("file_ops", True):
            for tool in [ReadFileTool(), WriteFileTool(), AppendFileTool(), ListDirTool(), DeleteFileTool(),
                         ExportFileBase64Tool(), ExportFileLinkTool()]:
                self._register(tool)

        if enabled.get("fetch_url", True):
            self._register(FetchURLTool())

        if enabled.get("web_search", True):
            self._register(WebSearchTool())

        if enabled.get("code_exec", True):
            exec_tool = ExecuteCodeTool()
            exec_tool.ENABLED = True
            exec_tool.TIMEOUT = enabled.get("code_exec_timeout", 15)
            self._register(exec_tool)
        else:
            exec_tool = ExecuteCodeTool()
            exec_tool.ENABLED = False
            self._register(exec_tool)

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
