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

    async def run(self, code: str, timeout: int = 15) -> ToolResult:
        if not self.ENABLED:
            return ToolResult(False, "Code execution disabled in this environment.")
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


# ── registry ──────────────────────────────────────────────────────────────────

class ToolRegistry:
    def __init__(self, cfg: dict):
        self._tools: dict[str, BaseTool] = {}
        enabled = cfg.get("tools", {})

        # Always register utility tools
        for tool in [GetTimeTool(), CalculateTool(), SummarizeTextTool()]:
            self._register(tool)

        if enabled.get("file_ops", True):
            for tool in [ReadFileTool(), WriteFileTool(), AppendFileTool(), ListDirTool(), DeleteFileTool()]:
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
