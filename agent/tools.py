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


# ── registry ──────────────────────────────────────────────────────────────────

class ToolRegistry:
    def __init__(self, cfg: dict):
        self._tools: dict[str, BaseTool] = {}
        enabled = cfg.get("tools", {})

        # Always register utility tools
        for tool in [GetTimeTool(), CalculateTool(), SummarizeTextTool(),
                     CountStatsTool(), RandomGenTool(), DateCalcTool(), StringTransformTool()]:
            self._register(tool)

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

    def register(self, tool: "BaseTool"):
        """Public method to register additional tools (e.g. strategic tools)."""
        self._tools[tool.name] = tool
