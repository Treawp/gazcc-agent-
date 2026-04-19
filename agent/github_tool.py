"""
agent/github_tool.py
═══════════════════════════════════════════════════════════════════════════════
  GAZCC — GITHUB INTELLIGENCE TOOLKIT
  ══════════════════════════════════════════════════════════════════════════════
  Tools:
    github_repo_info     — Metadata repo (stars, forks, desc, license, topics)
    github_list_files    — Directory tree (recursive atau satu level)
    github_read_file     — Baca isi file dari repo
    github_commits       — Recent commits + author + message
    github_contributors  — Top contributors + commit count
    github_languages     — Breakdown bahasa pemrograman + persentase
    github_search_code   — Search code di dalam repo (GitHub Code Search API)
    github_analyze       — Full analysis: semua info sekaligus dalam satu call

  REGISTER ke ToolRegistry:
    from .github_tool import register_github_tools
    register_github_tools(registry, cfg)

  CONFIG (config.yaml):
    tools:
      github_tools: true
      github_token: ""    # opsional, rate limit naik dari 60 → 5000 req/hr
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


# ── Shared GitHub HTTP client ─────────────────────────────────────────────────

GH_API = "https://api.github.com"
_TOKEN: str = ""          # set via set_github_token() atau env var


def set_github_token(token: str) -> None:
    global _TOKEN
    _TOKEN = token.strip()


def _headers() -> dict:
    h = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "GazccAgent/2.0",
    }
    token = _TOKEN or os.environ.get("GITHUB_TOKEN", "")
    if token:
        h["Authorization"] = f"token {token}"
    return h


def _headers_raw() -> dict:
    """Headers untuk fetch raw file content."""
    h = {
        "Accept": "application/vnd.github.v3.raw",
        "User-Agent": "GazccAgent/2.0",
    }
    token = _TOKEN or os.environ.get("GITHUB_TOKEN", "")
    if token:
        h["Authorization"] = f"token {token}"
    return h


def _parse_slug(repo: str) -> str | None:
    """Normalisasi input: URL GitHub atau 'owner/repo' → 'owner/repo'."""
    repo = repo.strip().rstrip("/")
    m = re.search(r"github\.com/([^/]+/[^/]+?)(?:\.git)?(?:/.*)?$", repo)
    if m:
        return m.group(1)
    if re.match(r"^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$", repo):
        return repo
    return None


def _fmt_num(n: int | None) -> str:
    if n is None:
        return "—"
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}k"
    return str(n)


def _time_ago(iso: str | None) -> str:
    if not iso:
        return "—"
    try:
        import datetime
        dt = datetime.datetime.fromisoformat(iso.replace("Z", "+00:00"))
        now = datetime.datetime.now(datetime.timezone.utc)
        diff = int((now - dt).total_seconds())
        if diff < 60:      return f"{diff}s ago"
        if diff < 3600:    return f"{diff//60}m ago"
        if diff < 86400:   return f"{diff//3600}h ago"
        if diff < 2592000: return f"{diff//86400}d ago"
        if diff < 31536000:return f"{diff//2592000}mo ago"
        return f"{diff//31536000}y ago"
    except Exception:
        return iso[:10]


async def _gh_get(endpoint: str, timeout: int = 15) -> tuple[bool, Any]:
    """
    GET GitHub API endpoint.
    Returns (success: bool, data: dict|list|str)
    """
    url = f"{GH_API}{endpoint}"
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(url, headers=_headers())
        if resp.status_code == 403:
            remaining = resp.headers.get("X-RateLimit-Remaining", "?")
            reset_ts = resp.headers.get("X-RateLimit-Reset", "")
            reset_str = ""
            if reset_ts.isdigit():
                wait = int(reset_ts) - int(time.time())
                reset_str = f" (reset in {max(0,wait)}s)"
            return False, f"Rate limit hit. Remaining: {remaining}{reset_str}. Set GITHUB_TOKEN untuk limit lebih tinggi."
        if resp.status_code == 404:
            return False, f"Not found: {endpoint}"
        if resp.status_code == 401:
            return False, "Unauthorized. Token tidak valid."
        if not resp.is_success:
            return False, f"GitHub API {resp.status_code}: {resp.text[:200]}"
        return True, resp.json()
    except httpx.TimeoutException:
        return False, f"Timeout after {timeout}s"
    except Exception as e:
        return False, f"HTTP error: {e}"


async def _gh_raw(endpoint: str, timeout: int = 15) -> tuple[bool, str]:
    """GET raw file content dari GitHub."""
    url = f"{GH_API}{endpoint}"
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(url, headers=_headers_raw())
        if not resp.is_success:
            return False, f"GitHub {resp.status_code}: {resp.text[:200]}"
        return True, resp.text
    except Exception as e:
        return False, f"Fetch error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# 1. github_repo_info — Metadata repo
# ══════════════════════════════════════════════════════════════════════════════

class GithubRepoInfoTool(BaseTool):
    name = "github_repo_info"
    description = (
        "Ambil metadata lengkap repositori GitHub: deskripsi, stars, forks, watchers, "
        "issues, license, topics, bahasa utama, tanggal dibuat/update, ukuran, dll. "
        "Input: URL GitHub atau 'owner/repo'."
    )
    parameters = "repo: str"

    async def run(self, repo: str) -> ToolResult:
        slug = _parse_slug(repo)
        if not slug:
            return ToolResult(False, f"Repo tidak valid: '{repo}'. Gunakan format 'owner/repo' atau URL GitHub.")

        ok, data = await _gh_get(f"/repos/{slug}")
        if not ok:
            return ToolResult(False, data)

        owner = data.get("owner", {})
        license_info = data.get("license") or {}
        lines = [
            f"╔══ REPO: {data['full_name']} ══",
            f"║ Description : {data.get('description') or '(none)'}",
            f"║ URL         : {data.get('html_url')}",
            f"║ Owner       : {owner.get('login')} ({owner.get('type')})",
            f"║ Visibility  : {'Private 🔒' if data.get('private') else 'Public 🌐'}",
            f"║ Fork        : {'Yes (forked repo)' if data.get('fork') else 'No (original)'}",
            f"╠══ STATS",
            f"║ ⭐ Stars     : {_fmt_num(data.get('stargazers_count'))}",
            f"║ 🍴 Forks     : {_fmt_num(data.get('forks_count'))}",
            f"║ 👁 Watchers  : {_fmt_num(data.get('watchers_count'))}",
            f"║ 🐛 Issues    : {_fmt_num(data.get('open_issues_count'))}",
            f"║ 💾 Size      : {data.get('size', 0)} KB",
            f"╠══ META",
            f"║ Language    : {data.get('language') or '—'}",
            f"║ License     : {license_info.get('spdx_id') or license_info.get('name') or '—'}",
            f"║ Branch      : {data.get('default_branch')}",
            f"║ Topics      : {', '.join(data.get('topics') or []) or '—'}",
            f"╠══ TIMESTAMPS",
            f"║ Created     : {data.get('created_at','')[:10]} ({_time_ago(data.get('created_at'))})",
            f"║ Updated     : {data.get('updated_at','')[:10]} ({_time_ago(data.get('updated_at'))})",
            f"║ Last Push   : {data.get('pushed_at','')[:10]} ({_time_ago(data.get('pushed_at'))})",
            f"╠══ FEATURES",
            f"║ Wiki        : {'✓' if data.get('has_wiki') else '✗'}  "
            f"  Issues: {'✓' if data.get('has_issues') else '✗'}  "
            f"  Pages: {'✓' if data.get('has_pages') else '✗'}",
            f"╚══════════════════════",
        ]
        return ToolResult(True, "\n".join(lines), {
            "full_name": data.get("full_name"),
            "stars": data.get("stargazers_count"),
            "language": data.get("language"),
            "private": data.get("private"),
        })


# ══════════════════════════════════════════════════════════════════════════════
# 2. github_list_files — Directory tree
# ══════════════════════════════════════════════════════════════════════════════

class GithubListFilesTool(BaseTool):
    name = "github_list_files"
    description = (
        "List file dan folder di repositori GitHub. "
        "Mode 'tree' = seluruh struktur rekursif. "
        "Mode 'dir' = satu level direktori tertentu. "
        "Berguna untuk memahami layout proyek sebelum baca file spesifik."
    )
    parameters = "repo: str, path: str = '', mode: str = 'tree', branch: str = 'HEAD', max_files: int = 200"

    async def run(
        self,
        repo: str,
        path: str = "",
        mode: str = "tree",
        branch: str = "HEAD",
        max_files: int = 200,
    ) -> ToolResult:
        slug = _parse_slug(repo)
        if not slug:
            return ToolResult(False, f"Repo tidak valid: '{repo}'")

        if mode == "tree":
            ok, data = await _gh_get(f"/repos/{slug}/git/trees/{branch}?recursive=1")
            if not ok:
                return ToolResult(False, data)

            tree = data.get("tree", [])
            truncated = data.get("truncated", False)

            # Filter by path prefix if specified
            if path:
                tree = [t for t in tree if t["path"].startswith(path.strip("/") + "/") or t["path"] == path.strip("/")]

            blobs = [t for t in tree if t["type"] == "blob"]
            dirs  = [t for t in tree if t["type"] == "tree"]

            # Build visual tree
            lines = [f"📁 {slug} ({len(blobs)} files, {len(dirs)} dirs)"]
            if truncated:
                lines.append("⚠ Tree truncated by GitHub (repo too large)")

            seen = set()
            for item in sorted(tree, key=lambda x: x["path"])[:max_files]:
                parts = item["path"].split("/")
                depth = len(parts) - 1
                indent = "  " * depth
                name = parts[-1]
                is_dir = item["type"] == "tree"
                icon = "📁" if is_dir else _file_icon(name)
                size_str = ""
                if not is_dir and item.get("size"):
                    s = item["size"]
                    size_str = f"  [{s/1024:.1f}KB]" if s >= 1024 else f"  [{s}B]"
                lines.append(f"{indent}{'└─' if depth else ''}  {icon} {name}{size_str}")

            if len(tree) > max_files:
                lines.append(f"... dan {len(tree) - max_files} item lainnya (set max_files lebih tinggi)")

            return ToolResult(True, "\n".join(lines), {
                "total_files": len(blobs),
                "total_dirs": len(dirs),
                "truncated": truncated,
            })

        else:  # mode == 'dir'
            endpoint = f"/repos/{slug}/contents/{path.strip('/')}" if path else f"/repos/{slug}/contents"
            ok, data = await _gh_get(endpoint)
            if not ok:
                return ToolResult(False, data)

            if not isinstance(data, list):
                return ToolResult(False, f"Path '{path}' bukan direktori.")

            lines = [f"📁 /{path or ''} ({len(data)} items)"]
            for item in sorted(data, key=lambda x: (x["type"] != "dir", x["name"])):
                icon = "📁" if item["type"] == "dir" else _file_icon(item["name"])
                size_str = ""
                if item["type"] == "file" and item.get("size"):
                    s = item["size"]
                    size_str = f"  [{s/1024:.1f}KB]" if s >= 1024 else f"  [{s}B]"
                lines.append(f"  {icon} {item['name']}{size_str}  ({item['type']})")

            return ToolResult(True, "\n".join(lines), {"count": len(data)})


def _file_icon(name: str) -> str:
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    return {
        "py": "🐍", "js": "📜", "ts": "📘", "jsx": "⚛", "tsx": "⚛",
        "json": "📋", "yaml": "⚙", "yml": "⚙", "toml": "⚙",
        "md": "📝", "rst": "📝", "txt": "📄",
        "sh": "🖥", "bash": "🖥", "zsh": "🖥",
        "html": "🌐", "css": "🎨", "scss": "🎨",
        "java": "☕", "go": "🐹", "rs": "⚙", "cpp": "⚙", "c": "⚙",
        "dockerfile": "🐳", "lock": "🔒",
        "png": "🖼", "jpg": "🖼", "svg": "🎨", "gif": "🖼",
        "pdf": "📕", "zip": "📦", "tar": "📦",
    }.get(ext, {
        "dockerfile": "🐳", "makefile": "🔧", "readme": "📖",
        "license": "⚖", "contributing": "🤝", "gitignore": "👻",
    }.get(name.lower(), "📄"))


# ══════════════════════════════════════════════════════════════════════════════
# 3. github_read_file — Baca isi file
# ══════════════════════════════════════════════════════════════════════════════

class GithubReadFileTool(BaseTool):
    name = "github_read_file"
    description = (
        "Baca isi file dari repositori GitHub. "
        "Mendukung semua jenis file teks: .py, .js, .json, .yaml, .md, dll. "
        "Bisa baca dari branch/tag/commit tertentu."
    )
    parameters = "repo: str, file_path: str, branch: str = 'HEAD', max_chars: int = 8000"

    async def run(
        self,
        repo: str,
        file_path: str,
        branch: str = "HEAD",
        max_chars: int = 8000,
    ) -> ToolResult:
        slug = _parse_slug(repo)
        if not slug:
            return ToolResult(False, f"Repo tidak valid: '{repo}'")

        file_path = file_path.strip("/")
        ok, content = await _gh_raw(f"/repos/{slug}/contents/{file_path}?ref={branch}")
        if not ok:
            return ToolResult(False, content)

        truncated = len(content) > max_chars
        display = content[:max_chars] + ("\n\n... [TRUNCATED - set max_chars lebih tinggi]" if truncated else "")

        header = f"── FILE: {file_path} ({len(content)} chars) {'[TRUNCATED]' if truncated else ''} ──\n"
        return ToolResult(True, header + display, {
            "path": file_path,
            "size": len(content),
            "truncated": truncated,
            "branch": branch,
        })


# ══════════════════════════════════════════════════════════════════════════════
# 4. github_commits — Recent commits
# ══════════════════════════════════════════════════════════════════════════════

class GithubCommitsTool(BaseTool):
    name = "github_commits"
    description = (
        "Ambil daftar recent commits dari repositori GitHub. "
        "Tampilkan SHA, author, tanggal, dan pesan commit. "
        "Opsional filter by branch atau path file tertentu."
    )
    parameters = "repo: str, branch: str = '', path: str = '', limit: int = 10"

    async def run(
        self,
        repo: str,
        branch: str = "",
        path: str = "",
        limit: int = 10,
    ) -> ToolResult:
        slug = _parse_slug(repo)
        if not slug:
            return ToolResult(False, f"Repo tidak valid: '{repo}'")

        limit = min(max(1, limit), 50)
        params = f"?per_page={limit}"
        if branch:
            params += f"&sha={branch}"
        if path:
            params += f"&path={path}"

        ok, data = await _gh_get(f"/repos/{slug}/commits{params}")
        if not ok:
            return ToolResult(False, data)

        lines = [f"📝 COMMITS — {slug}{f' [{branch}]' if branch else ''}  ({len(data)} shown)"]
        lines.append("─" * 60)
        for c in data:
            sha = c.get("sha", "")[:7]
            commit = c.get("commit", {})
            author_info = commit.get("author", {})
            author = author_info.get("name", "?")
            date  = author_info.get("date", "")[:10]
            msg   = commit.get("message", "").split("\n")[0][:80]
            lines.append(f"  {sha}  {date}  {author[:20]:<20}  {msg}")

        return ToolResult(True, "\n".join(lines), {"count": len(data)})


# ══════════════════════════════════════════════════════════════════════════════
# 5. github_contributors — Top contributors
# ══════════════════════════════════════════════════════════════════════════════

class GithubContributorsTool(BaseTool):
    name = "github_contributors"
    description = (
        "Ambil daftar top contributors repositori GitHub beserta jumlah commit. "
        "Diurutkan dari kontributor terbanyak commit-nya."
    )
    parameters = "repo: str, limit: int = 15"

    async def run(self, repo: str, limit: int = 15) -> ToolResult:
        slug = _parse_slug(repo)
        if not slug:
            return ToolResult(False, f"Repo tidak valid: '{repo}'")

        limit = min(max(1, limit), 30)
        ok, data = await _gh_get(f"/repos/{slug}/contributors?per_page={limit}&anon=false")
        if not ok:
            return ToolResult(False, data)

        lines = [f"👥 CONTRIBUTORS — {slug}  (top {len(data)})"]
        lines.append("─" * 50)
        for i, c in enumerate(data, 1):
            login = c.get("login", "?")
            contribs = c.get("contributions", 0)
            ctype = c.get("type", "User")
            bar = "█" * min(20, int(contribs / max(data[0].get("contributions",1), 1) * 20))
            lines.append(f"  #{i:>2}  {login:<25}  {contribs:>5} commits  {bar}")

        return ToolResult(True, "\n".join(lines), {
            "total_shown": len(data),
            "top_contributor": data[0].get("login") if data else None,
        })


# ══════════════════════════════════════════════════════════════════════════════
# 6. github_languages — Breakdown bahasa pemrograman
# ══════════════════════════════════════════════════════════════════════════════

class GithubLanguagesTool(BaseTool):
    name = "github_languages"
    description = (
        "Ambil breakdown bahasa pemrograman yang digunakan di repositori GitHub. "
        "Menampilkan persentase dan ukuran bytes per bahasa."
    )
    parameters = "repo: str"

    async def run(self, repo: str) -> ToolResult:
        slug = _parse_slug(repo)
        if not slug:
            return ToolResult(False, f"Repo tidak valid: '{repo}'")

        ok, data = await _gh_get(f"/repos/{slug}/languages")
        if not ok:
            return ToolResult(False, data)

        if not data:
            return ToolResult(True, "Tidak ada data bahasa (repo kosong?).", {})

        total = sum(data.values())
        sorted_langs = sorted(data.items(), key=lambda x: x[1], reverse=True)

        lines = [f"💻 LANGUAGES — {slug}  (total: {total/1024:.1f} KB)"]
        lines.append("─" * 50)
        for lang, byt in sorted_langs:
            pct = byt / total * 100
            bar = "█" * int(pct / 3)
            size_str = f"{byt/1024:.1f}KB" if byt >= 1024 else f"{byt}B"
            lines.append(f"  {lang:<20}  {pct:>6.2f}%  {bar:<34}  {size_str}")

        primary = sorted_langs[0][0]
        return ToolResult(True, "\n".join(lines), {
            "primary_language": primary,
            "language_count": len(data),
            "breakdown": {k: round(v/total*100, 2) for k,v in sorted_langs},
        })


# ══════════════════════════════════════════════════════════════════════════════
# 7. github_search_code — Search code dalam repo
# ══════════════════════════════════════════════════════════════════════════════

class GithubSearchCodeTool(BaseTool):
    name = "github_search_code"
    description = (
        "Cari kode di dalam repositori GitHub menggunakan GitHub Code Search API. "
        "Berguna untuk mencari fungsi, class, pattern, atau kata kunci di seluruh codebase. "
        "Catatan: Butuh token untuk akses penuh, tanpa token limit sangat ketat."
    )
    parameters = "repo: str, query: str, language: str = '', limit: int = 10"

    async def run(
        self,
        repo: str,
        query: str,
        language: str = "",
        limit: int = 10,
    ) -> ToolResult:
        slug = _parse_slug(repo)
        if not slug:
            return ToolResult(False, f"Repo tidak valid: '{repo}'")

        q = f"{query}+repo:{slug}"
        if language:
            q += f"+language:{language}"

        ok, data = await _gh_get(f"/search/code?q={q}&per_page={min(limit,10)}")
        if not ok:
            return ToolResult(False, data)

        items = data.get("items", [])
        total = data.get("total_count", 0)

        if not items:
            return ToolResult(True, f"Tidak ditemukan hasil untuk '{query}' di {slug}.", {"total": 0})

        lines = [f"🔍 CODE SEARCH: '{query}' in {slug}  ({total} total results, showing {len(items)})"]
        lines.append("─" * 60)
        for item in items:
            path = item.get("path", "")
            name = item.get("name", "")
            url  = item.get("html_url", "")
            lines.append(f"  📄 {path}")
            lines.append(f"     → {url}")

        return ToolResult(True, "\n".join(lines), {
            "total": total,
            "shown": len(items),
        })


# ══════════════════════════════════════════════════════════════════════════════
# 8. github_analyze — Full analysis dalam satu call
# ══════════════════════════════════════════════════════════════════════════════

CRITICAL_FILES = [
    "README.md", "readme.md", "README.rst",
    "requirements.txt", "package.json", "Pipfile",
    "pyproject.toml", "setup.py", "Cargo.toml", "go.mod",
    "Dockerfile", "docker-compose.yml", "docker-compose.yaml",
    ".env.example", "LICENSE", "LICENSE.md",
    ".gitignore", "CONTRIBUTING.md",
]


class GithubAnalyzeTool(BaseTool):
    name = "github_analyze"
    description = (
        "Full analysis repositori GitHub dalam satu call. "
        "Menggabungkan: metadata, struktur direktori, bahasa, kontributor, "
        "recent commits, deteksi file penting, dan analisis arsitektur. "
        "Gunakan ini sebagai entry point utama sebelum eksplorasi lebih dalam."
    )
    parameters = "repo: str, include_readme: bool = True, max_tree_files: int = 100"

    async def run(
        self,
        repo: str,
        include_readme: bool = True,
        max_tree_files: int = 100,
    ) -> ToolResult:
        slug = _parse_slug(repo)
        if not slug:
            return ToolResult(False, f"Repo tidak valid: '{repo}'")

        # Fire semua request parallel
        results = await asyncio.gather(
            _gh_get(f"/repos/{slug}"),
            _gh_get(f"/repos/{slug}/git/trees/HEAD?recursive=1"),
            _gh_get(f"/repos/{slug}/languages"),
            _gh_get(f"/repos/{slug}/contributors?per_page=5"),
            _gh_get(f"/repos/{slug}/commits?per_page=5"),
            return_exceptions=True,
        )

        (ok_r, repo_data), (ok_t, tree_data), (ok_l, langs_data), \
        (ok_c, contribs_data), (ok_cm, commits_data) = results

        if not ok_r:
            return ToolResult(False, f"Gagal ambil info repo: {repo_data}")

        # ── Section 1: Repo Info
        r = repo_data
        license_spdx = (r.get("license") or {}).get("spdx_id", "—")
        sections = []
        sections.append(
            f"╔══════════════════════════════════════════════════════\n"
            f"║  GITHUB ANALYSIS: {r['full_name']}\n"
            f"╚══════════════════════════════════════════════════════\n"
            f"\n📊 METADATA\n"
            f"  Description : {r.get('description') or '(none)'}\n"
            f"  URL         : {r.get('html_url')}\n"
            f"  ⭐ Stars    : {_fmt_num(r.get('stargazers_count'))}  "
            f"  🍴 Forks: {_fmt_num(r.get('forks_count'))}  "
            f"  🐛 Issues: {_fmt_num(r.get('open_issues_count'))}\n"
            f"  Language    : {r.get('language') or '—'}\n"
            f"  License     : {license_spdx}\n"
            f"  Branch      : {r.get('default_branch')}\n"
            f"  Topics      : {', '.join(r.get('topics') or []) or '—'}\n"
            f"  Created     : {r.get('created_at','')[:10]}  "
            f"  Last Push: {_time_ago(r.get('pushed_at'))}\n"
            f"  Size        : {r.get('size', 0)} KB"
        )

        # ── Section 2: Tree / Structure
        if ok_t and isinstance(tree_data, dict):
            tree = tree_data.get("tree", [])
            blobs = [t for t in tree if t["type"] == "blob"]
            dirs  = [t for t in tree if t["type"] == "tree"]

            # Detect important files
            existing = {t["path"] for t in tree}
            found_important = []
            missing_important = []
            for f in CRITICAL_FILES:
                matches = [p for p in existing if p == f or p.endswith("/" + f)]
                if matches:
                    found_important.append(matches[0])
                else:
                    missing_important.append(f)

            # Root-level files/dirs for quick overview
            root_items = [t for t in tree if "/" not in t["path"]]
            root_display = []
            for item in sorted(root_items, key=lambda x: (x["type"] != "tree", x["path"]))[:30]:
                icon = "📁" if item["type"] == "tree" else _file_icon(item["path"])
                root_display.append(f"  {icon} {item['path']}")

            sections.append(
                f"\n📁 STRUCTURE  ({len(blobs)} files, {len(dirs)} dirs"
                + (f", TRUNCATED" if tree_data.get("truncated") else "") + ")\n"
                + "\n".join(root_display[:max_tree_files])
                + (f"\n  ... (+{len(root_items)-30} more root items)" if len(root_items) > 30 else "")
            )

            found_str = "\n".join(f"  ✓ {f}" for f in found_important[:10]) or "  (none)"
            missing_str = "\n".join(f"  ✗ {f}" for f in missing_important[:6]) or "  (none)"
            sections.append(
                f"\n📋 IMPORTANT FILES\n"
                f"  FOUND:\n{found_str}\n"
                f"  MISSING:\n{missing_str}"
            )

            # Architecture detection
            arch_hints = _detect_architecture(existing)
            if arch_hints:
                sections.append(f"\n🏗 ARCHITECTURE HINTS\n" + "\n".join(f"  • {h}" for h in arch_hints))

        # ── Section 3: Languages
        if ok_l and isinstance(langs_data, dict) and langs_data:
            total_b = sum(langs_data.values())
            sorted_l = sorted(langs_data.items(), key=lambda x: x[1], reverse=True)
            lang_lines = []
            for lang, byt in sorted_l[:8]:
                pct = byt / total_b * 100
                bar = "█" * int(pct / 5)
                lang_lines.append(f"  {lang:<18} {pct:>6.1f}%  {bar}")
            sections.append(f"\n💻 LANGUAGES\n" + "\n".join(lang_lines))

        # ── Section 4: Recent commits
        if ok_cm and isinstance(commits_data, list) and commits_data:
            commit_lines = []
            for c in commits_data:
                sha = c.get("sha", "")[:7]
                cm = c.get("commit", {})
                author = (cm.get("author") or {}).get("name", "?")[:18]
                date   = (cm.get("author") or {}).get("date", "")[:10]
                msg    = cm.get("message", "").split("\n")[0][:60]
                commit_lines.append(f"  {sha}  {date}  {author:<18}  {msg}")
            sections.append(f"\n📝 RECENT COMMITS\n" + "\n".join(commit_lines))

        # ── Section 5: Top contributors
        if ok_c and isinstance(contribs_data, list) and contribs_data:
            contrib_lines = []
            for c in contribs_data:
                login = c.get("login", "?")[:20]
                n = c.get("contributions", 0)
                contrib_lines.append(f"  {login:<22}  {n} commits")
            sections.append(f"\n👥 TOP CONTRIBUTORS\n" + "\n".join(contrib_lines))

        # ── Section 6: README preview (async, independent)
        if include_readme and ok_t and isinstance(tree_data, dict):
            tree_paths = {t["path"] for t in tree_data.get("tree", [])}
            readme_path = next(
                (p for p in ["README.md", "readme.md", "README.rst", "Readme.md"]
                 if p in tree_paths), None
            )
            if readme_path:
                ok_readme, readme_content = await _gh_raw(
                    f"/repos/{slug}/contents/{readme_path}"
                )
                if ok_readme:
                    preview = readme_content[:1500]
                    if len(readme_content) > 1500:
                        preview += "\n\n... [README truncated, gunakan github_read_file untuk full content]"
                    sections.append(f"\n📖 README PREVIEW\n{'─'*50}\n{preview}")

        full_output = "\n".join(sections)
        return ToolResult(True, full_output, {
            "slug": slug,
            "stars": r.get("stargazers_count"),
            "language": r.get("language"),
            "files": len([t for t in (tree_data.get("tree", []) if ok_t and isinstance(tree_data, dict) else []) if t["type"] == "blob"]),
        })


def _detect_architecture(paths: set) -> list[str]:
    """Deteksi pola arsitektur dari nama file/folder."""
    hints = []

    # Framework detection
    if "package.json" in paths:
        if any("next.config" in p for p in paths):
            hints.append("Next.js project")
        elif any("vite.config" in p for p in paths):
            hints.append("Vite-based frontend")
        elif any("nuxt.config" in p for p in paths):
            hints.append("Nuxt.js project")
        else:
            hints.append("Node.js / npm project")

    if any(p.endswith("requirements.txt") or p == "pyproject.toml" or p.endswith("setup.py") for p in paths):
        if any("fastapi" in p or "app.py" in p or "main.py" in p for p in paths):
            hints.append("Python web app (likely FastAPI/Flask/Django)")
        elif any("train" in p or "model" in p or "notebook" in p or p.endswith(".ipynb") for p in paths):
            hints.append("Python ML/Data Science project")
        else:
            hints.append("Python project")

    if "Cargo.toml" in paths:
        hints.append("Rust project (Cargo)")
    if "go.mod" in paths:
        hints.append("Go module project")
    if "pom.xml" in paths or "build.gradle" in paths:
        hints.append("Java/JVM project")

    # Architecture patterns
    if any(p.startswith("src/") for p in paths):
        hints.append("src/ layout (standard package structure)")
    if any(p.startswith("tests/") or p.startswith("test/") for p in paths):
        hints.append("Has test suite (tests/ directory)")
    if "Dockerfile" in paths or "docker-compose.yml" in paths:
        hints.append("Docker-containerized")
    if ".github/workflows" in str(paths) or any("workflows" in p for p in paths):
        hints.append("GitHub Actions CI/CD configured")
    if "terraform" in str(paths).lower() or any(p.endswith(".tf") for p in paths):
        hints.append("Terraform IaC detected")
    if any("api/" in p or "/routes/" in p or "/controllers/" in p for p in paths):
        hints.append("REST API structure detected")
    if any("migrations/" in p or "alembic" in p for p in paths):
        hints.append("Database migrations present (SQLAlchemy/Alembic/Django)")
    if any(p.endswith(".proto") for p in paths):
        hints.append("gRPC / Protocol Buffers")
    if "kubernetes" in str(paths).lower() or any(p.endswith(".k8s.yaml") for p in paths):
        hints.append("Kubernetes manifests detected")

    return hints


# ══════════════════════════════════════════════════════════════════════════════
# REGISTER — panggil dari core.py atau tools.py
# ══════════════════════════════════════════════════════════════════════════════

GITHUB_TOOLS = [
    GithubRepoInfoTool,
    GithubListFilesTool,
    GithubReadFileTool,
    GithubCommitsTool,
    GithubContributorsTool,
    GithubLanguagesTool,
    GithubSearchCodeTool,
    GithubAnalyzeTool,
]


def register_github_tools(registry, cfg: dict | None = None) -> None:
    """
    Register semua GitHub tools ke ToolRegistry yang sudah ada.

    USAGE di agent/core.py:
        from .github_tool import register_github_tools
        register_github_tools(self._tools, self._cfg)

    atau di extra_tools.py / gazcc_tools_expansion.py:
        from .github_tool import register_github_tools
        register_github_tools(registry, cfg)

    CONFIG (config.yaml):
        tools:
          github_tools: true
          github_token: "ghp_xxxxx"   # opsional
    """
    cfg = cfg or {}
    tool_cfg = cfg.get("tools", {})

    if not tool_cfg.get("github_tools", True):
        return

    # Set token jika ada di config
    token = tool_cfg.get("github_token", "") or os.environ.get("GITHUB_TOKEN", "")
    if token:
        set_github_token(token)

    for cls in GITHUB_TOOLS:
        tool = cls()
        registry._register(tool)
