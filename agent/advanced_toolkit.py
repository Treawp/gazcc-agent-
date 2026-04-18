"""
GazccThinking Advanced Toolkit
Kemampuan: execute, filesystem, web search, unit test, git
"""

import os
import subprocess
import json
import shutil
import tempfile
from pathlib import Path
from typing import Optional
import urllib.request
import urllib.parse


# ─── CONFIG ──────────────────────────────────────────────────────────────────
SANDBOX_ROOT   = os.environ.get("SANDBOX_ROOT", "/tmp/gazcc_sandbox")
EXEC_TIMEOUT   = int(os.environ.get("EXEC_TIMEOUT", "15"))       # detik
SEARCH_API_KEY = os.environ.get("SERPER_API_KEY", "")            # opsional
MAX_OUTPUT     = 8000                                              # chars

os.makedirs(SANDBOX_ROOT, exist_ok=True)


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def _safe_path(path: str) -> str:
    """Pastikan path tidak keluar dari SANDBOX_ROOT."""
    resolved = os.path.realpath(os.path.join(SANDBOX_ROOT, path.lstrip("/")))
    if not resolved.startswith(os.path.realpath(SANDBOX_ROOT)):
        raise PermissionError(f"Path escape blocked: {path}")
    return resolved

def _truncate(text: str, limit: int = MAX_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + f"\n\n...[TRUNCATED {len(text)-limit} chars]...\n\n" + text[-half:]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. EXECUTE COMMAND
# ═══════════════════════════════════════════════════════════════════════════════

def execute_command(
    command: str,
    working_dir: str = ".",
    timeout: int = EXEC_TIMEOUT,
    env_vars: dict = None
) -> dict:
    """
    Jalankan perintah shell di dalam sandbox.
    Mendukung: python script.py, node app.js, ls, pip install, dsb.

    Args:
        command    : Perintah shell (string).
        working_dir: Direktori kerja relatif terhadap SANDBOX_ROOT.
        timeout    : Batas waktu eksekusi dalam detik (max 30).
        env_vars   : Dict environment variable tambahan.

    Returns:
        dict: stdout, stderr, exit_code, duration_ms, success
    """
    import time

    timeout = min(timeout, 30)  # hard cap
    cwd = _safe_path(working_dir)
    os.makedirs(cwd, exist_ok=True)

    env = os.environ.copy()
    env["HOME"] = SANDBOX_ROOT
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if env_vars:
        env.update(env_vars)

    # Blokir perintah berbahaya
    BLOCKED = ["rm -rf /", "mkfs", "dd if=", ":(){:|:&};:", "shutdown", "reboot", "curl | sh", "wget | sh"]
    for bad in BLOCKED:
        if bad in command:
            return {"success": False, "error": f"Blocked command pattern: '{bad}'",
                    "stdout": "", "stderr": "", "exit_code": -1, "duration_ms": 0}

    start = time.time()
    try:
        result = subprocess.run(
            command, shell=True, cwd=cwd, env=env,
            capture_output=True, text=True, timeout=timeout
        )
        duration = int((time.time() - start) * 1000)
        return {
            "success":     result.returncode == 0,
            "exit_code":   result.returncode,
            "stdout":      _truncate(result.stdout),
            "stderr":      _truncate(result.stderr),
            "duration_ms": duration,
            "command":     command,
            "cwd":         cwd
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Timeout after {timeout}s",
                "stdout": "", "stderr": "", "exit_code": -1, "duration_ms": timeout*1000}
    except Exception as e:
        return {"success": False, "error": str(e),
                "stdout": "", "stderr": "", "exit_code": -1, "duration_ms": 0}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. FILE SYSTEM MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

def file_system_manager(
    action: str,
    path: str,
    content: str = None,
    recursive: bool = True,
    max_depth: int = 5,
    extensions: list = None
) -> dict:
    """
    Kelola filesystem skala proyek: tree, baca, tulis, mkdir, delete, copy.

    Args:
        action    : 'tree' | 'read' | 'write' | 'mkdir' | 'delete' | 'copy' | 'search'
        path      : Path target (relatif terhadap SANDBOX_ROOT).
        content   : Untuk 'write' — konten file. Untuk 'copy' — path tujuan.
        recursive : Untuk 'tree' — apakah rekursif.
        max_depth : Untuk 'tree' — kedalaman maksimal (max 8).
        extensions: Untuk 'tree'/'search' — filter ekstensi, contoh ['.py', '.js'].

    Returns:
        dict dengan hasil sesuai action.
    """
    action = action.lower().strip()
    max_depth = min(max_depth, 8)

    if action == "tree":
        target = _safe_path(path)
        if not os.path.exists(target):
            return {"error": f"Path tidak ditemukan: {path}"}

        def _tree(p: Path, depth: int, prefix: str = "") -> list:
            if depth > max_depth:
                return [prefix + "  ... (max depth reached)"]
            lines = []
            try:
                items = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name))
            except PermissionError:
                return [prefix + "  [permission denied]"]
            for i, item in enumerate(items):
                is_last = (i == len(items) - 1)
                connector = "└── " if is_last else "├── "
                ext = item.suffix.lower()
                if extensions and item.is_file() and ext not in extensions:
                    continue
                size = f" ({item.stat().st_size:,} B)" if item.is_file() else ""
                lines.append(prefix + connector + item.name + size)
                if item.is_dir() and recursive:
                    ext_prefix = prefix + ("    " if is_last else "│   ")
                    lines.extend(_tree(item, depth + 1, ext_prefix))
            return lines

        p = Path(target)
        tree_lines = [p.name + "/"] + _tree(p, 1)
        flat = []
        for root, dirs, files in os.walk(target):
            for f in files:
                full = os.path.join(root, f)
                rel  = os.path.relpath(full, target)
                if not extensions or Path(f).suffix.lower() in extensions:
                    flat.append(rel)
        return {"action": "tree", "path": path, "tree": "\n".join(tree_lines),
                "flat": flat, "file_count": len(flat)}

    elif action == "read":
        target = _safe_path(path)
        if not os.path.isfile(target):
            return {"error": f"File tidak ditemukan: {path}"}
        size = os.path.getsize(target)
        try:
            with open(target, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            import base64
            with open(target, "rb") as f:
                text = base64.b64encode(f.read()).decode()
            return {"action": "read", "path": path, "content": text,
                    "encoding": "base64", "size_bytes": size}
        return {"action": "read", "path": path, "content": _truncate(text),
                "encoding": "utf-8", "size_bytes": size, "lines": text.count("\n")+1}

    elif action == "write":
        if content is None:
            return {"error": "content wajib diisi untuk action 'write'"}
        target = _safe_path(path)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(content)
        return {"action": "write", "path": path, "success": True,
                "bytes_written": len(content.encode("utf-8"))}

    elif action == "mkdir":
        target = _safe_path(path)
        os.makedirs(target, exist_ok=True)
        return {"action": "mkdir", "path": path, "success": True}

    elif action == "delete":
        target = _safe_path(path)
        if not os.path.exists(target):
            return {"error": f"Tidak ditemukan: {path}"}
        if os.path.isdir(target):
            shutil.rmtree(target)
        else:
            os.remove(target)
        return {"action": "delete", "path": path, "success": True}

    elif action == "copy":
        if not content:
            return {"error": "Untuk 'copy', isi 'content' dengan path tujuan"}
        src = _safe_path(path)
        dst = _safe_path(content)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        return {"action": "copy", "src": path, "dst": content, "success": True}

    elif action == "search":
        target = _safe_path(path)
        keyword = content or ""
        results = []
        for root, _, files in os.walk(target):
            for fname in files:
                fp = os.path.join(root, fname)
                ext = Path(fname).suffix.lower()
                if extensions and ext not in extensions:
                    continue
                try:
                    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                        for i, line in enumerate(f, 1):
                            if keyword.lower() in line.lower():
                                results.append({
                                    "file": os.path.relpath(fp, target),
                                    "line": i,
                                    "match": line.strip()
                                })
                except Exception:
                    pass
        return {"action": "search", "keyword": keyword, "results": results[:100],
                "total_matches": len(results)}

    return {"error": f"Unknown action: {action}. Gunakan: tree|read|write|mkdir|delete|copy|search"}


# ═══════════════════════════════════════════════════════════════════════════════
# 3. WEB SEARCH DOCUMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

def web_search_documentation(
    query: str,
    search_type: str = "docs",
    max_results: int = 5
) -> dict:
    """
    Cari dokumentasi API / library terbaru di internet.
    Gunakan Serper.dev (jika SERPER_API_KEY tersedia) atau DuckDuckGo fallback.

    Args:
        query       : Query pencarian, contoh "FastAPI WebSocket docs 2024".
        search_type : 'docs' | 'general' | 'stackoverflow' | 'github'
        max_results : Jumlah hasil (max 10).

    Returns:
        dict dengan list hasil: title, url, snippet.
    """
    max_results = min(max_results, 10)

    # Modifikasi query berdasarkan tipe
    prefixes = {
        "docs":         "site:docs. OR documentation ",
        "stackoverflow": "site:stackoverflow.com ",
        "github":        "site:github.com ",
        "general":       ""
    }
    full_query = prefixes.get(search_type, "") + query

    # ── Serper.dev (akurat, butuh API key) ────────────────────────────────────
    if SEARCH_API_KEY:
        try:
            payload = json.dumps({"q": full_query, "num": max_results}).encode()
            req = urllib.request.Request(
                "https://google.serper.dev/search",
                data=payload,
                headers={"X-API-KEY": SEARCH_API_KEY, "Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                data = json.loads(r.read())
            results = []
            for item in data.get("organic", [])[:max_results]:
                results.append({
                    "title":   item.get("title", ""),
                    "url":     item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "source":  "serper"
                })
            return {"query": query, "results": results, "engine": "serper"}
        except Exception as e:
            pass  # fallback ke DDG

    # ── DuckDuckGo Instant Answer API (fallback, no key needed) ───────────────
    try:
        encoded = urllib.parse.quote_plus(full_query)
        url = f"https://api.duckduckgo.com/?q={encoded}&format=json&no_redirect=1&no_html=1"
        req = urllib.request.Request(url, headers={"User-Agent": "GazccAgent/1.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())

        results = []
        # Abstract
        if data.get("AbstractText"):
            results.append({
                "title":   data.get("Heading", query),
                "url":     data.get("AbstractURL", ""),
                "snippet": data["AbstractText"][:500],
                "source":  "ddg_abstract"
            })
        # Related topics
        for topic in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append({
                    "title":   topic.get("Text", "")[:80],
                    "url":     topic.get("FirstURL", ""),
                    "snippet": topic.get("Text", "")[:300],
                    "source":  "ddg_related"
                })
        return {"query": query, "results": results[:max_results], "engine": "duckduckgo"}
    except Exception as e:
        return {"query": query, "results": [], "error": str(e),
                "tip": "Set SERPER_API_KEY untuk hasil yang lebih baik"}


# ═══════════════════════════════════════════════════════════════════════════════
# 4. UNIT TEST RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def unit_test_runner(
    test_path: str = ".",
    framework: str = "pytest",
    pattern: str = "test_*.py",
    coverage: bool = False,
    specific_test: str = None
) -> dict:
    """
    Jalankan unit test dan kembalikan hasil terstruktur.

    Args:
        test_path    : Path direktori atau file test (relatif ke SANDBOX_ROOT).
        framework    : 'pytest' | 'unittest' | 'jest' | 'mocha'
        pattern      : Glob pattern file test untuk pytest.
        coverage     : Aktifkan coverage report.
        specific_test: Jalankan test spesifik, contoh "test_auth.py::test_login"

    Returns:
        dict: passed, failed, errors, output, summary
    """
    target = _safe_path(test_path)

    if framework == "pytest":
        args = ["python", "-m", "pytest", target, "-v", "--tb=short", "--no-header",
                f"--collect-only=no", f"--pattern={pattern}"]
        if coverage:
            args += ["--cov=.", "--cov-report=term-missing"]
        if specific_test:
            args = ["python", "-m", "pytest", _safe_path(specific_test), "-v", "--tb=long"]
        cmd = " ".join(args)

    elif framework == "unittest":
        cmd = f"python -m unittest discover -s {target} -p '{pattern}' -v"

    elif framework == "jest":
        cmd = f"npx jest {target} --no-coverage --verbose 2>&1"
        if coverage:
            cmd = f"npx jest {target} --coverage --verbose 2>&1"

    elif framework == "mocha":
        cmd = f"npx mocha '{os.path.join(target, pattern)}' --reporter spec 2>&1"

    else:
        return {"error": f"Framework tidak dikenal: {framework}. Gunakan: pytest|unittest|jest|mocha"}

    result = execute_command(cmd, working_dir=test_path)
    output = result.get("stdout", "") + result.get("stderr", "")

    # Parse hasil pytest
    passed = failed = errors = skipped = 0
    if framework == "pytest":
        import re
        m = re.search(r"(\d+) passed", output)
        if m: passed = int(m.group(1))
        m = re.search(r"(\d+) failed", output)
        if m: failed = int(m.group(1))
        m = re.search(r"(\d+) error", output)
        if m: errors = int(m.group(1))
        m = re.search(r"(\d+) skipped", output)
        if m: skipped = int(m.group(1))

    return {
        "framework":  framework,
        "test_path":  test_path,
        "success":    result.get("exit_code", 1) == 0,
        "passed":     passed,
        "failed":     failed,
        "errors":     errors,
        "skipped":    skipped,
        "output":     _truncate(output),
        "exit_code":  result.get("exit_code"),
        "summary":    f"✅ {passed} passed | ❌ {failed} failed | ⚠️ {errors} errors | ⏭ {skipped} skipped"
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. GIT MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

def git_manager(
    action: str,
    repo_path: str = ".",
    message: str = None,
    branch: str = None,
    files: list = None,
    remote: str = "origin"
) -> dict:
    """
    Kelola version control Git layaknya developer profesional.

    Args:
        action    : 'status' | 'diff' | 'log' | 'commit' | 'branch' |
                    'checkout' | 'add' | 'init' | 'clone' | 'stash'
        repo_path : Path repo (relatif ke SANDBOX_ROOT).
        message   : Commit message (untuk 'commit').
        branch    : Nama branch (untuk 'branch' / 'checkout').
        files     : List file untuk 'add' (None = semua).
        remote    : Remote name (default 'origin').

    Returns:
        dict hasil operasi Git.
    """
    action = action.lower().strip()
    cwd    = _safe_path(repo_path)

    def _run(cmd):
        return execute_command(cmd, working_dir=repo_path, timeout=20)

    if action == "init":
        r = _run("git init && git config user.email 'agent@gazcc.ai' && git config user.name 'GazccAgent'")
        return {"action": "init", "success": r["success"], "output": r["stdout"] + r["stderr"]}

    elif action == "status":
        r = _run("git status --short && echo '---' && git branch --show-current")
        staged_r   = _run("git diff --cached --stat")
        unstaged_r = _run("git diff --stat")
        return {
            "action":   "status",
            "output":   r["stdout"],
            "staged":   staged_r["stdout"],
            "unstaged": unstaged_r["stdout"],
            "success":  r["success"]
        }

    elif action == "diff":
        target_files = " ".join(files) if files else ""
        r = _run(f"git diff {target_files}" if target_files else "git diff")
        staged = _run(f"git diff --cached {target_files}" if target_files else "git diff --cached")
        return {
            "action":  "diff",
            "unstaged": _truncate(r["stdout"]),
            "staged":   _truncate(staged["stdout"]),
            "success":  True
        }

    elif action == "log":
        r = _run("git log --oneline --graph --decorate -20")
        return {"action": "log", "output": r["stdout"], "success": r["success"]}

    elif action == "add":
        paths = " ".join(files) if files else "."
        r = _run(f"git add {paths}")
        return {"action": "add", "files": files or ["."], "success": r["success"],
                "output": r["stdout"] + r["stderr"]}

    elif action == "commit":
        if not message:
            return {"error": "commit message wajib diisi"}
        # Auto-add semua perubahan jika belum di-stage
        _run("git add -A")
        r = _run(f'git commit -m "{message}"')
        return {"action": "commit", "message": message, "success": r["success"],
                "output": r["stdout"] + r["stderr"]}

    elif action == "branch":
        if branch:
            r = _run(f"git branch {branch}")
            return {"action": "branch", "created": branch, "success": r["success"],
                    "output": r["stdout"] + r["stderr"]}
        else:
            r = _run("git branch -a")
            return {"action": "branch", "output": r["stdout"], "success": r["success"]}

    elif action == "checkout":
        if not branch:
            return {"error": "nama branch wajib diisi untuk checkout"}
        r = _run(f"git checkout {branch}")
        if not r["success"]:
            # Coba buat branch baru
            r = _run(f"git checkout -b {branch}")
        return {"action": "checkout", "branch": branch, "success": r["success"],
                "output": r["stdout"] + r["stderr"]}

    elif action == "stash":
        r = _run("git stash")
        return {"action": "stash", "success": r["success"], "output": r["stdout"]}

    elif action == "clone":
        if not message:  # pakai message sebagai URL
            return {"error": "isi 'message' dengan URL repo untuk clone"}
        # Hanya izinkan HTTPS GitHub/GitLab
        if not (message.startswith("https://github.com") or message.startswith("https://gitlab.com")):
            return {"error": "Hanya clone dari github.com atau gitlab.com yang diizinkan"}
        dest = _safe_path(branch or "cloned_repo")
        r = execute_command(f"git clone --depth=1 {message} {dest}", working_dir=".", timeout=30)
        return {"action": "clone", "url": message, "dest": dest,
                "success": r["success"], "output": _truncate(r["stdout"] + r["stderr"])}

    return {"error": f"Unknown action: {action}. Gunakan: status|diff|log|commit|branch|checkout|add|init|clone|stash"}


# ─── SELF-TEST ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== ADVANCED TOOLKIT SELF-TEST ===\n")

    # 1. Execute
    r = execute_command("echo 'hello from agent' && python3 -c 'print(2**10)'")
    print("execute_command:", r["stdout"].strip(), "| exit:", r["exit_code"])

    # 2. Filesystem
    file_system_manager("write", "project/main.py", content='print("gazcc")')
    file_system_manager("write", "project/test_main.py", content='def test_ok(): assert 1==1')
    r = file_system_manager("tree", "project")
    print("file_system_manager tree:", r["tree"])

    # 3. Git
    r = git_manager("init", "project")
    print("git init:", r["success"])
    r = git_manager("commit", "project", message="initial commit by GazccAgent")
    print("git commit:", r["output"].strip())

    # 4. Unit test
    r = unit_test_runner("project", framework="pytest")
    print("unit_test_runner:", r["summary"])

    print("\n=== ALL DONE ===")
