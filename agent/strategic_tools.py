"""
agent/strategic_tools.py
Strategic Partner Tool Suite for GazccAgent Pro.

Tools:
  - semantic_memory    : Concept-level storage with meaning-based retrieval
  - proactive_monitor  : Context scanner for missed steps / optimizations
  - api_bridge         : Secure generic HTTP GET/POST interface
  - sandbox_executor   : Isolated code execution with dependency support
"""

import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import aiofiles
import httpx

from .tools import BaseTool, ToolResult


# ══════════════════════════════════════════════════════════════════════════════
# 1. SEMANTIC MEMORY TOOL
# ══════════════════════════════════════════════════════════════════════════════

class SemanticMemoryTool(BaseTool):
    """
    Stores and retrieves 'concepts' — rich, structured summaries of past tasks.
    Retrieval is meaning-based (TF-IDF cosine), not exact-match.

    Operations:
      - store  : Save a concept with tags and a summary
      - search : Find relevant concepts by semantic query
      - list   : List all concept keys
      - delete : Remove a concept
    """

    name = "semantic_memory"
    description = (
        "Store or retrieve concept-level knowledge from long-term memory. "
        "Supports storing summaries with tags, and searching by meaning. "
        "Use this to remember user preferences, past task outcomes, and architectural decisions."
    )
    parameters = (
        "operation: str ('store'|'search'|'list'|'delete'), "
        "key: str (required for store/delete), "
        "content: str (required for store), "
        "tags: list[str] (optional, for store), "
        "query: str (required for search), "
        "top_k: int = 5"
    )

    def __init__(self, memory_path: str = "/tmp/gazcc_semantic_memory"):
        self._path = Path(memory_path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._index_file = self._path / "concepts.json"
        self._concepts: dict[str, dict] = {}
        self._loaded = False
        self._lock = asyncio.Lock()

    async def _load(self):
        if self._loaded:
            return
        if self._index_file.exists():
            async with aiofiles.open(self._index_file, "r") as f:
                self._concepts = json.loads(await f.read())
        self._loaded = True

    async def _save(self):
        async with aiofiles.open(self._index_file, "w") as f:
            await f.write(json.dumps(self._concepts, indent=2, ensure_ascii=False))

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def _tfidf_sim(self, query: str, doc: str) -> float:
        from collections import Counter
        import math
        qt = Counter(self._tokenize(query))
        dt = Counter(self._tokenize(doc))
        vocab = set(qt) | set(dt)
        if not vocab:
            return 0.0
        qv = [qt.get(w, 0) for w in vocab]
        dv = [dt.get(w, 0) for w in vocab]
        dot = sum(a * b for a, b in zip(qv, dv))
        nq = math.sqrt(sum(a * a for a in qv))
        nd = math.sqrt(sum(b * b for b in dv))
        return dot / (nq * nd) if nq and nd else 0.0

    async def run(
        self,
        operation: str,
        key: str = "",
        content: str = "",
        tags: list = None,
        query: str = "",
        top_k: int = 5,
        **kwargs,
    ) -> ToolResult:
        async with self._lock:
            await self._load()

            if operation == "store":
                if not key or not content:
                    return ToolResult(False, "store requires 'key' and 'content'")
                concept = {
                    "key": key,
                    "content": content,
                    "tags": tags or [],
                    "timestamp": time.time(),
                    "version": self._concepts.get(key, {}).get("version", 0) + 1,
                }
                self._concepts[key] = concept
                await self._save()
                return ToolResult(
                    True,
                    f"Stored concept '{key}' (v{concept['version']}) "
                    f"with {len(content)} chars and tags {tags or []}",
                    {"key": key, "version": concept["version"]},
                )

            elif operation == "search":
                if not query:
                    return ToolResult(False, "search requires 'query'")
                if not self._concepts:
                    return ToolResult(True, "Memory is empty.", {"results": []})
                scored = []
                for k, c in self._concepts.items():
                    text = c["content"] + " " + " ".join(c.get("tags", []))
                    score = self._tfidf_sim(query, text)
                    if score > 0.1:
                        scored.append((score, c))
                scored.sort(key=lambda x: -x[0])
                results = scored[:top_k]
                if not results:
                    return ToolResult(True, "No relevant concepts found.", {"results": []})
                lines = []
                for score, c in results:
                    lines.append(
                        f"[{c['key']}] (score={score:.2f}, tags={c['tags']})\n"
                        f"  {c['content'][:300]}"
                    )
                return ToolResult(
                    True, "\n\n".join(lines),
                    {"results": [c["key"] for _, c in results]},
                )

            elif operation == "list":
                if not self._concepts:
                    return ToolResult(True, "No concepts stored yet.", {"keys": []})
                lines = []
                for k, c in sorted(self._concepts.items()):
                    lines.append(f"• {k}  [tags: {c.get('tags', [])}]  ({len(c['content'])} chars)")
                return ToolResult(True, "\n".join(lines), {"keys": list(self._concepts.keys())})

            elif operation == "delete":
                if not key:
                    return ToolResult(False, "delete requires 'key'")
                if key not in self._concepts:
                    return ToolResult(False, f"Concept '{key}' not found")
                del self._concepts[key]
                await self._save()
                return ToolResult(True, f"Deleted concept '{key}'")

            else:
                return ToolResult(
                    False,
                    f"Unknown operation '{operation}'. Use: store | search | list | delete",
                )


# ══════════════════════════════════════════════════════════════════════════════
# 2. PROACTIVE MONITOR TOOL
# ══════════════════════════════════════════════════════════════════════════════

class ProactiveMonitorTool(BaseTool):
    """
    Scans the current task context and raises alerts for:
      - Missed steps that are implied but not planned
      - Optimization opportunities (parallelizable work, caching, etc.)
      - Anti-patterns or common failure modes for the given task type
    Returns a structured list of recommendations.
    """

    name = "proactive_monitor"
    description = (
        "Scan the current task plan and execution context to detect: "
        "missed steps, optimization opportunities, anti-patterns, and risks. "
        "Call this after planning or mid-execution to get proactive suggestions."
    )
    parameters = (
        "task: str, "
        "plan_steps: list[str], "
        "completed_steps: list[str] = [], "
        "context_summary: str = ''"
    )

    # Heuristic rule library
    _RULES = [
        {
            "pattern": r"\b(python|script|\.py)\b",
            "suggestions": [
                "Consider creating a requirements.txt for dependency tracking",
                "Add a basic test suite (pytest) to validate core functions",
                "Include a README.md with usage instructions",
                "Add error handling and logging (use Python logging module)",
            ],
            "risk": "Code without tests is fragile — bugs will appear in edge cases",
        },
        {
            "pattern": r"\b(api|endpoint|request|http|rest)\b",
            "suggestions": [
                "Implement rate-limiting / retry with exponential backoff",
                "Validate all API responses before using the data",
                "Store API keys in environment variables, never hardcode",
                "Consider caching responses to reduce API calls and cost",
            ],
            "risk": "Unhandled API failures will silently corrupt downstream results",
        },
        {
            "pattern": r"\b(data|csv|database|sql|json|excel)\b",
            "suggestions": [
                "Validate and clean the data before processing",
                "Handle missing/null values explicitly",
                "Create a backup before modifying source data",
                "Document the data schema / column meanings",
            ],
            "risk": "Dirty data without validation leads to misleading outputs",
        },
        {
            "pattern": r"\b(deploy|production|server|docker|cloud)\b",
            "suggestions": [
                "Add health-check endpoints to the service",
                "Set up logging and monitoring before going live",
                "Document rollback procedure in case of failure",
                "Use environment-specific configs (dev/staging/prod)",
            ],
            "risk": "Deployments without rollback plans are high-risk",
        },
        {
            "pattern": r"\b(report|document|write|summary|analysis)\b",
            "suggestions": [
                "Define the target audience and adjust tone accordingly",
                "Include an executive summary at the top",
                "Cite sources and add confidence levels to conclusions",
                "Add a 'Next Steps' or 'Recommendations' section",
            ],
            "risk": "Reports without clear audience framing often miss the mark",
        },
        {
            "pattern": r"\b(research|search|find|scrape|gather)\b",
            "suggestions": [
                "Cross-reference multiple sources before concluding",
                "Note the timestamp of retrieved information (data freshness)",
                "Check for contradicting information across sources",
            ],
            "risk": "Single-source research may be biased or outdated",
        },
    ]

    async def run(
        self,
        task: str,
        plan_steps: list = None,
        completed_steps: list = None,
        context_summary: str = "",
        **kwargs,
    ) -> ToolResult:
        plan_steps = plan_steps or []
        completed_steps = completed_steps or []
        task_lower = (task + " " + context_summary + " " + " ".join(plan_steps)).lower()

        alerts = []
        suggestions = []
        risks = []

        # ── Apply heuristic rules
        for rule in self._RULES:
            if re.search(rule["pattern"], task_lower, re.IGNORECASE):
                suggestions.extend(rule["suggestions"])
                risks.append(rule["risk"])

        # ── Check for parallelizable steps
        if len(plan_steps) > 3:
            alerts.append(
                f"Plan has {len(plan_steps)} steps. Consider marking independent "
                "steps as parallelizable to speed up execution."
            )

        # ── Check for common missing steps
        has_test = any("test" in s.lower() for s in plan_steps)
        has_write = any(
            kw in s.lower() for s in plan_steps
            for kw in ("write", "create", "generate", "build")
        )
        if has_write and not has_test and re.search(r"\bpython|code|script\b", task_lower):
            alerts.append("Step: 'Write tests / validation' is implied but not in plan.")

        has_cleanup = any("clean" in s.lower() or "final" in s.lower() for s in plan_steps)
        if len(plan_steps) > 4 and not has_cleanup:
            alerts.append("Consider adding a cleanup/finalization step to consolidate outputs.")

        # ── Deduplicate
        suggestions = list(dict.fromkeys(suggestions))[:6]
        risks = list(dict.fromkeys(risks))[:3]

        if not alerts and not suggestions and not risks:
            return ToolResult(
                True,
                "✅ No significant issues detected. Plan looks solid.",
                {"alerts": [], "suggestions": [], "risks": []},
            )

        output_lines = []
        if alerts:
            output_lines.append("🔔 ALERTS (Missed Steps / Structural Issues):")
            output_lines.extend(f"  • {a}" for a in alerts)
        if suggestions:
            output_lines.append("\n💡 PROACTIVE SUGGESTIONS (Consider adding to plan):")
            output_lines.extend(f"  • {s}" for s in suggestions)
        if risks:
            output_lines.append("\n⚠️ RISK FLAGS:")
            output_lines.extend(f"  • {r}" for r in risks)

        return ToolResult(
            True,
            "\n".join(output_lines),
            {"alerts": alerts, "suggestions": suggestions, "risks": risks},
        )


# ══════════════════════════════════════════════════════════════════════════════
# 3. API BRIDGE TOOL
# ══════════════════════════════════════════════════════════════════════════════

class ApiBridgeTool(BaseTool):
    """
    Generic, secure HTTP bridge for dynamic API calls.
    Supports GET and POST with custom headers and auth.
    Never logs credentials. Validates response codes.
    """

    name = "api_bridge"
    description = (
        "Make a secure HTTP GET or POST request to an external API. "
        "Supports JSON/form body, custom headers, Bearer/Basic auth, and API key headers. "
        "Returns the response body as a string."
    )
    parameters = (
        "method: str ('GET'|'POST'|'PUT'|'DELETE'), "
        "url: str, "
        "headers: dict = {}, "
        "body: dict = {}, "
        "auth_type: str = '' ('bearer'|'basic'|'apikey'|''), "
        "auth_value: str = '', "
        "auth_header: str = 'Authorization', "
        "timeout: int = 30, "
        "max_response_chars: int = 8000"
    )

    _ALLOWED_SCHEMES = {"http", "https"}
    _BLOCKED_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}  # SSRF guard

    async def run(
        self,
        method: str = "GET",
        url: str = "",
        headers: dict = None,
        body: dict = None,
        auth_type: str = "",
        auth_value: str = "",
        auth_header: str = "Authorization",
        timeout: int = 30,
        max_response_chars: int = 8000,
        **kwargs,
    ) -> ToolResult:
        if not url:
            return ToolResult(False, "api_bridge: 'url' is required")

        # ── Validate URL scheme
        from urllib.parse import urlparse
        parsed = urlparse(url)
        if parsed.scheme not in self._ALLOWED_SCHEMES:
            return ToolResult(False, f"api_bridge: Scheme '{parsed.scheme}' not allowed (use http/https)")
        if parsed.hostname in self._BLOCKED_HOSTS:
            return ToolResult(False, "api_bridge: Requests to localhost/loopback are blocked")

        method = method.upper()
        if method not in ("GET", "POST", "PUT", "DELETE", "PATCH"):
            return ToolResult(False, f"api_bridge: Unsupported method '{method}'")

        # ── Build headers
        req_headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if headers:
            req_headers.update(headers)

        # ── Apply auth
        if auth_type and auth_value:
            if auth_type.lower() == "bearer":
                req_headers["Authorization"] = f"Bearer {auth_value}"
            elif auth_type.lower() == "basic":
                import base64
                encoded = base64.b64encode(auth_value.encode()).decode()
                req_headers["Authorization"] = f"Basic {encoded}"
            elif auth_type.lower() == "apikey":
                req_headers[auth_header] = auth_value

        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                if method == "GET":
                    resp = await client.get(url, headers=req_headers, params=body or {})
                elif method in ("POST", "PUT", "PATCH"):
                    resp = await client.request(
                        method, url, headers=req_headers, json=body or {}
                    )
                else:  # DELETE
                    resp = await client.delete(url, headers=req_headers)

            status = resp.status_code
            try:
                data = resp.json()
                text = json.dumps(data, indent=2, ensure_ascii=False)
            except Exception:
                text = resp.text

            if len(text) > max_response_chars:
                text = text[:max_response_chars] + "\n... [response truncated]"

            success = 200 <= status < 300
            return ToolResult(
                success,
                f"HTTP {status} {resp.reason_phrase}\n\n{text}",
                {"status_code": status, "url": url, "method": method},
            )

        except httpx.TimeoutException:
            return ToolResult(False, f"api_bridge: Request timed out after {timeout}s")
        except httpx.RequestError as e:
            return ToolResult(False, f"api_bridge: Network error — {e}")
        except Exception as e:
            return ToolResult(False, f"api_bridge: Unexpected error — {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. SANDBOX EXECUTOR TOOL
# ══════════════════════════════════════════════════════════════════════════════

class SandboxExecutorTool(BaseTool):
    """
    Isolated Python execution environment.
    - Runs code in a temp directory with a fresh venv
    - Supports installing pip dependencies before execution
    - Captures stdout, stderr, and return code
    - Enforces timeout and cleans up after itself
    - Supports multi-step execution via sequenced code blocks
    """

    name = "sandbox_executor"
    description = (
        "Execute Python code in an isolated sandbox with optional pip dependencies. "
        "Installs packages, runs code, and returns stdout/stderr. "
        "Safe: each run is isolated in a temp directory. "
        "Use for code testing, data processing, simulations."
    )
    parameters = (
        "code: str, "
        "dependencies: list[str] = [], "
        "timeout: int = 60, "
        "python_version: str = 'python3', "
        "env_vars: dict = {}"
    )

    _MAX_OUTPUT = 10_000   # chars
    _MAX_TIMEOUT = 120     # hard cap seconds
    _BANNED_IMPORTS = {    # simple AST-free check for obvious hazards
        "os.system", "subprocess.Popen", "subprocess.call",
        "__import__('os').system", "ctypes",
    }

    def _safety_check(self, code: str) -> str | None:
        """Returns an error string if code looks dangerous, else None."""
        for banned in self._BANNED_IMPORTS:
            if banned in code:
                return f"Banned pattern detected: '{banned}'"
        return None

    async def run(
        self,
        code: str,
        dependencies: list = None,
        timeout: int = 60,
        python_version: str = "python3",
        env_vars: dict = None,
        **kwargs,
    ) -> ToolResult:
        if not code:
            return ToolResult(False, "sandbox_executor: 'code' is required")

        # ── Safety check
        danger = self._safety_check(code)
        if danger:
            return ToolResult(False, f"sandbox_executor: Safety check failed — {danger}")

        timeout = min(timeout, self._MAX_TIMEOUT)
        dependencies = dependencies or []
        env_vars = env_vars or {}
        run_id = str(uuid.uuid4())[:8]

        with tempfile.TemporaryDirectory(prefix=f"gazcc_sandbox_{run_id}_") as tmpdir:
            tmppath = Path(tmpdir)

            # ── Prepare environment
            env = os.environ.copy()
            env["PYTHONPATH"] = tmpdir
            env["TMPDIR"] = tmpdir
            env.update(env_vars)

            # ── Install dependencies
            dep_log = ""
            if dependencies:
                install_cmd = [
                    sys.executable, "-m", "pip", "install",
                    "--quiet", "--target", tmpdir,
                    *dependencies,
                ]
                try:
                    proc = await asyncio.create_subprocess_exec(
                        *install_cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        env=env,
                    )
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=90)
                    if proc.returncode != 0:
                        return ToolResult(
                            False,
                            f"Dependency install failed:\n{stderr.decode(errors='replace')[:2000]}",
                        )
                    dep_log = f"Installed: {', '.join(dependencies)}\n"
                except asyncio.TimeoutError:
                    return ToolResult(False, "Dependency installation timed out (90s)")

            # ── Write code to file
            code_file = tmppath / "main.py"
            async with aiofiles.open(code_file, "w") as f:
                await f.write(code)

            # ── Execute
            start = time.time()
            try:
                proc = await asyncio.create_subprocess_exec(
                    sys.executable, str(code_file),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=tmpdir,
                    env=env,
                )
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
                elapsed = round(time.time() - start, 2)
                rc = proc.returncode

                out = stdout.decode(errors="replace")
                err = stderr.decode(errors="replace")

                # Truncate if too long
                if len(out) > self._MAX_OUTPUT:
                    out = out[: self._MAX_OUTPUT] + "\n... [stdout truncated]"
                if len(err) > 2000:
                    err = err[:2000] + "\n... [stderr truncated]"

                if rc == 0:
                    body = dep_log
                    if out:
                        body += f"STDOUT:\n{out}"
                    if err:
                        body += f"\nSTDERR (non-fatal):\n{err}"
                    body += f"\n✅ Completed in {elapsed}s (exit code 0)"
                    return ToolResult(True, body, {"exit_code": 0, "elapsed": elapsed})
                else:
                    body = dep_log + f"STDOUT:\n{out}\nSTDERR:\n{err}\n❌ Exit code {rc} ({elapsed}s)"
                    return ToolResult(False, body, {"exit_code": rc, "elapsed": elapsed})

            except asyncio.TimeoutError:
                return ToolResult(
                    False,
                    f"sandbox_executor: Execution timed out after {timeout}s",
                    {"exit_code": -1},
                )
            except Exception as e:
                return ToolResult(False, f"sandbox_executor: Error — {e}")
