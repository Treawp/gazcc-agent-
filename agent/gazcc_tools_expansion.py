"""
agent/gazcc_tools_expansion.py
══════════════════════════════════════════════════════════════════════════════
  GAZCC TOOLS EXPANSION — 15 NEW TOOLS
  Drop-in addition to agent/tools.py. Import and register via ToolRegistry.

  INSTALL:
    from agent.gazcc_tools_expansion import register_expansion_tools
    register_expansion_tools(tool_registry, cfg)

  TOOLS:
    #01  diff_patch           — Unified diff + patch apply
    #02  regex_extract        — Regex pattern extraction with named groups
    #03  text_chunker         — Token-safe text splitting with overlap
    #04  env_manager          — .env file read/write/validate/mask
    #05  config_validator     — YAML/JSON schema validation
    #06  code_linter          — Static analysis: Python/Lua/JS/Bash
    #07  json_query           — JMESPath query over JSON data
    #08  template_render      — Jinja2 template rendering
    #09  task_queue           — In-memory FIFO task queue
    #10  notification_broadcast — Webhook/SSE push notifier
    #11  code_translator      — LLM-powered code language translation
    #12  lua_script_analyzer  — Luau/Roblox static analyzer (Delta Executor)
    #14  dependency_resolver  — requirements.txt / package.json conflict checker
    #18  image_metadata       — EXIF + dimensions + base64 encoder
    #20  agent_inspector      — Full agent state introspection dump

  REQUIREMENTS (add to requirements.txt):
    jmespath>=1.0.1
    jinja2>=3.1.0
    pyyaml>=6.0
    pillow>=10.0.0
    pyflakes>=3.0.0
    aiofiles>=23.0.0
    httpx>=0.27.0
══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import asyncio
import base64
import difflib
import importlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import deque
from pathlib import Path
from typing import Any

import aiofiles
import httpx

# ── Import BaseTool / ToolResult from sibling module ─────────────────────────
# Adjust the import path if this file is placed outside the agent/ folder.
try:
    from .tools import BaseTool, ToolResult
except ImportError:
    # Standalone / testing mode
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
        async def run(self, *args, **kwargs) -> ToolResult:
            raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# TOOL #01 — DIFF PATCH TOOL
# Compare two texts or files and produce a unified diff.
# Apply a unified diff patch to a target file.
# ══════════════════════════════════════════════════════════════════════════════

class DiffPatchTool(BaseTool):
    """
    USAGE:
      # Get diff between two strings
      diff_patch(operation="diff", source="old text", target="new text")

      # Get diff between two files
      diff_patch(operation="diff", source_file="/a.py", target_file="/b.py")

      # Apply a patch to a file
      diff_patch(operation="patch", target_file="/a.py", patch="--- a\\n+++ b\\n...")

      # Apply patch and write result back
      diff_patch(operation="patch", target_file="/a.py", patch="...", write=True)
    """
    name        = "diff_patch"
    description = (
        "Generate a unified diff between two texts/files, or apply a unified diff "
        "patch to a target file. Enables surgical edits — no full-file overwrite needed."
    )
    parameters  = (
        "operation: str ('diff'|'patch'), "
        "source: str = '' (text, for diff), "
        "target: str = '' (text, for diff), "
        "source_file: str = '' (path, for diff), "
        "target_file: str = '' (path, for diff or patch target), "
        "patch: str = '' (unified diff string, for patch), "
        "context_lines: int = 3, "
        "write: bool = False (write patched result back to target_file)"
    )

    async def run(
        self,
        operation: str,
        source: str = "",
        target: str = "",
        source_file: str = "",
        target_file: str = "",
        patch: str = "",
        context_lines: int = 3,
        write: bool = False,
    ) -> ToolResult:
        try:
            op = operation.strip().lower()

            # ── DIFF ──────────────────────────────────────────────────────────
            if op == "diff":
                # Load from files if provided
                if source_file:
                    p = Path(source_file)
                    if not p.exists():
                        return ToolResult(False, f"source_file not found: {source_file}")
                    source = p.read_text(errors="replace")

                if target_file and not target:
                    p = Path(target_file)
                    if not p.exists():
                        return ToolResult(False, f"target_file not found: {target_file}")
                    target = p.read_text(errors="replace")

                if not source and not target:
                    return ToolResult(False, "Provide source+target text or source_file+target_file.")

                src_lines  = source.splitlines(keepends=True)
                tgt_lines  = target.splitlines(keepends=True)
                src_label  = source_file or "source"
                tgt_label  = target_file or "target"

                diff_lines = list(difflib.unified_diff(
                    src_lines, tgt_lines,
                    fromfile=src_label,
                    tofile=tgt_label,
                    n=context_lines,
                ))

                if not diff_lines:
                    return ToolResult(True, "No differences found — files are identical.", {
                        "changed": False, "hunks": 0
                    })

                diff_str   = "".join(diff_lines)
                added      = sum(1 for l in diff_lines if l.startswith("+") and not l.startswith("+++"))
                removed    = sum(1 for l in diff_lines if l.startswith("-") and not l.startswith("---"))
                hunks      = sum(1 for l in diff_lines if l.startswith("@@"))

                return ToolResult(True, diff_str, {
                    "changed": True,
                    "lines_added": added,
                    "lines_removed": removed,
                    "hunks": hunks,
                })

            # ── PATCH ─────────────────────────────────────────────────────────
            elif op == "patch":
                if not patch:
                    return ToolResult(False, "patch string is required for operation='patch'.")
                if not target_file:
                    return ToolResult(False, "target_file is required for operation='patch'.")

                p = Path(target_file)
                if not p.exists():
                    return ToolResult(False, f"target_file not found: {target_file}")

                original = p.read_text(errors="replace")
                orig_lines = original.splitlines(keepends=True)

                # Apply patch via difflib's SequenceMatcher reconstruction
                patched_lines = self._apply_unified_patch(orig_lines, patch)
                if patched_lines is None:
                    return ToolResult(False, "Patch failed: hunks did not match target content.")

                patched_str = "".join(patched_lines)

                if write:
                    p.write_text(patched_str, encoding="utf-8")
                    return ToolResult(True, f"Patch applied and written to {target_file}.", {
                        "written": True, "path": str(p)
                    })
                else:
                    return ToolResult(True, patched_str, {"written": False})

            else:
                return ToolResult(False, f"Unknown operation: '{operation}'. Use 'diff' or 'patch'.")

        except Exception as e:
            return ToolResult(False, f"diff_patch error: {e}")

    def _apply_unified_patch(self, orig_lines: list[str], patch: str) -> list[str] | None:
        """Minimal unified diff patch applicator."""
        result       = list(orig_lines)
        patch_lines  = patch.splitlines(keepends=True)
        offset       = 0

        i = 0
        while i < len(patch_lines):
            line = patch_lines[i]
            if line.startswith("@@"):
                # Parse hunk header: @@ -start,count +start,count @@
                m = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
                if not m:
                    i += 1
                    continue
                orig_start = int(m.group(1)) - 1  # 0-indexed
                i += 1
                pos = orig_start + offset
                while i < len(patch_lines) and not patch_lines[i].startswith("@@"):
                    hl = patch_lines[i]
                    if hl.startswith("-"):
                        if pos < len(result):
                            result.pop(pos)
                            offset -= 1
                    elif hl.startswith("+"):
                        result.insert(pos, hl[1:])
                        pos    += 1
                        offset += 1
                    else:
                        pos += 1
                    i += 1
            else:
                i += 1

        return result


# ══════════════════════════════════════════════════════════════════════════════
# TOOL #02 — REGEX EXTRACT TOOL
# Deterministic, hallucination-free text extraction via regex patterns.
# ══════════════════════════════════════════════════════════════════════════════

class RegexExtractTool(BaseTool):
    """
    USAGE:
      regex_extract(pattern=r"\\d{4}-\\d{2}-\\d{2}", text="Date: 2025-04-18")
      regex_extract(pattern=r"(?P<key>\\w+)=(?P<val>\\S+)", text="a=1 b=2", all=True)
      regex_extract(pattern=r"error.*", file="/var/log/app.log", flags="IGNORECASE,MULTILINE")
    """
    name        = "regex_extract"
    description = (
        "Run a regex pattern over text or a file. Returns first match or all matches "
        "with named groups. Zero-hallucination structured extraction."
    )
    parameters  = (
        "pattern: str (regex pattern), "
        "text: str = '' (input text), "
        "file: str = '' (path to file to scan), "
        "all: bool = False (return all matches vs first only), "
        "flags: str = '' (comma-separated: IGNORECASE, MULTILINE, DOTALL)"
    )

    _FLAG_MAP = {
        "IGNORECASE": re.IGNORECASE,
        "MULTILINE":  re.MULTILINE,
        "DOTALL":     re.DOTALL,
        "VERBOSE":    re.VERBOSE,
    }

    async def run(
        self,
        pattern: str,
        text: str = "",
        file: str = "",
        all: bool = False,
        flags: str = "",
    ) -> ToolResult:
        try:
            if not pattern:
                return ToolResult(False, "pattern is required.")

            # Build flags
            compiled_flags = 0
            for f in flags.split(","):
                f = f.strip().upper()
                if f and f in self._FLAG_MAP:
                    compiled_flags |= self._FLAG_MAP[f]

            try:
                rx = re.compile(pattern, compiled_flags)
            except re.error as e:
                return ToolResult(False, f"Invalid regex pattern: {e}")

            # Load source text
            if file:
                p = Path(file)
                if not p.exists():
                    return ToolResult(False, f"File not found: {file}")
                if p.stat().st_size > 5 * 1024 * 1024:
                    return ToolResult(False, "File too large (>5MB) for regex scan.")
                async with aiofiles.open(p, "r", errors="replace") as fh:
                    text = await fh.read()

            if not text:
                return ToolResult(False, "Provide text or file to scan.")

            def match_to_dict(m: re.Match) -> dict:
                return {
                    "full_match": m.group(0),
                    "groups":     list(m.groups()),
                    "named":      m.groupdict(),
                    "span":       list(m.span()),
                }

            if all:
                matches = [match_to_dict(m) for m in rx.finditer(text)]
                if not matches:
                    return ToolResult(True, "No matches found.", {"count": 0, "matches": []})
                return ToolResult(
                    True,
                    json.dumps(matches, ensure_ascii=False, indent=2),
                    {"count": len(matches), "matches": matches},
                )
            else:
                m = rx.search(text)
                if not m:
                    return ToolResult(True, "No match found.", {"matched": False})
                result = match_to_dict(m)
                return ToolResult(
                    True,
                    json.dumps(result, ensure_ascii=False, indent=2),
                    {"matched": True, **result},
                )

        except Exception as e:
            return ToolResult(False, f"regex_extract error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TOOL #03 — TEXT CHUNKER TOOL
# Split large text into token-safe chunks with configurable overlap.
# Prevents context window overflow in LLM pipelines.
# ══════════════════════════════════════════════════════════════════════════════

class TextChunkerTool(BaseTool):
    """
    USAGE:
      text_chunker(text="...", max_tokens=2000, overlap_tokens=100)
      text_chunker(file="/path/to/large.txt", max_tokens=4000, strategy="paragraph")
      text_chunker(text="...", max_tokens=1000, strategy="sentence", overlap_tokens=50)
    """
    name        = "text_chunker"
    description = (
        "Split large text into token-safe chunks with configurable overlap. "
        "Strategies: 'char' (hard split), 'sentence' (sentence boundaries), "
        "'paragraph' (paragraph boundaries). Prevents LLM context overflow."
    )
    parameters  = (
        "text: str = '' (input text), "
        "file: str = '' (path to text file), "
        "max_tokens: int = 2000 (approx tokens per chunk, 1 token ≈ 4 chars), "
        "overlap_tokens: int = 50 (token overlap between chunks), "
        "strategy: str = 'paragraph' ('char'|'sentence'|'paragraph')"
    )

    async def run(
        self,
        text: str = "",
        file: str = "",
        max_tokens: int = 2000,
        overlap_tokens: int = 50,
        strategy: str = "paragraph",
    ) -> ToolResult:
        try:
            if file:
                p = Path(file)
                if not p.exists():
                    return ToolResult(False, f"File not found: {file}")
                async with aiofiles.open(p, "r", errors="replace") as fh:
                    text = await fh.read()

            if not text:
                return ToolResult(False, "Provide text or file.")

            max_chars     = max_tokens * 4
            overlap_chars = overlap_tokens * 4
            chunks        = []

            strat = strategy.lower()

            if strat == "char":
                chunks = self._chunk_by_chars(text, max_chars, overlap_chars)
            elif strat == "sentence":
                chunks = self._chunk_by_sentence(text, max_chars, overlap_chars)
            else:  # paragraph (default)
                chunks = self._chunk_by_paragraph(text, max_chars, overlap_chars)

            result = [
                {
                    "index":      i,
                    "text":       c,
                    "char_count": len(c),
                    "est_tokens": len(c) // 4,
                }
                for i, c in enumerate(chunks)
            ]

            summary = (
                f"{len(chunks)} chunks | strategy={strat} | "
                f"max_tokens={max_tokens} | overlap={overlap_tokens}"
            )

            return ToolResult(
                True,
                json.dumps(result, ensure_ascii=False, indent=2),
                {"total_chunks": len(chunks), "strategy": strat, "summary": summary},
            )

        except Exception as e:
            return ToolResult(False, f"text_chunker error: {e}")

    def _chunk_by_chars(self, text: str, max_chars: int, overlap: int) -> list[str]:
        chunks, start = [], 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            chunks.append(text[start:end])
            start = end - overlap if end < len(text) else end
        return chunks

    def _chunk_by_sentence(self, text: str, max_chars: int, overlap: int) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return self._pack_units(sentences, max_chars, overlap)

    def _chunk_by_paragraph(self, text: str, max_chars: int, overlap: int) -> list[str]:
        paragraphs = re.split(r"\n{2,}", text)
        return self._pack_units(paragraphs, max_chars, overlap)

    def _pack_units(self, units: list[str], max_chars: int, overlap: int) -> list[str]:
        chunks, current, overlap_buf = [], [], []
        current_len = 0

        for unit in units:
            if current_len + len(unit) + 1 > max_chars and current:
                chunks.append(" ".join(current))
                # Build overlap from tail of current
                overlap_text = " ".join(current)[-overlap:] if overlap else ""
                current      = [overlap_text, unit] if overlap_text else [unit]
                current_len  = sum(len(u) for u in current)
            else:
                current.append(unit)
                current_len += len(unit) + 1

        if current:
            chunks.append(" ".join(current))

        return [c.strip() for c in chunks if c.strip()]


# ══════════════════════════════════════════════════════════════════════════════
# TOOL #04 — ENV MANAGER TOOL
# Read, write, validate, and inspect .env files safely.
# Secrets are masked in output by default.
# ══════════════════════════════════════════════════════════════════════════════

class EnvManagerTool(BaseTool):
    """
    USAGE:
      env_manager(operation="read", file=".env")
      env_manager(operation="get", file=".env", key="OPENROUTER_API_KEY")
      env_manager(operation="set", file=".env", key="NEW_VAR", value="hello")
      env_manager(operation="validate", file=".env", required=["API_KEY","REDIS_URL"])
      env_manager(operation="list_missing", file=".env", required=["A","B","C"])
    """
    name        = "env_manager"
    description = (
        "Read, write, validate, and inspect .env files. Secrets are masked in output. "
        "Operations: read, get, set, delete, validate, list_missing."
    )
    parameters  = (
        "operation: str ('read'|'get'|'set'|'delete'|'validate'|'list_missing'), "
        "file: str = '.env' (path to .env file), "
        "key: str = '' (variable name for get/set/delete), "
        "value: str = '' (value for set), "
        "required: list[str] = [] (required keys for validate/list_missing), "
        "mask_secrets: bool = True (mask values containing 'key','secret','token','pass')"
    )

    _SENSITIVE = re.compile(r"key|secret|token|pass|auth|credential|api", re.IGNORECASE)

    async def run(
        self,
        operation: str,
        file: str = ".env",
        key: str = "",
        value: str = "",
        required: list | None = None,
        mask_secrets: bool = True,
    ) -> ToolResult:
        try:
            op   = operation.strip().lower()
            path = Path(file)
            required = required or []

            # Parse existing .env (or empty dict)
            env: dict[str, str] = {}
            if path.exists():
                env = self._parse_env(path.read_text(encoding="utf-8"))

            # ── Operations ───────────────────────────────────────────────────
            if op == "read":
                display = self._display(env, mask_secrets)
                return ToolResult(True, json.dumps(display, indent=2), {
                    "total_vars": len(env), "file": str(path)
                })

            elif op == "get":
                if not key:
                    return ToolResult(False, "key is required for operation='get'.")
                if key not in env:
                    return ToolResult(False, f"Key '{key}' not found in {file}.")
                val     = env[key]
                masked  = self._mask(key, val) if mask_secrets else val
                return ToolResult(True, masked, {"key": key, "found": True})

            elif op == "set":
                if not key:
                    return ToolResult(False, "key is required for operation='set'.")
                env[key] = value
                path.write_text(self._serialize_env(env), encoding="utf-8")
                return ToolResult(True, f"Set {key} in {file}.", {"key": key, "written": True})

            elif op == "delete":
                if not key:
                    return ToolResult(False, "key is required for operation='delete'.")
                if key not in env:
                    return ToolResult(False, f"Key '{key}' not found.")
                del env[key]
                path.write_text(self._serialize_env(env), encoding="utf-8")
                return ToolResult(True, f"Deleted {key} from {file}.", {"key": key})

            elif op == "validate":
                if not required:
                    return ToolResult(False, "required list is needed for operation='validate'.")
                missing = [k for k in required if k not in env or not env[k]]
                present = [k for k in required if k in env and env[k]]
                status  = "PASS" if not missing else "FAIL"
                return ToolResult(
                    not bool(missing),
                    json.dumps({"status": status, "present": present, "missing": missing}, indent=2),
                    {"status": status, "missing_count": len(missing)},
                )

            elif op == "list_missing":
                missing = [k for k in required if k not in env]
                return ToolResult(True, json.dumps(missing, indent=2), {
                    "missing_count": len(missing)
                })

            else:
                return ToolResult(False, f"Unknown operation: '{operation}'.")

        except Exception as e:
            return ToolResult(False, f"env_manager error: {e}")

    def _parse_env(self, content: str) -> dict[str, str]:
        env = {}
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip().strip('"').strip("'")
        return env

    def _serialize_env(self, env: dict[str, str]) -> str:
        return "\n".join(f'{k}="{v}"' for k, v in env.items()) + "\n"

    def _mask(self, key: str, val: str) -> str:
        if self._SENSITIVE.search(key) and len(val) > 4:
            return val[:4] + "*" * (len(val) - 4)
        return val

    def _display(self, env: dict, mask: bool) -> dict:
        return {k: (self._mask(k, v) if mask else v) for k, v in env.items()}


# ══════════════════════════════════════════════════════════════════════════════
# TOOL #05 — CONFIG VALIDATOR TOOL
# Validate YAML or JSON config files against an expected schema dict.
# ══════════════════════════════════════════════════════════════════════════════

class ConfigValidatorTool(BaseTool):
    """
    USAGE:
      config_validator(file="config.yaml", schema='{"llm":{"model":"str"},"agent":{"max_iterations":"int"}}')
      config_validator(file="package.json", required_keys=["name","version","dependencies"])
    """
    name        = "config_validator"
    description = (
        "Validate a YAML or JSON config file. Checks required keys exist, "
        "type correctness against a schema dict, and detects common misconfigurations."
    )
    parameters  = (
        "file: str (path to config file, .yaml/.yml/.json), "
        "schema: str = '' (JSON string defining {key: type_str} schema), "
        "required_keys: list[str] = [] (keys that must exist at top level), "
        "strict: bool = False (fail on unknown keys not in schema)"
    )

    async def run(
        self,
        file: str,
        schema: str = "",
        required_keys: list | None = None,
        strict: bool = False,
    ) -> ToolResult:
        try:
            p = Path(file)
            if not p.exists():
                return ToolResult(False, f"File not found: {file}")

            content = p.read_text(encoding="utf-8")
            suffix  = p.suffix.lower()

            # Parse config
            try:
                if suffix in (".yaml", ".yml"):
                    import yaml
                    cfg = yaml.safe_load(content)
                elif suffix == ".json":
                    cfg = json.loads(content)
                else:
                    return ToolResult(False, "Unsupported format. Use .yaml, .yml, or .json.")
            except Exception as pe:
                return ToolResult(False, f"Parse error in {file}: {pe}")

            if not isinstance(cfg, dict):
                return ToolResult(False, "Config root must be a dict/object.")

            errors   = []
            warnings = []
            required_keys = required_keys or []

            # Check required keys
            for rk in required_keys:
                if rk not in cfg:
                    errors.append(f"Missing required key: '{rk}'")

            # Schema validation
            if schema:
                try:
                    schema_dict = json.loads(schema)
                except json.JSONDecodeError as e:
                    return ToolResult(False, f"Invalid schema JSON: {e}")

                for sk, expected_type in schema_dict.items():
                    if isinstance(expected_type, dict):
                        # Nested: check sub-keys exist
                        if sk not in cfg:
                            errors.append(f"Missing section: '{sk}'")
                        elif not isinstance(cfg[sk], dict):
                            errors.append(f"Section '{sk}' should be a dict.")
                        else:
                            for subk, subt in expected_type.items():
                                if subk not in cfg[sk]:
                                    warnings.append(f"Missing key: '{sk}.{subk}'")
                    else:
                        if sk in cfg:
                            self._check_type(cfg[sk], expected_type, sk, errors)

                if strict:
                    for ck in cfg:
                        if ck not in schema_dict:
                            warnings.append(f"Unknown key (strict mode): '{ck}'")

            # Common misconfigurations
            self._check_common(cfg, warnings)

            status = "PASS" if not errors else "FAIL"
            report = {
                "status":   status,
                "file":     str(p),
                "errors":   errors,
                "warnings": warnings,
            }

            return ToolResult(
                not bool(errors),
                json.dumps(report, indent=2),
                {"status": status, "error_count": len(errors), "warning_count": len(warnings)},
            )

        except Exception as e:
            return ToolResult(False, f"config_validator error: {e}")

    def _check_type(self, val: Any, type_str: str, key: str, errors: list):
        type_map = {"str": str, "int": int, "float": float, "bool": bool, "list": list, "dict": dict}
        expected = type_map.get(type_str.lower())
        if expected and not isinstance(val, expected):
            errors.append(f"Key '{key}' expected {type_str}, got {type(val).__name__}.")

    def _check_common(self, cfg: dict, warnings: list):
        # Detect unresolved env var placeholders
        raw = json.dumps(cfg)
        placeholders = re.findall(r"\$\{([^}]+)\}", raw)
        for ph in placeholders:
            if not os.environ.get(ph):
                warnings.append(f"Env var '{ph}' is referenced but not set in environment.")


# ══════════════════════════════════════════════════════════════════════════════
# TOOL #06 — CODE LINTER TOOL
# Static analysis for Python, Lua, JavaScript, Bash.
# Returns structured error + warning list with line numbers.
# ══════════════════════════════════════════════════════════════════════════════

class CodeLinterTool(BaseTool):
    """
    USAGE:
      code_linter(code="x=1\nprint(y)", language="python")
      code_linter(file="/path/to/script.lua", language="lua")
      code_linter(file="/path/to/app.js", language="javascript")
    """
    name        = "code_linter"
    description = (
        "Static analysis for Python (pyflakes), Lua (luacheck), "
        "JavaScript (node --check), Bash (bash -n). "
        "Returns structured errors and warnings with line numbers."
    )
    parameters  = (
        "code: str = '' (source code string), "
        "file: str = '' (path to source file), "
        "language: str (python|lua|javascript|bash)"
    )

    async def run(
        self,
        code: str = "",
        file: str = "",
        language: str = "python",
    ) -> ToolResult:
        try:
            lang = language.strip().lower()

            # Resolve source
            if file:
                p = Path(file)
                if not p.exists():
                    return ToolResult(False, f"File not found: {file}")
                code = p.read_text(errors="replace")
                src_label = str(p)
            elif code:
                src_label = f"<inline_{lang}>"
            else:
                return ToolResult(False, "Provide code or file.")

            # Write to temp file if inline
            suffix_map = {"python": ".py", "lua": ".lua", "javascript": ".js", "bash": ".sh"}
            suffix = suffix_map.get(lang, ".txt")

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=suffix, delete=False, encoding="utf-8"
            ) as tmp:
                tmp.write(code)
                tmp_path = tmp.name

            try:
                issues = await self._lint(lang, tmp_path, src_label, code)
            finally:
                os.unlink(tmp_path)

            errors   = [i for i in issues if i["level"] == "error"]
            warnings = [i for i in issues if i["level"] == "warning"]
            status   = "CLEAN" if not issues else ("ERRORS" if errors else "WARNINGS")

            report = {
                "status":   status,
                "language": lang,
                "source":   src_label,
                "errors":   errors,
                "warnings": warnings,
                "total":    len(issues),
            }

            return ToolResult(
                not bool(errors),
                json.dumps(report, indent=2),
                {"status": status, "error_count": len(errors), "warning_count": len(warnings)},
            )

        except Exception as e:
            return ToolResult(False, f"code_linter error: {e}")

    async def _lint(self, lang: str, path: str, label: str, code: str) -> list[dict]:
        issues = []

        if lang == "python":
            issues = self._lint_python_builtin(code, label)
            # Try pyflakes if available
            if shutil.which("pyflakes"):
                issues = await self._run_linter_cmd(["pyflakes", path], label, "pyflakes")

        elif lang == "lua":
            if shutil.which("luacheck"):
                issues = await self._run_linter_cmd(
                    ["luacheck", path, "--no-color", "--formatter", "plain"],
                    label, "luacheck"
                )
            else:
                # Fallback: basic Luau pattern check
                issues = self._lint_lua_builtin(code, label)

        elif lang == "javascript":
            if shutil.which("node"):
                issues = await self._run_linter_cmd(
                    ["node", "--check", path], label, "node"
                )
            else:
                issues = [{"line": 0, "col": 0, "level": "warning",
                           "message": "node not found — skipping JS lint.", "source": "linter"}]

        elif lang == "bash":
            if shutil.which("bash"):
                issues = await self._run_linter_cmd(
                    ["bash", "-n", path], label, "bash"
                )

        return issues

    def _lint_python_builtin(self, code: str, label: str) -> list[dict]:
        """Fallback Python lint using compile()."""
        issues = []
        try:
            compile(code, label, "exec")
        except SyntaxError as e:
            issues.append({
                "line": e.lineno or 0, "col": e.offset or 0,
                "level": "error", "message": str(e), "source": "python_compile"
            })
        return issues

    def _lint_lua_builtin(self, code: str, label: str) -> list[dict]:
        """Basic Lua pattern checks without luacheck."""
        issues = []
        lines = code.splitlines()
        for i, line in enumerate(lines, 1):
            if re.search(r"\bgoto\b", line):
                issues.append({"line": i, "col": 0, "level": "warning",
                               "message": "goto is not supported in Luau (Roblox).",
                               "source": "lua_builtin"})
            if re.search(r"\bloadstring\b", line):
                issues.append({"line": i, "col": 0, "level": "warning",
                               "message": "loadstring may not work in all executors.",
                               "source": "lua_builtin"})
        return issues

    async def _run_linter_cmd(self, cmd: list, label: str, source: str) -> list[dict]:
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)
            output = (stdout + stderr).decode(errors="replace")
            return self._parse_linter_output(output, source)
        except asyncio.TimeoutError:
            return [{"line": 0, "col": 0, "level": "error",
                     "message": f"{source} timed out.", "source": source}]
        except Exception as e:
            return [{"line": 0, "col": 0, "level": "warning",
                     "message": f"{source} unavailable: {e}", "source": source}]

    def _parse_linter_output(self, output: str, source: str) -> list[dict]:
        issues = []
        pattern = re.compile(r"(?:.*?):?(\d+):?(\d+)?:?\s*(error|warning|E|W)\s*:?\s*(.+)", re.IGNORECASE)
        for line in output.splitlines():
            m = pattern.match(line.strip())
            if m:
                lvl = "error" if m.group(3).lower() in ("error", "e") else "warning"
                issues.append({
                    "line":    int(m.group(1)),
                    "col":     int(m.group(2)) if m.group(2) else 0,
                    "level":   lvl,
                    "message": m.group(4).strip(),
                    "source":  source,
                })
            elif line.strip() and ":" in line:
                # Catch unparsed output as raw warning
                issues.append({
                    "line": 0, "col": 0, "level": "warning",
                    "message": line.strip(), "source": source
                })
        return issues


# ══════════════════════════════════════════════════════════════════════════════
# TOOL #07 — JSON QUERY TOOL
# JMESPath query over JSON data. No subprocess, no code execution.
# ══════════════════════════════════════════════════════════════════════════════

class JsonQueryTool(BaseTool):
    """
    USAGE:
      json_query(data='{"agent":{"name":"Gazcc","tools":["file","web"]}}', query="agent.name")
      json_query(file="/path/to/data.json", query="tools[?active==`true`].name")
      json_query(data='[{"id":1},{"id":2}]', query="[*].id")
    """
    name        = "json_query"
    description = (
        "Run a JMESPath query over JSON data or a JSON file. "
        "Zero-subprocess structured JSON extraction. "
        "Supports filtering, projections, and nested access."
    )
    parameters  = (
        "data: str = '' (raw JSON string), "
        "file: str = '' (path to JSON file), "
        "query: str (JMESPath expression)"
    )

    async def run(
        self,
        data: str = "",
        file: str = "",
        query: str = "",
    ) -> ToolResult:
        try:
            try:
                import jmespath
            except ImportError:
                return ToolResult(False, "jmespath not installed. Run: pip install jmespath")

            if not query:
                return ToolResult(False, "query (JMESPath expression) is required.")

            # Load data
            if file:
                p = Path(file)
                if not p.exists():
                    return ToolResult(False, f"File not found: {file}")
                async with aiofiles.open(p, "r") as fh:
                    data = await fh.read()

            if not data:
                return ToolResult(False, "Provide data or file.")

            try:
                parsed = json.loads(data)
            except json.JSONDecodeError as e:
                return ToolResult(False, f"Invalid JSON: {e}")

            try:
                result = jmespath.search(query, parsed)
            except jmespath.exceptions.JMESPathError as e:
                return ToolResult(False, f"JMESPath error: {e}")

            if result is None:
                return ToolResult(True, "null — query matched nothing.", {"matched": False})

            result_str = json.dumps(result, ensure_ascii=False, indent=2)
            result_type = type(result).__name__
            count = len(result) if isinstance(result, (list, dict)) else 1

            return ToolResult(True, result_str, {
                "matched": True,
                "type":    result_type,
                "count":   count,
            })

        except Exception as e:
            return ToolResult(False, f"json_query error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TOOL #08 — TEMPLATE RENDER TOOL
# Jinja2 template rendering with context injection.
# Render HTML, config files, prompts, or any text template.
# ══════════════════════════════════════════════════════════════════════════════

class TemplateRenderTool(BaseTool):
    """
    USAGE:
      template_render(template="Hello {{ name }}!", context='{"name": "Gazcc"}')
      template_render(file="/path/to/template.html", context='{"title":"GazccAI","items":["a","b"]}')
      template_render(template="...", context='{}', output_file="/out/result.html")
    """
    name        = "template_render"
    description = (
        "Render a Jinja2 template string or file with a JSON context dict. "
        "Supports all Jinja2 features: loops, conditionals, filters, macros. "
        "Output can be returned as string or written to a file."
    )
    parameters  = (
        "template: str = '' (Jinja2 template string), "
        "file: str = '' (path to .j2 or any template file), "
        "context: str = '{}' (JSON string with template variables), "
        "output_file: str = '' (optional: write rendered output to this path)"
    )

    async def run(
        self,
        template: str = "",
        file: str = "",
        context: str = "{}",
        output_file: str = "",
    ) -> ToolResult:
        try:
            try:
                from jinja2 import Environment, FileSystemLoader, BaseLoader, TemplateSyntaxError
            except ImportError:
                return ToolResult(False, "jinja2 not installed. Run: pip install jinja2")

            # Load template
            if file:
                p = Path(file)
                if not p.exists():
                    return ToolResult(False, f"Template file not found: {file}")
                env = Environment(loader=FileSystemLoader(str(p.parent)))
                tmpl = env.get_template(p.name)
            elif template:
                env  = Environment(loader=BaseLoader())
                try:
                    tmpl = env.from_string(template)
                except TemplateSyntaxError as e:
                    return ToolResult(False, f"Jinja2 syntax error at line {e.lineno}: {e.message}")
            else:
                return ToolResult(False, "Provide template string or file.")

            # Parse context
            try:
                ctx = json.loads(context)
            except json.JSONDecodeError as e:
                return ToolResult(False, f"Invalid context JSON: {e}")

            # Render
            try:
                rendered = tmpl.render(**ctx)
            except Exception as e:
                return ToolResult(False, f"Jinja2 render error: {e}")

            # Optionally write output
            if output_file:
                op = Path(output_file)
                op.parent.mkdir(parents=True, exist_ok=True)
                op.write_text(rendered, encoding="utf-8")
                return ToolResult(True, f"Rendered and written to {output_file}.", {
                    "written": True, "path": str(op), "char_count": len(rendered)
                })

            return ToolResult(True, rendered, {
                "char_count": len(rendered), "written": False
            })

        except Exception as e:
            return ToolResult(False, f"template_render error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TOOL #09 — TASK QUEUE TOOL
# In-memory FIFO task queue for agent sub-task management.
# Persist across tool calls via class-level state.
# ══════════════════════════════════════════════════════════════════════════════

class TaskQueueTool(BaseTool):
    """
    USAGE:
      task_queue(operation="enqueue", queue="main", task="Fetch data from API")
      task_queue(operation="dequeue", queue="main")
      task_queue(operation="peek", queue="main")
      task_queue(operation="list", queue="main")
      task_queue(operation="clear", queue="main")
      task_queue(operation="status")
    """
    name        = "task_queue"
    description = (
        "In-memory FIFO task queue for managing agent sub-tasks. "
        "Supports multiple named queues. Operations: enqueue, dequeue, peek, list, clear, status."
    )
    parameters  = (
        "operation: str ('enqueue'|'dequeue'|'peek'|'list'|'clear'|'status'), "
        "queue: str = 'default' (queue name), "
        "task: str = '' (task string for enqueue), "
        "priority: int = 0 (higher = more important, for ordering insight)"
    )

    # Class-level shared state (singleton per process)
    _queues:    dict[str, deque] = {}
    _metadata:  dict[str, list]  = {}

    async def run(
        self,
        operation: str,
        queue: str = "default",
        task: str = "",
        priority: int = 0,
    ) -> ToolResult:
        try:
            op = operation.strip().lower()

            # Ensure queue exists
            if queue not in self._queues:
                self._queues[queue]   = deque()
                self._metadata[queue] = []

            q    = self._queues[queue]
            meta = self._metadata[queue]

            if op == "enqueue":
                if not task:
                    return ToolResult(False, "task string is required for enqueue.")
                entry = {
                    "id":        len(meta),
                    "task":      task,
                    "priority":  priority,
                    "queued_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "status":    "pending",
                }
                q.append(entry)
                meta.append(entry)
                return ToolResult(True, f"Enqueued task #{entry['id']} to queue '{queue}'.", {
                    "task_id": entry["id"], "queue_size": len(q)
                })

            elif op == "dequeue":
                if not q:
                    return ToolResult(True, f"Queue '{queue}' is empty.", {"empty": True})
                entry = q.popleft()
                entry["status"] = "dequeued"
                return ToolResult(True, json.dumps(entry, indent=2), {
                    "task_id": entry["id"], "remaining": len(q)
                })

            elif op == "peek":
                if not q:
                    return ToolResult(True, f"Queue '{queue}' is empty.", {"empty": True})
                entry = q[0]
                return ToolResult(True, json.dumps(entry, indent=2), {
                    "task_id": entry["id"], "queue_size": len(q)
                })

            elif op == "list":
                items = list(q)
                return ToolResult(True, json.dumps(items, indent=2), {
                    "queue": queue, "size": len(items)
                })

            elif op == "clear":
                size = len(q)
                q.clear()
                return ToolResult(True, f"Cleared {size} tasks from queue '{queue}'.", {
                    "cleared": size
                })

            elif op == "status":
                status = {
                    name: {"size": len(dq), "total_ever": len(self._metadata.get(name, []))}
                    for name, dq in self._queues.items()
                }
                return ToolResult(True, json.dumps(status, indent=2), {"queues": list(status.keys())})

            else:
                return ToolResult(False, f"Unknown operation: '{operation}'.")

        except Exception as e:
            return ToolResult(False, f"task_queue error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TOOL #10 — NOTIFICATION BROADCAST TOOL
# Push real-time notifications to webhooks or SSE endpoints.
# Lets GazccAI frontend receive live progress updates from the agent.
# ══════════════════════════════════════════════════════════════════════════════

class NotificationBroadcastTool(BaseTool):
    """
    USAGE:
      notification_broadcast(event="step_done", message="Step 3/7 complete", webhook="https://...")
      notification_broadcast(event="task_failed", message="Tool crashed", data='{"tool":"web_search"}')
      notification_broadcast(event="progress", message="50% done", level="info")
    """
    name        = "notification_broadcast"
    description = (
        "Push real-time event notifications to a webhook URL or SSE endpoint. "
        "Enables GazccAI frontend to receive live agent progress updates. "
        "Falls back to console log if no webhook configured."
    )
    parameters  = (
        "event: str (event name, e.g. 'step_done', 'task_failed', 'progress'), "
        "message: str (human-readable message), "
        "webhook: str = '' (URL to POST to; reads GAZCC_WEBHOOK_URL env if empty), "
        "data: str = '{}' (extra JSON payload), "
        "level: str = 'info' ('info'|'warning'|'error'|'success')"
    )

    async def run(
        self,
        event: str,
        message: str,
        webhook: str = "",
        data: str = "{}",
        level: str = "info",
    ) -> ToolResult:
        try:
            if not event:
                return ToolResult(False, "event is required.")

            # Resolve webhook URL
            url = webhook or os.environ.get("GAZCC_WEBHOOK_URL", "")

            # Build payload
            try:
                extra = json.loads(data)
            except json.JSONDecodeError:
                extra = {"raw": data}

            payload = {
                "event":     event,
                "message":   message,
                "level":     level,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "data":      extra,
            }

            # Attempt HTTP push if webhook URL is set
            if url:
                try:
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        resp = await client.post(
                            url,
                            json=payload,
                            headers={"Content-Type": "application/json"},
                        )
                    if resp.status_code < 300:
                        return ToolResult(True, f"Notification sent → {url} [{resp.status_code}]", {
                            "delivered": True, "status_code": resp.status_code, "event": event
                        })
                    else:
                        return ToolResult(False,
                            f"Webhook returned {resp.status_code}: {resp.text[:200]}",
                            {"delivered": False, "status_code": resp.status_code}
                        )
                except Exception as he:
                    return ToolResult(False, f"HTTP error posting to webhook: {he}")
            else:
                # Console fallback
                log_line = f"[BROADCAST][{level.upper()}] {event}: {message}"
                print(log_line)
                return ToolResult(True, log_line, {
                    "delivered": False, "fallback": "console",
                    "note": "Set GAZCC_WEBHOOK_URL or pass webhook param to push to endpoint."
                })

        except Exception as e:
            return ToolResult(False, f"notification_broadcast error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TOOL #11 — CODE TRANSLATOR TOOL
# Translate code from one language to another via LLM sub-call (OpenRouter).
# Specialization: Python → Luau, JS → Python, any pair.
# ══════════════════════════════════════════════════════════════════════════════

class CodeTranslatorTool(BaseTool):
    """
    USAGE:
      code_translator(code="def greet(n): return f'hi {n}'", from_lang="python", to_lang="lua")
      code_translator(file="/path/to/script.py", from_lang="python", to_lang="luau",
                      target_env="Roblox Delta Executor")
    """
    name        = "code_translator"
    description = (
        "Translate source code from one language to another using LLM. "
        "Specialization: Python→Luau (Roblox/Delta Executor compatible). "
        "Preserves logic, adapts idioms, annotates non-translatable patterns."
    )
    parameters  = (
        "code: str = '' (source code), "
        "file: str = '' (path to source file), "
        "from_lang: str (source language, e.g. 'python'), "
        "to_lang: str (target language, e.g. 'lua', 'luau', 'javascript'), "
        "target_env: str = '' (execution environment hint, e.g. 'Roblox Delta Executor'), "
        "notes: str = '' (extra instructions for the translator)"
    )

    async def run(
        self,
        code: str = "",
        file: str = "",
        from_lang: str = "",
        to_lang: str = "",
        target_env: str = "",
        notes: str = "",
    ) -> ToolResult:
        try:
            if not from_lang or not to_lang:
                return ToolResult(False, "from_lang and to_lang are required.")

            if file:
                p = Path(file)
                if not p.exists():
                    return ToolResult(False, f"File not found: {file}")
                code = p.read_text(errors="replace")

            if not code:
                return ToolResult(False, "Provide code or file.")

            api_key = os.environ.get("OPENROUTER_API_KEY", "")
            if not api_key:
                return ToolResult(False, "OPENROUTER_API_KEY not set in environment.")

            env_hint = f" targeting {target_env}" if target_env else ""
            extra    = f"\n\nAdditional notes: {notes}" if notes else ""

            system_prompt = (
                f"You are an expert code translator. Translate the provided {from_lang} code "
                f"into {to_lang}{env_hint}. "
                "Rules:\n"
                "1. Preserve all logic exactly — no simplification.\n"
                "2. Adapt language idioms correctly (not word-for-word).\n"
                "3. If a pattern cannot be translated, add a comment: -- TODO: [reason]\n"
                "4. Return ONLY the translated code block. No explanation, no markdown fences.\n"
                f"5. Ensure compatibility with {target_env if target_env else to_lang} runtime.{extra}"
            )

            payload = {
                "model":       "moonshotai/kimi-k2.5",
                "max_tokens":  4000,
                "temperature": 0.1,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": f"Translate this {from_lang} code:\n\n{code}"},
                ],
            }

            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    json=payload,
                    headers={
                        "Authorization":  f"Bearer {api_key}",
                        "Content-Type":   "application/json",
                        "HTTP-Referer":   "https://gazcc.vercel.app",
                        "X-Title":        "GazccThinking-CodeTranslator",
                    },
                )

            if resp.status_code != 200:
                return ToolResult(False, f"LLM API error {resp.status_code}: {resp.text[:300]}")

            data      = resp.json()
            translated = data["choices"][0]["message"]["content"].strip()

            # Strip any accidental markdown fences
            translated = re.sub(r"^```[\w]*\n?", "", translated)
            translated = re.sub(r"\n?```$", "", translated)

            return ToolResult(True, translated, {
                "from_lang":   from_lang,
                "to_lang":     to_lang,
                "target_env":  target_env,
                "char_count":  len(translated),
                "model_used":  data.get("model", "unknown"),
            })

        except Exception as e:
            return ToolResult(False, f"code_translator error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TOOL #12 — LUA SCRIPT ANALYZER TOOL
# Static analyzer for Luau/Roblox scripts.
# Detects Delta Executor incompatibilities, performance issues,
# remote abuse, anti-cheat triggers, and common exploit patterns.
# ══════════════════════════════════════════════════════════════════════════════

class LuaScriptAnalyzerTool(BaseTool):
    """
    USAGE:
      lua_script_analyzer(file="/path/to/script.lua")
      lua_script_analyzer(code="game:GetService('Players')...")
      lua_script_analyzer(code="...", strict=True)
    """
    name        = "lua_script_analyzer"
    description = (
        "Static analyzer for Luau/Roblox exploit scripts. "
        "Detects: Delta Executor incompatibilities, goto usage, loadstring misuse, "
        "RemoteEvent flood risk, infinite loop patterns, deprecated API calls, "
        "anti-cheat trigger patterns, and memory leak risks."
    )
    parameters  = (
        "code: str = '' (Luau source code), "
        "file: str = '' (path to .lua file), "
        "strict: bool = False (enable all experimental checks)"
    )

    # ── Pattern definitions ───────────────────────────────────────────────────
    _PATTERNS = [
        # (pattern, level, category, message)
        (r"\bgoto\b",
         "error", "compatibility",
         "goto is NOT supported in Luau (Roblox). Replace with boolean flag guards."),

        (r"\blocal\s+function\s+\w+\s*\(.*\)\s*\n(?:.*\n)*?.*\1\s*\(",
         "warning", "recursion",
         "Possible recursive function detected. Ensure base case exists."),

        (r"game:HttpGet\s*\(",
         "info", "network",
         "game:HttpGet() detected. Ensure HttpService is enabled and URL is trusted."),

        (r"loadstring\s*\(",
         "warning", "compatibility",
         "loadstring() may be blocked on some executors. Prefer direct require() or inline code."),

        (r"require\s*\(",
         "info", "compatibility",
         "require() usage detected. Verify module is accessible from executor context."),

        (r"while\s+true\s+do\s*\n(?!.*task\.wait|.*wait\()",
         "warning", "performance",
         "Infinite loop without wait() detected. Will cause server-side lag/kick."),

        (r"game\.Players\.LocalPlayer\.Character\.Humanoid\.Health\s*=\s*0",
         "error", "anti_cheat",
         "Self-kill via Health=0 detected. High anti-cheat trigger risk."),

        (r"FireServer\s*\(",
         "info", "remote_abuse",
         "RemoteEvent:FireServer() detected. Rate-limit to prevent remote flood ban."),

        (r"InvokeServer\s*\(",
         "info", "remote_abuse",
         "RemoteFunction:InvokeServer() detected. Ensure not called in a tight loop."),

        (r"Workspace\b|game\.Workspace\b",
         "warning", "deprecation",
         "game.Workspace is deprecated. Use workspace (lowercase) instead."),

        (r"\.Touched:Connect\b",
         "info", "performance",
         ".Touched event can fire rapidly. Debounce with a cooldown variable."),

        (r"pcall\s*\(\s*function",
         "info", "best_practice",
         "pcall() detected — good error handling practice."),

        (r"getgenv\s*\(\)|getrenv\s*\(\)|getfenv\s*\(",
         "warning", "compatibility",
         "Environment access functions may not work on all executors. Test in Delta."),

        (r"debug\.getinfo\s*\(|debug\.traceback\s*\(",
         "warning", "compatibility",
         "debug library access — restricted on some executors."),

        (r"syn\.\w+|fluxus\.\w+|krnl\.\w+|oxygen\.\w+",
         "warning", "compatibility",
         "Executor-specific API detected. Not compatible with Delta Executor."),

        (r"Drawing\.new\s*\(",
         "info", "drawing",
         "Drawing API detected. Ensure cleanup on script end to prevent ghost renders."),

        (r"game:GetService\s*\(\s*[\"']RunService[\"']\s*\)\s*\.\s*Heartbeat:Connect\b",
         "info", "performance",
         "Heartbeat loop detected. Keep logic minimal — fires 60x/sec."),

        (r"_G\.\w+",
         "warning", "scope",
         "_G global table usage detected. Prefer local variables or getgenv() for shared state."),

        (r"module\s+\w+\s*=\s*\{|return\s*\{",
         "info", "modulescript",
         "ModuleScript pattern detected. Verify module is not server-side only."),
    ]

    async def run(
        self,
        code: str = "",
        file: str = "",
        strict: bool = False,
    ) -> ToolResult:
        try:
            if file:
                p = Path(file)
                if not p.exists():
                    return ToolResult(False, f"File not found: {file}")
                code = p.read_text(errors="replace")

            if not code:
                return ToolResult(False, "Provide code or file.")

            issues   = []
            lines    = code.splitlines()
            line_count = len(lines)

            for pattern_str, level, category, message in self._PATTERNS:
                try:
                    rx = re.compile(pattern_str, re.MULTILINE)
                    for m in rx.finditer(code):
                        # Find line number
                        line_no = code[:m.start()].count("\n") + 1
                        issues.append({
                            "line":     line_no,
                            "level":    level,
                            "category": category,
                            "message":  message,
                            "snippet":  lines[line_no - 1].strip() if line_no <= line_count else "",
                        })
                except re.error:
                    continue

            # Additional strict checks
            if strict:
                issues += self._strict_checks(code, lines)

            # Deduplicate by (line, message)
            seen   = set()
            unique = []
            for issue in issues:
                key = (issue["line"], issue["message"][:40])
                if key not in seen:
                    seen.add(key)
                    unique.append(issue)

            unique.sort(key=lambda x: (
                {"error": 0, "warning": 1, "info": 2}.get(x["level"], 3),
                x["line"]
            ))

            errors   = [i for i in unique if i["level"] == "error"]
            warnings = [i for i in unique if i["level"] == "warning"]
            infos    = [i for i in unique if i["level"] == "info"]

            # Script stats
            stats = self._compute_stats(code, lines)

            report = {
                "status":       "ISSUES_FOUND" if unique else "CLEAN",
                "total_lines":  line_count,
                "errors":       errors,
                "warnings":     warnings,
                "info":         infos,
                "stats":        stats,
                "delta_compatible": not bool(errors),
            }

            return ToolResult(
                not bool(errors),
                json.dumps(report, indent=2, ensure_ascii=False),
                {
                    "status":       report["status"],
                    "error_count":  len(errors),
                    "warning_count":len(warnings),
                    "delta_compatible": report["delta_compatible"],
                },
            )

        except Exception as e:
            return ToolResult(False, f"lua_script_analyzer error: {e}")

    def _strict_checks(self, code: str, lines: list[str]) -> list[dict]:
        issues = []
        # Check for very long single lines (potential obfuscation)
        for i, line in enumerate(lines, 1):
            if len(line) > 500:
                issues.append({
                    "line": i, "level": "warning", "category": "obfuscation",
                    "message": f"Line {i} is {len(line)} chars long — possible obfuscation.",
                    "snippet": line[:80] + "...",
                })
        # Check for base64-encoded strings (common in obfuscated scripts)
        b64_pattern = re.compile(r'"([A-Za-z0-9+/]{50,}={0,2})"')
        for m in b64_pattern.finditer(code):
            line_no = code[:m.start()].count("\n") + 1
            issues.append({
                "line": line_no, "level": "warning", "category": "obfuscation",
                "message": "Possible base64-encoded string. Verify content before execution.",
                "snippet": m.group(0)[:60] + "...",
            })
        return issues

    def _compute_stats(self, code: str, lines: list[str]) -> dict:
        return {
            "total_lines":       len(lines),
            "non_empty_lines":   sum(1 for l in lines if l.strip()),
            "comment_lines":     sum(1 for l in lines if l.strip().startswith("--")),
            "function_count":    len(re.findall(r"\bfunction\b", code)),
            "remote_fire_count": len(re.findall(r"FireServer\s*\(|FireClient\s*\(|FireAllClients\s*\(", code)),
            "pcall_count":       len(re.findall(r"\bpcall\b", code)),
            "wait_count":        len(re.findall(r"\btask\.wait\b|\bwait\s*\(", code)),
            "drawing_count":     len(re.findall(r"Drawing\.new\s*\(", code)),
        }


# ══════════════════════════════════════════════════════════════════════════════
# TOOL #14 — DEPENDENCY RESOLVER TOOL
# Parse and analyze requirements.txt, package.json, vercel.json.
# Detect version conflicts, missing deps, and outdated packages.
# ══════════════════════════════════════════════════════════════════════════════

class DependencyResolverTool(BaseTool):
    """
    USAGE:
      dependency_resolver(file="requirements.txt")
      dependency_resolver(file="package.json")
      dependency_resolver(file="vercel.json")
      dependency_resolver(file="requirements.txt", check_installed=True)
    """
    name        = "dependency_resolver"
    description = (
        "Parse and analyze requirements.txt, package.json, or vercel.json. "
        "Detects version conflicts, missing packages, duplicate entries, "
        "and optionally checks what's actually installed."
    )
    parameters  = (
        "file: str (path to requirements.txt / package.json / vercel.json), "
        "check_installed: bool = False (compare against actually installed packages)"
    )

    async def run(
        self,
        file: str,
        check_installed: bool = False,
    ) -> ToolResult:
        try:
            p = Path(file)
            if not p.exists():
                return ToolResult(False, f"File not found: {file}")

            content = p.read_text(encoding="utf-8")
            fname   = p.name.lower()

            if fname == "requirements.txt" or fname.endswith(".txt"):
                report = self._parse_requirements(content, check_installed)
            elif fname == "package.json":
                report = self._parse_package_json(content)
            elif fname == "vercel.json":
                report = self._parse_vercel_json(content)
            else:
                return ToolResult(False, "Unsupported file. Use requirements.txt, package.json, or vercel.json.")

            status = "OK" if not report.get("errors") else "ISSUES_FOUND"
            report["file"]   = str(p)
            report["status"] = status

            return ToolResult(
                status == "OK",
                json.dumps(report, indent=2, ensure_ascii=False),
                {"status": status, "error_count": len(report.get("errors", []))},
            )

        except Exception as e:
            return ToolResult(False, f"dependency_resolver error: {e}")

    def _parse_requirements(self, content: str, check_installed: bool) -> dict:
        deps     = {}
        errors   = []
        warnings = []

        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Parse package==version or package>=version etc.
            m = re.match(r"^([A-Za-z0-9_\-]+)\s*([><=!~]+)?\s*([\d.*]+)?", line)
            if m:
                pkg  = m.group(1).lower()
                spec = (m.group(2) or "") + (m.group(3) or "")
                if pkg in deps:
                    errors.append(f"Duplicate entry: '{pkg}' appears more than once.")
                deps[pkg] = spec or "any"
            else:
                warnings.append(f"Could not parse line: '{line}'")

        installed = {}
        if check_installed:
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "list", "--format=json"],
                    capture_output=True, text=True, timeout=15
                )
                pkg_list  = json.loads(result.stdout)
                installed = {p["name"].lower(): p["version"] for p in pkg_list}
            except Exception as e:
                warnings.append(f"Could not check installed packages: {e}")

        missing = []
        if installed:
            missing = [pkg for pkg in deps if pkg not in installed]

        return {
            "type":              "requirements.txt",
            "total_deps":        len(deps),
            "dependencies":      deps,
            "errors":            errors,
            "warnings":          warnings,
            "missing_installed": missing,
        }

    def _parse_package_json(self, content: str) -> dict:
        errors = []
        warnings = []
        try:
            pkg = json.loads(content)
        except json.JSONDecodeError as e:
            return {"errors": [f"Invalid JSON: {e}"], "warnings": []}

        deps     = pkg.get("dependencies", {})
        dev_deps = pkg.get("devDependencies", {})
        all_deps = {**deps, **dev_deps}

        # Check for conflicts (same pkg in both)
        conflicts = [k for k in deps if k in dev_deps]
        for c in conflicts:
            errors.append(f"'{c}' appears in both dependencies and devDependencies.")

        # Check for wildcard versions
        for pkg_name, ver in all_deps.items():
            if ver in ("*", "latest"):
                warnings.append(f"'{pkg_name}' uses non-pinned version '{ver}'. Risk of breakage.")

        return {
            "type":           "package.json",
            "name":           pkg.get("name", "unknown"),
            "version":        pkg.get("version", "unknown"),
            "total_deps":     len(deps),
            "total_dev_deps": len(dev_deps),
            "dependencies":   deps,
            "devDependencies":dev_deps,
            "errors":         errors,
            "warnings":       warnings,
            "conflicts":      conflicts,
        }

    def _parse_vercel_json(self, content: str) -> dict:
        errors = []
        warnings = []
        try:
            cfg = json.loads(content)
        except json.JSONDecodeError as e:
            return {"errors": [f"Invalid JSON: {e}"], "warnings": []}

        # Check common Vercel misconfigs
        routes  = cfg.get("routes", [])
        builds  = cfg.get("builds", [])
        env_raw = cfg.get("env", {})
        funcs   = cfg.get("functions", {})

        if not builds and not cfg.get("framework"):
            warnings.append("No 'builds' or 'framework' specified. Vercel may auto-detect incorrectly.")

        for route in routes:
            if "src" not in route:
                errors.append(f"Route missing 'src' field: {route}")

        # Check for unresolved env refs
        env_str = json.dumps(env_raw)
        placeholders = re.findall(r"@([A-Z_]+)", env_str)
        for ph in placeholders:
            warnings.append(f"Env placeholder '@{ph}' — ensure it's set in Vercel Dashboard.")

        return {
            "type":      "vercel.json",
            "routes":    len(routes),
            "builds":    len(builds),
            "functions": len(funcs),
            "env_refs":  placeholders,
            "errors":    errors,
            "warnings":  warnings,
        }


# ══════════════════════════════════════════════════════════════════════════════
# TOOL #18 — IMAGE METADATA TOOL
# Extract EXIF, dimensions, format from image files.
# Encode image as base64 for LLM vision input.
# ══════════════════════════════════════════════════════════════════════════════

class ImageMetadataTool(BaseTool):
    """
    USAGE:
      image_metadata(file="/path/to/image.png")
      image_metadata(file="/path/to/photo.jpg", encode_base64=True)
      image_metadata(file="/path/to/img.webp", encode_base64=True, max_size_kb=500)
    """
    name        = "image_metadata"
    description = (
        "Extract image metadata (format, dimensions, mode, EXIF) from image files. "
        "Optionally encode image as base64 for LLM vision API input. "
        "Supports JPEG, PNG, WEBP, GIF, BMP."
    )
    parameters  = (
        "file: str (path to image file), "
        "encode_base64: bool = False (include base64 data URI), "
        "max_size_kb: int = 1024 (max file size to process in KB)"
    )

    async def run(
        self,
        file: str,
        encode_base64: bool = False,
        max_size_kb: int = 1024,
    ) -> ToolResult:
        try:
            try:
                from PIL import Image
                import PIL.ExifTags as ExifTags
                HAS_PILLOW = True
            except ImportError:
                HAS_PILLOW = False

            p = Path(file)
            if not p.exists():
                return ToolResult(False, f"File not found: {file}")

            file_size_kb = p.stat().st_size // 1024
            if file_size_kb > max_size_kb:
                return ToolResult(False,
                    f"File too large: {file_size_kb}KB > max {max_size_kb}KB.")

            # Read raw bytes
            raw = p.read_bytes()

            metadata: dict = {
                "file":          str(p),
                "filename":      p.name,
                "size_bytes":    len(raw),
                "size_kb":       round(len(raw) / 1024, 2),
                "extension":     p.suffix.lower(),
            }

            if HAS_PILLOW:
                with Image.open(p) as img:
                    metadata["format"]    = img.format or p.suffix.lstrip(".").upper()
                    metadata["mode"]      = img.mode
                    metadata["width"]     = img.width
                    metadata["height"]    = img.height
                    metadata["megapixels"]= round((img.width * img.height) / 1_000_000, 3)

                    # EXIF extraction
                    exif_data = {}
                    try:
                        exif_raw = img._getexif()
                        if exif_raw:
                            for tag_id, value in exif_raw.items():
                                tag = ExifTags.TAGS.get(tag_id, str(tag_id))
                                if isinstance(value, bytes):
                                    value = value.decode(errors="replace")
                                if isinstance(value, (str, int, float)):
                                    exif_data[tag] = value
                    except Exception:
                        pass
                    metadata["exif"] = exif_data
            else:
                metadata["note"] = "Pillow not installed — limited metadata. Run: pip install pillow"
                # Detect format from magic bytes
                magic = raw[:12]
                if magic[:8] == b"\x89PNG\r\n\x1a\n":
                    metadata["format"] = "PNG"
                elif magic[:3] == b"\xff\xd8\xff":
                    metadata["format"] = "JPEG"
                elif magic[:4] == b"RIFF" and magic[8:12] == b"WEBP":
                    metadata["format"] = "WEBP"
                elif magic[:6] in (b"GIF87a", b"GIF89a"):
                    metadata["format"] = "GIF"

            # Base64 encoding
            if encode_base64:
                b64     = base64.b64encode(raw).decode("ascii")
                fmt     = metadata.get("format", "jpeg").lower()
                mime    = f"image/{fmt}"
                data_uri = f"data:{mime};base64,{b64}"
                metadata["base64_data_uri"]   = data_uri
                metadata["base64_length"]     = len(b64)
                metadata["base64_size_kb"]    = round(len(b64) / 1024, 2)

            return ToolResult(
                True,
                json.dumps(metadata, indent=2, ensure_ascii=False),
                {
                    "format":    metadata.get("format"),
                    "width":     metadata.get("width"),
                    "height":    metadata.get("height"),
                    "has_exif":  bool(metadata.get("exif")),
                    "base64_encoded": encode_base64,
                },
            )

        except Exception as e:
            return ToolResult(False, f"image_metadata error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TOOL #20 — AGENT INSPECTOR TOOL
# Full introspection dump of GazccThinking agent state.
# Outputs active plan, tool history, memory snapshot, error log, token estimate.
# ══════════════════════════════════════════════════════════════════════════════

class AgentInspectorTool(BaseTool):
    """
    USAGE:
      agent_inspector(state_json='{"task":"...","steps":[...],"errors":[]}')
      agent_inspector(state_json='...', section="tools")
      agent_inspector(state_file="/tmp/gazcc_checkpoint.json")
    """
    name        = "agent_inspector"
    description = (
        "Full introspection dump of GazccThinking agent runtime state. "
        "Sections: all, plan, tools, memory, errors, tokens, health. "
        "Pass agent state as JSON string or checkpoint file path."
    )
    parameters  = (
        "state_json: str = '' (agent state as JSON string), "
        "state_file: str = '' (path to checkpoint JSON file), "
        "section: str = 'all' ('all'|'plan'|'tools'|'memory'|'errors'|'tokens'|'health')"
    )

    async def run(
        self,
        state_json: str = "",
        state_file: str = "",
        section: str = "all",
    ) -> ToolResult:
        try:
            # Load state
            if state_file:
                p = Path(state_file)
                if not p.exists():
                    return ToolResult(False, f"State file not found: {state_file}")
                async with aiofiles.open(p, "r") as fh:
                    state_json = await fh.read()

            if not state_json:
                return ToolResult(False, "Provide state_json or state_file.")

            try:
                state = json.loads(state_json)
            except json.JSONDecodeError as e:
                return ToolResult(False, f"Invalid state JSON: {e}")

            sec = section.lower()

            # ── Build inspection report ────────────────────────────────────────

            report = {}

            # Plan section
            plan_data = {
                "task":          state.get("task", "unknown"),
                "task_id":       state.get("task_id", "unknown"),
                "status":        state.get("status", "unknown"),
                "steps_total":   len(state.get("steps", [])),
                "steps_done":    sum(1 for s in state.get("steps", [])
                                    if s.get("status") == "done"),
                "steps_failed":  sum(1 for s in state.get("steps", [])
                                    if s.get("status") == "failed"),
                "current_step":  state.get("current_step", None),
                "plan_steps":    state.get("steps", []),
            }

            # Tools section
            tool_calls   = state.get("tool_calls", [])
            tool_summary = {}
            for tc in tool_calls:
                tn = tc.get("tool", "unknown")
                tool_summary[tn] = tool_summary.get(tn, 0) + 1

            tools_data = {
                "total_tool_calls":   len(tool_calls),
                "tool_call_counts":   tool_summary,
                "last_tool_call":     tool_calls[-1] if tool_calls else None,
                "failed_tool_calls":  [tc for tc in tool_calls if not tc.get("success", True)],
                "tool_call_history":  tool_calls[-10:],  # Last 10
            }

            # Memory section
            memory_data = {
                "memory_backend":   state.get("memory_backend", "unknown"),
                "memory_entries":   state.get("memory_count", 0),
                "context_snapshot": state.get("context", {}),
                "semantic_keys":    state.get("semantic_memory_keys", []),
            }

            # Errors section
            errors       = state.get("errors", [])
            errors_data  = {
                "error_count":   len(errors),
                "errors":        errors,
                "last_error":    errors[-1] if errors else None,
                "retry_count":   state.get("retry_count", 0),
            }

            # Token estimate section
            def est_tokens(s: str) -> int:
                return len(s) // 4

            raw_str = json.dumps(state)
            tokens_data = {
                "state_size_chars":    len(raw_str),
                "state_est_tokens":    est_tokens(raw_str),
                "task_est_tokens":     est_tokens(state.get("task", "")),
                "context_est_tokens":  est_tokens(json.dumps(state.get("context", {}))),
                "plan_est_tokens":     est_tokens(json.dumps(state.get("steps", []))),
                "note": "Token estimates: 1 token ≈ 4 chars (rough OpenAI/Claude estimate)",
            }

            # Health section
            all_tools     = list(tool_summary.keys())
            failed_tools  = list({tc.get("tool") for tc in tool_calls if not tc.get("success", True)})
            health_score  = 100
            health_issues = []

            if errors_data["error_count"] > 0:
                health_score -= min(40, errors_data["error_count"] * 10)
                health_issues.append(f"{errors_data['error_count']} error(s) recorded.")
            if plan_data["steps_failed"] > 0:
                health_score -= min(30, plan_data["steps_failed"] * 10)
                health_issues.append(f"{plan_data['steps_failed']} step(s) failed.")
            if failed_tools:
                health_score -= 10
                health_issues.append(f"Tools with failures: {failed_tools}")
            if tokens_data["state_est_tokens"] > 50000:
                health_issues.append("State size approaching context limit. Consider summarizing.")

            health_data = {
                "score":          max(0, health_score),
                "rating":         "HEALTHY" if health_score >= 80 else
                                  "DEGRADED" if health_score >= 50 else "CRITICAL",
                "issues":         health_issues,
                "failed_tools":   failed_tools,
                "uptime_seconds": state.get("elapsed_seconds", 0),
            }

            # ── Assemble output ───────────────────────────────────────────────

            if sec == "all":
                report = {
                    "plan":    plan_data,
                    "tools":   tools_data,
                    "memory":  memory_data,
                    "errors":  errors_data,
                    "tokens":  tokens_data,
                    "health":  health_data,
                    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            elif sec == "plan":
                report = plan_data
            elif sec == "tools":
                report = tools_data
            elif sec == "memory":
                report = memory_data
            elif sec == "errors":
                report = errors_data
            elif sec == "tokens":
                report = tokens_data
            elif sec == "health":
                report = health_data
            else:
                return ToolResult(False, f"Unknown section: '{section}'. Use: all/plan/tools/memory/errors/tokens/health")

            return ToolResult(
                True,
                json.dumps(report, indent=2, ensure_ascii=False),
                {
                    "section":       sec,
                    "health_score":  health_data["score"],
                    "health_rating": health_data["rating"],
                },
            )

        except Exception as e:
            return ToolResult(False, f"agent_inspector error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# REGISTRATION — Add all expansion tools to ToolRegistry
# ══════════════════════════════════════════════════════════════════════════════

def register_expansion_tools(registry, cfg: dict | None = None) -> None:
    """
    Register all 15 expansion tools into an existing ToolRegistry instance.

    USAGE (in agent/core.py or wherever ToolRegistry is initialized):
        from agent.gazcc_tools_expansion import register_expansion_tools
        register_expansion_tools(self.tool_registry, cfg)

    CONFIG KEYS (in config.yaml under 'tools:'):
        expansion_tools: true   # master switch
        code_translator: true   # requires OPENROUTER_API_KEY
        image_metadata: true    # requires pillow
    """
    cfg = cfg or {}
    tool_cfg = cfg.get("tools", {})

    expansion_enabled = tool_cfg.get("expansion_tools", True)
    if not expansion_enabled:
        return

    # Core expansion tools — always register
    core_tools = [
        DiffPatchTool(),
        RegexExtractTool(),
        TextChunkerTool(),
        EnvManagerTool(),
        ConfigValidatorTool(),
        CodeLinterTool(),
        JsonQueryTool(),
        TemplateRenderTool(),
        TaskQueueTool(),
        NotificationBroadcastTool(),
        LuaScriptAnalyzerTool(),
        DependencyResolverTool(),
        AgentInspectorTool(),
    ]

    for tool in core_tools:
        registry.register(tool)

    # Optional: CodeTranslatorTool (needs OPENROUTER_API_KEY)
    if tool_cfg.get("code_translator", True):
        registry.register(CodeTranslatorTool())

    # Optional: ImageMetadataTool (needs pillow)
    if tool_cfg.get("image_metadata", True):
        registry.register(ImageMetadataTool())


# ══════════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import asyncio

    async def _test():
        print("═" * 60)
        print("  GAZCC TOOLS EXPANSION — SELF TEST")
        print("═" * 60)

        tests = [
            ("DiffPatchTool",          DiffPatchTool(),
             {"operation": "diff", "source": "line1\nline2\nline3", "target": "line1\nLINE2\nline3"}),
            ("RegexExtractTool",       RegexExtractTool(),
             {"pattern": r"\d{4}", "text": "Year 2025 and 2026", "all": True}),
            ("TextChunkerTool",        TextChunkerTool(),
             {"text": "Hello world. " * 200, "max_tokens": 100, "strategy": "sentence"}),
            ("JsonQueryTool",          JsonQueryTool(),
             {"data": '{"agent":{"name":"Gazcc","active":true}}', "query": "agent.name"}),
            ("TaskQueueTool_enqueue",  TaskQueueTool(),
             {"operation": "enqueue", "queue": "test", "task": "Fetch API data"}),
            ("TaskQueueTool_dequeue",  TaskQueueTool(),
             {"operation": "dequeue", "queue": "test"}),
            ("LuaScriptAnalyzer",      LuaScriptAnalyzerTool(),
             {"code": 'while true do\n  game:GetService("Players")\n  goto continue\nend'}),
            ("TemplateRenderTool",     TemplateRenderTool(),
             {"template": "Agent: {{ name }}, Tools: {{ tools|join(', ') }}",
              "context": '{"name":"Gazcc","tools":["file","web","memory"]}'}),
        ]

        for name, tool, kwargs in tests:
            result = await tool.run(**kwargs)
            status = "✓ PASS" if result.success else "✗ FAIL"
            preview = result.output[:80].replace("\n", " ")
            print(f"  {status} | {name:30s} | {preview}")

        print("═" * 60)
        print("  All tests done. Check FAIL lines for issues.")
        print("═" * 60)

    asyncio.run(_test())
