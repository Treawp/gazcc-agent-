"""
agent/sandbox.py
Claude Sandbox Executor Agent v3.0
Autonomous code generation & execution with full
ANALYZE → CREATE → EXECUTE → VERIFY → DELIVER cycle.
"""

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import AsyncIterator

import httpx

from .tools import ToolRegistry, ToolResult


# ── System Prompt ─────────────────────────────────────────────────────────────

SANDBOX_SYSTEM = """\
You are Claude Sandbox Executor — an autonomous AI agent that CREATE, READ,
EDIT, EXECUTE, and MANAGE files and code in real-time.

EXECUTION CYCLE (always follow every phase):
1. ANALYZE   — understand requirements, identify files needed, plan steps
2. CREATE    — write complete working code with error handling & comments
3. EXECUTE   — run the code, capture output, check for errors
4. VERIFY    — confirm output matches expectation, fix any issues
5. DELIVER   — present final result, file structure, usage instructions

AVAILABLE TOOLS — use EXACTLY this format (no extra blank lines between):

[TOOL: create_file]
PARAMETERS: filename.ext|||file content here

[TOOL: read_file]
PARAMETERS: filename.ext

[TOOL: edit_file]
PARAMETERS: filename|||old_text|||new_text

[TOOL: append_file]
PARAMETERS: filename|||content_to_append

[TOOL: delete_file]
PARAMETERS: filename.ext

[TOOL: list_files]
PARAMETERS: .

[TOOL: execute_code]
PARAMETERS: python|||your code here

[TOOL: shell_command]
PARAMETERS: command here

[TOOL: install_package]
PARAMETERS: pip|||package_name

[TOOL: run_tests]
PARAMETERS: pytest tests/

RESPONSE FORMAT — always use this structure:
  💭 THINKING — show reasoning
  [TOOL: ...] — execute tools
  ⚡ EXECUTING — show what is running
  ✅ VERIFICATION — confirm success or report error
  📦 DELIVERY — files created, usage, next steps

MEMORY — save important context using:
  [[MEM: key=value]]

RULES:
- Never write placeholder code — everything must be functional
- When code fails: show exact error → analyze → fix → re-execute → confirm
- Never say "maybe try this" — actually FIX and RUN it
- Maintain context from previous messages
- Always include error handling, input validation, and comments
"""


# ── Tool dispatcher ───────────────────────────────────────────────────────────

class SandboxTools:
    """
    Wraps ToolRegistry and exposes sandbox-specific tool names.
    Handles parameter parsing for the |||‑delimited format.
    """

    def __init__(self, cfg: dict):
        # Force code_exec enabled for sandbox mode
        cfg_copy = {**cfg}
        cfg_copy.setdefault("tools", {})["code_exec"] = True
        self._reg = ToolRegistry(cfg_copy)

    async def dispatch(self, tool_name: str, params_str: str) -> ToolResult:
        parts = params_str.split("|||")

        # ── file tools ────────────────────────────────────────────────────────
        if tool_name == "create_file":
            if len(parts) < 2:
                return ToolResult(False, "create_file — format: filename|||content")
            path = parts[0].strip()
            content = "|||".join(parts[1:])
            return await self._reg.run("write_file", {"path": path, "content": content})

        if tool_name == "read_file":
            return await self._reg.run("read_file", {"path": parts[0].strip()})

        if tool_name == "append_file":
            if len(parts) < 2:
                return ToolResult(False, "append_file — format: filename|||content")
            return await self._reg.run("append_file", {
                "path": parts[0].strip(),
                "content": "|||".join(parts[1:]),
            })

        if tool_name == "delete_file":
            return await self._reg.run("delete_file", {"path": parts[0].strip()})

        if tool_name == "list_files":
            path = parts[0].strip() if parts and parts[0].strip() else "."
            return await self._reg.run("list_dir", {"path": path})

        # ── edit_file: read → replace → write ────────────────────────────────
        if tool_name == "edit_file":
            if len(parts) < 3:
                return ToolResult(False, "edit_file — format: filename|||old_text|||new_text")
            path = parts[0].strip()
            old_text = parts[1]
            new_text = "|||".join(parts[2:])
            read_res = await self._reg.run("read_file", {"path": path})
            if not read_res.success:
                return read_res
            if old_text not in read_res.output:
                return ToolResult(False, f"Text not found in {path}:\n{old_text[:200]}")
            new_content = read_res.output.replace(old_text, new_text, 1)
            return await self._reg.run("write_file", {"path": path, "content": new_content})

        # ── execute_code ──────────────────────────────────────────────────────
        if tool_name == "execute_code":
            if len(parts) < 2:
                return ToolResult(False, "execute_code — format: language|||code")
            lang = parts[0].strip().lower()
            code = "|||".join(parts[1:])
            if lang == "python":
                return await self._reg.run("execute_code", {"code": code})
            # Other languages via shell
            cmds = {"javascript": "node", "js": "node", "bash": "bash", "sh": "bash"}
            if lang in cmds:
                return await self._shell(f"{cmds[lang]} -e {code!r}")
            return ToolResult(False, f"Language '{lang}' not supported. Use python, js, or bash.")

        # ── shell_command ─────────────────────────────────────────────────────
        if tool_name == "shell_command":
            return await self._shell(params_str.strip())

        # ── install_package ───────────────────────────────────────────────────
        if tool_name == "install_package":
            if len(parts) < 2:
                return ToolResult(False, "install_package — format: pip|||package or npm|||package")
            manager = parts[0].strip().lower()
            package = parts[1].strip()
            if manager == "pip":
                return await self._shell(f"pip install {package} --quiet")
            if manager in ("npm", "node"):
                return await self._shell(f"npm install {package}")
            return ToolResult(False, f"Unknown package manager '{manager}'. Use pip or npm.")

        # ── run_tests ─────────────────────────────────────────────────────────
        if tool_name == "run_tests":
            return await self._shell(params_str.strip())

        return ToolResult(False, f"Unknown sandbox tool: '{tool_name}'")

    async def _shell(self, cmd: str) -> ToolResult:
        """Run a shell command via execute_code (reuses sandbox process isolation)."""
        code = (
            f"import subprocess, sys\n"
            f"r = subprocess.run({cmd!r}, shell=True, capture_output=True, text=True, timeout=30)\n"
            f"if r.stdout: print(r.stdout, end='')\n"
            f"if r.stderr: print(r.stderr, end='', file=sys.stderr)\n"
            f"sys.exit(r.returncode)\n"
        )
        return await self._reg.run("execute_code", {"code": code})


# ── Parsers ───────────────────────────────────────────────────────────────────

def _parse_tool_calls(text: str) -> list[dict]:
    """
    Extract [TOOL: name]\\nPARAMETERS: ... blocks from LLM output.
    Returns list of {"tool": str, "params": str}.
    """
    pattern = r"\[TOOL:\s*(\w+)\]\s*\nPARAMETERS:\s*(.*?)(?=\n\[TOOL:|\Z)"
    return [
        {"tool": m[0].strip(), "params": m[1].strip()}
        for m in re.findall(pattern, text, re.DOTALL)
    ]


def _parse_memory(text: str) -> dict:
    """Extract [[MEM: key=value]] markers from LLM output."""
    pattern = r"\[\[MEM:\s*(\w+)\s*=\s*(.+?)\]\]"
    return {k: v.strip() for k, v in re.findall(pattern, text)}


# ── Event type ────────────────────────────────────────────────────────────────

@dataclass
class SandboxEvent:
    type: str   # thinking | tool_call | tool_result | memory | response | error
    data: dict
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"type": self.type, "data": self.data, "timestamp": self.timestamp}


# ── Main class ────────────────────────────────────────────────────────────────

class SandboxExecutor:
    """
    Conversational code agent.
    Maintains message history and memory key-value store across turns.

    Usage:
        executor = SandboxExecutor(cfg)
        async for event in executor.chat("Buat script scraping"):
            print(event)
    """

    MAX_HISTORY = 20  # keep last N turns in context

    def __init__(self, cfg: dict):
        self._llm_cfg = cfg.get("llm", {})
        self._tools = SandboxTools(cfg)
        self._base_url = self._llm_cfg.get("base_url", "https://openrouter.ai/api/v1")
        self._model = self._llm_cfg.get("model", "minimax/minimax-m2.7")
        self._api_key = self._llm_cfg.get("api_key", "")
        self._history: list[dict] = []
        self._memory: dict[str, str] = {}

    # ── public API ────────────────────────────────────────────────────────────

    def reset(self):
        """Clear conversation history and memory."""
        self._history = []
        self._memory = {}

    @property
    def memory(self) -> dict:
        return dict(self._memory)

    @property
    def history(self) -> list[dict]:
        return list(self._history)

    def load_state(self, history: list[dict], memory: dict):
        """Restore state (for stateless API deployments)."""
        self._history = list(history)
        self._memory = dict(memory)

    async def chat(self, user_message: str) -> AsyncIterator[SandboxEvent]:
        """
        Process one user message.
        Yields SandboxEvent objects as the agent works.
        """
        self._history.append({"role": "user", "content": user_message})

        # Append memory context to system prompt
        mem_ctx = ""
        if self._memory:
            lines = "\n".join(f"  • {k}: {v}" for k, v in self._memory.items())
            mem_ctx = f"\n\n[SAVED CONTEXT]\n{lines}"

        messages = [
            {"role": "system", "content": SANDBOX_SYSTEM + mem_ctx},
            *self._history[-self.MAX_HISTORY:],
        ]

        yield SandboxEvent("thinking", {"msg": "Analyzing request..."})

        # ── Phase 1: initial LLM call ─────────────────────────────────────────
        try:
            llm_output = await self._call_llm(messages)
        except Exception as e:
            yield SandboxEvent("error", {"msg": f"LLM error: {e}"})
            self._history.pop()  # rollback
            return

        # ── Phase 2: execute all tool calls ───────────────────────────────────
        tool_calls = _parse_tool_calls(llm_output)
        observations: list[dict] = []

        for tc in tool_calls:
            yield SandboxEvent("tool_call", {
                "tool": tc["tool"],
                "params": tc["params"][:300],
            })
            result = await self._tools.dispatch(tc["tool"], tc["params"])
            obs_text = str(result)[:600]
            observations.append({
                "tool": tc["tool"],
                "success": result.success,
                "output": obs_text,
            })
            yield SandboxEvent("tool_result", {
                "tool": tc["tool"],
                "success": result.success,
                "output": obs_text,
            })

        # ── Phase 3: VERIFY + DELIVER follow-up if tools ran ─────────────────
        full_response = llm_output
        if observations:
            obs_block = "\n\n".join(
                f"[RESULT: {o['tool']}]\n{'✅' if o['success'] else '❌'} {o['output']}"
                for o in observations
            )
            messages.append({"role": "assistant", "content": llm_output})
            messages.append({
                "role": "user",
                "content": (
                    f"Tool execution results:\n{obs_block}\n\n"
                    "Now complete the VERIFY and DELIVER phases."
                ),
            })
            try:
                followup = await self._call_llm(messages)
                full_response = llm_output + "\n\n" + followup
            except Exception:
                pass  # use first response as fallback

        # ── Phase 4: persist memory markers ──────────────────────────────────
        new_mem = _parse_memory(full_response)
        if new_mem:
            self._memory.update(new_mem)
            yield SandboxEvent("memory", {"updated": new_mem, "store": dict(self._memory)})

        # ── Phase 5: store assistant turn and emit final response ─────────────
        self._history.append({"role": "assistant", "content": full_response})

        yield SandboxEvent("response", {
            "content": full_response,
            "tool_calls_executed": len(tool_calls),
            "memory": dict(self._memory),
        })

    # ── LLM call ──────────────────────────────────────────────────────────────

    async def _call_llm(self, messages: list[dict], _retry: int = 0) -> str:
        system_msg = next(
            (m["content"] for m in messages if m["role"] == "system"), ""
        )
        convo_parts = []
        for m in messages:
            if m["role"] == "system":
                continue
            label = "User" if m["role"] == "user" else "Assistant"
            convo_parts.append(f"{label}: {m['content']}")

        params: dict = {
            "question": "\n".join(convo_parts),
            "model": self._model,
        }
        if system_msg:
            params["system"] = system_msg

        headers = {"x-api-key": self._api_key}
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.get(self._base_url, params=params, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, dict):
                    return (
                        data.get("result")
                        or data.get("response")
                        or data.get("message")
                        or str(data)
                    )
                return str(data)
        except Exception as e:
            if _retry < 3:
                await asyncio.sleep(2 ** _retry)
                return await self._call_llm(messages, _retry + 1)
            raise RuntimeError(f"LLM call failed after retries: {e}") from e
