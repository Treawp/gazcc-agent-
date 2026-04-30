"""
agent/executor.py
Executes individual plan steps via ReAct mini-loop.
Handles retry, error correction, and self-healing.
"""

import asyncio
import json
import re
import time
import uuid
from typing import Any, Optional

import httpx

from .planner import Step
from .tools import ToolRegistry, ToolResult
from .fact_guard import FactGuardInterceptor



# ── ReAct prompt ──────────────────────────────────────────────────────────────

REACT_SYSTEM = """\
You are GazccAgent Pro. You have tools. Use them. Deliver results.

━━━ FILE CREATION — MANDATORY PROTOCOL ━━━
When user asks for a file/download, ALWAYS do EXACTLY this — no exceptions:

Thought: User wants a file. I will write it then export it.
Action: write_file
Action Input: {{"path": "/tmp/output.txt", "content": "...file content here..."}}
Observation: [file written]
Thought: Now export it as downloadable.
Action: export_file_base64
Action Input: {{"path": "/tmp/output.txt"}}
Observation: [FILE_EXPORT:output.txt:base64data...]
Thought: Done. Include the marker in Final Answer.
Final Answer: File siap! [FILE_EXPORT:output.txt:base64data...]

RULES (non-negotiable):
- NEVER say "saya tidak bisa membuat file" — write_file is ALWAYS available
- NEVER invent fake conditions like "[TOOLS TERSEDIA UNTUK TASK INI]" — does NOT exist
- NEVER refuse file creation — just DO IT with write_file + export_file_base64
- NEVER write a [FILE_EXPORT:...] marker manually — must come from export_file_base64 tool
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

AVAILABLE TOOLS:
{tool_schema}

FORMAT:
Thought: [reasoning]
Action: [tool_name]
Action Input: [JSON]
(after tool result → continue or conclude)
Final Answer: [complete result]

RULES:
- Action Input MUST be valid JSON
- Never repeat a failed Action with same input — change strategy
- Final Answer should be complete and production-quality
"""

CRITIC_REVIEW_PROMPT = """\
You are a Senior Software Architect reviewing this work:

STEP: {step}
RESULT: {result}

Provide a BRIEF architectural critique (2-3 sentences max):
1. Is there a structural flaw, missing case, or fragile assumption?
2. If yes: propose ONE specific, actionable improvement.
3. If the work is solid, say "LGTM — no architectural concerns."

Be specific. Avoid vague praise. Respond directly without preamble.
"""

REACT_USER = """\
STEP TO EXECUTE: {step_description}

CONTEXT FROM PREVIOUS STEPS:
{context}

MEMORY RECALL:
{memory}

Execute this step now. Think step by step.
"""


# ── parse ReAct output ────────────────────────────────────────────────────────

def _parse_react(text: str) -> dict:
    """Returns dict with keys: thought, action, action_input, final_answer, critic."""
    result: dict[str, Any] = {
        "thought": "",
        "action": None,
        "action_input": {},
        "final_answer": None,
        "critic": "",
    }

    # Extract Thought
    m = re.search(r"Thought:\s*(.+?)(?=\nAction:|\nFinal Answer:|\nCritic:|$)", text, re.DOTALL | re.IGNORECASE)
    if m:
        result["thought"] = m.group(1).strip()

    # Extract Critic
    critic = re.search(r"Critic:\s*(.+?)(?=\nThought:|\nAction:|\nFinal Answer:|$)", text, re.DOTALL | re.IGNORECASE)
    if critic:
        result["critic"] = critic.group(1).strip()

    # Extract Final Answer
    fa = re.search(r"Final Answer:\s*([\s\S]+)$", text, re.IGNORECASE)
    if fa:
        result["final_answer"] = fa.group(1).strip()
        return result

    # Extract Action
    act = re.search(r"Action:\s*(\w+)", text, re.IGNORECASE)
    if act:
        result["action"] = act.group(1).strip()

    # Extract Action Input (JSON)
    ai = re.search(r"Action Input:\s*(\{[\s\S]*?\})(?:\n|$)", text, re.IGNORECASE)
    if ai:
        raw_json = ai.group(1).strip()
        try:
            result["action_input"] = json.loads(raw_json)
        except json.JSONDecodeError:
            result["action_input"] = {"_raw": raw_json}

    return result


# ── Executor ──────────────────────────────────────────────────────────────────

class StepExecutor:
    MAX_REACT_TURNS = 8
    MAX_HISTORY_PAIRS = 6      # window: keep last N user/assistant pairs
    MAX_CONTEXT_CHARS = 3000   # truncate step context from previous steps
    MAX_MEMORY_CHARS = 1200    # truncate memory recall

    def __init__(self, llm_cfg: dict, tools: ToolRegistry, retry_limit: int = 3, agent_cfg: dict | None = None):
        self._cfg = llm_cfg
        self._tools = tools
        self._retry_limit = retry_limit
        self._base_url = llm_cfg.get("base_url", "https://api.covenant.sbs/api/ai/gemini")
        self._model = llm_cfg.get("model", "gemini")
        self._api_key = llm_cfg.get("api_key", "")
        self._fact_guard = FactGuardInterceptor(agent_cfg or {})

    def _build_system(self, step_tool_hint: str = "") -> str:
        """
        Per-step schema injection.
        If step has a tool_hint, inject full schema only for that tool + slim core.
        Otherwise use slim_schema_string() — saves ~1500-2800 tok vs full schema.
        """
        if step_tool_hint and hasattr(self._tools, "step_schema_string"):
            schema = self._tools.step_schema_string([step_tool_hint])
        elif hasattr(self._tools, "slim_schema_string"):
            schema = self._tools.slim_schema_string()
        else:
            schema = self._tools.schema_string()
        return REACT_SYSTEM.format(tool_schema=schema)

    @staticmethod
    def _window_messages(messages: list[dict], max_pairs: int) -> list[dict]:
        """
        Keep system message + first user message + last N user/assistant pairs.
        Prevents unbounded history growth during long ReAct loops.
        """
        if len(messages) <= 2 + max_pairs * 2:
            return messages
        system = [m for m in messages if m["role"] == "system"]
        turns = [m for m in messages if m["role"] != "system"]
        # Always keep first user message (task description)
        first = turns[:1]
        # Keep last max_pairs * 2 messages
        recent = turns[-(max_pairs * 2):]
        # Avoid duplicate if first is already in recent
        if recent and first and recent[0] == first[0]:
            return system + recent
        return system + first + recent

    async def execute_step(
        self,
        step: Step,
        context: str = "",
        memory_recall: str = "",
        on_event=None,
    ) -> tuple[bool, str]:
        """
        Runs a single plan step through the ReAct loop.
        Returns (success, result_string).
        on_event(event_type, data) called for streaming progress.
        """
        # Truncate inputs — don't let context bloat the prompt
        ctx_trimmed  = context[:self.MAX_CONTEXT_CHARS] + ("…" if len(context) > self.MAX_CONTEXT_CHARS else "") if context else "(none)"
        mem_trimmed  = memory_recall[:self.MAX_MEMORY_CHARS] + ("…" if len(memory_recall) > self.MAX_MEMORY_CHARS else "") if memory_recall else "(none)"

        messages = [
            {"role": "system", "content": self._build_system(getattr(step, "tool_hint", ""))},
            {
                "role": "user",
                "content": REACT_USER.format(
                    step_description=step.description,
                    context=ctx_trimmed,
                    memory=mem_trimmed,
                ),
            },
        ]

        for turn in range(self.MAX_REACT_TURNS):
            if on_event:
                on_event("turn", {"turn": turn + 1, "step_id": step.id})

            llm_output = await self._call_llm(messages, on_event=on_event)

            if on_event:
                on_event("llm_output", {"output": llm_output[:500], "step_id": step.id})

            parsed = _parse_react(llm_output)

            # ── Final Answer ──────────────────────────────────────────────────
            if parsed["final_answer"] is not None:
                answer = parsed["final_answer"]
                answer = await self._fact_guard.process(answer)  # ← FACT GUARD
                return True, answer

            # ── Tool call ─────────────────────────────────────────────────────
            action = parsed.get("action")
            if not action:
                # LLM didn't follow format — push it
                messages.append({"role": "assistant", "content": llm_output})
                messages.append({
                    "role": "user",
                    "content": (
                        "You must use the format:\n"
                        "Thought: ...\nAction: <tool_name>\nAction Input: {\"key\": \"value\"}\n\n"
                        "OR:\nThought: ...\nFinal Answer: <result>\n\n"
                        f"Available tools: {list(self._tools._tools.keys())}"
                    ),
                })
                continue

            action_input = parsed.get("action_input", {})

            # Handle raw string input
            if "_raw" in action_input:
                action_input = {"input": action_input["_raw"]}

            if on_event:
                on_event("tool_call", {"tool": action, "input": action_input, "step_id": step.id})

            tool_result: ToolResult = await self._tools.run(action, action_input)

            observation = str(tool_result)
            if len(observation) > 4000:
                observation = observation[:4000] + "\n... [truncated]"

            if on_event:
                on_event("observation", {"tool": action, "success": tool_result.success, "output": observation[:300], "step_id": step.id})

            messages.append({"role": "assistant", "content": llm_output})
            messages.append({"role": "user", "content": f"Observation: {observation}\n\nContinue."})

            # Window history to prevent unbounded growth
            messages = self._window_messages(messages, self.MAX_HISTORY_PAIRS)

        # Exhausted turns without Final Answer
        return False, "Max ReAct turns reached without completion."

    async def _call_llm(self, messages: list[dict], retry: int = 0, on_event=None) -> str:
        """Call Covenant LLM API (form-data format)."""
        url = self._base_url.rstrip('/')
        # Konversi messages[] → question + system (format Covenant)
        system = "Kamu adalah GazccAI, AI agent cerdas yang menyelesaikan task secara terstruktur."
        question = ""
        for m in messages:
            if m.get("role") == "system":
                system = m.get("content", system)
            elif m.get("role") == "user":
                question = m.get("content", "")
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(
                    url,
                    headers={"x-api-key": self._api_key},
                    files=[
                        ("question", (None, question)),
                        ("system", (None, system)),
                        ("sessionId", (None, str(uuid.uuid4())[:8])),
                    ],
                )
                resp.raise_for_status()
                data = resp.json()
            return data["data"]["result"] or ""
        except (httpx.HTTPStatusError, httpx.TimeoutException, KeyError) as e:
            if retry < self._retry_limit:
                await asyncio.sleep(2 ** retry)
                return await self._call_llm(messages, retry + 1, on_event=on_event)
            raise RuntimeError(f"LLM call failed after {self._retry_limit} retries: {e}") from e
