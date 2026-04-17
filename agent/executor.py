"""
agent/executor.py
Executes individual plan steps via ReAct mini-loop.
Handles retry, error correction, and self-healing.
"""

import asyncio
import json
import re
import time
from typing import Any, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .planner import Step
from .tools import ToolRegistry, ToolResult


# ── ReAct prompt ──────────────────────────────────────────────────────────────

REACT_SYSTEM = """\
You are GazccThinking, an autonomous AI agent. You execute tasks using available tools.

AVAILABLE TOOLS:
{tool_schema}

FORMAT — use EXACTLY this format, one block at a time:

Thought: [your reasoning about what to do next]
Action: [tool_name]
Action Input: [JSON object with tool arguments]

After receiving an Observation, emit the next Thought/Action/Action Input block.
When the step is fully complete, emit:

Thought: [final reasoning]
Final Answer: [complete result of this step]

RULES:
- Action Input MUST be valid JSON
- Never repeat a failed Action with the same input — change strategy
- If a tool fails twice, try a different approach or use what you already know
- Final Answer should be complete and useful for the next step
- Be concise in Thoughts; be thorough in Final Answers
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
    """Returns dict with keys: thought, action, action_input, final_answer."""
    result: dict[str, Any] = {
        "thought": "",
        "action": None,
        "action_input": {},
        "final_answer": None,
    }

    # Extract Thought
    m = re.search(r"Thought:\s*(.+?)(?=\nAction:|\nFinal Answer:|$)", text, re.DOTALL | re.IGNORECASE)
    if m:
        result["thought"] = m.group(1).strip()

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
            # Try to extract key-value pairs loosely
            result["action_input"] = {"_raw": raw_json}

    return result


# ── Executor ──────────────────────────────────────────────────────────────────

class StepExecutor:
    MAX_REACT_TURNS = 8

    def __init__(self, llm_cfg: dict, tools: ToolRegistry, retry_limit: int = 3):
        self._cfg = llm_cfg
        self._tools = tools
        self._retry_limit = retry_limit
        self._base_url = llm_cfg.get("base_url", "https://openrouter.ai/api/v1")
        self._model = llm_cfg.get("model", "anthropic/claude-sonnet-4-6")
        self._api_key = llm_cfg.get("api_key", "")

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
        messages = [
            {"role": "system", "content": REACT_SYSTEM.format(tool_schema=self._tools.schema_string())},
            {
                "role": "user",
                "content": REACT_USER.format(
                    step_description=step.description,
                    context=context[:3000] if context else "(none)",
                    memory=memory_recall[:1500] if memory_recall else "(none)",
                ),
            },
        ]

        for turn in range(self.MAX_REACT_TURNS):
            if on_event:
                on_event("turn", {"turn": turn + 1, "step_id": step.id})

            llm_output = await self._call_llm(messages)

            if on_event:
                on_event("llm_output", {"output": llm_output[:500], "step_id": step.id})

            parsed = _parse_react(llm_output)

            # ── Final Answer ──────────────────────────────────────────────────
            if parsed["final_answer"] is not None:
                return True, parsed["final_answer"]

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

        # Exhausted turns without Final Answer
        return False, "Max ReAct turns reached without completion."

    async def _call_llm(self, messages: list[dict], retry: int = 0) -> str:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "X-Title": "GazccThinking-Executor",
        }
        body = {
            "model": self._model,
            "messages": messages,
            "max_tokens": 2000,
            "temperature": 0.1,
        }
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(
                    f"{self._base_url}/chat/completions",
                    json=body,
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
        except (httpx.HTTPStatusError, httpx.TimeoutException, KeyError) as e:
            if retry < self._retry_limit:
                await asyncio.sleep(2 ** retry)
                return await self._call_llm(messages, retry + 1)
            raise RuntimeError(f"LLM call failed after {self._retry_limit} retries: {e}") from e
