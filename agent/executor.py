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

from .planner import Step
from .tools import ToolRegistry, ToolResult



# ── ReAct prompt ──────────────────────────────────────────────────────────────

REACT_SYSTEM = """\
You are GazccAgent Pro, operating as a Strategic Partner — not just an executor.
You combine two internal voices:

[EXECUTOR] — Gets things done. Uses tools efficiently. Delivers concrete results.
[CRITIC]   — Senior Architect mindset. Reviews EXECUTOR's work. Catches:
              • Architectural flaws (wrong abstraction, fragile design)
              • Missing error handling or edge cases
              • Opportunities for reuse or simplification
              • Security / correctness issues
              The CRITIC proposes SPECIFIC fixes, not vague complaints.

AVAILABLE TOOLS:
{tool_schema}

FORMAT — use EXACTLY this format, one block at a time:

Thought: [EXECUTOR reasoning about what to do next]
Action: [tool_name]
Action Input: [JSON object with tool arguments]

After an Observation, optionally add a CRITIC review:
Critic: [architectural observation or improvement — skip if nothing to add]

Then continue with next Thought/Action or conclude:

Thought: [final synthesis]
Final Answer: [complete result — include any CRITIC-suggested improvements inline]

RULES:
- Action Input MUST be valid JSON
- Never repeat a failed Action with the same input — change strategy
- CRITIC should speak after significant steps, not trivially
- If CRITIC identifies a flaw, EXECUTOR must address it before Final Answer
- Final Answer should be complete, reviewed, and production-quality
- Be concise in Thoughts; be thorough in Final Answers
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

    def __init__(self, llm_cfg: dict, tools: ToolRegistry, retry_limit: int = 3):
        self._cfg = llm_cfg
        self._tools = tools
        self._retry_limit = retry_limit
        self._base_url = llm_cfg.get("base_url", "https://openrouter.ai/api/v1")
        self._model = llm_cfg.get("model", "moonshotai/kimi-k2.5")
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

            llm_output = await self._call_llm(messages, on_event=on_event)

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

    async def _call_llm(self, messages: list[dict], retry: int = 0, on_event=None) -> str:
        """
        Call LLM and auto-continue if output was truncated (finish_reason == 'length').
        Concatenates all partial outputs into one complete response.
        Max 6 continuation rounds to prevent infinite loops.
        """
        url = f"{self._base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/gazcc",
            "X-Title": "GazccThinking",
        }

        full_output_parts: list[str] = []
        current_messages = list(messages)
        MAX_CONTINUATION_ROUNDS = 3

        for continuation_round in range(MAX_CONTINUATION_ROUNDS):
            payload = {
                "model": self._model,
                "messages": current_messages,
                "max_tokens": 4000,
                "temperature": 0.1,
            }
            try:
                async with httpx.AsyncClient(timeout=120) as client:
                    resp = await client.post(url, json=payload, headers=headers)
                    resp.raise_for_status()
                    data = resp.json()

                choice = data["choices"][0]
                content: str = choice["message"]["content"] or ""
                finish_reason: str = choice.get("finish_reason", "stop")

                full_output_parts.append(content)

                combined = "".join(full_output_parts)
                # Stop kalau selesai natural ATAU udah ada Final Answer di output
                if finish_reason != "length" or "Final Answer:" in combined:
                    break

                # Output was truncated — notify frontend and continue
                # This fires the existing on_event callback so the UI shows "Continuing..."
                if on_event:
                    on_event("continuation", {
                        "round": continuation_round + 1,
                        "msg": f"Output truncated (finish_reason=length), auto-continuing... round {continuation_round + 1}/{MAX_CONTINUATION_ROUNDS}"
                    })

                # Append the partial assistant turn and request continuation
                current_messages = list(current_messages) + [
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": (
                            "Your previous response was cut off because it was too long. "
                            "Continue EXACTLY from where you stopped — do NOT repeat any previous text. "
                            "Just continue the output seamlessly."
                        ),
                    },
                ]

            except (httpx.HTTPStatusError, httpx.TimeoutException, KeyError) as e:
                if retry < self._retry_limit:
                    await asyncio.sleep(2 ** retry)
                    return await self._call_llm(messages, retry + 1, on_event=on_event)
                raise RuntimeError(f"LLM call failed after {self._retry_limit} retries: {e}") from e

        return "".join(full_output_parts)
