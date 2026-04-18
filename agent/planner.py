"""
agent/planner.py
Decomposes a high-level task into concrete, ordered, dependency-aware steps.
Uses the same LLM as the agent but in a structured planning call.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Optional

import httpx


@dataclass
class Step:
    id: int
    description: str
    depends_on: list[int] = field(default_factory=list)
    tool_hint: str = ""          # suggested tool (non-binding)
    status: str = "pending"      # pending | running | done | failed | skipped
    result: str = ""
    retries: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "depends_on": self.depends_on,
            "tool_hint": self.tool_hint,
            "status": self.status,
            "result": self.result[:500] if self.result else "",
            "retries": self.retries,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Step":
        s = cls(
            id=d["id"],
            description=d["description"],
            depends_on=d.get("depends_on", []),
            tool_hint=d.get("tool_hint", ""),
        )
        s.status = d.get("status", "pending")
        s.result = d.get("result", "")
        s.retries = d.get("retries", 0)
        return s


@dataclass
class Plan:
    task: str
    steps: list[Step]
    goal: str = ""

    def pending_steps(self) -> list[Step]:
        done_ids = {s.id for s in self.steps if s.status == "done"}
        return [
            s for s in self.steps
            if s.status == "pending"
            and all(dep in done_ids for dep in s.depends_on)
        ]

    def is_complete(self) -> bool:
        return all(s.status in ("done", "skipped") for s in self.steps)

    def has_failed(self) -> bool:
        return any(s.status == "failed" for s in self.steps)

    def summary(self) -> str:
        lines = [f"Task: {self.task}", "Steps:"]
        icons = {"pending": "○", "running": "◌", "done": "✓", "failed": "✗", "skipped": "—"}
        for s in self.steps:
            icon = icons.get(s.status, "?")
            dep = f" [deps:{s.depends_on}]" if s.depends_on else ""
            hint = f" [{s.tool_hint}]" if s.tool_hint else ""
            lines.append(f"  {icon} [{s.id}]{dep}{hint} {s.description}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"task": self.task, "goal": self.goal, "steps": [s.to_dict() for s in self.steps]}

    @classmethod
    def from_dict(cls, d: dict) -> "Plan":
        steps = [Step.from_dict(s) for s in d.get("steps", [])]
        p = cls(task=d["task"], steps=steps, goal=d.get("goal", ""))
        return p


# ── Planner ───────────────────────────────────────────────────────────────────

PLAN_PROMPT = """\
You are a task planning AI. Decompose the following task into concrete, ordered steps.

TASK: {task}

AVAILABLE TOOLS:
{tools}

OUTPUT — respond ONLY with valid JSON in this exact structure:
{{
  "goal": "one-sentence description of the end goal",
  "steps": [
    {{
      "id": 1,
      "description": "exact action to perform",
      "depends_on": [],
      "tool_hint": "tool_name or empty string"
    }},
    ...
  ]
}}

RULES:
- Each step must be a single concrete action, not vague
- depends_on lists step IDs that MUST complete before this step
- tool_hint should match an available tool name exactly, or be empty
- Minimum 2 steps, maximum 15 steps
- Steps that can run independently should NOT depend on each other
- The final step should always produce or summarize the output
- Respond ONLY with JSON — no markdown, no explanation
"""


class Planner:
    def __init__(self, llm_cfg: dict, tool_schema: str):
        self._cfg = llm_cfg
        self._tool_schema = tool_schema
        self._base_url = llm_cfg.get("base_url", "https://openrouter.ai/api/v1")
        self._model = llm_cfg.get("model", "google/gemma-4-26b-a4b-it")
        self._api_key = llm_cfg.get("api_key", "")

    async def decompose(self, task: str) -> Plan:
        prompt = PLAN_PROMPT.format(task=task, tools=self._tool_schema)
        raw = await self._call_llm(prompt)
        return self._parse_plan(task, raw)

    async def _call_llm(self, prompt: str) -> str:
        # OpenRouter API: POST /v1/chat/completions (OpenAI-compatible)
        url = f"{self._base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/gazcc",
            "X-Title": "GazccThinking",
        }
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4000,
            "temperature": 0.1,
        }
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    def _parse_plan(self, task: str, raw: str) -> Plan:
        # Strip markdown fences if present
        clean = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
        clean = re.sub(r"\s*```$", "", clean.strip(), flags=re.MULTILINE)
        try:
            data = json.loads(clean)
            steps = [
                Step(
                    id=s["id"],
                    description=s["description"],
                    depends_on=s.get("depends_on", []),
                    tool_hint=s.get("tool_hint", ""),
                )
                for s in data.get("steps", [])
            ]
            return Plan(task=task, steps=steps, goal=data.get("goal", ""))
        except (json.JSONDecodeError, KeyError) as e:
            # Fallback: single step plan
            return Plan(
                task=task,
                goal=task,
                steps=[Step(id=1, description=task, depends_on=[])],
            )
