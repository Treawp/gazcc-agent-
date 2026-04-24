"""
agent/learning.py
GazccAgent Learning Component — tracks outcomes, learns from history,
improves future planning & execution based on past success/failure patterns.

Components:
  - OutcomeTracker  : records task & step outcomes with metadata
  - PatternAnalyzer : derives insights from recorded outcomes
  - EvaluationEngine: LLM-based quality check after task completion
  - LearningMemory  : persistent learning store (wraps FileMemory/Redis)
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

logger = logging.getLogger("gazcc.learning")


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class StepOutcome:
    step_id: int
    description: str
    tool_hint: str
    success: bool
    retries: int
    duration_s: float
    error: str = ""


@dataclass
class TaskOutcome:
    task_id: str
    task: str
    success: bool
    steps_done: int
    steps_total: int
    elapsed_s: float
    eval_score: float = 0.0          # 0-10, set by EvaluationEngine
    eval_summary: str = ""
    step_outcomes: list[StepOutcome] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "task": self.task,
            "success": self.success,
            "steps_done": self.steps_done,
            "steps_total": self.steps_total,
            "elapsed_s": round(self.elapsed_s, 2),
            "eval_score": self.eval_score,
            "eval_summary": self.eval_summary,
            "tools_used": self.tools_used,
            "step_outcomes": [
                {
                    "step_id": s.step_id,
                    "description": s.description,
                    "tool_hint": s.tool_hint,
                    "success": s.success,
                    "retries": s.retries,
                    "duration_s": round(s.duration_s, 2),
                    "error": s.error,
                }
                for s in self.step_outcomes
            ],
            "timestamp": self.timestamp,
        }


@dataclass
class LearningInsight:
    """Structured insight derived from past outcomes."""
    insight_type: str          # tool_preference | avoid_tool | step_pattern | risk_flag
    subject: str               # tool name, step pattern, etc.
    detail: str
    confidence: float          # 0.0 - 1.0
    source_tasks: int          # how many tasks contributed

    def to_prompt_line(self) -> str:
        return f"[{self.insight_type.upper()}] {self.subject}: {self.detail} (confidence={self.confidence:.0%}, from {self.source_tasks} tasks)"


# ── EvaluationEngine ──────────────────────────────────────────────────────────

EVAL_PROMPT = """\
You are GazccAgent Evaluator — a ruthless quality auditor.

ORIGINAL TASK:
{task}

AGENT OUTPUT:
{output}

EVALUATION CRITERIA:
1. Completeness — Did the agent fully address ALL aspects of the task?
2. Accuracy     — Is the output factually correct and logically sound?
3. Quality      — Is the output well-structured, clear, and actionable?
4. Efficiency   — Were there unnecessary steps or tool calls?
5. Goal Met     — Did the final output satisfy the user's original intent?

Respond ONLY with valid JSON:
{{
  "score": <float 0-10>,
  "completeness": <float 0-10>,
  "accuracy": <float 0-10>,
  "quality": <float 0-10>,
  "efficiency": <float 0-10>,
  "goal_met": <bool>,
  "strengths": ["what was done well"],
  "weaknesses": ["what was missing or wrong"],
  "retry_recommended": <bool>,
  "retry_reason": "why retry is needed, or empty string",
  "summary": "one-sentence overall verdict"
}}
"""


class EvaluationEngine:
    """Calls LLM to evaluate task output quality."""

    def __init__(self, llm_cfg: dict):
        self._cfg = llm_cfg
        self._base_url = llm_cfg.get("base_url", "https://openrouter.ai/api/v1")
        self._model = llm_cfg.get("model", "deepseek/deepseek-v4-flash")
        self._api_key = llm_cfg.get("api_key", "")

    async def evaluate(self, task: str, output: str) -> dict:
        """
        Returns dict with score (0-10), goal_met, retry_recommended, summary, etc.
        Never raises — returns safe defaults on error.
        """
        if not self._api_key:
            return self._default_eval()

        prompt = EVAL_PROMPT.format(
            task=task[:1500],
            output=output[:3000],
        )
        try:
            raw = await self._call_llm(prompt)
            clean = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
            clean = re.sub(r"\s*```$", "", clean.strip(), flags=re.MULTILINE)
            data = json.loads(clean)
            return data
        except Exception as e:
            logger.warning(f"EvaluationEngine error: {e}")
            return self._default_eval()

    def _default_eval(self) -> dict:
        return {
            "score": 7.0,
            "completeness": 7.0,
            "accuracy": 7.0,
            "quality": 7.0,
            "efficiency": 7.0,
            "goal_met": True,
            "strengths": [],
            "weaknesses": [],
            "retry_recommended": False,
            "retry_reason": "",
            "summary": "Evaluation skipped (no API key or LLM error)",
        }

    async def _call_llm(self, prompt: str) -> str:
        url = f"{self._base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/gazcc",
            "X-Title": "GazccThinking-Evaluator",
        }
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000,
            "temperature": 0.1,
        }
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]


# ── PatternAnalyzer ───────────────────────────────────────────────────────────

class PatternAnalyzer:
    """
    Derives actionable insights from a list of TaskOutcomes.
    These insights are injected into the Planner's memory context
    so future plans benefit from past experience.
    """

    def analyze(self, outcomes: list[TaskOutcome]) -> list[LearningInsight]:
        if not outcomes:
            return []

        insights: list[LearningInsight] = []

        # ── Tool performance analysis
        tool_stats: dict[str, dict] = {}
        for outcome in outcomes:
            for step in outcome.step_outcomes:
                t = step.tool_hint or "unknown"
                if t not in tool_stats:
                    tool_stats[t] = {"success": 0, "fail": 0, "retries": 0, "count": 0}
                tool_stats[t]["count"] += 1
                if step.success:
                    tool_stats[t]["success"] += 1
                else:
                    tool_stats[t]["fail"] += 1
                tool_stats[t]["retries"] += step.retries

        for tool, stats in tool_stats.items():
            if stats["count"] < 2:
                continue
            success_rate = stats["success"] / stats["count"]
            avg_retries = stats["retries"] / stats["count"]
            if success_rate >= 0.85:
                insights.append(LearningInsight(
                    insight_type="tool_preference",
                    subject=tool,
                    detail=f"Reliable tool — {success_rate:.0%} success rate across {stats['count']} uses",
                    confidence=min(success_rate, 0.95),
                    source_tasks=stats["count"],
                ))
            elif success_rate <= 0.4:
                insights.append(LearningInsight(
                    insight_type="avoid_tool",
                    subject=tool,
                    detail=f"Unreliable — only {success_rate:.0%} success, avg {avg_retries:.1f} retries. Consider alternative.",
                    confidence=0.7,
                    source_tasks=stats["count"],
                ))

        # ── Task success rate
        total = len(outcomes)
        successes = sum(1 for o in outcomes if o.success)
        if total >= 3:
            success_rate = successes / total
            if success_rate < 0.6:
                insights.append(LearningInsight(
                    insight_type="risk_flag",
                    subject="overall_task_success",
                    detail=f"Only {success_rate:.0%} of recent tasks succeeded. Agent may need more careful planning or tool selection.",
                    confidence=0.8,
                    source_tasks=total,
                ))

        # ── Avg eval scores
        scored = [o for o in outcomes if o.eval_score > 0]
        if scored:
            avg_score = sum(o.eval_score for o in scored) / len(scored)
            if avg_score < 6.0:
                insights.append(LearningInsight(
                    insight_type="risk_flag",
                    subject="output_quality",
                    detail=f"Average eval score is {avg_score:.1f}/10. Output quality needs improvement.",
                    confidence=0.75,
                    source_tasks=len(scored),
                ))
            elif avg_score >= 8.5:
                insights.append(LearningInsight(
                    insight_type="step_pattern",
                    subject="output_quality",
                    detail=f"High quality outputs — avg eval score {avg_score:.1f}/10. Keep current approach.",
                    confidence=0.85,
                    source_tasks=len(scored),
                ))

        return insights

    def to_planner_context(self, insights: list[LearningInsight]) -> str:
        """Format insights as a block for injecting into the planner prompt."""
        if not insights:
            return "(no learning insights yet)"
        lines = ["=== LEARNING INSIGHTS FROM PAST TASKS ==="]
        for ins in insights[:8]:  # cap at 8 to avoid token bloat
            lines.append(ins.to_prompt_line())
        lines.append("==========================================")
        return "\n".join(lines)


# ── OutcomeTracker ────────────────────────────────────────────────────────────

class OutcomeTracker:
    """
    Builds a TaskOutcome object during agent execution,
    then persists it to memory for future analysis.
    """

    def __init__(self, task_id: str, task: str):
        self.task_id = task_id
        self.task = task
        self._step_starts: dict[int, float] = {}
        self.outcome = TaskOutcome(
            task_id=task_id,
            task=task,
            success=False,
            steps_done=0,
            steps_total=0,
            elapsed_s=0.0,
        )

    def step_started(self, step_id: int):
        self._step_starts[step_id] = time.time()

    def step_completed(self, step_id: int, description: str, tool_hint: str,
                       success: bool, retries: int = 0, error: str = ""):
        duration = time.time() - self._step_starts.get(step_id, time.time())
        self.outcome.step_outcomes.append(StepOutcome(
            step_id=step_id,
            description=description,
            tool_hint=tool_hint,
            success=success,
            retries=retries,
            duration_s=duration,
            error=error,
        ))
        if tool_hint and tool_hint not in self.outcome.tools_used:
            self.outcome.tools_used.append(tool_hint)

    def finalize(self, success: bool, steps_done: int, steps_total: int,
                 elapsed_s: float, eval_result: dict | None = None):
        self.outcome.success = success
        self.outcome.steps_done = steps_done
        self.outcome.steps_total = steps_total
        self.outcome.elapsed_s = elapsed_s
        if eval_result:
            self.outcome.eval_score = float(eval_result.get("score", 0.0))
            self.outcome.eval_summary = eval_result.get("summary", "")


# ── LearningSystem ────────────────────────────────────────────────────────────

class LearningSystem:
    """
    Top-level learning facade. Wires together:
      OutcomeTracker → persist → PatternAnalyzer → insights → Planner context

    Usage in core.py:
        self._learning = LearningSystem(llm_cfg, memory_backend)

        # Start tracking
        tracker = self._learning.start_task(task_id, task)

        # Track steps
        tracker.step_started(step_id)
        tracker.step_completed(step_id, ...)

        # After task done — evaluate & learn
        eval_result = await self._learning.finalize_task(tracker, success, steps_done, steps_total, elapsed, output)

        # Inject into planner
        planner.set_learning_context(await self._learning.get_insights_context())
    """

    _OUTCOMES_KEY = "gazcc_learning_outcomes"
    _MAX_STORED = 50  # keep last N outcomes

    def __init__(self, llm_cfg: dict, memory_backend: Any):
        self._evaluator = EvaluationEngine(llm_cfg)
        self._analyzer = PatternAnalyzer()
        self._memory = memory_backend
        self._cached_insights: list[LearningInsight] = []

    def start_task(self, task_id: str, task: str) -> OutcomeTracker:
        return OutcomeTracker(task_id=task_id, task=task)

    async def finalize_task(
        self,
        tracker: OutcomeTracker,
        success: bool,
        steps_done: int,
        steps_total: int,
        elapsed_s: float,
        output: str,
    ) -> dict:
        """
        1. Evaluate output quality via LLM
        2. Finalize tracker
        3. Persist outcome to memory
        4. Recompute insights cache
        Returns eval_result dict.
        """
        eval_result = await self._evaluator.evaluate(tracker.task, output)
        tracker.finalize(success, steps_done, steps_total, elapsed_s, eval_result)

        # Persist
        await self._persist_outcome(tracker.outcome)

        # Recompute insights
        outcomes = await self._load_outcomes()
        self._cached_insights = self._analyzer.analyze(outcomes)

        logger.info(
            f"[LEARNING] Task {tracker.task_id} — "
            f"eval_score={tracker.outcome.eval_score:.1f}/10 | "
            f"goal_met={eval_result.get('goal_met')} | "
            f"retry={eval_result.get('retry_recommended')}"
        )
        return eval_result

    async def get_insights_context(self) -> str:
        """Return formatted insights string for planner injection."""
        if not self._cached_insights:
            outcomes = await self._load_outcomes()
            self._cached_insights = self._analyzer.analyze(outcomes)
        return self._analyzer.to_planner_context(self._cached_insights)

    async def _persist_outcome(self, outcome: TaskOutcome):
        """Append outcome to rolling list in memory (capped at _MAX_STORED)."""
        try:
            raw_entry = await self._memory.retrieve(self._OUTCOMES_KEY)
            existing: list[dict] = []
            if raw_entry:
                try:
                    existing = json.loads(raw_entry.content)
                except Exception:
                    existing = []

            existing.append(outcome.to_dict())
            # Keep last N
            if len(existing) > self._MAX_STORED:
                existing = existing[-self._MAX_STORED:]

            await self._memory.store(
                key=self._OUTCOMES_KEY,
                content=json.dumps(existing, ensure_ascii=False),
                metadata={"type": "learning_outcomes", "count": len(existing)},
            )
        except Exception as e:
            logger.warning(f"[LEARNING] Failed to persist outcome: {e}")

    async def _load_outcomes(self) -> list[TaskOutcome]:
        """Load stored outcomes from memory."""
        try:
            raw_entry = await self._memory.retrieve(self._OUTCOMES_KEY)
            if not raw_entry:
                return []
            data = json.loads(raw_entry.content)
            outcomes = []
            for d in data:
                o = TaskOutcome(
                    task_id=d["task_id"],
                    task=d["task"],
                    success=d["success"],
                    steps_done=d["steps_done"],
                    steps_total=d["steps_total"],
                    elapsed_s=d["elapsed_s"],
                    eval_score=d.get("eval_score", 0.0),
                    eval_summary=d.get("eval_summary", ""),
                    tools_used=d.get("tools_used", []),
                    timestamp=d.get("timestamp", 0.0),
                )
                o.step_outcomes = [
                    StepOutcome(
                        step_id=s["step_id"],
                        description=s["description"],
                        tool_hint=s["tool_hint"],
                        success=s["success"],
                        retries=s["retries"],
                        duration_s=s["duration_s"],
                        error=s.get("error", ""),
                    )
                    for s in d.get("step_outcomes", [])
                ]
                outcomes.append(o)
            return outcomes
        except Exception as e:
            logger.warning(f"[LEARNING] Failed to load outcomes: {e}")
            return []
