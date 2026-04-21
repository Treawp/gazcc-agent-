"""
agent/l99.py
═══════════════════════════════════════════════════════════════════════════════
  ██╗      ██████╗  █████╗
  ██║     ██╔════╝ ██╔══██╗
  ██║     ██║  ███╗╚██████║
  ██║     ██║   ██║ ╚═══██║
  ███████╗╚██████╔╝  █████╔╝
  ╚══════╝ ╚═════╝  ╚════╝

  GAZCC L99 — GOD-TIER AGENT UPGRADE
  ══════════════════════════════════════════════════════════════════════════════
  Protocol Stack:
    LAYER 0  │ EmpathyBuffer      — Sentiment & context scanner (anti-hallucination)
    LAYER 1  │ PreMortem Engine   — Risk & failure analysis BEFORE execution
    LAYER 2  │ DualPersona        — [ARCHITECT] strategy + [ENGINEER] precision
    LAYER 3  │ SelfEvolution      — Observes feedback, persists internal weights
    LAYER 4  │ GodTierPlanner     — Chain-of-Thought planning with structured output
    LAYER 5  │ GodTierExecutor    — Full ReAct loop with structured response format
    LAYER 6  │ GodTierAgent       — Drop-in replacement for GazccAgent

  HOW TO ACTIVATE:
    # Option A — Replace existing agent (recommended)
    from agent.l99 import GodTierAgent
    agent = GodTierAgent(config)
    result = await agent.run("your task")

    # Option B — Patch existing instance
    from agent.l99 import apply_l99_upgrade
    agent = apply_l99_upgrade(existing_gazcc_agent)

  OUTPUT FORMAT (every task response):
    [ANALYSIS]        — Hidden complexity breakdown & requirements
    [STRATEGY]        — Architectural approach, step-by-step
    [PRE_MORTEM]      — Risks, failure modes, contingency plans
    [IMPLEMENTATION]  — High-precision solution output
    [CRITICAL_REVIEW] — Self-critique, trade-offs, next improvements

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Optional

import aiofiles
import httpx

from .core import GazccAgent, AgentEvent, AgentResult
from .executor import StepExecutor, _parse_react
from .memory import build_memory
from .planner import Planner, Plan, Step
from .tools import ToolRegistry
from .strategic_tools import (
    SemanticMemoryTool,
    ProactiveMonitorTool,
    ApiBridgeTool,
    SandboxExecutorTool,
)

logger = logging.getLogger("gazcc.l99")


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 0 — EMPATHY BUFFER
# Sentiment & context scanner. Determines operational mode before ANY response.
# Prevents hallucination by routing input to correct processing pipeline.
# ══════════════════════════════════════════════════════════════════════════════

class OperationalMode:
    CONVERSATIONAL  = "CONVERSATIONAL"   # Appreciation detected → build rapport
    SELF_REFLECT    = "SELF_REFLECT"     # Correction/anger → analyze error, update
    EXECUTION       = "EXECUTION"        # Command/task → trigger full pipeline
    CLARIFICATION   = "CLARIFICATION"   # Ambiguous → ask before executing
    STANDARD        = "STANDARD"        # Default operational mode


@dataclass
class EmpathyResult:
    mode: str
    sentiment_score: float          # -1.0 (negative) → +1.0 (positive)
    detected_intent: str
    is_correction: bool
    is_ambiguous: bool
    urgency: str                    # LOW | MEDIUM | HIGH | CRITICAL
    summary: str


class EmpathyBuffer:
    """
    LAYER 0: Sentiment & Context Scanner.

    Analyzes user input BEFORE the agent acts.
    Determines the correct operational mode to prevent wasted cycles
    and 'hallucination drift' caused by misread intent.

    Uses multi-signal analysis:
      - Keyword matching (fast, zero-latency)
      - Sentence structure heuristics
      - Punctuation & capitalization signals
      - Correction pattern detection
    """

    # Sentiment signal dictionaries
    _APPRECIATION_SIGNALS = {
        "keywords": [
            "thanks", "thank you", "terima kasih", "makasih", "great",
            "awesome", "perfect", "nice", "good job", "well done",
            "bagus", "mantap", "keren", "hebat", "luar biasa", "oke banget",
        ],
        "patterns": [r"👍|🙏|❤️|⭐|✅"],
    }

    _CORRECTION_SIGNALS = {
        "keywords": [
            "wrong", "incorrect", "error", "salah", "bukan", "bukan begitu",
            "fix this", "that's not", "you misunderstood", "no no", "not right",
            "kenapa", "kok", "huh", "what", "wait", "hold on", "actually",
        ],
        "patterns": [r"!!!|WTF|wtf|harus\s+\w+\s+bukan"],
    }

    _COMMAND_SIGNALS = {
        "starters": [
            "create", "build", "make", "write", "generate", "implement",
            "analyze", "research", "find", "fix", "refactor", "deploy",
            "buatkan", "buat", "tulis", "cari", "analisis", "perbaiki",
            "tolong", "please", "can you", "could you", "i need", "bantu",
        ],
        "patterns": [r"step\s+\d|todo|todo list|\bDO\b|\bRUN\b"],
    }

    _AMBIGUITY_SIGNALS = [
        "maybe", "perhaps", "i think", "not sure", "or maybe", "mungkin",
        "kayaknya", "sepertinya", "entah", "gimana ya", "??"
    ]

    def analyze(self, user_input: str) -> EmpathyResult:
        text = user_input.strip()
        text_lower = text.lower()
        words = set(re.findall(r"\b\w+\b", text_lower))

        # ── Score signals ──────────────────────────────────────────────────
        appreciation_score = self._score(text_lower, words, self._APPRECIATION_SIGNALS)
        correction_score   = self._score(text_lower, words, self._CORRECTION_SIGNALS)
        command_score      = self._score_commands(text_lower, words)
        ambiguity_score    = sum(1 for s in self._AMBIGUITY_SIGNALS if s in text_lower)

        # ── Detect all-caps urgency (e.g. "FIX THIS NOW") ─────────────────
        caps_ratio  = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        has_urgency = caps_ratio > 0.4 or "urgent" in text_lower or "asap" in text_lower

        # ── Correction pattern: direct negation ───────────────────────────
        is_correction = (
            correction_score > 0 or
            bool(re.search(r"(bukan|not|salah|wrong)\s+\w+", text_lower)) or
            bool(re.search(r"(harusnya|seharusnya|should be)", text_lower))
        )

        # ── Ambiguity check ───────────────────────────────────────────────
        is_ambiguous = (
            ambiguity_score >= 2 or
            (text.endswith("?") and command_score == 0)
        )

        # ── Sentiment score ────────────────────────────────────────────────
        sentiment = min(1.0, appreciation_score * 0.5) - min(1.0, correction_score * 0.4)

        # ── Mode routing ───────────────────────────────────────────────────
        if appreciation_score > 0 and command_score == 0 and not is_correction:
            mode   = OperationalMode.CONVERSATIONAL
            intent = "User expressing appreciation or positive feedback"
        elif is_correction:
            mode   = OperationalMode.SELF_REFLECT
            intent = "User providing correction or expressing dissatisfaction"
        elif is_ambiguous and command_score < 2:
            mode   = OperationalMode.CLARIFICATION
            intent = "Ambiguous intent — clarification needed before execution"
        elif command_score > 0:
            mode   = OperationalMode.EXECUTION
            intent = "Task command detected — trigger full execution pipeline"
        else:
            mode   = OperationalMode.STANDARD
            intent = "Standard informational or conversational input"

        urgency = "CRITICAL" if has_urgency else (
            "HIGH" if command_score > 3 else (
                "MEDIUM" if command_score > 0 else "LOW"
            )
        )

        return EmpathyResult(
            mode=mode,
            sentiment_score=round(sentiment, 2),
            detected_intent=intent,
            is_correction=is_correction,
            is_ambiguous=is_ambiguous,
            urgency=urgency,
            summary=f"[{mode}] {intent} (urgency={urgency}, sentiment={sentiment:+.2f})",
        )

    def _score(self, text_lower: str, words: set, signals: dict) -> float:
        score = sum(1.0 for kw in signals.get("keywords", []) if kw in text_lower)
        for pat in signals.get("patterns", []):
            if re.search(pat, text_lower):
                score += 1.5
        return score

    def _score_commands(self, text_lower: str, words: set) -> float:
        score = 0.0
        for starter in self._COMMAND_SIGNALS["starters"]:
            if text_lower.startswith(starter) or f" {starter} " in text_lower:
                score += 1.5
        for pat in self._COMMAND_SIGNALS["patterns"]:
            if re.search(pat, text_lower):
                score += 1.0
        return score


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — PRE-MORTEM ENGINE
# Failure analysis BEFORE execution. Identifies risks, failure modes,
# and prepares contingency plans proactively.
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Risk:
    id: int
    description: str
    likelihood: str        # LOW | MEDIUM | HIGH
    impact: str            # LOW | MEDIUM | HIGH | CRITICAL
    mitigation: str
    contingency: str


@dataclass
class PreMortemReport:
    task: str
    risks: list[Risk]
    overall_risk_level: str
    go_no_go: str          # GO | CAUTION | NO_GO
    recommendations: list[str]
    timestamp: float = field(default_factory=time.time)

    def to_markdown(self) -> str:
        lines = [
            f"## [PRE_MORTEM] Risk Analysis",
            f"**Task:** {self.task}",
            f"**Overall Risk:** `{self.overall_risk_level}` | **Decision:** `{self.go_no_go}`",
            "",
            "### Risk Register",
        ]
        for r in self.risks:
            lines.append(
                f"- **R{r.id}** [{r.likelihood}/{r.impact}] {r.description}\n"
                f"  → Mitigation: {r.mitigation}\n"
                f"  → Contingency: {r.contingency}"
            )
        if self.recommendations:
            lines.append("\n### Strategic Recommendations")
            for rec in self.recommendations:
                lines.append(f"- {rec}")
        return "\n".join(lines)


class PreMortemEngine:
    """
    LAYER 1: Pre-Mortem Risk Analysis.

    Runs BEFORE execution. Simulates failure to identify:
      - Technical risks (API failures, race conditions, missing deps)
      - Strategic risks (wrong approach, scope creep)
      - Data risks (loss, corruption, privacy)
      - Time risks (blocking steps, infinite loops)

    Uses heuristic pattern matching for zero-latency analysis
    + LLM deep analysis for complex tasks.
    """

    _RISK_PATTERNS: list[dict] = [
        {
            "pattern": r"(api|endpoint|http|request|fetch|webhook)",
            "risk": "External API dependency — rate limits, downtime, schema changes",
            "likelihood": "MEDIUM", "impact": "HIGH",
            "mitigation": "Add retry logic with exponential backoff",
            "contingency": "Cache last successful response; degrade gracefully",
        },
        {
            "pattern": r"(database|db|sql|query|insert|delete|drop)",
            "risk": "Database operation — data loss/corruption risk on failure",
            "likelihood": "LOW", "impact": "CRITICAL",
            "mitigation": "Wrap in transactions; validate inputs; take backup",
            "contingency": "Rollback transaction; restore from checkpoint",
        },
        {
            "pattern": r"(loop|while|recursive|iterate|for each)",
            "risk": "Unbounded iteration — potential infinite loop or OOM",
            "likelihood": "MEDIUM", "impact": "HIGH",
            "mitigation": "Enforce max_iterations cap and timeout",
            "contingency": "Kill process; log state; resume from checkpoint",
        },
        {
            "pattern": r"(file|write|read|save|load|disk|path)",
            "risk": "File I/O — path traversal, permission errors, disk full",
            "likelihood": "MEDIUM", "impact": "MEDIUM",
            "mitigation": "Validate paths; check permissions; check disk space",
            "contingency": "Use temp dir fallback; surface error to user",
        },
        {
            "pattern": r"(auth|token|secret|password|key|credential)",
            "risk": "Credential handling — exposure in logs, memory, or output",
            "likelihood": "LOW", "impact": "CRITICAL",
            "mitigation": "Use env vars; never log secrets; mask in output",
            "contingency": "Rotate credentials immediately if exposed",
        },
        {
            "pattern": r"(parallel|concurrent|async|thread|race)",
            "risk": "Concurrency — race conditions, deadlocks, shared state",
            "likelihood": "MEDIUM", "impact": "HIGH",
            "mitigation": "Use locks/semaphores; design for idempotency",
            "contingency": "Serialize execution as fallback",
        },
        {
            "pattern": r"(deploy|production|prod|live|release)",
            "risk": "Production deployment — irreversible changes at scale",
            "likelihood": "LOW", "impact": "CRITICAL",
            "mitigation": "Blue/green or canary deployment; feature flags",
            "contingency": "Immediate rollback procedure; incident runbook",
        },
        {
            "pattern": r"(user data|personal|pii|gdpr|privacy)",
            "risk": "PII/privacy — regulatory exposure, trust damage",
            "likelihood": "LOW", "impact": "CRITICAL",
            "mitigation": "Minimize data collection; encrypt at rest/transit",
            "contingency": "Isolate breach; notify stakeholders per GDPR timeline",
        },
    ]

    def analyze(self, task: str, plan_steps: list[str] | None = None) -> PreMortemReport:
        task_lower = task.lower()
        full_text  = task_lower + " " + " ".join(plan_steps or []).lower()

        risks: list[Risk] = []
        for i, rp in enumerate(self._RISK_PATTERNS, start=1):
            if re.search(rp["pattern"], full_text, re.IGNORECASE):
                risks.append(Risk(
                    id=i,
                    description=rp["risk"],
                    likelihood=rp["likelihood"],
                    impact=rp["impact"],
                    mitigation=rp["mitigation"],
                    contingency=rp["contingency"],
                ))

        # ── Compute overall risk level ─────────────────────────────────────
        critical_count = sum(1 for r in risks if r.impact == "CRITICAL")
        high_count     = sum(1 for r in risks if r.impact == "HIGH")

        if critical_count >= 2:
            overall = "CRITICAL"
            go_no_go = "CAUTION"
        elif critical_count >= 1 or high_count >= 3:
            overall = "HIGH"
            go_no_go = "CAUTION"
        elif high_count >= 1 or len(risks) >= 3:
            overall = "MEDIUM"
            go_no_go = "GO"
        else:
            overall = "LOW"
            go_no_go = "GO"

        # ── Strategic recommendations ──────────────────────────────────────
        recs = []
        if not risks:
            recs.append("No significant risks detected — proceed with standard execution")
        else:
            if critical_count:
                recs.append(f"Address {critical_count} CRITICAL risk(s) before starting")
            if high_count:
                recs.append("Implement mitigations for HIGH-impact risks in first steps")
            recs.append("Enable checkpointing (already active in GazccAgent)")
            if len(risks) > 3:
                recs.append("Consider phasing execution — reduce blast radius of failure")

        return PreMortemReport(
            task=task,
            risks=risks,
            overall_risk_level=overall,
            go_no_go=go_no_go,
            recommendations=recs,
        )


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3 — SELF-EVOLUTION ENGINE
# Observes corrections and feedback. Persists lessons to disk.
# Injects learned context into every future planning/execution call.
# "A living, learning system."
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EvolutionEntry:
    timestamp: float
    trigger: str           # CORRECTION | FEEDBACK | ERROR_PATTERN | SUCCESS
    original_input: str
    correction_or_lesson: str
    domain: str            # e.g. "code", "planning", "tone", "strategy"
    applied_count: int = 0

    def to_dict(self) -> dict:
        return self.__dict__

    @classmethod
    def from_dict(cls, d: dict) -> "EvolutionEntry":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class SelfEvolutionEngine:
    """
    LAYER 3: Self-Evolution & Internal Weight Refinement.

    Persists learned corrections/patterns to `evolution_weights.json`.
    On every task start, injects the most relevant learned lessons
    into the system prompt — making the agent truly adaptive.

    Learns from:
      - Explicit corrections ("no, that's wrong, you should...")
      - Error patterns (repeated failures on similar tasks)
      - User feedback (appreciation → reinforce; frustration → adjust)
      - Successful strategies (replicate what worked)

    Storage: JSON file at `{memory_path}/evolution_weights.json`
    """

    MAX_ENTRIES = 200           # Cap to prevent context explosion
    MAX_INJECT  = 5             # Max lessons injected per task

    def __init__(self, memory_path: str = "/tmp/gazcc_evolution"):
        self._path    = Path(memory_path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._weights_file = self._path / "evolution_weights.json"
        self._entries: list[EvolutionEntry] = []
        self._loaded  = False
        self._lock    = asyncio.Lock()

    async def _load(self):
        if self._loaded:
            return
        if self._weights_file.exists():
            try:
                async with aiofiles.open(self._weights_file, "r") as f:
                    data = json.loads(await f.read())
                self._entries = [EvolutionEntry.from_dict(e) for e in data]
                logger.info(f"[SelfEvolution] Loaded {len(self._entries)} learned entries")
            except Exception as e:
                logger.warning(f"[SelfEvolution] Could not load weights: {e}")
        self._loaded = True

    async def _save(self):
        async with aiofiles.open(self._weights_file, "w") as f:
            await f.write(json.dumps(
                [e.to_dict() for e in self._entries[-self.MAX_ENTRIES:]],
                indent=2, ensure_ascii=False
            ))

    async def record_correction(
        self,
        original_input: str,
        correction: str,
        domain: str = "general",
    ):
        """Called when EmpathyBuffer detects a correction."""
        await self._load()
        async with self._lock:
            entry = EvolutionEntry(
                timestamp=time.time(),
                trigger="CORRECTION",
                original_input=original_input[:300],
                correction_or_lesson=correction[:500],
                domain=domain,
            )
            self._entries.append(entry)
            await self._save()
            logger.info(f"[SelfEvolution] Correction recorded: {correction[:80]}...")

    async def record_success(self, task: str, strategy_summary: str, domain: str = "general"):
        """Called when a task completes successfully — reinforce the strategy."""
        await self._load()
        async with self._lock:
            entry = EvolutionEntry(
                timestamp=time.time(),
                trigger="SUCCESS",
                original_input=task[:300],
                correction_or_lesson=f"SUCCESSFUL STRATEGY: {strategy_summary[:400]}",
                domain=domain,
            )
            self._entries.append(entry)
            await self._save()

    async def record_error_pattern(self, task: str, error: str, domain: str = "general"):
        """Called on repeated failures — learn what NOT to do."""
        await self._load()
        async with self._lock:
            # Deduplicate similar error patterns
            similar = [
                e for e in self._entries
                if e.trigger == "ERROR_PATTERN" and
                self._similarity(e.correction_or_lesson, error) > 0.7
            ]
            if similar:
                similar[-1].applied_count += 1
                await self._save()
                return

            entry = EvolutionEntry(
                timestamp=time.time(),
                trigger="ERROR_PATTERN",
                original_input=task[:300],
                correction_or_lesson=f"AVOID: {error[:400]}",
                domain=domain,
            )
            self._entries.append(entry)
            await self._save()

    def get_relevant_lessons(self, task: str, top_k: int | None = None) -> list[EvolutionEntry]:
        """Return most relevant learned lessons for a given task context."""
        if not self._entries:
            return []

        top_k = top_k or self.MAX_INJECT
        scored = []
        task_tokens = set(re.findall(r"\b\w+\b", task.lower()))

        for entry in self._entries:
            entry_tokens = set(re.findall(r"\b\w+\b", (
                entry.original_input + " " + entry.correction_or_lesson
            ).lower()))
            overlap = len(task_tokens & entry_tokens) / max(len(task_tokens), 1)

            # Boost corrections and recent entries
            recency_boost = 1.0 / (1.0 + (time.time() - entry.timestamp) / 86400)  # per day
            trigger_boost = 1.3 if entry.trigger == "CORRECTION" else 1.0

            score = overlap * trigger_boost * (0.7 + 0.3 * recency_boost)
            scored.append((score, entry))

        scored.sort(key=lambda x: -x[0])
        return [e for _, e in scored[:top_k] if _ > 0.1]

    def build_evolution_context(self, task: str) -> str:
        """Build the evolution context string to inject into system prompts."""
        lessons = self.get_relevant_lessons(task)
        if not lessons:
            return ""

        lines = ["## EVOLVED INTELLIGENCE — Learned from past interactions:"]
        for i, lesson in enumerate(lessons, 1):
            trigger_icon = {
                "CORRECTION": "⚠️ CORRECTION",
                "SUCCESS": "✅ PROVEN STRATEGY",
                "ERROR_PATTERN": "🚫 AVOID",
                "FEEDBACK": "💡 INSIGHT",
            }.get(lesson.trigger, "📌 NOTE")
            lines.append(f"{i}. [{trigger_icon}] {lesson.correction_or_lesson}")

        lines.append(
            "\nApply these lessons. This is your EVOLVED state — "
            "do not repeat past mistakes or ignore proven strategies."
        )
        return "\n".join(lines)

    def _similarity(self, a: str, b: str) -> float:
        ta = set(re.findall(r"\b\w+\b", a.lower()))
        tb = set(re.findall(r"\b\w+\b", b.lower()))
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / max(len(ta | tb), 1)

    async def get_stats(self) -> dict:
        await self._load()
        by_trigger: dict[str, int] = {}
        by_domain:  dict[str, int] = {}
        for e in self._entries:
            by_trigger[e.trigger] = by_trigger.get(e.trigger, 0) + 1
            by_domain[e.domain]   = by_domain.get(e.domain, 0) + 1
        return {
            "total_entries": len(self._entries),
            "by_trigger": by_trigger,
            "by_domain": by_domain,
            "evolution_file": str(self._weights_file),
        }


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2 + 4 — GOD-TIER PLANNER
# Dual-persona planner: [ARCHITECT] defines strategy, [ENGINEER] validates
# feasibility. Outputs structured [ANALYSIS][STRATEGY][PRE_MORTEM] blocks.
# ══════════════════════════════════════════════════════════════════════════════

GOD_TIER_PLAN_SYSTEM = """\
You are GazccAgent L99 — operating under the DUAL PERSONA protocol:

╔══════════════════════════════════════╗
║  [ARCHITECT] — Senior Solution Architect
║  Sees the WHOLE system. Defines strategy,
║  architecture, and the "why" behind each step.
║  Focuses on: scalability, maintainability,
║  failure modes, and hidden dependencies.
╚══════════════════════════════════════╝

╔══════════════════════════════════════╗
║  [ENGINEER] — Principal Engineer
║  Executes with PRECISION. Validates that
║  each step is technically feasible, has
║  the right tools, and handles edge cases.
║  Catches: vague steps, missing inputs,
║  wrong tool choices, impossible deps.
╚══════════════════════════════════════╝

EVOLVED INTELLIGENCE:
{evolution_context}

AVAILABLE TOOLS:
{tools}

TASK: {task}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT PROTOCOL — You MUST respond with EXACTLY this structure:

[ANALYSIS]
- Restate the core objective in your own words
- List hidden complexities, ambiguities, or unstated requirements
- Identify the key constraints (time, tools, data, permissions)
- Note any domain-specific knowledge required

[STRATEGY]
- High-level architectural approach chosen by [ARCHITECT]
- Why this approach over alternatives
- Critical path (the steps everything else depends on)
- Steps that can be parallelized (mark with "⚡ parallel")

[PRE_MORTEM]
- Top 3 ways this plan can FAIL (be specific)
- Mitigation for each failure mode
- Decision: GO / CAUTION / NO_GO

[PLAN]
Return a VALID JSON array of steps:
[
  {{
    "id": 1,
    "description": "Clear, actionable step description",
    "depends_on": [],
    "tool_hint": "tool_name_or_empty"
  }},
  ...
]

RULES:
- Every step must be independently executable
- No step should be "do everything" — be granular
- depends_on must reference valid step IDs only
- tool_hint should be one of: {tool_names}
- Minimum 3 steps, maximum 15 steps
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

PROACTIVE_SUGGESTION_SYSTEM = """\
You are a Strategic Partner analyzing a task for proactive opportunities.
Task: {task}
Completed steps: {steps}
Identify 2-3 related improvements, optimizations, or next steps the user may not have considered.
Return as JSON array: [{{"title": "...", "reason": "..."}}]
"""


class GodTierPlanner:
    """
    LAYER 4: God-Tier Planner.

    Extends the base Planner with:
      - Dual persona (Architect + Engineer)
      - Structured output format
      - Evolution context injection
      - Pre-mortem integration
      - Proactive suggestion generation
    """

    def __init__(self, llm_cfg: dict, tool_schema: str, evolution: SelfEvolutionEngine):
        self._cfg            = llm_cfg
        self._tool_schema    = tool_schema
        self._evolution      = evolution
        self._memory_context = ""
        self._base_url       = llm_cfg.get("base_url", "https://openrouter.ai/api/v1")
        self._model          = llm_cfg.get("model", "qwen/qwen3.5-flash-02-23")
        self._api_key        = llm_cfg.get("api_key", "")

        # Extract tool names from schema for planner hints
        self._tool_names = re.findall(r'"name"\s*:\s*"(\w+)"', tool_schema)

    def set_memory_context(self, ctx: str):
        self._memory_context = ctx

    async def decompose(self, task: str) -> Plan:
        """Decompose task into a dependency-aware Plan using dual-persona analysis."""
        evolution_ctx = self._evolution.build_evolution_context(task)

        prompt = GOD_TIER_PLAN_SYSTEM.format(
            task=task,
            tools=self._tool_schema,
            tool_names=", ".join(self._tool_names[:20]) or "read_file, write_file, web_search, execute_code",
            evolution_context=evolution_ctx or "(No evolved context yet — first interaction)",
        )

        if self._memory_context:
            prompt += f"\n\nRELEVANT MEMORY:\n{self._memory_context}"

        response = await self._call_llm([
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Decompose this task: {task}"},
        ])

        return self._parse_god_tier_response(task, response)

    def _parse_god_tier_response(self, task: str, response: str) -> Plan:
        """Extract structured blocks and JSON plan from god-tier response."""

        # Extract the goal from [ANALYSIS] block
        goal = task
        analysis_match = re.search(r'\[ANALYSIS\](.*?)(?=\[STRATEGY\]|\[PRE_MORTEM\]|\[PLAN\]|$)',
                                    response, re.DOTALL | re.IGNORECASE)
        if analysis_match:
            goal = analysis_match.group(1).strip()[:500]

        # Extract JSON plan from [PLAN] block
        plan_match = re.search(r'\[PLAN\](.*?)(?=\[|$)', response, re.DOTALL | re.IGNORECASE)
        json_source = plan_match.group(1).strip() if plan_match else response

        # Find JSON array
        arr_match = re.search(r'\[[\s\S]*\]', json_source)
        if not arr_match:
            # Fallback: try full response
            arr_match = re.search(r'\[[\s\S]*\]', response)

        steps: list[Step] = []
        if arr_match:
            try:
                raw_steps = json.loads(arr_match.group())
                for s in raw_steps:
                    steps.append(Step(
                        id=int(s.get("id", len(steps) + 1)),
                        description=str(s.get("description", "Unnamed step")),
                        depends_on=[int(d) for d in s.get("depends_on", [])],
                        tool_hint=str(s.get("tool_hint", "")),
                    ))
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.warning(f"[GodTierPlanner] JSON parse failed: {e}")

        if not steps:
            # Emergency fallback: single-step plan
            steps = [Step(id=1, description=task, depends_on=[], tool_hint="")]

        # Attach full structured analysis as plan metadata
        plan       = Plan(task=task, steps=steps, goal=goal)
        plan._god_tier_analysis = response  # type: ignore[attr-defined]
        return plan

    async def get_proactive_suggestions(self, task: str, completed_steps: list[str]) -> list[dict]:
        """Generate proactive suggestions for related improvements."""
        try:
            response = await self._call_llm([{
                "role": "user",
                "content": PROACTIVE_SUGGESTION_SYSTEM.format(
                    task=task,
                    steps="\n".join(completed_steps[:10]),
                )
            }])
            arr_match = re.search(r'\[[\s\S]*?\]', response)
            if arr_match:
                return json.loads(arr_match.group())
        except Exception:
            pass
        return []

    async def _call_llm(self, messages: list[dict]) -> str:
        url     = f"{self._base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/gazcc",
            "X-Title": "GazccL99",
        }
        payload = {
            "model": self._model,
            "messages": messages,
            "max_tokens": 4000,
            "temperature": 0.1,
        }
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2 + 5 — GOD-TIER EXECUTOR
# Dual-persona executor with structured output format.
# Every response: [ANALYSIS][STRATEGY][IMPLEMENTATION][CRITICAL_REVIEW]
# ══════════════════════════════════════════════════════════════════════════════

GOD_TIER_REACT_SYSTEM = """\
You are GazccAgent L99 — ULTRA-HIGH-LEVEL AUTONOMOUS INTELLIGENCE.

You operate with TWO voices:

[ARCHITECT] — Defines strategy. Before acting, map the problem space.
              Sees edge cases others miss. Thinks in systems.

[ENGINEER]  — Executes with surgical precision. Uses the right tool
              for the job. Never guesses. Verifies results.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EVOLVED INTELLIGENCE:
{evolution_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AVAILABLE TOOLS:
{tool_schema}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REACT FORMAT — use EXACTLY this format:

Thought: [ARCHITECT analysis first, then ENGINEER execution plan]
Action: [tool_name]
Action Input: {{"key": "value"}}

After Observation:
Critic: [ARCHITECT reviews the result — identify flaws or confirm quality]

Then either continue:
Thought: ...

Or conclude with the FULL STRUCTURED OUTPUT:
Final Answer:
[ANALYSIS]
(What was actually required — including hidden complexity)

[STRATEGY]
(Approach taken and why — architectural rationale)

[IMPLEMENTATION]
(The actual output, code, data, or result)

[CRITICAL_REVIEW]
(Honest self-critique: what could be better, trade-offs, next steps)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULES:
- Action Input MUST be valid JSON
- Never repeat a failed Action with the same input — change strategy
- The Critic must speak after every significant action
- If Critic identifies a flaw: ENGINEER must fix it before Final Answer
- Final Answer MUST use the 4-section structured format above
- Partial answers are not acceptable — be thorough or explain why you can't be
"""

GOD_TIER_REACT_USER = """\
STEP TO EXECUTE: {step_description}

CONTEXT FROM PREVIOUS STEPS:
{context}

MEMORY RECALL:
{memory}

Execute this step. Apply both [ARCHITECT] thinking and [ENGINEER] precision.
Your Final Answer must use the [ANALYSIS][STRATEGY][IMPLEMENTATION][CRITICAL_REVIEW] format.
"""


class GodTierExecutor:
    """
    LAYER 5: God-Tier Step Executor.

    Extends StepExecutor with:
      - Evolution context injection per step
      - Dual-persona system prompt
      - Structured output enforcement
      - Critic loop that blocks on architectural flaws
      - Error pattern recording in SelfEvolution
    """

    MAX_REACT_TURNS = 10  # Increased from base 8

    def __init__(
        self,
        llm_cfg: dict,
        tools: ToolRegistry,
        retry_limit: int,
        evolution: SelfEvolutionEngine,
    ):
        self._cfg         = llm_cfg
        self._tools       = tools
        self._retry_limit = retry_limit
        self._evolution   = evolution
        self._base_url    = llm_cfg.get("base_url", "https://openrouter.ai/api/v1")
        self._model       = llm_cfg.get("model", "qwen/qwen3.5-flash-02-23")
        self._api_key     = llm_cfg.get("api_key", "")

    async def execute_step(
        self,
        step: Step,
        context: str = "",
        memory_recall: str = "",
        on_event: Callable | None = None,
        task_for_evolution: str = "",
    ) -> tuple[bool, str]:
        """Execute a single plan step through the God-Tier ReAct loop."""

        evolution_ctx = self._evolution.build_evolution_context(
            step.description + " " + task_for_evolution
        )

        system_prompt = GOD_TIER_REACT_SYSTEM.format(
            evolution_context=evolution_ctx or "(No evolved context yet)",
            tool_schema=self._tools.schema_string(),
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": GOD_TIER_REACT_USER.format(
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

            # ── Final Answer reached ───────────────────────────────────────
            if parsed["final_answer"] is not None:
                final = parsed["final_answer"]

                # Record successful strategy for evolution
                await self._evolution.record_success(
                    task=step.description,
                    strategy_summary=final[:400],
                    domain=self._infer_domain(step.description),
                )

                return True, final

            # ── Critic detected a flaw — must fix before concluding ───────
            critic = parsed.get("critic", "")
            if critic and "LGTM" not in critic and "no architectural concerns" not in critic.lower():
                if on_event:
                    on_event("critic_flag", {"critic": critic, "step_id": step.id})
                # Inject critic feedback into conversation
                messages.append({"role": "assistant", "content": llm_output})
                messages.append({
                    "role": "user",
                    "content": (
                        f"[ARCHITECT] raised a concern: {critic}\n\n"
                        "[ENGINEER] must address this flaw before concluding. "
                        "Continue execution with the fix applied."
                    ),
                })
                continue

            # ── Tool call ─────────────────────────────────────────────────
            action = parsed.get("action")
            if not action:
                messages.append({"role": "assistant", "content": llm_output})
                messages.append({
                    "role": "user",
                    "content": (
                        "Follow the format:\n"
                        "Thought: [analysis]\nAction: <tool>\nAction Input: {}\n\n"
                        "OR conclude with:\nFinal Answer:\n[ANALYSIS]...[STRATEGY]..."
                        "[IMPLEMENTATION]...[CRITICAL_REVIEW]...\n\n"
                        f"Available tools: {list(self._tools._tools.keys())}"
                    ),
                })
                continue

            action_input = parsed.get("action_input", {})
            if "_raw" in action_input:
                action_input = {"input": action_input["_raw"]}

            if on_event:
                on_event("tool_call", {"tool": action, "input": action_input, "step_id": step.id})

            tool_result = await self._tools.run(action, action_input)
            observation = str(tool_result)

            if len(observation) > 4000:
                observation = observation[:4000] + "\n... [truncated]"

            if on_event:
                on_event("observation", {
                    "tool": action,
                    "success": tool_result.success,
                    "output": observation[:300],
                    "step_id": step.id,
                })

            messages.append({"role": "assistant", "content": llm_output})
            messages.append({"role": "user", "content": f"Observation: {observation}\n\nContinue."})

        # ── Exhausted turns ───────────────────────────────────────────────
        err = "Max ReAct turns reached without completion."
        await self._evolution.record_error_pattern(
            task=step.description,
            error=f"Turn exhaustion on step: {step.description[:200]}",
            domain=self._infer_domain(step.description),
        )
        return False, err

    async def _call_llm(self, messages: list[dict], retry: int = 0) -> str:
        url     = f"{self._base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/gazcc",
            "X-Title": "GazccL99",
        }
        payload = {
            "model": self._model,
            "messages": messages,
            "max_tokens": 4000,
            "temperature": 0.1,
        }
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
        except (httpx.HTTPStatusError, httpx.TimeoutException, KeyError) as e:
            if retry < self._retry_limit:
                await asyncio.sleep(2 ** retry)
                return await self._call_llm(messages, retry + 1)
            raise RuntimeError(f"LLM call failed: {e}") from e

    def _infer_domain(self, text: str) -> str:
        text_lower = text.lower()
        if any(w in text_lower for w in ["code", "python", "function", "class", "script"]):
            return "code"
        if any(w in text_lower for w in ["plan", "strategy", "architect", "design"]):
            return "planning"
        if any(w in text_lower for w in ["tone", "response", "answer", "explain"]):
            return "communication"
        if any(w in text_lower for w in ["deploy", "docker", "ci", "test"]):
            return "devops"
        return "general"


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 6 — GOD-TIER AGENT
# Full orchestration. Drop-in replacement for GazccAgent.
# ══════════════════════════════════════════════════════════════════════════════

class GodTierAgent:
    """
    ██████████████████████████████████████████████████████████████████
    ██                                                              ██
    ██   GAZCC L99 — GOD-TIER AUTONOMOUS INTELLIGENCE              ██
    ██   Drop-in replacement for GazccAgent                        ██
    ██                                                              ██
    ██   All layers active:                                         ██
    ██   [0] EmpathyBuffer     [1] PreMortem                       ██
    ██   [2] DualPersona       [3] SelfEvolution                   ██
    ██   [4] GodTierPlanner    [5] GodTierExecutor                 ██
    ██                                                              ██
    ██████████████████████████████████████████████████████████████████

    Usage:
        agent = GodTierAgent(config)
        result = await agent.run("Build a REST API with authentication")

        # Stream events:
        async for event in agent.stream("Research and summarize AI trends"):
            print(event)

        # Handle corrections (auto-recorded for self-evolution):
        await agent.provide_correction("That was wrong", "You should have...")

        # Check evolution stats:
        stats = await agent.evolution_stats()
    """

    def __init__(self, cfg: dict):
        self._cfg         = cfg
        self._agent_cfg   = cfg.get("agent", {})
        self._llm_cfg     = self._resolve_env(cfg.get("llm", {}))
        self._max_iter    = self._agent_cfg.get("max_iterations", 20)
        self._timeout     = self._agent_cfg.get("timeout", 900)
        self._ckpt_int    = self._agent_cfg.get("checkpoint_interval", 3)
        self._retry_limit = self._agent_cfg.get("retry_limit", 3)

        # ── L99 Components ─────────────────────────────────────────────
        _mem_path    = cfg.get("memory", {}).get("file_path", "/tmp/gazcc_memory")
        _evol_path   = _mem_path + "_evolution"

        self._empathy   = EmpathyBuffer()
        self._premortem = PreMortemEngine()
        self._evolution = SelfEvolutionEngine(memory_path=_evol_path)

        # ── Infrastructure ─────────────────────────────────────────────
        self._memory = build_memory(cfg.get("memory", {}))
        self._tools  = ToolRegistry(cfg)

        # Register strategic tools
        _sem_mem = SemanticMemoryTool(memory_path=_mem_path + "_semantic")
        self._tools.register(_sem_mem)
        self._tools.register(ProactiveMonitorTool())
        self._tools.register(ApiBridgeTool())
        self._tools.register(SandboxExecutorTool())

        self._planner  = GodTierPlanner(self._llm_cfg, self._tools.schema_string(), self._evolution)
        self._executor = GodTierExecutor(self._llm_cfg, self._tools, self._retry_limit, self._evolution)
        self._sem_mem  = _sem_mem

        self._events:          list[AgentEvent]    = []
        self._event_callbacks: list[Callable]      = []
        self._last_result:     AgentResult | None  = None

        # Track conversation context for empathy buffer
        self._last_user_input: str = ""
        self._interaction_count:int = 0

        logger.info("[GazccL99] God-Tier Agent initialized — all layers active")

    # ── Public API ─────────────────────────────────────────────────────────

    async def run(self, task: str, task_id: str | None = None) -> AgentResult:
        """Run task to completion. Returns AgentResult."""
        task_id = task_id or str(uuid.uuid4())[:8]
        async for _ in self._run_gen(task, task_id):
            pass
        return self._last_result  # type: ignore

    async def stream(self, task: str, task_id: str | None = None) -> AsyncIterator[AgentEvent]:
        """Yield AgentEvent objects as the agent works."""
        task_id = task_id or str(uuid.uuid4())[:8]
        async for event in self._run_gen(task, task_id):
            yield event

    def on_event(self, callback: Callable[[AgentEvent], None]):
        self._event_callbacks.append(callback)

    async def provide_correction(self, original_input: str, correction: str, domain: str = "general"):
        """
        Explicitly provide a correction. The agent will learn from it
        and apply the lesson in all future tasks.
        """
        await self._evolution.record_correction(original_input, correction, domain)
        logger.info(f"[GazccL99] Correction recorded and internalized.")

    async def evolution_stats(self) -> dict:
        """Return self-evolution statistics."""
        return await self._evolution.get_stats()

    # ── Core Execution Loop ────────────────────────────────────────────────

    async def _run_gen(self, task: str, task_id: str) -> AsyncIterator[AgentEvent]:
        self._events      = []
        self._last_result = None
        start             = time.time()
        self._last_user_input = task
        self._interaction_count += 1

        # ── Import TaskState here to avoid circular issue ──────────────
        from .memory import TaskState
        state_store = TaskState(self._memory, task_id)

        try:
            # ── LAYER 0: Empathy Buffer ────────────────────────────────
            empathy_result = self._empathy.analyze(task)
            yield self._emit("empathy_scan", {
                "mode": empathy_result.mode,
                "sentiment": empathy_result.sentiment_score,
                "urgency": empathy_result.urgency,
                "summary": empathy_result.summary,
            })

            # Handle non-execution modes
            if empathy_result.mode == OperationalMode.SELF_REFLECT:
                await self._evolution.record_correction(
                    original_input=task,
                    correction=task,
                    domain="general",
                )
                yield self._emit("self_reflect", {
                    "msg": "Correction detected — internalizing for self-evolution",
                    "lesson_recorded": True,
                })

            elif empathy_result.mode == OperationalMode.CONVERSATIONAL:
                # Skip full pipeline for appreciation — acknowledge only
                yield self._emit("task_start", {"task_id": task_id, "task": task, "mode": "conversational"})
                self._last_result = AgentResult(
                    task_id=task_id, task=task, success=True,
                    output="[CONVERSATIONAL MODE] Acknowledgment processed.",
                    steps_done=0, steps_total=0, elapsed=time.time() - start,
                )
                yield self._emit("task_done", {"task_id": task_id, "mode": "conversational"})
                return

            # ── LAYER 1: Pre-Mortem Analysis ───────────────────────────
            premortem_report = self._premortem.analyze(task)
            yield self._emit("premortem", {
                "risk_level": premortem_report.overall_risk_level,
                "go_no_go": premortem_report.go_no_go,
                "risks_found": len(premortem_report.risks),
                "report": premortem_report.to_markdown(),
            })

            # ── Start task ─────────────────────────────────────────────
            yield self._emit("task_start", {
                "task_id": task_id,
                "task": task,
                "mode": empathy_result.mode,
                "risk_level": premortem_report.overall_risk_level,
            })

            # ── Check checkpoint ───────────────────────────────────────
            plan = await self._load_checkpoint(state_store, task)
            if plan:
                yield self._emit("checkpoint", {"msg": "Resumed from checkpoint", "steps": len(plan.steps)})
            else:
                # ── LAYER 4: God-Tier Planning ─────────────────────────
                yield self._emit("planning", {"msg": "ARCHITECT + ENGINEER decomposing task..."})

                mem_ctx = await self._memory.search(task, top_k=3)
                self._planner.set_memory_context(
                    "\n".join(f"[{e.key}]: {e.content[:200]}" for e in mem_ctx)
                )

                plan = await asyncio.wait_for(
                    self._planner.decompose(task), timeout=90
                )

                yield self._emit("plan_ready", {
                    "goal": plan.goal,
                    "steps": [s.to_dict() for s in plan.steps],
                    "analysis": getattr(plan, "_god_tier_analysis", "")[:600],
                })

                # Proactive suggestions
                try:
                    suggestions = await asyncio.wait_for(
                        self._planner.get_proactive_suggestions(task, [s.description for s in plan.steps]),
                        timeout=30,
                    )
                    if suggestions:
                        yield self._emit("proactive_suggestions", {
                            "msg": "[ARCHITECT] Strategic Partner detected related opportunities:",
                            "suggestions": suggestions,
                        })
                except Exception:
                    pass

                await self._save_checkpoint(state_store, plan)

            # ── LAYER 5: God-Tier Execution ────────────────────────────
            iteration     = 0
            context_parts: list[str] = []

            while not plan.is_complete() and iteration < self._max_iter:
                if time.time() - start > self._timeout:
                    yield self._emit("error", {"msg": f"Timeout after {self._timeout}s"})
                    break

                ready = plan.pending_steps()
                if not ready:
                    if plan.has_failed():
                        yield self._emit("error", {"msg": "Blocked by failed steps"})
                    break

                for step in ready:
                    step.status = "running"
                    yield self._emit("step_start", {
                        "step_id": step.id,
                        "description": step.description,
                        "tool_hint": step.tool_hint,
                    })

                    mem_results = await self._memory.search(step.description, top_k=3)
                    mem_text    = "\n".join(f"• [{e.key}]: {e.content[:300]}" for e in mem_results)
                    context_str = "\n---\n".join(context_parts[-5:])

                    exec_events: list[dict] = []
                    def _on_exec(etype: str, edata: dict):
                        exec_events.append({"type": etype, "data": edata})

                    success, result_str = False, ""
                    for attempt in range(self._retry_limit):
                        try:
                            success, result_str = await asyncio.wait_for(
                                self._executor.execute_step(
                                    step, context_str, mem_text,
                                    on_event=_on_exec,
                                    task_for_evolution=task,
                                ),
                                timeout=self._timeout - (time.time() - start),
                            )
                            break
                        except asyncio.TimeoutError:
                            success, result_str = False, "Step timed out"
                            break
                        except Exception as e:
                            if attempt < self._retry_limit - 1:
                                await asyncio.sleep(2 ** attempt)
                            else:
                                success, result_str = False, str(e)

                    for ev in exec_events:
                        yield self._emit(ev["type"], ev["data"])

                    step.result = result_str
                    if success:
                        step.status = "done"
                        context_parts.append(f"[Step {step.id}: {step.description}]\n{result_str}")
                        await self._memory.store(
                            key=f"{task_id}_step_{step.id}",
                            content=result_str,
                            metadata={"task_id": task_id, "step": step.description},
                        )
                        yield self._emit("step_done", {
                            "step_id": step.id,
                            "description": step.description,
                            "result_preview": result_str[:300],
                        })
                    else:
                        step.retries += 1
                        if step.retries >= self._retry_limit:
                            step.status = "failed"
                            await self._evolution.record_error_pattern(
                                task=step.description,
                                error=result_str[:300],
                            )
                            yield self._emit("step_failed", {
                                "step_id": step.id,
                                "description": step.description,
                                "reason": result_str,
                            })
                        else:
                            step.status = "pending"
                            yield self._emit("step_retry", {
                                "step_id": step.id,
                                "attempt": step.retries,
                            })

                    iteration += 1
                    if iteration % self._ckpt_int == 0:
                        await self._save_checkpoint(state_store, plan)
                        yield self._emit("checkpoint", {"iteration": iteration})

            # ── Finalize ───────────────────────────────────────────────
            elapsed    = time.time() - start
            done_steps = [s for s in plan.steps if s.status == "done"]
            output     = self._compile_god_tier_output(task, plan, context_parts, premortem_report)

            if plan.is_complete() and not plan.has_failed():
                await self._memory.store(
                    key=f"{task_id}_final",
                    content=output,
                    metadata={"task_id": task_id, "task": task},
                )
                await state_store.delete()
                self._last_result = AgentResult(
                    task_id=task_id, task=task, success=True, output=output,
                    steps_done=len(done_steps), steps_total=len(plan.steps),
                    elapsed=elapsed, events=list(self._events),
                )
                yield self._emit("task_done", {
                    "task_id": task_id,
                    "steps_done": len(done_steps),
                    "elapsed": round(elapsed, 2),
                    "output_preview": output[:400],
                })
            else:
                failed  = [s for s in plan.steps if s.status == "failed"]
                err_msg = f"{len(failed)} step(s) failed"
                self._last_result = AgentResult(
                    task_id=task_id, task=task, success=False, output=output,
                    steps_done=len(done_steps), steps_total=len(plan.steps),
                    elapsed=elapsed, events=list(self._events), error=err_msg,
                )
                yield self._emit("task_failed", {
                    "task_id": task_id,
                    "reason": err_msg,
                    "elapsed": round(elapsed, 2),
                })

        except Exception as e:
            logger.exception("[GazccL99] Agent crashed")
            elapsed = time.time() - start
            self._last_result = AgentResult(
                task_id=task_id, task=task, success=False,
                output="", steps_done=0, steps_total=0,
                elapsed=elapsed, error=str(e),
            )
            yield self._emit("error", {"msg": str(e), "task_id": task_id})

    # ── Output Compilation ─────────────────────────────────────────────────

    def _compile_god_tier_output(
        self,
        task: str,
        plan: Plan,
        context_parts: list[str],
        premortem: PreMortemReport,
    ) -> str:
        """
        Compiles the final structured output in God-Tier format:
        [ANALYSIS] [STRATEGY] [PRE_MORTEM] [IMPLEMENTATION] [CRITICAL_REVIEW]
        """

        # Extract structured blocks from the last step's result if available
        last_result = context_parts[-1] if context_parts else ""

        def _extract_block(text: str, block: str) -> str:
            m = re.search(
                rf'\[{block}\](.*?)(?=\[ANALYSIS\]|\[STRATEGY\]|\[PRE_MORTEM\]'
                rf'|\[IMPLEMENTATION\]|\[CRITICAL_REVIEW\]|$)',
                text, re.DOTALL | re.IGNORECASE
            )
            return m.group(1).strip() if m else ""

        # ── Build structured sections ──────────────────────────────────
        analysis_text = _extract_block(last_result, "ANALYSIS") or (
            f"Task '{task}' involved {len(plan.steps)} steps. "
            f"Goal: {plan.goal[:300]}"
        )

        strategy_text = _extract_block(last_result, "STRATEGY") or (
            "\n".join(
                f"Step {s.id}: {s.description}"
                f"{' [deps: ' + str(s.depends_on) + ']' if s.depends_on else ''}"
                for s in plan.steps
            )
        )

        implementation_text = _extract_block(last_result, "IMPLEMENTATION") or (
            "\n\n".join(
                f"### Step {s.id}: {s.description}\n{s.result}"
                for s in plan.steps if s.result
            )
        )

        critical_review_text = _extract_block(last_result, "CRITICAL_REVIEW") or (
            f"Completed {len([s for s in plan.steps if s.status == 'done'])}/{len(plan.steps)} steps. "
            f"Risk level: {premortem.overall_risk_level}. "
            "Review implementation for production readiness."
        )

        return "\n\n".join([
            f"# Task: {task}",
            f"## [ANALYSIS]",
            analysis_text,
            f"## [STRATEGY]",
            strategy_text,
            premortem.to_markdown(),
            f"## [IMPLEMENTATION]",
            implementation_text,
            f"## [CRITICAL_REVIEW]",
            critical_review_text,
        ])

    # ── Helpers ────────────────────────────────────────────────────────────

    def _emit(self, etype: str, data: dict) -> AgentEvent:
        ev = AgentEvent(type=etype, data=data)
        self._events.append(ev)
        logger.info(str(ev))
        for cb in self._event_callbacks:
            try:
                if asyncio.iscoroutinefunction(cb):
                    asyncio.create_task(cb(ev))
                else:
                    cb(ev)
            except Exception:
                pass
        return ev

    async def _save_checkpoint(self, state_store, plan: Plan):
        await state_store.save(plan.to_dict())

    async def _load_checkpoint(self, state_store, task: str) -> Plan | None:
        data = await state_store.load()
        if data and data.get("task") == task:
            plan = Plan.from_dict(data)
            has_pending = any(s.status in ("pending", "running", "failed") for s in plan.steps)
            if has_pending:
                for s in plan.steps:
                    if s.status == "running":
                        s.status = "pending"
                return plan
        return None

    @staticmethod
    def _resolve_env(cfg: dict) -> dict:
        resolved = {}
        for k, v in cfg.items():
            if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
                env_key = v[2:-1]
                resolved[k] = os.environ.get(env_key, "")
            else:
                resolved[k] = v
        return resolved


# ══════════════════════════════════════════════════════════════════════════════
# PATCH FUNCTION — Apply L99 upgrade to existing GazccAgent instance
# ══════════════════════════════════════════════════════════════════════════════

def apply_l99_upgrade(agent: GazccAgent) -> GodTierAgent:
    """
    Upgrade an existing GazccAgent instance to GodTierAgent.
    Preserves memory, tools, and config from the original instance.

    Usage:
        existing_agent = GazccAgent(config)
        god_agent = apply_l99_upgrade(existing_agent)
    """
    logger.info("[L99] Applying God-Tier upgrade to existing GazccAgent...")
    god_agent = GodTierAgent(agent._cfg)

    # Migrate memory if possible
    try:
        god_agent._memory = agent._memory
        god_agent._tools  = agent._tools
        logger.info("[L99] Memory and tools migrated from existing agent")
    except AttributeError:
        logger.warning("[L99] Could not migrate existing agent state — using fresh instance")

    logger.info("[L99] ✅ Upgrade complete — Agent is now operating at L99")
    return god_agent


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Main agent (drop-in replacement)
    "GodTierAgent",

    # Layer components (for custom integration)
    "EmpathyBuffer",
    "EmpathyResult",
    "OperationalMode",
    "PreMortemEngine",
    "PreMortemReport",
    "Risk",
    "SelfEvolutionEngine",
    "EvolutionEntry",
    "GodTierPlanner",
    "GodTierExecutor",

    # Patch utility
    "apply_l99_upgrade",
]
