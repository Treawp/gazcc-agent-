"""
agent/core.py
GazccThinking Agent — autonomous ReAct agent with planning, memory,
checkpointing, self-correction, and event streaming.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Optional

from .executor import StepExecutor
from .memory import FileMemory, RedisMemory, TaskState, build_memory
from .planner import Plan, Planner, Step
from .tools import ToolRegistry
from .strategic_tools import (
    SemanticMemoryTool,
    ProactiveMonitorTool,
    ApiBridgeTool,
    SandboxExecutorTool,
)
from .gazcc_tools_expansion import register_expansion_tools
from .extra_tools import register_extra_tools
from .github_tool import register_github_tools
from .skill_tools import register_skill_tools
from .python_ai_tools import register_python_ai_tools
from .new_tools import register_new_tools
from .fact_guard import register_fact_guard, FactGuardInterceptor
from .learning import LearningSystem

logger = logging.getLogger("gazcc.agent")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ── AgentEvent ────────────────────────────────────────────────────────────────

@dataclass
class AgentEvent:
    type: str           # task_start | plan_ready | step_start | tool_call |
                        # observation | step_done | step_failed | task_done |
                        # task_failed | checkpoint | error
    data: dict
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"type": self.type, "data": self.data, "timestamp": self.timestamp}

    def __str__(self) -> str:
        return f"[{self.type.upper()}] {json.dumps(self.data, ensure_ascii=False)}"


# ── AgentResult ───────────────────────────────────────────────────────────────

@dataclass
class AgentResult:
    task_id: str
    task: str
    success: bool
    output: str
    steps_done: int
    steps_total: int
    elapsed: float
    events: list[AgentEvent] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "task": self.task,
            "success": self.success,
            "output": self.output,
            "steps_done": self.steps_done,
            "steps_total": self.steps_total,
            "elapsed_seconds": round(self.elapsed, 2),
            "error": self.error,
        }


# ── Agent ─────────────────────────────────────────────────────────────────────

class GazccAgent:
    """
    Autonomous AI agent.

    Usage:
        agent = GazccAgent(config)
        result = await agent.run("Research quantum computing and write a report")

    For streaming events:
        async for event in agent.stream("..."):
            print(event)
    """

    def __init__(self, cfg: dict):
        self._cfg = cfg
        self._agent_cfg = cfg.get("agent", {})
        self._llm_cfg = self._resolve_env(cfg.get("llm", {}))
        self._max_iter = self._agent_cfg.get("max_iterations", 20)
        self._timeout = self._agent_cfg.get("timeout", 900)
        self._ckpt_interval = self._agent_cfg.get("checkpoint_interval", 3)
        self._retry_limit = self._agent_cfg.get("retry_limit", 3)

        # Build components
        self._memory = build_memory(cfg.get("memory", {}))
        self._tools = ToolRegistry(cfg)

        # ── Register Strategic Tools
        _mem_path = cfg.get("memory", {}).get("file_path", "/tmp/gazcc_memory")
        _sem_mem = SemanticMemoryTool(memory_path=_mem_path + "_semantic")
        self._tools.register(_sem_mem)
        self._tools.register(ProactiveMonitorTool())
        self._tools.register(ApiBridgeTool())
        self._tools.register(SandboxExecutorTool())

        # ── Register Expansion Tools (CodeTranslator, ImageMetadata, etc.)
        register_expansion_tools(self._tools, self._cfg)
        register_extra_tools(self._tools, self._cfg)
        register_github_tools(self._tools, self._cfg)
        register_skill_tools(self._tools, self._cfg)
        register_python_ai_tools(self._tools, self._cfg)
        register_new_tools(self._tools, self._cfg)
        register_fact_guard(self._tools, self._cfg)

        self._fact_guard = FactGuardInterceptor(self._cfg)

        self._planner = Planner(self._llm_cfg, self._tools.slim_schema_string())
        self._executor = StepExecutor(self._llm_cfg, self._tools, self._retry_limit, agent_cfg=self._cfg)
        self._sem_mem = _sem_mem  # direct reference for pre-task memory recall
        self._learning = LearningSystem(self._llm_cfg, self._memory)

        self._events: list[AgentEvent] = []
        self._event_callbacks: list[Callable[[AgentEvent], None]] = []

    # ── public API ────────────────────────────────────────────────────────────

    async def run(self, task: str, task_id: str | None = None) -> AgentResult:
        """Run task to completion, block until done."""
        task_id = task_id or str(uuid.uuid4())[:8]
        result = None
        async for _ in self._run_gen(task, task_id):
            pass
        return self._last_result

    async def stream(self, task: str, task_id: str | None = None) -> AsyncIterator[AgentEvent]:
        """Yield AgentEvent objects as the agent works."""
        task_id = task_id or str(uuid.uuid4())[:8]
        async for event in self._run_gen(task, task_id):
            yield event

    def on_event(self, callback: Callable[[AgentEvent], None]):
        """Register a callback for every event (sync or async)."""
        self._event_callbacks.append(callback)

    # ── internals ─────────────────────────────────────────────────────────────

    async def _run_gen(self, task: str, task_id: str) -> AsyncIterator[AgentEvent]:
        self._events = []
        self._last_result = None
        start = time.time()
        state_store = TaskState(self._memory, task_id)

        try:
            # ── 1. emit start ──────────────────────────────────────────────
            yield self._emit("task_start", {"task_id": task_id, "task": task})

            # ── Init learning tracker for this task
            _tracker = self._learning.start_task(task_id, task)

            # ── 2. check for existing checkpoint ──────────────────────────
            plan = await self._load_checkpoint(state_store, task)
            if plan:
                yield self._emit("checkpoint", {"msg": "Resumed from checkpoint", "steps": len(plan.steps)})
            else:
                # ── 3. plan ────────────────────────────────────────────────
                yield self._emit("planning", {"msg": "Decomposing task..."})

                # Inject past memory context into planner (Contextual Memory Integration)
                mem_ctx = await self._memory.search(task, top_k=3)
                mem_ctx_text = "\n".join(
                    f"[{e.key}]: {e.content[:200]}" for e in mem_ctx
                )
                self._planner.set_memory_context(mem_ctx_text)

                # Inject learning insights into planner
                learning_ctx = await self._learning.get_insights_context()
                self._planner.set_learning_context(learning_ctx)

                plan = await asyncio.wait_for(
                    self._planner.decompose(task),
                    timeout=60,
                )
                yield self._emit("plan_ready", {
                    "goal": plan.goal,
                    "steps": [s.to_dict() for s in plan.steps],
                })

                # ── Proactive Agency: suggest related tasks ────────────────
                step_descs = [s.description for s in plan.steps]
                try:
                    suggestions = await asyncio.wait_for(
                        self._planner.get_proactive_suggestions(task, step_descs),
                        timeout=30,
                    )
                    if suggestions:
                        yield self._emit("proactive_suggestions", {
                            "msg": "Strategic Partner detected related tasks you may want to consider:",
                            "suggestions": suggestions,
                        })
                except Exception:
                    pass  # Non-critical — don't block execution

                await self._save_checkpoint(state_store, plan)

            # ── 4. execution loop ──────────────────────────────────────────
            iteration = 0
            context_parts: list[str] = []

            while not plan.is_complete() and iteration < self._max_iter:
                if time.time() - start > self._timeout:
                    yield self._emit("error", {"msg": f"Timeout after {self._timeout}s"})
                    break

                ready = plan.pending_steps()
                if not ready:
                    if plan.has_failed():
                        yield self._emit("error", {"msg": "No executable steps — blocked by failures"})
                        break
                    break

                # Execute ready steps (could be parallelized; serial for safety)
                for step in ready:
                    step.status = "running"
                    _tracker.step_started(step.id)
                    yield self._emit("step_start", {"step_id": step.id, "description": step.description, "tool_hint": step.tool_hint})

                    # Memory recall relevant to this step
                    mem_results = await self._memory.search(step.description, top_k=3)
                    mem_text = "\n".join(
                        f"• [{e.key}]: {e.content[:200]}" for e in mem_results
                    )[:1200]  # hard cap ~300 tok

                    # Context from prev steps — last 3 only, 800 chars each
                    context_str = "\n---\n".join(
                        p[:800] for p in context_parts[-3:]
                    )

                    # Executor event bridge
                    exec_events: list[dict] = []

                    def _on_exec(etype: str, edata: dict):
                        exec_events.append({"type": etype, "data": edata})

                    # Execute with retry
                    success, result_str = False, ""
                    for attempt in range(self._retry_limit):
                        try:
                            success, result_str = await asyncio.wait_for(
                                self._executor.execute_step(
                                    step, context_str, mem_text, on_event=_on_exec
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

                    # Emit sub-events from executor
                    for ev in exec_events:
                        yield self._emit(ev["type"], ev["data"])

                    # Update step
                    step.result = result_str
                    if success:
                        step.status = "done"
                        context_parts.append(f"[Step {step.id}: {step.description}]\n{result_str}")
                        # Store result in long-term memory
                        await self._memory.store(
                            key=f"{task_id}_step_{step.id}",
                            content=result_str,
                            metadata={"task_id": task_id, "step": step.description},
                        )
                        _tracker.step_completed(step.id, step.description, step.tool_hint, success=True, retries=step.retries)
                        yield self._emit("step_done", {
                            "step_id": step.id,
                            "description": step.description,
                            "result_preview": result_str[:300],
                        })
                    else:
                        step.retries += 1
                        if step.retries >= self._retry_limit:
                            step.status = "failed"
                            _tracker.step_completed(step.id, step.description, step.tool_hint, success=False, retries=step.retries, error=result_str)
                            yield self._emit("step_failed", {
                                "step_id": step.id,
                                "description": step.description,
                                "reason": result_str,
                            })
                        else:
                            step.status = "pending"  # retry next iteration
                            yield self._emit("step_retry", {
                                "step_id": step.id,
                                "attempt": step.retries,
                            })

                    iteration += 1

                    # Checkpoint every N steps
                    if iteration % self._ckpt_interval == 0:
                        await self._save_checkpoint(state_store, plan)
                        yield self._emit("checkpoint", {"iteration": iteration})

            # ── 5. finalize ────────────────────────────────────────────────
            elapsed = time.time() - start
            done_steps = [s for s in plan.steps if s.status == "done"]
            final_output = self._compile_output(task, plan, context_parts)
            final_output = await self._fact_guard.process(final_output)  # ← FACT GUARD

            if plan.is_complete() and not plan.has_failed():
                # ── EVALUASI: quality check output before declaring done ──
                yield self._emit("evaluating", {"msg": "Evaluating output quality..."})
                eval_result = await self._learning.finalize_task(
                    _tracker,
                    success=True,
                    steps_done=len(done_steps),
                    steps_total=len(plan.steps),
                    elapsed_s=elapsed,
                    output=final_output,
                )
                yield self._emit("evaluation_done", {
                    "score": eval_result.get("score"),
                    "goal_met": eval_result.get("goal_met"),
                    "summary": eval_result.get("summary"),
                    "retry_recommended": eval_result.get("retry_recommended", False),
                    "weaknesses": eval_result.get("weaknesses", []),
                })

                # Store final output in memory
                await self._memory.store(
                    key=f"{task_id}_final",
                    content=final_output,
                    metadata={"task_id": task_id, "task": task},
                )
                await state_store.delete()
                self._last_result = AgentResult(
                    task_id=task_id, task=task, success=True,
                    output=final_output,
                    steps_done=len(done_steps), steps_total=len(plan.steps),
                    elapsed=elapsed, events=list(self._events),
                )
                yield self._emit("task_done", {
                    "task_id": task_id,
                    "steps_done": len(done_steps),
                    "elapsed": round(elapsed, 2),
                    "eval_score": eval_result.get("score"),
                    "output_preview": final_output[:400],
                })
            else:
                failed = [s for s in plan.steps if s.status == "failed"]
                err_msg = f"{len(failed)} step(s) failed"
                # Still finalize learning on failure
                await self._learning.finalize_task(
                    _tracker,
                    success=False,
                    steps_done=len(done_steps),
                    steps_total=len(plan.steps),
                    elapsed_s=elapsed,
                    output=final_output,
                )
                self._last_result = AgentResult(
                    task_id=task_id, task=task, success=False,
                    output=final_output,
                    steps_done=len(done_steps), steps_total=len(plan.steps),
                    elapsed=elapsed, events=list(self._events), error=err_msg,
                )
                yield self._emit("task_failed", {"task_id": task_id, "reason": err_msg, "elapsed": round(elapsed, 2)})

        except Exception as e:
            logger.exception("Agent crashed")
            elapsed = time.time() - start
            self._last_result = AgentResult(
                task_id=task_id, task=task, success=False,
                output="", steps_done=0, steps_total=0,
                elapsed=elapsed, error=str(e),
            )
            yield self._emit("error", {"msg": str(e), "task_id": task_id})

    # ── helpers ───────────────────────────────────────────────────────────────

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

    def _compile_output(self, task: str, plan: Plan, context_parts: list[str]) -> str:
        lines = [
            f"# Task: {task}",
            f"## Goal: {plan.goal}",
            "",
            "## Results",
        ]
        for step in plan.steps:
            if step.result:
                lines.append(f"\n### Step {step.id}: {step.description}")
                lines.append(step.result)
        if context_parts:
            last = context_parts[-1]
            if last not in "\n".join(lines):
                lines.append("\n## Final Output")
                lines.append(last)
        return "\n".join(lines)

    async def _save_checkpoint(self, state_store: TaskState, plan: Plan):
        await state_store.save(plan.to_dict())

    async def _load_checkpoint(self, state_store: TaskState, task: str) -> Plan | None:
        data = await state_store.load()
        if data and data.get("task") == task:
            plan = Plan.from_dict(data)
            # Only resume if there are pending/running steps
            has_pending = any(s.status in ("pending", "running", "failed") for s in plan.steps)
            if has_pending:
                # Reset running → pending (may have been interrupted)
                for s in plan.steps:
                    if s.status == "running":
                        s.status = "pending"
                return plan
        return None

    @staticmethod
    def _resolve_env(cfg: dict) -> dict:
        """Replace ${VAR} in config values with env vars."""
        resolved = {}
        for k, v in cfg.items():
            if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
                env_key = v[2:-1]
                resolved[k] = os.environ.get(env_key, "")
            else:
                resolved[k] = v
        return resolved
