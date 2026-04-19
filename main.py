"""
main.py
Local CLI runner for GazccThinking Agent.
Usage:
    python main.py "Research quantum computing, write a report, save to file"
    python main.py --task-id abc123 "Resume a previous task"
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

# ── colour helpers ────────────────────────────────────────────────────────────

RESET = "\033[0m"
BOLD  = "\033[1m"
CYAN  = "\033[96m"
GREEN = "\033[92m"
YELLOW= "\033[93m"
RED   = "\033[91m"
DIM   = "\033[2m"
PINK  = "\033[95m"

def c(text, color): return f"{color}{text}{RESET}"

EVENT_COLORS = {
    "task_start":  CYAN,
    "planning":    YELLOW,
    "plan_ready":  BOLD,
    "step_start":  CYAN,
    "tool_call":   YELLOW,
    "observation": DIM,
    "step_done":   GREEN,
    "step_failed": RED,
    "step_retry":  YELLOW,
    "task_done":   f"{BOLD}{GREEN}",
    "task_failed": f"{BOLD}{RED}",
    "checkpoint":  DIM,
    "error":       RED,
}

ICONS = {
    "task_start":  "🚀",
    "planning":    "🧠",
    "plan_ready":  "📋",
    "step_start":  "▶",
    "tool_call":   "🔧",
    "observation": "👁",
    "step_done":   "✓",
    "step_failed": "✗",
    "step_retry":  "↺",
    "task_done":   "🎉",
    "task_failed": "💥",
    "checkpoint":  "💾",
    "error":       "⚠",
}

def print_event(event):
    color = EVENT_COLORS.get(event.type, "")
    icon  = ICONS.get(event.type, "•")
    ts    = time.strftime("%H:%M:%S", time.localtime(event.timestamp))

    data  = event.data
    etype = event.type.upper()

    print(f"{DIM}[{ts}]{RESET} {color}{icon} {etype}{RESET}", end=" ")

    if event.type == "task_start":
        print(f"{BOLD}{data.get('task', '')}{RESET}")
    elif event.type == "plan_ready":
        print(f"Goal: {data.get('goal', '')}")
        for s in data.get("steps", []):
            deps = f" (deps:{s['depends_on']})" if s.get("depends_on") else ""
            hint = f" [{s['tool_hint']}]" if s.get("tool_hint") else ""
            print(f"  {DIM}[{s['id']}]{deps}{hint}{RESET} {s['description']}")
    elif event.type == "step_start":
        print(f"[{data.get('step_id')}] {data.get('description', '')}")
    elif event.type == "tool_call":
        inp = json.dumps(data.get("input", {}), ensure_ascii=False)
        print(f"{data.get('tool')}({inp[:120]})")
    elif event.type == "observation":
        status = "ok" if data.get("success") else "FAIL"
        print(f"{data.get('tool')} [{status}] → {data.get('output', '')[:200]}")
    elif event.type == "step_done":
        print(f"[{data.get('step_id')}] {data.get('description', '')}")
        preview = data.get("result_preview", "")
        if preview:
            print(f"   {DIM}{preview[:200]}{RESET}")
    elif event.type == "step_failed":
        print(f"[{data.get('step_id')}] {data.get('reason', '')}")
    elif event.type == "task_done":
        elapsed = data.get("elapsed", 0)
        print(f"Task complete in {elapsed:.1f}s ({data.get('steps_done')} steps)")
        print()
        preview = data.get("output_preview", "")
        if preview:
            print(c("── Output Preview ──────────────────────", CYAN))
            print(preview)
            print()
    elif event.type == "task_failed":
        print(f"Reason: {data.get('reason', '')}")
    elif event.type == "error":
        print(f"{data.get('msg', '')}")
    elif event.type == "checkpoint":
        print(f"State saved. {data.get('msg', '')}")
    else:
        print(json.dumps(data, ensure_ascii=False)[:200])


# ── load config ───────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)

    # Resolve env vars in llm.api_key
    llm = cfg.get("llm", {})
    api_key_raw = llm.get("api_key", "")
    if isinstance(api_key_raw, str) and api_key_raw.startswith("${"):
        env_var = api_key_raw[2:-1]
        llm["api_key"] = os.environ.get(env_var, "")
    cfg["llm"] = llm

    # Force file backend for local run
    if cfg.get("memory", {}).get("backend") == "redis":
        cfg["memory"]["backend"] = "file"

    return cfg


# ── main ──────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(
        description="GazccThinking — Autonomous AI Agent",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("task", nargs="+", help="Task to execute")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--task-id", default=None, help="Task ID (for resume)")
    parser.add_argument("--output", default=None, help="Save final output to file")
    parser.add_argument("--json-events", action="store_true", help="Print events as JSON")
    args = parser.parse_args()

    task = " ".join(args.task)
    cfg  = load_config(args.config)

    api_key = cfg.get("llm", {}).get("api_key", "")
    if not api_key:
        print(c("✗ OPENROUTER_API_KEY not set. Export it or put in .env", RED))
        sys.exit(1)

    print()
    print(c("╔══════════════════════════════════════════════╗", CYAN))
    print(c("║       GazccThinking Agent  v1.0              ║", CYAN))
    print(c("╚══════════════════════════════════════════════╝", CYAN))
    print()

    from agent.core import GazccAgent

    agent = GazccAgent(cfg)

    # Launch memory viewer if configured
    viewer_cfg = cfg.get("memory_viewer", {})
    if viewer_cfg.get("enabled", True) and viewer_cfg.get("launch_on_start", True):
        try:
            from agent.memory_viewer import start_viewer
            asyncio.create_task(
                start_viewer(
                    agent._memory,
                    host=viewer_cfg.get("host", "127.0.0.1"),
                    port=viewer_cfg.get("port", 37777),
                )
            )
            print(c(f"  Memory Viewer → http://{viewer_cfg.get('host','127.0.0.1')}:{viewer_cfg.get('port',37777)}", DIM))
        except Exception as ve:
            print(c(f"  Memory viewer skipped: {ve}", DIM))

    async for event in agent.stream(task, task_id=args.task_id):
        if args.json_events:
            print(json.dumps(event.to_dict(), ensure_ascii=False))
            sys.stdout.flush()
        else:
            print_event(event)

    result = agent._last_result
    if result is None:
        print(c("No result produced.", RED))
        sys.exit(1)

    if result.success:
        # Write final output
        out_path = args.output or f"output_{result.task_id}.md"
        Path(out_path).write_text(result.output, encoding="utf-8")
        print()
        print(c(f"✓ Output saved → {out_path}", GREEN))
        print(c(f"  Steps: {result.steps_done}/{result.steps_total} | Time: {result.elapsed:.1f}s", DIM))
    else:
        print()
        print(c(f"✗ Task failed: {result.error}", RED))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
