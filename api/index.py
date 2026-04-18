"""
api/index.py
Vercel serverless entry point.
Exposes GazccThinking Agent via FastAPI REST + Server-Sent Events.

Endpoints:
  POST /api/run          – submit task, stream events via SSE
  POST /api/task         – submit task async (returns task_id immediately)
  GET  /api/task/{id}    – poll task status
  GET  /api/memory       – list memory keys
  GET  /api/tools        – list available tools
  GET  /                 – health check
"""

import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import AsyncIterator

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

load_dotenv()

# ── load config ───────────────────────────────────────────────────────────────

def _load_cfg() -> dict:
    cfg_path = Path(__file__).parent.parent / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Resolve env vars
    llm = cfg.get("llm", {})
    api_key_raw = llm.get("api_key", "")
    if isinstance(api_key_raw, str) and api_key_raw.startswith("${"):
        env_var = api_key_raw[2:-1]
        llm["api_key"] = os.environ.get(env_var, "")
    cfg["llm"] = llm

    # Vercel: force /tmp for file backend, or use redis if configured
    redis_url = os.environ.get("UPSTASH_REDIS_REST_URL", "")
    if redis_url:
        cfg["memory"]["backend"] = "redis"
    else:
        cfg["memory"]["backend"] = "file"
        cfg["memory"]["file_path"] = "/tmp/gazcc_memory"

    # Disable code exec on Vercel (security)
    cfg.setdefault("tools", {})["code_exec"] = os.environ.get("ALLOW_CODE_EXEC", "false").lower() == "true"

    return cfg


CFG = _load_cfg()

# ── app ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="GazccThinking Agent API",
    description="Autonomous AI Agent — Powered by GazccThinking",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory task tracker (Vercel: each invocation is isolated, use Redis for persistence)
_tasks: dict[str, dict] = {}


# ── models ────────────────────────────────────────────────────────────────────

class TaskRequest(BaseModel):
    task: str
    task_id: str | None = None
    stream: bool = True


class TaskStatus(BaseModel):
    task_id: str
    status: str          # running | done | failed
    task: str
    events: list[dict]
    output: str = ""
    error: str = ""
    elapsed: float = 0.0
    steps_done: int = 0
    steps_total: int = 0


class SandboxRequest(BaseModel):
    message: str
    session_id: str | None = None
    history: list[dict] = []
    memory: dict = {}


# Sandbox session store (history + memory per session)
_sandbox_sessions: dict[str, dict] = {}


# ── SSE helper ────────────────────────────────────────────────────────────────

async def _sse_gen(task: str, task_id: str) -> AsyncIterator[str]:
    """Stream AgentEvent objects as Server-Sent Events."""
    from agent.core import GazccAgent

    agent = GazccAgent(CFG)
    _tasks[task_id] = {"status": "running", "task": task, "events": [], "output": "", "error": ""}

    try:
        async for event in agent.stream(task, task_id=task_id):
            ev_dict = event.to_dict()
            _tasks[task_id]["events"].append(ev_dict)

            if event.type == "task_done":
                _tasks[task_id]["status"] = "done"
                _tasks[task_id]["output"] = agent._last_result.output if agent._last_result else ""
            elif event.type == "task_failed":
                _tasks[task_id]["status"] = "failed"
                _tasks[task_id]["error"] = event.data.get("reason", "")
            elif event.type == "error":
                _tasks[task_id]["status"] = "failed"
                _tasks[task_id]["error"] = event.data.get("msg", "")

            yield f"data: {json.dumps(ev_dict, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0)  # yield control

        # Final result
        if agent._last_result:
            r = agent._last_result
            _tasks[task_id].update({
                "elapsed": r.elapsed,
                "steps_done": r.steps_done,
                "steps_total": r.steps_total,
                "output": r.output,
            })
            final = {
                "type": "stream_end",
                "data": {
                    "task_id": task_id,
                    "success": r.success,
                    "output": r.output,
                    "steps_done": r.steps_done,
                    "steps_total": r.steps_total,
                    "elapsed": round(r.elapsed, 2),
                },
                "timestamp": time.time(),
            }
            yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"

    except Exception as e:
        err_ev = {"type": "error", "data": {"msg": str(e)}, "timestamp": time.time()}
        _tasks[task_id]["status"] = "failed"
        _tasks[task_id]["error"] = str(e)
        yield f"data: {json.dumps(err_ev)}\n\n"
    finally:
        yield "data: [DONE]\n\n"


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return JSONResponse({
        "agent": "GazccThinking",
        "version": "1.0.0",
        "status": "online",
        "endpoints": {
            "POST /api/run":            "Stream task events (SSE)",
            "POST /api/task":           "Submit task, get task_id",
            "GET /api/task/{id}":       "Poll task status",
            "GET /api/memory":          "List memory entries",
            "GET /api/tools":           "List available tools",
            "POST /api/sandbox":        "Sandbox Executor — chat with code agent (SSE)",
            "POST /api/sandbox/reset":  "Reset sandbox session",
            "GET /api/sandbox/{id}":    "Get sandbox session state",
        },
    })


@app.post("/api/run")
async def run_task(req: TaskRequest):
    """
    Submit a task and receive a Server-Sent Events stream of AgentEvent objects.
    Each event: `data: {"type": "...", "data": {...}, "timestamp": ...}`
    Final event: `data: [DONE]`
    """
    task_id = req.task_id or str(uuid.uuid4())[:8]
    if not req.task.strip():
        raise HTTPException(400, "task cannot be empty")
    api_key = CFG.get("llm", {}).get("api_key", "")
    if not api_key:
        raise HTTPException(500, "OPENROUTER_API_KEY not configured")

    return StreamingResponse(
        _sse_gen(req.task, task_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Task-Id": task_id,
        },
    )


@app.post("/api/task")
async def submit_task(req: TaskRequest):
    """Submit task and immediately return task_id. Poll /api/task/{id} for status."""
    task_id = req.task_id or str(uuid.uuid4())[:8]
    if not req.task.strip():
        raise HTTPException(400, "task cannot be empty")

    _tasks[task_id] = {"status": "running", "task": req.task, "events": [], "output": "", "error": ""}

    async def _bg():
        async for _ in _sse_gen(req.task, task_id):
            pass

    asyncio.create_task(_bg())
    return {"task_id": task_id, "status": "running", "poll_url": f"/api/task/{task_id}"}


@app.get("/api/task/{task_id}")
async def get_task(task_id: str):
    """Poll task status and events."""
    state = _tasks.get(task_id)
    if state is None:
        raise HTTPException(404, f"Task {task_id!r} not found")
    return JSONResponse(state)


@app.get("/api/memory")
async def list_memory(q: str = Query("", description="Search query")):
    """List or search memory entries."""
    from agent.memory import build_memory
    mem = build_memory(CFG.get("memory", {}))
    if q:
        entries = await mem.search(q, top_k=10)
        return {"query": q, "results": [{"key": e.key, "content": e.content[:500], "metadata": e.metadata} for e in entries]}
    keys = await mem.all_keys()
    return {"keys": keys, "count": len(keys)}


@app.get("/api/tools")
async def list_tools():
    """List all registered tools."""
    from agent.tools import ToolRegistry
    reg = ToolRegistry(CFG)
    return {"tools": reg.list_all()}


# ── Sandbox Executor endpoints ────────────────────────────────────────────────

async def _sandbox_sse(session_id: str, message: str, history: list, memory: dict) -> AsyncIterator[str]:
    """SSE generator for sandbox chat."""
    from agent.sandbox import SandboxExecutor

    executor = SandboxExecutor(CFG)
    executor.load_state(history, memory)

    try:
        async for event in executor.chat(message):
            ev_dict = event.to_dict()

            # Keep session state up-to-date on each turn
            if event.type == "response":
                _sandbox_sessions[session_id] = {
                    "history": executor.history,
                    "memory": executor.memory,
                }

            yield f"data: {json.dumps(ev_dict, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0)

    except Exception as e:
        err = {"type": "error", "data": {"msg": str(e)}, "timestamp": time.time()}
        yield f"data: {json.dumps(err)}\n\n"
    finally:
        yield "data: [DONE]\n\n"


@app.post("/api/sandbox")
async def sandbox_chat(req: SandboxRequest):
    """
    Send a message to the Sandbox Executor and receive an SSE stream of events.

    Pass session_id to continue a previous session, or omit to start fresh.
    Each event: data: {"type": "...", "data": {...}, "timestamp": ...}
    Types: thinking | tool_call | tool_result | memory | response | error
    Final event: data: [DONE]
    """
    if not req.message.strip():
        raise HTTPException(400, "message cannot be empty")
    api_key = CFG.get("llm", {}).get("api_key", "")
    if not api_key:
        raise HTTPException(500, "OPENROUTER_API_KEY not configured")

    session_id = req.session_id or str(uuid.uuid4())[:8]

    # Restore session if it exists, otherwise use history/memory from request body
    if session_id in _sandbox_sessions:
        session = _sandbox_sessions[session_id]
        history = session["history"]
        memory = session["memory"]
    else:
        history = req.history
        memory = req.memory
        _sandbox_sessions[session_id] = {"history": history, "memory": memory}

    return StreamingResponse(
        _sandbox_sse(session_id, req.message, history, memory),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Session-Id": session_id,
        },
    )


@app.post("/api/sandbox/reset")
async def sandbox_reset(session_id: str = Query(..., description="Session ID to reset")):
    """Clear conversation history and memory for a sandbox session."""
    if session_id in _sandbox_sessions:
        del _sandbox_sessions[session_id]
        return {"status": "reset", "session_id": session_id}
    return {"status": "not_found", "session_id": session_id}


@app.get("/api/sandbox/{session_id}")
async def sandbox_state(session_id: str):
    """Get current history and memory for a sandbox session."""
    session = _sandbox_sessions.get(session_id)
    if session is None:
        raise HTTPException(404, f"Session '{session_id}' not found")
    return {
        "session_id": session_id,
        "turns": len(session["history"]) // 2,
        "memory": session["memory"],
        "history_length": len(session["history"]),
    }


# ── demo UI ───────────────────────────────────────────────────────────────────

@app.get("/ui", response_class=HTMLResponse)
async def demo_ui():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GazccThinking Agent</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#04060a;color:#dde3f0;font-family:'Segoe UI',sans-serif;min-height:100vh;display:flex;flex-direction:column}
header{padding:18px 24px;border-bottom:1px solid rgba(0,212,255,0.1);display:flex;align-items:center;gap:12px}
.logo{width:40px;height:40px;border-radius:12px;background:linear-gradient(135deg,#7b2fff,#00d4ff);display:flex;align-items:center;justify-content:center;font-size:20px}
h1{font-size:18px;font-weight:700;background:linear-gradient(90deg,#7b2fff,#00d4ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.sub{font-size:11px;color:#4a5980;margin-top:2px}
main{flex:1;padding:24px;max-width:900px;margin:0 auto;width:100%;display:flex;flex-direction:column;gap:16px}
.input-row{display:flex;gap:10px}
#taskInput{flex:1;background:rgba(0,0,0,0.4);border:1px solid #1a2540;border-radius:12px;padding:12px 16px;color:#dde3f0;font-size:14px;outline:none;transition:border-color .2s}
#taskInput:focus{border-color:#00d4ff}
#runBtn{padding:12px 24px;border-radius:12px;background:linear-gradient(135deg,#7b2fff,#00d4ff);border:none;color:#fff;font-weight:700;cursor:pointer;font-size:14px;white-space:nowrap}
#runBtn:disabled{opacity:.4;cursor:not-allowed}
#log{background:rgba(0,0,0,0.3);border:1px solid #1a2540;border-radius:12px;padding:16px;min-height:400px;max-height:60vh;overflow-y:auto;font-family:'JetBrains Mono',monospace;font-size:12px;line-height:1.8}
.ev{margin-bottom:4px;padding:2px 0}
.ev .ts{color:#4a5980;margin-right:8px}
.ev .icon{margin-right:4px}
.ev.task_start .label,.ev.task_done .label{color:#00d4ff;font-weight:700}
.ev.step_done .label{color:#06d6a0}
.ev.step_failed .label,.ev.task_failed .label,.ev.error .label{color:#f72585}
.ev.tool_call .label{color:#ffd60a}
.ev.plan_ready .label{color:#fff;font-weight:600}
.ev .label{color:#7b2fff}
.step-list{margin-left:16px;margin-top:4px}
.step-list .step{color:#4a5980;padding:1px 0}
.step-list .step.done{color:#06d6a0}
#statusBar{font-size:11px;color:#4a5980;text-align:right}
</style>
</head>
<body>
<header>
  <div class="logo">🧠</div>
  <div>
    <h1>GazccThinking Agent</h1>
    <div class="sub">Autonomous AI — ReAct loop · Multi-step · Self-correcting</div>
  </div>
</header>
<main>
  <div class="input-row">
    <input id="taskInput" placeholder='Try: "Research quantum computing, write a report, save to report.md"' value="">
    <button id="runBtn" onclick="runTask()">▶ Run</button>
  </div>
  <div id="statusBar">Ready</div>
  <div id="log"><span style="color:#4a5980">// events will appear here...</span></div>
</main>
<script>
const ICONS = {task_start:'🚀',planning:'🧠',plan_ready:'📋',step_start:'▶',tool_call:'🔧',
  observation:'👁',step_done:'✓',step_failed:'✗',step_retry:'↺',task_done:'🎉',
  task_failed:'💥',checkpoint:'💾',error:'⚠',stream_end:'⬛'};

function ts(epoch){ return new Date(epoch*1000).toTimeString().slice(0,8); }

function appendEvent(ev){
  const log = document.getElementById('log');
  const row = document.createElement('div');
  row.className = 'ev ' + ev.type;
  const d = ev.data || {};
  let body = '';
  if(ev.type==='task_start') body = `<strong>${d.task}</strong>`;
  else if(ev.type==='plan_ready'){
    body = `Goal: ${d.goal}<div class="step-list">${(d.steps||[]).map(s=>`<div class="step">[${s.id}] ${s.description}</div>`).join('')}</div>`;
  }
  else if(ev.type==='step_start') body = `[${d.step_id}] ${d.description}`;
  else if(ev.type==='step_done') body = `[${d.step_id}] ✓ ${d.description}`;
  else if(ev.type==='tool_call') body = `${d.tool}(${JSON.stringify(d.input||{}).slice(0,100)})`;
  else if(ev.type==='observation') body = `${d.tool} → ${(d.output||'').slice(0,200)}`;
  else if(ev.type==='task_done') body = `Done in ${d.elapsed}s · ${d.steps_done} steps<br><small style="color:#4a5980">${(d.output_preview||'').slice(0,300)}</small>`;
  else if(ev.type==='stream_end') body = `<strong>Output ready</strong> · ${d.steps_done} steps · ${d.elapsed}s`;
  else body = JSON.stringify(d).slice(0,200);

  row.innerHTML = `<span class="ts">${ts(ev.timestamp)}</span><span class="icon">${ICONS[ev.type]||'·'}</span><span class="label">${ev.type.toUpperCase()}</span> ${body}`;
  if(log.firstChild && log.firstChild.textContent.includes('events will appear')){log.innerHTML='';}
  log.appendChild(row);
  log.scrollTop = log.scrollHeight;
}

async function runTask(){
  const task = document.getElementById('taskInput').value.trim();
  if(!task) return;
  document.getElementById('log').innerHTML = '';
  const btn = document.getElementById('runBtn');
  btn.disabled = true;
  document.getElementById('statusBar').textContent = 'Running...';

  try{
    const res = await fetch('/api/run', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({task})
    });
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    while(true){
      const {done, value} = await reader.read();
      if(done) break;
      buf += dec.decode(value, {stream:true});
      const lines = buf.split('\n');
      buf = lines.pop();
      for(const line of lines){
        if(!line.startsWith('data: ')) continue;
        const raw = line.slice(6).trim();
        if(raw==='[DONE]'){ document.getElementById('statusBar').textContent='Complete'; break; }
        try{ appendEvent(JSON.parse(raw)); }catch(e){}
      }
    }
  }catch(e){
    appendEvent({type:'error',data:{msg:e.message},timestamp:Date.now()/1000});
  }
  btn.disabled = false;
}

document.getElementById('taskInput').addEventListener('keydown',e=>{if(e.key==='Enter')runTask();});
</script>
</body>
</html>"""


# ── /api/proxy — OpenRouter chat proxy (dipanggil frontend) ───────────────────

import httpx as _httpx

@app.get("/api/proxy")
async def proxy_chat(
    question: str = Query(..., description="User message / full prompt"),
    system: str = Query("You are a helpful AI assistant.", description="System prompt"),
    key: str = Query("", description="OpenRouter API key"),
    model: str = Query("", description="Model ID"),
):
    cfg_llm = CFG.get("llm", {})
    api_key = key or cfg_llm.get("api_key", "") or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise HTTPException(500, "OPENROUTER_API_KEY not configured")

    selected_model = model or cfg_llm.get("model", "google/gemma-4-26b-a4b-it")
    base_url = cfg_llm.get("base_url", "https://openrouter.ai/api/v1").rstrip("/")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://gazccai.vercel.app",
        "X-Title": "GazccAI",
    }
    payload = {
        "model": selected_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
        "max_tokens": 4000,
        "temperature": 0.7,
    }

    try:
        async with _httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(f"{base_url}/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            result = data["choices"][0]["message"]["content"]
            return JSONResponse({"status": True, "data": {"result": result}})
    except _httpx.HTTPStatusError as e:
        return JSONResponse({"status": False, "message": f"OpenRouter HTTP {e.response.status_code}: {e.response.text[:200]}"})
    except Exception as e:
        return JSONResponse({"status": False, "message": str(e)})


# ── /api/download/{token} — serve exported files ──────────────────────────────

import base64 as _base64
from fastapi.responses import Response as _Response

@app.get("/api/download/{token}")
async def download_file(token: str):
    """
    Serve file yang disimpan Agent di Redis via export_file_link tool.
    Token di-generate Agent, berlaku 1 jam.
    """
    redis_url   = os.environ.get("UPSTASH_REDIS_REST_URL", "")
    redis_token = os.environ.get("UPSTASH_REDIS_REST_TOKEN", "")

    if not redis_url or not redis_token:
        raise HTTPException(404, "File storage not configured (Redis unavailable)")

    redis_key = f"gazcc_export:{token}"
    try:
        async with _httpx.AsyncClient(timeout=10) as client:
            r = await client.get(
                f"{redis_url}/get/{redis_key}",
                headers={"Authorization": f"Bearer {redis_token}"},
            )
            r.raise_for_status()
            result = r.json()

        raw_val = result.get("result")
        if not raw_val:
            raise HTTPException(404, "File not found or expired (link valid for 1 hour)")

        meta = json.loads(raw_val)
        filename = meta["filename"]
        mime     = meta.get("mime", "application/octet-stream")
        data     = _base64.b64decode(meta["data"])

        return _Response(
            content=data,
            media_type=mime,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(len(data)),
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Download error: {e}")
