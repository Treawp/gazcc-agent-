"""
api/index.py
Vercel serverless entry point.
Exposes GazccThinking Agent via FastAPI REST + Server-Sent Events.

Endpoints:
  POST /api/run          – submit task, stream events via SSE
  POST /api/task         – submit task async (returns task_id immediately, NO timeout)
  GET  /api/task/{id}    – poll task status & steps
  GET  /api/stream/{id}  – SSE stream untuk task yang sudah berjalan
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
from typing import AsyncIterator, Dict, Any

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
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
    version="1.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Task store ────────────────────────────────────────────────────────────────
# In-memory dengan TTL cleanup.
# Untuk multi-instance Vercel: ganti dengan Upstash Redis (lihat komentar di bawah).
_tasks: Dict[str, Dict[str, Any]] = {}

# SSE event queues: task_id → asyncio.Queue
# Hanya ada selama ada subscriber aktif.
_event_queues: Dict[str, asyncio.Queue] = {}


def _task_init(task_id: str, task: str) -> Dict:
    """Buat entry task baru di store."""
    entry = {
        "task_id": task_id,
        "status": "pending",   # pending | running | done | failed
        "task": task,
        "events": [],
        "output": "",
        "error": "",
        "elapsed": 0.0,
        "steps_done": 0,
        "steps_total": 0,
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    _tasks[task_id] = entry
    return entry


def _task_push_event(task_id: str, ev_dict: dict):
    """Simpan event ke store + kirim ke SSE queue jika ada subscriber."""
    entry = _tasks.get(task_id)
    if entry:
        entry["events"].append(ev_dict)
        entry["updated_at"] = time.time()

    q = _event_queues.get(task_id)
    if q:
        try:
            q.put_nowait(ev_dict)
        except asyncio.QueueFull:
            pass  # Subscriber lambat, skip event ini


# ── models ────────────────────────────────────────────────────────────────────

class TaskRequest(BaseModel):
    task: str
    task_id: str | None = None
    stream: bool = True


class TaskStatus(BaseModel):
    task_id: str
    status: str          # pending | running | done | failed
    task: str
    events: list[dict]
    output: str = ""
    error: str = ""
    elapsed: float = 0.0
    steps_done: int = 0
    steps_total: int = 0


# ── SSE generator (dipakai oleh /api/run dan /api/stream/{id}) ────────────────

async def _sse_gen(task: str, task_id: str) -> AsyncIterator[str]:
    """Stream AgentEvent objects as Server-Sent Events."""
    from agent.core import GazccAgent

    agent = GazccAgent(CFG)
    entry = _tasks.get(task_id)
    if entry:
        entry["status"] = "running"
        entry["updated_at"] = time.time()

    try:
        async for event in agent.stream(task, task_id=task_id):
            ev_dict = event.to_dict()
            _task_push_event(task_id, ev_dict)

            if event.type == "task_done":
                r = agent._last_result
                if r:
                    entry = _tasks.get(task_id, {})
                    entry.update({
                        "status": "done",
                        "output": r.output,
                        "elapsed": r.elapsed,
                        "steps_done": r.steps_done,
                        "steps_total": r.steps_total,
                        "updated_at": time.time(),
                    })
            elif event.type in ("task_failed", "error"):
                entry = _tasks.get(task_id, {})
                entry.update({
                    "status": "failed",
                    "error": event.data.get("reason") or event.data.get("msg", ""),
                    "updated_at": time.time(),
                })

            yield f"data: {json.dumps(ev_dict, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0)  # yield control ke event loop

        # Final stream_end event
        if agent._last_result:
            r = agent._last_result
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
            _task_push_event(task_id, final)
            yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"

    except Exception as e:
        err_ev = {"type": "error", "data": {"msg": str(e)}, "timestamp": time.time()}
        _task_push_event(task_id, err_ev)
        entry = _tasks.get(task_id, {})
        entry.update({"status": "failed", "error": str(e), "updated_at": time.time()})
        yield f"data: {json.dumps(err_ev)}\n\n"
    finally:
        # Sentinel agar SSE subscriber tahu stream selesai
        _task_push_event(task_id, {"type": "_eof"})
        yield "data: [DONE]\n\n"


# ── Background runner (untuk /api/task) ───────────────────────────────────────

async def _run_background(task: str, task_id: str):
    """
    Jalankan agent sampai selesai di background.
    Dipanggil via FastAPI BackgroundTasks — tidak block response.
    Bisa berjalan 5-10 menit tanpa kena Vercel 90s timeout karena
    request /api/task sudah selesai (202) sebelum ini berjalan.
    """
    async for _ in _sse_gen(task, task_id):
        pass  # Event sudah di-push ke _task_push_event, tidak perlu di-yield


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return JSONResponse({
        "agent": "GazccThinking",
        "version": "1.1.0",
        "status": "online",
        "endpoints": {
            "POST /api/run":          "Stream task events langsung (SSE)",
            "POST /api/task":         "Submit task async, return task_id (no timeout)",
            "GET  /api/task/{id}":    "Poll status + events",
            "GET  /api/stream/{id}":  "SSE stream untuk task background",
            "GET  /api/memory":       "List memory entries",
            "GET  /api/tools":        "List available tools",
        },
    })


@app.post("/api/run")
async def run_task(req: TaskRequest):
    """
    Submit task → terima SSE stream langsung.
    Cocok untuk task pendek. Untuk task panjang (>90s) pakai /api/task.
    """
    task_id = req.task_id or str(uuid.uuid4())[:8]
    if not req.task.strip():
        raise HTTPException(400, "task cannot be empty")
    api_key = CFG.get("llm", {}).get("api_key", "")
    if not api_key:
        raise HTTPException(500, "OPENROUTER_API_KEY not configured")

    _task_init(task_id, req.task)

    return StreamingResponse(
        _sse_gen(req.task, task_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Task-Id": task_id,
        },
    )


@app.post("/api/task", status_code=202)
async def submit_task(req: TaskRequest, background_tasks: BackgroundTasks):
    """
    Submit task → langsung return task_id (HTTP 202, tidak tunggu).
    Agent berjalan di background via FastAPI BackgroundTasks.
    Poll /api/task/{id} setiap 5 detik untuk cek status.
    Task bisa berjalan 5-10 menit tanpa timeout.
    """
    task_id = req.task_id or str(uuid.uuid4())[:8]
    if not req.task.strip():
        raise HTTPException(400, "task cannot be empty")
    api_key = CFG.get("llm", {}).get("api_key", "")
    if not api_key:
        raise HTTPException(500, "OPENROUTER_API_KEY not configured")

    # Init task di store dengan status pending
    _task_init(task_id, req.task)

    # Daftarkan ke BackgroundTasks — AMAN, tidak drop saat response selesai
    background_tasks.add_task(_run_background, req.task, task_id)

    return {
        "id": task_id,
        "task_id": task_id,
        "status": "pending",
        "message": "Task diterima. Agent berjalan di background.",
        "poll_url": f"/api/task/{task_id}",
        "stream_url": f"/api/stream/{task_id}",
    }


@app.get("/api/task/{task_id}")
async def get_task(task_id: str):
    """
    Poll status task.
    Response: status (pending|running|done|failed), output, events, elapsed.
    """
    entry = _tasks.get(task_id)
    if entry is None:
        raise HTTPException(404, f"Task '{task_id}' tidak ditemukan")
    return JSONResponse(entry)


@app.get("/api/stream/{task_id}")
async def stream_task(task_id: str):
    """
    SSE stream untuk task yang sudah disubmit via /api/task.
    Jika task sudah selesai sebelum klien connect → kirim semua events lalu tutup.
    """
    entry = _tasks.get(task_id)
    if entry is None:
        raise HTTPException(404, f"Task '{task_id}' tidak ditemukan")

    async def generator():
        # Jika sudah selesai: replay semua events langsung
        if entry["status"] in ("done", "failed"):
            for ev in entry.get("events", []):
                if ev.get("type") != "_eof":
                    yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Task masih running: subscribe ke queue
        q: asyncio.Queue = asyncio.Queue(maxsize=500)
        _event_queues[task_id] = q

        # Replay events yang sudah ada sebelum subscribe
        for ev in entry.get("events", []):
            if ev.get("type") != "_eof":
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"

        try:
            while True:
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    yield ": ping\n\n"  # Keepalive agar koneksi tidak drop
                    continue

                if ev.get("type") == "_eof":
                    yield "data: [DONE]\n\n"
                    break

                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"

                if ev.get("type") in ("stream_end", "task_done", "task_failed", "error"):
                    yield "data: [DONE]\n\n"
                    break
        finally:
            _event_queues.pop(task_id, None)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


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


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "tasks_tracked": len(_tasks),
        "active_streams": len(_event_queues),
        "timestamp": time.time(),
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
      const lines = buf.split('\\n');
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
