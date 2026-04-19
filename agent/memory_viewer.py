"""
agent/memory_viewer.py
═══════════════════════════════════════════════════════════════════════════════
  GAZCC — MEMORY VIEWER
  ════════════════════════════════════════════════════════════════════════════

  Lightweight FastAPI server on http://localhost:37777
  Runs as background asyncio task alongside the main agent.

  Endpoints:
    GET /                          — Web UI (real-time memory stream)
    GET /api/observations          — List all observations (paginated)
    GET /api/observation/{obs_id}  — Get single observation by ID
    GET /api/stream                — SSE feed of live memory events
    GET /api/stats                 — Memory stats (count, total tokens, etc.)
    POST /api/search               — Search memory { "query": "...", "top_k": 5 }

  Launch:
    from agent.memory_viewer import start_viewer
    asyncio.create_task(start_viewer(memory_backend))
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Optional

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
    from pydantic import BaseModel
    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False

from .memory_features import bus, MemoryEvent

_VIEWER_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gazcc Memory Viewer</title>
<style>
  :root{--bg:#0e1117;--panel:#161b27;--border:#21293b;--accent:#4f8ef7;
        --green:#3ddc84;--yellow:#f7c948;--red:#f75f5f;--dim:#5b6a8a;
        --text:#e0e6f0;--mono:'JetBrains Mono',monospace}
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:var(--mono);font-size:13px}
  header{display:flex;align-items:center;gap:12px;padding:14px 20px;
         border-bottom:1px solid var(--border);background:var(--panel)}
  header h1{font-size:16px;font-weight:700;letter-spacing:.5px}
  header .badge{background:var(--accent);color:#fff;padding:2px 8px;border-radius:10px;
                font-size:11px;font-weight:600}
  #stats{display:flex;gap:24px;padding:10px 20px;
         border-bottom:1px solid var(--border);background:var(--panel)}
  .stat{display:flex;flex-direction:column}
  .stat .label{font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:.8px}
  .stat .value{font-size:18px;font-weight:700;color:var(--accent)}
  #controls{padding:10px 20px;display:flex;gap:8px;border-bottom:1px solid var(--border)}
  #search{flex:1;background:var(--panel);border:1px solid var(--border);color:var(--text);
          padding:6px 12px;border-radius:6px;font-family:var(--mono);font-size:12px;outline:none}
  #search:focus{border-color:var(--accent)}
  button{background:var(--accent);color:#fff;border:none;padding:6px 14px;
         border-radius:6px;cursor:pointer;font-family:var(--mono);font-size:12px;font-weight:600}
  button:hover{opacity:.85}
  #log{padding:12px 20px;overflow-y:auto;height:calc(100vh - 190px)}
  .entry{border:1px solid var(--border);border-radius:8px;margin-bottom:10px;
         background:var(--panel);overflow:hidden}
  .entry-header{display:flex;align-items:center;gap:10px;padding:8px 12px;
                border-bottom:1px solid var(--border);background:rgba(79,142,247,.06)}
  .tier{font-size:10px;font-weight:700;padding:1px 7px;border-radius:4px;text-transform:uppercase}
  .tier-store{background:#1a3a1a;color:var(--green)}
  .tier-search{background:#1a1a3a;color:var(--accent)}
  .tier-delete{background:#3a1a1a;color:var(--red)}
  .tier-retrieve{background:#2a2a1a;color:var(--yellow)}
  .entry-key{font-weight:600;color:var(--accent);flex:1}
  .entry-obs{font-size:10px;color:var(--dim);font-family:var(--mono)}
  .entry-ts{font-size:10px;color:var(--dim)}
  .entry-tokens{font-size:10px;color:var(--yellow);margin-left:auto}
  .entry-body{padding:10px 12px;font-size:12px;line-height:1.6;color:#c8d0e0;
              white-space:pre-wrap;word-break:break-word;max-height:200px;overflow-y:auto}
  .citation-link{font-size:10px;color:var(--dim);text-decoration:none;margin-left:4px}
  .citation-link:hover{color:var(--accent)}
  #status{font-size:11px;color:var(--green);margin-left:auto}
  #status.disconnected{color:var(--red)}
  .empty{text-align:center;color:var(--dim);padding:40px}
</style>
</head>
<body>
<header>
  <h1>⚡ Gazcc Memory Viewer</h1>
  <span class="badge">LIVE</span>
  <span id="status">● connected</span>
</header>
<div id="stats">
  <div class="stat"><span class="label">Observations</span><span class="value" id="s-count">0</span></div>
  <div class="stat"><span class="label">Total Tokens</span><span class="value" id="s-tokens">0</span></div>
  <div class="stat"><span class="label">Live Events</span><span class="value" id="s-live">0</span></div>
</div>
<div id="controls">
  <input id="search" placeholder="Search memory... (press Enter)" />
  <button onclick="doSearch()">Search</button>
  <button onclick="loadAll()">Load All</button>
  <button onclick="clearLog()" style="background:#3a1a1a;color:var(--red)">Clear View</button>
</div>
<div id="log"><div class="empty">Waiting for memory events…</div></div>

<script>
let liveCount=0, totalTokens=0, obsCount=0;

function ts(t){return new Date(t*1000).toLocaleTimeString()}
function addEntry(e, prepend=false){
  const log=document.getElementById('log');
  if(log.querySelector('.empty')) log.innerHTML='';
  const tier=e.type||'store';
  const d=document.createElement('div');
  d.className='entry';
  const obsShort=(e.obs_id||'').slice(0,8);
  const link=e.obs_id?`<a class="citation-link" href="/api/observation/${e.obs_id}" target="_blank">#${obsShort}</a>`:'';
  d.innerHTML=`
    <div class="entry-header">
      <span class="tier tier-${tier}">${tier}</span>
      <span class="entry-key">${e.key||'-'}</span>
      ${link}
      <span class="entry-tokens">~${e.tokens||0} tok</span>
      <span class="entry-ts">${ts(e.ts||Date.now()/1000)}</span>
    </div>
    <div class="entry-body">${(e.preview||'').replace(/</g,'&lt;')}</div>`;
  if(prepend) log.insertBefore(d,log.firstChild);
  else log.appendChild(d);
}

// SSE stream
function connect(){
  const es=new EventSource('/api/stream');
  es.onmessage=ev=>{
    try{
      const data=JSON.parse(ev.data);
      liveCount++;
      if(data.tokens) totalTokens+=data.tokens;
      document.getElementById('s-live').textContent=liveCount;
      document.getElementById('s-tokens').textContent=totalTokens;
      addEntry(data,true);
    }catch(_){}
  };
  es.onerror=()=>{
    document.getElementById('status').textContent='● disconnected';
    document.getElementById('status').className='disconnected';
    setTimeout(connect,3000);
  };
  es.onopen=()=>{
    document.getElementById('status').textContent='● connected';
    document.getElementById('status').className='';
  };
}

async function loadAll(){
  const r=await fetch('/api/observations?limit=100');
  const data=await r.json();
  const log=document.getElementById('log');
  log.innerHTML='';
  obsCount=data.total||0;
  document.getElementById('s-count').textContent=obsCount;
  let tok=0;
  (data.entries||[]).forEach(e=>{tok+=e.token_cost||0;addEntry({
    type:'store', key:e.key, obs_id:e.obs_id, preview:e.content?.slice(0,300)||'',
    tokens:e.token_cost||0, ts:e.timestamp||Date.now()/1000
  })});
  totalTokens=tok;
  document.getElementById('s-tokens').textContent=tok;
}

async function doSearch(){
  const q=document.getElementById('search').value.trim();
  if(!q) return;
  const r=await fetch('/api/search',{
    method:'POST', headers:{'Content-Type':'application/json'},
    body:JSON.stringify({query:q, top_k:10})
  });
  const data=await r.json();
  const log=document.getElementById('log');
  log.innerHTML=`<div style="padding:8px 12px;color:var(--yellow);font-size:11px">
    🔍 Results for: "${q}" (${(data.results||[]).length} found)</div>`;
  (data.results||[]).forEach(e=>addEntry({
    type:'search', key:e.key, obs_id:e.obs_id,
    preview:e.content?.slice(0,300)||'', tokens:e.token_cost||0,
    ts:e.timestamp||Date.now()/1000
  }));
}

function clearLog(){document.getElementById('log').innerHTML='<div class="empty">Cleared.</div>';}
document.getElementById('search').addEventListener('keydown',e=>{if(e.key==='Enter')doSearch()});

connect();
loadAll();
</script>
</body>
</html>
"""


def _build_viewer_app(memory_backend) -> "FastAPI":
    app = FastAPI(title="Gazcc Memory Viewer", version="1.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    class SearchRequest(BaseModel):
        query: str
        top_k: int = 10

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return HTMLResponse(_VIEWER_HTML)

    @app.get("/api/observations")
    async def list_observations(
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ):
        if hasattr(memory_backend, "all_entries"):
            entries = await memory_backend.all_entries()
        else:
            keys = await memory_backend.all_keys()
            entries = []
            for k in keys:
                e = await memory_backend.retrieve(k)
                if e:
                    entries.append(e)
            entries.sort(key=lambda x: x.timestamp, reverse=True)

        total = len(entries)
        page = entries[offset: offset + limit]
        return JSONResponse({
            "total": total,
            "offset": offset,
            "limit": limit,
            "entries": [e.to_dict() for e in page],
        })

    @app.get("/api/observation/{obs_id}")
    async def get_observation(obs_id: str):
        if hasattr(memory_backend, "retrieve_by_obs_id"):
            entry = await memory_backend.retrieve_by_obs_id(obs_id)
        else:
            # Fallback: linear scan
            keys = await memory_backend.all_keys()
            entry = None
            for k in keys:
                e = await memory_backend.retrieve(k)
                if e and e.obs_id == obs_id:
                    entry = e
                    break
        if not entry:
            raise HTTPException(404, f"obs_id '{obs_id}' not found")
        return JSONResponse(entry.to_dict())

    @app.get("/api/stream")
    async def sse_stream():
        q = bus.subscribe()

        async def generator():
            # Send heartbeat first
            yield ": connected\n\n"
            try:
                while True:
                    try:
                        event: MemoryEvent = await asyncio.wait_for(q.get(), timeout=25)
                        yield event.to_sse()
                    except asyncio.TimeoutError:
                        yield ": heartbeat\n\n"
            except asyncio.CancelledError:
                pass
            finally:
                bus.unsubscribe(q)

        return StreamingResponse(
            generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/stats")
    async def stats():
        if hasattr(memory_backend, "all_entries"):
            entries = await memory_backend.all_entries()
        else:
            keys = await memory_backend.all_keys()
            entries = []
            for k in keys:
                e = await memory_backend.retrieve(k)
                if e:
                    entries.append(e)

        total_tokens = sum(getattr(e, "token_cost", 0) for e in entries)
        return JSONResponse({
            "count":        len(entries),
            "total_tokens": total_tokens,
            "oldest":       min((e.timestamp for e in entries), default=0),
            "newest":       max((e.timestamp for e in entries), default=0),
        })

    @app.post("/api/search")
    async def search_memory(req: SearchRequest):
        if hasattr(memory_backend, "search_with_scores"):
            scored = await memory_backend.search_with_scores(req.query, top_k=req.top_k)
            results = [
                {**e.to_dict(), "score": round(s, 4)}
                for s, e in scored
            ]
        else:
            entries = await memory_backend.search(req.query, top_k=req.top_k)
            results = [e.to_dict() for e in entries]
        return JSONResponse({"query": req.query, "results": results})

    return app


async def start_viewer(memory_backend, host: str = "127.0.0.1", port: int = 37777):
    """
    Launch memory viewer as background asyncio task.

    Usage in main.py:
        from agent.memory_viewer import start_viewer
        asyncio.create_task(start_viewer(agent._memory))
    """
    if not _HAS_FASTAPI:
        import logging
        logging.getLogger("gazcc.viewer").warning(
            "FastAPI/uvicorn not installed — memory viewer disabled"
        )
        return

    app = _build_viewer_app(memory_backend)
    config = uvicorn.Config(app, host=host, port=port, log_level="warning", loop="none")
    server = uvicorn.Server(config)

    import logging
    logging.getLogger("gazcc.viewer").info(
        f"Memory viewer → http://{host}:{port}"
    )
    await server.serve()
