# GazccThinking Agent

Autonomous AI Agent — ReAct loop, multi-step planning, memory, self-correction.

## Architecture

```
gazcc-agent/
├── main.py              ← CLI runner (local)
├── config.yaml          ← All config
├── vercel.json          ← Vercel deploy config
├── requirements.txt
├── .env.example
├── agent/
│   ├── core.py          ← GazccAgent class + event loop
│   ├── planner.py       ← Task → Plan (LLM decomposition)
│   ├── executor.py      ← Step → Result (ReAct mini-loop)
│   ├── tools.py         ← File, web, code, math tools
│   └── memory.py        ← Vector search memory (file or Redis)
└── api/
    └── index.py         ← FastAPI app (Vercel entry point)
```

## ReAct Loop

```
Task
 └─► Planner (LLM) → Plan [Step1, Step2, Step3...]
       └─► For each ready Step:
             Thought → Action → Tool → Observation → (repeat)
             → Final Answer → next Step
       └─► Compile output → save to memory
```

---

## Local Setup

```bash
git clone <your-repo>
cd gazcc-agent

pip install -r requirements.txt

cp .env.example .env
# Edit .env: add OPENROUTER_API_KEY

# Run a task
python main.py "Research quantum computing, write a report, save to file"

# With output file
python main.py --output report.md "Research quantum computing and summarize key concepts"

# JSON event stream
python main.py --json-events "List files in current dir"
```

---

## Deploy to Vercel

### 1. Push to GitHub
```bash
git init
git add .
git commit -m "GazccThinking Agent"
git remote add origin https://github.com/youruser/gazcc-agent
git push -u origin main
```

### 2. Import on Vercel
- Go to https://vercel.com/new
- Import your GitHub repo
- Framework: **Other**
- Root Directory: `.`

### 3. Set Environment Variables
In Vercel Project → Settings → Environment Variables:

| Key | Value |
|-----|-------|
| `OPENROUTER_API_KEY` | `sk-or-v1-...` |
| `UPSTASH_REDIS_REST_URL` | `https://xxx.upstash.io` |
| `UPSTASH_REDIS_REST_TOKEN` | `AXxx...` |
| `ALLOW_CODE_EXEC` | `false` |

### 4. (Optional) Upstash Redis
- Create free account at https://console.upstash.com
- Create Redis database → copy REST URL and Token
- Add to Vercel env vars above

### 5. Deploy
Vercel auto-deploys on every push to main.

---

## API Endpoints

### `POST /api/run` — Stream events (SSE)
```bash
curl -N -X POST https://your-app.vercel.app/api/run \
  -H "Content-Type: application/json" \
  -d '{"task": "Research quantum computing and write a summary"}'
```

Response: Server-Sent Events stream
```
data: {"type":"task_start","data":{"task":"..."},"timestamp":...}
data: {"type":"plan_ready","data":{"goal":"...","steps":[...]},...}
data: {"type":"step_start","data":{"step_id":1,"description":"..."},...}
data: {"type":"tool_call","data":{"tool":"web_search","input":{...}},...}
data: {"type":"observation","data":{"tool":"web_search","output":"..."},...}
data: {"type":"step_done","data":{"step_id":1,...},...}
data: {"type":"task_done","data":{"elapsed":42.1,"steps_done":5},...}
data: [DONE]
```

### `POST /api/task` — Submit async, poll later
```bash
curl -X POST https://your-app.vercel.app/api/task \
  -H "Content-Type: application/json" \
  -d '{"task": "Search for Python async best practices"}'
# → {"task_id": "abc123", "status": "running", "poll_url": "/api/task/abc123"}

curl https://your-app.vercel.app/api/task/abc123
# → {"status": "done", "output": "...", "events": [...]}
```

### `GET /api/tools` — List tools
```bash
curl https://your-app.vercel.app/api/tools
```

### `GET /api/memory?q=quantum` — Search memory
```bash
curl "https://your-app.vercel.app/api/memory?q=quantum+computing"
```

### `GET /ui` — Browser UI
Open `https://your-app.vercel.app/ui` in browser.

---

## Tools Available

| Tool | Description |
|------|-------------|
| `web_search` | DuckDuckGo search |
| `fetch_url` | Fetch + parse any URL |
| `read_file` | Read file content |
| `write_file` | Write/create file |
| `append_file` | Append to file |
| `list_dir` | List directory |
| `delete_file` | Delete file |
| `execute_code` | Run Python (local only) |
| `calculate` | Safe math eval |
| `get_time` | Current UTC time |
| `summarize_text` | Truncate long text |

---

## Example Tasks

```bash
# Research + write
python main.py "Research the latest developments in quantum computing, write a comprehensive report, and save it to quantum_report.md"

# Code analysis
python main.py "Read main.py, analyze its structure, and write a summary to analysis.md"

# Web + file
python main.py "Search for Python async best practices, fetch the top result, and save key points to async_notes.md"

# Math + report
python main.py "Calculate compound interest for 10000 at 5% over 20 years and write the results to investment.md"
```

---

## Vercel Limitations

| Limit | Value |
|-------|-------|
| Max execution time | 300s (Hobby) / 900s (Pro) |
| Memory | 1024 MB |
| File system | `/tmp` only (ephemeral) |
| Persistent storage | Upstash Redis |

For tasks > 5 minutes, run locally with `python main.py`.

---

## Config (config.yaml)

```yaml
llm:
  model: anthropic/claude-sonnet-4-6   # change model here
  max_tokens: 4000
  temperature: 0.1

agent:
  max_iterations: 20
  timeout: 900
  retry_limit: 3

memory:
  backend: redis   # or 'file' for local
  max_entries: 500
```
