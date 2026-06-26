# RCA Agent — Project Structure & Setup Guide

Everything you need to understand the codebase, set it up, and deploy it.

---

## 1. Folder structure

```
rca-agent/
│
├── backend/
│   ├── app.py               ← Flask server — all API routes and RCA graph nodes
│   ├── requirements.txt     ← Python dependencies
│   ├── .env.example         ← Copy to .env and fill in your keys (never commit .env)
│   ├── .env                 ← YOUR SECRETS — git-ignored
│   └── rca_agent.db         ← SQLite database (auto-created on first run)
│
├── frontend/
│   └── index.html           ← Entire frontend (auth, analysis UI, history) — one file
│
├── docs/
│   └── STRUCTURE.md         ← This file
│
├── .gitignore
├── README.md
└── render.yaml              ← One-click Render.com deploy config (optional)
```

---

## 2. API keys you need

### Groq API key (required)
1. Go to **https://console.groq.com**
2. Sign up (free)
3. Click **API Keys → Create API Key**
4. Copy the key — it starts with `gsk_`

> Groq gives you fast, free inference on Llama 3 70B, Mixtral, and Gemma. The free tier is generous enough to run this entire project.

### Where to put it
Create `backend/.env` (copy from `.env.example`):

```env
GROQ_API_KEY=gsk_your_actual_key_here
GROQ_MODEL=llama3-70b-8192
JWT_SECRET=any_long_random_string_here
DB_PATH=rca_agent.db
PORT=5000
FLASK_ENV=development
```

**Never commit `.env` to git.** Add it to `.gitignore` (already included below).

---

## 3. How the pieces connect

```
Browser (index.html)
    │
    │  HTTP requests (fetch API)
    │  Bearer token in Authorization header
    ▼
Flask backend (app.py)  ←→  SQLite (rca_agent.db)
    │
    │  HTTP POST to api.groq.com/openai/v1/chat/completions
    ▼
Groq API (Llama 3 70B)
```

The frontend is served by Flask as a static file. There is no separate Node/React server — Flask serves `index.html` for all routes and the JS fetches `/api/*` endpoints.

---

## 4. Database tables

### `users`
| Column     | Type    | Notes                        |
|------------|---------|------------------------------|
| id         | INTEGER | Auto-increment primary key   |
| email      | TEXT    | Unique, used for login       |
| password   | TEXT    | SHA-256 hashed (not plain)   |
| name       | TEXT    | Display name                 |
| created_at | TEXT    | ISO datetime                 |

### `queries`
| Column       | Type    | Notes                              |
|--------------|---------|------------------------------------|
| id           | INTEGER | Auto-increment primary key         |
| user_id      | INTEGER | Foreign key → users.id             |
| method       | TEXT    | "5whys" or "fishbone"              |
| problem_text | TEXT    | The problem the user described     |
| log_input    | TEXT    | Raw logs (nullable)                |
| result       | TEXT    | Full JSON result from RCA graph    |
| confidence   | REAL    | 0.0–1.0 float                      |
| created_at   | TEXT    | ISO datetime                       |

---

## 5. RCA graph — how the nodes work

### 5-Whys flow

```
intake_node  →  why_node(1)  →  why_node(2)  →  ...  →  why_node(5)
                                                           ↓
                                                     synthesis_node
                                                           ↓
                                                      action_node
```

Each `why_node` call:
- Receives the full state (problem + prior chain)
- Makes one Groq API call with a strict JSON-output prompt
- Appends `{why, cause, is_root, reasoning}` to `state["chain"]`
- Sets `state["stop"] = True` if `is_root` is true
- The loop exits early if stop is triggered

### Fishbone flow

```
intake_node  →  fishbone_node (6 branches in one call)  →  synthesis_node  →  action_node
```

The `fishbone_node` maps causes to People / Process / Tools / Environment / Materials / Measurement in a single Groq call, returning structured JSON.

### System prompt steering

Every node uses this as its system prompt:
```
"You are a rigorous root-cause analysis engine.
You ONLY output valid JSON matching the schema given.
No markdown, no extra keys, no explanation outside the JSON."
```

This is the LLM steering technique — the model's persona is locked to a strict analyst that never deviates from the schema.

---

## 6. API endpoints

| Method | Path                  | Auth | Description                        |
|--------|-----------------------|------|------------------------------------|
| POST   | /api/auth/register    | No   | Create account, returns JWT token  |
| POST   | /api/auth/login       | No   | Login, returns JWT token           |
| GET    | /api/auth/me          | Yes  | Get current user info              |
| POST   | /api/analyse          | Yes  | Run RCA analysis, saves to DB      |
| GET    | /api/history          | Yes  | List user's past 50 analyses       |
| GET    | /api/history/:id      | Yes  | Get one full analysis result       |
| DELETE | /api/history/:id      | Yes  | Delete one analysis                |
| GET    | /api/health           | No   | Health check                       |

All authenticated routes require: `Authorization: Bearer <token>`

### POST /api/analyse — request body

```json
{
  "method":       "5whys",
  "problem_text": "Deployment fails intermittently in production",
  "log_input":    "[2024-06-25 14:32:01] ERROR: pip install exit code 1",
  "use_logs":     true
}
```

### POST /api/analyse — response

```json
{
  "ok": true,
  "result": {
    "problem_text": "...",
    "chain": [
      { "why": "...", "cause": "...", "is_root": false, "reasoning": "..." },
      ...
    ],
    "root_cause": "...",
    "confidence": 0.88,
    "confidence_reason": "...",
    "actions": [
      { "action": "...", "priority": "high", "effort": "hours" },
      ...
    ]
  }
}
```

---

## 7. Local setup — step by step

```bash
# 1. Clone your repo
git clone https://github.com/yourname/rca-agent.git
cd rca-agent

# 2. Set up Python environment
cd backend
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your .env
cp .env.example .env
# Now open .env and add your GROQ_API_KEY and a random JWT_SECRET

# 5. Run the server
python app.py
# ✓ RCA Agent running on http://localhost:5000

# 6. Open the app
# Go to http://localhost:5000 in your browser
```

---

## 8. Deployment — Render.com (free tier)

1. Push your project to a GitHub repo
2. Go to **https://render.com → New → Web Service**
3. Connect your GitHub repo
4. Settings:
   - **Root directory**: `backend`
   - **Build command**: `pip install -r requirements.txt`
   - **Start command**: `gunicorn app:app`
   - **Environment**: Python 3
5. Add environment variables (from your `.env`):
   - `GROQ_API_KEY`
   - `JWT_SECRET`
   - `GROQ_MODEL`
   - `DB_PATH` = `/opt/render/project/src/rca_agent.db`
6. Click **Deploy**

> Note: Render's free tier spins down after 15 min of inactivity. The first request after sleep takes ~30s. Upgrade to Starter ($7/mo) for always-on.

---

## 9. `.gitignore`

Create this file at the root:

```
# Secrets
backend/.env

# Python
backend/venv/
backend/__pycache__/
backend/*.pyc
backend/rca_agent.db

# OS
.DS_Store
Thumbs.db
```

---

## 10. Groq model options

| Model                | Speed    | Quality  | Context  | Notes                    |
|----------------------|----------|----------|----------|--------------------------|
| llama3-70b-8192      | Fast     | ★★★★★   | 8K       | Best default             |
| llama3-8b-8192       | Fastest  | ★★★☆☆   | 8K       | Use if rate-limited      |
| mixtral-8x7b-32768   | Fast     | ★★★★☆   | 32K      | Good for long logs       |
| gemma2-9b-it         | Fast     | ★★★☆☆   | 8K       | Lightweight option       |

Set your choice in `.env` under `GROQ_MODEL`.

---

## 11. Troubleshooting

| Problem | Fix |
|---------|-----|
| `GROQ_API_KEY not set` | Make sure `.env` exists in `backend/` and has `GROQ_API_KEY=gsk_...` |
| `502 Groq API error` | Check your API key is valid at console.groq.com. Check rate limits. |
| `401 Invalid token` | JWT_SECRET changed. Log out and log back in. |
| Database errors | Delete `rca_agent.db` and restart — it recreates automatically. |
| CORS errors in browser | You're running frontend on a different port than backend. Either serve from Flask (correct) or add your frontend URL to `CORS(app, origins=["http://localhost:3000"])`. |

---

## 12. What each file does — quick reference

| File | What it does |
|------|-------------|
| `backend/app.py` | The entire backend: Flask routes, RCA graph nodes (log_parser_node, why_node, fishbone_node, synthesis_node, action_node), auth (JWT), database (SQLite) |
| `backend/requirements.txt` | Python packages: Flask, flask-cors, PyJWT, requests, gunicorn |
| `backend/.env` | Your secrets. Never commit this. |
| `backend/rca_agent.db` | SQLite database, auto-created. Contains users and queries tables. |
| `frontend/index.html` | The entire frontend: login/register screens, analysis form, result rendering, history page. Pure HTML/CSS/JS — no build step needed. |
