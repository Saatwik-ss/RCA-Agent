"""
RCA Agent - Flask Backend
Uses Groq API for LLM calls, SQLite for storage, JWT for auth.
"""

import os
import json
import sqlite3
import hashlib
import secrets
import time
from datetime import datetime, timedelta
from functools import wraps
from flask import Flask, request, jsonify, send_from_directory, g
from flask_cors import CORS
import jwt
import requests

app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app, supports_credentials=True)

# ─── Config ──────────────────────────────────────────────────────────────────
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL     = os.getenv("GROQ_MODEL", "llama3-70b-8192")
JWT_SECRET     = os.getenv("JWT_SECRET", secrets.token_hex(32))
DB_PATH        = os.getenv("DB_PATH", "rca_agent.db")
GROQ_URL       = "https://api.groq.com/openai/v1/chat/completions"

# ─── Database ─────────────────────────────────────────────────────────────────
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    db = g.pop("db", None)
    if db:
        db.close()

def init_db():
    db = sqlite3.connect(DB_PATH)
    db.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            email       TEXT    UNIQUE NOT NULL,
            password    TEXT    NOT NULL,
            name        TEXT    NOT NULL,
            created_at  TEXT    DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS queries (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      INTEGER NOT NULL,
            method       TEXT    NOT NULL,
            problem_text TEXT    NOT NULL,
            log_input    TEXT,
            result       TEXT    NOT NULL,
            confidence   REAL,
            created_at   TEXT    DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)
    db.commit()
    db.close()
    print("✓ Database initialised")

# ─── Auth helpers ──────────────────────────────────────────────────────────────
def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def make_token(user_id: int, email: str) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "exp": datetime.utcnow() + timedelta(days=7)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        if not token:
            return jsonify({"error": "No token provided"}), 401
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
            g.user_id = payload["sub"]
            g.email   = payload["email"]
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401
        return f(*args, **kwargs)
    return decorated

# ─── Groq LLM call ────────────────────────────────────────────────────────────
def call_groq(system_prompt: str, user_prompt: str, max_tokens: int = 800) -> dict:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3
    }
    resp = requests.post(GROQ_URL, headers=headers, json=body, timeout=30)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"].strip()

    # Strip markdown code fences if present
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
        content = content.rsplit("```", 1)[0].strip()

    return json.loads(content)

# ─── RCA Graph Nodes ──────────────────────────────────────────────────────────

SYS_STRICT_JSON = (
    "You are a rigorous root-cause analysis engine. "
    "You ONLY output valid JSON matching the schema given. "
    "No markdown, no extra keys, no explanation outside the JSON."
)

def log_parser_node(log_text: str) -> dict:
    """Parse raw log text into a structured problem statement."""
    return call_groq(
        SYS_STRICT_JSON,
        f"""Parse these deployment logs and extract the core problem.
Output JSON:
{{
  "problem_text": "one-sentence description of the failure",
  "error_type": "category of error (e.g. network, build, config)",
  "affected_service": "service or component that failed",
  "key_signals": ["list", "of", "key", "log", "signals"]
}}

LOGS:
{log_text}"""
    )

def why_node(state: dict, step: int) -> dict:
    """Single why-step graph node."""
    prior_cause = state["chain"][-1]["cause"] if state["chain"] else state["problem_text"]
    result = call_groq(
        SYS_STRICT_JSON,
        f"""You are on step {step} of a 5-Whys root cause analysis.

Problem being investigated: {state['problem_text']}
Current cause (from step {step-1}): {prior_cause}
Prior chain: {json.dumps(state['chain'])}

Ask ONE focused "why" about the current cause and identify the upstream cause.
Output JSON:
{{
  "why": "the question you are asking",
  "cause": "the upstream cause you identified",
  "is_root": true or false,
  "reasoning": "one sentence explaining your reasoning"
}}

Rules:
- is_root = true ONLY if this cause is a process/system/config failure (not another symptom)
- Never repeat a prior cause
- Be specific, not generic"""
    )
    state["chain"].append(result)
    if result.get("is_root"):
        state["stop"] = True
    return state

def fishbone_node(state: dict) -> dict:
    """Fishbone analysis across 6 standard branches."""
    result = call_groq(
        SYS_STRICT_JSON,
        f"""Perform a Fishbone (Ishikawa) root cause analysis on this problem:

Problem: {state['problem_text']}
Context: {state.get('context', 'None')}

Map causes to exactly these 6 branches. Output JSON:
{{
  "branches": {{
    "People":      "cause related to people, skills, training",
    "Process":     "cause related to process, workflow, procedure",
    "Tools":       "cause related to tools, software, equipment",
    "Environment": "cause related to environment, infrastructure, network",
    "Materials":   "cause related to inputs, dependencies, data",
    "Measurement": "cause related to monitoring, alerting, metrics"
  }},
  "dominant_branch": "which branch is the primary cause",
  "dominant_cause": "the specific dominant cause in one sentence"
}}""",
        max_tokens=1000
    )
    state["fishbone"] = result
    return state

def synthesis_node(state: dict, method: str) -> dict:
    """Synthesise root cause and confidence from the chain."""
    if method == "5whys":
        chain_summary = json.dumps(state["chain"])
    else:
        chain_summary = json.dumps(state.get("fishbone", {}))

    result = call_groq(
        SYS_STRICT_JSON,
        f"""Synthesise the final root cause from this analysis.

Problem: {state['problem_text']}
Analysis: {chain_summary}

Output JSON:
{{
  "root_cause": "clear one-to-two sentence root cause statement",
  "confidence": 0.0 to 1.0 (how confident you are this is the TRUE root, not a symptom),
  "confidence_reason": "why you assigned this confidence score"
}}"""
    )
    state["root_cause"]        = result["root_cause"]
    state["confidence"]        = result["confidence"]
    state["confidence_reason"] = result["confidence_reason"]
    return state

def action_node(state: dict) -> dict:
    """Generate corrective actions tied to the root cause."""
    result = call_groq(
        SYS_STRICT_JSON,
        f"""Generate corrective actions for this root cause.

Problem: {state['problem_text']}
Root cause: {state['root_cause']}

Output JSON:
{{
  "actions": [
    {{"action": "what to do", "priority": "high|medium|low", "effort": "hours|days|weeks"}},
    {{"action": "what to do", "priority": "high|medium|low", "effort": "hours|days|weeks"}},
    {{"action": "what to do", "priority": "high|medium|low", "effort": "hours|days|weeks"}},
    {{"action": "what to do", "priority": "high|medium|low", "effort": "hours|days|weeks"}}
  ]
}}

Rules:
- First action must directly fix the root cause
- Include one preventive action
- Be specific and actionable""",
        max_tokens=600
    )
    state["actions"] = result["actions"]
    return state

# ─── RCA Orchestrator ─────────────────────────────────────────────────────────

def run_5whys(problem_text: str, context: str = "") -> dict:
    state = {
        "problem_text": problem_text,
        "context": context,
        "chain": [],
        "stop": False
    }
    for i in range(1, 6):
        state = why_node(state, i)
        if state["stop"]:
            break
    state = synthesis_node(state, "5whys")
    state = action_node(state)
    return state

def run_fishbone(problem_text: str, context: str = "") -> dict:
    state = {
        "problem_text": problem_text,
        "context": context,
        "chain": []
    }
    state = fishbone_node(state)
    state = synthesis_node(state, "fishbone")
    state = action_node(state)
    return state

# ─── Routes: Auth ─────────────────────────────────────────────────────────────

@app.route("/api/auth/register", methods=["POST"])
def register():
    data = request.get_json()
    name     = data.get("name", "").strip()
    email    = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not all([name, email, password]):
        return jsonify({"error": "All fields required"}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400

    db = get_db()
    try:
        db.execute(
            "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
            (name, email, hash_password(password))
        )
        db.commit()
        user = db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        return jsonify({
            "token": make_token(user["id"], user["email"]),
            "user": {"id": user["id"], "name": user["name"], "email": user["email"]}
        })
    except sqlite3.IntegrityError:
        return jsonify({"error": "Email already registered"}), 409

@app.route("/api/auth/login", methods=["POST"])
def login():
    data     = request.get_json()
    email    = data.get("email", "").strip().lower()
    password = data.get("password", "")

    db   = get_db()
    user = db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()

    if not user or user["password"] != hash_password(password):
        return jsonify({"error": "Invalid email or password"}), 401

    return jsonify({
        "token": make_token(user["id"], user["email"]),
        "user": {"id": user["id"], "name": user["name"], "email": user["email"]}
    })

@app.route("/api/auth/me", methods=["GET"])
@require_auth
def me():
    db   = get_db()
    user = db.execute("SELECT id, name, email, created_at FROM users WHERE id = ?", (g.user_id,)).fetchone()
    return jsonify(dict(user))

# ─── Routes: RCA ──────────────────────────────────────────────────────────────

@app.route("/api/analyse", methods=["POST"])
@require_auth
def analyse():
    data        = request.get_json()
    method      = data.get("method", "5whys")          # "5whys" | "fishbone"
    problem     = data.get("problem_text", "").strip()
    log_input   = data.get("log_input", "").strip()
    use_logs    = data.get("use_logs", False)

    if not problem and not log_input:
        return jsonify({"error": "Provide a problem description or logs"}), 400

    try:
        context = ""

        # Log parser node — if logs provided, extract structured problem
        if use_logs and log_input:
            parsed = log_parser_node(log_input)
            if not problem:
                problem = parsed["problem_text"]
            context = json.dumps(parsed)

        # Run the graph
        if method == "fishbone":
            result = run_fishbone(problem, context)
        else:
            result = run_5whys(problem, context)

        # Save to DB
        db = get_db()
        db.execute(
            """INSERT INTO queries
               (user_id, method, problem_text, log_input, result, confidence)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (g.user_id, method, problem, log_input or None,
             json.dumps(result), result.get("confidence"))
        )
        db.commit()

        return jsonify({"ok": True, "result": result})

    except ValueError as e:
        return jsonify({"error": str(e)}), 500
    except requests.HTTPError as e:
        return jsonify({"error": f"Groq API error: {e.response.status_code}"}), 502
    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

# ─── Routes: History ──────────────────────────────────────────────────────────

@app.route("/api/history", methods=["GET"])
@require_auth
def history():
    db   = get_db()
    rows = db.execute(
        """SELECT id, method, problem_text, confidence, created_at
           FROM queries WHERE user_id = ?
           ORDER BY created_at DESC LIMIT 50""",
        (g.user_id,)
    ).fetchall()
    return jsonify([dict(r) for r in rows])

@app.route("/api/history/<int:qid>", methods=["GET"])
@require_auth
def get_query(qid):
    db  = get_db()
    row = db.execute(
        "SELECT * FROM queries WHERE id = ? AND user_id = ?",
        (qid, g.user_id)
    ).fetchone()
    if not row:
        return jsonify({"error": "Not found"}), 404
    r = dict(row)
    r["result"] = json.loads(r["result"])
    return jsonify(r)

@app.route("/api/history/<int:qid>", methods=["DELETE"])
@require_auth
def delete_query(qid):
    db = get_db()
    db.execute("DELETE FROM queries WHERE id = ? AND user_id = ?", (qid, g.user_id))
    db.commit()
    return jsonify({"ok": True})

# ─── Routes: Static ───────────────────────────────────────────────────────────

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path):
    if path and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "model": GROQ_MODEL})

# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_db()
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_ENV") == "development"
    print(f"✓ RCA Agent running on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
