"""
RCA Agent - Flask Backend (no auth)
Uses Groq API for LLM calls, SQLite for storage.
"""

import os
import json
import sqlite3
import time
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, g
from flask_cors import CORS
import requests

app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)
from dotenv import load_dotenv
load_dotenv()

# ─── Config ───────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama3-70b-8192")
DB_PATH      = os.getenv("DB_PATH", "rca_agent.db")
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"

# Fixed guest user — no login required
GUEST_USER_ID = 1

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
        CREATE TABLE IF NOT EXISTS queries (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      INTEGER NOT NULL DEFAULT 1,
            method       TEXT    NOT NULL,
            problem_text TEXT    NOT NULL,
            log_input    TEXT,
            result       TEXT    NOT NULL,
            confidence   REAL,
            created_at   TEXT    DEFAULT (datetime('now'))
        );
    """)
    db.commit()
    db.close()
    print("✓ Database initialised")

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

# ─── RCA Orchestrators ────────────────────────────────────────────────────────

def run_5whys(problem_text: str, context: str = "") -> dict:
    state = {"problem_text": problem_text, "context": context, "chain": [], "stop": False}
    for i in range(1, 6):
        state = why_node(state, i)
        if state["stop"]:
            break
    state = synthesis_node(state, "5whys")
    state = action_node(state)
    return state

def run_fishbone(problem_text: str, context: str = "") -> dict:
    state = {"problem_text": problem_text, "context": context, "chain": []}
    state = fishbone_node(state)
    state = synthesis_node(state, "fishbone")
    state = action_node(state)
    return state

# ─── Routes: RCA ──────────────────────────────────────────────────────────────

@app.route("/api/analyse", methods=["POST"])
def analyse():
    data      = request.get_json()
    method    = data.get("method", "5whys")
    problem   = data.get("problem_text", "").strip()
    log_input = data.get("log_input", "").strip()
    use_logs  = data.get("use_logs", False)

    if not problem and not log_input:
        return jsonify({"error": "Provide a problem description or logs"}), 400

    try:
        context = ""
        if use_logs and log_input:
            parsed = log_parser_node(log_input)
            if not problem:
                problem = parsed["problem_text"]
            context = json.dumps(parsed)

        if method == "fishbone":
            result = run_fishbone(problem, context)
        else:
            result = run_5whys(problem, context)

        db = get_db()
        db.execute(
            """INSERT INTO queries
               (user_id, method, problem_text, log_input, result, confidence)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (GUEST_USER_ID, method, problem, log_input or None,
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
def history():
    db   = get_db()
    rows = db.execute(
        """SELECT id, method, problem_text, confidence, created_at
           FROM queries WHERE user_id = ?
           ORDER BY created_at DESC LIMIT 50""",
        (GUEST_USER_ID,)
    ).fetchall()
    return jsonify([dict(r) for r in rows])

@app.route("/api/history/<int:qid>", methods=["GET"])
def get_query(qid):
    db  = get_db()
    row = db.execute(
        "SELECT * FROM queries WHERE id = ? AND user_id = ?",
        (qid, GUEST_USER_ID)
    ).fetchone()
    if not row:
        return jsonify({"error": "Not found"}), 404
    r = dict(row)
    r["result"] = json.loads(r["result"])
    return jsonify(r)

@app.route("/api/history/<int:qid>", methods=["DELETE"])
def delete_query(qid):
    db = get_db()
    db.execute("DELETE FROM queries WHERE id = ? AND user_id = ?", (qid, GUEST_USER_ID))
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
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_ENV") == "development"
    print(f"✓ RCA Agent running on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
