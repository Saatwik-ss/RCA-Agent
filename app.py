"""
RCA Agent v2 - Enhanced Backend
Uses: Groq API (LLM), Supabase (PostgreSQL), Vector DB (knowledge graph)
Features: Entity extraction, knowledge retrieval, interactive reasoning with clarifications
"""

import os
import json
import time
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, g
from flask_cors import CORS
import requests
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

# ─── Config ───────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
DATABASE_URL = os.getenv("DATABASE_URL", "")
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is not set!")

GUEST_USER_ID = 1

# ─── Database Setup ───────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="./frontend", static_url_path="")
CORS(app)

def get_db():
    if "db" not in g:
        g.db = psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
        g.db.autocommit = False
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    db = g.pop("db", None)
    if db and not db.closed:
        db.close()

def init_db():
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    # Main queries table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS queries (
            id              SERIAL PRIMARY KEY,
            user_id         INTEGER NOT NULL DEFAULT 1,
            session_id      TEXT UNIQUE NOT NULL,
            method          TEXT NOT NULL,
            problem_text    TEXT NOT NULL,
            log_input       TEXT,
            entities        JSONB,
            retrieved_knowledge JSONB,
            result          JSONB,
            confidence      REAL,
            created_at      TIMESTAMP DEFAULT NOW()
        );
    """)
    
    # Clarification/interaction log
    cur.execute("""
        CREATE TABLE IF NOT EXISTS clarifications (
            id              SERIAL PRIMARY KEY,
            query_id        INTEGER REFERENCES queries(id) ON DELETE CASCADE,
            step_number     INTEGER,
            question        TEXT NOT NULL,
            user_response   TEXT,
            llm_reasoning   JSONB,
            created_at      TIMESTAMP DEFAULT NOW()
        );
    """)
    
    # Mock knowledge graph entries
    cur.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_base (
            id              SERIAL PRIMARY KEY,
            entity_type     TEXT,
            entity_name     TEXT,
            description     TEXT,
            related_causes  JSONB,
            fixes           JSONB,
            created_at      TIMESTAMP DEFAULT NOW()
        );
    """)
    
    conn.commit()
    cur.close()
    conn.close()
    print("✓ Database initialised with new schema")

# ─── LLM Calls ────────────────────────────────────────────────────────────────

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
            {"role": "user", "content": user_prompt}
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

SYS_STRICT_JSON = (
    "You are a rigorous root-cause analysis engine. "
    "You ONLY output valid JSON matching the schema given. "
    "No markdown, no extra keys, no explanation outside the JSON."
)

# ─── Entity Extraction ─────────────────────────────────────────────────────────

def extract_entities(log_text: str, problem_text: str) -> dict:
    """
    Extract structured entities from logs and problem description.
    Common entities: frameworks, services, error types, stack, etc.
    """
    result = call_groq(
        SYS_STRICT_JSON,
        f"""Analyze these logs and problem, extract key entities:

Problem: {problem_text}

Logs:
{log_text}

Output JSON (extract all that apply):
{{
  "frameworks": ["Node.js", "Python", "Java", ...],
  "services": ["Database", "API", "Queue", "Cache", ...],
  "platforms": ["Docker", "Kubernetes", "Render", "AWS", ...],
  "error_types": ["Network", "Timeout", "Permission", "Configuration", ...],
  "files_mentioned": ["package.json", "config.yaml", ...],
  "key_timestamps": ["when errors occur"],
  "error_messages": ["exact error strings from logs"]
}}""",
        max_tokens=600
    )
    return result

# ─── Knowledge Graph / Vector DB Query ────────────────────────────────────────

def mock_knowledge_base() -> dict:
    """
    Mock knowledge base. In production, query Pinecone/Weaviate/Milvus.
    Returns related causes, documentation, historical incidents, dependencies.
    """
    return {
        "Node.js": {
            "common_causes": [
                "Missing or incorrect 'start' script in package.json",
                "Dependency version mismatch",
                "Memory leak or insufficient heap size",
                "Port already in use",
                "Environment variables not loaded"
            ],
            "fixes": [
                "Verify package.json has 'start' script: npm start",
                "Run npm install to update dependencies",
                "Increase Node memory: node --max-old-space-size=4096 app.js",
                "Check if port is occupied: lsof -i :3000",
                "Ensure .env file is properly configured"
            ],
            "related_dependencies": ["express", "dotenv", "cors"]
        },
        "Render": {
            "common_causes": [
                "Build script failed during deployment",
                "Environment variables not set in dashboard",
                "Free tier memory/timeout limits exceeded",
                "Log streaming not configured",
                "Health check endpoint missing"
            ],
            "fixes": [
                "Check build command in render.yaml",
                "Set all env vars in Render dashboard",
                "Add health check endpoint /health",
                "Increase plan for higher limits",
                "Review Render logs in dashboard"
            ]
        },
        "Database": {
            "common_causes": [
                "Connection string malformed",
                "Database server down or unreachable",
                "Authentication credentials wrong",
                "Query timeout",
                "Connection pool exhausted"
            ],
            "fixes": [
                "Verify connection string format",
                "Check database server status",
                "Test credentials independently",
                "Increase query timeout threshold",
                "Increase connection pool size"
            ]
        }
    }

def retrieve_knowledge(entities: dict) -> dict:
    """
    Query knowledge base for relevant information based on extracted entities.
    """
    kb = mock_knowledge_base()
    retrieved = {}
    
    for entity in entities.get("frameworks", []) + entities.get("services", []) + entities.get("platforms", []):
        if entity in kb:
            retrieved[entity] = kb[entity]
    
    return {
        "retrieved_entities": list(retrieved.keys()),
        "knowledge": retrieved,
        "total_matches": len(retrieved)
    }

# ─── Interactive Reasoning with Clarifications ────────────────────────────────

def ask_clarification(session_id: str, step: int, question: str) -> str:
    """
    Store a clarification question that needs user input.
    Returns a placeholder that the frontend will fill.
    """
    db = get_db()
    cur = db.cursor()
    
    qid = None
    cur.execute(
        """SELECT id FROM queries WHERE session_id = %s""",
        (session_id,)
    )
    row = cur.fetchone()
    if row:
        qid = row['id']
    
    if qid:
        cur.execute(
            """INSERT INTO clarifications (query_id, step_number, question)
               VALUES (%s, %s, %s) RETURNING id""",
            (qid, step, question)
        )
        cid = cur.fetchone()['id']
        db.commit()
        cur.close()
        return f"CLARIFY_{cid}"
    
    cur.close()
    return ""

def reasoning_node_with_clarifications(state: dict, step: int, session_id: str) -> dict:
    """
    Enhanced reasoning that can ask for clarifications.
    Returns both reasoning and potential clarification questions.
    """
    prior_cause = state.get("chain", [])[-1]["cause"] if state.get("chain") else state["problem_text"]
    knowledge_context = json.dumps(state.get("retrieved_knowledge", {}).get("knowledge", {}))
    
    clarification_policy = (
        "Do not ask the user any questions. Where evidence is incomplete, make the most "
        "probable assumption, identify it in your reasoning, and set is_ambiguous to false."
        if state.get("skip_clarifications") else
        "Ask ONE focused question about the current cause when more information is needed."
    )
    result = call_groq(
        SYS_STRICT_JSON,
        f"""You are on step {step} of root cause analysis.

Problem: {state['problem_text']}
Current cause: {prior_cause}
Retrieved knowledge: {knowledge_context}

Your tasks:
1. {clarification_policy}
2. Identify if you need user clarification (is_ambiguous)
3. Identify the upstream cause based on available evidence

Output JSON:
{{
  "why": "the question you are asking about this cause",
  "is_ambiguous": true/false (set true if you need user clarification),
  "clarification_needed": "if is_ambiguous=true, what specifically needs clarification?",
  "cause": "the upstream cause you identified or 'AWAITING_CLARIFICATION'",
  "is_root": true/false (only if not ambiguous),
  "reasoning": "one sentence explaining your reasoning"
}}""",
        max_tokens=600
    )
    
    state["chain"].append(result)
    
    # If LLM needs clarification, pause and wait for user input
    if result.get("is_ambiguous") and not state.get("skip_clarifications"):
        state["awaiting_clarification"] = True
        state["clarification_step"] = step
        state["clarification_needed"] = result.get("clarification_needed")
    elif result.get("is_root"):
        state["stop"] = True
    
    return state

def process_clarification_response(session_id: str, clarification_id: int, user_response: str,
                                   state: dict, skipped: bool = False) -> dict:
    """
    Process user's clarification response and continue reasoning.
    """
    db = get_db()
    cur = db.cursor()
    
    # Store either the user's answer or their choice to proceed on available evidence.
    recorded_response = user_response if not skipped else "Skipped: proceed with the most probable evidence-based assumption."
    cur.execute(
        """UPDATE clarifications SET user_response = %s WHERE id = %s""",
        (recorded_response, clarification_id)
    )
    db.commit()
    cur.close()
    
    # Update the last chain item with user's input
    if state.get("chain"):
        last_item = state["chain"][-1]
        last_item["user_clarification"] = recorded_response
        
        # Re-run reasoning with clarification
        state["awaiting_clarification"] = False
        
        # Ask LLM to re-evaluate with clarification
        prior_cause = last_item.get("cause", state["problem_text"])
        step = state.get("clarification_step", 1)
        
        clarification_instruction = (
            "The user chose not to provide more information. Do not ask another question. "
            "Proceed using the most probable explanation supported by the problem and prior reasoning. "
            "State the key assumption explicitly in the reasoning."
            if skipped else
            f"User's clarification: {user_response}"
        )
        result = call_groq(
            SYS_STRICT_JSON,
            f"""Determine the next most likely cause in this root-cause analysis.

Previous question: {last_item.get('why')}
{clarification_instruction}

Output JSON:
{{
  "cause": "the upstream cause now that we have clarification",
  "is_root": true/false,
  "reasoning": "one sentence"
}}""",
            max_tokens=400
        )
        
        last_item.update(result)
        if skipped:
            last_item["assumption_used"] = True
        if result.get("is_root"):
            state["stop"] = True
    
    return state

# ─── RCA Orchestrators ────────────────────────────────────────────────────────

def run_enhanced_rca(problem_text: str, log_input: str = "", session_id: str = "") -> dict:
    """
    New enhanced RCA workflow:
    1. Parse logs and extract entities
    2. Retrieve relevant knowledge
    3. Run interactive reasoning with potential clarifications
    4. Synthesize and generate actions
    """
    if not session_id:
        session_id = str(uuid.uuid4())
    
    # Step 1: Entity extraction
    entities = extract_entities(log_input, problem_text)
    
    # Step 2: Knowledge retrieval
    knowledge = retrieve_knowledge(entities)
    
    # Step 3: Initialize reasoning state
    state = {
        "problem_text": problem_text,
        "entities": entities,
        "retrieved_knowledge": knowledge,
        "chain": [],
        "analysis_method": "five_whys",
        "stop": False,
        "awaiting_clarification": False,
        "clarification_step": None,
        "skip_clarifications": False
    }
    
    # Step 4: Run reasoning loop (5 steps max)
    for i in range(1, 6):
        if state["stop"]:
            break
        
        state = reasoning_node_with_clarifications(state, i, session_id)
        
        # If awaiting clarification, return state and pause
        if state.get("awaiting_clarification"):
            state["session_id"] = session_id
            state["current_step"] = i
            return state
    
    # Step 5: Synthesis (if not awaiting clarification)
    if not state.get("awaiting_clarification"):
        state = synthesis_node(state, "enhanced")
        state = action_node(state)
        state["session_id"] = session_id
    
    return state

def run_fishbone_rca(problem_text: str, log_input: str = "", session_id: str = "") -> dict:
    """Run a Fishbone (Ishikawa) analysis across the six common cause categories."""
    if not session_id:
        session_id = str(uuid.uuid4())

    entities = extract_entities(log_input, problem_text)
    knowledge = retrieve_knowledge(entities)
    knowledge_context = json.dumps(knowledge.get("knowledge", {}))
    fishbone = call_groq(
        SYS_STRICT_JSON,
        f"""Perform a Fishbone (Ishikawa) root-cause analysis.

Problem: {problem_text}
Logs: {log_input}
Retrieved knowledge: {knowledge_context}

Consider each category: People, Process, Technology, Materials, Measurement, Environment.
Only include causes plausibly related to the supplied evidence. Do not invent facts.

Output JSON:
{{
  "fishbone": [
    {{"category": "People", "potential_causes": ["cause"], "evidence": "evidence or 'No direct evidence'"}},
    {{"category": "Process", "potential_causes": ["cause"], "evidence": "..."}},
    {{"category": "Technology", "potential_causes": ["cause"], "evidence": "..."}},
    {{"category": "Materials", "potential_causes": ["cause"], "evidence": "..."}},
    {{"category": "Measurement", "potential_causes": ["cause"], "evidence": "..."}},
    {{"category": "Environment", "potential_causes": ["cause"], "evidence": "..."}}
  ],
  "most_likely_causes": ["ranked cause"]
}}""",
        max_tokens=1000
    )
    state = {
        "problem_text": problem_text,
        "entities": entities,
        "retrieved_knowledge": knowledge,
        "fishbone": fishbone.get("fishbone", []),
        "chain": [{"cause": cause, "reasoning": "Ranked as a likely Fishbone cause."}
                  for cause in fishbone.get("most_likely_causes", [])],
        "analysis_method": "fishbone",
        "stop": True,
        "awaiting_clarification": False,
        "session_id": session_id
    }
    state = synthesis_node(state, "fishbone")
    return action_node(state)

def synthesis_node(state: dict, method: str) -> dict:
    """Synthesize final root cause from reasoning chain."""
    chain_summary = json.dumps(state.get("chain", []))
    
    result = call_groq(
        SYS_STRICT_JSON,
        f"""Synthesise the final root cause from this analysis chain.

Problem: {state['problem_text']}
Reasoning chain: {chain_summary}

Output JSON:
{{
  "root_cause": "clear one-to-two sentence root cause statement",
  "confidence": 0.0 to 1.0,
  "confidence_reason": "why you assigned this confidence score",
  "supporting_evidence": "brief summary of evidence from the chain"
}}""",
        max_tokens=600
    )
    
    state["root_cause"] = result["root_cause"]
    state["confidence"] = result["confidence"]
    state["confidence_reason"] = result["confidence_reason"]
    state["supporting_evidence"] = result.get("supporting_evidence", "")
    return state

def action_node(state: dict) -> dict:
    """Generate corrective actions."""
    result = call_groq(
        SYS_STRICT_JSON,
        f"""Generate 4 corrective actions for this root cause.

Problem: {state['problem_text']}
Root cause: {state.get('root_cause', 'Not determined')}

Output JSON:
{{
  "actions": [
    {{"action": "what to do", "priority": "high|medium|low", "effort": "hours|days|weeks"}},
    {{"action": "...", ...}},
    {{"action": "...", ...}},
    {{"action": "...", ...}}
  ]
}}

Rules:
- First action must directly fix the root cause
- Include one preventive action
- Be specific and actionable""",
        max_tokens=600
    )
    
    state["actions"] = result.get("actions", [])
    return state

# ─── Routes: Analysis ─────────────────────────────────────────────────────────

@app.route("/api/analyse", methods=["POST"])
def analyse():
    data = request.get_json(silent=True) or {}
    problem = data.get("problem_text", "").strip()
    log_input = data.get("log_input", "").strip()
    method = data.get("analysis_method", "five_whys")
    
    if not problem and not log_input:
        return jsonify({"error": "Provide a problem description or logs"}), 400
    if method not in {"five_whys", "fishbone"}:
        return jsonify({"error": "Choose either Five Whys or Fishbone analysis"}), 400
    
    try:
        session_id = str(uuid.uuid4())
        result = (run_fishbone_rca(problem, log_input, session_id)
                  if method == "fishbone" else run_enhanced_rca(problem, log_input, session_id))
        
        # Store in database
        db = get_db()
        cur = db.cursor()
        cur.execute(
            """INSERT INTO queries
               (user_id, session_id, method, problem_text, log_input, 
                entities, retrieved_knowledge, result, confidence)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
               RETURNING id""",
            (GUEST_USER_ID, session_id, method, problem, log_input or None,
             json.dumps(result.get("entities")), 
             json.dumps(result.get("retrieved_knowledge")),
             json.dumps({k: v for k, v in result.items() 
                        if k not in ["entities", "retrieved_knowledge", "session_id"]}),
             result.get("confidence"))
        )
        query_id = cur.fetchone()['id']
        db.commit()
        cur.close()
        
        result["query_id"] = query_id
        result["session_id"] = session_id
        
        return jsonify({"ok": True, "result": result, "session_id": session_id})
    
    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

@app.route("/api/clarify", methods=["POST"])
def submit_clarification():
    """Submit user's clarification response and continue analysis."""
    data = request.get_json()
    session_id = data.get("session_id", "")
    clarification_id = data.get("clarification_id")
    user_response = data.get("response", "").strip()
    skipped = bool(data.get("skip_clarification"))
    
    if not session_id or not clarification_id or (not user_response and not skipped):
        return jsonify({"error": "Missing required fields"}), 400
    
    try:
        db = get_db()
        cur = db.cursor()
        
        # Get the query and its current state
        cur.execute(
            """SELECT id, result FROM queries WHERE session_id = %s""",
            (session_id,)
        )
        row = cur.fetchone()
        if not row:
            cur.close()
            return jsonify({"error": "Session not found"}), 404
        
        query_id = row['id']
        stored_state = row['result'] or {}
        state = stored_state if isinstance(stored_state, dict) else json.loads(stored_state)
        cur.close()
        
        # Process clarification
        state = process_clarification_response(session_id, clarification_id, user_response, state, skipped)
        if skipped:
            state["skip_clarifications"] = True
        
        # If root cause still not determined, continue reasoning
        if not state.get("stop") and not state.get("awaiting_clarification"):
            for i in range(state.get("clarification_step", 2), 6):
                if state["stop"]:
                    break
                state = reasoning_node_with_clarifications(state, i, session_id)
                if state.get("awaiting_clarification"):
                    break
        
        # If analysis complete
        if state.get("stop") or not state.get("awaiting_clarification"):
            if not state.get("root_cause"):
                state = synthesis_node(state, "enhanced")
                state = action_node(state)
        
        # Update database
        db = get_db()
        cur = db.cursor()
        cur.execute(
            """UPDATE queries SET result = %s, confidence = %s 
               WHERE session_id = %s""",
            (json.dumps({k: v for k, v in state.items() 
                        if k not in ["entities", "retrieved_knowledge"]}),
             state.get("confidence"),
             session_id)
        )
        db.commit()
        cur.close()
        
        return jsonify({
            "ok": True,
            "result": state,
            "session_id": session_id,
            "awaiting_clarification": state.get("awaiting_clarification", False)
        })
    
    except Exception as e:
        return jsonify({"error": f"Clarification failed: {str(e)}"}), 500

# ─── Routes: History ─────────────────────────────────────────────────────────

@app.route("/api/history", methods=["GET"])
def history():
    db = get_db()
    cur = db.cursor()
    cur.execute(
        """SELECT id, method, problem_text, confidence, created_at
           FROM queries WHERE user_id = %s
           ORDER BY created_at DESC LIMIT 50""",
        (GUEST_USER_ID,)
    )
    rows = cur.fetchall()
    cur.close()
    return jsonify([dict(r) for r in rows])

@app.route("/api/history/<int:qid>", methods=["GET"])
def get_query(qid):
    db = get_db()
    cur = db.cursor()
    cur.execute(
        "SELECT * FROM queries WHERE id = %s AND user_id = %s",
        (qid, GUEST_USER_ID)
    )
    row = cur.fetchone()
    cur.close()
    if not row:
        return jsonify({"error": "Not found"}), 404
    r = dict(row)
    if r.get('result'):
        r["result"] = json.loads(r["result"])
    if r.get('entities'):
        r["entities"] = json.loads(r["entities"])
    if r.get('retrieved_knowledge'):
        r["retrieved_knowledge"] = json.loads(r["retrieved_knowledge"])
    return jsonify(r)

@app.route("/api/history/<int:qid>", methods=["DELETE"])
def delete_query(qid):
    db = get_db()
    cur = db.cursor()
    cur.execute("DELETE FROM queries WHERE id = %s AND user_id = %s", (qid, GUEST_USER_ID))
    db.commit()
    cur.close()
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

init_db()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_ENV") == "development"
    print(f"✓ Enhanced RCA Agent running on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
