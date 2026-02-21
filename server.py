import os
import json
import subprocess
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

# ============================================================
# Config
# ============================================================
GRAPH_FILE_PATH = "knowledge_graph.json"
STORE_FILE_PATH = "./store/"
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")

ALLOWED_ORIGINS = [
    "https://sage-bunny-f8e38a.netlify.app",
    "http://localhost:3000",
    "http://localhost:5173",
]

os.makedirs(STORE_FILE_PATH, exist_ok=True)

app = Flask(__name__)
CORS(
    app,
    resources={r"/*": {"origins": ALLOWED_ORIGINS}},
    supports_credentials=False,
    methods=["GET", "POST", "PATCH", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# ============================================================
# OpenAI client (lazy init, won't crash boot)
# ============================================================
def get_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)

# ============================================================
# Load Knowledge Graph
# ============================================================
def load_graph():
    with open(GRAPH_FILE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

knowledge_graph = load_graph()

# ============================================================
# Small Utilities
# ============================================================
def normalize(s: str) -> str:
    if s is None:
        return ""
    return str(s).replace("\r\n", "\n").replace("\r", "\n").strip()

def last_non_empty_line(stdout: str) -> str:
    lines = [normalize(ln) for ln in (stdout or "").splitlines()]
    lines = [ln for ln in lines if ln != ""]
    return lines[-1] if lines else ""

def allowed_fail_nodes_for_problem(problem_data: dict) -> list[str]:
    """
    Allowed node IDs are exactly the fail_node_id listed in test_cases
    that also exist in graph_nodes.
    """
    ids = []
    for tc in problem_data.get("test_cases", []):
        fid = tc.get("fail_node_id")
        if fid:
            ids.append(fid)

    # de-dup preserve order
    seen = set()
    out = []
    graph_nodes = knowledge_graph.get("graph_nodes", {})
    for x in ids:
        if x not in seen and x in graph_nodes:
            seen.add(x)
            out.append(x)
    return out

# ============================================================
# AI Classifier (label-only)
# ============================================================
def ai_classify_fail_node(problem_id: str, student_code: str, failures: list, problem_data: dict) -> str | None:
    """
    Uses LLM to pick ONE concept_id from allowed list.
    Returns None if no API key / error / invalid output.
    """
    client = get_openai_client()
    if not client:
        return None

    allowed = allowed_fail_nodes_for_problem(problem_data)
    if not allowed:
        return None

    # Compact failures (keep tokens low)
    compact_failures = []
    for f in failures:
        compact_failures.append({
            "case_id": f.get("case_id"),
            "input": f.get("input"),
            "expected": f.get("expected"),
            "actual": f.get("actual"),
            "stderr": (f.get("stderr") or "")[:250],
        })

    payload = {
        "problem_id": problem_id,
        "allowed_concept_ids": allowed,
        "student_code": (student_code or "")[:6000],
        "failures": compact_failures,
    }

    system_prompt = (
        "You are a strict classifier for a programming tutor.\n"
        "Pick exactly ONE concept_id from allowed_concept_ids.\n"
        "Return ONLY JSON: {\"concept_id\":\"...\",\"confidence\":0.0}\n"
        "No markdown, no extra keys, do not invent new IDs."
    )

    try:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            temperature=0,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload)},
            ],
        )

        text = (resp.output_text or "").strip()
        obj = json.loads(text)
        cid = obj.get("concept_id")

        if cid in allowed:
            return cid

        return None

    except Exception as e:
        print("AI classify error:", e)
        return None

# ============================================================
# Routes
# ============================================================
@app.route("/")
def health():
    return "Knowledge Graph Server is Online!"

@app.route("/submit-code", methods=["POST", "OPTIONS"])
def submit_code():
    if request.method == "OPTIONS":
        return ("", 204)

    data = request.get_json(force=True) or {}
    problem_id = data.get("problem_id")
    code = data.get("code", "")

    if not problem_id or problem_id not in knowledge_graph.get("problems", {}):
        return jsonify({"status": "error", "message": "Invalid or missing problem_id"}), 400
    if not isinstance(code, str) or code.strip() == "":
        return jsonify({"status": "error", "message": "Missing code"}), 400

    problem_data = knowledge_graph["problems"][problem_id]
    test_cases = problem_data.get("test_cases", [])
    if not test_cases:
        return jsonify({"status": "error", "message": f"No test cases configured for {problem_id}"}), 500

    # Save submission
    safe_name = "".join(c for c in problem_id if c.isalnum() or c in ("_", "-"))
    file_path = os.path.join(STORE_FILE_PATH, f"{safe_name}_submission.py")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(code)

    failures = []
    outputs_log = []

    for case in test_cases:
        input_val = str(case.get("input", ""))
        expected = normalize(case.get("expected_output", ""))

        try:
            result = subprocess.run(
                ["python3", file_path, input_val],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except subprocess.TimeoutExpired:
            return jsonify({
                "status": "failed",
                "hint": "Your code timed out. Do you have an infinite loop?",
                "concept_gap": "Complexity",
            })

        # Runtime crash => generic runtime hint
        if result.returncode != 0:
            return jsonify({
                "status": "failed",
                "hint": "Your code crashed. Check for invalid indexing / None access / syntax issues.",
                "concept_gap": "Runtime Error",
                "stderr": (result.stderr or "").strip(),
            })

        actual = last_non_empty_line(result.stdout or "")
        outputs_log.append(f"In: {input_val} | Out: {actual}")

        if actual != expected:
            failures.append({
                "case_id": case.get("id"),
                "fail_node_id": case.get("fail_node_id"),
                "input": input_val,
                "expected": expected,
                "actual": actual,
                "stderr": (result.stderr or "").strip(),
            })

    if not failures:
        return jsonify({
            "status": "success",
            "message": "All test cases passed!",
            "stdout": "\n".join(outputs_log),
        })

    # 1) AI picks best concept node from allowed list
    chosen_node_id = ai_classify_fail_node(problem_id, code, failures, problem_data)

    # 2) fallback: first failed testcase node
    if not chosen_node_id:
        chosen_node_id = failures[0].get("fail_node_id")

    node_data = knowledge_graph.get("graph_nodes", {}).get(chosen_node_id, {}) if chosen_node_id else {}

    # Show the testcase matching the chosen node, if present
    shown = next((f for f in failures if f.get("fail_node_id") == chosen_node_id), failures[0])

    return jsonify({
        "status": "failed",
        "hint": node_data.get("hint_text", "Check your logic."),
        "failed_test_case": shown.get("input", ""),
        "student_output": shown.get("actual", ""),
        "expected_output": shown.get("expected", ""),
        "concept_gap": node_data.get("related_concept", "General"),
        # keep during dev; remove later
        "debug_chosen_node_id": chosen_node_id,
        "debug_failures": failures,
    })

@app.route("/request-hint", methods=["POST", "OPTIONS"])
def request_hint():
    """
    For interactive visual puzzles / MCQ puzzles:
    Client sends:
      {
        "problem_id": "...",
        "remaining_concept_ids": ["nodeA","nodeB",...]
      }

    We return the first concept_id that exists in graph_nodes.
    """
    if request.method == "OPTIONS":
        return ("", 204)

    data = request.get_json(force=True) or {}
    problem_id = data.get("problem_id")
    ids = data.get("remaining_concept_ids", [])

    if not problem_id or problem_id not in knowledge_graph.get("problems", {}):
        return jsonify({"status": "error", "message": f"Invalid problem_id: {problem_id}"}), 400
    if not isinstance(ids, list) or not ids:
        return jsonify({"status": "error", "message": "No concept/error ids provided"}), 400

    chosen_id = None
    graph_nodes = knowledge_graph.get("graph_nodes", {})
    for cid in ids:
        if cid in graph_nodes:
            chosen_id = cid
            break

    if not chosen_id:
        return jsonify({"status": "error", "message": "No matching graph_nodes for given concept ids"}), 400

    node = graph_nodes[chosen_id]
    return jsonify({
        "status": "success",
        "concept_id": chosen_id,
        "hint": node.get("hint_text", "Think about this part again."),
        "related_concept": node.get("related_concept", "General"),
    })

if __name__ == "__main__":
    app.run(debug=True)