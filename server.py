import os
import json
import csv
import io
import glob
import subprocess
from flask import Flask, request, jsonify, Response
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

# If you want to control logging without editing code, set on Railway:
# AI_LOG=1
AI_LOG = os.environ.get("AI_LOG", "1") in ("1", "true", "True", "yes", "YES")

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
# OpenAI client (lazy init so server doesn't crash at boot)
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
# def ai_classify_fail_node(problem_id: str, student_code: str, failures: list, problem_data: dict) -> tuple[str | None, float | None]:
#     """
#     Uses LLM to pick ONE concept_id from allowed list.
#     Returns (concept_id, confidence). If unavailable/invalid -> (None, None)
#     """
#     client = get_openai_client()
#     if not client:
#         return None, None

#     allowed = allowed_fail_nodes_for_problem(problem_data)
#     if not allowed:
#         return None, None

#     compact_failures = []
#     for f in failures:
#         compact_failures.append({
#             "case_id": f.get("case_id"),
#             "input": f.get("input"),
#             "expected": f.get("expected"),
#             "actual": f.get("actual"),
#             "stderr": (f.get("stderr") or "")[:250],
#         })

#     payload = {
#         "problem_id": problem_id,
#         "allowed_concept_ids": allowed,
#         "student_code": (student_code or "")[:6000],
#         "failures": compact_failures,
#     }

#     system_prompt = (
#         "You are a strict classifier for a programming tutor.\n"
#         "Pick exactly ONE concept_id from allowed_concept_ids.\n"
#         "Return ONLY JSON: {\"concept_id\":\"...\",\"confidence\":0.0}\n"
#         "No markdown, no extra keys, do not invent new IDs."
#     )

#     try:
#         resp = client.responses.create(
#             model=OPENAI_MODEL,
#             temperature=0,
#             input=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": json.dumps(payload)},
#             ],
#         )

#         text = (resp.output_text or "").strip()
#         obj = json.loads(text)

#         cid = obj.get("concept_id")
#         conf = obj.get("confidence", None)

#         try:
#             conf = float(conf) if conf is not None else None
#         except Exception:
#             conf = None

#         if cid in allowed:
#             if AI_LOG:
#                 print(f"[AI] problem={problem_id} picked={cid} confidence={conf}")
#             return cid, conf

#         if AI_LOG:
#             print(f"[AI] INVALID concept_id returned: {cid} (allowed count={len(allowed)})")
#         return None, conf

#     except Exception as e:
#         print("[AI] classify error:", e)
#         return None, None

def ai_pick_node_and_microhint(
    problem_id: str,
    student_code: str,
    failures: list,
    problem_data: dict,
) -> tuple[str | None, float | None, str | None]:
    """
    Uses LLM to:
    1) Pick ONE concept_id from allowed list OR "__generic__"
    2) Generate ONE short micro-hint sentence (<=25 words)

    Returns:
        (concept_id_or_None, confidence, micro_hint_or_None)
    """

    client = get_openai_client()
    if not client:
        return None, None, None

    allowed = allowed_fail_nodes_for_problem(problem_data)
    if not allowed:
        return None, None, None

    # Add generic escape option
    allowed_with_generic = allowed + ["__generic__"]

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
        "allowed_concept_ids": allowed_with_generic,
        "student_code": (student_code or "")[:6000],
        "failures": compact_failures,
    }

    system_prompt = (
        "You are a strict programming tutor assistant.\n\n"
        "Tasks:\n"
        "1) Pick exactly ONE concept_id from allowed_concept_ids.\n"
        "   If none are a good fit, choose \"__generic__\".\n"
        "2) Write ONE short micro_hint sentence (max 25 words).\n\n"
        "Rules:\n"
        "- The micro_hint must reference the observed failure (expected vs actual, crash, timeout, etc).\n"
        "- Suggest a likely cause area (e.g., pointer update, base case, loop condition, head update).\n"
        "- Do NOT mention concept_id names.\n"
        "- Do NOT be overly confident.\n"
        "- If confidence is below 0.55, choose \"__generic__\".\n\n"
        "Return ONLY JSON in this format:\n"
        "{\"concept_id\":\"...\",\"confidence\":0.0,\"micro_hint\":\"...\"}\n"
        "No markdown. No extra keys."
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
        conf = obj.get("confidence", None)
        micro_hint = obj.get("micro_hint")

        try:
            conf = float(conf) if conf is not None else None
        except Exception:
            conf = None

        # Validate concept_id
        if cid not in allowed_with_generic:
            if AI_LOG:
                print(f"[AI] INVALID concept_id returned: {cid}")
            return None, conf, micro_hint

        if AI_LOG:
            print(f"[AI] picked={cid} confidence={conf}")

        # If generic or low confidence → treat as no node
        if cid == "__generic__" or (conf is not None and conf < 0.55):
            return None, conf, micro_hint

        return cid, conf, micro_hint

    except Exception as e:
        print("[AI] pick_node_and_microhint error:", e)
        return None, None, None
# ============================================================
# Health
# ============================================================
@app.route("/")
def health():
    return "Knowledge Graph Server is Online!"

# ============================================================
# /submit-code  (coding puzzles)
# ============================================================
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
    # chosen_node_id, confidence = ai_classify_fail_node(problem_id, code, failures, problem_data)
    chosen_node_id, confidence, micro_hint = ai_pick_node_and_microhint(problem_id, code, failures, problem_data)

    # # 2) fallback: first failed testcase node
    # if not chosen_node_id:
    #     chosen_node_id = failures[0].get("fail_node_id")

    # node_data = knowledge_graph.get("graph_nodes", {}).get(chosen_node_id, {}) if chosen_node_id else {}

    # # Show testcase matching the chosen node (less confusing)
    # shown = next((f for f in failures if f.get("fail_node_id") == chosen_node_id), failures[0])

    # return jsonify({
    #     "status": "failed",
    #     "hint": node_data.get("hint_text", "Check your logic."),
    #     "failed_test_case": shown.get("input", ""),
    #     "student_output": shown.get("actual", ""),
    #     "expected_output": shown.get("expected", ""),
    #     "concept_gap": node_data.get("related_concept", "General"),
    #     # Keep during dev; remove later
    #     "debug_chosen_node_id": chosen_node_id,
    #     "debug_ai_confidence": confidence,
    #     "debug_failures": failures,
    # })
    # 2) Always pick a node to show as fallback (for concept_gap + safe tip)
    fallback_node_id = chosen_node_id or failures[0].get("fail_node_id")

    node_data = knowledge_graph.get("graph_nodes", {}).get(fallback_node_id, {}) if fallback_node_id else {}

    # Show testcase matching the node we decided to show (less confusing)
    shown = next((f for f in failures if f.get("fail_node_id") == fallback_node_id), failures[0])

    # Compose final hint: AI micro-hint + KG hint_text
    kg_hint = node_data.get("hint_text")
    if micro_hint and kg_hint:
        final_hint = micro_hint.strip() + "\n\nTip: " + kg_hint
    elif micro_hint:
        final_hint = micro_hint.strip()
    else:
        final_hint = kg_hint or "Check your logic."

    return jsonify({
        "status": "failed",
        "hint": final_hint,
        "failed_test_case": shown.get("input", ""),
        "student_output": shown.get("actual", ""),
        "expected_output": shown.get("expected", ""),
        "concept_gap": node_data.get("related_concept", "General"),
        # Keep during dev; remove later
        "debug_chosen_node_id": fallback_node_id,
        "debug_ai_confidence": confidence,
        "debug_ai_micro_hint": micro_hint,
        "debug_failures": failures,
    })
# ============================================================
# /request-hint  (visual/interactive puzzles)
# ============================================================
@app.route("/request-hint", methods=["POST", "OPTIONS"])
def request_hint():
    """
    Client sends:
      {
        "problem_id": "...",
        "remaining_concept_ids": ["nodeA","nodeB",...]
      }

    Deterministic: return first concept_id that exists in graph_nodes.
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

    graph_nodes = knowledge_graph.get("graph_nodes", {})
    chosen_id = next((cid for cid in ids if cid in graph_nodes), None)

    if not chosen_id:
        return jsonify({"status": "error", "message": "No matching graph_nodes for given concept ids"}), 400

    node = graph_nodes[chosen_id]
    return jsonify({
        "status": "success",
        "concept_id": chosen_id,
        "hint": node.get("hint_text", "Think about this part again."),
        "related_concept": node.get("related_concept", "General"),
    })

# ============================================================
# /api/login  (simple test login)
# ============================================================
@app.route("/api/login", methods=["POST", "OPTIONS"])
def api_login():
    if request.method == "OPTIONS":
        return ("", 204)

    data = request.get_json(force=True) or {}
    username = data.get("username", "")
    password = data.get("password", "")

    is_test_account = isinstance(username, str) and username.startswith("testAccount")
    is_test_account = is_test_account and username[len("testAccount"):].isdigit()
    if is_test_account and password == "password@123":
        return jsonify({"status": "success", "user": username})

    return jsonify({"status": "error", "message": "Invalid credentials"}), 401

# ============================================================
# Playerdata storage (GET/PATCH)
# ============================================================
@app.route("/playerdata/<int:player_id>", methods=["GET", "PATCH", "OPTIONS"])
def playerdata(player_id):
    if request.method == "OPTIONS":
        return ("", 204)

    path = os.path.join(STORE_FILE_PATH, f"playerdata_{player_id}.json")

    if request.method == "GET":
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                try:
                    return jsonify(json.load(f) or {})
                except Exception:
                    return jsonify({})
        return jsonify({})

    # PATCH
    data = request.get_json(silent=True) or {}

    store = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                store = json.load(f) or {}
            except Exception:
                store = {}

    # Unity sends: { "name": "...", "time": ..., "attempts": ..., "solved": ... }
    if "name" in data:
        entry = {
            "puzzle_name": data.get("name", ""),
            "time": data.get("time", 0),
            "attempts": data.get("attempts", 0),
            "solved": int(bool(data.get("solved", False))),
        }
        key = entry["puzzle_name"] or str(len(store))
        store[key] = entry
    elif isinstance(data, dict):
        # If client sends a dict of entries, merge them
        for k, v in data.items():
            if isinstance(v, dict):
                store[k] = v

    with open(path, "w", encoding="utf-8") as f:
        json.dump(store, f)

    return jsonify({"ok": True, "player_id": player_id})

# ============================================================
# CSV Export
# ============================================================
def _iter_playerdata_files():
    """Yield (player_id:int, filepath:str) for all playerdata_*.json files."""
    pattern = os.path.join(STORE_FILE_PATH, "playerdata_*.json")
    for path in sorted(glob.glob(pattern)):
        base = os.path.basename(path)  # e.g. playerdata_12.json
        try:
            player_id = int(base.replace("playerdata_", "").replace(".json", ""))
            yield player_id, path
        except ValueError:
            continue

@app.route("/playerdata/export.csv", methods=["GET", "OPTIONS"])
def export_playerdata_csv():
    if request.method == "OPTIONS":
        return ("", 204)

    player_id_filter = request.args.get("player_id")
    solved_only = request.args.get("solved_only") in ("1", "true", "True", "yes", "YES")

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["player_id", "puzzle_name", "time", "attempts", "solved"])

    def write_rows_for_player(pid: int, store: dict):
        for _, entry in (store or {}).items():
            if not isinstance(entry, dict):
                continue
            puzzle_name = entry.get("puzzle_name", "")
            t = entry.get("time", 0)
            attempts = entry.get("attempts", 0)
            solved = int(entry.get("solved", 0))

            if solved_only and solved != 1:
                continue

            writer.writerow([pid, puzzle_name, t, attempts, solved])

    # Export single player
    if player_id_filter:
        try:
            pid = int(player_id_filter)
        except ValueError:
            return jsonify({"ok": False, "message": "player_id must be an integer"}), 400

        path = os.path.join(STORE_FILE_PATH, f"playerdata_{pid}.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                try:
                    store = json.load(f) or {}
                except Exception:
                    store = {}
            write_rows_for_player(pid, store)

        csv_text = output.getvalue()
        return Response(
            csv_text,
            mimetype="text/csv",
            headers={"Content-Disposition": f'attachment; filename="playerdata_{pid}.csv"'}
        )

    # Export all players
    for pid, path in _iter_playerdata_files():
        with open(path, "r", encoding="utf-8") as f:
            try:
                store = json.load(f) or {}
            except Exception:
                store = {}
        write_rows_for_player(pid, store)

    csv_text = output.getvalue()
    return Response(
        csv_text,
        mimetype="text/csv",
        headers={"Content-Disposition": 'attachment; filename="playerdata_all.csv"'}
    )

if __name__ == "__main__":
    app.run(debug=True)
