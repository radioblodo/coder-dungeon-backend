import os
import json
import csv
import io
import glob
import subprocess
from collections import Counter, defaultdict 
from typing import Dict, List, Set, Tuple 
from datetime import datetime, timezone, timedelta
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

def ai_explain_failure(
    chosen_node_id: str | None,
    concept_label: str,
    student_code: str,
    failure: dict,
) -> str | None:
    client = get_openai_client()
    if not client:
        return None

    payload = {
        "chosen_node_id": chosen_node_id,
        "concept_name": concept_label,
        "student_code": (student_code or "")[:6000],
        "test_case": {
            "case_id": failure.get("case_id"),
            "input": failure.get("input"),
            "expected": failure.get("expected"),
            "actual": failure.get("actual"),
            "stderr": (failure.get("stderr") or "")[:250],
        },
    }

    system_prompt = (
        "You are a strict programming tutor assistant. "
        "Follow the user instruction exactly. Return only the sentence."
    )
    user_prompt = (
        f"The student is struggling with {concept_label}. "
        "Looking at their code, explain why they failed this specific test case in 15 words."
    )

    try:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            temperature=0,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "user", "content": json.dumps(payload)},
            ],
        )
        micro_hint = (resp.output_text or "").strip()
        if micro_hint:
            words = micro_hint.split()
            if len(words) > 15:
                micro_hint = " ".join(words[:15])
        return micro_hint or None
    except Exception as e:
        print("[AI] explain_failure error:", e)
        return None
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
    player_id = data.get("player_id")
    attempt_no = data.get("attempt_no")

    if player_id is None:
        return jsonify({"status": "error", "message": "Missing player_id"}), 400
    try:
        player_id = int(player_id)
    except Exception:
        return jsonify({"status": "error", "message": "player_id must be an integer"}), 400

    try:
        attempt_no = int(attempt_no)
    except Exception:
        attempt_no = None

    if attempt_no is None or attempt_no < 1:
        attempt_no = next_attempt_no_for_player(player_id, problem_id)

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
        prev_fail_nodes = []
        for entry in reversed(_load_attempt_logs(player_id)):
            if entry.get("puzzle_name") == problem_id:
                prev_fail_nodes = entry.get("failed_node_ids") or []
                break

        mastery = load_student_mastery(player_id)
        mastery, _ = update_mastery(
            mastery,
            knowledge_graph,
            prev_fail_nodes,
            [],
            recovery_factor=0.5,
        )
        save_student_mastery(player_id, mastery)

        attempt_entry = {
            "player_id": player_id,
            "puzzle_name": problem_id,
            "attempt_no": attempt_no,
            "timestamp": now_iso_sg(),
            "solved": 1,
            "failed_node_ids": [],
            "concept_counts": {},
            "concept_mastery_log": {},
        }
        append_attempt_log(player_id, attempt_entry)

        return jsonify({
            "status": "success",
            "message": "All test cases passed!",
            "stdout": "\n".join(outputs_log),
        })

    failed_node_ids = [f.get("fail_node_id") for f in failures if f.get("fail_node_id")]
    concept_counts = concept_counts_from_failures(failures)

    prev_fail_nodes = []
    for entry in reversed(_load_attempt_logs(player_id)):
        if entry.get("puzzle_name") == problem_id:
            prev_fail_nodes = entry.get("failed_node_ids") or []
            break

    mastery = load_student_mastery(player_id)
    mastery, _ = update_mastery(
        mastery,
        knowledge_graph,
        prev_fail_nodes,
        failed_node_ids,
        recovery_factor=0.5,
    )
    save_student_mastery(player_id, mastery)

    graph_nodes = knowledge_graph.get("graph_nodes", {})
    concept_mastery_log = {}
    for node_id in failed_node_ids:
        node = graph_nodes.get(node_id, {})
        concept_id = node.get("concept_id")
        if concept_id:
            concept_mastery_log[concept_id] = mastery.get(concept_id, 0.5)

    def mastery_for_node(node_id: str) -> float:
        node = graph_nodes.get(node_id, {})
        concept_id = node.get("concept_id")
        if concept_id in mastery:
            return mastery[concept_id]
        return 0.5

    chosen_node_id = (
        min(
            failed_node_ids,
            key=lambda node_id: (mastery_for_node(node_id), node_id),
        )
        if failed_node_ids
        else None
    )
    if not chosen_node_id:
        chosen_node_id = failures[0].get("fail_node_id")

    attempt_entry = {
        "player_id": player_id,
        "puzzle_name": problem_id,
        "attempt_no": attempt_no,
        "timestamp": now_iso_sg(),
        "solved": 0,
        "failed_node_ids": failed_node_ids,
        "concept_counts": concept_counts,
        "concept_mastery_log": concept_mastery_log,
        "ai_chosen_node_id": chosen_node_id,
        "ai_confidence": None,
    }
    append_attempt_log(player_id, attempt_entry)
    
    node_data = knowledge_graph.get("graph_nodes", {}).get(chosen_node_id, {}) if chosen_node_id else {}
    concept_id = node_data.get("concept_id")
    concept = knowledge_graph.get("concepts", {}).get(concept_id, {}) if concept_id else {}
    concept_label = concept.get("label") or concept_id or "General"

    shown = next((f for f in failures if f.get("fail_node_id") == chosen_node_id), failures[0])

    micro_hint = ai_explain_failure(chosen_node_id, concept_label, code, shown)

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
        "concept_gap": node_data.get("concept_id", "General"),
        "debug_chosen_node_id": chosen_node_id,
        "debug_ai_confidence": None,
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
    player_id = data.get("player_id")
    ids = data.get("remaining_concept_ids", [])

    if player_id is None:
        return jsonify({"status": "error", "message": "Missing player_id"}), 400
    try:
        player_id = int(player_id)
    except Exception:
        return jsonify({"status": "error", "message": "player_id must be an integer"}), 400

    if not problem_id or problem_id not in knowledge_graph.get("problems", {}):
        return jsonify({"status": "error", "message": f"Invalid problem_id: {problem_id}"}), 400
    if not isinstance(ids, list) or not ids:
        return jsonify({"status": "error", "message": "No concept/error ids provided"}), 400

    graph_nodes = knowledge_graph.get("graph_nodes", {})
    chosen_id = next((cid for cid in ids if cid in graph_nodes), None)

    if not chosen_id:
        return jsonify({"status": "error", "message": "No matching graph_nodes for given concept ids"}), 400

    node = graph_nodes[chosen_id]
    concept_id = node.get("concept_id")
    concepts = knowledge_graph.get("concepts", {})
    concept = concepts.get(concept_id, {}) if concept_id else {}
    concept_label = concept.get("label") or concept_id or "General"
    generic_hint = concept.get("generic_hint")
    specific_hint = node.get("hint_text", "Think about this part again.")
    if generic_hint:
        hint_text = f"{concept_label}: {generic_hint}\n\nTip: {specific_hint}"
    else:
        hint_text = specific_hint
    attempt_entry = {
        "player_id": player_id,
        "puzzle_name": problem_id,
        "attempt_no": next_attempt_no_for_player(player_id, problem_id),
        "timestamp": now_iso_sg(),
        "solved": 0,
        "failed_node_ids": [chosen_id],
        "concept_counts": {concept_id or "General": 1},
    }
    append_attempt_log(player_id, attempt_entry)
    return jsonify({
        "status": "success",
        "concept_id": chosen_id,
        "hint": hint_text,
        "concept_id": concept_id or "General",
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
    suffix = username[len("testAccount"):] if is_test_account else ""
    is_test_account = is_test_account and suffix.isdigit()
    if is_test_account and password == "password@123":
        return jsonify({"status": "success", "user": username, "id": int(suffix)})

    return jsonify({"status": "error", "message": "Invalid credentials"}), 401

@app.route("/token", methods=["POST", "OPTIONS"])
def token():
    if request.method == "OPTIONS":
        return ("", 204)
    return jsonify({"ok": True}), 200

# ============================================================
# Tracking student mastery of concepts based on failed nodes
# ============================================================
def clamp(x: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, x))

def aggregate_penalties(knowledge_graph: dict, fail_node_ids: Set[str],) -> Dict[str, float]:
    """
    Sum penalties per concept across the given fail nodes.
    """
    delta = defaultdict(float)
    graph_nodes = knowledge_graph.get("graph_nodes", {})

    for node_id in fail_node_ids:
        node = graph_nodes.get(node_id, {})
        penalties = node.get("penalties", {}) or {}
        for concept_id, p in penalties.items():
            delta[concept_id] += float(p)
    return dict(delta)


def update_mastery(
    mastery: Dict[str, float],
    knowledge_graph: dict,
    prev_fail_nodes: List[str],
    curr_fail_nodes: List[str],
    recovery_factor: float = 0.5,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Returns:
      (updated_mastery, applied_delta_by_concept)

    Behaviour:
      - Penalize concepts for CURRENT fail nodes.
      - Reward concepts for RESOLVED fail nodes (those that were failing last attempt but not now),
        by adding recovery_factor * abs(penalty).
    """
    prev_set = set(prev_fail_nodes or [])
    curr_set = set(curr_fail_nodes or [])

    resolved = prev_set - curr_set
    current = curr_set

    penalty_delta = aggregate_penalties(knowledge_graph, current)

    resolved_penalties = aggregate_penalties(knowledge_graph, resolved)
    recovery_delta = {c: recovery_factor * abs(p) for c, p in resolved_penalties.items()}

    total_delta = defaultdict(float)
    for c, d in penalty_delta.items():
        total_delta[c] += d
    for c, d in recovery_delta.items():
        total_delta[c] += d

    updated = dict(mastery)  # copy
    for concept_id, d in total_delta.items():
        old = float(updated.get(concept_id, 0.0))
        updated[concept_id] = clamp(old + d)

    return updated, dict(total_delta)

def load_student_mastery(player_id: int) -> dict:
    path = os.path.join(STORE_FILE_PATH, f"mastery_{player_id}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    # Default state if new student
    return {cid: 0.5 for cid in knowledge_graph.get("concepts", {})}

def save_student_mastery(player_id: int, mastery: dict):
    path = os.path.join(STORE_FILE_PATH, f"mastery_{player_id}.json")
    with open(path, "w") as f:
        json.dump(mastery, f)
# ============================================================
# Playerdata storage (GET/PATCH)
# ============================================================
def now_iso_sg() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat()

def concept_counts_from_failures(failures: list[dict]) -> dict:
    """
    Map fail_node_id -> graph_nodes[fail_node_id].concept_id and count.
    Returns dict like {"concept_ll_stable_partition": 2, ...}
    """
    graph_nodes = knowledge_graph.get("graph_nodes", {})
    c = Counter()

    for f in failures:
        fid = f.get("fail_node_id")
        node = graph_nodes.get(fid, {})
        concept_id = node.get("concept_id")  # NEW FIELD you’re adding
        if concept_id:
            c[concept_id] += 1

    return dict(c)
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

def append_attempt_log(player_id: int, entry: dict) -> None:
    """
    Append a single attempt entry to store/attemptlog_<player_id>.json
    File format: a JSON list of entries.
    """
    path = os.path.join(STORE_FILE_PATH, f"attemptlog_{player_id}.json")

    logs = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                logs = json.load(f) or []
            if not isinstance(logs, list):
                logs = []
        except Exception:
            logs = []

    logs.append(entry)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(logs, f)

def _load_attempt_logs(player_id: int) -> list:
    path = os.path.join(STORE_FILE_PATH, f"attemptlog_{player_id}.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                logs = json.load(f) or []
            if isinstance(logs, list):
                return logs
        except Exception:
            return []
    return []

def next_attempt_no_for_player(player_id: int, puzzle_name: str) -> int:
    logs = _load_attempt_logs(player_id)
    max_no = 0
    solved_seen = False
    for e in logs:
        if e.get("puzzle_name") != puzzle_name:
            continue
        try:
            no = int(e.get("attempt_no") or 0)
        except Exception:
            no = 0
        if no > max_no:
            max_no = no
        if int(e.get("solved", 0)) == 1:
            solved_seen = True
    if max_no == 0:
        return 1
    if solved_seen:
        return max_no
    return max_no + 1
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

def _iter_attemptlog_files():
    pattern = os.path.join(STORE_FILE_PATH, "attemptlog_*.json")
    for path in sorted(glob.glob(pattern)):
        base = os.path.basename(path)  # attemptlog_12.json
        try:
            player_id = int(base.replace("attemptlog_", "").replace(".json", ""))
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

@app.route("/attemptlog/export.csv", methods=["GET", "OPTIONS"])
def export_attemptlog_csv():
    if request.method == "OPTIONS":
        return ("", 204)

    player_id_filter = request.args.get("player_id")

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "player_id",
        "puzzle_name",
        "attempt_no",
        "timestamp",
        "solved",
        "failed_node_ids_json",
        "concept_counts_json",
        "concept_mastery_log_json",
    ])

    def write_rows(pid: int, logs: list):
        for e in logs or []:
            if not isinstance(e, dict):
                continue
            writer.writerow([
                pid,
                e.get("puzzle_name", ""),
                e.get("attempt_no", ""),
                e.get("timestamp", ""),
                int(e.get("solved", 0)),
                json.dumps(e.get("failed_node_ids", []), ensure_ascii=False),
                json.dumps(e.get("concept_counts", {}), ensure_ascii=False),
                json.dumps(e.get("concept_mastery_log", {}), ensure_ascii=False),
            ])

    # Single player
    if player_id_filter:
        try:
            pid = int(player_id_filter)
        except ValueError:
            return jsonify({"ok": False, "message": "player_id must be an integer"}), 400

        path = os.path.join(STORE_FILE_PATH, f"attemptlog_{pid}.json")
        logs = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                try:
                    logs = json.load(f) or []
                except Exception:
                    logs = []
        write_rows(pid, logs)

        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": f'attachment; filename="attemptlog_{pid}.csv"'}
        )

    # All players
    for pid, path in _iter_attemptlog_files():
        with open(path, "r", encoding="utf-8") as f:
            try:
                logs = json.load(f) or []
            except Exception:
                logs = []
        write_rows(pid, logs)

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": 'attachment; filename="attemptlog_all.csv"'}
    )

if __name__ == "__main__":
    app.run(debug=True)
