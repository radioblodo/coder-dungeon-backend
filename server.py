import os
import json
import subprocess
from flask import Flask, request, jsonify
from flask_cors import CORS 

app = Flask(__name__)

CORS(
    app,
    resources={r"/*": {"origins": [
        "https://sage-bunny-f8e38a.netlify.app",   # your Netlify site
        "http://localhost:3000",                  # if you test locally
        "http://localhost:5173"
    ]}},
    supports_credentials=False,
    methods=["GET", "POST", "PATCH", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"]
)

STORE_FILE_PATH = "./store/"
GRAPH_FILE_PATH = "knowledge_graph.json"

# Ensure store directory exists
os.makedirs(STORE_FILE_PATH, exist_ok=True)


# 1. LOAD THE GRAPH INTO MEMORY
def load_graph():
    with open(GRAPH_FILE_PATH, "r") as f:
        return json.load(f)


knowledge_graph = load_graph()


@app.route("/")
def health_check():
    return "Knowledge Graph Server is Online!"

USERS = {
    "testAccount1": "password@123"
}

@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        return ("", 204)

@app.after_request
def add_cors_headers(resp):
    origin = request.headers.get("Origin")
    allowed = {
        "https://sage-bunny-f8e38a.netlify.app",
        "http://localhost:3000",
        "http://localhost:5173",
    }
    if origin in allowed:
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Vary"] = "Origin"
        resp.headers["Access-Control-Allow-Credentials"] = "true"
        resp.headers["Access-Control-Allow-Methods"] = "GET,POST,PATCH,PUT,DELETE,OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return resp

@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json(force=True) or {}
    username = data.get("username", "")
    password = data.get("password", "")

    if USERS.get(username) == password:
        return jsonify({"ok": True, "username": username})
    return jsonify({"ok": False, "message": "Invalid username or password"}), 401

# --- Senior compatibility endpoints ---

@app.route("/token", methods=["POST", "OPTIONS"])
def token():
    if request.method == "OPTIONS":
        return ("", 204)

    # Try parse JSON or form
    data = request.get_json(silent=True) or request.form.to_dict() or {}

    username = (
        data.get("username")
        or data.get("userName")
        or data.get("email")
        or "testAccount1"   # fallback for now
    )
    password = (
        data.get("password")
        or data.get("pass")
        or data.get("pwd")
        or None
    )

    # If password is provided, validate it. If not, allow issuing a token (compat).
    if password is not None and USERS.get(username) != password:
        return jsonify({"message": "Invalid username or password"}), 401

    return jsonify({
        "access_token": "dummy-token",
        "token_type": "bearer",
        "id": 0,
        "username": username
    })



@app.route("/playerdata/<int:player_id>", methods=["GET", "PATCH", "OPTIONS"])
def playerdata(player_id):
    if request.method == "OPTIONS":
        return ("", 204)

    path = os.path.join(STORE_FILE_PATH, f"playerdata_{player_id}.json")

    if request.method == "GET":
        if os.path.exists(path):
            with open(path, "r") as f:
                return jsonify(json.load(f))
        return jsonify({})

    data = request.get_json(silent=True) or {}
    with open(path, "w") as f:
        json.dump(data, f)

    return jsonify({"ok": True, "player_id": player_id})


# 2. THE GENERIC SUBMISSION ENGINE
# Unity sends: file, problem_id (e.g., "l1_c1_p1")
@app.route("/submit-code", methods=["POST", "OPTIONS"])
def submit_code():
    if request.method == "OPTIONS":
        return ("", 204)

    # --- 1) Get problem_id + python file path (supports JSON OR multipart) ---
    problem_id = None
    file_path = None

    # Case A: Unity non-local path sends JSON: {"problem_id": "...", "code": "..."}
    if request.is_json:
        data = request.get_json(silent=True) or {}
        problem_id = data.get("problem_id")
        code = data.get("code", "")

        if not problem_id:
            return jsonify({"status": "error", "message": "Missing problem_id"}), 400
        if not isinstance(code, str) or code.strip() == "":
            return jsonify({"status": "error", "message": "Missing code"}), 400

        safe_name = "".join(c for c in problem_id if c.isalnum() or c in ("_", "-"))
        file_name = f"{safe_name}_submission.py"
        file_path = os.path.join(STORE_FILE_PATH, file_name)

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code)
        except Exception as e:
            return jsonify({"status": "error", "message": f"Failed to save code: {e}"}), 500

    # Case B: Local path sends multipart form-data with "file" + "problem_id"
    else:
        problem_id = request.form.get("problem_id")
        if not problem_id:
            return jsonify({"status": "error", "message": "Missing problem_id"}), 400

        if "file" not in request.files:
            return jsonify({
                "status": "error",
                "message": "No file uploaded",
                "debug": {
                    "content_type": request.content_type,
                    "form_keys": list(request.form.keys()),
                    "file_keys": list(request.files.keys())
                }
            }), 400

        uploaded_file = request.files["file"]
        if not uploaded_file or uploaded_file.filename == "":
            return jsonify({"status": "error", "message": "Empty file upload"}), 400

        file_name = uploaded_file.filename
        file_path = os.path.join(STORE_FILE_PATH, file_name)

        try:
            uploaded_file.save(file_path)
        except Exception as e:
            return jsonify({"status": "error", "message": f"Failed to save file: {e}"}), 500

    print(f"Evaluating submission for: {problem_id}")

    # --- 2) Validate problem_id ---
    if problem_id not in knowledge_graph.get("problems", {}):
        return jsonify({"status": "error", "message": f"Invalid Problem ID: {problem_id}"}), 400

    problem_data = knowledge_graph["problems"][problem_id]
    test_cases = problem_data.get("test_cases", [])
    if not test_cases:
        return jsonify({"status": "error", "message": f"No test cases configured for {problem_id}"}), 500

    outputs_log = ""

    # --- 3) Execute code against test cases ---
    try:
        for case in test_cases:
            input_val = str(case.get("input", ""))
            expected_val = str(case.get("expected_output", ""))

            result = subprocess.run(
                ["python3", file_path, input_val],
                capture_output=True,
                text=True,
                timeout=5,
            )

            raw = (result.stdout or "")
            lines = [ln.strip() for ln in raw.splitlines() if ln.strip() != ""]
            actual_output = lines[-1] if lines else ""
            stderr_output = (result.stderr or "").strip()

            outputs_log += f"In: {input_val} | Out: {actual_output}\n"

            if stderr_output or actual_output != expected_val:
                fail_node_id = case.get("fail_node_id")

                node_data = knowledge_graph.get("graph_nodes", {}).get(fail_node_id, {}) if fail_node_id else {}
                hint_message = node_data.get("hint_text", "Check your logic.")
                concept = node_data.get("related_concept", "General")

                return jsonify({
                    "status": "failed",
                    "hint": hint_message,
                    "failed_test_case": input_val,
                    "student_output": actual_output,
                    "expected_output": expected_val,
                    "concept_gap": concept,
                    "stderr": stderr_output,
                })

        return jsonify({
            "status": "success",
            "message": "All test cases passed!",
            "stdout": outputs_log,
        })

    except subprocess.TimeoutExpired:
        return jsonify({
            "status": "failed",
            "hint": "Your code timed out. Do you have an infinite loop?",
            "concept_gap": "Complexity",
        })

    except Exception as e:
        print(e)
        return jsonify({"status": "error", "message": str(e)}), 500



# Endpoint 2: For non-coding puzzle
@app.route("/request-hint", methods=["POST", "OPTIONS"])
def request_hint():
    print("Processing hint request...")
    data = request.get_json()

    # Debugging (Print out JSON)
    print(data)

    problem_id = data.get("problem_id")
    ids = data.get("remaining_concept_ids", [])

    # --- Basic validation ---
    if problem_id not in knowledge_graph["problems"]:
        return jsonify({"status": "error", "message": "Invalid problem_id: " + problem_id})

    if not ids:
        return jsonify({"status": "error", "message": "No concept/error ids provided"})

    chosen_id = None
    chosen_node = None

    # --- Problem-specific logic ---

    # Old style: list of 'remaining concepts', pick the first that exists
    if problem_id == "l1_c3_p1":
        print("Providing hints for problem l1_c3_p1...")
        for cid in ids:
            node = knowledge_graph["graph_nodes"].get(cid)
            if node:
                chosen_id = cid
                chosen_node = node
                break
        if chosen_node is None:
            return jsonify(
                {
                    "status": "error",
                    "message": "No matching graph_nodes for given concept ids",
                }
            )

    # New style: grid puzzles – list contains exactly ONE error node id
    elif problem_id in (
        "l2_c1_p1",
        "l2_c1_p2",
        "l2_c1_p3",
        "l2_c2_p1",
        "l2_c2_p2",
        "l3_c1_p1",
        "l3_c1_p2",
        "l3_c1_p3",
        "l3_c2_p2",
        "l3_c2_p3",
        "l4_c1_p1_q1",
        "l4_c1_p1_q2",
        "l4_c1_p1_q3",
        "l4_c1_p2_q1",
        "l4_c1_p2_q2",
        "l4_c1_p2_q3",
        "l4_c1_p2_q4",
        "l4_c1_p3_q1",
        "l4_c1_p3_q2",
        "l4_c1_p3_q3",
        "l5_c1_p1",
        "l5_c1_p2",
        "l6_c1_p1",
        "l6_c1_p2"
    ):
        print(f"Providing hints for problem {problem_id}...")
        cid = ids[0]  # we expect exactly one
        node = knowledge_graph["graph_nodes"].get(cid)
        if not node:
            return jsonify(
                {"status": "error", "message": f"No graph node for id {cid}"}
            )
        chosen_id = cid
        chosen_node = node

    else:
        print(f"Hint request for unknown problem {problem_id}")
        return jsonify(
            {"status": "error", "message": f"No hint logic configured for {problem_id}"}
        )

    # --- Common success response ---
    return jsonify(
        {
            "status": "success",
            "concept_id": chosen_id,
            "hint": chosen_node.get(
                "hint_text", "Think about this part of the code again."
            ),
            "related_concept": chosen_node.get("related_concept", "General"),
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
