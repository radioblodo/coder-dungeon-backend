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
    supports_credentials=True,
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

@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json(force=True) or {}
    username = data.get("username", "")
    password = data.get("password", "")

    if USERS.get(username) == password:
        return jsonify({"ok": True, "username": username})
    return jsonify({"ok": False, "message": "Invalid username or password"}), 401

# 2. THE GENERIC SUBMISSION ENGINE
# Unity sends: file, problem_id (e.g., "l1_c1_p1")
@app.route("/submit-code", methods=["POST"])
def submit_code():
    problem_id = request.form.get("problem_id")

    print(f"Evaluating submission for: {problem_id}")

    # A. Validate Problem ID
    if problem_id not in knowledge_graph["problems"]:
        return jsonify({"status": "error", "message": "Invalid Problem ID"})

    # B. Save File
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded"})

    uploaded_file = request.files["file"]
    file_name = uploaded_file.filename
    file_path = os.path.join(STORE_FILE_PATH, file_name)
    uploaded_file.save(file_path)

    # C. Retrieve Test Cases from Graph (NOT from Unity)
    # This prevents students from faking inputs
    problem_data = knowledge_graph["problems"][problem_id]
    test_cases = problem_data["test_cases"]

    outputs_log = ""

    try:
        # D. Run Loop
        for case in test_cases:
            input_val = case["input"]
            expected_val = case["expected_output"]

            # Execute Student Code
            # Timeout added to prevent infinite loops freezing your server
            result = subprocess.run(
                ["python3", file_path, input_val],
                capture_output=True,
                text=True,
                timeout=5,
            )

            actual_output = result.stdout.strip()  # Remove trailing newlines
            outputs_log += f"In: {input_val} | Out: {actual_output}\n"

            # E. CHECK FOR FAILURE (The Graph Logic)
            # If crash (stderr) OR wrong output
            if result.stderr or actual_output != expected_val:

                # 1. Identify the Node
                fail_node_id = case["fail_node_id"]

                # 2. Retrieve the Hint
                node_data = knowledge_graph["graph_nodes"].get(fail_node_id, {})
                hint_message = node_data.get("hint_text", "Check your logic.")
                concept = node_data.get("related_concept", "General")

                return jsonify(
                    {
                        "status": "failed",
                        "hint": hint_message,
                        "failed_test_case": input_val,
                        "student_output": actual_output,
                        "expected_output": expected_val,
                        "concept_gap": concept,
                        "stderr": result.stderr,
                    }
                )

        # F. If loop finishes without returning, Success!
        return jsonify(
            {
                "status": "success",
                "message": "All test cases passed!",
                "stdout": outputs_log,
            }
        )

    except subprocess.TimeoutExpired:
        return jsonify(
            {
                "status": "failed",
                "hint": "Your code timed out. Do you have an infinite loop?",
                "concept_gap": "Complexity",
            }
        )
    except Exception as e:
        print(e)
        return jsonify({"status": "error", "message": str(e)})


# Endpoint 2: For non-coding puzzle
@app.route("/request-hint", methods=["POST"])
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
        "l5_c1_p1",
        "l5_c1_p2"
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
