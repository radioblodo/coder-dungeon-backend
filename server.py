import os
import json
import subprocess
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import csv
import io
import glob
import re

EXC_LINE_RE = re.compile(r"^(\w+Error|Exception):\s*(.*)$")

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

os.makedirs(STORE_FILE_PATH, exist_ok=True)

# -------- Helpers --------
def load_graph():
    with open(GRAPH_FILE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_line(s: str) -> str:
    """Normalize stdout/expected strings to avoid invisible mismatch."""
    if s is None:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n").strip()
    s = s.replace("\ufeff", "")   # BOM
    s = s.replace("\u200b", "")   # zero-width space
    s = s.replace("\x00", "")     # null
    return s

def parse_answer_from_stdout(stdout: str) -> str:
    """Take the last non-empty, normalized line as the answer."""
    raw = stdout or ""
    lines = []
    for ln in raw.splitlines():
        nl = normalize_line(ln)
        if nl != "":
            lines.append(nl)
    return lines[-1] if lines else ""

def is_known_buggy_driver_crash(stderr: str) -> bool:
    """
    Whitelist the known template-driver bug:
    When words == [], driver prints 'empty!' then crashes at print(words[i]).
    """
    if not stderr:
        return False
    return ("IndexError: list index out of range" in stderr) and ("print(words[i])" in stderr)

# -------- Load knowledge graph --------
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

    data = request.get_json(silent=True) or request.form.to_dict() or {}

    username = (
        data.get("username")
        or data.get("userName")
        or data.get("email")
        or "testAccount1"
    )
    password = (
        data.get("password")
        or data.get("pass")
        or data.get("pwd")
        or None
    )

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
          with open(path, "r", encoding="utf-8") as f:                                                                                 
              return jsonify(json.load(f))                                                                                             
      return jsonify({})                                                                                                               
                                                                                                                                       
  data = request.get_json(silent=True) or {}                                                                                           
                                                                                                                                       
  # Load existing store (dict of puzzle entries)                                                                                       
  store = {}                                                                                                                           
  if os.path.exists(path):                                                                                                             
      with open(path, "r", encoding="utf-8") as f:                                                                                     
          try:                                                                                                                         
              store = json.load(f) or {}                                                                                               
          except Exception:                                                                                                            
              store = {}                                                                                                               
                                                                                                                                       
  # Normalize PATCH payload into PuzzleData shape                                                                                      
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
      # If client sends full dict already, merge it                                                                                    
      for k, v in data.items():                                                                                                        
          if isinstance(v, dict):                                                                                                      
              store[k] = v                                                                                                             
                                                                                                                                       
  with open(path, "w", encoding="utf-8") as f:                                                                                         
      json.dump(store, f)                                                                                                              
                                                                                                                                       
  return jsonify({"ok": True, "player_id": player_id}) 

# 2. THE GENERIC SUBMISSION ENGINE

# Helper function to evaluate code against test cases
def extract_exception(stderr: str):
    """
    Return (exc_name, exc_msg) from stderr, or (None, None).
    Tries to read the last 'XxxError: message' line.
    """
    if not stderr:
        return None, None
    lines = [ln.strip() for ln in stderr.splitlines() if ln.strip()]
    for ln in reversed(lines):
        m = EXC_LINE_RE.match(ln)
        if m:
            return m.group(1), m.group(2)
    return None, None

def rule_matches(rule_match: dict, exc_name: str, exc_msg: str, stderr: str) -> bool:
    """
    rule_match can include:
      - exception: "AttributeError"
      - contains: ["NoneType", "item"]
      - regex: "pattern"
    """
    if not rule_match:
        return False

    if "exception" in rule_match and exc_name != rule_match["exception"]:
        return False

    haystack = (stderr or "") + "\n" + (exc_msg or "")

    contains = rule_match.get("contains")
    if contains:
        for token in contains:
            if token not in haystack:
                return False

    pattern = rule_match.get("regex")
    if pattern and not re.search(pattern, haystack):
        return False

    return True

def classify_runtime_fail_node(kg: dict, problem_data: dict, stderr: str) -> str:
    """
    Return a fail_node_id for this runtime error.
    Order: problem runtime_rules -> global runtime_rules -> generic runtime node.
    """
    exc_name, exc_msg = extract_exception(stderr)

    # 1) per-problem
    for rule in problem_data.get("runtime_rules", []):
        if rule_matches(rule.get("match", {}), exc_name, exc_msg, stderr):
            return rule.get("fail_node_id", "node_runtime_generic")

    # 2) global
    for rule in kg.get("runtime_rules", []):
        if rule_matches(rule.get("match", {}), exc_name, exc_msg, stderr):
            return rule.get("fail_node_id", "node_runtime_generic")

    return "node_runtime_generic"

import re

# -----------------------
# Helpers
# -----------------------

def _to_int(s):
    try:
        return int(str(s).strip())
    except:
        return None

def _split_csv(s):
    s = str(s).strip()
    if s == "":
        return []
    return [x.strip() for x in s.split(",") if x.strip() != ""]

def _to_int_list(csv_str):
    try:
        return [int(x) for x in _split_csv(csv_str)]
    except:
        return None

def _same_multiset(a, b):
    # a, b are lists
    return sorted(a) == sorted(b)

def _all_deltas_equal(failures, delta_target):
    deltas = []
    for f in failures:
        a = _to_int(f.get("actual"))
        e = _to_int(f.get("expected"))
        if a is None or e is None:
            return False
        deltas.append(a - e)
    return (len(deltas) > 0) and all(d == delta_target for d in deltas)

def _failed_case_ids(failures):
    return {f.get("case_id") for f in failures if f.get("case_id")}

def _first_fail_node(failures):
    return failures[0].get("fail_node_id") if failures else None


# -----------------------
# Main dispatcher
# -----------------------

def classify_wrong_answer_fail_node(problem_id: str, failures: list) -> str:
    """
    Master wrong-answer classifier. Dispatches to per-problem classifiers.
    Falls back to the first failing test case's fail_node_id.
    """
    if not failures:
        return None

    # Optional generic rule (safe + useful): consistent off-by-one
    # You can keep this or remove it; it helps multiple problems.
    # NOTE: For max-depth, "+1 everywhere" is common.
    # We'll still allow per-problem classifier to override.
    # (We apply generic later as fallback.)

    dispatch = {
        "l1_c1_p1": classify_wrong_answer_l1_c1_p1,
        "l2_c2_p3": classify_wrong_answer_l2_c2_p3,
        "l3_c2_p1": classify_wrong_answer_l3_c2_p1,
        "l4_c1_p4": classify_wrong_answer_l4_c1_p4,
        "l6_c3_p1": classify_wrong_answer_l6_c3_p1,
    }

    fn = dispatch.get(problem_id)
    if fn:
        node = fn(failures)
        if node:
            return node

    # Generic fallback: consistent +1 or -1
    if _all_deltas_equal(failures, 1) or _all_deltas_equal(failures, -1):
        return _first_fail_node(failures)

    return _first_fail_node(failures)


# ============================================================
# 1) l1_c1_p1 — Move evens to back, preserve relative order
# ============================================================

def classify_wrong_answer_l1_c1_p1(failures: list) -> str:
    failed = _failed_case_ids(failures)

    MIXED1 = "evens_tc_01"
    ALL_EVEN = "evens_tc_02"
    ALL_ODD = "evens_tc_03"
    STABILITY = "evens_tc_04"

    # If only edge case fails, return that edge hint
    if failed == {ALL_EVEN}:
        return "node_evens_all_even"
    if failed == {ALL_ODD}:
        return "node_evens_all_odd"

    # If stability test fails, check whether it's truly a stability/order issue
    for f in failures:
        if f.get("case_id") == STABILITY:
            inp = _to_int_list(f.get("input"))
            act = _to_int_list(f.get("actual"))
            exp = _to_int_list(f.get("expected"))
            if inp and act and exp:
                exp_odds = [x for x in inp if x % 2 != 0]
                exp_evens = [x for x in inp if x % 2 == 0]
                act_odds = [x for x in act if x % 2 != 0]
                act_evens = [x for x in act if x % 2 == 0]

                # They partitioned into odds then evens, but order inside group differs
                if _same_multiset(act_odds, exp_odds) and _same_multiset(act_evens, exp_evens):
                    if act_odds != exp_odds or act_evens != exp_evens:
                        return "node_evens_stability"

            # If we can't parse, still treat tc_04 failing as stability (most diagnostic)
            return "node_evens_stability"

    # Otherwise core logic
    return "node_evens_core"


# ============================================================
# 2) l2_c2_p3 — Sort a stack using push/pop + one temp stack
# ============================================================

def classify_wrong_answer_l2_c2_p3(failures: list) -> str:
    failed = _failed_case_ids(failures)

    BASIC = "sortstack_tc_01_basic"
    ALREADY = "sortstack_tc_02_already_sorted"
    REVERSE = "sortstack_tc_03_reverse_sorted"
    DUP = "sortstack_tc_04_with_duplicates"
    SINGLE = "sortstack_tc_05_single"

    # Most specific edge case first
    if failed == {SINGLE}:
        return "l2_c2_p3_node_sortstack_single"

    # Duplicates often need careful handling; if only that fails, report duplicates node
    if failed == {DUP}:
        return "l2_c2_p3_node_sortstack_duplicates"

    # If already sorted input fails but basic passes, they may be "over-shuffling" or wrong direction
    if ALREADY in failed and BASIC not in failed and REVERSE not in failed:
        return "l2_c2_p3_node_sortstack_already_sorted"

    # If reverse-sorted fails but already-sorted passes, likely not moving items correctly back/forth
    if REVERSE in failed and ALREADY not in failed:
        return "l2_c2_p3_node_sortstack_reverse_sorted"

    # If basic fails, it's core
    if BASIC in failed:
        return "l2_c2_p3_node_sortstack_core"

    # Otherwise pick the most diagnostic among those failed
    priority = [
        (DUP, "l2_c2_p3_node_sortstack_duplicates"),
        (REVERSE, "l2_c2_p3_node_sortstack_reverse_sorted"),
        (ALREADY, "l2_c2_p3_node_sortstack_already_sorted"),
        (SINGLE, "l2_c2_p3_node_sortstack_single"),
        (BASIC, "l2_c2_p3_node_sortstack_core"),
    ]
    for cid, node in priority:
        if cid in failed:
            return node

    return failures[0].get("fail_node_id")


# ============================================================
# 3) l3_c2_p1 — Max depth of binary tree (depth of single node = 0)
# ============================================================

def classify_wrong_answer_l3_c2_p1(failures: list) -> str:
    failed = _failed_case_ids(failures)

    SINGLE = "maxdepth_tc_02_single"
    TWO = "maxdepth_tc_03_two_levels"
    LEFT = "maxdepth_tc_04_left_skewed"
    RIGHT = "maxdepth_tc_05_right_skewed"
    UNBAL = "maxdepth_tc_06_unbalanced_deeper_left"
    SPARSE = "maxdepth_tc_07_sparse_with_nulls"
    COMPLETE = "maxdepth_tc_08_complete_three_levels"
    ZIGZAG = "maxdepth_tc_09_zigzag_chain"

    # ---------- NEW RULE 0: missing +1 / not accumulating depth ----------
    # If two-level expects 1 but student outputs 0, it's almost certainly "forgot +1".
    two_fail = next((f for f in failures if f.get("case_id") == TWO), None)
    if two_fail:
        a = _to_int(two_fail.get("actual"))
        e = _to_int(two_fail.get("expected"))
        if a == 0 and e == 1:
            return "l3_c2_p1_node_maxdepth_basic_height"

    # More general: many deeper cases output 0 while expected > 0
    deep_cases = 0
    deep_zero = 0
    for f in failures:
        a = _to_int(f.get("actual"))
        e = _to_int(f.get("expected"))
        if a is None or e is None:
            continue
        if e >= 2:              # deeper than 2 levels
            deep_cases += 1
            if a == 0:
                deep_zero += 1
    if deep_cases >= 2 and deep_zero == deep_cases:
        return "l3_c2_p1_node_maxdepth_basic_height"
    
    # NEW: detect "ignores left subtree" (more precise)
    left_fail = next((f for f in failures if f.get("case_id") == LEFT), None)
    right_fail = next((f for f in failures if f.get("case_id") == RIGHT), None)

    if left_fail:
        aL = _to_int(left_fail.get("actual"))
        eL = _to_int(left_fail.get("expected"))

        aR = _to_int(right_fail.get("actual")) if right_fail else None
        eR = _to_int(right_fail.get("expected")) if right_fail else None

        # left expected deep but output tiny
        left_tiny = (aL is not None and eL is not None and eL >= 2 and aL <= 1)

        # right is "not tiny" (meaning: they *are* able to traverse some depth on right)
        right_not_tiny = (aR is not None and eR is not None and aR >= 2)

        if left_tiny and right_not_tiny:
            return "l3_c2_p1_node_maxdepth_skewed_left"
            
    # Directional diagnosis (prevents "single node" from stealing the hint)
    if LEFT in failed and RIGHT not in failed:
        return "l3_c2_p1_node_maxdepth_skewed_left"
    if RIGHT in failed and LEFT not in failed:
        return "l3_c2_p1_node_maxdepth_skewed_right"

    # Zigzag special
    if ZIGZAG in failed and LEFT not in failed and RIGHT not in failed:
        return "l3_c2_p1_node_maxdepth_zigzag_structure"

    # Sparse vs complete
    if SPARSE in failed and COMPLETE not in failed:
        return "l3_c2_p1_node_maxdepth_null_children_handling"

    # Unbalanced deeper-left
    if UNBAL in failed and TWO not in failed:
        return "l3_c2_p1_node_maxdepth_unbalanced_tree"

    # Off-by-one definition mismatch: actual = expected + 1 across failures
    # (Common “node-count height” instead of “edge depth”.)
    if _all_deltas_equal(failures, 1):
        # Your closest node right now is single_node, but ideally create a dedicated node later.
        return "l3_c2_p1_node_maxdepth_single_node"

    # Forgotten +1 typically shows up in two-level failure
    if TWO in failed:
        return "l3_c2_p1_node_maxdepth_basic_height"

    # Fallback: if they only fail SINGLE, return SINGLE node
    if failed == {SINGLE}:
        return "l3_c2_p1_node_maxdepth_single_node"

    # Else first mapped node
    return failures[0].get("fail_node_id")


# ============================================================
# 4) l4_c1_p4 — Min in rotated sorted array (with duplicates)
# ============================================================

def classify_wrong_answer_l4_c1_p4(failures: list) -> str:
    failed = _failed_case_ids(failures)

    SORTED = "min_tc_01_sorted_no_rotation"
    ROT = "min_tc_02_simple_rotation"
    SINGLE = "min_tc_03_single_element"
    TWO = "min_tc_04_two_elements_rotated"
    DUP_EQ = "min_tc_05_duplicates_not_rotated"
    DUP_TRICKY = "min_tc_06_duplicates_rotated_tricky"
    DUP_PIVOT = "min_tc_07_duplicates_with_pivot"
    NEG = "min_tc_08_negative_values_rotated"

    # Strong edge cases first
    if failed == {SINGLE}:
        return "node_min_single_element"
    if TWO in failed:
        return "node_min_boundary_two_elements"
    if NEG in failed:
        return "node_min_handle_negatives"

    # Duplicate-related diagnostics
    # If all-equal duplicates fail, likely special-case needed
    if failed == {DUP_EQ}:
        return "node_min_duplicates_equal_values"

    # Pivot duplicates is most specific
    if DUP_PIVOT in failed:
        return "node_min_duplicates_shrink_high"

    # Tricky duplicates rotated (breaks sorted check)
    if DUP_TRICKY in failed:
        return "node_min_duplicates_break_sorted_check"

    # Rotation logic: passes sorted but fails rotation
    if ROT in failed and SORTED not in failed:
        return "node_min_rotation_binary_search"
    if ROT in failed and SORTED in failed:
        # both failing usually means core binary-search logic is off
        return "node_min_rotation_binary_search"

    # If only sorted-no-rotation fails, likely wrong “already sorted segment” short-circuit
    if failed == {SORTED}:
        return "node_min_already_sorted_segment"

    # Otherwise choose most diagnostic by priority
    priority = [
        (TWO, "node_min_boundary_two_elements"),
        (SINGLE, "node_min_single_element"),
        (NEG, "node_min_handle_negatives"),
        (DUP_PIVOT, "node_min_duplicates_shrink_high"),
        (DUP_TRICKY, "node_min_duplicates_break_sorted_check"),
        (DUP_EQ, "node_min_duplicates_equal_values"),
        (ROT, "node_min_rotation_binary_search"),
        (SORTED, "node_min_already_sorted_segment"),
    ]
    for cid, node in priority:
        if cid in failed:
            return node

    return failures[0].get("fail_node_id")


# ============================================================
# 5) l6_c3_p1 — Trie words with prefix in alphabetical order
# ============================================================

def classify_wrong_answer_l6_c3_p1(failures: list) -> str:
    failed = _failed_case_ids(failures)

    CORE = "trie_prefix_tc_01"
    ALPHA = "trie_prefix_tc_02"
    NOTFOUND = "trie_prefix_tc_03"
    WORDVS = "trie_prefix_tc_04"
    SIBLING = "trie_prefix_tc_05"

    # If not-found case fails, it's usually most important
    if NOTFOUND in failed:
        return "node_trie_prefix_not_found"

    # If alphabetical case fails, verify it's an ordering issue (optional)
    for f in failures:
        if f.get("case_id") == ALPHA:
            exp = _split_csv(f.get("expected"))
            act_raw = str(f.get("actual", "")).strip()
            act = _split_csv(act_raw)

            # If same set but different order -> alphabetical node
            if exp and act and _same_multiset(exp, act) and exp != act:
                return "node_trie_prefix_alphabetical"

            return "node_trie_prefix_alphabetical"

    # Word vs prefix (prefix itself is a word) is very specific
    if WORDVS in failed:
        return "node_trie_prefix_word_vs_prefix"

    # Sibling traversal (linked-list siblings) is specific
    if SIBLING in failed:
        return "node_trie_prefix_sibling_traversal"

    # Otherwise core
    return "node_trie_prefix_core"

@app.route("/submit-code", methods=["POST", "OPTIONS"])
def submit_code():
    if request.method == "OPTIONS":
        return ("", 204)

    problem_id = None
    file_path = None

    # Case A: Unity sends JSON: {"problem_id": "...", "code": "..."}
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

    # Case B: multipart form-data with "file" + "problem_id"
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

    if problem_id not in knowledge_graph.get("problems", {}):
        return jsonify({"status": "error", "message": f"Invalid Problem ID: {problem_id}"}), 400

    problem_data = knowledge_graph["problems"][problem_id]
    test_cases = problem_data.get("test_cases", [])
    if not test_cases:
        return jsonify({"status": "error", "message": f"No test cases configured for {problem_id}"}), 500

    outputs_log = ""
    failures = []   # NEW: collect wrong-answer failures
    try:
        for case in test_cases:
            input_val = str(case.get("input", ""))
            expected_val = normalize_line(str(case.get("expected_output", "")))

            result = subprocess.run(
                ["python3", file_path, input_val],
                capture_output=True,
                text=True,
                timeout=5,
            )

            raw_out = result.stdout or ""
            raw_err = result.stderr or ""

            actual_output = parse_answer_from_stdout(raw_out)
            outputs_log += f"In: {input_val} | Out: {actual_output}\n"

            # --- Runtime error handling ---
            if result.returncode != 0:
                # Special-case: known buggy driver crash AFTER printing correct expected output
                if actual_output == expected_val and is_known_buggy_driver_crash(raw_err):
                    # treat this test case as passed
                    continue

                # fail_node_id = case.get("fail_node_id")
                fail_node_id = classify_runtime_fail_node(knowledge_graph, problem_data, raw_err)
                node_data = knowledge_graph.get("graph_nodes", {}).get(fail_node_id, {}) if fail_node_id else {}
                return jsonify({
                    "status": "failed",
                    "hint": node_data.get("hint_text", "Your code crashed."),
                    "failed_test_case": input_val,
                    "student_output": actual_output,
                    "expected_output": expected_val,
                    "concept_gap": node_data.get("related_concept", "Runtime Error"),
                    "stderr": raw_err.strip(),
                    "debug_raw_stdout": repr(raw_out),
                    "debug_raw_stderr": repr(raw_err),
                })


            # --- Wrong answer handling (collect, don't early return) ---
            if actual_output != expected_val:
                failures.append({
                    "case_id": case.get("id"),
                    "fail_node_id": case.get("fail_node_id"),
                    "input": input_val,
                    "expected": expected_val,
                    "actual": actual_output,
                    "stderr": raw_err.strip(),
                    "raw_stdout": raw_out,   # optional
                })
                # keep running remaining tests
                continue
        
        # If any wrong-answer failures happened, choose best hint node now
        if failures:
            chosen_node_id = classify_wrong_answer_fail_node(problem_id, failures)
            node_data = knowledge_graph.get("graph_nodes", {}).get(chosen_node_id, {}) if chosen_node_id else {}

            # optionally choose the most diagnostic failure to show
            shown = failures[0]

            return jsonify({
                "status": "failed",
                "hint": node_data.get("hint_text", "Check your logic."),
                "failed_test_case": shown["input"],
                "student_output": shown["actual"],
                "expected_output": shown["expected"],
                "concept_gap": node_data.get("related_concept", "General"),
                "stderr": shown.get("stderr", ""),
                "debug_failures": failures,  # keep during dev, remove in production
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
    data = request.get_json() or {}
    print(data)

    problem_id = data.get("problem_id")
    ids = data.get("remaining_concept_ids", [])

    if problem_id not in knowledge_graph.get("problems", {}):
        return jsonify({"status": "error", "message": "Invalid problem_id: " + str(problem_id)})

    if not ids:
        return jsonify({"status": "error", "message": "No concept/error ids provided"})

    chosen_id = None
    chosen_node = None

    # Old style
    if problem_id == "l1_c3_p1":
        print("Providing hints for problem l1_c3_p1...")
        for cid in ids:
            node = knowledge_graph.get("graph_nodes", {}).get(cid)
            if node:
                chosen_id = cid
                chosen_node = node
                break
        if chosen_node is None:
            return jsonify({
                "status": "error",
                "message": "No matching graph_nodes for given concept ids",
            })

    # New style (single node id)
    elif problem_id in (
        "l1_c4_p1",
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
        "l6_c1_p2",
        "l6_c3_p1",
    ):
        print(f"Providing hints for problem {problem_id}...")
        cid = ids[0]
        node = knowledge_graph.get("graph_nodes", {}).get(cid)
        if not node:
            return jsonify({"status": "error", "message": f"No graph node for id {cid}"})
        chosen_id = cid
        chosen_node = node

    else:
        print(f"Hint request for unknown problem {problem_id}")
        return jsonify({"status": "error", "message": f"No hint logic configured for {problem_id}"})

    return jsonify({
        "status": "success",
        "concept_id": chosen_id,
        "hint": chosen_node.get("hint_text", "Think about this part of the code again."),
        "related_concept": chosen_node.get("related_concept", "General"),
    })

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

    # Optional query params:
    #   ?player_id=12        -> export only one player
    #   ?solved_only=1       -> export only solved attempts
    player_id_filter = request.args.get("player_id")
    solved_only = request.args.get("solved_only") in ("1", "true", "True", "yes", "YES")

    # CSV output in memory
    output = io.StringIO()
    writer = csv.writer(output)

    # Header row
    writer.writerow(["player_id", "puzzle_name", "time", "attempts", "solved"])

    def write_rows_for_player(pid: int, store: dict):
        # Your store is a dict of entries: { key: {puzzle_name,time,attempts,solved}, ... }
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

    # Export a single player
    if player_id_filter:
        try:
            pid = int(player_id_filter)
        except ValueError:
            return jsonify({"ok": False, "message": "player_id must be an integer"}), 400

        path = os.path.join(STORE_FILE_PATH, f"playerdata_{pid}.json")
        if not os.path.exists(path):
            # return empty CSV with header
            csv_text = output.getvalue()
            return Response(
                csv_text,
                mimetype="text/csv",
                headers={"Content-Disposition": f'attachment; filename="playerdata_{pid}.csv"'}
            )

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
