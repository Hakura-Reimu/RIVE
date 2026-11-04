"""One research cycle of a Phase A candidate (handoff doc 3.7.3), as an explicit state machine.

The model only produces stage outputs; every fact (counts, metrics, test results) comes from the
controller's own records, and every stage output is schema-validated and fact-checked.
"""
import ast
import hashlib
import json
import random
import re
import shutil
from collections import Counter
from pathlib import Path

import jsonschema

from rive.benchmark.stats import paired_bootstrap
from rive.research import schemas
from rive.runtime.toolcall import task_text

ROUTER = "agent/tool_router.py"
MAX_ROUTER_LINES = 200
FORBIDDEN_IMPORTS = {"os", "sys", "subprocess", "socket", "shutil", "pathlib", "ctypes", "multiprocessing",
                     "threading", "importlib", "builtins", "urllib", "http", "signal", "inspect", "gc"}
FORBIDDEN_CALLS = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "getattr", "setattr"}

SYSTEM = (
    "You are the research decision-maker for an AI agent. The agent answers user requests by calling tools. "
    "A Python program called the tool router decides which tools the language model is shown for each request, "
    "in what order, and may add a short hint to the system prompt. You improve the agent only by changing the "
    "router, based on evidence from development tests. You cannot change the model, the tests or the scoring. "
    "Be concrete: cite failure types, counts and example ids from the feedback you are given."
)

ROUTER_CONTRACT = (
    "Router contract: `select(task, tools, history=None)` returns {\"tools\": [names], \"hint\": str}. "
    "`task` is the user's request text; `tools` is a list of dicts with 'name', 'description' and 'parameters' "
    "(JSON Schema). Return only names from `tools`, without duplicates, in the order the model should see them; "
    "an empty list means the model sees no tools. The hint is at most 2000 characters. Use only the Python "
    "standard library (no os, sys, subprocess, socket, file access, eval or exec). Each call must finish within "
    "1 second."
)


class StageFailed(Exception):
    pass


# Same rules as benchmark.bfcl.classify, in words. In the main run the model often filed "the router hid the
# right tool" cases (no_call) under missing_call, so the definitions are now part of the feedback.
FAILURE_MEANINGS = {
    "no_call": "a tool call was expected, but the model answered without calling any tool",
    "wrong_tool": "the model called at least one function that is not in the expected answer",
    "missing_call": "every call used an expected function, but there were fewer calls than expected "
                    "(for example 2 of 3 parallel calls)",
    "extra_call": "every call used an expected function, but there were more calls than expected",
    "invalid_arguments": "right functions and number of calls, but some argument values were wrong",
    "parse_error": "the model wrote a tool call that could not be parsed",
    "irrelevance_violation": "no offered tool fits the request, but the model called one anyway",
    "router_error": "the router raised an error or took too long on this task",
}


def observe(results, tasks_by_id, max_examples=8, seed=0):
    """Development feedback built by the program, not interpreted by the model (handoff doc 3.7.3 OBSERVE).

    Examples are drawn per failure type with `seed` (the cycle number): with fixed first-two examples the
    main run kept tuning the router to the same two irrelevance requests for ten cycles.
    """
    rng = random.Random(seed)
    counts = Counter(r["failure"] for r in results if r["failure"])
    examples = []
    for failure, _ in counts.most_common():
        failing = [r for r in results if r["failure"] == failure]
        for r in rng.sample(failing, min(2, len(failing))):
            t = tasks_by_id[r["id"]]
            expected = [next(iter(g)) for g in t["ground_truth"]]
            ex = {"id": r["id"], "failure": failure, "request": task_text(t)[:400],
                  "tools_offered": len(t["functions"]), "tools_shown": r.get("n_tools_shown"),
                  "expected_calls": expected, "model_calls": [c[0] for c in r.get("calls", [])]}
            if "shown" in r and expected:
                ex["expected_tools_shown"] = all(e in r["shown"] for e in expected)
            examples.append(ex)
        if len(examples) >= max_examples:
            break
    by_cat = {}
    for r in results:
        by_cat.setdefault(tasks_by_id[r["id"]].get("category", "all"), []).append(r["correct"])
    hidden = [r for r in results if "shown" in r and any(
        next(iter(g)) not in r["shown"] for g in tasks_by_id[r["id"]]["ground_truth"])]
    return {"n_tasks": len(results), "accuracy": round(sum(r["correct"] for r in results) / len(results), 4),
            "accuracy_by_category": {c: round(sum(v) / len(v), 4) for c, v in sorted(by_cat.items())},
            "failure_counts": dict(counts.most_common()),
            "failure_meanings": {f: FAILURE_MEANINGS[f] for f in counts if f in FAILURE_MEANINGS},
            "router_recall": {"mean_tools_shown": round(sum(r.get("n_tools_shown", 0) for r in results) /
                                                        len(results), 2),
                              "tasks_with_an_expected_tool_hidden": len(hidden),
                              "of_which_failed": sum(not r["correct"] for r in hidden)},
            "failure_examples": examples[:max_examples]}


def check_router_source(source):
    """Static checks before anything runs; returns a list of problems (empty if fine)."""
    return check_python_source(source, "select", MAX_ROUTER_LINES, "router")


def check_python_source(source, function, max_lines, what="file"):
    """Line cap, syntax, the required top-level function, and no forbidden imports or calls."""
    problems = []
    if len(source.splitlines()) > max_lines:
        problems.append(f"{what} has more than {max_lines} lines")
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return problems + [f"syntax error: {e}"]
    if not any(isinstance(n, ast.FunctionDef) and n.name == function for n in tree.body):
        problems.append(f"no top-level function named {function}")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for name in names:
                if name.split(".")[0] in FORBIDDEN_IMPORTS:
                    problems.append(f"forbidden import: {name}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            problems.append(f"forbidden call: {node.func.id}()")
    return problems


def strip_docstrings(tree):
    """Removes the docstrings of a parsed module in place and returns it."""
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) and isinstance(
                getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
            body.pop(0)
    return tree


def normalized_hash(source):
    """Hash ignoring comments, docstrings and formatting, for duplicate detection."""
    return hashlib.sha256(ast.dump(strip_docstrings(ast.parse(source))).encode()).hexdigest()[:16]


def extract_python(text):
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    return m.group(1) if m else text


class Stages:
    """Structured stage outputs from the research model, shared by the Phase A and Phase C loops."""

    def __init__(self, llm, thinking=False, retries=2, temperature=0.7):
        self.llm, self.thinking, self.retries = llm, thinking, retries
        # Greedy decoding made every cycle with the same parent produce the same router (first pilot);
        # sampling with a per-cycle seed keeps cycles different but reproducible.
        self.temperature = temperature
        self.seed = 0

    def _structured(self, messages, schema, check=None):
        """Free analysis first, then constrained JSON; validated, retried with the error on failure.
        check(obj) -> a problem the schema cannot express, or None."""
        analysis = self.llm.chat(messages + [{"role": "user", "content": "First reason about this in plain text "
                                              "(at most 250 words). Do not output JSON yet."}],
                                 max_tokens=700, thinking=self.thinking, temperature=self.temperature, seed=self.seed)
        convo = messages + [{"role": "assistant", "content": analysis},
                            {"role": "user", "content": "Now give your answer as JSON matching the required schema."}]
        errors = []
        for _ in range(self.retries + 1):
            raw = self.llm.chat(convo, schema=schema, max_tokens=1500, temperature=self.temperature, seed=self.seed)
            try:
                obj = json.loads(raw)
                jsonschema.validate(obj, schema)
            except (json.JSONDecodeError, jsonschema.ValidationError) as e:
                errors.append(str(e).splitlines()[0][:300])
            else:
                problem = check(obj) if check else None
                if not problem:
                    return obj, {"attempts": len(errors) + 1, "errors": errors, "analysis": analysis}
                errors.append(problem)
            convo = convo + [{"role": "assistant", "content": raw},
                             {"role": "user", "content": f"That was not valid: {errors[-1]}. Output corrected JSON."}]
        raise StageFailed(f"no valid output after {len(errors)} attempts: {errors[-1]}")


class ResearchLoop(Stages):
    def __init__(self, llm, evaluate, make_sandbox, sample_tasks, run_dir, thinking=False, retries=2,
                 seen_hashes=None, leak_scanner=None, temperature=0.7):
        """evaluate(candidate_dir) -> results over the fixed dev tasks (same order every call).
        make_sandbox(candidate_dir) -> CandidateSandbox-like context manager. sample_tasks: a few dev tasks
        for the router smoke test. leak_scanner: LeakScanner over protected items (checked before evaluation)."""
        super().__init__(llm, thinking, retries, temperature)
        self.evaluate, self.make_sandbox = evaluate, make_sandbox
        self.sample_tasks, self.run_dir = sample_tasks, Path(run_dir)
        self.seen_hashes = set(seen_hashes or ())
        self.leak_scanner = leak_scanner

    # ---------- the cycle ----------

    def run_cycle(self, cycle_id, parent_dir, parent_results, tasks_by_id, history_notes):
        cycle_dir = self.run_dir / f"cycle_{cycle_id:03d}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        record = {"cycle": cycle_id, "parent": str(parent_dir), "status": "started", "checks": {}, "stages": {}}
        self.seed = cycle_id
        try:
            self._run(record, cycle_dir, Path(parent_dir), parent_results, tasks_by_id, history_notes)
        except StageFailed as e:
            record["status"] = "stage_failed"
            record["error"] = str(e)
        if "note" not in record:
            # Rejected attempts are history too, so the next cycle does not propose the same thing again.
            tried = record["stages"].get("hypothesize", {}).get("output", {})
            chosen = next((h["hypothesis"] for h in tried.get("hypotheses", [])
                           if h["hypothesis_id"] == tried.get("selected")), "no hypothesis reached")
            reason = {"duplicate": "it produced the same router as an earlier attempt",
                      "leak_violation": "its code contained protected test content",
                      "invalid_patch": "the code failed validation: " + "; ".join(
                          record["stages"].get("implement", {}).get("problems", []))[:200],
                      "stage_failed": "a stage produced no valid output"}.get(record["status"], record["status"])
            record["note"] = f"Cycle {cycle_id} (not evaluated): {chosen[:160]} -> rejected because {reason}."
        (cycle_dir / "record.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
        return record

    def _run(self, record, cycle_dir, parent_dir, parent_results, tasks_by_id, history_notes):
        feedback = observe(parent_results, tasks_by_id, seed=self.seed)
        router_src = (parent_dir / ROUTER).read_text()
        # Every earlier attempt, older ones shortened: with only the last five, the main run proposed
        # "raise the top-N limit" again in cycles 4, 8 and 13.
        older = [n[:200] for n in history_notes[:-5]]
        history = "\n".join(f"- {n}" for n in older + history_notes[-5:]) or "(none yet)"
        base = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"{ROUTER_CONTRACT}\n\nCurrent router:\n```python\n{router_src}```\n\n"
                                            f"Development feedback (JSON):\n{json.dumps(feedback, ensure_ascii=False)}"
                                            f"\n\nConclusions of earlier experiments:\n{history}"}]
        record["stages"]["observe"] = feedback

        # DIAGNOSE (the user's "寻找问题")
        diag, meta = self._structured(base + [{"role": "user", "content": (
            "DIAGNOSE: pick the single most valuable problem to work on now. Evidence must use the exact failure "
            "types, counts and example ids from the feedback; an example id belongs only to the failure type it "
            "is listed with. Do not return to an approach that an earlier experiment tried without a significant "
            "improvement unless you change it substantially.")}], schemas.DIAGNOSE)
        counts = feedback["failure_counts"]
        failing_ids = {}
        for ex in feedback["failure_examples"]:
            failing_ids.setdefault(ex["failure"], set()).add(ex["id"])
        evidence_ok = counts.get(diag["selected_problem"], 0) > 0 and all(
            e["count"] == counts.get(e["failure_type"], 0)
            and set(e["example_ids"]) <= failing_ids.get(e["failure_type"], set()) for e in diag["evidence"])
        record["stages"]["diagnose"] = {"output": diag, **meta}
        record["checks"]["diagnose_evidence_matches_records"] = evidence_ok

        # HYPOTHESIZE
        hyp, meta = self._structured(base + [
            {"role": "user", "content": f"Your diagnosis:\n{json.dumps(diag, ensure_ascii=False)}\n\nHYPOTHESIZE: "
                                        "propose exactly two different, testable hypotheses for fixing it by changing "
                                        "the router, each with a measurable prediction and the unchanged current "
                                        "router as control; then select one."}], schemas.HYPOTHESIZE)
        chosen = next(h for h in hyp["hypotheses"] if h["hypothesis_id"] == hyp["selected"])
        record["stages"]["hypothesize"] = {"output": hyp, **meta}
        record["checks"]["hypothesis_testable"] = bool(chosen["prediction"]["metric"] and chosen["control"].strip())

        # DESIGN
        design, meta = self._structured(base + [
            {"role": "user", "content": f"Selected hypothesis:\n{json.dumps(chosen, ensure_ascii=False)}\n\nDESIGN: "
                                        "write the experiment plan: concrete change steps to the router, the primary "
                                        "metric and when to roll back."}], schemas.DESIGN)
        record["stages"]["design"] = {"output": design, **meta}

        # IMPLEMENT + VALIDATE (basic validity), with bounded repair attempts
        child_dir = cycle_dir / "candidate"
        if child_dir.exists():
            shutil.rmtree(child_dir)
        shutil.copytree(parent_dir, child_dir)
        prompt = base + [{"role": "user", "content": (
            f"Plan:\n{json.dumps(design, ensure_ascii=False)}\n\nIMPLEMENT: write the complete new "
            f"{ROUTER} implementing this plan. Output the whole file in one ```python code block.")}]
        source, problems = None, []
        for attempt in range(self.retries + 1):
            text = self.llm.chat(prompt, max_tokens=3000, thinking=self.thinking, temperature=self.temperature,
                                 seed=self.seed + attempt)
            source = extract_python(text)
            problems = check_router_source(source)
            if not problems:
                (child_dir / ROUTER).write_text(source)
                problems = self._smoke_test(child_dir)
            if not problems:
                break
            prompt = prompt + [{"role": "assistant", "content": text},
                               {"role": "user", "content": "The router failed validation:\n- " + "\n- ".join(problems)
                                                           + "\nFix it and output the whole file again."}]
        record["stages"]["implement"] = {"attempts": attempt + 1, "problems": problems,
                                         "lines": len((source or "").splitlines())}
        record["checks"]["patch_valid"] = not problems
        if problems:
            record["status"] = "invalid_patch"
            return
        code_hash = normalized_hash(source)
        record["code_hash"] = code_hash
        if code_hash in self.seen_hashes:
            record["status"] = "duplicate"
            return
        self.seen_hashes.add(code_hash)

        if self.leak_scanner is not None:
            files = {str(p.relative_to(child_dir)): p.read_text(errors="replace")
                     for p in child_dir.rglob("*") if p.is_file() and p.suffix in (".py", ".jsonl", ".jinja", ".md")}
            scan = self.leak_scanner.scan(files)
            record["leak_scan"] = scan
            record["checks"]["leak_scan_clean"] = not scan["violations"]
            if scan["violations"]:
                record["status"] = "leak_violation"  # never evaluated; flagged in the report
                return

        # VALIDATE: paired comparison with the parent on the same development tasks
        child_results = self.evaluate(child_dir)
        by_id = {r["id"]: r for r in child_results}
        parent_vec = [int(r["correct"]) for r in parent_results]
        child_vec = [int(by_id[r["id"]]["correct"]) for r in parent_results]
        diff, p = paired_bootstrap(parent_vec, child_vec)
        child_feedback = observe(child_results, tasks_by_id)
        metrics = {"parent_accuracy": feedback["accuracy"], "child_accuracy": child_feedback["accuracy"],
                   "diff": round(diff, 4), "p_one_sided": round(p, 4),
                   "parent_failures": counts, "child_failures": child_feedback["failure_counts"]}
        record["metrics"] = metrics
        (cycle_dir / "dev_results.json").write_text(json.dumps(child_results, ensure_ascii=False))

        # REFLECT: facts are given, not asked
        reflect, meta = self._structured(base + [
            {"role": "user", "content": f"Hypothesis:\n{json.dumps(chosen, ensure_ascii=False)}\n\nMeasured result "
                                        f"(recorded by the controller):\n{json.dumps(metrics)}\n\nREFLECT: was the "
                                        "hypothesis supported? What is the limitation and the next direction?"}],
            schemas.REFLECT)
        record["stages"]["reflect"] = {"output": reflect, **meta}
        record["checks"]["improved_on_dev"] = diff > 0 and p < 0.05
        record["note"] = (f"Cycle {record['cycle']}: {chosen['hypothesis'][:160]} -> accuracy "
                          f"{metrics['parent_accuracy']:.3f} to {metrics['child_accuracy']:.3f} "
                          f"(p={metrics['p_one_sided']}); {reflect['reusable_finding'][:160]}")
        record["status"] = "evaluated"
        record["child_dir"] = str(child_dir)

    def _smoke_test(self, candidate_dir):
        problems = []
        with self.make_sandbox(candidate_dir) as sb:
            for t in self.sample_tasks:
                r = sb.select(task_text(t), t["functions"])
                if not r.get("ok"):
                    problems.append(f"on task {t['id']}: {r.get('error', 'failed')[:300]}")
                    break
        return problems
