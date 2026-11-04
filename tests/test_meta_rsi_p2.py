"""P2 of Meta-RSI (docs/meta-rsi-design.md section 7): sampling seeds, the evaluation cache's guard, the self_test
component, retargeted task sets, the improver's own configuration, the meta loop and the watchdog. Tests marked
docker need docker and the alter-judge image."""
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import httpx
import openai
import pytest

from rive.meta_rsi.bugfix import make_task, retarget
from rive.meta_rsi.components import DEFAULT_CONFIG, component_docs, make_agent
from rive.meta_rsi.config_improver import M0C, ConfigImprover, tracking_table
from rive.meta_rsi.meta_loop import MetaLoop, archive_table, best_version, choose_parent, meta_prompt
from rive.meta_rsi.reset_meta import EvaluationFailed, Evaluator, finish_test, reset_meta
from rive.runtime.workflow import run_workflow_tasks
from rive.sandbox.container import CandidateSandbox

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "candidate_template"
docker = pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available")
FENCE = "`" * 3


def agent(tmp_path, workflow):
    cand = tmp_path / "cand"
    shutil.copytree(TEMPLATE, cand)
    (cand / "agent" / "workflow.py").write_text(workflow)
    for p in [cand, *cand.rglob("*")]:
        p.chmod(0o755 if p.is_dir() else 0o644)
    return cand


class SeedLLM:
    """Records each call's seed; the call numbered `down` fails as if the server were not there."""

    def __init__(self, down=None):
        self.seeds, self.down = [], down

    def generate(self, messages, max_tokens, temperature, thinking, seed=0):
        self.seeds.append(seed)
        if len(self.seeds) == self.down:
            raise openai.APIConnectionError(request=httpx.Request("POST", "http://127.0.0.1:8000/v1"))
        return {"text": f"answer {seed}", "finish_reason": "stop", "prompt_tokens": 3, "completion_tokens": 3}


@docker
def test_each_call_of_a_task_samples_with_its_own_seed_and_server_failures_are_counted(tmp_path):
    wf = ("def solve(task, ctx):\n"
          "    out = []\n"
          "    for _ in range(3):\n"
          "        try:\n"
          "            out.append(ctx.llm([{'role': 'user', 'content': 'x'}], temperature=0.7))\n"
          "        except RuntimeError:\n"
          "            out.append('failed')\n"
          "    return {'answer': ' | '.join(out)}\n")
    task = {"id": "t", "domain": "code", "public": {"domain": "code", "prompt": "x"}, "item": {}}
    llm = SeedLLM()
    res = run_workflow_tasks([task], agent(tmp_path, wf), llm, workers=1, judge_fn=lambda t, a: (True, None, {}))
    assert llm.seeds == [0, 1, 2] and res[0]["answer"] == "answer 0 | answer 1 | answer 2"
    assert res[0]["server_errors"] == 0
    llm = SeedLLM()
    run_workflow_tasks([task], agent(tmp_path / "s", wf), llm, workers=1, judge_fn=lambda t, a: (True, None, {}),
                       seed_base=2000)
    assert llm.seeds == [2000, 2001, 2002]
    llm = SeedLLM(down=2)
    res = run_workflow_tasks([task], agent(tmp_path / "b", wf), llm, workers=1,
                             judge_fn=lambda t, a: (True, None, {}))
    assert res[0]["server_errors"] == 1 and "failed" in res[0]["answer"]


def test_an_evaluation_with_server_failures_is_not_kept(tmp_path):
    a = tmp_path / "a"
    shutil.copytree(TEMPLATE, a)
    outcomes = [[{"id": "x", "correct": False, "server_errors": 2}], [{"id": "x", "correct": True, "server_errors": 0}]]
    evaluate = Evaluator(lambda ts, d: outcomes.pop(0), tmp_path / "evals")
    with pytest.raises(EvaluationFailed):
        evaluate(a, "dev", [])
    assert list((tmp_path / "evals").iterdir()) == []
    assert evaluate(a, "dev", [])[0]["correct"] and evaluate(a, "dev", [])[0]["correct"]  # the second is kept
    assert [p.name for p in (tmp_path / "evals").iterdir()][0].endswith("_dev.json") and outcomes == []


class Levels:
    """Proposes agents of the given levels and records the context of each proposal."""
    name = "levels"

    def __init__(self, levels):
        self.levels, self.contexts = levels, []

    def start(self, run_dir):
        self.run_dir = Path(run_dir)

    def propose(self, child_id, parent_dir, results, tasks_by_id, notes, seed, context=None):
        self.contexts.append(context)
        child = self.run_dir / child_id
        shutil.copytree(parent_dir, child)
        (child / "agent" / "level.txt").write_text(str(self.levels[len(self.contexts) - 1]))
        return {"id": child_id, "status": "valid", "dir": str(child), "module": "config",
                "record": {"hypothesis": {"hypothesis": "h"}, "config": {"level": self.levels[len(self.contexts) - 1]}}}

    def usage(self):
        return {}


def level_score(agent_dir, task_list):
    level = int((Path(agent_dir) / "agent" / "level.txt").read_text())
    return [{"id": t["id"], "correct": i < level, "failure": None if i < level else "wrong_answer", "tokens": 10 * i}
            for i, t in enumerate(task_list)]


def test_reset_meta_gives_context_and_can_leave_the_test_pack_for_later(tmp_path):
    a0 = tmp_path / "a0"
    shutil.copytree(TEMPLATE, a0)
    (a0 / "agent" / "level.txt").write_text("4")
    packs = {k: [{"id": f"{k}{i}", "domain": "code"} for i in range(n)] for k, n in (("dev", 10), ("select", 20),
                                                                                      ("test", 30))}
    seen = []
    evaluate = Evaluator(lambda ts, d: seen.append(ts[0]["id"][:-1]) or level_score(d, ts), tmp_path / "evals")
    improver = Levels([9, 8])
    out = reset_meta(improver, a0, packs, tmp_path / "run", evaluate, rounds=2, margin=3, seed=1, test=False)
    assert "test" not in seen and "test_solved" not in out and out["select_solved"] == 9
    first, second = improver.contexts
    assert (first["round"], first["rounds"], first["history"]) == (1, 2, [])
    assert first["start"]["select_solved"] == 4 and first["start"]["pack_sizes"] == {"dev": 10, "select": 20,
                                                                                       "test": 30}
    assert second["history"][0]["kept"] and second["history"][0]["select_tokens"] == 95  # mean of 0, 10, ... 190
    done = finish_test(out, evaluate, packs)
    assert done["test_solved"] == 9 and seen.count("test") == 1


ITEM = {"id": 7, "prompt": "Write a function to add one.", "reference": "def inc(x):\n    return x + 1",
        "tests": ["assert inc(1) == 2", "assert inc(0) == 1"], "test_imports": []}


def test_a_task_keeps_its_buggy_code_in_another_setting():
    task = make_task(ITEM, {"code": "def inc(x):\n    return x", "mutations": [[3, "binop", None]],
                            "failing": [1]}, "nospec_fail_1")
    assert task["public"]["failing_test"] == "assert inc(0) == 1" and "Task:" not in task["public"]["prompt"]
    spec = retarget(task, "spec_nofail_1")
    assert spec["public"]["buggy_code"] == task["public"]["buggy_code"] and spec["public"]["failing_test"] is None
    assert spec["public"]["prompt"].startswith("Task: Write a function to add one.")
    assert "fails this test" not in spec["public"]["prompt"] and spec["id"] != task["id"]
    assert spec["item"]["tests"] == ITEM["tests"] and spec["item"]["setting"] == "spec_nofail_1"
    assert retarget(spec, "nospec_nofail_1")["public"]["statement"] == ""
    assert retarget(task, "nospec_fail_1")["public"] == task["public"]
    with pytest.raises(ValueError):
        retarget(task, "spec_nofail_2")  # one bug, not two


class JsonLLM:
    """Analysis as plain text; as JSON, the given objects in turn."""

    def __init__(self, objects):
        self.objects, self.prompts, self.usage = list(objects), [], {"completion_tokens": 0}

    def chat(self, messages, schema=None, **kw):
        self.prompts.append(messages)
        return "Some analysis." if schema is None else json.dumps(self.objects.pop(0))


def test_the_improver_shows_its_strategy_memory_table_and_round(tmp_path):
    retry = {**DEFAULT_CONFIG, "test_retry": 2}
    proposal = {"diagnosis": "d", "hypothesis": "h", "prediction": "p", "predicted_gain": 4, "keep_probability": 0.7,
                "config": {**retry, "review": True}}
    llm = JsonLLM([{**proposal, "keep_probability": 1.5}, proposal])  # the first forecast is out of range
    improver = ConfigImprover("x", lambda: llm, config={**M0C, "policy": "Change one field at a time.",
                                                        "memory": "test_retry helped on set A (+14).",
                                                        "tracking": True}, docs=component_docs(2048))
    improver.start(tmp_path / "run")
    a0 = make_agent(tmp_path / "a0", DEFAULT_CONFIG)
    results = [{"id": "d1", "domain": "code", "correct": False, "failure": "wrong_answer", "answer": "x"}]
    tasks = {"d1": {"id": "d1", "public": {"prompt": "fix f"}}}
    child = improver.propose("r1", a0, results, tasks, [], seed=1, context={
        "round": 1, "rounds": 4, "history": [],
        "start": {"dev_solved": 25, "select_solved": 40, "select_tokens": 500,
                  "pack_sizes": {"dev": 58, "select": 67}}})
    assert child["record"]["predicted_gain"] == 4 and child["record"]["keep_probability"] == 0.7
    assert child["record"]["stages"]["propose"]["attempts"] == 2  # the out-of-range forecast was sent back
    assert "forecast" in llm.prompts[0][1]["content"] and "of 67 selection tasks" in llm.prompts[0][1]["content"]
    history = [{"round": 1, "status": "valid", "config": retry, "select_solved": 54, "select_gain": 14, "kept": True,
                "dev_solved": 37, "select_tokens": 700}]
    context = {"round": 2, "rounds": 4, "history": history, "start": {
        "dev_solved": 25, "select_solved": 40, "select_tokens": 500, "pack_sizes": {"dev": 58, "select": 67}}}
    user = improver.prompt(retry, results, tasks, ["r1: kept"], 2, context)[1]["content"]
    assert "Change one field at a time." in user and "test_retry helped on set A" in user
    assert "| r1 | test_retry 0->2 | 54 | +14 | yes | 37 | 700 |" in user and "| start | - | 40 |" in user
    assert "This is change 2 of 4" in user and "share 2048 tokens" in user and "r1: kept" not in user
    m0c = ConfigImprover("m0c", lambda: llm)
    m0c.start(tmp_path / "run2")
    user = m0c.prompt(retry, results, tasks, ["r1: kept"], 2, context)[1]["content"]
    assert "- r1: kept" in user and "Performance table" not in user and "Your strategy" not in user


def test_the_table_marks_proposals_that_were_not_valid():
    context = {"start": {"dev_solved": 1, "select_solved": 2, "select_tokens": 3, "pack_sizes": {"dev": 5,
                                                                                                  "select": 6}},
               "history": [{"round": 1, "status": "stage_failed", "config": None}]}
    assert "| r1 | no valid proposal (stage_failed) |" in tracking_table(DEFAULT_CONFIG, context)


def version(vid, imp, parent=None, children=(), config=None):
    return {"id": vid, "parent": parent, "generation": int(vid[1:]), "config": config or {**M0C, "policy": vid},
            "children": list(children), "imp": imp}


def test_parents_are_drawn_by_score_and_fewer_children():
    archive = [version("v0", {"a_s1": 0}), version("v1", {"a_s1": 20}, "v0"), version("v2", {"a_s1": 20}, "v0",
                                                                                         ["v3", "v4", "v5"])]
    rng = random.Random(0)
    drawn = [choose_parent(archive, rng)["id"] for _ in range(2000)]
    assert drawn.count("v1") > 2.5 * drawn.count("v2") and drawn.count("v1") > 3 * drawn.count("v0")
    assert best_version(archive)["id"] == "v1"
    table = archive_table(archive, archive[1], tracking=False)
    assert "| v1 | v0 | policy " in table and "v2" not in table
    assert "v2" in archive_table(archive, archive[1], tracking=True)


def test_the_meta_loop_grows_the_archive_and_resumes(tmp_path):
    scored = []

    def score_run(vid, config, name, seed):
        scored.append((vid, name, seed))
        return {"set": name, "seed": seed, "imp": len(config["policy"]), "start_config": DEFAULT_CONFIG,
                "rounds": [{"round": 1, "status": "valid", "config": {**DEFAULT_CONFIG, "review": True},
                            "hypothesis": "h", "select_solved": 44, "select_gain": 4, "kept": True}]}
    better = {**M0C, "policy": "Read the table first.", "tracking": True}
    llm = JsonLLM([{"diagnosis": "d", "hypothesis": "h", "prediction": "p", "improver": M0C},  # refused: exists
                   {"diagnosis": "d", "hypothesis": "h", "prediction": "p", "improver": better}])
    loop = MetaLoop(tmp_path / "meta", score_run, [("a", 1), ("b", 1)], lambda: llm)
    archive = loop.run(M0C, generations=1)
    assert [v["id"] for v in archive] == ["v0", "v1"] and archive[1]["config"] == better
    assert archive[0]["children"] == ["v1"] and archive[1]["imp"] == {"a_s1": 21, "b_s1": 21}
    prompt = llm.prompts[0][1]["content"]
    assert "Run a seed 1: imp@4 +0" in prompt and "r1: review false->true" in prompt and "imp@4" in prompt
    # Stopped before v1 was scored, it is scored when the loop starts again, and nothing is proposed twice.
    archive[1]["imp"]["b_s1"] = None
    loop.save(archive)
    llm.objects = []
    scored.clear()
    loop.run(M0C, generations=1)
    assert ("v1", "b", 1) in scored and ("v1", "a", 1) not in scored
    assert json.loads((tmp_path / "meta" / "archive.json").read_text())[1]["imp"]["b_s1"] == 21


def test_the_meta_prompt_shows_the_parents_memory_and_strategy():
    parent = version("v1", {"a_s1": 3}, "v0", config={**M0C, "memory": "self_test helped (+9).",
                                                       "meta_policy": "Keep what worked."})
    text = meta_prompt(parent, [version("v0", {"a_s1": 0}), parent], [])[1]["content"]
    assert "Your memory:\nself_test helped (+9)." in text and "Keep what worked." in text
    assert "What the agent's settings do" in text and "self_test (true/false)" in text


def run_workflow(tmp_path, config, task, replies):
    cand = make_agent(tmp_path / "agent", config)
    for p in [cand, *cand.rglob("*")]:
        p.chmod(0o755 if p.is_dir() else 0o644)
    sent = []

    def on_llm(req):
        sent.append(req)
        return {"ok": True, "text": replies[min(len(sent), len(replies)) - 1]}
    with CandidateSandbox(cand, memory="768m") as sb:
        done = sb.solve(task, 60, on_llm)
    assert done["ok"], done
    return done, sent


@docker
def test_self_test_writes_asserts_and_retries_against_them(tmp_path):
    task = {"domain": "code", "source": "t1_mbpp", "prompt": "Task: add one\n\nfix inc", "failing_test": None,
            "statement": "add one", "buggy_code": "def inc(x):\n    return x"}
    asserts = f"{FENCE}python\nassert inc(1) == 2\nassert inc(\nnot an assert\n{FENCE}"
    wrong = f"{FENCE}python\ndef inc(x):\n    return x\n{FENCE}"
    right = f"{FENCE}python\ndef inc(x):\n    return x + 1\n{FENCE}"
    done, sent = run_workflow(tmp_path, {**DEFAULT_CONFIG, "self_test": True, "test_retry": 2,
                                         "feedback": "stderr_and_test"}, task, [asserts, wrong, right])
    assert "x + 1" in done["answer"] and len(sent) == 3
    first = sent[0]["messages"][-1]["content"]
    assert first.startswith("Task: add one") and "Do not fix it yet" in first and "def inc(x)" in first
    retry = sent[2]["messages"][-1]["content"]
    assert "with these tests" in retry and "The tests:\nassert inc(1) == 2" in retry and "not an assert" not in retry
    # With a failing test shown, self_test does nothing; without self_test, no test means no retry.
    done, sent = run_workflow(tmp_path / "shown", {**DEFAULT_CONFIG, "self_test": True, "test_retry": 2},
                              {**task, "failing_test": "assert inc(1) == 2"}, [wrong, right])
    assert len(sent) == 2 and "with the failing test" in sent[1]["messages"][-1]["content"]
    done, sent = run_workflow(tmp_path / "none", {**DEFAULT_CONFIG, "test_retry": 2}, task, [wrong])
    assert len(sent) == 1


def test_gepa_knobs_are_read_back_within_the_schema():
    from rive.meta_rsi.gepa_baseline import to_candidate, to_config
    config, problems = to_config(to_candidate(DEFAULT_CONFIG))
    assert config == DEFAULT_CONFIG and problems == []
    knobs = json.dumps({"attempts": 9, "temperature": 0.6, "select": "best", "review": True, "self_test": "yes"})
    config, problems = to_config({"system_prompt": "s", "instruction": "i", "knobs": knobs})
    assert (config["attempts"], config["temperature"], config["review"]) == (5, 0.5, True)
    assert config["select"] == "first" and config["self_test"] is False and len(problems) == 2
    config, problems = to_config({"system_prompt": "", "instruction": "", "knobs": "not json"})
    assert config == DEFAULT_CONFIG and "not a JSON object" in problems[0]


class KnobReflector:
    """GEPA's reflection model: new knobs with retries for the knobs component, a sentence for the texts."""

    def __init__(self):
        self.usage, self.prompts = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}, []

    def chat(self, messages, **kw):
        self.prompts.append(messages[-1]["content"])
        if "knobs" in messages[-1]["content"]:
            return f'{FENCE}\n{{"test_retry": 2, "attempts": 9}}\n{FENCE}'
        return f"{FENCE}\nFix the bug carefully.\n{FENCE}"


def test_gepa_runs_on_the_component_agent_and_keeps_the_better_knobs(tmp_path):
    from rive.meta_rsi.components import read_config
    from rive.meta_rsi.gepa_baseline import run_gepa
    packs = {"dev": [{"id": f"d{i}", "public": {"prompt": f"fix {i}"}} for i in range(6)],
             "select": [{"id": f"s{i}", "public": {"prompt": f"fix {i}"}} for i in range(6)]}
    asked = []

    def evaluate(agent, pack, tasks):  # an agent that retries solves every task; otherwise every third
        asked.append(pack)
        retry = read_config(agent)["test_retry"] >= 2
        return [{"id": t["id"], "correct": retry or i % 3 == 0, "failure": None, "answer": "x", "tokens": 5,
                 "llm_calls": 1, "detail": {"tests": ["pass"]}} for i, t in enumerate(tasks)]
    llm = KnobReflector()
    out = run_gepa(evaluate, packs, tmp_path / "run", llm, budget=40, seed=1, minibatch=3, docs="test_retry (0-3)")
    assert out["config"]["test_retry"] == 2 and out["config"]["attempts"] == 5 and out["select_solved"] == 6
    assert read_config(out["final_agent"])["test_retry"] == 2
    assert "select" in asked and any(p.startswith("gepa_") for p in asked)
    knob_prompt = next(p for p in llm.prompts if "knobs" in p)
    assert "test_retry (0-3)" in knob_prompt and "hidden tests" in knob_prompt


def test_the_event_log_only_appends_and_survives_a_cut_off_line(tmp_path):
    from rive.infra.events import EventLog, read_events
    log = EventLog(tmp_path / "events.jsonl", "abc")
    log.log("run_start", run="r1")
    log.log("run_done", run="r1")
    with (tmp_path / "events.jsonl").open("a") as f:
        f.write('{"seq": 3, "event": "cut')  # a crash in the middle of a write
    again = EventLog(tmp_path / "events.jsonl", "abc")
    again.log("run_start", run="r2")
    events = read_events(tmp_path / "events.jsonl")
    assert [e["event"] for e in events] == ["run_start", "run_done", "run_start"]
    assert events[-1]["seq"] == 4 and events[0]["code"] == "abc"


def test_the_manifest_records_what_it_can(tmp_path):
    from rive.infra.manifest import manifest
    data = tmp_path / "pack.json"
    data.write_text("[]")
    m = manifest(tmp_path, tmp_path / "no-model", [data], images=())
    assert m["model"] == {"dir": str(tmp_path / "no-model"), "present": False}
    assert m["data"][str(data)] == "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
    assert m["git_commit"] is None and m["packages"]["jsonschema"]


def test_the_watchdog_restarts_a_failed_chain(tmp_path):
    marker, beat = tmp_path / "ran_once", tmp_path / "beat"
    chain = tmp_path / "chain.sh"
    chain.write_text(f"if [ -e {marker} ]; then exit 0; fi\ntouch {marker}\nexit 4\n")
    log = tmp_path / "watchdog.log"
    done = subprocess.run([sys.executable, str(ROOT / "scripts" / "watchdog.py"), "--heartbeat", str(beat),
                           "--every", "0.2", "--tries", "2", "--no-cleanup", "--log", str(log), "--", "bash",
                           str(chain)], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stdout + done.stderr
    text = log.read_text()
    assert "restart 1 of 2: the chain exited with 4" in text and "chain finished" in text


def test_the_watchdog_stops_a_stalled_chain_and_gives_up(tmp_path):
    log = tmp_path / "watchdog.log"
    done = subprocess.run([sys.executable, str(ROOT / "scripts" / "watchdog.py"), "--heartbeat",
                           str(tmp_path / "beat"), "--every", "0.2", "--stall-minutes", "0.01", "--tries", "1",
                           "--no-cleanup", "--log", str(log), "--", "sleep", "30"],
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 1
    text = log.read_text()
    assert "restart 1 of 1: no progress" in text and "giving up after 1 restarts: no progress" in text
