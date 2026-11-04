"""P2 of Meta-RSI, the meta level (docs/meta-rsi-design.md section 7). Steps, each resumable:

  python scripts/meta_rsi_p2.py check      # the server samples differently with different seeds
  python scripts/meta_rsi_p2.py packs      # the candidate task sets, from the calibration packs (same buggy code)
  python scripts/meta_rsi_p2.py scan       # P2.0: A0 and hand-picked configurations, development and selection only
  python scripts/meta_rsi_p2.py m0c        # P2.0: m0c, seeds 1-2, on the task sets with room; no test pack
  python scripts/meta_rsi_p2.py choose     # P2.0: training and held-out task sets by the rules written beforehand
  python scripts/meta_rsi_p2.py passk      # after P2.0: selection tasks solved by any of 8 samples of A0
  python scripts/meta_rsi_p2.py repeat     # after P2.0b: the choice's selection evaluations, made again
  python scripts/meta_rsi_p2.py r002       # R-002's pilot: m0c, Reflexion-style, M_hand and GEPA, 8 seeds
  python scripts/meta_rsi_p2.py control    # P2.1: M_hand against m0c on the held-out set, 8 seeds
  python scripts/meta_rsi_p2.py loop       # P2.2: the meta loop on the training sets
  python scripts/meta_rsi_p2.py transfer   # P2.3: the best version against m0c on the held-out set, 8 seeds
  python scripts/meta_rsi_p2.py report

A run is kept per improver configuration, task set and seed, so the same run serves every step that needs it (m0c's
runs from P2.0 are its first seeds in P2.1 and its scores as version v0 in P2.2). Evaluations are kept per agent and
pack. The heartbeat file is touched as tasks finish, for scripts/watchdog.py.
"""
import argparse
import hashlib
import json
import math
import os
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from rive.llm import LLM  # noqa: E402
from rive.meta_rsi.bugfix import judge, retarget  # noqa: E402
from rive.meta_rsi.components import DEFAULT_CONFIG, component_docs, make_agent  # noqa: E402
from rive.meta_rsi.config_improver import M0C, ConfigImprover, config_key  # noqa: E402
from rive.meta_rsi.meta_loop import MetaLoop, best_version, score  # noqa: E402
from rive.meta_rsi.reset_meta import (EvaluationFailed, Evaluator, finish_test, reset_meta, solved,  # noqa: E402
                                        write_json)
from rive.infra.events import EventLog, code_hash  # noqa: E402
from rive.infra.manifest import write_manifest  # noqa: E402
from rive.runtime.workflow import run_workflow_tasks  # noqa: E402
from meta_rsi_calibrate import sign_flip  # noqa: E402

CODE = [*(ROOT / "rive" / "meta_rsi").glob("*.py"), ROOT / "rive" / "runtime" / "workflow.py",
        Path(__file__).resolve()]

# name: (setting for one bug, setting for two bugs, token budget per task). The packs are the calibration packs
# (failing test shown, no statement), retargeted: the buggy code stays, what the agent is shown changes.
SETS = {"spec_nofail": ("spec_nofail_1", "spec_nofail_2", 8192),
        "nofail": ("nospec_nofail_1", "nospec_nofail_2", 8192),
        "fail_b2048": ("nospec_fail_1", "nospec_fail_2", 2048)}
SHOWS_TEST = {"spec_nofail": False, "nofail": False, "fail_b2048": True}
CODE_ONLY = "Return only the corrected code, with no explanation."
# name: (changes to A0, kind of lever). Configurations that need a shown test are not scanned where none is shown
# (they would be A0), and self_test ones not where one is (no effect there).
SCAN = {
    "a0": ({}, None),
    "review": ({"review": True}, "review"),
    "sample4_majority": ({"attempts": 4, "temperature": 0.7, "select": "majority", "max_tokens": 1024}, "sampling"),
    "code_only": ({"instruction": CODE_ONLY}, "brevity"),
    "retry2": ({"test_retry": 2}, "test"),
    "retry2_review": ({"test_retry": 2, "review": True}, "test"),
    "sample4_pass": ({"attempts": 4, "temperature": 0.7, "select": "passes_test", "max_tokens": 1024}, "test"),
    "code_only_retry3_1024": ({"instruction": CODE_ONLY, "test_retry": 3, "max_tokens": 1024}, "test"),
    "selftest_retry2": ({"self_test": True, "test_retry": 2}, "self_test"),
    "selftest_retry2_review": ({"self_test": True, "test_retry": 2, "review": True}, "self_test"),
    "selftest_sample4_pass": ({"self_test": True, "attempts": 4, "temperature": 0.7, "select": "passes_test",
                               "max_tokens": 1024}, "self_test"),
    # P2.0b (docs/rive-design.md section 6.7): choosing among samples by the results of model-written calls.
    "agree5": ({"attempts": 5, "temperature": 0.7, "select": "agreement", "max_tokens": 1024}, "agreement"),
    "agree5_t10": ({"attempts": 5, "temperature": 1.0, "select": "agreement", "max_tokens": 1024}, "agreement"),
    "agree5_code_only": ({"instruction": CODE_ONLY, "attempts": 5, "temperature": 0.7, "select": "agreement",
                          "max_tokens": 1024}, "agreement"),
}
ROOM, UNSATURATED = 8, 4  # P2.0's rules (section 7.4), in selection tasks
PASSK, PASSK_TEMPERATURE = 8, 0.7
REPEATS = 2
P20_SEEDS, SEEDS, GENERATIONS = (1, 2), tuple(range(1, 9)), 6
# P2.1's hand-written improver: general rules only, nothing about which settings win on which task set.
M_HAND = {"policy": (
    "1. First look at a task example in the feedback: does the task show a failing test, and does it show the "
    "problem statement? passes_test and test_retry need a test: without a shown failing test they do nothing "
    "unless self_test writes one, and self_test can only write good asserts when the task says what the function "
    "should do.\n"
    "2. Start with the setting that addresses the most common failure for this kind of task. Change one or two "
    "fields per proposal, so the table shows what each change did.\n"
    "3. Read the performance table: build on the best kept configuration, and do not repeat a direction that lost "
    "selection tasks or gained fewer than 3.\n"
    "4. Watch the cost: all calls of a task share the token budget. If answers are cut off or the tokens per task "
    "are near the budget, lower max_tokens or ask for code only before adding calls.\n"
    "5. Several attempts help only with a temperature above 0 and a way to choose among them (a test or "
    "majority)."), "meta_policy": "", "memory": "", "tracking": True, "temperature": 0.7}
# R-002's Reflexion-style fixed improver: verbal reflection on its own earlier changes, nothing about the components.
REFLEXION = {"policy": (
    "Reflect before you propose. For every earlier change in the performance table, say in one sentence why it "
    "gained or lost selection tasks, judged from the feedback, and what that teaches. Do not repeat a change that "
    "lost tasks or gained fewer than 3, and build on the configuration kept so far. Base the new configuration on "
    "these lessons and say in the diagnosis which lesson it uses."),
    "meta_policy": "", "memory": "", "tracking": True, "temperature": 0.7}
GEPA_BUDGET = 400  # evaluations, about one m0c run (451 in P2.0b); GEPA checks it only between iterations


class P2:
    def __init__(self, args):
        self.args, self.out = args, Path(args.out_dir)
        self.heartbeat = self.out / "heartbeat"
        self.out.mkdir(parents=True, exist_ok=True)
        self.beat()
        self.events = EventLog(self.out / "events.jsonl", code_hash(CODE))
        self.events.log("step_start", step=args.step)
        self.solver = LLM(model=args.model)
        self.a0 = make_agent(self.out / "agents" / "a0", DEFAULT_CONFIG)
        self._evaluators, self._packs = {}, {}

    def beat(self):
        self.heartbeat.touch()

    def packs(self, name):
        if name not in self._packs:
            self._packs[name] = json.loads((self.out / "packs" / f"{name}.json").read_text())
        return self._packs[name]

    def evaluate(self, name):
        """The evaluator of task set `name`: its token budget, its packs kept apart from other sets'."""
        if name not in self._evaluators:
            budget = SETS[name][2]

            def run(tasks, d):
                self.events.log("evaluation_start", agent=d.name, set=name, tasks=len(tasks))
                try:
                    results = run_workflow_tasks(tasks, d, self.solver, workers=self.args.workers,
                                                 token_budget=budget, judge_fn=judge, progress=self.beat)
                except Exception as e:
                    self.events.log("evaluation_failed", agent=d.name, set=name, error=str(e)[:300])
                    raise
                self.events.log("evaluation_done", agent=d.name, set=name, solved=solved(results),
                                server_errors=sum(r.get("server_errors", 0) for r in results))
                return results
            base = Evaluator(run, self.out / "evals")
            self._evaluators[name] = lambda agent, pack, tasks: base(agent, f"{name}_{pack}", tasks)
        return self._evaluators[name]

    def counted(self, name, tally):
        """Set `name`'s evaluator, adding every evaluation a run asks for to `tally`, kept or not: the run's budget
        in rollouts and tokens (docs/rive-design.md section 6.2), the same however much was already kept."""
        evaluate = self.evaluate(name)

        def run(agent, pack, tasks):
            results = evaluate(agent, pack, tasks)
            if pack != "test":  # the final test score measures the run; it is not part of its search
                tally["rollouts"] += len(results)
                tally["completion_tokens"] += sum(r.get("tokens", 0) for r in results)
                tally["prompt_tokens"] += sum(r.get("prompt_tokens", 0) for r in results)
            return results
        return run

    def scanned(self, name):
        """{configuration name: (development solved, selection solved, mean tokens)} for the configurations
        scanned on set `name`."""
        out = {}
        for cname, (change, kind) in SCAN.items():
            if kind == "test" and not SHOWS_TEST[name] or kind == "self_test" and SHOWS_TEST[name]:
                continue
            agent = make_agent(self.out / "scan" / cname, {**DEFAULT_CONFIG, **change})
            dev = self.evaluate(name)(agent, "dev", self.packs(name)["dev"])
            sel = self.evaluate(name)(agent, "select", self.packs(name)["select"])
            out[cname] = (solved(dev), solved(sel), round(statistics.mean(r["tokens"] for r in dev + sel)))
        return out

    def a0_test(self, name):
        return solved(self.evaluate(name)(self.a0, "test", self.packs(name)["test"]))

    def run(self, label, config, name, seed, test=True):
        """One RESET-META run of the improver set by `config` on task set `name`, kept by configuration, set and
        seed. test=True adds the final agent's test score and imp (its gain over A0 on the test pack)."""
        key = hashlib.sha256(config_key(config).encode()).hexdigest()[:10]
        path = self.out / "runs" / f"{key}_{name}_s{seed}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            result = json.loads(path.read_text())
        else:
            self.events.log("run_start", run=path.stem, improver=label, set=name, seed=seed)
            improver = ConfigImprover(label, lambda: LLM(model=self.args.model), config=config,
                                      docs=component_docs(SETS[name][2]))
            tally = {"rollouts": 0, "completion_tokens": 0, "prompt_tokens": 0}
            result = reset_meta(improver, self.a0, self.packs(name), path.with_suffix(""), self.counted(name, tally),
                                rounds=self.args.rounds, margin=self.args.margin, seed=seed, test=False)
            usage = result["improver_usage"]
            result.update(config=config, set=name, start_config=DEFAULT_CONFIG,
                          budget={**tally, "improver_completion_tokens": usage.get("completion_tokens", 0),
                                  "improver_prompt_tokens": usage.get("prompt_tokens", 0),
                                  "improver_calls": usage.get("calls", 0)})
            write_json(path, result)
            self.events.log("run_done", run=path.stem, select_solved=result["select_solved"],
                            kept=sum(bool(s.get("kept")) for s in result["rounds"]))
            print(f"run {label} on {name} seed {seed}: rounds "
                  f"{[(s['status'], s.get('select_gain'), s.get('kept')) for s in result['rounds']]}", flush=True)
        if test and "test_solved" not in result:
            result = finish_test(result, self.evaluate(name), self.packs(name))
            result["imp"] = result["test_solved"] - self.a0_test(name)
            write_json(path, result)
        return result

    def gepa(self, label, name, seed, budget, minibatch=3, test=True, folder="runs"):
        """One GEPA run (rive/meta_rsi/gepa_baseline.py) on set `name`, kept by label, set and seed, with the same
        budget accounting as an improver's run."""
        from rive.meta_rsi.gepa_baseline import run_gepa
        path = self.out / folder / f"{label}_{name}_s{seed}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            result = json.loads(path.read_text())
        else:
            self.events.log("run_start", run=path.stem, improver=label, set=name, seed=seed)
            tally = {"rollouts": 0, "completion_tokens": 0, "prompt_tokens": 0}
            llm = LLM(model=self.args.model)
            out = run_gepa(self.counted(name, tally), self.packs(name), path.with_suffix(""), llm, budget, seed=seed,
                           minibatch=minibatch, docs=component_docs(SETS[name][2]))
            result = {"improver": label, "set": name, "seed": seed, **out, "rounds": [],
                      "budget": {**tally, "improver_completion_tokens": llm.usage["completion_tokens"],
                                 "improver_prompt_tokens": llm.usage["prompt_tokens"],
                                 "improver_calls": llm.usage["calls"]},
                      "improver_usage": dict(llm.usage), "start_config": DEFAULT_CONFIG}
            write_json(path, result)
            self.events.log("run_done", run=path.stem, select_solved=result["select_solved"])
            print(f"gepa {label} on {name} seed {seed}: select {result['select_solved']}, "
                  f"{len(result['candidates'])} candidates, {result['metric_calls']} metric calls", flush=True)
        if test and "test_solved" not in result:
            result = finish_test(result, self.evaluate(name), self.packs(name))
            result["imp"] = result["test_solved"] - self.a0_test(name)
            write_json(path, result)
        return result

    def choice(self):
        return json.loads((self.out / "choice.json").read_text())


def check_step(p2):
    """With one seed the server repeats its sample; with another it does not (the fix in run_workflow_tasks)."""
    messages = [{"role": "user", "content": "Write four lines about the sea."}]
    a, b, c = (p2.solver.generate(messages, max_tokens=80, temperature=0.7, seed=s)["text"] for s in (0, 0, 1))
    seeds = {"same_seed_same_sample": a == b, "other_seed_other_sample": a != c}
    print(json.dumps(seeds), flush=True)
    data = [Path(p2.args.packs), *sorted((p2.out / "packs").glob("*.json"))]
    write_manifest(p2.out / "manifest.json", repo=ROOT, model_dir="~/alter/models/Qwen3.5-4B", data_files=data,
                   extra={"seed_check": seeds, "args": vars(p2.args)})
    if a == c:
        sys.exit("different seeds gave the same sample")


def packs_step(p2):
    base = json.loads(Path(p2.args.packs).read_text())
    for name, (one, two, _) in SETS.items():
        path = p2.out / "packs" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        packs = {k: [retarget(t, one if len(t["item"]["mutations"]) == 1 else two) for t in v]
                 for k, v in base.items()}
        write_json(path, packs)
        print(f"packs {name}: {({k: len(v) for k, v in packs.items()})}", flush=True)


def scan_step(p2):
    for name in SETS:
        for cname, (dev, sel, tokens) in p2.scanned(name).items():
            print(f"scan {name} {cname}: dev {dev}, select {sel}, mean tokens {tokens}", flush=True)


def room(p2, name):
    scan = p2.scanned(name)
    best = max((c for c in scan if c != "a0"), key=lambda c: scan[c][1])
    return best, scan[best][1] - scan["a0"][1]


def m0c_step(p2):
    for name in SETS:
        best, gain = room(p2, name)
        if gain < ROOM:
            print(f"m0c {name}: skipped, the best configuration ({best}) gains only {gain:+d} selection tasks",
                  flush=True)
            continue
        for seed in P20_SEEDS:
            p2.run("m0c", M0C, name, seed, test=False)


def choose_step(p2):
    """P2.0's rules (section 7.4): room, not saturated by m0c, a different kind of best configuration."""
    sets = {}
    for name in SETS:
        best, gain = room(p2, name)
        scan = p2.scanned(name)
        entry = {"a0_select": scan["a0"][1], "best": best, "best_kind": SCAN[best][1], "best_select": scan[best][1],
                 "room": gain >= ROOM}
        if entry["room"]:
            runs = [p2.run("m0c", M0C, name, s, test=False) for s in P20_SEEDS]
            finals = [r["select_solved"] for r in runs]
            kept = [sum(bool(s.get("kept")) for s in r["rounds"]) for r in runs]
            entry.update(m0c_select=finals, m0c_kept=kept,
                         gap=scan[best][1] - statistics.mean(finals),
                         unsaturated=scan[best][1] - statistics.mean(finals) >= UNSATURATED or min(kept) == 0)
        sets[name] = entry
    ok = sorted((n for n, e in sets.items() if e["room"] and e["unsaturated"]), key=lambda n: -sets[n]["gap"])
    held_out = training = None
    for h in ok:  # the held-out set: the largest gap above m0c among those with a kind no training set shares
        rest = [n for n in ok if n != h and sets[n]["best_kind"] != sets[h]["best_kind"]]
        if rest:
            held_out, training = h, rest[:2]
            break
    if len(ok) == 1:  # the held-out set alone; training sets are chosen before P2.2 (docs/rive-design.md 6.7)
        held_out = ok[0]
    choice ={"sets": sets, "held_out": held_out, "training": training}
    write_json(p2.out / "choice.json", choice)
    print(json.dumps(choice, indent=1), flush=True)


def gepa_smoke_step(p2):
    """GEPA end to end with the local model on a small budget (no test pack): does it run and propose?"""
    r = p2.gepa("gepa_smoke", "spec_nofail", 1, budget=p2.args.smoke_budget, test=False, folder="smoke")
    print(json.dumps({k: r[k] for k in ("select_solved", "metric_calls", "reflection_calls", "config",
                                        "config_problems", "budget")}, indent=1), flush=True)


def passk_step(p2):
    """Selection tasks solved by any of PASSK samples of A0 at temperature PASSK_TEMPERATURE, each sample with its
    own seeds, judged by the hidden tests: the room a perfect choice among samples would have, set against the best
    scanned configuration (docs/rive-design.md section 6.6)."""
    agent = make_agent(p2.out / "scan" / "a0_sampled", {**DEFAULT_CONFIG, "temperature": PASSK_TEMPERATURE})
    scan = json.loads(Path(p2.args.out).read_text())["scan"]
    out = {}
    for name in SETS:
        tasks, samples = p2.packs(name)["select"], []
        for i in range(PASSK):
            path = p2.out / "passk" / f"{name}_s{i + 1}.json"
            if not path.exists():
                results = run_workflow_tasks(tasks, agent, p2.solver, workers=p2.args.workers,
                                             token_budget=SETS[name][2], judge_fn=judge, progress=p2.beat,
                                             seed_base=1000 * (i + 1))
                if any(r.get("server_errors", 0) for r in results):
                    raise EvaluationFailed(f"the model server failed calls in sample {i + 1} of {name}")
                path.parent.mkdir(parents=True, exist_ok=True)
                write_json(path, results)
            samples.append([r["correct"] for r in json.loads(path.read_text())])
        out[name] = {"pass1_mean": statistics.mean(sum(s) for s in samples),
                     "pass_at": {k: sum(any(s[t] for s in samples[:k]) for t in range(len(tasks)))
                                 for k in (1, 2, 4, PASSK)},
                     "a0": scan[name]["a0"][1], "best_scanned": max(v[1] for v in scan[name].values())}
        print(name, out[name], flush=True)
    write_json(p2.out / "passk.json", out)


def repeat_step(p2):
    """On each set with room, the selection evaluations behind the choice (A0, the best scanned configuration, m0c's
    final agents) made REPEATS more times past the cache with other seeds, and P2.0's rules on the means: the same
    agent's score moves between runs (docs/rive-design.md section 6.8)."""
    out = {}
    for name, entry in p2.choice()["sets"].items():
        if not entry["room"]:
            continue
        best = entry["best"]
        agents = {"a0": p2.out / "scan" / "a0", best: p2.out / "scan" / best}
        agents.update({f"m0c_s{s}": Path(p2.run("m0c", M0C, name, s, test=False)["final_agent"])
                       for s in P20_SEEDS})
        tasks, scores = p2.packs(name)["select"], {}
        for label, agent in agents.items():
            scores[label] = [solved(p2.evaluate(name)(agent, "select", tasks))]
            for i in range(REPEATS):
                path = p2.out / "repeat" / f"{name}_{label}_r{i + 1}.json"
                if not path.exists():
                    results = run_workflow_tasks(tasks, agent, p2.solver, workers=p2.args.workers,
                                                 token_budget=SETS[name][2], judge_fn=judge, progress=p2.beat,
                                                 seed_base=100000 * (i + 1))
                    if any(r.get("server_errors", 0) for r in results):
                        raise EvaluationFailed(f"the model server failed calls in repeat {i + 1} of {label}")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    write_json(path, results)
                scores[label].append(solved(json.loads(path.read_text())))
        mean = {k: statistics.mean(v) for k, v in scores.items()}
        m0c = statistics.mean(mean[f"m0c_s{s}"] for s in P20_SEEDS)
        out[name] = {"scores": scores, "mean": mean, "room": mean[best] - mean["a0"], "gap": mean[best] - m0c,
                     "room_ok": mean[best] - mean["a0"] >= ROOM,
                     "unsaturated": mean[best] - m0c >= UNSATURATED or min(entry["m0c_kept"]) == 0}
        print(name, json.dumps(out[name]), flush=True)
    write_json(p2.out / "repeat.json", out)


def bootstrap(values, n=10000):
    """95% interval of the mean of `values` (one per seed), the seeds resampled."""
    rng = random.Random(0)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(n))
    return [means[int(0.025 * n)], means[int(0.975 * n) - 1]]


def r002_step(p2):
    """R-002's pilot on the held-out set (docs/rive-design.md section 6.9): m0c, the Reflexion-style improver, M_hand
    and GEPA from A0, SEEDS each, paired by seed. Seed by seed, so a stopped run leaves the groups even."""
    name = p2.choice()["held_out"]
    groups = {"m0c": lambda s: p2.run("m0c", M0C, name, s),
              "reflexion": lambda s: p2.run("reflexion", REFLEXION, name, s),
              "m_hand": lambda s: p2.run("m_hand", M_HAND, name, s),
              "gepa": lambda s: p2.gepa("gepa", name, s, GEPA_BUDGET)}
    runs = {g: {} for g in groups}
    for s in SEEDS:
        for g, run in groups.items():
            runs[g][s] = run(s)
            print(f"r002 {g} seed {s}: select {runs[g][s]['select_solved']}, imp {runs[g][s]['imp']}", flush=True)
    out = {"set": name, "a0_test": p2.a0_test(name), "seeds": list(SEEDS), "groups": {}, "vs_m0c": {}}
    for g, by in runs.items():
        imp = [by[s]["imp"] for s in SEEDS]
        out["groups"][g] = {"imp": imp, "mean_imp": statistics.mean(imp), "ci": bootstrap(imp),
                            "p_vs_a0": sign_flip(imp), "select": [by[s]["select_solved"] for s in SEEDS],
                            "rollouts": [by[s]["budget"]["rollouts"] for s in SEEDS],
                            "completion_tokens": [by[s]["budget"]["completion_tokens"] for s in SEEDS],
                            "improver_calls": [by[s]["budget"]["improver_calls"] for s in SEEDS]}
    for g in ("reflexion", "m_hand", "gepa"):
        d = [runs[g][s]["imp"] - runs["m0c"][s]["imp"] for s in SEEDS]
        sd = statistics.stdev(d)
        out["vs_m0c"][g] = {"diffs": d, "mean_diff": statistics.mean(d), "ci": bootstrap(d), "p": sign_flip(d),
                            "sd": sd, "detectable": 2.8 * sd / len(d) ** 0.5,
                            "seeds_for_3": math.ceil((2.8 * sd / 3) ** 2) if sd else 1,
                            "passed": statistics.mean(d) > 0 and sign_flip(d) < 0.05}
    write_json(p2.out / "r002.json", out)
    print(json.dumps({k: out[k] for k in ("a0_test", "vs_m0c")}), flush=True)


def paired(p2, label, config, name, against):
    """imp of `config` and of the m0c runs `against`, paired by seed, with the exact sign-flip test."""
    mine = {s: p2.run(label, config, name, s)["imp"] for s in SEEDS}
    diffs = [mine[s] - against[s] for s in SEEDS]
    return {"imp": mine, "m0c_imp": against, "diffs": diffs, "mean_diff": statistics.mean(diffs),
            "p": sign_flip(diffs), "passed": statistics.mean(diffs) > 0 and sign_flip(diffs) < 0.05}


def control_step(p2):
    name = p2.choice()["held_out"]
    m0c = {s: p2.run("m0c", M0C, name, s)["imp"] for s in SEEDS}
    result = paired(p2, "m_hand", M_HAND, name, m0c)
    write_json(p2.out / "control.json", result)
    print(json.dumps(result), flush=True)


def loop(p2):
    training = p2.choice()["training"]
    runs = [(n, s) for n in training for s in ((1, 2) if len(training) == 1 else (1,))]
    return MetaLoop(p2.out / "meta", lambda vid, config, name, seed: p2.run(vid, config, name, seed), runs,
                    lambda: LLM(model=p2.args.model))


def loop_step(p2):
    archive = loop(p2).run(M0C, GENERATIONS)
    for v in archive:
        print(f"{v['id']} (parent {v.get('parent')}): score {score(v)}, imp {v.get('imp')}", flush=True)


def transfer_step(p2):
    name, best = p2.choice()["held_out"], best_version(loop(p2).load())
    m0c = {s: p2.run("m0c", M0C, name, s)["imp"] for s in SEEDS}
    result = {"version": best["id"], "config": best["config"], **paired(p2, best["id"], best["config"], name, m0c)}
    write_json(p2.out / "transfer.json", result)
    print(json.dumps(result), flush=True)


def report_step(p2):
    out = {"scan": {n: p2.scanned(n) for n in SETS}}
    for part in ("choice", "repeat", "r002", "control", "transfer"):
        path = p2.out / f"{part}.json"
        if path.exists():
            out[part] = json.loads(path.read_text())
    archive = loop(p2).load() if (p2.out / "choice.json").exists() and (p2.out / "meta" / "archive.json").exists() \
        else []
    out["archive"] = [{k: v.get(k) for k in ("id", "parent", "generation", "config", "proposal", "imp", "status")}
                      | {"score": score(v)} for v in archive]
    out["runs"], forecasts = {}, []
    for path in sorted((p2.out / "runs").glob("*.json")):
        r = json.loads(path.read_text())
        out["runs"][path.stem] = {"improver": r["improver"], "set": r["set"], "seed": r["seed"],
                                  "select_solved": r["select_solved"], "test_solved": r.get("test_solved"),
                                  "imp": r.get("imp"), "budget": r.get("budget"),
                                  "rounds": [{k: s.get(k) for k in ("status", "config", "select_gain", "kept",
                                                                    "predicted_gain", "keep_probability",
                                                                    "hypothesis")} for s in r["rounds"]],
                                  "improver_tokens": r["improver_usage"].get("completion_tokens")}
        forecasts += [(r["improver"], s) for s in r["rounds"]
                      if s.get("predicted_gain") is not None and "select_gain" in s]
    # The improvers' forecasts against what happened (docs/rive-design.md section 3, item 6).
    out["forecasts"] = {}
    for label in sorted({lab for lab, _ in forecasts}):
        mine = [s for lab, s in forecasts if lab == label]
        out["forecasts"][label] = {
            "n": len(mine), "gain_mse": statistics.mean((s["predicted_gain"] - s["select_gain"]) ** 2 for s in mine),
            "brier": statistics.mean((s["keep_probability"] - bool(s["kept"])) ** 2 for s in mine),
            "mean_keep_probability": statistics.mean(s["keep_probability"] for s in mine),
            "kept_rate": statistics.mean(bool(s["kept"]) for s in mine)}
    write_json(Path(p2.args.out), out)
    print(json.dumps({k: v for k, v in out.items() if k not in ("runs", "archive")}, indent=1), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=("check", "packs", "scan", "m0c", "choose", "control", "loop", "transfer",
                                    "report", "gepa_smoke", "passk", "repeat", "r002"))
    p.add_argument("--smoke-budget", type=int, default=90)
    p.add_argument("--out-dir", default=os.path.expanduser("~/alter/meta-rsi/p2"))
    p.add_argument("--packs", default=os.path.expanduser("~/alter/meta-rsi/calibration/packs.json"))
    p.add_argument("--model", default="qwen3.5-4b")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--margin", type=int, default=3)
    p.add_argument("--out", default=str(ROOT / "results" / "meta_rsi_p2.json"))
    args = p.parse_args()
    steps = {"check": check_step, "packs": packs_step, "scan": scan_step, "m0c": m0c_step, "choose": choose_step,
             "control": control_step, "loop": loop_step, "transfer": transfer_step, "report": report_step,
             "gepa_smoke": gepa_smoke_step, "passk": passk_step, "repeat": repeat_step, "r002": r002_step}
    p2 = P2(args)
    steps[args.step](p2)
    p2.events.log("step_done", step=args.step)


if __name__ == "__main__":
    main()
