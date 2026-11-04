"""T2's environment check (docs/rive-design.md section 6.11, set again in 6.17): can LiveCodeBench program synthesis
tell improvers apart? Steps, each resumable:

  python scripts/rive_t2.py packs    # dev, select and test packs from LiveCodeBench release v1 and test6
  python scripts/rive_t2.py scan     # A0 and hand-picked configurations on the selection and test packs
  python scripts/rive_t2.py repeat   # A0's and the best configuration's selection evaluations, made REPEATS more
  python scripts/rive_t2.py gate     # the rules the scan decides alone; m0c runs only if they hold (6.17)
  python scripts/rive_t2.py m0c      # m0c from A0, seeds 1-2 (RESET-META, margin MARGIN)
  python scripts/rive_t2.py check    # the rules written beforehand
  python scripts/rive_t2.py report
  python scripts/rive_t2.py smoke    # every scanned configuration on SMOKE development tasks, in <out-dir>/smoke

The model is served as synth.T2Model describes: the card's sampling, and thinking through the e1 adapter with a
thinking budget. Evaluations are kept per agent and pack, so a configuration the scan and m0c both reach is scored
once. The heartbeat file is touched as tasks finish, for scripts/watchdog.py.
"""
import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rive.infra.events import EventLog, code_hash  # noqa: E402
from rive.infra.manifest import write_manifest  # noqa: E402
from rive.llm import LLM  # noqa: E402
from rive.meta_rsi import synth  # noqa: E402
from rive.meta_rsi.config_improver import M0C, ConfigImprover  # noqa: E402
from rive.meta_rsi.reset_meta import (EvaluationFailed, Evaluator, agent_hash, mean_tokens, reset_meta,  # noqa: E402
                                     solved, write_json)
from rive.runtime.workflow import run_workflow_tasks  # noqa: E402

CODE = [*(ROOT / "rive" / "meta_rsi" / f for f in ("synth.py", "synth_workflow.py", "config_improver.py",
                                                  "reset_meta.py")),
        ROOT / "rive" / "runtime" / "workflow.py", Path(__file__).resolve()]
EXPECTED = {"example_retry": 2, "feedback": "got_and_expected"}
THINK = {"thinking": True, "temperature": 0.6}  # the card's temperature with thinking
# name: (changes to A0, lever families it uses). Section 6.17: six of 6.11's twelve, the thinking ones now with the e1
# adapter and a thinking budget (synth.T2Model).
SCAN = {
    "a0": ({}, ()),
    "retry2_expected": (EXPECTED, ("examples",)),
    "review": ({"review": True}, ("review",)),
    "think": ({**THINK, "max_tokens": 8192}, ("thinking",)),
    "retry2_expected_review": ({**EXPECTED, "review": True}, ("examples", "review")),
    "think4096_retry1": ({**THINK, "max_tokens": 4096, "example_retry": 1, "feedback": "got_and_expected"},
                         ("thinking", "examples")),
}
MARGIN = 8  # selection tasks a change must gain to be kept (section 6.11)
ROOM, UNSATURATED, CORRELATION = 20, 10, 0.8
REPEATS = 1
SEEDS = (1, 2)
ROUNDS = 4
SMOKE = 6  # development tasks per configuration in the smoke test


class T2:
    def __init__(self, args):
        self.args, self.out = args, Path(args.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.heartbeat = self.out / "heartbeat"
        self.beat()
        self.events = EventLog(self.out / "events.jsonl", code_hash(CODE))
        self.events.log("step_start", step=args.step)
        self.solver = synth.T2Model(model=args.model, adapter=args.adapter)
        self.evaluate = Evaluator(self.run_tasks, self.out / ("smoke/evals" if args.step == "smoke" else "evals"))

    def beat(self):
        self.heartbeat.touch()

    def run_tasks(self, tasks, agent_dir, seed_base=0):
        start, forced = time.time(), self.solver.usage["forced"]
        results = run_workflow_tasks(tasks, agent_dir, self.solver, workers=self.args.workers,
                                     token_budget=synth.TOKEN_BUDGET, judge_fn=synth.judge, progress=self.beat,
                                     seed_base=seed_base, max_call_tokens=synth.TOKEN_BUDGET)
        seconds = time.time() - start
        print(f"{time.strftime('%F %T')} {Path(agent_dir).name} on {len(tasks)} tasks (seed base {seed_base}): "
              f"solved {solved(results)}, {mean_tokens(results)} tokens a task, "
              f"{sum(r['truncated_calls'] > 0 for r in results)} cut off, "
              f"{self.solver.usage['forced'] - forced} thinking budgets forced, {seconds:.0f} s, "
              f"{sum(r['tokens'] for r in results) / seconds:.0f} tokens/s", flush=True)
        return results

    def packs(self):
        return json.loads((self.out / "packs.json").read_text())

    def agent(self, name):
        return synth.make_agent(self.out / "scan" / name, {**synth.DEFAULT_CONFIG, **SCAN[name][0]})

    def scanned(self):
        """{name: {"select", "test", "select_tokens", "select_cut_off"}} for every configuration scanned so far."""
        packs, out = self.packs(), {}
        for name in SCAN:
            agent = self.out / "scan" / name
            if not agent.exists():
                continue
            row = {}
            for pack in ("select", "test"):
                path = self.out / "evals" / f"{agent_hash(agent)}_{pack}.json"
                if path.exists():
                    results = json.loads(path.read_text())
                    row[pack] = solved(results)
                    if pack == "select":
                        row["select_tokens"] = mean_tokens(results)
                        row["select_cut_off"] = sum(r.get("truncated_calls", 0) > 0 for r in results)
                        row["select_tasks"] = len(packs["select"])
            out[name] = row
        return out


def packs_step(t2):
    packs = synth.make_packs(synth.load_tasks())
    write_json(t2.out / "packs.json", packs)
    data = [synth.DATA / f"{s}.jsonl" for s in synth.SOURCES]
    write_manifest(t2.out / "manifest.json", repo=ROOT, model_dir="~/alter/models/Qwen3.5-4B", data_files=data,
                   extra={"args": vars(t2.args), "pack_sizes": {k: len(v) for k, v in packs.items()}})
    print("packs", json.dumps({k: len(v) for k, v in packs.items()}), flush=True)


def scan_step(t2):
    packs = t2.packs()
    for name in SCAN:
        agent = t2.agent(name)
        for pack in (("dev", "select", "test") if name == "a0" else ("select", "test")):
            t2.evaluate(agent, pack, packs[pack])
        print(f"scan {name}: {json.dumps(t2.scanned()[name])}", flush=True)


def best_scanned(t2):
    rows = t2.scanned()
    return max((n for n in rows if n != "a0"), key=lambda n: rows[n]["select"])


def flips(a, b):
    return sum(x["correct"] != y["correct"] for x, y in zip(a, b))


def repeat_step(t2):
    """A0's and the best configuration's selection evaluations made REPEATS more times past the cache, with other
    seeds: how much the same agent's score moves (section 6.10)."""
    packs, out = t2.packs(), {}
    for name in ("a0", best_scanned(t2)):
        agent = t2.agent(name)
        runs = [t2.evaluate(agent, "select", packs["select"])]
        for i in range(REPEATS):
            path = t2.out / "repeat" / f"{name}_r{i + 1}.json"
            if not path.exists():
                results = t2.run_tasks(packs["select"], agent, seed_base=100000 * (i + 1))
                if any(r.get("server_errors", 0) for r in results):
                    raise EvaluationFailed(f"the model server failed calls in repeat {i + 1} of {name}")
                path.parent.mkdir(parents=True, exist_ok=True)
                write_json(path, results)
            runs.append(json.loads(path.read_text()))
        pairs = [flips(runs[i], runs[j]) for i in range(len(runs)) for j in range(i + 1, len(runs))]
        out[name] = {"scores": [solved(r) for r in runs], "flips": pairs,
                     "sd_of_difference": math.sqrt(statistics.mean(pairs))}
        print(f"repeat {name}: {json.dumps(out[name])}", flush=True)
    write_json(t2.out / "repeat.json", out)


def m0c_step(t2):
    gate = json.loads((t2.out / "gate.json").read_text())
    if not gate["go_on"]:
        print("m0c skipped: the scan alone already fails the rules (gate.json)", flush=True)
        return
    packs, a0 = t2.packs(), t2.agent("a0")
    for seed in SEEDS:
        path = t2.out / "runs" / f"m0c_s{seed}.json"
        if path.exists():
            continue
        improver = ConfigImprover("m0c", lambda: LLM(model=t2.args.model), M0C, docs=synth.component_docs(),
                                  schema=synth.CONFIG_SCHEMA, make=synth.make_agent, system=synth.SYSTEM)
        result = reset_meta(improver, a0, packs, t2.out / "runs" / f"m0c_s{seed}", t2.evaluate, rounds=ROUNDS,
                            margin=MARGIN, seed=seed)
        write_json(path, result)
        print(f"m0c seed {seed}: select {result['select_solved']}, test {result['test_solved']}, kept "
              f"{sum(bool(r.get('kept')) for r in result['rounds'])}", flush=True)


def pearson(xs, ys):
    return statistics.correlation(xs, ys) if len(set(xs)) > 1 and len(set(ys)) > 1 else None


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2
        i = j + 1
    return out


def scan_rules(t2):
    """The rules of section 6.11 that the scan decides alone (room, the selection pack predicting the test pack, two
    levers), and whether the margin is wide enough for the noise the repeats measured."""
    rows = t2.scanned()
    a0, best = rows["a0"]["select"], best_scanned(t2)
    names = [n for n in SCAN if "test" in rows.get(n, {})]
    sel, test = [rows[n]["select"] for n in names], [rows[n]["test"] for n in names]
    # Reported only: R-002's trouble was among good configurations, where rule 3 is weakest.
    near = [n for n in names if rows[n]["select"] >= rows[best]["select"] - UNSATURATED]
    single = {}
    for n in names:
        if len(SCAN[n][1]) == 1:
            family = SCAN[n][1][0]
            single[family] = max(single.get(family, -1), rows[n]["select"])
    families = sorted(f for f, s in single.items() if s - a0 >= MARGIN)
    combined = max((rows[n]["select"] for n in names if len(SCAN[n][1]) > 1), default=None)
    best_single = max(single.values())
    repeat = json.loads((t2.out / "repeat.json").read_text())
    noise = max(v["sd_of_difference"] for v in repeat.values())
    out = {"a0_select": a0, "best": best, "best_select": rows[best]["select"],
           "room": rows[best]["select"] - a0, "pearson": pearson(sel, test),
           "spearman": pearson(ranks(sel), ranks(test)), "near_best": near,
           "near_best_pearson": pearson([rows[n]["select"] for n in near], [rows[n]["test"] for n in near]),
           "single_family_best": single,
           "families_past_margin": families, "best_combined": combined, "best_single": best_single,
           "noise_sd_of_difference": noise}
    out.update(room_ok=out["room"] >= ROOM, correlated=out["pearson"] is not None and out["pearson"] >= CORRELATION,
               two_levers=len(families) >= 2 and combined is not None and combined - best_single >= MARGIN,
               margin_ok=2 * noise <= MARGIN)
    return out


def gate_step(t2):
    """Section 6.17: m0c (rule 2) runs only when the scan's own rules hold; otherwise T2 fails without it."""
    out = scan_rules(t2)
    out["go_on"] = all(out[k] for k in ("room_ok", "correlated", "two_levers"))
    write_json(t2.out / "gate.json", out)
    print("gate", json.dumps(out), flush=True)


def check_step(t2):
    """The rules of section 6.11 on the scan, the repeats and m0c's runs (rule 2 stays open when m0c was skipped)."""
    out = scan_rules(t2)
    paths = [t2.out / "runs" / f"m0c_s{s}.json" for s in SEEDS]
    out.update(m0c_select=None, m0c_kept=None, gap=None, unsaturated=None)
    if all(p.exists() for p in paths):
        runs = [json.loads(p.read_text()) for p in paths]
        kept = [sum(bool(x.get("kept")) for x in r["rounds"]) for r in runs]
        gap = out["best_select"] - statistics.mean(r["select_solved"] for r in runs)
        out.update(m0c_select=[r["select_solved"] for r in runs], m0c_kept=kept, gap=gap,
                   unsaturated=gap >= UNSATURATED or min(kept) == 0)
    out["qualifies"] = all(out[k] for k in ("room_ok", "unsaturated", "correlated", "two_levers"))
    write_json(t2.out / "check.json", out)
    print("check", json.dumps(out), flush=True)


def smoke_step(t2):
    """Every scanned configuration on the first SMOKE development tasks, with the real server and sandbox."""
    tasks = t2.packs()["dev"][:SMOKE]
    for name in SCAN:
        t2.evaluate(t2.agent(name), "dev_smoke", tasks)


def report_step(t2):
    report = {"packs": {k: len(v) for k, v in t2.packs().items()}, "scan": t2.scanned()}
    for name in ("repeat", "gate", "check"):
        path = t2.out / f"{name}.json"
        if path.exists():
            report[name] = json.loads(path.read_text())
    report["runs"] = {}
    for s in SEEDS:
        path = t2.out / "runs" / f"m0c_s{s}.json"
        if path.exists():
            r = json.loads(path.read_text())
            report["runs"][f"m0c_s{s}"] = {k: r[k] for k in ("start", "rounds", "select_solved", "test_solved",
                                                              "improver_usage")}
    forecasts = [x for r in report["runs"].values() for x in r["rounds"] if x.get("keep_probability") is not None
                 and "kept" in x]
    if forecasts:
        report["forecasts"] = {
            "n": len(forecasts), "kept_rate": statistics.mean(bool(x["kept"]) for x in forecasts),
            "mean_keep_probability": statistics.mean(x["keep_probability"] for x in forecasts),
            "brier": statistics.mean((x["keep_probability"] - bool(x["kept"])) ** 2 for x in forecasts),
            "gain_mse": statistics.mean((x["predicted_gain"] - x["select_gain"]) ** 2 for x in forecasts)}
    write_json(Path(t2.args.out), report)
    print(json.dumps({k: report[k] for k in ("packs", "scan")}, indent=1), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=("packs", "scan", "repeat", "gate", "m0c", "check", "report", "smoke"))
    p.add_argument("--out-dir", default=os.path.expanduser("~/alter/rive/t2b"))  # 6.11's run is in ~/alter/rive/t2
    p.add_argument("--model", default="qwen3.5-4b")
    p.add_argument("--adapter", default=synth.THINKING_ADAPTER, help="LoRA adapter for thinking calls")
    p.add_argument("--workers", type=int, default=16)  # the server's max-num-seqs; 14-24% faster than 12 (6.17)
    p.add_argument("--out", default=str(ROOT / "results" / "rive_t2b.json"))
    args = p.parse_args()
    steps = {"packs": packs_step, "scan": scan_step, "repeat": repeat_step, "gate": gate_step, "m0c": m0c_step,
             "check": check_step, "report": report_step, "smoke": smoke_step}
    t2 = T2(args)
    steps[args.step](t2)
    t2.events.log("step_done", step=args.step)


if __name__ == "__main__":
    main()
