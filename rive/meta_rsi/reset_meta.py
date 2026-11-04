"""RESET-META (docs/meta-rsi-design.md section 4): an improver improves the same initial agent A0 under a fixed
budget of proposals, and the trusted controller decides what is kept.

Each round the improver gets the current agent and its development results and proposes one change. A valid
change is run on the selection pack; it replaces the current agent only when it solves at least `margin` more
selection tasks, and then its development results become the next feedback. The improver hears the outcome as a
count, never the selection tasks. After the last round the current agent is scored once on the test pack. MIG is
the difference of two improvers' test scores, paired by run seed.
"""
import hashlib
import json
import os
import statistics
from pathlib import Path

from rive.research.agent_loop import candidate_files


def agent_hash(agent_dir):
    """The content of a candidate directory (research notes excluded)."""
    h = hashlib.sha256()
    for path, data in sorted(candidate_files(agent_dir).items()):
        h.update(path.encode() + b"\0" + hashlib.sha256(data).digest())
    return h.hexdigest()[:16]


def solved(results):
    return sum(r["correct"] for r in results)


def mean_tokens(results):
    return round(statistics.mean(r.get("tokens", 0) for r in results)) if results else 0


def write_json(path, obj):
    """Written whole or not at all, so a run stopped mid-write leaves no broken file behind."""
    tmp = Path(f"{path}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
    os.replace(tmp, path)


class EvaluationFailed(RuntimeError):
    """The model server failed during an evaluation, so its scores say nothing about the agent."""


class Evaluator:
    """Runs agents on the packs. Results are kept per agent content and pack, so an agent that two runs both end
    with is scored once and both runs see the same score. An evaluation in which the server failed a model call is
    not kept: it raises, and the run stops (to be started again once the server is back)."""

    def __init__(self, run, cache_dir):
        """run(tasks, agent_dir) -> per-task results."""
        self.run, self.cache = run, Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)

    def __call__(self, agent_dir, pack, tasks):
        path = self.cache / f"{agent_hash(agent_dir)}_{pack}.json"
        if path.exists():
            return json.loads(path.read_text())
        results = self.run(tasks, Path(agent_dir))
        failed = sum(r.get("server_errors", 0) > 0 for r in results)
        if failed:
            raise EvaluationFailed(f"the model server failed calls in {failed} of {len(results)} tasks of {pack}")
        write_json(path, results)
        return results


def reset_meta(improver, a0_dir, packs, run_dir, evaluate, rounds=4, margin=3, seed=1, test=True):
    """One RESET-META run of `improver` from A0. packs: {"dev", "select", "test": tasks}. test=False leaves the
    test pack alone (finish_test() adds it later). Each proposal also gets a context: the round, the number of
    rounds, the start's scores and the rounds so far (what a tracking improver tabulates)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    improver.start(run_dir)
    current = Path(a0_dir)
    dev, sel = evaluate(current, "dev", packs["dev"]), evaluate(current, "select", packs["select"])
    start = {"dev_solved": solved(dev), "select_solved": solved(sel), "select_tokens": mean_tokens(sel),
             "pack_sizes": {k: len(v) for k, v in packs.items()}}
    tasks_by_id = {t["id"]: t for t in packs["dev"]}
    history, notes = [], []
    for r in range(1, rounds + 1):
        context = {"round": r, "rounds": rounds, "start": start, "history": [dict(s) for s in history],
                   "margin": margin}
        child = improver.propose(f"r{r}", current, dev, tasks_by_id, list(notes), seed=seed * 100 + r,
                                 context=context)
        record = child.get("record") or {}
        step = {"round": r, "status": child["status"], "module": child.get("module"),
                "hypothesis": record.get("hypothesis", {}).get("hypothesis"), "checks": record.get("checks"),
                "attempts": record.get("stages", {}).get("implement", {}).get("attempts"),
                "config": record.get("config"),  # the configuration-only improver's proposal and forecast
                "predicted_gain": record.get("predicted_gain"), "keep_probability": record.get("keep_probability")}
        if child["status"] == "valid":
            child_sel = evaluate(child["dir"], "select", packs["select"])
            gain = solved(child_sel) - solved(sel)
            step.update(select_solved=solved(child_sel), select_gain=gain, kept=gain >= margin,
                        select_tokens=mean_tokens(child_sel))
            notes.append(f"{child['id']} ({child.get('module')}): {(step['hypothesis'] or '')[:160]} -> "
                         f"{'kept' if step['kept'] else 'not kept'}: {gain:+d} selection tasks against the agent "
                         "it changed")
            if step["kept"]:  # only the agent kept needs development results: they are the next feedback
                child_dev = evaluate(child["dir"], "dev", packs["dev"])
                step["dev_solved"] = solved(child_dev)
                current, dev, sel = Path(child["dir"]), child_dev, child_sel
        else:
            notes.append(child.get("note") or f"{child['id']}: {child['status']}")
        history.append(step)
    result = {"improver": improver.name, "seed": seed, "start": start, "rounds": history,
              "final_agent": str(current), "final_hash": agent_hash(current), "dev_solved": solved(dev),
              "select_solved": solved(sel), "improver_usage": improver.usage()}
    return finish_test(result, evaluate, packs) if test else result


def finish_test(result, evaluate, packs):
    """A run's final agent scored on the test pack."""
    test = evaluate(result["final_agent"], "test", packs["test"])
    return {**result, "test_solved": solved(test), "test_tasks": len(test), "test_rate": solved(test) / len(test),
            "test_results": [{"id": r["id"], "correct": r["correct"], "failure": r["failure"]} for r in test]}
