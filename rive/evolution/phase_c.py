"""Wiring for a Phase C run (docs/phase-c-design.md section 4): the archive, the research step, the workflow runtime
and the judges, as the callables rive.evolution.controller.Evolution takes."""
import json
import random
from pathlib import Path

from rive.benchmark.hard import judge
from rive.research.agent_loop import affected_domains, assign_problem, blocked_kinds, kind_of
from rive.runtime.workflow import run_workflow_tasks

REJECTED = {"invalid_patch", "duplicate", "leak_violation", "stage_failed"}


def experiment_history(archive):
    """The research experiments so far, oldest first, as blocked_kinds() takes them: what each worked on, whether it
    passed the development gate (rejected before evaluation counts as no; no result at all, as for a child whose
    evaluation was interrupted, is None) and its hypothesis."""
    records = archive.experiment_records()
    gates = {e["detail"]["candidate"]: e["detail"]["pass_dev"] for e in archive.events("gates")}
    out = []
    for c in archive.candidates():
        if c["origin"] == "seed":
            continue
        record = records.get(c["id"], {})
        success = gates[c["id"]] if c["id"] in gates else False if c["status"] in REJECTED else None
        out.append({"id": c["id"], "kind": kind_of(record), "success": success,
                    "hypothesis": record.get("hypothesis", {}).get("hypothesis", "")})
    return out


class PhaseC:
    def __init__(self, llm, run_dir, archive, research, easy_tasks, selection_tasks, workers=12, history=15,
                 judge_fn=judge, allocate=True):
        """research: an AgentResearch. Every evaluation is saved under run_dir/evals/<candidate>/; a parent's latest
        development results (with answers and judge details) are the feedback its children's research starts from.
        allocate: give the experiments of one generation different problems (agent_loop.assign_problem)."""
        self.llm, self.run_dir, self.archive, self.step = llm, Path(run_dir), archive, research
        self.easy_tasks, self.selection_tasks = easy_tasks, selection_tasks
        self.workers, self.history, self.judge_fn, self.allocate = workers, history, judge_fn, allocate
        self.latest, self.tasks_by_id, self.made = {}, {}, 0
        self.generation, self.assigned = None, []

    def _run(self, cid, tasks, name):
        results = run_workflow_tasks(tasks, Path(self.archive.candidate(cid)["dir"]), self.llm, workers=self.workers,
                                     judge_fn=self.judge_fn)
        out = self.run_dir / "evals" / cid / f"{name}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, ensure_ascii=False))
        return results

    def evaluate(self, cid, tasks):
        self.tasks_by_id.update({t["id"]: t for t in tasks})
        n = len(list((self.run_dir / "evals" / cid).glob("dev_*.json")))
        self.latest[cid] = self._run(cid, tasks, f"dev_{n + 1:02d}")
        return self.latest[cid]

    def easy_guard(self, cid):
        return self._run(cid, self.easy_tasks, "easy")

    def selection(self, cid):
        return self._run(cid, self.selection_tasks, "selection")

    def research(self, parent, generation):
        self.made += 1
        if generation != self.generation:
            self.generation, self.assigned = generation, []
        notes = [c["note"] for c in self.archive.candidates() if c["note"]][-self.history:]
        blocked = blocked_kinds(experiment_history(self.archive))
        results = self.feedback(parent)
        problem = assign_problem(results, blocked, self.assigned, random.Random(self.made)) if self.allocate else None
        if problem:
            self.assigned.append(problem)
        child = self.step.propose(f"g{generation:02d}c{self.made:03d}", Path(self.archive.candidate(parent)["dir"]),
                                  results, self.tasks_by_id, notes, seed=self.made, blocked=blocked, problem=problem)
        if child["status"] == "valid":
            # A change checked to leave the other domains as its parent had them is judged on its own domain when
            # the parent is the champion; any other parent differs from the champion in changes of its own.
            champion = self.archive.champion()
            child["domains"] = ([child["local_to"]] if child.get("local_to") and parent == champion else
                                affected_domains(child["dir"], self.archive.candidate(champion)["dir"]))
        return child

    def feedback(self, parent):
        """The parent's latest development results. A parent judged only on the domains its prompts change was run
        only on those; its other domains behave as the champion's, so they come from the champion's latest run."""
        results = self.latest[parent]
        have = {r["domain"] for r in results}
        champion = self.archive.champion()
        if champion != parent and champion in self.latest:
            results = results + [r for r in self.latest[champion] if r["domain"] not in have]
        return results

    def reflect(self, child, verdict, champion_runs, child_runs):
        return self.step.reflect(child, verdict, champion_runs, child_runs)

    def restore(self, dev_tasks):
        """Rebuilds what this object keeps in memory from the archive and run_dir/evals after an interruption.
        Returns the easy and selection results already computed, as {(kind, candidate): results}."""
        self.tasks_by_id.update({t["id"]: t for t in dev_tasks})
        candidates = self.archive.candidates()
        self.made = sum(c["origin"] != "seed" for c in candidates)
        if self.step is not None:  # so a change made before the interruption still counts as a duplicate
            self.step.seen_hashes.update(c["code_hash"] for c in candidates if c["code_hash"])
        cached = {}
        for d in sorted((self.run_dir / "evals").glob("*")):
            devs = sorted(d.glob("dev_*.json"))
            if devs:
                self.latest[d.name] = json.loads(devs[-1].read_text())
            for kind in ("easy", "selection"):
                if (d / f"{kind}.json").exists():
                    cached[(kind, d.name)] = json.loads((d / f"{kind}.json").read_text())
        return cached


def resume(archive, glue, evolution, done, dev_tasks, seed=0):
    """Continues a run after an interruption, such as a Windows update restarting WSL. `done` lists the finished
    generations (as in the run's summary.json). Candidates of the unfinished generation are marked "interrupted",
    and the state the controller and the glue keep in memory is rebuilt. Returns the next generation number."""
    g = len(done) + 1
    for c in archive.candidates(g):
        archive.set_status(c["id"], "interrupted", c["note"])
    stagnant = 0
    for gen in reversed(done):
        if gen["champion_after"] != gen["champion_before"]:
            break
        stagnant += 1
    evolution.restore(stagnant, glue.restore(dev_tasks), f"{seed}-resume-{g}")
    return g
