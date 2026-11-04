"""The meta level of P2 (docs/meta-rsi-design.md section 7): an archive of improver configurations, each scored by
imp@4 on the training task sets, from which a parent is drawn to propose the next version of itself.

The parent proposes with its own strategy for changing itself (meta_policy), its memory and its settings: the
configuration-level counterpart of a HyperAgents meta agent editing itself. Parent selection, scoring and the task
sets stay fixed, outside what any version can change. A version sees the total test scores of the training task sets
and the rounds of its own runs, never a test task.
"""
import json
import math
import random
import statistics
from pathlib import Path

from rive.meta_rsi.components import COMPONENT_DOCS
from rive.meta_rsi.config_improver import IMPROVER_SCHEMA, changes, config_key
from rive.meta_rsi.reset_meta import write_json
from rive.research.loop import StageFailed, Stages

META_SYSTEM = (
    "You improve an improver: a program that improves an AI agent which fixes bugs in short Python functions with "
    "a small language model. The improver changes only the agent's configuration, a few times per run. You change "
    "only the improver's own configuration, based on how its versions did. You cannot change the agent's "
    "components, the tasks, the scoring or how versions are chosen."
)
META_DOCS = """The improver's configuration:
- policy (text, at most 1500 characters): its strategy for choosing agent configurations, shown to it in every
  round of a run.
- meta_policy (text, at most 1000 characters): its strategy for changing its own configuration, shown to it in
  this step.
- memory (text, at most 1500 characters): what earlier runs taught, each point with its evidence; shown to it in
  every run and in this step. Nothing writes to it during a run; only this step can change it.
- tracking (true/false): true shows it a table of everything tried so far, with what each try changed and how it
  scored (in a run: that run's changes; in this step: every improver version); false shows one line per earlier
  change, and in this step only its own line and its parent's.
- temperature (0.3, 0.7, 1.0): its sampling temperature.
A version is scored by imp@4: starting from the initial agent, it makes 4 changes on each training task set, each
kept only if it solves at least 3 more selection tasks; the score is how many more test tasks the configuration it
ends with solves than the initial agent. The improver will later also be used on task sets it has not seen."""
META_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["diagnosis", "hypothesis", "prediction", "improver"],
    "properties": {"diagnosis": {"type": "string", "maxLength": 800},
                   "hypothesis": {"type": "string", "maxLength": 400},
                   "prediction": {"type": "string", "maxLength": 300},
                   "improver": IMPROVER_SCHEMA},
}
SCALE = 5.0  # test tasks: how sharply a better score raises the chance of being drawn as a parent


def score(version):
    """Mean imp@4 over the version's runs; None until all of them are done."""
    runs = version.get("imp", {})
    if not runs or any(v is None for v in runs.values()):
        return None
    return statistics.mean(runs.values())


def choose_parent(archive, rng):
    """HyperAgents' rule: the better a version scored and the fewer children it has, the likelier it is drawn."""
    scored = [v for v in archive if score(v) is not None]
    mid = statistics.median(score(v) for v in scored)
    weights = [1 / (1 + math.exp(-(score(v) - mid) / SCALE)) / (1 + len(v["children"])) for v in scored]
    return rng.choices(scored, weights)[0]


def archive_table(archive, parent, tracking):
    """Every version (tracking) or only the parent and its own parent, with their scores."""
    by_id = {v["id"]: v for v in archive}
    shown = archive if tracking else [by_id[i] for i in (parent.get("parent"), parent["id"]) if i in by_id]
    rows = ["| version | parent | fields changed from its parent | imp@4 per run | mean |", "|---|---|---|---|---|"]
    for v in shown:
        if v.get("config") is None:
            rows.append(f"| {v['id']} | {v.get('parent') or '-'} | no valid proposal | - | - |")
            continue
        base = by_id[v["parent"]]["config"] if v.get("parent") in by_id else v["config"]
        runs = ", ".join(f"{k}: {x:+d}" if x is not None else f"{k}: -" for k, x in sorted(v.get("imp", {}).items()))
        mean = f"{score(v):+.1f}" if score(v) is not None else "-"
        changed = changes(base, v["config"]) if v.get("parent") else "start"
        rows.append(f"| {v['id']} | {v.get('parent') or '-'} | {changed} | {runs} | {mean} |")
    return "\n".join(rows)


def run_lines(run):
    """One run of a version as a few lines: each round's change and outcome, then the final imp@4."""
    lines = [f"Run {run['set']} seed {run['seed']}: imp@4 {run['imp']:+d}"]
    base = run["start_config"]
    for s in run["rounds"]:
        if s.get("config") is None or "select_solved" not in s:
            lines.append(f"  r{s['round']}: no valid proposal ({s['status']})")
            continue
        lines.append(f"  r{s['round']}: {changes(base, s['config'])}; hypothesis: {(s.get('hypothesis') or '')[:160]}; "
                     f"selection {s['select_gain']:+d}, {'kept' if s['kept'] else 'not kept'}")
        if s["kept"]:
            base = s["config"]
    return "\n".join(lines)


def meta_prompt(parent, archive, parent_runs, docs=COMPONENT_DOCS):
    c = parent["config"]
    parts = [META_DOCS, "What the agent's settings do (what the improver chooses from):\n" + docs]
    if c["meta_policy"]:
        parts.append(f"Your strategy for changing yourself:\n{c['meta_policy']}")
    if c["memory"]:
        parts.append(f"Your memory:\n{c['memory']}")
    parts += [f"Your current configuration (version {parent['id']}):\n{json.dumps(c, ensure_ascii=False)}",
              "Improver versions so far:\n" + archive_table(archive, parent, c["tracking"]),
              "Your runs:\n" + ("\n".join(run_lines(r) for r in parent_runs) or "(none)"),
              "PROPOSE: diagnose what limited your score, choose a new configuration of yourself that should score "
              "higher, and predict its effect. Give the complete configuration."]
    return [{"role": "system", "content": META_SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]


def propose_version(parent, archive, parent_runs, make_llm, seed, docs=COMPONENT_DOCS, retries=2):
    """The parent's proposal for its next version: (output, stage record, token use). Raises StageFailed."""
    stages = Stages(make_llm(), False, retries, parent["config"]["temperature"])
    stages.seed = seed
    tried = {config_key(v["config"]) for v in archive}

    def untried(obj):
        if config_key(obj["improver"]) in tried:
            return "that configuration is an existing version; propose a different one"
    out, meta = stages._structured(meta_prompt(parent, archive, parent_runs, docs), META_SCHEMA, check=untried)
    return out, meta, dict(stages.llm.usage)


class MetaLoop:
    """The archive on disk (archive.json, written after every change) and the generations that grow it.
    score_run(version_id, config, set_name, seed) -> a run summary with "imp" (and "rounds", "start_config")."""

    def __init__(self, out_dir, score_run, runs, make_llm, docs=COMPONENT_DOCS):
        """runs: [(set name, seed)] every version is scored on."""
        self.out, self.score_run, self.runs = Path(out_dir), score_run, list(runs)
        self.make_llm, self.docs = make_llm, docs
        self.path = self.out / "archive.json"
        self.out.mkdir(parents=True, exist_ok=True)

    def load(self):
        return json.loads(self.path.read_text()) if self.path.exists() else []

    def save(self, archive):
        write_json(self.path, archive)

    def finish(self, archive, version):
        """Score the version's missing runs, saving after each."""
        for set_name, seed in self.runs:
            key = f"{set_name}_s{seed}"
            if version["imp"].get(key) is None:
                run = self.score_run(version["id"], version["config"], set_name, seed)
                version["imp"][key] = run["imp"]
                self.save(archive)

    def run(self, start_config, generations):
        archive = self.load()
        if not archive:
            archive = [{"id": "v0", "parent": None, "generation": 0, "config": start_config, "children": [],
                        "imp": {f"{s}_s{seed}": None for s, seed in self.runs}}]
            self.save(archive)
        for version in archive:
            self.finish(archive, version)
        for g in range(len(archive), generations + 1):
            parent = choose_parent(archive, random.Random(f"meta-parent-{g}"))
            parent_runs = [self.score_run(parent["id"], parent["config"], s, seed) for s, seed in self.runs]
            try:
                out, meta, usage = propose_version(parent, archive, parent_runs, self.make_llm, seed=1000 + g,
                                                   docs=self.docs)
            except StageFailed as e:  # recorded; the generation is spent
                archive.append({"id": f"v{g}", "parent": parent["id"], "generation": g, "config": None,
                                "status": "stage_failed", "error": str(e)[:300], "children": [], "imp": {}})
                self.save(archive)
                continue
            version = {"id": f"v{g}", "parent": parent["id"], "generation": g, "config": out["improver"],
                       "proposal": {k: out[k] for k in ("diagnosis", "hypothesis", "prediction")},
                       "stages": meta, "meta_usage": usage, "children": [],
                       "imp": {f"{s}_s{seed}": None for s, seed in self.runs}}
            parent["children"].append(version["id"])
            archive.append(version)
            self.save(archive)
            self.finish(archive, version)
        return archive


def best_version(archive):
    """The version with the highest training score (the earliest on a tie)."""
    scored = [v for v in archive if v.get("config") and score(v) is not None]
    return max(scored, key=lambda v: (score(v), -v["generation"]))
