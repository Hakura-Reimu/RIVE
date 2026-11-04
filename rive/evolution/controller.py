"""Phase C evolution controller (docs/phase-c-design.md section 4). Each generation: one stratified batch of
development tasks shared by every candidate, the champion re-run on it as the control, parents from the trunk or
the archive (Thompson sampling over clade results), research, the four promotion gates, archive records, a report
and the stagnation rule.

Since C5 (docs/phase-c-design.md section 8) the development gates compare a child with all of the champion's runs
since it became champion, with the run-to-run noise measured rather than assumed; a child that differs from the
champion only in prompts is judged on those prompts' domains; a passing child runs a second time before the
selection retest; and a slot whose child was not evaluated gets another try.

The expensive parts are injected, so the control logic is testable without a model:
  research(parent_id, generation) -> {"id", "dir", "status", "module", "record", "note", "code_hash", "domains"?}
      status "valid" means the child's code passed validation and can be evaluated; "domains", when present, are
      the only domains its change can affect, and it is evaluated and judged on those alone
  evaluate(candidate_id, tasks) -> per-item results ({"id", "domain", "correct", ...}) in task order
  easy_guard(candidate_id) -> per-item results on the fixed easy-set sample
  selection(candidate_id) -> per-item results on the selection split
  reflect(child, verdict, champion_runs, child_runs) -> note, optional (REFLECT stage); verdict is judge()'s
"""
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

from rive.archive.report import generation_report, to_markdown
from rive.benchmark.stats import paired_bootstrap_macro

# The initial candidate's 11 development runs in C5, as accuracy per domain (results/c5_main_analysis.json): the
# prior for run-to-run noise while a champion has few runs of its own.
PRIOR_SIZES = {"math": 39, "code": 66, "rtl": 80}
PRIOR_RUNS = {"math": tuple(k / 39 for k in (8, 9, 10, 9, 8, 10, 11, 12, 11, 9, 11)),
              "code": tuple(k / 66 for k in (17, 18, 18, 21, 17, 16, 16, 21, 19, 17, 19)),
              "rtl": tuple(k / 80 for k in (21, 22, 23, 20, 21, 24, 24, 23, 22, 22, 21))}


@dataclass
class Settings:
    slots: int = 4
    base_history_share: float = 0.2  # handoff doc 3.5 suggests 80/20 as a starting value, not an optimum
    history_steps: tuple = (0.35, 0.5)
    patience: int = 5  # generations without a promotion before the history share goes up
    alpha: float = 0.05
    prior_weight: int = 10  # the noise prior counts as this many champion runs (C5: 11 runs, 10 degrees of freedom)
    # Share of champion-solved items a run of the champion itself gets wrong: 11.5% in the median of C5's seed runs.
    regression_prior: float = 0.115
    replicate: bool = True  # a child that passes the development gates runs again before the selection retest
    refills: int = 2  # extra research tries per generation for slots whose child could not be evaluated
    easy_max_net_loss: int = 2
    novelty_weight: float = 0.1


def sample_task_set(tasks, sizes, rng):
    """sizes[domain] tasks from each domain, sorted by id: the batch every candidate of a generation runs."""
    out = []
    for domain, k in sizes.items():
        pool = [t for t in tasks if t["domain"] == domain]
        out += rng.sample(pool, min(k, len(pool)))
    return sorted(out, key=lambda t: t["id"])


def compare(reference, child):
    """Paired comparison on the same items: domain-balanced difference, one-sided p, and the regression share
    (items the reference solved that the child gets wrong; docs/phase-a-design.md section 14). Used for the
    selection retest, where each side has one run."""
    by_id = {r["id"]: r for r in child}
    ref = [int(r["correct"]) for r in reference]
    new = [int(by_id[r["id"]]["correct"]) for r in reference]
    diff, p = paired_bootstrap_macro(ref, new, [r["domain"] for r in reference])
    solved = [i for i, x in enumerate(ref) if x]
    regression = sum(not new[i] for i in solved) / len(solved) if solved else 0.0
    return {"diff": diff, "p": p, "regression": regression}


def only(results, domains):
    """The results in `domains`; all of them when domains is None."""
    return [r for r in results if domains is None or r["domain"] in domains]


def accuracy(results, domain):
    rs = [r for r in results if r["domain"] == domain]
    return sum(r["correct"] for r in rs) / len(rs)


def score(results, domains):
    """Mean accuracy over `domains`: the domain-balanced mean when they are all of them."""
    return sum(accuracy(results, d) for d in domains) / len(domains)


def prior_sd(domains, sizes):
    """Run-to-run standard deviation of score() over `domains` in the C5 seed's runs, scaled from C5's numbers of
    items to `sizes` ({domain: items}). Domains are averaged run by run, so noise that moves a whole run is kept."""
    known = [d for d in domains if d in PRIOR_RUNS] or list(PRIOR_RUNS)
    per_run = [sum(PRIOR_RUNS[d][k] for d in known) / len(known) for k in range(len(PRIOR_RUNS[known[0]]))]
    return statistics.stdev(per_run) * math.sqrt(sum(PRIOR_SIZES[d] for d in known) / sum(sizes.values()))


def t_sf(t, df):
    """P(T > t) for Student's t with df degrees of freedom, by a normal approximation (0.051 at the 5% point of
    df = 10)."""
    z = t * (1 - 1 / (4 * df)) / math.sqrt(1 + t * t / (2 * df))
    return 0.5 * math.erfc(z / math.sqrt(2))


def pooled(runs):
    """Per item: the share of `runs` that solved it."""
    seen = {}
    for run in runs:
        for r in run:
            seen.setdefault(r["id"], []).append(int(r["correct"]))
    return {i: sum(v) / len(v) for i, v in seen.items()}


def regression(rate, run):
    """Share of the champion's solved items that `run` gets wrong, each weighted by how often the champion solves
    it (rate: pooled() over the champion's runs)."""
    solved = sum(rate[r["id"]] for r in run)
    return sum(rate[r["id"]] * (1 - int(r["correct"])) for r in run) / solved if solved else 0.0


def judge(champion_runs, child_runs, domains, s):
    """The development gates (docs/phase-c-design.md section 8, after C5).

    Dev: the child's mean score over `domains` against the mean of the champion's runs on the same items, one-sided
    t test. The run-to-run noise comes from the champion's runs, pulled towards the C5 seed's (s.prior_weight runs'
    worth) while they are few. In C5 one champion run compared item by item put the noise at 1.9 points where it
    was 2.6, and passed 10% of unchanged reruns.
    Regression: the child's loss of champion-solved items against the loss the champion's runs show among
    themselves, one-sided, its spread binomial over the items the champion solves: a domain with few of them loses
    a much more variable share. C5's fixed 15% stopped 26% of unchanged reruns."""
    champ = [[r for r in run if r["domain"] in domains] for run in champion_runs]
    kid = [[r for r in run if r["domain"] in domains] for run in child_runs]
    k, n, w = len(champ), len(kid), s.prior_weight
    champ_scores, kid_scores = [score(r, domains) for r in champ], [score(r, domains) for r in kid]
    sizes = {d: sum(r["domain"] == d for r in champ[0]) for d in domains}
    own_var = statistics.variance(champ_scores) if k > 1 else 0.0
    var = (w * prior_sd(domains, sizes) ** 2 + (k - 1) * own_var) / (w + k - 1)
    diff = statistics.mean(kid_scores) - statistics.mean(champ_scores)
    t, df = diff / math.sqrt(var * (1 / n + 1 / k)), w + k - 1
    rate = pooled(champ)
    solved = sum(rate.values())
    # An independent champion run loses sum p(1 - p) / sum p of them; k / (k - 1) undoes the small-sample bias.
    own_loss = sum(p * (1 - p) for p in rate.values()) * k / (k - 1) / solved if k > 1 and solved else 0.0
    expected = (w * s.regression_prior + (k - 1) * own_loss) / (w + k - 1)
    lost = statistics.mean(regression(rate, run) for run in kid)
    p_lost = t_sf((lost - expected) / math.sqrt(expected * (1 - expected) / (solved * n)), df) if solved else 1.0
    p = t_sf(t, df)
    return {"domains": list(domains), "diff": diff, "p": p, "p_worse": t_sf(-t, df), "sd": math.sqrt(var),
            "champion_runs": k, "child_runs": n, "regression": lost, "expected_regression": expected,
            "p_regression": p_lost, "pass_dev": diff > 0 and p < s.alpha, "pass_regression": p_lost >= s.alpha}


def history_share(stagnant, s):
    """(share of parent slots drawn from the archive, stagnation level). The share rises every `patience`
    generations without a promotion (handoff doc 3.5, level 1). Level 2 calls for a bigger model, which the
    4070 cannot run, so it is only recorded."""
    steps = [s.base_history_share, *s.history_steps]
    if stagnant >= s.patience * len(steps):
        return steps[-1], 2
    i = min(stagnant // s.patience, len(steps) - 1)
    return steps[i], 0 if i == 0 else 1


def novelty(archive, pool):
    """1 - share of the pool that changed the same module: rarer kinds of change get a bonus."""
    modules = {cid: archive.candidate(cid)["module"] for cid in pool}
    return {cid: 1 - sum(m == mod for m in modules.values()) / len(pool) for cid, mod in modules.items()}


def choose_parents(archive, champion, s, share, rng):
    """One (parent, source) per slot: the trunk (the champion) with probability 1 - share, otherwise the archive
    candidate with the best draw from Beta(1 + clade wins, 1 + clade losses) plus a novelty bonus (HGM)."""
    pool = [cid for cid in archive.evaluated() if cid != champion]
    bonus = novelty(archive, pool) if pool else {}
    parents = []
    for _ in range(s.slots):
        if not pool or rng.random() >= share:
            parents.append((champion, "trunk"))
            continue

        def draw(cid):
            st = archive.clade_stats(cid)
            return rng.betavariate(1 + st["wins"], 1 + st["losses"]) + s.novelty_weight * bonus[cid]
        parents.append((max(pool, key=draw), "archive"))
    return parents


class Evolution:
    def __init__(self, archive, research, evaluate, easy_guard, selection, dev_tasks, sizes, settings=None,
                 seed=0, report_dir=None, reflect=None):
        self.archive, self.research, self.evaluate = archive, research, evaluate
        self.easy_guard, self.selection, self.reflect = easy_guard, selection, reflect
        self.dev_tasks, self.sizes = dev_tasks, sizes
        self.settings = settings or Settings()
        self.rng = random.Random(seed)
        self.report_dir = Path(report_dir) if report_dir else None
        self.stagnant = 0
        self._cache = {}

    def restore(self, stagnant, cached, seed):
        """After an interruption: generations since the last promotion, easy and selection results already computed
        ({(kind, candidate): results}), and a new random stream for the generations still to run."""
        self.stagnant, self.rng = stagnant, random.Random(seed)
        self._cache.update(cached)

    def _once(self, kind, cid):
        """Easy-set and selection results per candidate are computed once and archived."""
        if (kind, cid) not in self._cache:
            results = (self.easy_guard if kind == "easy" else self.selection)(cid)
            self.archive.add_evaluation(cid, kind, kind, results)
            self._cache[(kind, cid)] = results
        return self._cache[(kind, cid)]

    def champion_runs(self, champion, tasks):
        """The champion's development runs that cover `tasks`, cut to them, since it became champion: its runs as a
        challenger are what got it promoted, so they would flatter it."""
        ids = {t["id"] for t in tasks}
        runs = self.archive.runs(champion, since=self.archive.promoted_at(champion))
        return [[r for r in run if r["id"] in ids] for run in runs if ids <= {r["id"] for r in run}]

    def run_generation(self, g):
        a, s = self.archive, self.settings
        tasks = sample_task_set(self.dev_tasks, self.sizes, self.rng)
        task_set = f"gen-{g:03d}"
        champion = a.champion()
        champ_results = self.evaluate(champion, tasks)
        a.add_evaluation(champion, task_set, "dev", champ_results, {"role": "champion"})
        champ_runs = self.champion_runs(champion, tasks)
        all_domains = sorted({t["domain"] for t in tasks})
        share, level = history_share(self.stagnant, s)
        a.log("generation_start", {"champion": champion, "history_share": share, "stagnation_level": level,
                                   "tasks": len(tasks), "champion_runs": len(champ_runs)}, g)

        passing = []
        slots, refills = choose_parents(a, champion, s, share, self.rng), s.refills
        while slots:
            parent, source = slots.pop(0)
            child = self.research(parent, g)
            a.add_candidate(child["id"], parent, g, child["dir"], child.get("origin", "research"), source,
                            child.get("module"), child.get("code_hash"), child["status"], child.get("note"))
            if child["status"] != "valid":
                a.add_experiment(child["id"], child.get("record", {}))
                if refills:  # C5 generations 4-10 lost 8 of 28 slots this way
                    refills -= 1
                    slots.insert(0, (parent, source))
                continue
            domains = child.get("domains") or all_domains
            own = [t for t in tasks if t["domain"] in domains]
            runs = [self.evaluate(child["id"], own)]
            verdict = judge(champ_runs, runs, domains, s)
            if s.replicate and verdict["pass_dev"] and verdict["pass_regression"]:
                runs.append(self.evaluate(child["id"], own))
                verdict = {**judge(champ_runs, runs, domains, s), "first_run": verdict}
            for n, results in enumerate(runs, 1):
                last = n == len(runs)  # one row per child carries the comparison: the lineage statistics count rows
                a.add_evaluation(child["id"], task_set, "dev", results, {"domains": domains, "run": n},
                                 reference_id=champion if last else None,
                                 vs_reference=verdict["diff"] if last else None)
            note = self.reflect(child, {**verdict, "alpha": s.alpha}, champ_runs, runs) if self.reflect else None
            a.set_status(child["id"], "evaluated", note)
            a.add_experiment(child["id"], child.get("record", {}))
            gates = {"pass_dev": verdict["pass_dev"], "pass_regression": verdict["pass_regression"]}
            if all(gates.values()):
                net = (sum(r["correct"] for r in only(self._once("easy", champion), child.get("domains"))) -
                       sum(r["correct"] for r in only(self._once("easy", child["id"]), child.get("domains"))))
                gates["pass_easy"] = net <= s.easy_max_net_loss
            a.log("gates", {"candidate": child["id"], **verdict, **gates}, g)
            if len(gates) == 3 and all(gates.values()):
                passing.append((verdict["diff"], child["id"], child.get("domains")))

        promoted = None
        if passing:
            # One selection retest per generation, for the best child only: fewer adaptive uses of the split.
            _, best, domains = max(passing)
            sel = compare(only(self._once("selection", champion), domains),
                          only(self._once("selection", best), domains))
            passed = sel["diff"] > 0 and sel["p"] < s.alpha
            a.log("selection_retest", {"candidate": best, "domains": domains, **sel, "passed": passed}, g)
            if passed:
                a.promote(best, g, {"selection_diff": sel["diff"], "selection_p": sel["p"]})
                promoted = best
        self.stagnant = 0 if promoted else self.stagnant + 1

        report = generation_report(a, g, task_set)
        if self.report_dir:
            self.report_dir.mkdir(parents=True, exist_ok=True)
            (self.report_dir / f"gen_{g:03d}.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
            (self.report_dir / f"gen_{g:03d}.md").write_text(to_markdown(report))
        return report
