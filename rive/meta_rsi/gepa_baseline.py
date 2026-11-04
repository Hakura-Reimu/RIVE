"""GEPA as a baseline improver of the component agent (docs/rive-design.md section 6.2, R-002).

GEPA (arXiv:2507.19457, `pip install gepa`) evolves named text components. The agent's configuration becomes three:
the system prompt, the instruction, and "knobs", the other settings as JSON, which are read back with every value
moved to the nearest one the schema allows. GEPA's minibatches come from the development pack and its Pareto set is
the selection pack, the same feedback and selection data RESET-META gives an improver; it reflects with the same
local model. Its budget is metric calls (one task run of one candidate), set to what the compared improver spent.
"""
import hashlib
import json
from pathlib import Path

from gepa import EvaluationBatch, optimize

from rive.meta_rsi.components import CONFIG_SCHEMA, DEFAULT_CONFIG, make_agent
from rive.research.agent_loop import _judge_note

TEXTS = ("system_prompt", "instruction")
KNOB_NAMES = [k for k in CONFIG_SCHEMA["properties"] if k not in TEXTS]
CUT = 600

KNOBS_TEMPLATE = """An AI agent fixes buggy Python functions. Besides its texts, it is set by these knobs:
<docs>

The current knobs, as JSON:
```
<curr_param>
```

Below are task inputs, the agent's answers with these knobs, and feedback on each (the hidden tests' verdicts, the
tokens used, whether a model call was cut off):
```
<side_info>
```

Find what the knobs make the agent do wrong, and choose new knob values that should make it solve more tasks.
Give the complete new knobs as a JSON object within ``` blocks."""


def to_candidate(config):
    knobs = {k: config[k] for k in KNOB_NAMES}
    return {"system_prompt": config["system_prompt"], "instruction": config["instruction"],
            "knobs": json.dumps(knobs)}


def nearest(value, allowed):
    if isinstance(allowed[0], bool) or isinstance(allowed[0], str):
        return value if value in allowed else None
    try:
        return min(allowed, key=lambda a: abs(a - float(value)))
    except (TypeError, ValueError):
        return None


def to_config(candidate):
    """The configuration a candidate stands for, and what had to be corrected to get it."""
    props, problems = CONFIG_SCHEMA["properties"], []
    try:
        knobs = json.loads(candidate.get("knobs") or "{}")
        if not isinstance(knobs, dict):
            raise ValueError("not an object")
    except ValueError as e:
        knobs, problems = {}, [f"knobs are not a JSON object ({e}); defaults used"]
    config = dict(DEFAULT_CONFIG)
    for k in TEXTS:
        config[k] = (candidate.get(k) or "")[:props[k]["maxLength"]]
    for k in KNOB_NAMES:
        if k not in knobs:
            continue
        allowed = props[k]["enum"] if "enum" in props[k] else [False, True]
        v = nearest(knobs[k], allowed)
        if v is None:
            problems.append(f"{k}={knobs[k]!r} is not allowed; kept {config[k]!r}")
        else:
            config[k] = v
    return config, problems


class ConfigAdapter:
    """GEPA's adapter for the component agent. evaluate(agent_dir, pack, tasks) -> per-task results (an
    Evaluator: kept per agent and pack; minibatches are packs of their own, named by their task ids)."""
    propose_new_texts = None  # GEPA's own reflective proposal, with the knobs template below

    def __init__(self, evaluate, agents_dir, select_ids):
        self.run, self.agents_dir, self.select_ids = evaluate, Path(agents_dir), select_ids

    def agent(self, config):
        key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
        return make_agent(self.agents_dir / key, config)

    def evaluate_batch(self, batch, candidate):
        config, problems = to_config(candidate)
        ids = [t["id"] for t in batch]
        pack = "select" if ids == self.select_ids else "gepa_" + hashlib.sha256(" ".join(ids).encode()).hexdigest()[:12]
        return self.run(self.agent(config), pack, batch), problems

    def evaluate(self, batch, candidate, capture_traces=False):
        results, problems = self.evaluate_batch(batch, candidate)
        scores = [1.0 if r["correct"] else 0.0 for r in results]
        traces = [{"task": t, "result": r, "problems": problems} for t, r in zip(batch, results)] \
            if capture_traces else None
        return EvaluationBatch(outputs=[r.get("answer", "") for r in results], scores=scores, trajectories=traces)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        rows = []
        for tr in eval_batch.trajectories or []:
            r = tr["result"]
            feedback = ("solved. " if r["correct"] else "wrong. ") + _judge_note(r)
            feedback += f" Tokens {r.get('tokens')}, model calls {r.get('llm_calls')}"
            if r.get("truncated_calls"):
                feedback += ", a call was cut off by its token limit"
            if tr["problems"]:
                feedback += ". Knob problems: " + "; ".join(tr["problems"])
            rows.append({"Inputs": tr["task"]["public"]["prompt"][:CUT],
                         "Generated Outputs": (r.get("answer") or "")[-CUT:], "Feedback": feedback})
        return {c: rows for c in components_to_update}


class LocalReflector:
    """GEPA's reflection model: the local model through rive.llm.LLM, thinking off, a new seed per call."""

    def __init__(self, llm, temperature=0.7, max_tokens=1500):
        self.llm, self.temperature, self.max_tokens, self.calls = llm, temperature, max_tokens, 0

    def __call__(self, prompt):
        self.calls += 1
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        return self.llm.chat(messages, max_tokens=self.max_tokens, temperature=self.temperature, seed=self.calls)


def run_gepa(evaluate, packs, run_dir, llm, budget, seed=1, minibatch=3, docs=""):
    """One GEPA run from A0 on a task set. Returns its final configuration, the selection score GEPA chose it by,
    and what it explored."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    adapter = ConfigAdapter(evaluate, run_dir / "agents", [t["id"] for t in packs["select"]])
    reflector = LocalReflector(llm)
    result = optimize(seed_candidate=to_candidate(DEFAULT_CONFIG), trainset=packs["dev"], valset=packs["select"],
                      adapter=adapter, reflection_lm=reflector, reflection_minibatch_size=minibatch,
                      reflection_prompt_template={"knobs": KNOBS_TEMPLATE.replace("<docs>", docs)},
                      max_metric_calls=budget, use_merge=False, candidate_selection_strategy="pareto",
                      module_selector="round_robin", skip_perfect_score=True, seed=seed,
                      run_dir=str(run_dir / "gepa"), display_progress_bar=False, raise_on_exception=False)
    config, problems = to_config(result.best_candidate)
    return {"config": config, "config_problems": problems,
            "select_solved": round(result.val_aggregate_scores[result.best_idx] * len(packs["select"])),
            "candidates": [{"config": to_config(c)[0], "select_rate": s}
                           for c, s in zip(result.candidates, result.val_aggregate_scores)],
            "metric_calls": result.total_metric_calls, "reflection_calls": reflector.calls,
            "final_agent": str(adapter.agent(config))}
