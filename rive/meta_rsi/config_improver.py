"""The configuration-only improver (docs/meta-rsi-design.md sections 6 and 7): it reads the development feedback and
the current configuration of the component agent and proposes a new configuration, generated under the JSON schema,
so it cannot write a broken agent.

The improver is itself set by a configuration (IMPROVER_SCHEMA, section 7), the layer the meta level changes: its
strategy texts, its memory of earlier runs, whether it sees a performance table, and its sampling temperature.
M0C, all empty and off, is the configuration-only M0 of section 6.
"""
import json
from pathlib import Path

import jsonschema

from rive.meta_rsi.components import COMPONENT_DOCS, CONFIG_SCHEMA, make_agent, read_config
from rive.research.agent_loop import observe_agent
from rive.research.loop import StageFailed, Stages

SYSTEM = (
    "You are the research decision-maker for an AI agent that fixes bugs in short Python functions with a small "
    "language model. You improve the agent only by changing its configuration, based on evidence from development "
    "tests. You cannot change the model, the tests or the scoring. Be concrete: cite failure types, counts and "
    "example ids from the feedback you are given."
)
# predicted_gain and keep_probability are the proposal's forecast, registered before it is run (docs/rive-design.md
# section 3, item 6). Their ranges are checked after generation: the server's grammar is given no numeric bounds.
PROPOSAL_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["diagnosis", "hypothesis", "prediction", "predicted_gain", "keep_probability", "config"],
    "properties": {"diagnosis": {"type": "string", "maxLength": 600},
                   "hypothesis": {"type": "string", "maxLength": 400},
                   "prediction": {"type": "string", "maxLength": 300},
                   "predicted_gain": {"type": "integer"},
                   "keep_probability": {"type": "number"},
                   "config": CONFIG_SCHEMA},
}
IMPROVER_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["policy", "meta_policy", "memory", "tracking", "temperature"],
    "properties": {"policy": {"type": "string", "maxLength": 1500},
                   "meta_policy": {"type": "string", "maxLength": 1000},
                   "memory": {"type": "string", "maxLength": 1500},
                   "tracking": {"type": "boolean"},
                   "temperature": {"type": "number", "enum": [0.3, 0.7, 1.0]}},
}
M0C = {"policy": "", "meta_policy": "", "memory": "", "tracking": False, "temperature": 0.7}
TEXT_SHOWN = 60  # characters of a changed text field in the performance table


def config_key(config):
    return json.dumps(config, sort_keys=True)


def changes(before, after):
    """What `after` changes from `before`, one "field old->new" per changed field."""
    def show(v):
        text = json.dumps(v, ensure_ascii=False)
        return text if len(text) <= TEXT_SHOWN else text[:TEXT_SHOWN] + '..."'
    return ", ".join(f"{k} {show(before.get(k))}->{show(v)}" for k, v in after.items() if before.get(k) != v) \
        or "nothing"


def tracking_table(start_config, context):
    """The performance table of a run so far: every change with what it changed, its scores and its cost."""
    start, sizes = context["start"], context["start"]["pack_sizes"]
    rows = [f"Performance table of this run (selection pack {sizes['select']} tasks, development pack "
            f"{sizes['dev']} tasks; tokens: mean completion tokens per selection task):",
            "| change | fields changed from the configuration it changed | selection solved | gain | kept | "
            "development solved | tokens |", "|---|---|---|---|---|---|---|",
            f"| start | - | {start['select_solved']} | - | - | {start['dev_solved']} | {start['select_tokens']} |"]
    base = start_config
    for s in context["history"]:
        if s.get("config") is None or "select_solved" not in s:
            rows.append(f"| r{s['round']} | no valid proposal ({s['status']}) | - | - | - | - | - |")
            continue
        rows.append(f"| r{s['round']} | {changes(base, s['config'])} | {s['select_solved']} | {s['select_gain']:+d} | "
                    f"{'yes' if s['kept'] else 'no'} | {s.get('dev_solved', '-')} | {s.get('select_tokens', '-')} |")
        if s["kept"]:
            base = s["config"]
    return "\n".join(rows)


class ConfigImprover(Stages):
    def __init__(self, name, make_llm, config=None, docs=COMPONENT_DOCS, retries=2, schema=CONFIG_SCHEMA,
                 make=make_agent, system=SYSTEM):
        """make_llm: a new model client for each run, so its token use is the run's own. config: the improver's
        own configuration (IMPROVER_SCHEMA; M0C when None). docs: what the agent's settings do (component_docs()).
        schema, make and system: the agent's configuration space, how an agent is made from a configuration, and
        the system message; T1's by default, rive/meta_rsi/synth.py's for T2."""
        self.name, self.make_llm, self.docs, self.retries = name, make_llm, docs, retries
        self.make, self.system = make, system
        self.proposal_schema = {**PROPOSAL_SCHEMA, "properties": {**PROPOSAL_SCHEMA["properties"], "config": schema}}
        self.config = {**M0C, **(config or {})}
        jsonschema.validate(self.config, IMPROVER_SCHEMA)

    def start(self, run_dir):
        super().__init__(self.make_llm(), False, self.retries, self.config["temperature"])
        self.run_dir, self.tried, self.start_config = Path(run_dir), set(), None

    def prompt(self, current, results, tasks_by_id, notes, seed, context):
        c = self.config
        if c["tracking"] and context:
            history = tracking_table(self.start_config, context)
        else:
            history = "Earlier changes and how they did:\n" + ("\n".join(f"- {n}" for n in notes) or "(none yet)")
        parts = [self.docs]
        if c["policy"]:
            parts.append(f"Your strategy:\n{c['policy']}")
        if c["memory"]:
            parts.append("What your earlier runs taught (from other task sets; check it against this one):\n"
                         + c["memory"])
        parts += [f"Current configuration:\n{json.dumps(current)}",
                  "Development feedback (JSON):\n"
                  + json.dumps(observe_agent(results, tasks_by_id, seed=seed), ensure_ascii=False), history]
        if context:
            parts.append(f"This is change {context['round']} of {context['rounds']}; after the last one, the "
                         "configuration kept so far is final.")
        size = context["start"]["pack_sizes"]["select"] if context else "the"
        margin = context.get("margin", 3) if context else 3
        parts.append("PROPOSE: diagnose why the agent fails, choose a new configuration that should fix more tasks, "
                     "and predict its effect. Give the complete configuration. Also forecast, before it is run: "
                     f"predicted_gain, how many more of {size} selection tasks it will solve than the current "
                     "configuration (negative if fewer), and keep_probability, the probability (0 to 1) that it is "
                     f"kept, which needs at least {margin} more.")
        return [{"role": "system", "content": self.system}, {"role": "user", "content": "\n\n".join(parts)}]

    def propose(self, child_id, parent_dir, results, tasks_by_id, notes, seed, context=None):
        self.seed = seed
        current = read_config(parent_dir)
        if self.start_config is None:
            self.start_config = current
        self.tried.add(config_key(current))
        messages = self.prompt(current, results, tasks_by_id, notes, seed, context)

        def untried(obj):
            if config_key(obj["config"]) in self.tried:
                return "that configuration is the current one or was already tried; propose a different one"
            if not 0 <= obj["keep_probability"] <= 1:
                return "keep_probability must be between 0 and 1"
            if abs(obj["predicted_gain"]) > 100:
                return "predicted_gain must be a number of selection tasks"
        try:
            out, meta = self._structured(messages, self.proposal_schema, check=untried)
        except StageFailed as e:
            return {"id": child_id, "status": "stage_failed", "dir": None, "module": "config", "record": {},
                    "note": f"{child_id}: no valid proposal ({str(e)[:160]})"}
        self.tried.add(config_key(out["config"]))
        child = self.make(self.run_dir / "candidates" / child_id, out["config"])
        record = {"hypothesis": {"hypothesis": out["hypothesis"]}, "diagnosis": out["diagnosis"],
                  "prediction": out["prediction"], "predicted_gain": out["predicted_gain"],
                  "keep_probability": out["keep_probability"], "config": out["config"], "stages": {"propose": meta}}
        return {"id": child_id, "status": "valid", "dir": str(child), "module": "config", "record": record,
                "note": None}

    def usage(self):
        return dict(self.llm.usage)
