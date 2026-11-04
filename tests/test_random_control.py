import random
import types

from rive.research.loop import check_router_source
from rive.research.random_control import RULE_BASELINE, TEMPLATES, random_router

TOOLS = [{"name": "get_weather", "description": "weather forecast for a city"},
         {"name": "add_numbers", "description": "add two numbers"},
         {"name": "send_email", "description": "send an email"}]


def load(source):
    module = types.ModuleType("router")
    exec(compile(source, "router", "exec"), module.__dict__)
    return module.select


def test_every_template_passes_static_checks_and_contract():
    rng = random.Random(0)
    for _ in range(40):
        name, params, src = random_router(rng)
        assert check_router_source(src) == [], name
        out = load(src)("what is the weather in Paris", TOOLS)
        names = [t["name"] for t in TOOLS]
        assert set(out["tools"]) <= set(names) and len(out["tools"]) == len(set(out["tools"]))
        assert isinstance(out["hint"], str)
    assert len(TEMPLATES) == 5


def test_rule_baseline_ranks_relevant_tool_first():
    assert load(RULE_BASELINE)("what is the weather in Paris", TOOLS)["tools"][0] == "get_weather"
