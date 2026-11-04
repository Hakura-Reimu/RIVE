from rive.benchmark.bfcl import as_tools, classify, parse_tool_calls

FUNCS = [{"name": "math.f", "parameters": {"type": "dict", "properties": {"x": {"type": "integer"}}}},
         {"name": "g", "parameters": {"type": "dict", "properties": {"y": {"type": "string"},
                                                                       "z": {"type": "float"}}}}]
GT = [{"math.f": {"x": [1]}}, {"math.f": {"x": [2]}}, {"g": {"y": ["hi"], "z": ["", 0.5]}}]


def call(name, **args):
    return f"<tool_call>\n<function={name}>\n" + "".join(
        f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in args.items()) + "</function>\n</tool_call>\n"


def score(text, gt=GT, irrelevance=False):
    return classify(text, parse_tool_calls(text, FUNCS), gt, irrelevance)


def test_dotted_names_round_trip():
    assert as_tools(FUNCS)[0]["function"]["name"] == "math_f"
    assert as_tools(FUNCS)[0]["function"]["parameters"]["type"] == "object"
    assert parse_tool_calls(call("math_f", x=3), FUNCS) == [("math.f", {"x": 3})]


def test_parallel_calls_in_any_order_and_optional_param():
    assert score(call("g", y="hi") + call("math_f", x=2) + call("math_f", x=1)) is None
    assert score(call("g", y="hi", z=0.5) + call("math_f", x=2) + call("math_f", x=1)) is None


def test_failure_taxonomy():
    assert score("The answer is 42.") == "no_call"
    assert score("<tool_call>\n{not json}\n</tool_call>") == "parse_error"
    assert score(call("h", q=1) + call("math_f", x=2) + call("g", y="hi")) == "wrong_tool"
    assert score(call("math_f", x=1) + call("g", y="hi")) == "missing_call"
    assert score(call("math_f", x=1) + call("math_f", x=2) + call("g", y="hi") + call("g", y="hi")) == "extra_call"
    assert score(call("math_f", x=1) + call("math_f", x=3) + call("g", y="hi")) == "invalid_arguments"


def test_irrelevance():
    assert score("None of these tools fit.", gt=[], irrelevance=True) is None
    assert score(call("g", y="hi"), gt=[], irrelevance=True) == "irrelevance_violation"


def test_json_fallback_format():
    text = '<tool_call>\n{"name": "math_f", "arguments": {"x": 1}}\n</tool_call>'
    assert parse_tool_calls(text, FUNCS) == [("math.f", {"x": 1})]
