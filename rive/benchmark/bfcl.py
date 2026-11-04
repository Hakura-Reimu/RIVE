"""BFCL-style tool calls: parse Qwen3.5's tool-call output and score it against BFCL ground truth.

Ported from phase0/baseline_eval.py (validated there); adds a failure taxonomy for OBSERVE.
"""
import ast
import json
import re

TYPE_MAP = {"dict": "object", "float": "number", "tuple": "array", "any": "string"}

FAILURE_TYPES = ("no_call", "wrong_tool", "missing_call", "extra_call", "invalid_arguments", "parse_error",
                 "irrelevance_violation")


def to_json_schema(node):
    """BFCL uses Python-ish type names; map them to JSON Schema for the chat template."""
    if isinstance(node, dict):
        out = {k: to_json_schema(v) for k, v in node.items()}
        if isinstance(out.get("type"), str):
            out["type"] = TYPE_MAP.get(out["type"], out["type"])
        return out
    if isinstance(node, list):
        return [to_json_schema(v) for v in node]
    return node


def tool_name(name):
    """Dots are not allowed in tool names; BFCL names like math.hypot become math_hypot."""
    return name.replace(".", "_")


def as_tools(functions):
    return [{"type": "function", "function": {"name": tool_name(f["name"]), "description": f.get("description", ""),
                                              "parameters": to_json_schema(f["parameters"])}} for f in functions]


def convert_value(raw, schema):
    s = raw.strip()
    t = (schema or {}).get("type")
    try:
        if t == "integer":
            v = float(s)
            return int(v) if v.is_integer() else v
        if t == "number":
            return float(s)
        if t == "boolean":
            return s.lower() in ("true", "1")
        if t == "string":
            return s
        return json.loads(s)
    except Exception:
        try:
            return ast.literal_eval(s)
        except Exception:
            return s


def parse_tool_calls(text, functions):
    """Calls as (original BFCL name, args). Handles the XML format and a JSON fallback."""
    schemas = {tool_name(f["name"]): (f["name"], to_json_schema(f["parameters"]).get("properties", {}))
               for f in functions}
    calls = []
    for name, body in re.findall(r"<function=([^>\n]+)>(.*?)</function>", text, re.S):
        orig, props = schemas.get(name.strip(), (name.strip(), {}))
        args = {}
        for pname, raw in re.findall(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", body, re.S):
            args[pname.strip()] = convert_value(raw, props.get(pname.strip()))
        calls.append((orig, args))
    if not calls:
        for blob in re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S):
            try:
                obj = json.loads(blob)
                name = obj.get("name", "")
                calls.append((schemas.get(name, (name, {}))[0], obj.get("arguments", {})))
            except json.JSONDecodeError:
                pass
    return calls


def std_str(s):
    return re.sub(r"[ ,./\-_*^]", "", s).lower().replace("'", '"')


def value_ok(value, acceptable):
    for acc in acceptable:
        if acc == "":
            continue
        if isinstance(acc, str) and isinstance(value, str) and std_str(acc) == std_str(value):
            return True
        if (isinstance(acc, (int, float)) and isinstance(value, (int, float))
                and not isinstance(acc, bool) and not isinstance(value, bool) and float(acc) == float(value)):
            return True
        if isinstance(acc, list) and isinstance(value, list) and len(acc) == len(value) and all(
                value_ok(v, [a]) for v, a in zip(value, acc)):
            return True
        if isinstance(acc, dict) and isinstance(value, dict) and dict_ok(value, acc):
            return True
        if acc == value:
            return True
    return False


def dict_ok(args, expected):
    if set(args) - set(expected):
        return False
    for key, acceptable in expected.items():
        options = acceptable if isinstance(acceptable, list) else [acceptable]
        if key in args:
            if not value_ok(args[key], options):
                return False
        elif "" not in options:
            return False
    return True


def calls_match(calls, expected):
    """Every expected (func, params) matched by a distinct model call, in any order."""
    if len(calls) != len(expected):
        return False
    if not expected:
        return True
    func, params = expected[0]
    for i, (name, args) in enumerate(calls):
        if name == func and dict_ok(args, params) and calls_match(calls[:i] + calls[i + 1:], expected[1:]):
            return True
    return False


def expected_calls(ground_truth):
    return [next(iter(gt.items())) for gt in ground_truth]


def classify(text, calls, ground_truth, irrelevance):
    """None if correct, else one FAILURE_TYPES entry. Decided by program, never by the model."""
    if irrelevance:
        return "irrelevance_violation" if calls else None
    expected = expected_calls(ground_truth)
    if not calls:
        return "parse_error" if ("<tool_call>" in text or "<function=" in text) else "no_call"
    if calls_match(calls, expected):
        return None
    wanted = {f for f, _ in expected}
    if any(name not in wanted for name, _ in calls):
        return "wrong_tool"
    if len(calls) < len(expected):
        return "missing_call"
    if len(calls) > len(expected):
        return "extra_call"
    return "invalid_arguments"
