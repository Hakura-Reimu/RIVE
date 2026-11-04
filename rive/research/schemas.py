"""JSON Schemas for the model's structured outputs. vLLM constrains decoding to them; the controller
validates again and fact-checks fields that refer to recorded data."""
from rive.benchmark.bfcl import FAILURE_TYPES

FAILURES = list(FAILURE_TYPES) + ["router_error"]
TARGET_FILES = ["agent/tool_router.py"]
METRICS = ["accuracy"] + [f"{f}_count" for f in FAILURES]

DIAGNOSE = {
    "type": "object",
    "properties": {
        "selected_problem": {"type": "string", "enum": FAILURES},
        "evidence": {
            "type": "array", "minItems": 1, "maxItems": 4,
            "items": {"type": "object",
                      "properties": {"failure_type": {"type": "string", "enum": FAILURES},
                                     "count": {"type": "integer", "minimum": 0},
                                     "example_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 3}},
                      "required": ["failure_type", "count", "example_ids"], "additionalProperties": False}},
        "suspected_cause": {"type": "string", "enum": ["tool_router", "prompt_hint", "model_knowledge", "other"]},
        "explanation": {"type": "string", "maxLength": 800},
        "recommended_next_action": {"type": "string", "maxLength": 400},
    },
    "required": ["selected_problem", "evidence", "suspected_cause", "explanation", "recommended_next_action"],
    "additionalProperties": False,
}

HYPOTHESIS = {
    "type": "object",
    "properties": {
        "hypothesis_id": {"type": "string", "enum": ["H1", "H2"]},
        "hypothesis": {"type": "string", "maxLength": 500},
        "target_file": {"type": "string", "enum": TARGET_FILES},
        "proposed_change": {"type": "string", "maxLength": 600},
        "prediction": {"type": "object",
                       "properties": {"metric": {"type": "string", "enum": METRICS},
                                      "direction": {"type": "string", "enum": ["increase", "decrease"]}},
                       "required": ["metric", "direction"], "additionalProperties": False},
        "control": {"type": "string", "maxLength": 200},
        "potential_regression": {"type": "string", "maxLength": 300},
    },
    "required": ["hypothesis_id", "hypothesis", "target_file", "proposed_change", "prediction", "control",
                 "potential_regression"],
    "additionalProperties": False,
}

HYPOTHESIZE = {
    "type": "object",
    "properties": {
        "hypotheses": {"type": "array", "minItems": 2, "maxItems": 2, "items": HYPOTHESIS},
        "selected": {"type": "string", "enum": ["H1", "H2"]},
        "why_selected": {"type": "string", "maxLength": 400},
    },
    "required": ["hypotheses", "selected", "why_selected"],
    "additionalProperties": False,
}

DESIGN = {
    "type": "object",
    "properties": {
        "target_file": {"type": "string", "enum": TARGET_FILES},
        "change_steps": {"type": "array", "minItems": 1, "maxItems": 6, "items": {"type": "string", "maxLength": 300}},
        "primary_metric": {"type": "string", "enum": METRICS},
        "expected_direction": {"type": "string", "enum": ["increase", "decrease"]},
        "rollback_condition": {"type": "string", "maxLength": 300},
    },
    "required": ["target_file", "change_steps", "primary_metric", "expected_direction", "rollback_condition"],
    "additionalProperties": False,
}

# ---------- Phase C: one module of a maths / code / RTL agent (docs/phase-c-design.md section 3) ----------

AGENT_DOMAINS = ["math", "code", "rtl"]
AGENT_FAILURES = ["wrong_answer", "no_final_answer", "compile_error", "mismatch", "time_limit", "runtime_error",
                  "load_error", "judge_error", "workflow_error"]
# "tool" means a new or changed tools/<tool_name>.py together with the workflow change that calls it
AGENT_TARGETS = ["agent/workflow.py", "agent/prompts/math.md", "agent/prompts/code.md", "agent/prompts/rtl.md",
                 "tool"]
AGENT_METRICS = (["macro_accuracy", "completion_tokens"] + [f"{d}_accuracy" for d in AGENT_DOMAINS]
                 + [f"{f}_count" for f in AGENT_FAILURES])
_PREDICTION = {"type": "object",
               "properties": {"metric": {"type": "string", "enum": AGENT_METRICS},
                              "direction": {"type": "string", "enum": ["increase", "decrease"]}},
               "required": ["metric", "direction"], "additionalProperties": False}

AGENT_DIAGNOSE = {
    "type": "object",
    "properties": {
        "selected_domain": {"type": "string", "enum": AGENT_DOMAINS},
        "selected_problem": {"type": "string", "enum": AGENT_FAILURES},
        "evidence": {
            "type": "array", "minItems": 1, "maxItems": 4,
            "items": {"type": "object",
                      "properties": {"domain": {"type": "string", "enum": AGENT_DOMAINS},
                                     "failure_type": {"type": "string", "enum": AGENT_FAILURES},
                                     "count": {"type": "integer", "minimum": 0},
                                     "example_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 3}},
                      "required": ["domain", "failure_type", "count", "example_ids"], "additionalProperties": False}},
        "suspected_cause": {"type": "string",
                            "enum": ["workflow", "prompt", "missing_tool", "token_budget", "model_capability", "other"]},
        "explanation": {"type": "string", "maxLength": 800},
        "recommended_next_action": {"type": "string", "maxLength": 400},
    },
    "required": ["selected_domain", "selected_problem", "evidence", "suspected_cause", "explanation",
                 "recommended_next_action"],
    "additionalProperties": False,
}

AGENT_HYPOTHESIS = {
    "type": "object",
    "properties": {
        "hypothesis_id": {"type": "string", "enum": ["H1", "H2"]},
        "hypothesis": {"type": "string", "maxLength": 500},
        "target": {"type": "string", "enum": AGENT_TARGETS},
        "tool_name": {"type": "string", "maxLength": 30},
        "proposed_change": {"type": "string", "maxLength": 600},
        "prediction": _PREDICTION,
        "control": {"type": "string", "maxLength": 200},
        "potential_regression": {"type": "string", "maxLength": 300},
    },
    "required": ["hypothesis_id", "hypothesis", "target", "tool_name", "proposed_change", "prediction", "control",
                 "potential_regression"],
    "additionalProperties": False,
}

AGENT_HYPOTHESIZE = {
    "type": "object",
    "properties": {
        "hypotheses": {"type": "array", "minItems": 2, "maxItems": 2, "items": AGENT_HYPOTHESIS},
        "selected": {"type": "string", "enum": ["H1", "H2"]},
        "why_selected": {"type": "string", "maxLength": 400},
    },
    "required": ["hypotheses", "selected", "why_selected"],
    "additionalProperties": False,
}

AGENT_DESIGN = {
    "type": "object",
    "properties": {
        "target": {"type": "string", "enum": AGENT_TARGETS},
        "tool_name": {"type": "string", "maxLength": 30},
        "change_steps": {"type": "array", "minItems": 1, "maxItems": 6, "items": {"type": "string", "maxLength": 300}},
        "primary_metric": {"type": "string", "enum": AGENT_METRICS},
        "expected_direction": {"type": "string", "enum": ["increase", "decrease"]},
        "rollback_condition": {"type": "string", "maxLength": 300},
    },
    "required": ["target", "tool_name", "change_steps", "primary_metric", "expected_direction", "rollback_condition"],
    "additionalProperties": False,
}

REFLECT = {
    "type": "object",
    "properties": {
        "conclusion": {"type": "string", "enum": ["supported", "refuted", "inconclusive"]},
        "summary": {"type": "string", "maxLength": 600},
        "limitations": {"type": "string", "maxLength": 400},
        "next_direction": {"type": "string", "maxLength": 400},
        "reusable_finding": {"type": "string", "maxLength": 300},
    },
    "required": ["conclusion", "summary", "limitations", "next_direction", "reusable_finding"],
    "additionalProperties": False,
}

# Phase C: the controller decides the conclusion from the measurements (docs/phase-c-design.md section 8: in C5 the
# model wrote "refuted" for 24 of 26 results, 20 of which were within run-to-run noise); the model explains it.
AGENT_REFLECT = {
    "type": "object",
    "properties": {k: v for k, v in REFLECT["properties"].items() if k != "conclusion"},
    "required": [k for k in REFLECT["required"] if k != "conclusion"],
    "additionalProperties": False,
}
