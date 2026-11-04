"""Workflow: how the agent answers one maths, code or RTL task (Phase C). solve(task, ctx) -> {"answer": str}.

task: public fields only: "domain" (math / code / rtl), "source", "prompt" (the full Phase 0 prompt) and
      domain fields: math "problem"; code "statement", "starter_code", "testtype", "func_name";
      rtl "spec", "module", "language".
ctx.llm(messages, max_tokens=1024, thinking=False, temperature=0.0) -> str
      One model call. All calls of one task share a completion-token budget; past it, calls fail.
ctx.tool(name, args) -> dict
      Public tools: python_exec(code, stdin="", seconds=10) and verilog_sim(design, testbench, top=None,
      seconds=20); or a tool from tools/<name>.py.
ctx.prompt(name) -> str
      Text of agent/prompts/<name>.md ("" if there is none).

This initial version answers in one call with the Phase 0 prompt, which reproduces the Phase 0 baseline.
"""


def solve(task, ctx):
    system = ctx.prompt(task["domain"])
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": task["prompt"]}]
    return {"answer": ctx.llm(messages, max_tokens=4096)}
