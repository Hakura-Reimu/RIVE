Tools written by this candidate. Each tool is one `<name>.py` defining:

- `SCHEMA`: a dict describing the arguments and the result;
- `run(args, public)`: does the work and returns a JSON-serializable result. `public.python_exec(...)` and
  `public.verilog_sim(...)` are available; tools do not start processes or open files themselves;
- `test(public)`: raises AssertionError when the tool is wrong. The controller runs it before the tool can be used.

`rive/research/agent_loop.py` (`TOOL_EXAMPLE`) has a complete tool that passes these checks.
