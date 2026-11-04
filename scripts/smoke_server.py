"""Check the vLLM server: plain completion, chat, and JSON-Schema-constrained chat (DIAGNOSE schema)."""
import json
import sys
import time
from pathlib import Path

import jsonschema

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rive.llm import LLM  # noqa: E402
from rive.research import schemas  # noqa: E402

llm = LLM()
t = time.perf_counter()
print("complete:", repr(llm.complete("The capital of France is", max_tokens=8)))
print("chat:", repr(llm.chat([{"role": "user", "content": "Say hello in five words."}], max_tokens=32)))
feedback = {"accuracy": 0.8, "failure_counts": {"wrong_tool": 9, "irrelevance_violation": 21},
            "failure_examples": [{"id": "multiple_3@k32", "failure": "wrong_tool"},
                                 {"id": "irrelevance_7@k32", "failure": "irrelevance_violation"}]}
raw = llm.chat([{"role": "user", "content": "Diagnose the most important problem from this feedback and answer "
                                            f"in JSON:\n{json.dumps(feedback)}"}], schema=schemas.DIAGNOSE,
               max_tokens=800)
obj = json.loads(raw)
jsonschema.validate(obj, schemas.DIAGNOSE)
print("structured DIAGNOSE ok:", json.dumps(obj)[:400])
print(f"usage {llm.usage}, {time.perf_counter() - t:.1f}s")
