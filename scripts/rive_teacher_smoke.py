"""One tiny request through the teacher gateway (rive/infra/teacher.py): does the key work, and what is logged?

  python scripts/rive_teacher_smoke.py
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rive.infra.teacher import TeacherGateway  # noqa: E402

log_dir = Path(os.path.expanduser("~/alter/rive/teacher"))
gw = TeacherGateway(log_dir)
effort = sys.argv[1] if len(sys.argv) > 1 else "none"
if effort == "none":
    answer = gw.ask("Answer with one word.", "Reply with the single word OK.", "smoke", max_output_tokens=16,
                    effort="none")
else:  # with thinking: a short question that needs a little reasoning
    answer = gw.ask("Answer with the number only.", "How many prime numbers are there between 10 and 30?",
                    "smoke_thinking", max_output_tokens=2000, effort=effort)
print("answer:", answer.strip()[:40])
print("logged:", json.loads((log_dir / "usage.jsonl").read_text().splitlines()[-1]))
