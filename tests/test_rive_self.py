"""Self-taught data (scripts/rive_self.py, section 6.19): which of the model's own answers are kept."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import rive_self  # noqa: E402

FENCE = "`" * 3


def sample(i, body, correct=True, cut=0):
    return {"id": i, "correct": correct, "truncated_calls": cut, "answer": f"Sure.\n{FENCE}python\n{body}\n{FENCE}"}


def test_only_passing_whole_answers_are_kept():
    kept = rive_self.pick([sample("a", "x = 1", correct=False), sample("a", "x = 2", cut=1), sample("b", "y = 1")])
    assert [s["id"] for s in kept] == ["b"]


def test_answers_with_the_same_code_count_once():
    kept = rive_self.pick([sample("a", "x = 1"), sample("a", "x  =  1"), sample("a", "x = 2"), sample("a", "x = 3")])
    codes = [s["answer"] for s in kept]
    assert len(kept) == rive_self.KEEP and len(set(" ".join(c.split()) for c in codes)) == rive_self.KEEP


def test_the_choice_is_fixed_per_problem():
    samples = [sample("a", f"x = {k}") for k in range(4)]
    assert rive_self.pick(samples) == rive_self.pick(list(reversed(samples)))
