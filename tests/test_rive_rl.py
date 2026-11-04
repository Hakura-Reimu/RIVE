import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import rive_rl  # noqa: E402


def test_advantages_are_reward_minus_the_problems_mean():
    rows = [{"id": "a", "correct": True}, {"id": "a", "correct": False}, {"id": "a", "correct": False},
            {"id": "a", "correct": False}, {"id": "b", "correct": True}, {"id": "b", "correct": True}]
    assert rive_rl.advantages(rows) == [0.75, -0.25, -0.25, -0.25, 0.0, 0.0]


def test_batches_cover_an_epoch_without_repeats_and_are_fixed():
    ids = [f"p{i:03d}" for i in range(193)]
    epoch = [p for it in range(6) for p in rive_rl.batch_for(it, ids, 32)]
    assert len(epoch) == len(set(epoch)) == 192
    assert rive_rl.batch_for(3, ids, 32) == rive_rl.batch_for(3, list(reversed(ids)), 32)
    assert rive_rl.batch_for(6, ids, 32) != rive_rl.batch_for(0, ids, 32)  # the next epoch is shuffled anew


def test_policy_names():
    assert rive_rl.name_of(0) == rive_rl.BASE
    assert rive_rl.name_of(1) == "rl-0000"
