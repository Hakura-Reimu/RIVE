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


def test_token_ids_read_the_servers_logprobs():
    from types import SimpleNamespace as N
    lp = N(content=[N(token="token_id:12", logprob=-0.1234567), N(token="token_id:248046", logprob=-2.0)])
    assert rive_rl.token_ids(lp) == ([12, 248046], [-0.12346, -2.0])
    assert rive_rl.token_ids(None) == ([], [])


def test_dev_drop_stop_needs_two_checks_in_a_row_under_the_base(tmp_path):
    import json
    (tmp_path / "dev").mkdir()

    def check(n, rate):
        for d in rive_rl.DEV_SIZE:
            rows = [{"correct": i < rate * 100} for i in range(100)]
            (tmp_path / "dev" / f"{n:04d}_{d}.json").write_text(json.dumps(rows))
    check(0, 0.66)
    check(15, 0.60)
    assert not rive_rl.dev_drop_stop(tmp_path)  # one check under the line
    check(30, 0.64)
    assert not rive_rl.dev_drop_stop(tmp_path)  # 0.64 is within 5 points
    check(45, 0.61)
    check(60, 0.50)
    assert rive_rl.dev_drop_stop(tmp_path)
