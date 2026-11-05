import sys
from argparse import Namespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

torch = pytest.importorskip("torch")
import rive_rl_train as tr  # noqa: E402

ARGS = Namespace(tis_cap=2.0, clip_low=0.2, clip_high=0.28, kl=0.04)


def test_v2_terms_at_the_old_policy_are_the_advantage_and_no_kl():
    lp = torch.log(torch.tensor([0.5, 0.25]))
    terms, stats = tr.v2_terms(lp, lp.clone(), lp.clone(), lp.clone(), 0.5, ARGS)
    assert torch.allclose(terms, torch.tensor([-0.5, -0.5]))
    assert stats["clipped"] == 0 and stats["capped"] == 0 and abs(stats["k3"]) < 1e-9


def test_v2_terms_clip_cap_and_kl():
    old = torch.log(torch.tensor([0.5, 0.5, 0.5]))
    lp = torch.log(torch.tensor([0.9, 0.3, 0.5]))  # ratios 1.8, 0.6, 1.0
    mu = torch.log(torch.tensor([0.5, 0.1, float("nan")]))  # pi_old / mu: 1, 5 (capped at 2), none (1)
    terms, stats = tr.v2_terms(lp, old, old.clone(), mu, 1.0, ARGS)
    # A > 0: the first ratio is clipped at 1.28; the second is below 1 - 0.2 but not clipped for A > 0
    k3 = torch.exp(old - lp) - (old - lp) - 1
    expected = -(torch.tensor([1.0, 2.0, 1.0]) * torch.tensor([1.28, 0.6, 1.0]) - 0.04 * k3)
    assert torch.allclose(terms, expected, atol=1e-6)
    assert stats["clipped"] == 1 and stats["capped"] == 1 and stats["with_mu"] == 2
    _, neg = tr.v2_terms(lp, old, old.clone(), mu, -1.0, ARGS)
    assert neg["clipped"] == 1  # A < 0: now the second ratio (0.6 < 0.8) is the clipped one


def test_encode_ids_adds_the_end_of_turn_only_when_missing():
    class Tok:
        def apply_chat_template(self, *a, **k):
            return "P"

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [1, 2]}

        def convert_tokens_to_ids(self, t):
            return 99
    ids, labels, mu = tr.encode_ids(Tok(), "q", [5, 6], [-0.1, -0.2])
    assert ids == [1, 2, 5, 6, 99] and labels == [-100, -100, 5, 6, 99] and mu == [-0.1, -0.2, None]
    ids, _, mu = tr.encode_ids(Tok(), "q", [5, 99], [-0.1, -0.3])
    assert ids == [1, 2, 5, 99] and mu == [-0.1, -0.3]
