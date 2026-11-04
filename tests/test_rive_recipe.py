import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import rive_recipe  # noqa: E402

TRAINED = {"self-s1": 0.12, "gentle-s1": 0.2}


def served(base=0.3, self_s1=0.15, gentle=0.22):
    return {"qwen3.5-4b": base, "self-s1": self_s1, "gentle-s1": gentle}


def test_a_served_adapter_that_does_not_lower_its_training_loss_is_case_p():
    r = rive_recipe.reading([38, 38], [27, 28], [38, 37], [112, 113], served(self_s1=0.25), TRAINED)
    assert not r["faithful"] and r["case"] == "P"


def test_the_cases_follow_the_dev_gaps():
    assert rive_recipe.reading([38, 38], [27, 28], [37, 36], [112, 113], served(), TRAINED)["case"] == "R1"
    assert rive_recipe.reading([38, 38], [27, 28], [30, 31], [112, 113], served(), TRAINED)["case"] == "R2"
    assert rive_recipe.reading([38, 38], [35, 34], [30, 31], [112, 113], served(), TRAINED)["case"] == "R3"


def test_the_select_gain_is_against_the_base_mean():
    r = rive_recipe.reading([38, 38], [27, 28], [37, 36], [127, 129], served(), TRAINED)
    assert r["useful"] and abs(r["gain"] - (128 - 112 - 2 / 3)) < 1e-9
