"""Fixed assignment of every evaluation item to exactly one use (handoff doc 3.4)."""
import hashlib

SPLITS = (("dev", 0.40), ("selection", 0.70), ("anchor", 0.80), ("sealed", 1.00))
SALT = "alter-phaseA-v1"


def split_of(item_id: str, salt: str = SALT) -> str:
    h = int.from_bytes(hashlib.sha256(f"{salt}:{item_id}".encode()).digest()[:8], "big") / 2**64
    for name, upper in SPLITS:
        if h < upper:
            return name
    return SPLITS[-1][0]
