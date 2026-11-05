"""Prompt-level improvers for RIVE's baselines (docs/rive-design.md section 6.27): OPRO (Yang et al. 2023, arXiv
2309.03409) and APE (Zhou et al. 2022, arXiv 2211.01910), as PenguinHarness specifies them in its
opro-rsi-toolkit and ape-rsi-toolkit skills (github.com/Prism-Shadow/penguin-harness, Apache-2.0). The model and
the scorer are passed in, so the same code serves any domain:

  propose(prompt) -> text           one call of the optimizer model (RIVE: the 4B itself, so the result is self-made)
  score(instruction) -> 0-100       the target's pass rate on the selection problems with this instruction

The instruction is the text appended to every problem's prompt (T2's `instruction` setting, at most 500
characters); everything else about the target stays fixed. Departures from PenguinHarness, besides running the
target as one model call instead of a tool-using agent: the optimizer is called directly, so their "call no tools"
prefix is left out; the score is a pass rate, not a rubric.
"""
import random

OPRO_SEED = "Let's solve the problem."
MAX_INSTRUCTION = 500

OPRO_HEAD = ("Your task is to generate the instruction <INS>. Below are some previous instructions with their scores. "
             "The score ranges from 0 to 100.\n\n")
OPRO_TAIL = ("Generate an instruction that is different from all the instructions <INS> above, and has a higher score "
             "than all the instructions <INS> above. The instruction should begin with <INS> and end with </INS>. "
             "The instruction should be concise, effective, and generally applicable to all problems above.")
APE_HEAD = "I gave a friend an instruction. Based on the instruction they produced the following input-output pairs:\n\n"
APE_TAIL = "The instruction was to"


def _key(text):
    """Exact duplicates up to surrounding and repeated whitespace."""
    return " ".join(text.split())


# ---- OPRO -------------------------------------------------------------------------------------------------------

def opro_meta_prompt(history, exemplars, answers=None, threshold=0.0, max_num_instructions=20):
    """history: [(instruction, score)] in measuring order; exemplars: problem texts; answers: their gold answers
    (PenguinHarness's few_shot_qa_pairs), or None."""
    kept = sorted(history, key=lambda h: h[1])[-max_num_instructions:]  # stable: ties keep measuring order
    kept = [h for h in kept if h[1] >= threshold]
    parts = [OPRO_HEAD]
    for ins, sc in kept:
        parts.append(f"text:\n{ins}\nscore:\n{round(sc)}\n\n")
    parts.append("Below are some problems.\n\n")
    for i, problem in enumerate(exemplars):
        parts.append(f"Problem:\n<INS>\n{problem}\n\n")
        if answers is not None:
            parts.append(f"Ground truth answer:\n{answers[i]}\n\n")
    parts.append(OPRO_TAIL)
    return "".join(parts)


def parse_instruction(reply):
    """The text between <INS> and </INS> (from the start or to the end where a tag is missing), stripped, without
    "**", first letter capitalized; None for a reply with no text."""
    if reply is None or not reply.strip():
        return None
    text = reply
    if "<INS>" in text:
        text = text.split("<INS>", 1)[1]
    if "</INS>" in text:
        text = text.split("</INS>", 1)[0]
    text = text.strip().replace("**", "").strip()
    return text[:1].upper() + text[1:]


def why_rejected(ins, seen):
    """None if the instruction may be measured, else the reason (PenguinHarness's skip rules)."""
    if ins is None:
        return "no text"
    if _key(ins) in seen:
        return "duplicate"
    if len(ins) > MAX_INSTRUCTION:
        return "longer than 500 characters"
    if "INS" in ins:
        return "contains INS"
    return None


def opro(propose, score, problems, steps=2, per_step=2, k=3, answers=None, threshold=0.0, max_num_instructions=20,
         seed_instruction=OPRO_SEED, log=print):
    """problems: {id: problem text}, the exemplar pool; answers: {id: gold answer}, or None. Exemplars are drawn as
    PenguinHarness draws them: random.Random(step).sample(sorted ids, k), shown in id order. Returns the best
    instruction (ties: the earliest, so the seed beats a tying candidate), its score and the whole record."""
    history = [(seed_instruction, score(seed_instruction))]
    seen = {_key(seed_instruction)}
    record = {"measured": [{"step": 0, "instruction": seed_instruction, "score": history[0][1]}], "rejected": []}
    log(f"opro: seed scores {history[0][1]:.1f}")
    ids = sorted(problems)
    for step in range(1, steps + 1):
        picked = sorted(random.Random(step).sample(ids, min(k, len(ids))))
        meta = opro_meta_prompt(history, [problems[i] for i in picked],
                                [answers[i] for i in picked] if answers is not None else None,
                                threshold, max_num_instructions)
        batch = [parse_instruction(propose(meta)) for _ in range(per_step)]
        for ins in batch:
            reason = why_rejected(ins, seen)
            if reason:
                record["rejected"].append({"step": step, "instruction": ins, "reason": reason})
                continue
            seen.add(_key(ins))
            sc = score(ins)
            history.append((ins, sc))
            record["measured"].append({"step": step, "instruction": ins, "score": sc})
            log(f"opro: step {step}: {sc:.1f}  {ins[:80]!r}")
    best = max(range(len(history)), key=lambda i: (history[i][1], -i))
    return {"best": history[best][0], "best_score": history[best][1], "seed_score": history[0][1], **record}


# ---- APE --------------------------------------------------------------------------------------------------------

def ape_prompt(demos):
    """demos: [(input, output)] in drawn order."""
    return APE_HEAD + "".join(f"Input: {i}\nOutput: {o}\n\n" for i, o in demos) + APE_TAIL


def ape_clean(reply):
    text = (reply or "").strip()
    if text.startswith(APE_TAIL):
        text = text[len(APE_TAIL):].strip()
    return text or None


def ape(propose, score, demos, num_subsamples=1, num_demos=5, per_subsample=4, seed=0, baseline="", log=print):
    """demos: [(input, output)] pairs taken out of the selection problems. One pass: generate every candidate,
    then measure each; the best is kept even below the baseline (reported as a regression)."""
    pool, seen, subsets = [], set(), []
    for i in range(1, num_subsamples + 1):
        picked = random.Random(seed + i).sample(range(len(demos)), min(num_demos, len(demos)))
        subsets.append(picked)
        prompt = ape_prompt([demos[j] for j in picked])
        for _ in range(per_subsample):
            ins = ape_clean(propose(prompt))
            if ins is not None and len(ins) <= MAX_INSTRUCTION and _key(ins) not in seen:
                seen.add(_key(ins))
                pool.append(ins)
    base = score(baseline)
    log(f"ape: baseline scores {base:.1f}; {len(pool)} candidates")
    scores = []
    for ins in pool:
        scores.append(score(ins))
        log(f"ape: {scores[-1]:.1f}  {ins[:80]!r}")
    if not pool:
        return {"best": baseline, "best_score": base, "baseline_score": base, "candidates": [], "subsets": subsets,
                "regression": False, "note": "no valid candidate"}
    best_score, best = sorted(zip(scores, pool), reverse=True)[0]  # ties: the instruction text descending
    return {"best": best, "best_score": best_score, "baseline_score": base, "subsets": subsets,
            "candidates": [{"instruction": p, "score": s} for p, s in zip(pool, scores)],
            "regression": best_score < base}
