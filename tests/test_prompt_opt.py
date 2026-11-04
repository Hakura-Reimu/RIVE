from rive.meta_rsi import prompt_opt as po


def test_meta_prompt_sorts_history_and_shows_exemplars():
    meta = po.opro_meta_prompt([("b", 70.4), ("a", 20.0), ("c", 55.5)], ["P1", "P2"], answers=["1", "2"])
    assert meta.startswith(po.OPRO_HEAD) and meta.endswith(po.OPRO_TAIL)
    assert meta.index("text:\na\nscore:\n20") < meta.index("text:\nc\nscore:\n56") < meta.index("text:\nb\nscore:\n70")
    assert "Problem:\n<INS>\nP1\n\nGround truth answer:\n1" in meta
    assert "Ground truth" not in po.opro_meta_prompt([("a", 1)], ["P1"])
    # at most max_num_instructions, the best ones; below the threshold left out
    meta = po.opro_meta_prompt([(str(i), i) for i in range(30)], [], threshold=12, max_num_instructions=20)
    assert "text:\n9\n" not in meta and "text:\n12\n" in meta and "text:\n29\n" in meta


def test_parse_instruction():
    assert po.parse_instruction("blah <INS>think **carefully** first.</INS> more") == "Think carefully first."
    assert po.parse_instruction("no tags at all") == "No tags at all"
    assert po.parse_instruction("<INS>") == ""  # an explicit empty instruction is valid
    assert po.parse_instruction("   ") is None
    seen = {po._key("Think  carefully")}
    assert po.why_rejected("Think carefully", seen) == "duplicate"
    assert po.why_rejected("x" * 501, seen) == "longer than 500 characters"
    assert po.why_rejected("Use INS here", seen) == "contains INS"
    assert po.why_rejected("Fresh", seen) is None


def test_opro_keeps_the_best_and_the_earliest_on_ties():
    replies = iter(["<INS>alpha</INS>", "<INS>beta</INS>", "<INS>alpha</INS>", "<INS>gamma</INS>"])
    scores = {po.OPRO_SEED: 40, "Alpha": 60, "Beta": 30, "Gamma": 60}
    seen_prompts = []

    def propose(prompt):
        seen_prompts.append(prompt)
        return next(replies)
    out = po.opro(propose, lambda ins: scores[ins], {"p1": "P1", "p2": "P2", "p3": "P3", "p4": "P4"}, steps=2,
                  per_step=2, k=2, log=lambda m: None)
    assert out["best"] == "Alpha" and out["best_score"] == 60 and out["seed_score"] == 40
    assert [r["reason"] for r in out["rejected"]] == ["duplicate"]
    assert len(out["measured"]) == 4  # seed, alpha, beta, gamma
    assert "text:\nAlpha\nscore:\n60" in seen_prompts[2]  # step 2 sees step 1's scores


def test_ape_generates_once_and_keeps_the_best_even_below_the_baseline():
    replies = iter(["The instruction was to add the numbers.", "add the numbers.", "  ", "multiply them."])
    out = po.ape(lambda p: next(replies), lambda ins: {"": 80, "add the numbers.": 50, "multiply them.": 20}[ins],
                 [("1 2", "3"), ("2 2", "4"), ("5 1", "6")], num_subsamples=1, num_demos=2, per_subsample=4,
                 log=lambda m: None)
    assert [c["instruction"] for c in out["candidates"]] == ["add the numbers.", "multiply them."]
    assert out["best"] == "add the numbers." and out["regression"] is True and out["baseline_score"] == 80
    prompt = po.ape_prompt([("1 2", "3")])
    assert prompt.startswith(po.APE_HEAD) and "Input: 1 2\nOutput: 3\n\n" in prompt and prompt.endswith(po.APE_TAIL)
