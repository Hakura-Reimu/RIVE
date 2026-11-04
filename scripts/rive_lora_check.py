"""Does the e1 edit work as a LoRA adapter on the original model (docs/rive-design.md section 6.15)? Against the
FP8 server started with the adapter (scripts/rive_lora_check_chain.sh), on T2's development pack:
  - lora_think: thinking at 8192 tokens through the adapter, against 6.13's e1 (the edited copy);
  - lora_plain: no thinking and no adapter (the model card's sampling, as 6.12's a0_card), against the plain server;
  - lora_mixed: 8 tasks of each at the same time, to see that requests with and without the adapter can share a batch.
  python scripts/rive_lora_check.py run lora_think lora_plain lora_mixed
  python scripts/rive_lora_check.py report
Writes <out-dir>/<run>.json (default ~/alter/rive/loracheck) and results/rive_lora_check.json."""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import rive_cutoff_diag as diag  # noqa: E402
import rive_think_fix as fix  # noqa: E402

ADAPTER = "nowait-e1"
THINK_CAPS = (8192, None, 12, 0, False)
PLAIN_CAP = 4096
# What the runs are compared with (6.13 and the FP8 check of 6.12, same pack, same settings).
E1 = {"solved": (35, 41), "cut": (17, 23), "mean_tokens": 4837, "recheck_lines_per_1k": 0.02,
      "tokens_per_second": round(357965 / 639)}
PLAIN = {"solved": 32, "mean_tokens": 1510}
SLACK, TOKENS = 5, 0.15


def think(client, tokenizer, task):
    return fix.generate(client, tokenizer, task, THINK_CAPS, {}, model=ADAPTER)


def plain(client, task):
    return {**diag.generate(client, task, False, PLAIN_CAP, diag.CARD_INSTRUCT), "forced": False}


def run(name, tasks, out):
    from openai import OpenAI
    from transformers import AutoTokenizer
    client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY", timeout=1800)
    names = [m.id for m in client.models.list().data]
    assert ADAPTER in names, names
    tokenizer = AutoTokenizer.from_pretrained(fix.MODEL)
    start = time.time()
    with ThreadPoolExecutor(max_workers=12) as pool:
        if name == "lora_think":
            rows = [dict(r, thinking=True) for r in pool.map(lambda t: think(client, tokenizer, t), tasks)]
        elif name == "lora_plain":
            rows = [dict(r, thinking=False) for r in pool.map(lambda t: plain(client, t), tasks)]
        else:
            a = [pool.submit(think, client, tokenizer, t) for t in tasks[:8]]
            b = [pool.submit(plain, client, t) for t in tasks[:8]]
            rows = [dict(f.result(), thinking=True) for f in a] + [dict(f.result(), thinking=False) for f in b]
    by_id = {t["id"]: t for t in tasks}
    with ThreadPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(lambda r: diag.judge(by_id[r["id"]], r, r["thinking"]), rows))
    seconds = time.time() - start  # with the judge, as the run times of 6.13
    (out / f"{name}.json").write_text(json.dumps({"seconds": seconds, "rows": rows}))
    print(f"{time.strftime('%F %T')} run {name}: {seconds:.0f} s, {json.dumps(summary(rows, seconds))}", flush=True)


def summary(rows, seconds):
    s = fix.summary(rows)
    s["tokens_per_second"] = round(sum(r["tokens"] for r in rows) / seconds)
    return s


def report(out):
    res = {}
    for name in ("lora_think", "lora_plain", "lora_mixed"):
        if (out / f"{name}.json").exists():
            d = json.loads((out / f"{name}.json").read_text())
            res[name] = summary(d["rows"], d["seconds"])
    if "lora_think" in res:
        s = res["lora_think"]
        res["same_as_e1"] = {
            "recheck_gone": s["recheck_lines_per_1k"] <= 0.1,
            "tokens": abs(s["mean_tokens"] / E1["mean_tokens"] - 1) <= TOKENS,
            "solved": E1["solved"][0] - SLACK <= s["solved"] <= E1["solved"][1] + SLACK,
            "cut": s["cut"] <= 30,
            "speed_vs_merged": round(s["tokens_per_second"] / E1["tokens_per_second"], 2)}
        res["same_as_e1"]["all"] = all(v for k, v in res["same_as_e1"].items() if k != "speed_vs_merged")
    if "lora_plain" in res:
        s = res["lora_plain"]
        res["plain_unchanged"] = {"solved": abs(s["solved"] - PLAIN["solved"]) <= SLACK,
                                  "tokens": abs(s["mean_tokens"] / PLAIN["mean_tokens"] - 1) <= TOKENS}
    if "lora_mixed" in res:
        d = json.loads((out / "lora_mixed.json").read_text())["rows"]
        res["mixed"] = {"think_recheck_per_1k": fix.summary([r for r in d if r["thinking"]])["recheck_lines_per_1k"],
                        "plain_recheck_per_1k": fix.summary([r for r in d if not r["thinking"]])[
                            "recheck_lines_per_1k"]}
    (ROOT / "results/rive_lora_check.json").write_text(json.dumps(res, indent=1) + "\n")
    print("report", json.dumps(res), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=("run", "report"))
    p.add_argument("names", nargs="*")
    p.add_argument("--out-dir", default=str(Path.home() / "alter/rive/loracheck"))
    args = p.parse_args()
    tasks = json.loads(diag.PACKS.read_text())["dev"]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.step == "run":
        for name in args.names:
            if not (out / f"{name}.json").exists():
                run(name, tasks, out)
    report(out)


if __name__ == "__main__":
    main()
