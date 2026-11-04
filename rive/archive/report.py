"""Per-generation report built only from archive records (handoff doc 7.1, docs/phase-c-design.md section 7)."""

VERDICTS = {"supported": "支持", "refuted": "驳倒", "inconclusive": "看不出"}


def generation_report(archive, generation, task_set):
    """task_set: the generation's shared development tasks, on which every candidate and the champion ran."""
    records = archive.experiment_records()
    rows = []
    for c in archive.candidates(generation):
        ev = archive.evaluation(c["id"], task_set)
        rec = records.get(c["id"], {})
        rows.append({"id": c["id"], "parent": c["parent_id"], "parent_source": c["parent_source"],
                     "origin": c["origin"], "module": c["module"], "status": c["status"], "note": c["note"],
                     "macro": ev["summary"]["macro"] if ev else None,
                     "by_domain": ev["summary"]["by_domain"] if ev else None,
                     "domains": ev["settings"].get("domains") if ev else None,
                     "vs_champion": ev["vs_reference"] if ev else None,
                     "tokens": ev["summary"]["tokens"] if ev else None,
                     "verdict": rec.get("verdict"), "prediction_hit": rec.get("prediction_check", {}).get("hit")})
    events = [e for e in archive.events() if e["generation"] == generation]
    return {"generation": generation, "task_set": task_set, "champion_after": archive.champion(),
            "candidates": rows, "events": [{"kind": e["kind"], **e["detail"]} for e in events]}


def _pct(x):
    return "—" if x is None else f"{100 * x:.1f}%"


def to_markdown(report):
    lines = [f"# 第 {report['generation']} 代", "",
             f"题目批次 `{report['task_set']}`；本代结束时的冠军：`{report['champion_after']}`。", "",
             "| 候选 | 亲代（来源） | 改动模块 | 状态 | 判定领域 | 得分 | 比冠军 | 结论 | 预测 | token |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in report["candidates"]:
        vs = "—" if r["vs_champion"] is None else f"{100 * r['vs_champion']:+.1f} 个百分点"
        hit = {True: "对", False: "错"}.get(r.get("prediction_hit"), "—")
        lines.append(f"| `{r['id']}` | `{r['parent']}`（{r['parent_source'] or r['origin']}） | {r['module'] or '—'} "
                     f"| {r['status']} | {', '.join(r.get('domains') or []) or '—'} | {_pct(r['macro'])} | {vs} "
                     f"| {VERDICTS.get(r.get('verdict'), '—')} | {hit} | {r['tokens'] or '—'} |")
    judged = [r for r in report["candidates"] if r.get("verdict")]
    if judged:
        hits = [r.get("prediction_hit") for r in judged]
        lines += ["", f"预测的方向：对 {hits.count(True)} 个，错 {hits.count(False)} 个，没变或测不了 {hits.count(None)} 个。"]
    if report["events"]:
        lines += ["", "事件：", ""] + [f"- {e['kind']}：" + "，".join(f"{k} = {v}" for k, v in e.items() if k != "kind")
                                     for e in report["events"]]
    return "\n".join(lines) + "\n"
