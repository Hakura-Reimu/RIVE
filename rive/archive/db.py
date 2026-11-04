"""Evolution archive (handoff doc 3.5, docs/phase-c-design.md section 7): every candidate, experiment, evaluation
and per-item result, with lineage, in one SQLite file. Nothing is deleted; failed candidates stay queryable."""
import json
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS candidates (
    id TEXT PRIMARY KEY, parent_id TEXT REFERENCES candidates(id), generation INTEGER NOT NULL,
    created REAL NOT NULL, dir TEXT NOT NULL, origin TEXT NOT NULL, parent_source TEXT, module TEXT,
    code_hash TEXT, status TEXT NOT NULL, note TEXT);
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT, candidate_id TEXT NOT NULL REFERENCES candidates(id),
    created REAL NOT NULL, record TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT, candidate_id TEXT NOT NULL REFERENCES candidates(id),
    task_set TEXT NOT NULL, split TEXT NOT NULL, created REAL NOT NULL, settings TEXT NOT NULL,
    summary TEXT NOT NULL, reference_id TEXT REFERENCES candidates(id), vs_reference REAL);
CREATE TABLE IF NOT EXISTS item_results (
    evaluation_id INTEGER NOT NULL REFERENCES evaluations(id), item_id TEXT NOT NULL, domain TEXT NOT NULL,
    correct INTEGER NOT NULL, failure TEXT, tokens INTEGER, seconds REAL, PRIMARY KEY (evaluation_id, item_id));
CREATE TABLE IF NOT EXISTS tools (
    name TEXT NOT NULL, version INTEGER NOT NULL, candidate_id TEXT NOT NULL REFERENCES candidates(id),
    created REAL NOT NULL, source_hash TEXT NOT NULL, schema TEXT NOT NULL, tests_passed INTEGER NOT NULL,
    PRIMARY KEY (name, version));
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, generation INTEGER, kind TEXT NOT NULL,
    detail TEXT NOT NULL);
"""


def summarize(results):
    """Overall accuracy, the domain-balanced mean (the Phase C primary metric) and per-domain accuracy."""
    by_domain = defaultdict(list)
    for r in results:
        by_domain[r["domain"]].append(r["correct"])
    per = {d: sum(v) / len(v) for d, v in sorted(by_domain.items())}
    return {"n": len(results), "accuracy": sum(r["correct"] for r in results) / len(results),
            "macro": sum(per.values()) / len(per), "by_domain": per,
            "tokens": sum(r.get("tokens") or 0 for r in results)}


class Archive:
    def __init__(self, path):
        self.path = Path(path)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def close(self):
        self.db.close()

    # ---------- writing ----------

    def add_candidate(self, cid, parent_id, generation, directory, origin, parent_source=None, module=None,
                      code_hash=None, status="created", note=None):
        """origin: how it was made (seed / research / random_template / direct_proposal);
        parent_source: how its parent was chosen (trunk / archive)."""
        with self.db:
            self.db.execute("INSERT INTO candidates VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                            (cid, parent_id, generation, time.time(), str(directory), origin, parent_source, module,
                             code_hash, status, note))

    def set_status(self, cid, status, note=None):
        with self.db:
            self.db.execute("UPDATE candidates SET status = ?, note = COALESCE(?, note) WHERE id = ?",
                            (status, note, cid))

    def add_experiment(self, cid, record):
        with self.db:
            return self.db.execute("INSERT INTO experiments (candidate_id, created, record) VALUES (?,?,?)",
                                   (cid, time.time(), json.dumps(record, ensure_ascii=False))).lastrowid

    def add_evaluation(self, cid, task_set, split, results, settings=None, reference_id=None, vs_reference=None):
        """results: [{"id", "domain", "correct", "failure"?, "tokens"?, "seconds"?}] in any order.
        With reference_id, the difference against that candidate is stored: vs_reference when given (the gate's
        own measure), otherwise the macro difference against its evaluation of the same task set (the champion is
        always evaluated on the same tasks, docs/phase-c-design.md section 4)."""
        summary = summarize(results)
        vs = vs_reference
        if reference_id is not None and vs is None:
            ref = self.evaluation(reference_id, task_set)
            if ref is None:
                raise ValueError(f"{reference_id} has no evaluation on {task_set}")
            vs = summary["macro"] - ref["summary"]["macro"]
        with self.db:
            eid = self.db.execute(
                "INSERT INTO evaluations (candidate_id, task_set, split, created, settings, summary, reference_id,"
                " vs_reference) VALUES (?,?,?,?,?,?,?,?)",
                (cid, task_set, split, time.time(), json.dumps(settings or {}), json.dumps(summary), reference_id,
                 vs)).lastrowid
            self.db.executemany("INSERT INTO item_results VALUES (?,?,?,?,?,?,?)",
                                [(eid, r["id"], r["domain"], int(r["correct"]), r.get("failure"), r.get("tokens"),
                                  r.get("seconds")) for r in results])
        return eid

    def register_tool(self, name, cid, source_hash, schema, tests_passed):
        """New version of a tool; tools outlive the candidate that made them (handoff doc 3.2)."""
        with self.db:
            version = self.db.execute("SELECT COALESCE(MAX(version), 0) + 1 FROM tools WHERE name = ?",
                                      (name,)).fetchone()[0]
            self.db.execute("INSERT INTO tools VALUES (?,?,?,?,?,?,?)",
                            (name, version, cid, time.time(), source_hash, json.dumps(schema), int(tests_passed)))
        return version

    def log(self, kind, detail, generation=None):
        with self.db:
            self.db.execute("INSERT INTO events (ts, generation, kind, detail) VALUES (?,?,?,?)",
                            (time.time(), generation, kind, json.dumps(detail, ensure_ascii=False)))

    def promote(self, cid, generation, evidence):
        self.log("promotion", {"candidate": cid, **evidence}, generation)

    # ---------- reading ----------

    def candidate(self, cid):
        row = self.db.execute("SELECT * FROM candidates WHERE id = ?", (cid,)).fetchone()
        return dict(row) if row else None

    def candidates(self, generation=None):
        sql, args = "SELECT * FROM candidates", ()
        if generation is not None:
            sql, args = sql + " WHERE generation = ?", (generation,)
        return [dict(r) for r in self.db.execute(sql + " ORDER BY created", args)]

    def evaluation(self, cid, task_set):
        row = self.db.execute("SELECT * FROM evaluations WHERE candidate_id = ? AND task_set = ? ORDER BY id DESC",
                              (cid, task_set)).fetchone()
        if row is None:
            return None
        out = dict(row)
        out["summary"], out["settings"] = json.loads(out["summary"]), json.loads(out["settings"])
        return out

    def experiment_records(self):
        """{candidate: its latest experiment record}."""
        return {r["candidate_id"]: json.loads(r["record"])
                for r in self.db.execute("SELECT candidate_id, record FROM experiments ORDER BY id")}

    def item_results(self, evaluation_id):
        return [dict(r) for r in self.db.execute("SELECT * FROM item_results WHERE evaluation_id = ? ORDER BY item_id",
                                                 (evaluation_id,))]

    def events(self, kind=None):
        sql, args = "SELECT * FROM events", ()
        if kind is not None:
            sql, args = sql + " WHERE kind = ?", (kind,)
        return [{**dict(r), "detail": json.loads(r["detail"])} for r in self.db.execute(sql + " ORDER BY id", args)]

    def champion(self):
        """Latest promoted candidate, else the seed."""
        promos = self.events("promotion")
        if promos:
            return promos[-1]["detail"]["candidate"]
        row = self.db.execute("SELECT id FROM candidates WHERE origin = 'seed' ORDER BY created").fetchone()
        return row["id"] if row else None

    def promoted_at(self, cid):
        """When cid last became champion; 0 for a candidate never promoted, such as the seed."""
        return max((e["ts"] for e in self.events("promotion") if e["detail"]["candidate"] == cid), default=0.0)

    def runs(self, cid, split="dev", since=0.0):
        """Per-item results of each of cid's evaluations on `split` made at or after `since`, oldest first; each item
        carries its id under "id" as well, like the results that were stored."""
        rows = self.db.execute("SELECT id FROM evaluations WHERE candidate_id = ? AND split = ? AND created >= ? "
                               "ORDER BY id", (cid, split, since)).fetchall()
        return [[dict(r, id=r["item_id"]) for r in self.item_results(e["id"])] for e in rows]

    def evaluated(self, split="dev"):
        """Candidates with at least one evaluation on `split` (their code ran), oldest first."""
        rows = self.db.execute("SELECT c.id FROM candidates c WHERE EXISTS (SELECT 1 FROM evaluations e WHERE "
                               "e.candidate_id = c.id AND e.split = ?) ORDER BY c.created", (split,))
        return [r["id"] for r in rows]

    def ancestors(self, cid):
        """[cid, parent, grandparent, ...] up to the seed."""
        chain = []
        while cid is not None:
            chain.append(cid)
            cid = self.db.execute("SELECT parent_id FROM candidates WHERE id = ?", (cid,)).fetchone()["parent_id"]
        return chain

    def clade(self, cid):
        """cid and all its descendants."""
        rows = self.db.execute("WITH RECURSIVE c(id) AS (SELECT ? UNION ALL SELECT k.id FROM candidates k JOIN c "
                               "ON k.parent_id = c.id) SELECT id FROM c", (cid,))
        return [r["id"] for r in rows]

    def clade_stats(self, cid, split="dev"):
        """Wins and losses of the whole clade against the reference (champion) on the same tasks.
        HGM estimates a candidate's potential from its descendants, not from its own score
        (handoff doc 3.5); a Beta(1 + wins, 1 + losses) over these feeds Thompson sampling."""
        members = self.clade(cid)
        marks = ",".join("?" * len(members))
        rows = self.db.execute(f"SELECT vs_reference FROM evaluations WHERE candidate_id IN ({marks}) AND split = ? "
                               "AND vs_reference IS NOT NULL", (*members, split)).fetchall()
        wins = sum(r["vs_reference"] > 0 for r in rows)
        return {"members": len(members), "evaluated": len(rows), "wins": wins, "losses": len(rows) - wins}
