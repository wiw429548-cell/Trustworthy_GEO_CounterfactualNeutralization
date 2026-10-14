"""Human validation sheets (exported automatically; fill them in and run `python -m neutralens.annotate score`).

neutralization.csv : (original, neutral) pairs -> facts_kept (1/0), rhetoric_removed (1/0), content_added (1/0)
manipulation.csv   : documents (benign / rewritten) -> manipulative (1/0)
                     -> validates that the rewrite families are perceived as intended
The family of every row is stored only in *_key.csv (do NOT give the key files to annotators).
"""
import argparse
import csv
import random

import numpy as np

from .build import G, h
from .utils import ROOT, load_config, read_jsonl, save_json

OUT = ROOT / "results" / "annotation"


def export(cfg, n_neutral=150, n_manip=200):
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg["seed"])
    neutral = {r["hash"]: r["neutral"] for r in read_jsonl(G / "neutral.jsonl")}
    qs = {q["qid"]: q for q in read_jsonl(G / "queries.jsonl") if q["split"] == "test"}
    var = [v for v in read_jsonl(G / "variants.jsonl") if v["qid"] in qs]
    docs = [("benign", q["qid"], q["sources"][q["target"]]) for q in qs.values()] + [(v["family"], v["qid"], v["text"]) for v in var]
    rng.shuffle(docs)
    with open(OUT / "neutralization.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "query", "original", "neutral", "facts_kept", "rhetoric_removed", "content_added"])
        sel = [d for d in docs if h(d[2]) in neutral][:n_neutral]
        for k, (fam, qid, t) in enumerate(sel):
            w.writerow([k, qs[qid]["query"], t[:2500], neutral[h(t)][:2500], "", "", ""])
    _key(sel, "neutralization_key.csv")
    with open(OUT / "manipulation.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "query", "document", "manipulative"])
        for k, (fam, qid, t) in enumerate(docs[:n_manip]):
            w.writerow([k, qs[qid]["query"], t[:2500], ""])
    _key(docs[:n_manip], "manipulation_key.csv")


def _key(rows, name):
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "family", "qid"])
        for k, (fam, qid, _) in enumerate(rows):
            w.writerow([k, fam, qid])


def _rows(name, col):
    p = OUT / name
    if not p.exists():
        return []
    key = {r["id"]: r["family"] for r in csv.DictReader(open(OUT / name.replace(".csv", "_key.csv"), encoding="utf-8"))}
    rows = [r for r in csv.DictReader(open(p, encoding="utf-8")) if r.get(col, "").strip() in ("0", "1")]
    for r in rows:
        r["family"] = key[r["id"]]
    return rows


def score():
    res = {}
    r = _rows("neutralization.csv", "facts_kept")
    if r:
        res["neutralization"] = {c: float(np.mean([int(x[c]) for x in r if x[c].strip() in ("0", "1")]))
                                 for c in ("facts_kept", "rhetoric_removed", "content_added")} | {"n": len(r)}
    r = _rows("manipulation.csv", "manipulative")
    if r:
        fams = sorted({x["family"] for x in r})
        res["manipulation_perceived"] = {f: float(np.mean([int(x["manipulative"]) for x in r if x["family"] == f]))
                                         for f in fams} | {"n": len(r)}
    save_json(res, OUT / "human_validation.json")
    print(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["export", "score"])
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    export(load_config(a.config)) if a.action == "export" else score()


if __name__ == "__main__":
    main()
