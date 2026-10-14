"""LLM judges on engine answers (TEST queries, sample 0 of every context).

mode quality (utility of defences): answer quality 1-10 against the ORIGINAL (benign) sources of the query
    --conds all          : benign + every rewrite family (main engine: utility on clean AND on legitimately optimised
                           pages, and quality under attack)
    --conds benign_legit : benign + legitimate-optimisation families only (other engines; cheaper)
    every context of the gens file is judged except benign_n* (detection negatives)
    -> data/judge/<out>.jsonl   (out defaults to <engine slug>; baselines file -> <slug>_baselines)
mode si (GEO's subjective impression, Aggarwal et al. KDD'24): 7 sub-metrics (relevance, influence, uniqueness,
    subjective position, subjective count, click likelihood, diversity) of the TARGET source in the answer, rated 0-10
    in one call and averaged to [0, 1]; contexts plain / neutral_target, every condition, the first
    `si.n_queries` test queries -> data/judge/<slug>_si.jsonl
Unparsable judge outputs are stored as None and excluded; parse-failure rates in data/usage/judge_<out>.json
"""
import argparse
import re

from . import prompts as P
from .build import G, h, test_subset
from .utils import DATA, load_config, read_jsonl, save_json, slug, write_jsonl


def _quality(cfg, a, llm, qd):
    legit = set(cfg["families"]["legit"])
    keep_cond = (lambda c: True) if a.conds == "all" else (lambda c: c == "benign" or c in legit)
    gens = [r for r in read_jsonl(DATA / "engine" / slug(a.engine) / a.gens)
            if r["qid"] in qd and r["sample"] == 0 and keep_cond(r["cond"]) and not r["ctx"].startswith("benign_n")]
    prompts = []
    for r in gens:
        q = qd[r["qid"]]
        src = "\n\n".join(f"[{i + 1}] {s}" for i, s in enumerate(q["sources"]) if s)
        prompts.append(P.QUALITY.format(query=q["query"], sources=src, answer=r["answer"]))
    outs = llm.chat(prompts, max_tokens=8, tag="quality")
    rows, bad = [], 0
    for r, o in zip(gens, outs):
        m = re.search(r"(\d+(?:\.\d+)?)", o)
        sc = float(m.group(1)) if m else None
        if sc is not None and not 1 <= sc <= 10:
            sc = None
        bad += sc is None
        rows.append({"qid": r["qid"], "cond": r["cond"], "ctx": r["ctx"], "score": sc, "raw": o[:40]})
    return rows, bad


SI_KEYS = ["REL", "INF", "UNI", "POS", "CNT", "CLK", "DIV"]


def _si(cfg, a, llm, qd):
    n = cfg.get("si", {}).get("n_queries", 150)
    ids = {q["qid"] for q in test_subset(list(qd.values()), n)}
    var = {(v["qid"], v["family"]): v["text"] for v in read_jsonl(G / "variants.jsonl")}
    neutral = {r["hash"]: r["neutral"] for r in read_jsonl(G / "neutral.jsonl")}
    gens, prompts = [], []
    for r in read_jsonl(DATA / "engine" / slug(a.engine) / a.gens):
        if r["qid"] not in ids or r["sample"] != 0 or r["ctx"] not in ("plain", "neutral_target"):
            continue
        q = qd[r["qid"]]
        t = q["target"]
        if t not in r["order"] or (r["cond"] != "benign" and (r["qid"], r["cond"]) not in var):
            continue
        doc = q["sources"][t] if r["cond"] == "benign" else var[(r["qid"], r["cond"])]
        shown = doc if r["ctx"] == "plain" else neutral.get(h(doc), doc)
        idx = r["order"].index(t) + 1
        gens.append(r)
        prompts.append(P.SUBJ_IMPRESSION.format(idx=idx, query=q["query"], source=" ".join(shown.split()[:400]),
                                                answer=r["answer"]))
    outs = llm.chat(prompts, max_tokens=60, tag="si")
    rows, bad = [], 0
    for r, o in zip(gens, outs):
        vals = {k: float(m.group(1)) for k in SI_KEYS for m in [re.search(rf"{k}\s*=\s*(\d+(?:\.\d+)?)", o)] if m}
        ok = len(vals) == len(SI_KEYS) and all(0 <= v <= 10 for v in vals.values())
        bad += not ok
        rows.append({"qid": r["qid"], "cond": r["cond"], "ctx": r["ctx"],
                     "si": sum(vals.values()) / (10 * len(SI_KEYS)) if ok else None, "sub": vals if ok else None})
    return rows, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--mode", default="quality", choices=["quality", "si"])
    ap.add_argument("--gens", default="gens.jsonl", help="engine output file to judge")
    ap.add_argument("--out", default=None, help="output name (data/judge/<out>.jsonl)")
    ap.add_argument("--conds", default="all", choices=["all", "benign_legit"])
    a = ap.parse_args()
    cfg = load_config(a.config)
    from .llm import LLM
    out = a.out or (slug(a.engine) + ("_si" if a.mode == "si" else ""))
    qd = {q["qid"]: q for q in read_jsonl(G / "queries.jsonl") if q["split"] == "test"}
    llm = LLM(cfg["models"]["judge"], max_model_len=cfg["models"]["max_model_len"], gpu_mem=cfg["models"]["gpu_mem"])
    rows, bad = (_si if a.mode == "si" else _quality)(cfg, a, llm, qd)
    write_jsonl(rows, DATA / "judge" / f"{out}.jsonl")
    save_json({"judge": cfg["models"]["judge"], "mode": a.mode, "gens": a.gens, "n": len(rows),
               "parse_fail_rate": bad / max(1, len(rows)), **llm.usage_snapshot()},
              DATA / "usage" / f"judge_{out}.json")


if __name__ == "__main__":
    main()
