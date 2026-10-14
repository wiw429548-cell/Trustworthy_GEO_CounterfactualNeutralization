"""Counterfactual influence gap from engine generations.

M(d) = visibility(d | original context) - visibility(N(d) | same context with d replaced by its neutral version),
averaged over samples (= source orders). Positive M = influence that does not come from the facts of d.
"""
from collections import defaultdict

import numpy as np

from .utils import DATA, read_jsonl, slug


def load_gens(engine, file="gens.jsonl", metric="shares", n_samples=None):
    """metric: shares (PAWC, main) | shares_cite (citation counts) | shares_wc (word count, no position decay) |
    shares_strict (GEO's PAWC exactly as in Aggarwal et al.: only sentences with an explicit [i] citation count, no
    word-overlap attribution of uncited sentences; recomputed from the stored answer and source order)"""
    shares = defaultdict(list)   # (qid, cond, ctx) -> list of share vectors
    answers = {}
    p = DATA / "engine" / slug(engine) / file
    if not p.exists():
        return {}, {}
    if metric == "shares_strict":
        from .engine import pawc_shares
    for r in read_jsonl(p):
        if n_samples is not None and r["sample"] >= n_samples:
            continue
        if metric == "shares_strict":
            v = pawc_shares(r["answer"], len(r["shares"]), r["order"], None)
        else:
            v = r.get(metric, r["shares"])
        shares[(r["qid"], r["cond"], r["ctx"])].append(v)
        if r["sample"] == 0:
            answers[(r["qid"], r["cond"], r["ctx"])] = r["answer"]
    mean = {k: np.mean(np.array(v), 0) for k, v in shares.items()}
    return mean, answers


def gaps(engine, queries, **kw):
    """-> dict (qid, cond) -> {'M', 'vis_plain', 'vis_neutral'} for the target slot,
           dict qid -> {'j', 'M'} for one benign non-target source"""
    mean, _ = load_gens(engine, **kw)
    by_q = defaultdict(list)
    for key in mean:
        by_q[key[0]].append(key)
    tgt, oth = {}, {}
    for q in queries:
        t = q["target"]
        keys = by_q.get(q["qid"], [])
        for c in {k[1] for k in keys}:
            p, n = mean.get((q["qid"], c, "plain")), mean.get((q["qid"], c, "neutral_target"))
            if p is None or n is None:
                continue
            tgt[(q["qid"], c)] = {"M": float(p[t] - n[t]), "vis_plain": float(p[t]), "vis_neutral": float(n[t])}
        for (qid, c, ctx) in keys:
            if ctx.startswith("benign_n") and (qid, "benign", "plain") in mean:
                j = int(ctx[len("benign_n"):])
                oth[qid] = {"j": j, "M": float(mean[(qid, "benign", "plain")][j] - mean[(qid, c, ctx)][j])}
    return tgt, oth, mean
