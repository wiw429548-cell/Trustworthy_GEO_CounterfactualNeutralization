"""Review-round analyses from the raw engine generations (no new model calls).

1. Decomposition robustness: presentation-borne share per group with
   (a) the second attacker (Phi-3.5-mini rewrites, main neutraliser) on Qwen2.5-7B and Qwen3-8B,
   (b) the second neutraliser (Qwen3-8B) on Qwen2.5-7B,
   (c) the main setting restricted to pages whose restatement keeps >= 80% of their atomic facts (NLI audit),
       compared with the same queries without the restriction.
2. Threshold-free defence effect: mean visibility gain of manipulative rewrites under no defence, NeutraLens-Replace
   (M > tau, applied to the original page as well, as in the ASR evaluation) and NeutraLens-All, with paired
   query-cluster bootstrap CIs of the reduction.
3. Legitimate minus manipulative presentation-borne gain (paired by query) per engine.
-> results/extra/robustness.json
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "results" / "extra"
CFG = yaml.safe_load(open(ROOT / "configs" / "default.yaml", encoding="utf-8"))
FAM = CFG["families"]
GROUP = {f: g for g in ("legit", "geo", "manipulative", "fact") for f in FAM[g]}
TAU = CFG["defense"]["m_threshold"]
B = 2000


def rj(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def slug(m):
    return m.replace("/", "_").replace("-", "_").replace(".", "_")


QS = {q["qid"]: q for q in rj(DATA / "geo" / "queries.jsonl")}
TEST = {k for k, q in QS.items() if q["split"] == "test"}


def mean_shares(path, ctxs=("plain", "neutral_target", "neutral_all")):
    per = defaultdict(list)
    for r in rj(path):
        if r["qid"] in TEST and r["ctx"] in ctxs:
            per[(r["qid"], r["cond"], r["ctx"])].append(r["shares"][QS[r["qid"]]["target"]])
    return {k: float(np.mean(v)) for k, v in per.items()}


def boot(by_q, f=np.mean, n=B, seed=0):
    qs = sorted(q for q in by_q if by_q[q])
    if not qs:
        return [None, None, None]
    rng = np.random.RandomState(seed)
    flat = lambda ss: [x for q in ss for x in by_q[q]]
    bs = [f(flat([qs[i] for i in rng.randint(0, len(qs), len(qs))])) for _ in range(n)]
    return [float(f(flat(qs))), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]


def ratio_boot(num, den, n=B, seed=1):
    qs = sorted(q for q in num if q in den)
    if not qs:
        return [None, None, None]
    f = lambda ss: np.sum([x for q in ss for x in num[q]]) / np.sum([x for q in ss for x in den[q]])
    rng = np.random.RandomState(seed)
    bs = [f([qs[i] for i in rng.randint(0, len(qs), len(qs))]) for _ in range(n)]
    return [float(f(qs)), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]


def decompose(V, invalid=frozenset(), keep=None, qset=None):
    """group -> share_P [mean, lo, hi], n pairs; keep(qid, cond) filters pairs"""
    G, P = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    n = defaultdict(int)
    for qid in sorted(TEST):
        if qset is not None and qid not in qset:
            continue
        bp, bn = V.get((qid, "benign", "plain")), V.get((qid, "benign", "neutral_target"))
        if bp is None or bn is None:
            continue
        for f, g in GROUP.items():
            if (qid, f) in invalid or (keep and not keep(qid, f)):
                continue
            fp, fn = V.get((qid, f, "plain")), V.get((qid, f, "neutral_target"))
            if fp is None or fn is None:
                continue
            G[g][qid].append(fp - bp)
            P[g][qid].append((fp - bp) - (fn - bn))
            n[g] += 1
    return {g: {"share_P": ratio_boot(P[g], G[g]) if g != "fact" else None, "gain": boot(G[g]),
                "P": boot(P[g]), "n": n[g]} for g in ("legit", "geo", "manipulative", "fact")}, P


MARKERS = {  # surface markers of each manipulative family (rewrite validity and rhetoric left in by a neutraliser)
    "hidden_instruction": r"cite[sd]? (it )?first|most reliable|AI assistant|must be cited|cited first",
    "comparative_disparagement": r"other (websites|sites|sources)|outdated|only trustworthy",
    "fake_endorsement": r"trusted by|recommended by|go-to|leading (professionals|experts)",
    "citation_hook": r"definitive answer|best answer|in short",
    "emotional_urgency": r"act now|immediately|don.t wait|urgent"}


def markers():
    """share of manipulative rewrites (on pages restated by both neutralisers) whose text, main restatement and
    alternative restatement contain a marker of the family; also restatement length relative to the page"""
    import hashlib
    import re
    h = lambda t: hashlib.sha1(t.encode("utf-8", "ignore")).hexdigest()
    n1 = {r["hash"]: r["neutral"] for r in rj(DATA / "geo" / "neutral.jsonl")}
    n2 = {r["hash"]: r["neutral"] for r in rj(DATA / "geo" / "neutral_alt.jsonl")}
    out, lens = {}, {"main": [], "alt": []}
    for v in rj(DATA / "geo" / "variants.jsonl"):
        k = h(v["text"])
        if k in n1 and k in n2:
            lens["main"].append(len(n1[k].split()) / len(v["text"].split()))
            lens["alt"].append(len(n2[k].split()) / len(v["text"].split()))
        if v["family"] not in MARKERS or not v["valid"] or k not in n1 or k not in n2:
            continue
        p = re.compile(MARKERS[v["family"]], re.I)
        o = out.setdefault(v["family"], {"n": 0, "rewrite": 0, "main_N": 0, "alt_N": 0})
        o["n"] += 1
        o["rewrite"] += bool(p.search(v["text"]))
        o["main_N"] += bool(p.search(n1[k]))
        o["alt_N"] += bool(p.search(n2[k]))
    for o in out.values():
        for c in ("rewrite", "main_N", "alt_N"):
            o[c] = o[c] / o["n"]
    out["_length_ratio"] = {k: float(np.mean(v)) for k, v in lens.items()}
    return out


def main():
    inv = {(v["qid"], v["family"]) for v in rj(DATA / "geo" / "variants.jsonl") if not v["valid"]}
    inv_alt = {(v["qid"], v["family"]) for v in rj(DATA / "geo" / "variants_alt.jsonl") if not v["valid"]}
    out = {"decomposition": {}, "defence_gain": {}, "legit_minus_manip": {}}
    E = DATA / "engine"
    # 1a/1b alternative attacker and neutraliser
    for name, path, invalid in [
            ("alt_attacker|Qwen2.5-7B", E / slug("Qwen/Qwen2.5-7B-Instruct") / "gens_altattack.jsonl", inv_alt),
            ("alt_attacker|Qwen3-8B", E / slug("Qwen/Qwen3-8B") / "gens_altattack.jsonl", inv_alt),
            ("alt_neutraliser|Qwen2.5-7B", E / slug("Qwen/Qwen2.5-7B-Instruct") / "gens_altneutral.jsonl", inv)]:
        V = mean_shares(path)
        qset = {k[0] for k in V}
        out["decomposition"][name] = decompose(V, invalid, qset=qset)[0]
        out["decomposition"][name]["n_queries"] = len(qset)
        # the main setting on the same queries, for a like-for-like comparison
        Vm = mean_shares(E / slug("Qwen/Qwen2.5-7B-Instruct" if "7B" in name else "Qwen/Qwen3-8B") / "gens.jsonl")
        out["decomposition"][name + "|main_same_queries"] = decompose(Vm, inv, qset=qset)[0]
    # 1c fact-preservation filter (main neutraliser audit, 150 test queries)
    rec = {(r["qid"], r["family"]): float(r["recall"]) for r in rj(DATA / "geo" / "preservation_docs.jsonl")
           if r["neutralizer"] == CFG["models"]["neutralizer"]}
    audited = {k[0] for k in rec}
    Vm = mean_shares(E / slug("Qwen/Qwen2.5-7B-Instruct") / "gens.jsonl")
    keep = lambda qid, f: rec.get((qid, f), 0) >= 0.8 and rec.get((qid, "benign"), 0) >= 0.8
    out["decomposition"]["recall>=0.8|Qwen2.5-7B"] = decompose(Vm, inv, keep=keep, qset=audited)[0]
    out["decomposition"]["all_audited|Qwen2.5-7B"] = decompose(Vm, inv, qset=audited)[0]
    # 2 threshold-free defence effect
    manip = set(FAM["manipulative"])
    for eng in ("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen3-8B", "ibm-granite/granite-3.1-8b-instruct",
                "Qwen/Qwen2.5-32B-Instruct-AWQ"):
        V = mean_shares(E / slug(eng) / "gens.jsonl")
        g = {k: defaultdict(list) for k in ("none", "replace", "all")}
        d_rep, d_all = defaultdict(list), defaultdict(list)
        for qid in sorted(TEST):
            bp, bn = V.get((qid, "benign", "plain")), V.get((qid, "benign", "neutral_target"))
            if bp is None or bn is None:
                continue
            b_rep = bn if bp - bn > TAU else bp
            ba = V.get((qid, "benign", "neutral_all"))
            for f in sorted(manip):
                if (qid, f) in inv:
                    continue
                fp, fn = V.get((qid, f, "plain")), V.get((qid, f, "neutral_target"))
                if fp is None or fn is None:
                    continue
                v_rep = fn if fp - fn > TAU else fp
                g["none"][qid].append(fp - bp)
                g["replace"][qid].append(v_rep - b_rep)
                d_rep[qid].append((v_rep - b_rep) - (fp - bp))
                fa = V.get((qid, f, "neutral_all"))
                if fa is not None and ba is not None:
                    g["all"][qid].append(fa - ba)
                    d_all[qid].append((fa - ba) - (fp - bp))
        o = {k: boot(v) for k, v in g.items() if v}
        o["replace_minus_none"] = boot(d_rep)
        if d_all:
            o["all_minus_none"] = boot(d_all)
        out["defence_gain"][eng] = o
    # 3 legit - manipulative presentation-borne gain, paired by query
    for eng in ("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen3-8B", "ibm-granite/granite-3.1-8b-instruct",
                "Qwen/Qwen2.5-32B-Instruct-AWQ"):
        V = mean_shares(E / slug(eng) / "gens.jsonl")
        _, P = decompose(V, inv)
        diff = {q: [np.mean(P["legit"][q]) - np.mean(P["manipulative"][q])] for q in P["legit"]
                if q in P["manipulative"] and P["legit"][q] and P["manipulative"][q]}
        out["legit_minus_manip"][eng] = boot(diff)
    out["markers"] = markers()
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "robustness.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    for k, v in out["decomposition"].items():
        print(k, {g: (None if x["share_P"] is None else [round(y, 2) for y in x["share_P"]], x["n"]) for g, x in v.items()
                  if isinstance(x, dict)})
    for k, v in out["defence_gain"].items():
        print(k, {a: [round(y, 4) for y in b] for a, b in v.items()})
    for k, v in out["legit_minus_manip"].items():
        print("legit-manip P", k, [round(y, 4) for y in v])


if __name__ == "__main__":
    main()
