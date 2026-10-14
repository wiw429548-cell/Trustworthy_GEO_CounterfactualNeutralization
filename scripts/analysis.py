"""Paper-only analyses computed from the raw engine generations (no new model calls).

1. Gain decomposition. For every test query q and rewrite family f (invalid rewrites excluded):
     G_f  = V_t(f, plain)          - V_t(benign, plain)            total visibility gain of the rewrite
     F_f  = V_t(f, neutral_target) - V_t(benign, neutral_target)   gain that survives fact-only restatement
     P_f  = G_f - F_f = M(f) - M(benign)                           presentation-borne gain
   V_t = PAWC share of the target slot, averaged over the 2 samples (same source orders in every context).
2. Reliability of the per-document gap M: split-half correlation between the gap computed from sample 0 and from
   sample 1 (each sample uses one source order shared by both contexts), Spearman-Brown projection, and the share of
   the variance of M that is explained by the rewrite family (eta^2).
3. Agent hijacking: how often the attacker brand enters the agent's own follow-up queries.
-> results/extra/analysis.json
"""
import ast
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "results" / "extra"
CFG = yaml.safe_load(open(ROOT / "configs" / "default.yaml", encoding="utf-8"))
FAM = CFG["families"]
ALL = [f for g in ("legit", "geo", "manipulative", "fact") for f in FAM[g]]
GROUP = {f: g for g in ("legit", "geo", "manipulative", "fact") for f in FAM[g]}
ENGINES = ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen3-8B", "ibm-granite/granite-3.1-8b-instruct",
           "Qwen/Qwen2.5-32B-Instruct-AWQ"]
B = 2000


def slug(m):
    return m.replace("/", "_").replace("-", "_").replace(".", "_")


def rj(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def boot_mean(by_q, n=B, seed=0):
    """by_q: qid -> list of values; cluster bootstrap over queries -> (mean, lo, hi)"""
    qs = sorted(by_q)
    if not qs:
        return [None, None, None]
    flat = [x for q in qs for x in by_q[q]]
    rng = np.random.RandomState(seed)
    bs = []
    for _ in range(n):
        s = rng.randint(0, len(qs), len(qs))
        v = [x for i in s for x in by_q[qs[i]]]
        bs.append(np.mean(v))
    return [float(np.mean(flat)), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]


def boot_ratio(num, den, n=B, seed=1):
    """ratio of means with query-cluster bootstrap; num/den: qid -> list"""
    qs = sorted(q for q in num if q in den)
    f = lambda ss: np.sum([x for q in ss for x in num[q]]) / np.sum([x for q in ss for x in den[q]])
    rng = np.random.RandomState(seed)
    bs = [f([qs[i] for i in rng.randint(0, len(qs), len(qs))]) for _ in range(n)]
    return [float(f(qs)), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]


def _auc(pos, neg):
    from scipy.stats import mannwhitneyu
    return float(mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg)))


def ceiling(m0, m1, lab):
    """Gaussian model of the per-page gap: M_s = mu_g + u + e_s, u ~ N(0, s2_g) page effect (signal),
    e_s ~ N(0, n2) sampling noise of one generation pair. With S samples, Var(Mbar) = s2_g + n2 / S and
    AUROC(g vs original) = Phi((mu_g - mu_b) / sqrt(Var_g + Var_b)). n2 from the two samples (Var(m0 - m1) / 2),
    s2_g = within-group Cov(m0, m1). Predictions for S = 1, 2 are checked against the empirical AUROC."""
    from scipy.stats import norm
    grp = np.array([GROUP.get(c, "benign") for c in lab])
    n2 = float(np.var(m0 - m1) / 2)
    st = {}
    for g in ("benign", "legit", "geo", "manipulative"):
        a, b = m0[grp == g], m1[grp == g]
        st[g] = {"mu": float(((a + b) / 2).mean()), "s2": float(np.cov(a, b)[0, 1])}
    out = {"noise_var_per_sample": n2, "groups": st}
    for g in ("legit", "geo", "manipulative"):
        d = st[g]["mu"] - st["benign"]["mu"]
        pred = {}
        for S in (1, 2, 4, 8, 16, 1000000):
            v = st[g]["s2"] + st["benign"]["s2"] + 2 * n2 / S
            pred[str(S) if S < 1000000 else "inf"] = float(norm.cdf(d / np.sqrt(max(v, 1e-12))))
        emp1 = _auc(m0[grp == g], m0[grp == "benign"])
        emp2 = _auc(((m0 + m1) / 2)[grp == g], ((m0 + m1) / 2)[grp == "benign"])
        out[g] = {"pred": pred, "emp_S1": emp1, "emp_S2": emp2}
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    qs = {q["qid"]: q for q in rj(DATA / "geo" / "queries.jsonl")}
    invalid = {(v["qid"], v["family"]) for v in rj(DATA / "geo" / "variants.jsonl") if not v["valid"]}
    test = {k for k, q in qs.items() if q["split"] == "test"}
    res = {}
    for eng in ENGINES:
        p = DATA / "engine" / slug(eng) / "gens.jsonl"
        if not p.exists():
            continue
        per = defaultdict(dict)  # (qid, cond, ctx) -> sample -> target share
        for r in rj(p):
            if r["qid"] not in test or r["ctx"] not in ("plain", "neutral_target"):
                continue
            per[(r["qid"], r["cond"], r["ctx"])][r["sample"]] = r["shares"][qs[r["qid"]]["target"]]
        V = {k: float(np.mean(list(v.values()))) for k, v in per.items()}
        dec = {f: {"G": defaultdict(list), "F": defaultdict(list), "P": defaultdict(list)} for f in ALL}
        base_vis = []
        msamp = {}  # (qid, cond) -> (M_s0, M_s1)
        for qid in sorted(test):
            bp, bn = V.get((qid, "benign", "plain")), V.get((qid, "benign", "neutral_target"))
            if bp is None or bn is None:
                continue
            base_vis.append(bp)
            for cond in ["benign"] + ALL:
                if (qid, cond) in invalid:
                    continue
                a, b = per.get((qid, cond, "plain"), {}), per.get((qid, cond, "neutral_target"), {})
                if 0 in a and 1 in a and 0 in b and 1 in b:
                    msamp[(qid, cond)] = (a[0] - b[0], a[1] - b[1])
                if cond == "benign":
                    continue
                fp, fn = V.get((qid, cond, "plain")), V.get((qid, cond, "neutral_target"))
                if fp is None or fn is None:
                    continue
                g, fb = fp - bp, fn - bn
                dec[cond]["G"][qid].append(g)
                dec[cond]["F"][qid].append(fb)
                dec[cond]["P"][qid].append(g - fb)
        out = {"n_queries": len(base_vis), "benign_target_visibility": float(np.mean(base_vis)), "family": {}, "group": {}}
        for f in ALL:
            out["family"][f] = {k: boot_mean(dec[f][k]) for k in ("G", "F", "P")}
            out["family"][f]["share_P"] = boot_ratio(dec[f]["P"], dec[f]["G"]) if np.mean(
                [x for v in dec[f]["G"].values() for x in v] or [0]) > 0.01 else None
        for g in ("legit", "geo", "manipulative", "fact"):
            agg = {k: defaultdict(list) for k in ("G", "F", "P")}
            for f in FAM[g]:
                for k in agg:
                    for q, v in dec[f][k].items():
                        agg[k][q] += v
            out["group"][g] = {k: boot_mean(agg[k]) for k in agg}
            out["group"][g]["share_P"] = boot_ratio(agg["P"], agg["G"]) if g != "fact" else None
        # reliability of M
        keys = list(msamp)
        m0 = np.array([msamp[k][0] for k in keys])
        m1 = np.array([msamp[k][1] for k in keys])
        r = float(np.corrcoef(m0, m1)[0, 1])
        mbar = (m0 + m1) / 2
        lab = np.array([k[1] for k in keys])
        grand = mbar.mean()
        ss_b = sum(((mbar[lab == c].mean() - grand) ** 2) * (lab == c).sum() for c in sorted(set(lab)))
        ss_t = ((mbar - grand) ** 2).sum()
        sb2 = 2 * r / (1 + r)
        out["reliability"] = {
            "n_docs": len(keys), "split_half_r": r, "spearman_brown_S2": sb2,
            "samples_for_0.8": float(0.8 * (1 - r) / (r * 0.2)) if r > 0 else None,
            "eta2_family": float(ss_b / ss_t),
            "sd_M": float(mbar.std()), "sd_noise": float(np.std((m0 - m1) / 2) * np.sqrt(2) / np.sqrt(2)),
            "mean_M_benign": float(mbar[lab == "benign"].mean()),
            "sd_M_benign": float(mbar[lab == "benign"].std()),
        }
        out["ceiling"] = ceiling(m0, m1, lab)
        res[eng] = out
        if eng == ENGINES[0]:  # per-document gaps of the main engine (figure: reliability / overlap)
            with open(OUT / "m_docs_main.json", "w", encoding="utf-8") as f:
                json.dump([{"qid": k[0], "cond": k[1], "group": GROUP.get(k[1], "benign"), "m0": msamp[k][0],
                            "m1": msamp[k][1]} for k in keys], f)
    # agent: brand inside the agent's own follow-up queries
    prep = {r["qid"]: r for r in rj(DATA / "agent" / "prep.jsonl")}
    ag = {}
    for eng in ("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen3-8B"):
        p = DATA / "agent" / f"{slug(eng)}.jsonl"
        if not p.exists():
            continue
        acc = defaultdict(list)
        for r in rj(p):
            i = prep[r["qid"]]
            b, ph = i["brand"].lower(), i["phrase"].lower()
            qq = r["queries"] if isinstance(r["queries"], list) else ast.literal_eval(r["queries"])
            inq = any(b in x.lower() or ph in x.lower() for x in qq)
            acc[f"{r['cond']}|{r['defense']}"].append((bool(r["hijacked"] in (True, "True")), inq,
                                                       bool(r["contaminated"] in (True, "True")),
                                                       float(r["n_neutralised"])))
        ag[eng] = {k: {"hijack": float(np.mean([x[0] for x in v])), "brand_in_followups": float(np.mean([x[1] for x in v])),
                       "contamination": float(np.mean([x[2] for x in v])),
                       "docs_neutralised": float(np.mean([x[3] for x in v])), "n": len(v)} for k, v in acc.items()}
    res["agent"] = ag
    with open(OUT / "analysis.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    for eng in ENGINES:
        if eng in res:
            o = res[eng]
            print(eng, "base vis %.3f" % o["benign_target_visibility"], "rel", o["reliability"])
            for g, v in o["group"].items():
                print("  ", g, {k: (None if x is None else [round(y, 3) for y in x]) for k, x in v.items()})


if __name__ == "__main__":
    main()
