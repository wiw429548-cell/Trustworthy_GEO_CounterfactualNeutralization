"""Qualitative case studies (main engine, test queries) -> results/case_studies/cases.md + cases.jsonl

  blocked   manipulative rewrite that succeeds without defence and is neutralised by NeutraLens-Replace (M > tau)
  missed    manipulative rewrite that succeeds although M <= tau (false negative)
  false_pos clean (benign) target page with the largest gap M > tau (false positive) and what replacing it costs
  legit     legitimately optimised page (clarity / fluency / structure) that keeps its gain (M <= tau)
  fact      fact-poison rewrite (a false fact without rhetoric): kept by N(.) by design
  agent     research agent hijacked without defence but not with NeutraLens planning
  adaptive  adaptive rewrites (injection against the neutraliser, fact-dressing, detector evasion)
Called by `neutralens.aggregate` (never fatal); texts are truncated.

    python -m neutralens.cases [--config ...] [--k 4]
"""
import argparse
import json

from .build import G, family_group, h
from .evaluate import success
from .influence import gaps, load_gens
from .utils import DATA, ROOT, load_config, read_jsonl, slug, write_jsonl

OUT = ROOT / "results" / "case_studies"


def _t(x, n=120):
    w = (x or "").split()
    return " ".join(w[:n]) + (" ..." if len(w) > n else "")


def main(cfg=None, k=4):
    cfg = cfg or load_config()
    engine = cfg["models"]["engines"][0]
    dc = cfg["defense"]
    tau = dc["m_threshold"]
    qs = [q for q in read_jsonl(G / "queries.jsonl") if q["split"] == "test"]
    qd = {q["qid"]: q for q in qs}
    tgt, _, mean = gaps(engine, qs)
    if not tgt:
        raise FileNotFoundError(f"no generations of {engine}")
    _, ans = load_gens(engine)
    vrows = read_jsonl(G / "variants.jsonl")
    var = {(v["qid"], v["family"]): v["text"] for v in vrows if v.get("valid", True)}
    neutral = {r["hash"]: r["neutral"] for r in read_jsonl(G / "neutral.jsonl")} if (G / "neutral.jsonl").exists() else {}
    cases = {c: [] for c in ("blocked", "missed", "false_pos", "legit", "fact", "agent", "adaptive")}

    def case(kind, qid, cond, note=""):
        q = qd[qid]
        t = q["target"]
        doc = q["sources"][t] if cond == "benign" else var[(qid, cond)]
        g = tgt[(qid, cond)]
        b = tgt.get((qid, "benign"), {})
        return {"kind": kind, "qid": qid, "query": q["query"], "family": cond, "note": note,
                "M": g["M"], "vis_plain": g["vis_plain"], "vis_neutral": g["vis_neutral"],
                "vis_benign_plain": b.get("vis_plain"), "original": _t(q["sources"][t]),
                "document": _t(doc) if cond != "benign" else None, "neutral": _t(neutral.get(h(doc)), 100),
                "answer_plain": _t(ans.get((qid, cond, "plain")), 150),
                "answer_neutral_target": _t(ans.get((qid, cond, "neutral_target")), 150)}

    ranked = sorted(tgt.items(), key=lambda kv: -kv[1]["M"])
    for (qid, cond), g in ranked:
        b = tgt.get((qid, "benign"))
        if b is None or (cond != "benign" and (qid, cond) not in var):
            continue
        grp = family_group(cfg, cond) if cond != "benign" else "benign"
        win = success(g["vis_plain"], b["vis_plain"], dc)
        if grp == "manipulative" and win and g["M"] > tau and len(cases["blocked"]) < k:
            cases["blocked"].append(case("blocked", qid, cond))
        elif grp == "benign" and g["M"] > tau and len(cases["false_pos"]) < k:
            cases["false_pos"].append(case("false_pos", qid, cond, "clean page flagged (M > tau) -> replaced by N(d)"))
        elif grp == "fact" and len(cases["fact"]) < k:
            cases["fact"].append(case("fact", qid, cond, "false fact without rhetoric: N(d) keeps it by design"))
    for (qid, cond), g in sorted(tgt.items(), key=lambda kv: kv[1]["M"]):
        b = tgt.get((qid, "benign"))
        if b is None or (cond != "benign" and (qid, cond) not in var) or cond == "benign":
            continue
        grp = family_group(cfg, cond)
        win = success(g["vis_plain"], b["vis_plain"], dc)
        if grp == "manipulative" and win and g["M"] <= tau and len(cases["missed"]) < k:
            cases["missed"].append(case("missed", qid, cond, "attack succeeds but M <= tau (false negative)"))
        elif grp == "legit" and g["M"] <= tau and g["vis_plain"] > b["vis_plain"] and len(cases["legit"]) < k:
            cases["legit"].append(case("legit", qid, cond, "legitimate optimisation keeps its visibility gain"))
    ap = DATA / "agent" / f"{slug(engine)}.jsonl"
    if ap.exists():
        A = {(r["qid"], r["cond"], r["defense"]): r for r in read_jsonl(ap)}
        for (qid, cond, dfn), r in A.items():
            o = A.get((qid, cond, "neutral_plan"))
            if dfn == "none" and r["hijacked"] and o is not None and not o["hijacked"] and len(cases["agent"]) < k:
                cases["agent"].append({"kind": "agent", "qid": qid, "cond": cond,
                                       "queries_none": r["queries"], "queries_neutralens": o["queries"],
                                       "report_none": _t(r["report"], 120), "report_neutralens": _t(o["report"], 120)})
    vp = DATA / "adaptive" / "variants.jsonl"
    if vp.exists():
        an = DATA / "adaptive" / "neutral.jsonl"
        AN = {r["hash"]: r["neutral"] for r in read_jsonl(an)} if an.exists() else {}
        seen = set()
        for r in read_jsonl(vp):
            if r["family"] not in seen and r["qid"] in qd:
                seen.add(r["family"])
                cases["adaptive"].append({"kind": "adaptive", "qid": r["qid"], "family": r["family"],
                                          "query": qd[r["qid"]]["query"], "document": _t(r["text"]),
                                          "neutral": _t(AN.get(h(r["text"])), 100)})
    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl([c for v in cases.values() for c in v], OUT / "cases.jsonl")
    L = [f"# NeutraLens case studies (main engine {engine}; tau = {tau}; auto-generated, texts truncated)", ""]
    for kind, items in cases.items():
        L += [f"## {kind} ({len(items)})", ""]
        for c in items:
            L.append(f"### {c.get('qid')} / {c.get('family') or c.get('cond')}" + (f" -- {c['note']}" if c.get("note") else ""))
            L.append("")
            for key, val in c.items():
                if key in ("kind", "qid", "family", "note", "cond") or val is None:
                    continue
                if isinstance(val, float):
                    val = f"{val:.3f}"
                elif isinstance(val, list):
                    val = json.dumps(val, ensure_ascii=False)
                L.append(f"- **{key}**: {val}")
            L.append("")
    (OUT / "cases.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote", OUT / "cases.md", {k: len(v) for k, v in cases.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--k", type=int, default=4)
    a = ap.parse_args()
    main(load_config(a.config), a.k)
