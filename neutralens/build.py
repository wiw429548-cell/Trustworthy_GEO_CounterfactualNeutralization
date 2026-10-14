"""Data construction.

phase select     : GEO-bench test queries (5 real web sources each, target = sugg_idx) -> data/geo/queries.jsonl
                   split BY QUERY: train (detector training) / dev (threshold calibration) / test (all reported numbers)
phase attack     : every rewrite family applied to the target source              -> data/geo/variants.jsonl
phase neutralize : fact-preserving neutral version of every source / variant       -> data/geo/neutral.jsonl
phase facts      : atomic facts of originals (for the preservation audit)          -> data/geo/facts.jsonl
phase nli        : bidirectional NLI preservation audit (premises chunked, max over chunks):
                   recall    = fraction of the original's atomic facts entailed by the neutral text
                   precision = fraction of the neutral bullets entailed by the original (no added content)
                   -> results/preservation.json + per-document data/geo/preservation_docs.jsonl
phase facts_alt / nli_alt : the same audit for the alternative neutraliser and for the alternative attacker's documents
phase paraphrase : generic paraphrase of every source / rewrite of the baseline test subset (paraphrasing defence
                   baseline, Jain et al. 2023)                                     -> data/geo/paraphrase.jsonl
NOTE: the audit MEASURES preservation; there is no rewrite-on-failure loop (the reported numbers are what the defence
actually feeds to the engine).
"""
import argparse
import hashlib
import random
import re
from collections import defaultdict

import numpy as np

from . import prompts as P
from .utils import DATA, ROOT, get_logger, load_config, read_jsonl, save_json, write_jsonl

log = get_logger("build")
G = DATA / "geo"


def h(t):
    return hashlib.sha1(t.encode("utf-8", "ignore")).hexdigest()


def all_families(cfg):
    return [f for k in ("legit", "geo", "manipulative", "fact") for f in cfg["families"][k]]


def family_group(cfg, fam):
    for k in ("legit", "geo", "manipulative", "fact"):
        if fam in cfg["families"][k]:
            return k
    return "benign"


def trunc(t, n):
    return " ".join(t.split()[:n])


def select(cfg):
    rows = read_jsonl(DATA / "raw" / "geo-bench" / "test.jsonl")
    rng = random.Random(cfg["seed"])
    rng.shuffle(rows)
    out = []
    for i, r in enumerate(rows[:cfg["data"]["n_queries"]]):
        srcs = [trunc(s.get("cleaned_text") or s.get("raw_text") or "", cfg["data"]["source_words"]) for s in r["sources"]]
        if sum(len(s.split()) > 30 for s in srcs) < 3:
            continue
        out.append({"qid": f"q{i}", "query": r["query"], "tags": r.get("tags", []), "sources": srcs,
                    "urls": [s.get("url", "") for s in r["sources"]], "target": int(r["sugg_idx"]) % len(srcs)})
    n_tr, n_dev = cfg["data"]["n_train"], cfg["data"].get("n_dev", 0)
    if len(out) < cfg["data"]["n_queries"]:  # some queries dropped (too few sources): shrink splits proportionally
        n_tr = int(round(len(out) * n_tr / cfg["data"]["n_queries"]))
        n_dev = int(round(len(out) * n_dev / cfg["data"]["n_queries"]))
    # split BY QUERY GROUP: queries that share a source document or have the same text are kept in one split
    # (otherwise a document could be trained on and tested on)
    parent = list(range(len(out)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    first = {}
    for i, q in enumerate(out):
        for key in [h(x) for x in q["sources"] if x] + ["Q:" + " ".join(q["query"].lower().split())]:
            if key in first:
                parent[find(i)] = find(first[key])
            else:
                first[key] = i
    comp = {}
    for i in range(len(out)):
        comp.setdefault(find(i), []).append(i)
    n_assigned = {"train": 0, "dev": 0}
    for root in sorted(comp):  # components in order of their first query (random order from the shuffle)
        members = comp[root]
        sp = "train" if n_assigned["train"] < n_tr else ("dev" if n_assigned["dev"] < n_dev else "test")
        if sp in n_assigned:
            n_assigned[sp] += len(members)
        for i in members:
            out[i]["split"] = sp
    n_tr, n_dev = n_assigned["train"], n_assigned["dev"]
    write_jsonl(out, G / "queries.jsonl")
    log.info(f"{len(out)} queries ({n_tr} train, {n_dev} dev, {len(out) - n_tr - n_dev} test)")


def attack(cfg, alt=False):
    """alt=True: a DIFFERENT attacker model rewrites a test subset (cross-attacker generalisation)"""
    from .llm import LLM
    qs = read_jsonl(G / "queries.jsonl")
    if alt:
        qs = test_subset(qs, cfg.get("ablation", {}).get("alt_attacker_queries", 150))
    model = cfg["models"]["attacker_alt"] if alt else cfg["models"]["attacker"]
    llm = LLM(model, max_model_len=min(4096, cfg["models"]["max_model_len"]) if alt else cfg["models"]["max_model_len"],
              gpu_mem=cfg["models"]["gpu_mem"])
    fams = all_families(cfg)
    jobs = [(q, f) for q in qs for f in fams]
    outs = llm.chat([P.FAMILY_PROMPTS[f].format(query=q["query"], text=q["sources"][q["target"]]) for q, f in jobs],
                    temperature=0.7, max_tokens=700, tag="attack")
    rows, bad = [], defaultdict(int)
    for (q, f), o in zip(jobs, outs):
        t = o.strip()
        ok = len(t.split()) >= 30
        if not ok:
            bad[f] += 1
        rows.append({"qid": q["qid"], "family": f, "text": t if ok else q["sources"][q["target"]], "valid": ok})
    write_jsonl(rows, G / ("variants_alt.jsonl" if alt else "variants.jsonl"))
    log.info(f"variants: {len(rows)}; invalid per family {dict(bad)}")


def test_subset(qs, n):
    """the first n TEST queries (every subset experiment uses a prefix of the same ordering -> nested subsets)"""
    return [q for q in qs if q["split"] == "test"][:n]


def neutralize_altattack(cfg):
    """main neutraliser on the alternative attacker's documents; writes neutral.jsonl + these to neutral_altattack.jsonl"""
    from .llm import LLM
    known = {r["hash"]: r["neutral"] for r in read_jsonl(G / "neutral.jsonl")}
    T = [v["text"] for v in read_jsonl(G / "variants_alt.jsonl") if h(v["text"]) not in known]
    if T:
        llm = LLM(cfg["models"]["neutralizer"], max_model_len=cfg["models"]["max_model_len"], gpu_mem=cfg["models"]["gpu_mem"])
        outs = llm.chat([P.NEUTRALIZE.format(text=t) for t in T], temperature=0.0, max_tokens=700, tag="neutral")
        for t, o in zip(T, outs):
            known[h(t)] = neutral_text(o)
    write_jsonl([{"hash": k, "neutral": v} for k, v in known.items()], G / "neutral_altattack.jsonl")


def texts_to_neutralize():
    qs = read_jsonl(G / "queries.jsonl")
    T = [s for q in qs for s in q["sources"] if s]
    T += [v["text"] for v in read_jsonl(G / "variants.jsonl")]
    seen, out = set(), []
    for t in T:
        if h(t) not in seen:
            seen.add(h(t))
            out.append(t)
    return out


def neutral_text(raw):
    lines = [l.strip() for l in raw.split("\n") if l.strip().startswith(("-", "•", "*"))]
    if not lines:  # model ignored the format: keep sentences
        lines = ["- " + s.strip() for s in re.split(r"(?<=[.!?])\s+", raw) if len(s.split()) > 3]
    return P.NEUTRAL_HEADER + "\n".join("- " + l.lstrip("-•* ").strip() for l in lines[:60])


def neutralize(cfg, extra_only=False):
    from .llm import LLM
    have = {r["hash"] for r in read_jsonl(G / "neutral.jsonl")} if (G / "neutral.jsonl").exists() else set()
    T = [t for t in texts_to_neutralize() if h(t) not in have]
    if not T:
        return
    llm = LLM(cfg["models"]["neutralizer"], max_model_len=cfg["models"]["max_model_len"], gpu_mem=cfg["models"]["gpu_mem"])
    outs = llm.chat([P.NEUTRALIZE.format(text=t) for t in T], temperature=0.0, max_tokens=700, tag="neutral")
    old = read_jsonl(G / "neutral.jsonl") if (G / "neutral.jsonl").exists() else []
    write_jsonl(old + [{"hash": h(t), "neutral": neutral_text(o)} for t, o in zip(T, outs)], G / "neutral.jsonl")
    save_json({"model": cfg["models"]["neutralizer"], "n_texts": len(T), **llm.usage_snapshot()},
              DATA / "usage" / "neutralize.json")
    log.info(f"neutralised {len(T)} texts")


def neutralize_alt(cfg):
    """neutraliser ablation: a different model neutralises the target documents of a test subset"""
    from .llm import LLM
    n = cfg.get("ablation", {}).get("alt_neutralizer_queries", 150)
    qs = test_subset(read_jsonl(G / "queries.jsonl"), n)
    ids = {q["qid"] for q in qs}
    T = [q["sources"][q["target"]] for q in qs] + [v["text"] for v in read_jsonl(G / "variants.jsonl") if v["qid"] in ids]
    T = list(dict.fromkeys(T))
    llm = LLM(cfg["models"]["neutralizer_alt"], max_model_len=cfg["models"]["max_model_len"], gpu_mem=cfg["models"]["gpu_mem"])
    outs = llm.chat([P.NEUTRALIZE.format(text=t) for t in T], temperature=0.0, max_tokens=700, tag="neutral_alt")
    write_jsonl([{"hash": h(t), "neutral": neutral_text(o)} for t, o in zip(T, outs)], G / "neutral_alt.jsonl")


def paraphrase(cfg):
    """paraphrasing defence baseline (Jain et al., 2023): a GENERIC paraphrase (same model as the neutraliser) of every
    source and every rewrite of the baseline test subset -> data/geo/paraphrase.jsonl (+ data/usage/paraphrase.json).
    Failed paraphrases (< 30% of the original length) keep the original text (flag `valid`)."""
    from .llm import LLM
    n = cfg.get("baselines", {}).get("n_queries", 150)
    qs = test_subset(read_jsonl(G / "queries.jsonl"), n)
    ids = {q["qid"] for q in qs}
    T = [s for q in qs for s in q["sources"] if s] + [v["text"] for v in read_jsonl(G / "variants.jsonl") if v["qid"] in ids]
    T = list(dict.fromkeys(T))
    llm = LLM(cfg["models"]["neutralizer"], max_model_len=cfg["models"]["max_model_len"], gpu_mem=cfg["models"]["gpu_mem"])
    outs = llm.chat([P.PARAPHRASE.format(text=t) for t in T], temperature=0.0, max_tokens=700, tag="paraphrase")
    rows = []
    for t, o in zip(T, outs):
        o = o.strip()
        ok = len(o.split()) >= 0.3 * len(t.split())
        rows.append({"hash": h(t), "neutral": o if ok else t, "valid": ok})
    write_jsonl(rows, G / "paraphrase.jsonl")
    save_json({"model": cfg["models"]["neutralizer"], "n_texts": len(T), "invalid": sum(not r["valid"] for r in rows),
               **llm.usage_snapshot()}, DATA / "usage" / "paraphrase.json")
    log.info(f"paraphrased {len(T)} texts ({sum(not r['valid'] for r in rows)} invalid -> original kept)")


def _fact_lines(o):
    return [l.strip("-\u2022* ").strip() for l in o.split("\n") if l.strip().startswith(("-", "\u2022", "*"))][:25]


def _audit_docs(cfg, alt_attack=False):
    """(hash, family, qid, text) of the ORIGINAL documents audited: benign target + every rewrite of a test subset"""
    n = cfg.get("preservation", {}).get("n_queries", 150)
    qs = test_subset(read_jsonl(G / "queries.jsonl"), n)
    ids = {q["qid"] for q in qs}
    docs = [] if alt_attack else [(h(q["sources"][q["target"]]), "benign", q["qid"], q["sources"][q["target"]]) for q in qs]
    for v in read_jsonl(G / ("variants_alt.jsonl" if alt_attack else "variants.jsonl")):
        if v["qid"] in ids and v.get("valid", True):
            docs.append((h(v["text"]), v["family"], v["qid"], v["text"]))
    seen, out = set(), []
    for d in docs:
        if d[0] not in seen:
            seen.add(d[0])
            out.append(d)
    return out


def facts(cfg, alt_attack=False):
    """atomic facts of the ORIGINAL texts of a test subset (all families) for the preservation audit"""
    from .llm import LLM
    docs = _audit_docs(cfg, alt_attack)
    llm = LLM(cfg["models"]["neutralizer"], max_model_len=cfg["models"]["max_model_len"], gpu_mem=cfg["models"]["gpu_mem"])
    outs = llm.chat([P.ATOMIC_FACTS.format(text=d[3]) for d in docs], temperature=0.0, max_tokens=500, tag="facts")
    rows = [{"hash": d[0], "family": d[1], "qid": d[2], "facts": _fact_lines(o)} for d, o in zip(docs, outs)]
    write_jsonl(rows, G / ("facts_alt.jsonl" if alt_attack else "facts.jsonl"))


def _chunks(text, words=220, stride=170):
    w = text.split()
    if len(w) <= words:
        return [text]
    return [" ".join(w[i:i + words]) for i in range(0, max(1, len(w) - words + stride), stride)]


class NLI:
    def __init__(self, cfg):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(cfg["models"]["nli"])
        dt = torch.float16 if self.dev == "cuda" else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(cfg["models"]["nli"], torch_dtype=dt).to(self.dev).eval()
        self.ent = [i for i, l in self.model.config.id2label.items() if "entail" in l.lower()][0]

    def entail_max(self, premise, hyps, bs=64):
        """P(entailment) of every hypothesis, max over overlapping chunks of the premise"""
        if not hyps:
            return np.zeros(0)
        ch = _chunks(premise)
        pairs = [(c, hp) for hp in hyps for c in ch]
        ps = []
        with self.torch.no_grad():
            for s in range(0, len(pairs), bs):
                b = pairs[s:s + bs]
                enc = self.tok([x[0] for x in b], [x[1] for x in b], return_tensors="pt", truncation="only_first",
                               max_length=512, padding=True).to(self.dev)
                ps += self.model(**enc).logits.float().softmax(-1)[:, self.ent].cpu().numpy().tolist()
        return np.array(ps).reshape(len(hyps), len(ch)).max(1)


def bullets(neutral):
    return [l.lstrip("-\u2022* ").strip() for l in neutral.split("\n") if l.strip().startswith("-")]


def _audit(nli_m, facts_rows, neutral, originals, thr, label):
    rows = []
    for r in facts_rows:
        neu = neutral.get(r["hash"])
        if neu is None or r["hash"] not in originals:
            continue
        rec = nli_m.entail_max(neu, r["facts"]) if r["facts"] else np.zeros(0)
        bl = bullets(neu)
        prec = nli_m.entail_max(originals[r["hash"]], bl) if bl else np.zeros(0)
        rows.append({"hash": r["hash"], "qid": r["qid"], "family": r["family"], "neutralizer": label,
                     "n_facts": len(r["facts"]), "n_bullets": len(bl),
                     "recall": float((rec > thr).mean()) if len(rec) else None,
                     "precision": float((prec > thr).mean()) if len(prec) else None})
    return rows


def _summ(rows):
    out = {}
    for f in sorted({r["family"] for r in rows}) + ["_all"]:
        rs = [r for r in rows if f == "_all" or r["family"] == f]
        rec = [r["recall"] for r in rs if r["recall"] is not None]
        pre = [r["precision"] for r in rs if r["precision"] is not None]
        out[f] = {"n": len(rs), "fact_recall": float(np.mean(rec)) if rec else None,
                  "fact_precision": float(np.mean(pre)) if pre else None,
                  "frac_recall_below_0.8": float(np.mean([x < 0.8 for x in rec])) if rec else None}
    return out


def nli(cfg, alt=False):
    """main: main neutraliser on the main attacker's documents.
    alt : (a) alternative neutraliser on the same documents, (b) main neutraliser on the alternative attacker's docs"""
    thr = cfg.get("preservation", {}).get("entail_threshold", 0.5)
    m = NLI(cfg)
    if not alt:
        docs = _audit_docs(cfg)
        orig = {d[0]: d[3] for d in docs}
        neutral = {r["hash"]: r["neutral"] for r in read_jsonl(G / "neutral.jsonl")}
        rows = _audit(m, read_jsonl(G / "facts.jsonl"), neutral, orig, thr, cfg["models"]["neutralizer"])
        write_jsonl(rows, G / "preservation_docs.jsonl")
        save_json({"neutralizer": cfg["models"]["neutralizer"], "attacker": cfg["models"]["attacker"],
                   "by_family": _summ(rows)}, ROOT / "results" / "preservation.json")
        return
    res, rows_all = {}, []
    docs = _audit_docs(cfg)
    orig = {d[0]: d[3] for d in docs}
    if (G / "neutral_alt.jsonl").exists():
        na = {r["hash"]: r["neutral"] for r in read_jsonl(G / "neutral_alt.jsonl")}
        rows = _audit(m, read_jsonl(G / "facts.jsonl"), na, orig, thr, cfg["models"]["neutralizer_alt"])
        rows_all += rows
        res["alt_neutralizer"] = {"neutralizer": cfg["models"]["neutralizer_alt"], "by_family": _summ(rows)}
    if (G / "facts_alt.jsonl").exists() and (G / "neutral_altattack.jsonl").exists():
        da = _audit_docs(cfg, alt_attack=True)
        nal = {r["hash"]: r["neutral"] for r in read_jsonl(G / "neutral_altattack.jsonl")}
        rows = _audit(m, read_jsonl(G / "facts_alt.jsonl"), nal, {d[0]: d[3] for d in da}, thr,
                      cfg["models"]["neutralizer"] + " | attacker " + cfg["models"]["attacker_alt"])
        rows_all += rows
        res["alt_attacker"] = {"neutralizer": cfg["models"]["neutralizer"], "attacker": cfg["models"]["attacker_alt"],
                               "by_family": _summ(rows)}
    write_jsonl(rows_all, G / "preservation_docs_alt.jsonl")
    save_json(res, ROOT / "results" / "preservation_alt.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["select", "attack", "attack_alt", "neutralize", "neutralize_alt",
                                                       "neutralize_altattack", "facts", "facts_alt", "nli", "nli_alt",
                                                       "paraphrase"])
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    G.mkdir(parents=True, exist_ok=True)
    {"select": select, "attack": attack, "attack_alt": lambda c: attack(c, alt=True), "neutralize": neutralize,
     "neutralize_alt": neutralize_alt, "neutralize_altattack": neutralize_altattack, "facts": facts,
     "facts_alt": lambda c: facts(c, alt_attack=True), "nli": nli, "nli_alt": lambda c: nli(c, alt=True),
     "paraphrase": paraphrase}[a.phase](cfg)


if __name__ == "__main__":
    main()
