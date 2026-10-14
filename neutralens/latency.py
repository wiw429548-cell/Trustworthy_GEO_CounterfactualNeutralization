"""Per-query wall-clock latency of every defence (online serving, batch size 1 = one user query with its 5 sources).

Each model is loaded in its own subprocess (one vLLM engine per process); every operation is timed WITHOUT the LLM
cache (LLM._run) on the first `latency.n_queries` test queries after two warm-up calls; median / p90 are reported.
  engine         answer one query (plain context)                       engine model
  engine_spot    the same with datamarked sources (spotlighting: more prompt tokens)
  engine_cf      S plain + S x 5 counterfactual answers in one batch     (exact NeutraLens screening all 5 sources)
  neutralize5    neutralise the 5 sources in one batch                   neutraliser model
  paraphrase5    paraphrase the 5 sources in one batch                   neutraliser model
  guard5 / sci5  LLM classifiers on the 5 sources                        judge model
  amort5 / ppl5 / pi5  amortised detector / perplexity / prompt-injection classifier on the 5 sources
Defence latency = sum of its operations (online), and the variant in which neutral / paraphrased versions are
pre-computed at indexing time (they are query-independent) -> results/latency.json
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

from . import prompts as P
from .build import G, test_subset
from .utils import DATA, ROOT, get_logger, load_config, read_jsonl, save_json

log = get_logger("latency")
OUT = DATA / "latency"


def _queries(cfg):
    return test_subset(read_jsonl(G / "queries.jsonl"), cfg.get("latency", {}).get("n_queries", 20))


def _timeit(fn, items, warm=2):
    for it in items[:warm]:
        fn(it)
    ts = []
    for it in items:
        t0 = time.perf_counter()
        fn(it)
        ts.append(time.perf_counter() - t0)
    return {"median_s": float(np.median(ts)), "p90_s": float(np.quantile(ts, 0.9)), "mean_s": float(np.mean(ts)),
            "n": len(ts)}


def _engine_msgs(q, src, system=None):
    body = "\n\n".join(f"[{k + 1}] {s}" for k, s in enumerate(x for x in src if x))
    return ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": P.ENGINE.format(query=q["query"], sources=body)}]


def part_llm(cfg, model, ops):
    from .llm import LLM
    qs = _queries(cfg)
    llm = LLM(model, max_model_len=cfg["models"]["max_model_len"], gpu_mem=cfg["models"]["gpu_mem"])
    ec = cfg["engine"]
    S = ec["samples"]

    def run(msgs_list, max_tokens, temperature):
        llm._run([llm._truncate(llm.render(m), max_tokens) for m in msgs_list], temperature, max_tokens, 1, None)
    res = {}
    for op in ops:
        if op == "engine":
            f = lambda q: run([_engine_msgs(q, q["sources"])], ec["max_tokens"], ec["temperature"])
        elif op == "engine_spot":
            f = lambda q: run([_engine_msgs(q, [P.SPOTLIGHT_MARK.join(s.split()) for s in q["sources"] if s],
                                            P.SPOTLIGHT_SYSTEM)], ec["max_tokens"], ec["temperature"])
        elif op == "engine_cf":  # source texts stand in for N(d) (same length class); timing only
            f = lambda q: run([_engine_msgs(q, q["sources"])] * S + [_engine_msgs(q, q["sources"]) for _ in q["sources"]
                                                                      for _ in range(S)], ec["max_tokens"], ec["temperature"])
        elif op == "neutralize5":
            f = lambda q: run([P.NEUTRALIZE.format(text=s) for s in q["sources"] if s], 700, 0.0)
        elif op == "paraphrase5":
            f = lambda q: run([P.PARAPHRASE.format(text=s) for s in q["sources"] if s], 700, 0.0)
        elif op == "guard5":
            f = lambda q: run([P.GUARD.format(text=s[:6000]) for s in q["sources"] if s], 4, 0.0)
        elif op == "sci5":
            f = lambda q: run([P.SCI.format(text=s[:6000]) for s in q["sources"] if s], 40, 0.0)
        else:
            raise ValueError(op)
        res[op] = {"model": model, **_timeit(f, qs)}
        log.info(f"{op}: {res[op]}")
    return res


def part_enc(cfg):
    import torch

    from .detectors import D, load_amortized, pi_scorer
    qs = _queries(cfg)
    res = {}
    if (D / "amort_all_model").exists():
        sc = load_amortized("amort_all")
        res["amort5"] = {"model": cfg["models"]["detector_backbone"],
                         **_timeit(lambda q: sc([s for s in q["sources"] if s], [q["query"]] * sum(map(bool, q["sources"]))), qs)}
    pi, name = pi_scorer(cfg)
    res["pi5"] = {"model": name, **_timeit(lambda q: pi([s for s in q["sources"] if s]), qs)}
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(cfg["models"]["ppl_lm"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(cfg["models"]["ppl_lm"], torch_dtype=torch.float16 if dev == "cuda" else
                                             torch.float32).to(dev).eval()

    def ppl(q):
        enc = tok([s for s in q["sources"] if s], return_tensors="pt", padding=True, truncation=True, max_length=512).to(dev)
        with torch.no_grad():
            m(**enc)
        if dev == "cuda":
            torch.cuda.synchronize()
    res["ppl5"] = {"model": cfg["models"]["ppl_lm"], **_timeit(ppl, qs)}
    return res


def combine(cfg, ops):
    g = lambda k: (ops.get(k) or {}).get("median_s")

    def tot(*ks):
        v = [g(k) for k in ks]
        return float(sum(v)) if all(x is not None for x in v) else None
    D = {"none": ("engine",), "shield": ("engine",), "ignore_instr": ("engine",), "spotlight": ("engine_spot",),
         "paraphrase_all": ("paraphrase5", "engine"), "neutral_all": ("neutralize5", "engine"),
         "filter:ppl": ("ppl5", "engine"), "filter:pi": ("pi5", "engine"), "filter:guard": ("guard5", "engine"),
         "filter:sci": ("sci5", "engine"), "filter:amort_all": ("amort5", "engine"),
         "replace:amort_all": ("amort5", "neutralize5", "engine"), "replace:M_fixed": ("neutralize5", "engine_cf")}
    pre = {"neutralize5", "paraphrase5"}  # query-independent: can be computed once per document at indexing time
    out = {}
    for k, v in D.items():
        out[k] = {"online_s": tot(*v), "indexed_s": tot(*[x for x in v if x not in pre]), "ops": list(v)}
    base = out["none"]["online_s"]
    for v in out.values():
        v["online_overhead_x"] = v["online_s"] / base if base and v["online_s"] is not None else None
        v["indexed_overhead_x"] = v["indexed_s"] / base if base and v["indexed_s"] is not None else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--part", default=None, choices=["llm", "enc"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--ops", default="")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.part:
        res = part_llm(cfg, a.model, a.ops.split(",")) if a.part == "llm" else part_enc(cfg)
        save_json(res, a.out)
        return
    mc = cfg["models"]
    plan = {}
    for model, ops in [(mc["engines"][0], ["engine", "engine_spot", "engine_cf"]),
                       (mc["neutralizer"], ["neutralize5", "paraphrase5"]), (mc["judge"], ["guard5", "sci5"])]:
        plan.setdefault(model, []).extend(ops)
    parts = [("llm", m, ops) for m, ops in plan.items()] + [("enc", None, [])]
    cargs = ["--config", a.config] if a.config else []
    ops, errors = {}, {}
    for i, (kind, model, o) in enumerate(parts):
        out = OUT / f"part{i}.json"
        cmd = [sys.executable, "-m", "neutralens.latency", "--part", kind, "--out", str(out), *cargs]
        if model:
            cmd += ["--model", model, "--ops", ",".join(o)]
        rc = subprocess.call(cmd, cwd=ROOT, env=dict(os.environ, WWW_CACHE_SHARD=f"latency_{i}"))
        if rc == 0 and out.exists():
            ops.update(json.loads(out.read_text(encoding="utf-8")))
        else:
            errors[f"{kind}:{model}"] = f"rc={rc}"
    if not ops:
        raise SystemExit(f"latency: every part failed {errors}")
    try:
        import torch
        hw = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        hw = "unknown"
    save_json({"engine": mc["engines"][0], "hardware": hw, "backend": os.environ.get("LLM_BACKEND", "vllm"),
               "n_queries": len(_queries(cfg)),
               "batch": "one query (5 sources) at a time", "ops": ops, "defences": combine(cfg, ops), "errors": errors},
              ROOT / "results" / "latency.json")


if __name__ == "__main__":
    main()
