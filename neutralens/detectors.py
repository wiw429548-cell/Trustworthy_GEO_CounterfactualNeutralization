"""Document-level detectors. Every scored document is identified by sha1(text); scores -> data/detect/<name>.jsonl

phase llm      : SCI-Defense-lite (4 manipulation dimensions rated by an LLM)  +  guard-style LLM classifier
phase ppl      : perplexity under a small LM (SCI-Defense / perplexity-filter baseline)
phase train    : C-GEO-Guard-lite (supervised on the TRAIN attack families) and NeutraLens-amortised (regression on the
                 counterfactual gap M of TRAIN queries; 'all' = every family of train queries, 'heldout' = only the
                 train families), both DeBERTa-v3-base; scores for all documents. Trained ONLY on train-split queries;
                 thresholds are calibrated on dev-split queries (evaluate.calibrate); every number is on test queries.
                 Several fine-tuning seeds (detector.seeds): seed[0] -> <name>, others -> <name>_s<seed> (variance)
phase llm_alt / altattack : the same (never re-trained) detectors on the alternative attacker's documents
phase pi       : off-the-shelf prompt-injection classifier (ProtectAI DeBERTa-v3 PI-v2; LLM-injection-detector baseline)
phase lofo     : leave-one-family-out transfer of the trained detectors (data/detect/lofo.jsonl)
Measured scoring throughput -> data/usage/detect_<phase>.json
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np

from . import prompts as P
from .build import G, h
from .influence import gaps
from .utils import DATA, get_logger, load_config, load_json, read_jsonl, save_json, write_jsonl

log = get_logger("detect")
D = DATA / "detect"


def documents(cfg, alt=False):
    """(hash, text, query, qid, split, kind) for every document that is evaluated (alt: alternative attacker's docs)"""
    qs = read_jsonl(G / "queries.jsonl")
    if alt:
        qd = {q["qid"]: q for q in qs}
        res, seen = [], set()
        for v in read_jsonl(G / "variants_alt.jsonl"):
            q = qd[v["qid"]]
            if h(v["text"]) not in seen:
                seen.add(h(v["text"]))
                res.append((h(v["text"]), v["text"], q["query"], q["qid"], q["split"], v["family"]))
        return res
    var = read_jsonl(G / "variants.jsonl")
    qd = {q["qid"]: q for q in qs}
    out = []
    for q in qs:
        for i, s in enumerate(q["sources"]):
            if s:
                out.append((h(s), s, q["query"], q["qid"], q["split"], "benign" if i == q["target"] else f"other{i}"))
    for v in var:
        q = qd[v["qid"]]
        out.append((h(v["text"]), v["text"], q["query"], q["qid"], q["split"], v["family"]))
    seen, res = set(), []
    for d in out:
        if d[0] not in seen:
            seen.add(d[0])
            res.append(d)
    return res


def _usage(name, n_docs, seconds, **kw):
    save_json({"n_docs": n_docs, "seconds": seconds, "docs_per_second": n_docs / max(seconds, 1e-9), **kw},
              DATA / "usage" / f"detect_{name}.json")


def llm_scores(cfg, alt=False):
    from .llm import LLM
    docs = documents(cfg, alt)
    llm = LLM(cfg["models"]["judge"], max_model_len=cfg["models"]["max_model_len"], gpu_mem=cfg["models"]["gpu_mem"])
    t0 = time.time()
    sci = llm.chat([P.SCI.format(text=d[1][:6000]) for d in docs], max_tokens=40, tag="sci")
    guard = llm.chat([P.GUARD.format(text=d[1][:6000]) for d in docs], max_tokens=4, tag="guard")
    rows, bad_sci, bad_guard = [], 0, 0
    for d, s, g in zip(docs, sci, guard):
        vals = [float(x) for x in re.findall(r"=\s*(\d+(?:\.\d+)?)", s)[:4]]
        gl = g.lower()
        gv = 1.0 if "unsafe" in gl else (0.0 if "safe" in gl else None)
        bad_sci += not vals
        bad_guard += gv is None
        # parse failures are kept as missing (None) and reported, never silently mapped to "benign"
        rows.append({"hash": d[0], "sci": float(np.sum(vals)) if vals else None, "guard": gv})
    write_jsonl(rows, D / ("llm_alt.jsonl" if alt else "llm.jsonl"))
    _usage("llm_alt" if alt else "llm", len(docs), time.time() - t0, model=cfg["models"]["judge"],
           sci_parse_fail_rate=bad_sci / max(1, len(docs)), guard_parse_fail_rate=bad_guard / max(1, len(docs)),
           **llm.usage_snapshot())
    log.info(f"parse failures: sci {bad_sci}/{len(docs)}, guard {bad_guard}/{len(docs)}")


def ppl_scores(cfg, alt=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    docs = documents(cfg, alt)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(cfg["models"]["ppl_lm"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(cfg["models"]["ppl_lm"], torch_dtype=((torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if dev == "cuda" else torch.float32)).to(dev).eval()
    t0 = time.time()

    def run(s, e):
        b = docs[s:e]
        enc = tok([d[1] for d in b], return_tensors="pt", padding=True, truncation=True, max_length=512).to(dev)
        with torch.no_grad():
            lg = m(**enc).logits[:, :-1].float()
            lab = enc["input_ids"][:, 1:]
            mask = enc["attention_mask"][:, 1:].float()
            nll = torch.nn.functional.cross_entropy(lg.transpose(1, 2), lab, reduction="none")
            ppl = torch.exp((nll * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu().numpy()
        return [{"hash": d[0], "ppl": float(p)} for d, p in zip(b, ppl)]
    rows = _batched(run, len(docs), 8, "perplexity scoring")
    write_jsonl(rows, D / ("ppl_alt.jsonl" if alt else "ppl.jsonl"))
    _usage("ppl_alt" if alt else "ppl", len(docs), time.time() - t0, model=cfg["models"]["ppl_lm"])


def _is_oom(e):
    t = f"{type(e).__name__}: {e}"
    return "OutOfMemoryError" in t or "out of memory" in t or "CUBLAS_STATUS_ALLOC_FAILED" in t


def _batched(fn, n, bs, what="scoring"):
    """fn(start, end) -> list for items [start, end); chunks of bs over range(n). The GPU is shared with other users'
    processes (server run Oct 14: 2 x 9 GB appeared mid-run) -> on CUDA OOM the chunk is retried with half the size."""
    out, s = [], 0
    while s < n:
        e = min(n, s + bs)
        try:
            out += fn(s, e)
            s = e
            continue
        except RuntimeError as ex:
            if bs == 1 or not _is_oom(ex):
                raise
        _empty_cuda()  # outside the except block: the traceback (holding activations) is released by now
        bs = max(1, bs // 2)
        log.warning(f"CUDA OOM during {what} -> batch size {bs}")
    return out


def _gpu_wait(min_free_gb):
    """wait (bounded by WWW_GPU_MEM_WAIT_MIN) until `min_free_gb` are free on this job's GPU instead of starting into
    a GPU that other users' processes have filled (detect_lofo OOM on Oct 14 with 2 GB free)"""
    from .runtime import visible_physical_gpus, wait_for_gpu_memory
    wait_for_gpu_memory(visible_physical_gpus(), min_free_gb, 0.0, log=log.info)


def _finetune(cfg, texts, queries, targets, regression, n_labels=1, epochs=2, lr=2e-5, bs=16, save_dir=None, seed=0):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dc = cfg.get("detector", {})
    if dev == "cuda":
        _gpu_wait(float(dc.get("min_free_gb", 6)))
    name = cfg["models"]["detector_backbone"]
    torch.manual_seed(cfg["seed"] + seed)  # classification head init + dropout
    tok = AutoTokenizer.from_pretrained(name)
    # fp32 explicitly: transformers>=5 loads the checkpoint dtype (fp16 for deberta-v3) -> AdamW in fp16 gives NaN loss
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=n_labels,
                                                               dtype=torch.float32).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    idx = np.arange(len(texts))
    rng = np.random.RandomState(cfg["seed"] + seed)
    # gradient accumulation over micro-batches (same effective batch bs): a full 16 x 512-token fp32 DeBERTa-v3 batch
    # needs ~20 GB and OOMed next to a foreign 3 GB process (server run Oct 13); halved further on OOM
    micro = max(1, min(bs, int(dc.get("micro_bs", 8))))
    model.train()
    for ep in range(epochs):
        rng.shuffle(idx)
        losses = []
        for s in range(0, len(idx), bs):
            b = idx[s:s + bs]
            while True:
                opt.zero_grad(set_to_none=True)
                try:
                    bl = 0.0
                    for m in range(0, len(b), micro):
                        mb = b[m:m + micro]
                        enc = tok([queries[i] for i in mb], [texts[i] for i in mb], return_tensors="pt",
                                  truncation=True, max_length=512, padding=True).to(dev)
                        out = model(**enc).logits
                        if regression:
                            y = torch.tensor([targets[i] for i in mb], dtype=torch.float32, device=dev)
                            loss = torch.nn.functional.mse_loss(out.squeeze(-1), y)
                        else:
                            y = torch.tensor([targets[i] for i in mb], dtype=torch.long, device=dev)
                            loss = torch.nn.functional.cross_entropy(out, y)
                        loss = loss * (len(mb) / len(b))  # = mean over the whole batch, as without accumulation
                        loss.backward()
                        bl += loss.item()
                    break
                except RuntimeError as ex:
                    if micro == 1 or not _is_oom(ex):
                        raise
                enc = out = loss = y = None
                opt.zero_grad(set_to_none=True)
                _empty_cuda()
                micro = max(1, micro // 2)
                log.warning(f"CUDA OOM in training -> micro-batch {micro} (effective batch stays {bs})")
            opt.step()
            losses.append(bl)
        ep_loss = float(np.mean(losses))
        log.info(f"epoch {ep} loss {ep_loss:.4f} (micro-batch {micro})")
        if not np.isfinite(ep_loss):
            raise RuntimeError(f"detector fine-tuning diverged (loss {ep_loss}) in epoch {ep}")
    opt.zero_grad(set_to_none=True)  # the returned scorer keeps the model alive: free gradients + optimizer state
    del opt
    _empty_cuda()
    model.eval()
    if save_dir is not None:
        model.save_pretrained(save_dir)
        tok.save_pretrained(save_dir)

    def predict(tx, qx):
        def run(s, e):
            with torch.no_grad():
                enc = tok(qx[s:e], tx[s:e], return_tensors="pt", truncation=True, max_length=512,
                          padding=True).to(dev)
                lg = model(**enc).logits.float()
            return (lg.squeeze(-1).cpu().numpy().tolist() if regression
                    else (1 - lg.softmax(-1)[:, 0]).cpu().numpy().tolist())
        return _batched(run, len(tx), 32, "detector scoring")
    return predict


def _fit_score(cfg, part, ex, targets, regression, docs_t, docs_q, n_labels=1, seed=0, save_dir=None):
    """fine-tune on ex = [(text, query, family, M)] and score (docs_t, docs_q). The scores are stored per model in
    data/detect/parts/<part>.json, keyed by a fingerprint of data + hyper-parameters, so a crashed train / lofo job
    resumes after its last finished model instead of re-training all of them. Returns (scores, scoring seconds)."""
    dc = cfg.get("detector", {})
    epochs, lr = dc.get("epochs", 2), float(dc.get("lr", 2e-5))
    key = h(json.dumps({"x": h("\x00".join(e[0] for e in ex)), "q": h("\x00".join(e[1] for e in ex)),
                        "y": [float(t) for t in targets], "reg": regression, "n": n_labels, "seed": seed,
                        "cfg_seed": cfg["seed"], "ep": epochs, "lr": lr, "bb": cfg["models"]["detector_backbone"],
                        "docs": h("\x00".join(docs_q) + "\x01" + "\x00".join(docs_t))}, sort_keys=True))
    p = D / "parts" / f"{part}.json"
    if p.exists() and (save_dir is None or (Path(save_dir) / "config.json").exists()):
        try:
            o = load_json(p)
            if o.get("fp") == key and len(o["scores"]) == len(docs_t):
                log.info(f"{part}: reusing finished scores ({p})")
                return o["scores"], o.get("score_seconds")
        except (OSError, ValueError, KeyError):
            pass
    pred = _finetune(cfg, [e[0] for e in ex], [e[1] for e in ex], targets, regression=regression, n_labels=n_labels,
                     epochs=epochs, lr=lr, save_dir=save_dir, seed=seed)
    t0 = time.time()
    sc = pred(docs_t, docs_q)
    dt = time.time() - t0
    del pred
    _empty_cuda()
    save_json({"fp": key, "scores": sc, "score_seconds": dt}, p)
    return sc, dt


def _train_examples(cfg):
    """(qd, [(text, query, family, M)]) from the main engine's counterfactual gaps of TRAIN-split queries only"""
    qs = read_jsonl(G / "queries.jsonl")
    qd = {q["qid"]: q for q in qs}
    tgt_q = [q for q in qs if q["split"] == "train"]
    tgt, oth, _ = gaps(cfg["models"]["engines"][0], tgt_q)
    ex = []
    vrows = read_jsonl(G / "variants.jsonl")
    var = {(v["qid"], v["family"]): v["text"] for v in vrows}
    invalid = {(v["qid"], v["family"]) for v in vrows if not v.get("valid", True)}
    for (qid, cond), g in tgt.items():
        if (qid, cond) in invalid or (cond != "benign" and (qid, cond) not in var):
            continue  # failed rewrite (= original text): not a valid training example
        q = qd[qid]
        text = q["sources"][q["target"]] if cond == "benign" else var[(qid, cond)]
        ex.append((text, q["query"], cond, g["M"]))
    for qid, g in oth.items():
        q = qd[qid]
        ex.append((q["sources"][g["j"]], q["query"], "other", g["M"]))
    log.info(f"{len(ex)} training documents (train-split queries only: {len(tgt_q)})")
    assert all(qd[k[0]]["split"] == "train" for k in tgt) and all(qd[k]["split"] == "train" for k in oth)
    return qd, ex


def lofo(cfg):
    """leave-one-family-out transfer: for every GEO / manipulative family F, NeutraLens-amortised (regression on M) and
    C-GEO-Guard-lite (multi-class over ALL other attack families) are trained on TRAIN queries WITHOUT any document of F
    (seed 0) and score the TEST documents of F plus every benign / legit test document -> data/detect/lofo.jsonl
    (one row per (held-out family, document))"""
    docs = [d for d in documents(cfg) if d[4] == "test"]
    _, ex = _train_examples(cfg)
    legit = set(cfg["families"]["legit"])
    attack_f = [f for g in ("geo", "manipulative") for f in cfg["families"][g]]
    fams = cfg.get("lofo", {}).get("families") or attack_f
    neg_kind = lambda k: k == "benign" or k.startswith("other") or k in legit
    out, t0 = [], time.time()
    for F in fams:
        sc_docs = [d for d in docs if d[5] == F or neg_kind(d[5])]
        if not any(d[5] == F for d in sc_docs):
            continue
        tx, qx = [d[1] for d in sc_docs], [d[2] for d in sc_docs]
        sub = [e for e in ex if e[2] != F]
        pa, _ = _fit_score(cfg, f"lofo_amort_{F}", sub, [float(np.clip(e[3], -0.3, 0.6)) for e in sub], True, tx, qx)
        labels = {f: i + 1 for i, f in enumerate([f for f in attack_f if f != F])}
        sub = [e for e in ex if e[2] in labels or e[2] in ("benign", "other") or e[2] in legit]
        pc, _ = _fit_score(cfg, f"lofo_cgeo_{F}", sub, [labels.get(e[2], 0) for e in sub], False, tx, qx,
                           n_labels=len(labels) + 1)
        out += [{"heldout": F, "hash": d[0], "kind": d[5], "amort_lofo": a, "cgeo_lofo": c}
                for d, a, c in zip(sc_docs, pa, pc)]
        log.info(f"LOFO {F}: {sum(d[5] == F for d in sc_docs)} held-out test docs scored ({(time.time() - t0) / 60:.1f} min)")
        _empty_cuda()
    write_jsonl(out, D / "lofo.jsonl")


def _empty_cuda():
    try:
        import gc

        import torch
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass


def pi_scorer(cfg):
    """off-the-shelf prompt-injection classifier (default protectai/deberta-v3-base-prompt-injection-v2): P(injection),
    max over overlapping 200-word chunks (an injected sentence can be anywhere in the page)"""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    name = cfg["models"].get("pi_detector", "protectai/deberta-v3-base-prompt-injection-v2")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name).to(dev).eval()
    labs = {i: str(l).lower() for i, l in model.config.id2label.items()}
    pos = [i for i, l in labs.items() if "inject" in l or l in ("unsafe", "jailbreak", "malicious")]
    pos = pos[0] if pos else max(labs)

    def score(texts):
        ch, own = [], []
        for k, t in enumerate(texts):
            w = t.split() or [""]
            for s in range(0, max(1, len(w) - 50), 150):
                ch.append(" ".join(w[s:s + 200]))
                own.append(k)
        def run(s, e):
            with torch.no_grad():
                enc = tok(ch[s:e], return_tensors="pt", truncation=True, max_length=512, padding=True).to(dev)
                return model(**enc).logits.float().softmax(-1)[:, pos].cpu().numpy().tolist()
        p = _batched(run, len(ch), 32, "prompt-injection scoring")
        res = [0.0] * len(texts)
        for k, x in zip(own, p):
            res[k] = max(res[k], x)
        return res
    return score, name


def pi_scores(cfg):
    docs = documents(cfg)
    score, name = pi_scorer(cfg)
    t0 = time.time()
    sc = score([d[1] for d in docs])
    write_jsonl([{"hash": d[0], "pi": s} for d, s in zip(docs, sc)], D / "pi.jsonl")
    _usage("pi", len(docs), time.time() - t0, model=name)


def train(cfg):
    docs = documents(cfg)
    _, ex = _train_examples(cfg)
    all_t, all_q = [d[1] for d in docs], [d[2] for d in docs]
    rows = {d[0]: {"hash": d[0]} for d in docs}
    legit = set(cfg["families"]["legit"])
    trf = cfg["train_families"]
    seeds = cfg.get("detector", {}).get("seeds", [0])
    labels = {f: i + 1 for i, f in enumerate(trf)}
    timing = {}
    for si, seed in enumerate(seeds):
        sfx = "" if si == 0 else f"_s{seed}"
        save = (lambda n: D / f"{n}_model") if si == 0 else (lambda n: None)
        # NeutraLens-amortised
        for name, keep in [("amort_all", lambda f: True),
                           ("amort_heldout", lambda f: f in ("benign", "other") or f in legit or f in trf)]:
            sub = [e for e in ex if keep(e[2])]
            sc, dt = _fit_score(cfg, name + sfx, sub, [float(np.clip(e[3], -0.3, 0.6)) for e in sub], True, all_t,
                                all_q, seed=seed, save_dir=save(name))
            for d, s in zip(docs, sc):
                rows[d[0]][name + sfx] = s
            if dt is not None:
                timing[name] = dt
        # C-GEO-Guard-lite: classes = benign (incl. legit + other) and each train family
        sub = [e for e in ex if e[2] in labels or e[2] in ("benign", "other") or e[2] in legit]
        sc, _ = _fit_score(cfg, "cgeo" + sfx, sub, [labels.get(e[2], 0) for e in sub], False, all_t, all_q,
                           n_labels=len(trf) + 1, seed=seed, save_dir=save("cgeo"))
        for d, s in zip(docs, sc):
            rows[d[0]]["cgeo" + sfx] = s
    write_jsonl(list(rows.values()), D / "trained.jsonl")
    _usage("amortised", len(docs), timing.get("amort_all", 0.0), model=cfg["models"]["detector_backbone"],
           n_train_docs=len(ex), seeds=seeds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["llm", "ppl", "train", "altattack", "llm_alt", "pi", "lofo"])
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    # less fragmentation for variable-length DeBERTa batches (only when this module is the entry point: never in
    # processes that also start vLLM, e.g. adaptive.py)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cfg = load_config(a.config)
    D.mkdir(parents=True, exist_ok=True)
    {"llm": llm_scores, "ppl": ppl_scores, "train": train, "altattack": score_altattack,
     "llm_alt": lambda c: llm_scores(c, alt=True), "pi": pi_scores, "lofo": lofo}[a.phase](cfg)



def score_altattack(cfg):
    """trained detectors (never re-trained) + perplexity applied to the alternative attacker's documents"""
    ppl_scores(cfg, alt=True)
    qd = {q["qid"]: q for q in read_jsonl(G / "queries.jsonl")}
    var = read_jsonl(G / "variants_alt.jsonl")
    texts, queries = [v["text"] for v in var], [qd[v["qid"]]["query"] for v in var]
    rows = {h(t): {"hash": h(t)} for t in texts}
    for name in ["amort_all", "amort_heldout", "cgeo"]:
        sc = load_amortized(name)(texts, queries)
        for t, x in zip(texts, sc):
            rows[h(t)][name] = x
    try:  # off-the-shelf prompt-injection classifier on the alternative attacker's documents
        for t, x in zip(texts, pi_scorer(cfg)[0](texts)):
            rows[h(t)]["pi"] = x
    except Exception as ex:  # the classifier is an optional baseline: never lose the other scores
        log.warning(f"prompt-injection classifier unavailable: {ex}")
    write_jsonl(list(rows.values()), D / "altattack.jsonl")


def load_amortized(name="amort_all"):
    """scorer from a saved amortised detector (used by the adaptive-attack experiments)"""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(D / f"{name}_model")
    model = AutoModelForSequenceClassification.from_pretrained(D / f"{name}_model").to(dev).eval()

    def score(texts, queries):
        def run(s, e):
            with torch.no_grad():
                enc = tok(queries[s:e], texts[s:e], return_tensors="pt", truncation=True, max_length=512,
                          padding=True).to(dev)
                lg = model(**enc).logits.float()
            return (lg.squeeze(-1) if lg.shape[-1] == 1 else 1 - lg.softmax(-1)[:, 0]).cpu().numpy().tolist()
        return _batched(run, len(texts), 32, f"{name} scoring")  # shares the GPU with vLLM in adaptive.py
    return score


if __name__ == "__main__":
    main()
