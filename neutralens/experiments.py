"""Registry of every experiment of the NeutraLens paper -> scheduler jobs.

List order = scheduling priority (the scheduler is first-fit in list order).  Evaluation and aggregation are NOT jobs:
run_all.py calls `python -m neutralens.evaluate --all` and `python -m neutralens.aggregate` after all jobs, so they
always see every finished output (and are re-run on resume).  --quick restricts the job list only (same data, same
splits, nested test subsets), never the data.
"""
import os
import re

from .scheduler import Job, py, visible_gpus
from .utils import DATA, ROOT, load_config, slug

R = ROOT / "results"
G = DATA / "geo"


def gated_accessible(model):
    """True if a gated HF model can actually be downloaded with the current HF_TOKEN (checked once per model)."""
    import urllib.request
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        return False
    base = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    req = urllib.request.Request(f"{base}/{model}/resolve/main/config.json", method="HEAD",
                                 headers={"Authorization": f"Bearer {tok}"})
    try:
        urllib.request.urlopen(req, timeout=30)
        return True
    except Exception:
        return False


def gpus_for(model, gpu_gb=24.0):
    """1 GPU if the weights leave >= ~1/3 of the card for the KV cache, else 2 (TP=2; clamped by the scheduler)."""
    m = re.search(r"(\d+(?:\.\d+)?)[Bb]\b", model.split("/")[-1].replace("-", " "))
    if not m:
        return 1
    gb = float(m.group(1)) * (0.55 if re.search(r"awq|gptq|int4", model, re.I) else 2.0)
    return 2 if gb > 0.65 * gpu_gb else 1


def build_jobs(config_path=None, quick=False):
    cfg = load_config(config_path)
    mc = cfg["models"]
    c = ["--config", config_path] if config_path else []
    gpu_gb = float(mc.get("gpu_gb", 24))
    ng = lambda m: gpus_for(m, gpu_gb)
    engines = []
    for m in mc["engines"]:
        if m.startswith("meta-llama/") and not gated_accessible(m):
            fb = mc.get("engine_fallback")
            print(f"[experiments] no access to gated model {m} (HF_TOKEN missing or access not granted) -> "
                  f"{'replaced by ' + fb if fb else 'skipped'}", flush=True)
            m = fb
        if m and m not in engines:
            engines.append(m)
    main_e = engines[0]
    E = lambda e: f"engine_{slug(e)}"
    ab, ec = cfg.get("ablation", {}), cfg["engine"]
    test = ["--splits", "test"]

    J = [Job("select", py("neutralens.build", "--phase", "select", *c), G / "queries.jsonl", n_gpus=0),
         Job("attack", py("neutralens.build", "--phase", "attack", *c), G / "variants.jsonl", ["select"],
             n_gpus=ng(mc["attacker"])),
         Job("neutralize", py("neutralens.build", "--phase", "neutralize", *c), G / "neutral.jsonl", ["attack"],
             n_gpus=ng(mc["neutralizer"])),
         # main engine: every split (train/dev labels for the amortised detector + dev calibration of M)
         Job(E(main_e), py("neutralens.engine", "--engine", main_e, *c), DATA / "engine" / slug(main_e) / "gens.jsonl",
             ["neutralize"], n_gpus=ng(main_e)),
         Job("detect_llm", py("neutralens.detectors", "--phase", "llm", *c), DATA / "detect" / "llm.jsonl", ["attack"],
             n_gpus=ng(mc["judge"])),
         Job("detect_ppl", py("neutralens.detectors", "--phase", "ppl", *c), DATA / "detect" / "ppl.jsonl", ["attack"]),
         Job("facts", py("neutralens.build", "--phase", "facts", *c), G / "facts.jsonl", ["attack"],
             n_gpus=ng(mc["neutralizer"])),
         Job("nli", py("neutralens.build", "--phase", "nli", *c), R / "preservation.json", ["facts", "neutralize"]),
         Job(f"judge_{slug(main_e)}", py("neutralens.judge", "--engine", main_e, "--conds", "all", *c),
             DATA / "judge" / f"{slug(main_e)}.jsonl", [E(main_e)], n_gpus=ng(mc["judge"])),
         Job("detect_train", py("neutralens.detectors", "--phase", "train", *c), DATA / "detect" / "trained.jsonl",
             [E(main_e)])]
    if quick:
        return J
    # other engine families: TEST split, reduced context set
    for e in engines[1:]:
        J.append(Job(E(e), py("neutralens.engine", "--engine", e, *test, "--ctx", ec["extra_engine_ctx"], *c),
                     DATA / "engine" / slug(e) / "gens.jsonl", ["neutralize"], n_gpus=ng(e)))
        J.append(Job(f"judge_{slug(e)}", py("neutralens.judge", "--engine", e, "--conds", "benign_legit", *c),
                     DATA / "judge" / f"{slug(e)}.jsonl", [E(e)], n_gpus=ng(mc["judge"])))
    # cross-attacker generalisation: a different attacker LLM, nothing re-trained or re-calibrated
    J += [Job("attack_alt", py("neutralens.build", "--phase", "attack_alt", *c), G / "variants_alt.jsonl", ["select"],
              n_gpus=ng(mc["attacker_alt"])),
          Job("neutralize_altattack", py("neutralens.build", "--phase", "neutralize_altattack", *c),
              G / "neutral_altattack.jsonl", ["attack_alt", "neutralize"], n_gpus=ng(mc["neutralizer"]))]
    aa_args = ["--max_test_queries", ab.get("alt_attacker_queries", 150), *test, "--ctx", "plain,neutral_target,benign_n",
               "--variants_file", "variants_alt.jsonl", "--neutral_file", "neutral_altattack.jsonl",
               "--out", "gens_altattack.jsonl", *c]
    for e in engines[:2]:
        J.append(Job(f"engine_altattack_{slug(e)}", py("neutralens.engine", "--engine", e, *aa_args),
                     DATA / "engine" / slug(e) / "gens_altattack.jsonl", ["neutralize_altattack", E(e)], n_gpus=ng(e)))
    J += [Job("detect_altattack", py("neutralens.detectors", "--phase", "altattack", *c), DATA / "detect" / "altattack.jsonl",
              ["attack_alt", "detect_train"]),
          Job("detect_llm_alt", py("neutralens.detectors", "--phase", "llm_alt", *c), DATA / "detect" / "llm_alt.jsonl",
              ["attack_alt"], n_gpus=ng(mc["judge"]))]
    # neutraliser ablation: a different model family neutralises, the main engine re-answers (test subset)
    J += [Job("neutralize_alt", py("neutralens.build", "--phase", "neutralize_alt", *c), G / "neutral_alt.jsonl",
              ["attack"], n_gpus=ng(mc["neutralizer_alt"])),
          Job("engine_altneutral", py("neutralens.engine", "--engine", main_e, "--max_test_queries",
                                      ab.get("alt_neutralizer_queries", 150), *test, "--ctx", "plain,neutral_target",
                                      "--neutral_file", "neutral_alt.jsonl", "--out", "gens_altneutral.jsonl", *c),
              DATA / "engine" / slug(main_e) / "gens_altneutral.jsonl", ["neutralize_alt", E(main_e)], n_gpus=ng(main_e)),
          Job("facts_alt", py("neutralens.build", "--phase", "facts_alt", *c), G / "facts_alt.jsonl", ["attack_alt"],
              n_gpus=ng(mc["neutralizer"])),
          Job("nli_alt", py("neutralens.build", "--phase", "nli_alt", *c), R / "preservation_alt.json",
              ["facts", "facts_alt", "neutralize_alt", "neutralize_altattack"])]
    # larger generative engine on a test subset
    se = mc.get("scale_engine")
    n_vis = len(visible_gpus())
    if se and n_vis and ng(se) > n_vis and mc.get("scale_engine_fallback"):
        # 32B-AWQ needs TP=2 on 24 GB cards; on one card it never fits next to other users' processes (Oct 13/14: two
        # 6 h job attempts failed). Same rule as the gated-Llama fallback: replace it and say so in the log.
        print(f"[experiments] scale engine {se} needs {ng(se)} GPUs but only {n_vis} visible -> replaced by "
              f"{mc['scale_engine_fallback']} (set models.scale_engine_fallback: null to force {se})", flush=True)
        se = mc["scale_engine_fallback"]
    if se and se not in engines:  # (an engine already in `engines` has its full run)
        J.append(Job(E(se), py("neutralens.engine", "--engine", se, "--max_test_queries",
                               ab.get("scale_engine_queries", 120), *test, "--ctx",
                               ab.get("scale_engine_ctx", "plain,neutral_target,removed"), *c),
                     DATA / "engine" / slug(se) / "gens.jsonl", ["neutralize"], n_gpus=ng(se)))
    # multi-turn research agent (attacker pages by the attacker, neutralisation by the neutraliser)
    J += [Job("agent_prep_attack", py("neutralens.agent", "--phase", "prep_attack", *c), DATA / "agent" / "prep.jsonl",
              ["select"], n_gpus=ng(mc["attacker"])),
          Job("agent_prep_neutral", py("neutralens.agent", "--phase", "prep_neutral", *c),
              DATA / "agent" / "neutral_agent.jsonl", ["agent_prep_attack", "neutralize"], n_gpus=ng(mc["neutralizer"]))]
    for e in engines[:2]:
        J.append(Job(f"agent_{slug(e)}", py("neutralens.agent", "--phase", "run", "--engine", e, *c),
                     R / "agent" / f"{slug(e)}.json", ["agent_prep_neutral"], n_gpus=ng(e)))
    J += [Job("adaptive", py("neutralens.adaptive", *c), R / "adaptive.json", ["detect_train", E(main_e)],
              n_gpus=ng(main_e))]
    # reviewer additions: literature defences under the same engines, GEO subjective impression, off-the-shelf
    # prompt-injection classifier, leave-one-family-out transfer, latency
    bc = cfg.get("baselines", {})
    J.append(Job("paraphrase", py("neutralens.build", "--phase", "paraphrase", *c), G / "paraphrase.jsonl", ["attack"],
                 n_gpus=ng(mc["neutralizer"])))
    for e, nq, ctx in [(main_e, bc.get("n_queries", 150),
                        bc.get("main_ctx", "paraphrase_all,paraphrase_target,spotlight,ignore_instr"))] + \
                      [(e, bc.get("extra_engine_queries", 100), bc.get("extra_ctx", "paraphrase_all,spotlight,ignore_instr"))
                       for e in engines[1:2]]:
        J.append(Job(f"engine_baselines_{slug(e)}",
                     py("neutralens.engine", "--engine", e, "--max_test_queries", nq, *test, "--ctx", ctx,
                        "--out", "gens_baselines.jsonl", *c),
                     DATA / "engine" / slug(e) / "gens_baselines.jsonl", ["paraphrase", "neutralize", E(e)], n_gpus=ng(e)))
        J.append(Job(f"judge_baselines_{slug(e)}",
                     py("neutralens.judge", "--engine", e, "--gens", "gens_baselines.jsonl", "--out",
                        f"{slug(e)}_baselines", *c),
                     DATA / "judge" / f"{slug(e)}_baselines.jsonl", [f"engine_baselines_{slug(e)}"],
                     n_gpus=ng(mc["judge"])))
    J += [Job(f"judge_si_{slug(main_e)}", py("neutralens.judge", "--engine", main_e, "--mode", "si", *c),
              DATA / "judge" / f"{slug(main_e)}_si.jsonl", [E(main_e)], n_gpus=ng(mc["judge"])),
          Job("detect_pi", py("neutralens.detectors", "--phase", "pi", *c), DATA / "detect" / "pi.jsonl", ["attack"]),
          Job("detect_lofo", py("neutralens.detectors", "--phase", "lofo", *c), DATA / "detect" / "lofo.jsonl",
              [E(main_e)]),
          Job("latency", py("neutralens.latency", *c), R / "latency.json",
              ["neutralize", "detect_train", "detect_pi", "paraphrase"],
              n_gpus=max(ng(main_e), ng(mc["neutralizer"]), ng(mc["judge"]))),
          Job("annotation_export", py("neutralens.annotate", "export", *c), R / "annotation" / "neutralization.csv",
              ["neutralize"], n_gpus=0)]
    return J
