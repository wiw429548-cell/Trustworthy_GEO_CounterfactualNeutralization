"""One-command entry point for ALL experiments of the NeutraLens paper.

    CUDA_VISIBLE_DEVICES=0,1 nohup python run_all.py > run.log 2>&1 &

(1) download GEO-bench  (2) query-grouped train/dev/test split, rewrite families, fact-preserving neutralisation,
    bidirectional NLI preservation audit  (3) generative-engine runs (counterfactual contexts) for every engine family
(4) detectors (LLM / PPL / trained, 3 seeds) and answer judge  (5) ablations: second neutraliser, second attacker,
    larger engine; multi-turn agent hijacking; adaptive attacks  (6) `evaluate --all` (dev-calibrated thresholds,
    query-cluster bootstrap CIs) and `aggregate` -> results/summary.md, results/latex/*.tex, results/figures/*
Every job is resumable (done-file per job; LLM outputs cached in data/cache/).  Progress: logs/_status.json, run.log.
Options: --quick (main engine only, no ablations / agent / adaptive)  --config path  --skip_download
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
from neutralens import runtime  # noqa: E402  (stdlib-only module)

runtime.setup_env()  # VLLM_USE_FLASHINFER_SAMPLER=0 (no nvcc on the server), UTF-8, PCI_BUS_ID order, ...

REQS = [("yaml", "pyyaml"),
        ("numpy", "numpy"),
        ("sklearn", "scikit-learn"),
        ("matplotlib", "matplotlib"),
        ("transformers", "transformers>=4.51"),
        ("sentence_transformers", "sentence-transformers>=3.0"),
        ("sentencepiece", "sentencepiece"),
        ("google.protobuf", "protobuf"),
        ("pandas", "pandas"),
        ("scipy", "scipy"),
        ("pyarrow", "pyarrow")]


def check_env():
    """installs only MISSING packages (never upgrades torch; serialised across projects; see runtime.py)."""
    reqs = list(REQS)
    if os.environ.get("LLM_BACKEND", "vllm") != "hf":
        reqs.append(("vllm", "vllm>=0.8.5"))
    runtime.ensure_packages(reqs)
    import torch
    print(f"[run_all] python {sys.version.split()[0]} torch {torch.__version__} cuda={torch.cuda.is_available()} "
          f"VLLM_USE_FLASHINFER_SAMPLER={os.environ.get('VLLM_USE_FLASHINFER_SAMPLER')}", flush=True)
    runtime.report_gpus(lambda s: print(s, flush=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--config", default=None)
    ap.add_argument("--skip_download", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    check_env()
    from neutralens import download
    from neutralens.experiments import build_jobs
    from neutralens.scheduler import run_jobs

    if not a.skip_download:
        try:
            download.main()
        except Exception as ex:  # a dead mirror must not stop a run whose data is already on disk
            if not (ROOT / "data" / "raw" / "geo-bench" / "test.jsonl").exists():
                raise
            print(f"[run_all] download failed ({type(ex).__name__}: {ex}); using the existing data/raw/geo-bench",
                  flush=True)
    summary = run_jobs(build_jobs(a.config, a.quick), ROOT / "logs", ROOT)
    cfg_args = ["--config", a.config] if a.config else []
    # evaluation / aggregation always run (also after partial failures) on whatever outputs exist
    with open(ROOT / "logs" / "evaluate.log", "w", encoding="utf-8") as f:
        rc_eval = subprocess.call([sys.executable, "-m", "neutralens.evaluate", "--all", *cfg_args], cwd=ROOT,
                                  stdout=f, stderr=subprocess.STDOUT)
    with open(ROOT / "logs" / "aggregate.log", "w", encoding="utf-8") as f:
        rc_agg = subprocess.call([sys.executable, "-m", "neutralens.aggregate", *cfg_args], cwd=ROOT,
                                 stdout=f, stderr=subprocess.STDOUT)
    print(f"[run_all] evaluate rc={rc_eval} aggregate rc={rc_agg} (logs/evaluate.log, logs/aggregate.log)", flush=True)
    print(f"[run_all] finished in {(time.time() - t0) / 3600:.1f} h; failed={summary['failed']} "
          f"skipped={summary['skipped']}; see results/summary.md", flush=True)


if __name__ == "__main__":
    main()
