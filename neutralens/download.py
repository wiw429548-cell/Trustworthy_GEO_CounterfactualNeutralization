"""GEO-bench (Aggarwal et al., KDD'24): queries with 5 cleaned top-search-result sources. Honors HF_ENDPOINT.
Counter-GEO-Bench is gated (manual approval on Hugging Face) and is therefore not downloaded automatically."""
import os
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HF = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def fetch(url, dst, retries=5):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            dst.write_bytes(urllib.request.urlopen(req, timeout=300).read())
            print("[download] ok", dst, flush=True)
            return
        except Exception as e:  # noqa
            print(f"[download] retry {a + 1}: {e}", flush=True)
            time.sleep(5 * (a + 1))
    raise RuntimeError(url)


def main():
    for f in ["train.jsonl", "val.jsonl", "test.jsonl", "README.md"]:
        fetch(f"{HF}/datasets/GEO-Optim/geo-bench/resolve/main/{f}", ROOT / "data" / "raw" / "geo-bench" / f)


if __name__ == "__main__":
    main()
