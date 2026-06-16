"""DPG-Bench prompts (the standard LONG / dense-prompt benchmark for T2I).

DPG-Bench ("Dense Prompt Graph") ships with the ELLA repo; prompts average ~80
words and pack many objects/attributes/relations, which is exactly the regime we
want for E26 seed-alignment on long prompts. We only need the prompt strings here
(the official DPG score is a heavy mPLUG/VQA pipeline; E26 uses long-aware CLIP-T).

The full prompt table is one CSV in the ELLA repo with several rows per prompt
(one per scoring proposition); we cache it locally and dedup by item_id. Mirrors
`compbench.py::load_compbench_prompts` in spirit: a stride-sampled balanced slice.
"""
import csv
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dpg_bench_data")
CSV_URL = "https://raw.githubusercontent.com/TencentQQGYLab/ELLA/main/dpg_bench/dpg_bench.csv"
CSV_PATH = os.path.join(DATA, "dpg_bench.csv")


def _ensure_csv():
    if os.path.exists(CSV_PATH):
        return
    os.makedirs(DATA, exist_ok=True)
    print(f"[dpg] downloading {CSV_URL}", flush=True)
    req = urllib.request.Request(CSV_URL, headers={"User-Agent": "curl/8"})
    data = urllib.request.urlopen(req, timeout=60).read()
    with open(CSV_PATH, "wb") as f:
        f.write(data)
    print(f"[dpg] cached {len(data)} bytes -> {CSV_PATH}", flush=True)


def _all_prompts():
    """Unique (item_id, text), preserving CSV order (one row per proposition)."""
    _ensure_csv()
    seen, out = set(), []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid, txt = row["item_id"], row["text"].strip()
            if pid not in seen:
                seen.add(pid)
                out.append((pid, txt))
    return out


def load_dpg_prompts(n=16, min_words=45, max_words=120, stride_seed=0):
    """Balanced stride-sampled slice of LONG DPG-Bench prompts.

    Filters to prompts with `min_words..max_words` words (drops the few extreme
    outliers that blow past CLIP context even when chunked), then stride-samples
    `n` of them deterministically. Returns list of (pid, prompt).
    """
    pool = [(pid, t) for pid, t in _all_prompts()
            if min_words <= len(t.split()) <= max_words]
    if n is None or n >= len(pool):
        return pool
    stride = max(1, len(pool) // n)
    picked = pool[stride_seed::stride][:n]
    return picked


if __name__ == "__main__":
    ps = load_dpg_prompts(n=int(sys.argv[1]) if len(sys.argv) > 1 else 8)
    wc = [len(t.split()) for _, t in ps]
    print(f"{len(ps)} prompts, words avg={sum(wc)/len(wc):.1f} "
          f"min={min(wc)} max={max(wc)}\n")
    for pid, t in ps:
        print(f"[{pid}] ({len(t.split())}w) {t[:160]}{'...' if len(t) > 160 else ''}\n")
