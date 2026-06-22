"""Frozen canonical evaluation fixture — so cross-experiment comparisons are valid
by construction (same prompts, same seeds, everywhere).

`canonical_prompts()` + `SEEDS` are the small, FROZEN, reproducible set every new
experiment should default to for its comparison runs. The prompt slice is taken
deterministically from the committed GenEval metadata (geneval_data/), spanning all
six tags, plus a few hand-authored dense prompts for the long-prompt regime GenEval
lacks — so the set reproduces anywhere with no downloads.

For headline runs against a full benchmark, use the thin re-exports
(`geneval_all`, `compbench_prompts`, `dpg_prompts`); those are larger and, for
CompBench/DPG, download their data on first use (not part of the frozen set).

Bump FIXTURE_VERSION if you ever change the canonical set, and record the version a
run used in its manifest (manifest.py).
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE_VERSION = "v1"

# Canonical seeds: small + fixed. Use all of them, or SEEDS[:k] for quick passes.
SEEDS = (0, 1, 2, 3)

# How many prompts to take per GenEval tag for the frozen set (file order = stable).
_GENEVAL_PER_TAG = 2

# Long / dense prompts (DPG-Bench style) for the regime GenEval's short captions miss.
_LONG_PROMPTS = [
    "a weathered red fishing boat moored at a stone harbour at dawn, gulls overhead, "
    "fog rolling over distant green hills, reflections rippling on the calm water",
    "a cozy wooden library interior with floor-to-ceiling shelves, a brass ladder, "
    "an orange tabby cat asleep on a leather armchair, warm afternoon light through tall windows",
    "a bustling night market street with paper lanterns, steaming food stalls, "
    "neon signs in the rain, crowds with umbrellas, puddles mirroring the colourful lights",
]


def _geneval_rows():
    path = os.path.join(HERE, "geneval_data", "evaluation_metadata.jsonl")
    return [json.loads(line) for line in open(path)]


def canonical_prompts():
    """The frozen comparison set: a balanced GenEval slice + the long prompts.
    Returns a list of {id, prompt, tag, source} dicts in a stable order."""
    out, seen = [], {}
    for r in _geneval_rows():
        tag = r["tag"]
        if seen.get(tag, 0) >= _GENEVAL_PER_TAG:
            continue
        seen[tag] = seen.get(tag, 0) + 1
        out.append({"id": f"geneval_{tag}_{seen[tag]}", "prompt": r["prompt"],
                    "tag": tag, "source": "geneval"})
    for i, p in enumerate(_LONG_PROMPTS):
        out.append({"id": f"long_{i + 1}", "prompt": p, "tag": "long", "source": "canonical"})
    return out


def geneval_all():
    """All 553 committed GenEval prompts (for a headline GenEval run)."""
    return [r["prompt"] for r in _geneval_rows()]


def compbench_prompts(**kw):
    """Full-benchmark CompBench prompts (downloads compbench_data/ on first use)."""
    from compbench import load_compbench_prompts
    return load_compbench_prompts(**kw)


def dpg_prompts(**kw):
    """Full-benchmark DPG-Bench long prompts (downloads the CSV on first use)."""
    from dpg_bench import load_dpg_prompts
    return load_dpg_prompts(**kw)


if __name__ == "__main__":
    ps = canonical_prompts()
    print(f"fixture {FIXTURE_VERSION}: {len(ps)} canonical prompts, seeds={SEEDS}")
    for p in ps:
        print(f"  [{p['tag']:13s}] {p['prompt'][:70]}")
