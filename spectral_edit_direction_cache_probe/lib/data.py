"""E51 data selection: the PIE-Bench subset with (source_prompt, target_prompt) pairs.

PIE-Bench is the only repo subset that ships an explicit source prompt AND a target prompt
per example, which is exactly what delta_edit = v_edit - v_src needs. All 24 examples are
used (8 task types x 3), already category-balanced.
"""
from __future__ import annotations
import json
import config as C


def load_examples():
    mf = C.PIEBENCH / "manifest.jsonl"
    rows = [json.loads(l) for l in mf.read_text().splitlines() if l.strip()]
    out = []
    for r in rows:
        tt = r["task_type"]
        src = C.PIEBENCH.parent.parent.parent / r["source"]   # row paths are baseline_establishment-root relative
        # row["source"] is like data/benchmark_subsets/piebench/xxx_src.png under baseline_establishment/
        src = (C.REPO / "baseline_establishment" / r["source"]).resolve()
        out.append(dict(
            id=r["id"],
            task_type=tt,
            category=C.CATEGORY[tt],
            scope=C.SCOPE[tt],
            source_image=str(src),
            source_image_rel=C.rel(src),
            source_prompt=r["source_prompt"],
            target_prompt=r["target_prompt"],
            instruction=r.get("instruction", ""),
        ))
    return out


def pareto_subset(examples):
    """One example per task type (the first of each), spanning all 8 categories
    and both local/global scope — used for the dense per-step diagnostics + Pareto sweep."""
    seen, sub = set(), []
    for e in examples:
        if e["task_type"] not in seen:
            seen.add(e["task_type"])
            sub.append(e["id"])
    return sub


def build():
    C.ensure_dirs()
    ex = load_examples()
    sub = pareto_subset(ex)
    manifest = dict(
        phase=C.PHASE_VERSION, n=len(ex), size=C.SIZE, steps=C.STEPS,
        guidance=C.GUIDANCE, strength=C.STRENGTH, seed=C.SEED,
        pareto_subset=sub, examples=ex,
    )
    C.write_json(C.DIAG / "examples.json", manifest)
    print(f"[data] {len(ex)} PIE-Bench examples, {len(sub)} in Pareto subset: {sub}")
    return manifest


if __name__ == "__main__":
    build()
