"""E50 evaluation: compute the E49 metric suite on every E50 generation. Run in anaconda env.

Reuses baseline_establishment/lib/metrics.full_metrics so E50 numbers are directly comparable to
the E49 baseline. Emits metrics/e50_metrics.csv (one row per generation) and a summary JSON with
per-experiment / per-spectral-op / per-task / per-prompt aggregates and the headline leakage gap
(DINO_content - DINO_style).
"""
from __future__ import annotations
import sys, json, csv, statistics as st
from pathlib import Path
import config as C

sys.path.insert(0, str(C.E49 / "lib"))
import metrics as M   # noqa: E402


def load_results():
    p = C.MANIFESTS / "e50_results.jsonl"
    return [json.loads(l) for l in open(p) if l.strip()]


def _p(rel):
    return C.REPO / rel if rel else None


def evaluate():
    res = [r for r in load_results() if r.get("ok")]
    rows = []
    for r in res:
        out = _p(r["output"])
        if not out or not out.exists():
            continue
        content = _p(r["content_ref"])
        style = _p(r.get("style_ref"))
        m = M.full_metrics(out, content, style,
                           target_prompt=r.get("target_prompt"),
                           source_prompt=r.get("source_prompt"))
        # headline leakage gap (content-vs-style movement); only meaningful when a style ref exists
        if "dino_style" in m:
            m["leak_gap"] = m["dino_content"] - m["dino_style"]          # high = content kept, style not copied
            m["clip_leak_gap"] = m["clipI_content"] - m["clipI_style"]
        row = dict(job_id=r["job_id"], experiment=r["experiment"], bucket=r["bucket"],
                   source_id=r["source_id"], style_id=r.get("style_id"),
                   task_type=r["task_type"], spectral_op=r["spectral_op"],
                   prompt_key=r.get("prompt_key"), style_category=r.get("style_category"),
                   seconds=r.get("seconds"), output=r["output"], **{k: round(v, 5) for k, v in m.items()})
        rows.append(row)

    # ---- CSV
    cols = sorted({k for row in rows for k in row}, key=lambda c: (
        c not in ("job_id", "experiment", "bucket", "spectral_op", "prompt_key", "task_type",
                  "source_id", "style_id", "style_category"), c))
    C.METRICS.mkdir(parents=True, exist_ok=True)
    with open(C.METRICS / "e50_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    # ---- summary aggregates
    def agg(group_key, metric_keys, subset=None):
        out = {}
        for row in rows:
            if subset and not subset(row):
                continue
            g = row.get(group_key)
            out.setdefault(g, {k: [] for k in metric_keys})
            for k in metric_keys:
                if k in row:
                    out[g][k].append(row[k])
        return {g: {k: round(st.mean(v), 4) for k, v in d.items() if v} for g, d in out.items()}

    edit_keys = ["dino_content", "clipI_content", "lpips_content", "clipT_target", "clipT_gain"]
    style_keys = ["dino_content", "clipI_content", "clipI_style", "dino_style",
                  "colorhist_style", "fourier_style", "leak_gap", "clip_leak_gap"]

    summary = dict(
        phase=C.PHASE_VERSION,
        n_generations=len(rows),
        source_by_op=agg("spectral_op", edit_keys, lambda r: r["experiment"] == "spectral_source"),
        source_by_task=agg("task_type", edit_keys, lambda r: r["experiment"] == "spectral_source"),
        reference_by_op=agg("spectral_op", style_keys, lambda r: r["experiment"] == "spectral_reference"
                            or (r["experiment"] == "spectral_reference")),
        reference_all=agg("experiment", style_keys, lambda r: r["bucket"] in ("spectral_reference", "kontext_baseline_replay")),
        reference_op_full=agg("spectral_op", style_keys,
                              lambda r: r["bucket"] in ("spectral_reference", "kontext_baseline_replay")),
        prompt_by_key=agg("prompt_key", style_keys, lambda r: r["experiment"] == "prompt_variants"),
    )
    C.write_json(C.METRICS / "e50_summary.json", summary)
    print(f"evaluated {len(rows)} generations -> e50_metrics.csv + e50_summary.json")
    print("source_by_op:", json.dumps(summary["source_by_op"], indent=0)[:400])
    print("reference_op_full:", json.dumps(summary["reference_op_full"], indent=0)[:500])
    print("prompt_by_key:", json.dumps(summary["prompt_by_key"], indent=0)[:400])
    return rows, summary


if __name__ == "__main__":
    evaluate()
