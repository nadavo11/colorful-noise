"""Metrics pass. Reads every outputs/<model>/manifest_<phase>.jsonl, computes the metric
row per generated image, writes the machine-readable metrics CSV + summary JSON.

Run in the anaconda env (transformers + lpips + torchvision available there).
  python evaluate.py --phase pilot
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from collections import defaultdict
import config as C
import metrics as M


def load_outputs(phase):
    recs = []
    for mid in C.MODELS:
        man = C.OUT / mid / f"manifest_{phase}.jsonl"
        if not man.exists():
            continue
        for line in man.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("ok") and r.get("output"):
                recs.append(r)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="pilot")
    a = ap.parse_args()
    recs = load_outputs(a.phase)
    print(f"evaluating {len(recs)} outputs ({a.phase})")
    rows = []
    for i, r in enumerate(recs):
        out = C.ROOT / r["output"]
        content = C.ROOT / Path(r["content"]).relative_to(C.ROOT) if str(r["content"]).startswith(str(C.ROOT)) else Path(r["content"])
        style = r.get("style")
        style = (Path(style) if style else None)
        try:
            m = M.full_metrics(out, content, style=style,
                               target_prompt=r.get("prompt") or r.get("instruction"),
                               source_prompt=r.get("source_prompt") or None)
        except Exception as e:
            print("  metric ERR", r["job_id"], repr(e)[:120]); continue
        row = dict(model=r["model"], job_id=r["job_id"], kind=r["kind"],
                   benchmark=r.get("benchmark", ""), task_type=r.get("task_type", ""),
                   pair_type=r.get("pair_type", ""), style_category=r.get("style_category", ""),
                   seconds=r.get("seconds", ""), output=r["output"], **m)
        rows.append(row)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(recs)}")
    # write CSV
    cols = sorted({k for row in rows for k in row}, key=lambda k: (k not in
                  ["model", "job_id", "kind", "benchmark", "task_type", "pair_type",
                   "style_category"], k))
    csv_path = C.METRICS / "baseline_establishment_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for row in rows:
            w.writerow(row)
    # summary: per-model and per-(model,task) means
    def agg(group_key):
        g = defaultdict(list)
        for row in rows:
            g[group_key(row)].append(row)
        out = {}
        for k, items in g.items():
            num = defaultdict(list)
            for it in items:
                for kk, vv in it.items():
                    if isinstance(vv, (int, float)):
                        num[kk].append(vv)
            out[k] = {kk: round(sum(v) / len(v), 4) for kk, v in num.items() if v}
            out[k]["n"] = len(items)
        return out
    summary = dict(
        phase=a.phase, n_outputs=len(rows),
        per_model={str(k): v for k, v in agg(lambda r: r["model"]).items()},
        per_model_task={f"{k[0]}|{k[1]}": v for k, v in
                        agg(lambda r: (r["model"], r["task_type"])).items()},
        per_model_pairtype={f"{k[0]}|{k[1]}": v for k, v in
                            agg(lambda r: (r["model"], r["pair_type"])).items() if k[1]},
    )
    C.write_json(C.METRICS / "baseline_establishment_summary.json", summary)
    print(f"wrote {csv_path} ({len(rows)} rows) and summary.json")


if __name__ == "__main__":
    main()
