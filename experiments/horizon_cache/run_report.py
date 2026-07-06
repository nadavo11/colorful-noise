"""Build the HorizonCache HTML report + summary md/json from a finished run.

    python -m horizon_cache.run_report --gen-dir ../results/horizon_cache/gen_XXXX \
        --rollout-csv ../metrics/horizon_cache/action_dataset.csv \
        --v1-diag ../metrics/horizon_cache/v1_diag.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
REPO = HERE.parents[1]

from horizon_cache import capability
from horizon_cache.report import build


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO)).decode().strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True)
    ap.add_argument("--rollout-csv", default="")
    ap.add_argument("--v1-diag", default="")
    ap.add_argument("--reports-dir", default=str(REPO / "reports"))
    args = ap.parse_args()

    reports = Path(args.reports_dir)
    assets = reports / "horizon_cache_assets"
    v1_diag = None
    if args.v1_diag and Path(args.v1_diag).exists():
        v1_diag = json.loads(Path(args.v1_diag).read_text())
    rollout = Path(args.rollout_csv) if args.rollout_csv and Path(args.rollout_csv).exists() else None

    sj = build(Path(args.gen_dir), rollout, v1_diag,
               reports / "horizon_cache.html", reports / "horizon_cache_summary.md",
               reports / "horizon_cache_summary.json", assets,
               capability.detect(), git_hash())
    print(json.dumps({"html": str(reports / "horizon_cache.html"),
                      "status": sj["status"], "best_gen": sj["best_methods"]["generation"],
                      "verdicts": sj["verdicts"]}, indent=2))


if __name__ == "__main__":
    main()
