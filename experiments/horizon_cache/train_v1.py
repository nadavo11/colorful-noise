"""Train HorizonCache-v1 — a small tabular safe-horizon predictor from rollout labels.

Uses sklearn HistGradientBoosting (lightgbm/xgboost preferred if installed). Asymmetric
cost is imposed via sample weights: mislabeling a state as jump/cache when it should be
fresh is *dangerous* (hurts quality), so those errors are up-weighted; a false-fresh only
costs speed and is cheap. We fit on causal features only (FEATURE_NAMES) and report a
confusion matrix + false-jump / false-cache rates.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .policy import FEATURE_NAMES
from .scheduler import ACTIONS


def _cost_weight(label: str) -> float:
    # asymmetric: penalize letting a state be under-refreshed (false jump/cache) more.
    return {"fresh": 1.0, "cache": 1.5, "jump_1.25": 2.0, "jump_1.5": 2.5, "jump_2.0": 3.0}[label]


def train(csv_path: Path, out_bundle: Path, mode: str = "classify") -> dict[str, Any]:
    import csv as _csv
    rows = list(_csv.DictReader(open(csv_path)))
    if len(rows) < 20:
        return {"status": "SKIPPED", "reason": f"too few states ({len(rows)})", "n": len(rows)}

    X = np.array([[float(r[k]) for k in FEATURE_NAMES] for r in rows], dtype=np.float64)
    y_action = [r["label_action"] for r in rows]
    weights = np.array([_cost_weight(a) for a in y_action])

    rng = np.random.RandomState(0)
    idx = rng.permutation(len(rows))
    cut = int(0.75 * len(rows))
    tr, te = idx[:cut], idx[cut:]

    try:
        import lightgbm as lgb  # noqa
        backend = "lightgbm"
    except Exception:
        backend = "sklearn_histgb"

    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    if mode == "ordinal":
        y = np.array([float(r["label_horizon"]) for r in rows])
        model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, max_depth=4)
        model.fit(X[tr], y[tr], sample_weight=weights[tr])
        pred_te = model.predict(X[te])
        from .scheduler import horizon_to_action
        pred_action = [horizon_to_action(float(v)) for v in pred_te]
        true_action = [y_action[i] for i in te]
        classes = ACTIONS
    else:
        y = np.array(y_action)
        model = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, max_depth=4)
        model.fit(X[tr], y[tr], sample_weight=weights[tr])
        pred_action = list(model.predict(X[te]))
        true_action = list(y[te])
        classes = sorted(set(y_action), key=lambda a: ACTIONS.index(a))

    # confusion matrix + safety rates
    cm = {t: {p: 0 for p in ACTIONS} for t in ACTIONS}
    for t, p in zip(true_action, pred_action):
        cm[t][p] += 1
    n_te = len(true_action)
    # false jump: predicted a jump when truth was fresh (dangerous)
    false_jump = sum(1 for t, p in zip(true_action, pred_action) if p.startswith("jump") and t == "fresh")
    false_cache = sum(1 for t, p in zip(true_action, pred_action) if p == "cache" and t == "fresh")
    over_fresh = sum(1 for t, p in zip(true_action, pred_action) if p == "fresh" and t != "fresh")
    acc = sum(1 for t, p in zip(true_action, pred_action) if t == p) / max(1, n_te)

    import joblib
    out_bundle.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "classes": classes, "mode": mode,
                 "features": FEATURE_NAMES, "backend": backend}, out_bundle)

    diag = {
        "status": "DONE", "backend": backend, "mode": mode, "n_total": len(rows),
        "n_test": n_te, "accuracy": round(acc, 4),
        "false_jump_rate": round(false_jump / max(1, n_te), 4),
        "false_cache_rate": round(false_cache / max(1, n_te), 4),
        "conservative_fresh_overpredict_rate": round(over_fresh / max(1, n_te), 4),
        "confusion_matrix": cm, "bundle": str(out_bundle),
    }
    # feature importances via permutation (cheap, robust across backends)
    try:
        from sklearn.inspection import permutation_importance
        pi = permutation_importance(model, X[te], y[te], n_repeats=5, random_state=0,
                                    scoring="neg_mean_absolute_error" if mode == "ordinal" else "accuracy")
        diag["feature_importance"] = {FEATURE_NAMES[i]: round(float(pi.importances_mean[i]), 4)
                                      for i in np.argsort(pi.importances_mean)[::-1][:10]}
    except Exception as e:
        diag["feature_importance_error"] = str(e)
    return diag


def train_frontier(csv_path: Path, out_bundle: Path) -> dict[str, Any]:
    """Binary v1 on FRONTIER-IMPROVEMENT labels: predict jump-helpful (1) vs harmful (0)
    from causal features only. This is the correct target (full-rollout, compounding-aware),
    unlike latent-L2 (too strict) or short-horizon decoded-PSNR (too lenient)."""
    import csv as _csv
    rows = list(_csv.DictReader(open(csv_path)))
    if len(rows) < 16:
        return {"status": "SKIPPED", "reason": f"too few states ({len(rows)})", "n": len(rows)}
    y = np.array([int(r["label_jump_helpful"]) for r in rows])
    X = np.array([[float(r[k]) for k in FEATURE_NAMES] for r in rows], dtype=np.float64)
    pos = int(y.sum()); base = max(pos, len(y) - pos) / len(y)  # majority-class baseline
    if pos == 0 or pos == len(y):
        return {"status": "DEGENERATE", "reason": f"single class (helpful={pos}/{len(y)})",
                "n": len(rows), "frac_helpful": round(pos / len(y), 3)}
    rng = np.random.RandomState(0); idx = rng.permutation(len(rows)); cut = int(0.75 * len(rows))
    tr, te = idx[:cut], idx[cut:]
    from sklearn.ensemble import HistGradientBoostingClassifier
    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_depth=3)
    model.fit(X[tr], y[tr])
    pred = model.predict(X[te])
    acc = float((pred == y[te]).mean())
    import joblib
    out_bundle.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "classes": [0, 1], "mode": "frontier_binary",
                 "features": FEATURE_NAMES}, out_bundle)
    diag = {"status": "DONE", "task": "frontier_binary_jump_helpful", "n_total": len(rows),
            "frac_helpful": round(pos / len(y), 3), "majority_baseline": round(base, 3),
            "n_test": len(te), "test_accuracy": round(acc, 3), "bundle": str(out_bundle)}
    try:
        from sklearn.inspection import permutation_importance
        pi = permutation_importance(model, X[te], y[te], n_repeats=5, random_state=0, scoring="accuracy")
        diag["feature_importance"] = {FEATURE_NAMES[i]: round(float(pi.importances_mean[i]), 4)
                                      for i in np.argsort(pi.importances_mean)[::-1][:8]}
    except Exception as e:
        diag["feature_importance_error"] = str(e)
    return diag


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="results/horizon_cache/v1_bundle.joblib")
    ap.add_argument("--mode", choices=["classify", "ordinal"], default="classify")
    a = ap.parse_args()
    print(json.dumps(train(Path(a.csv), Path(a.out), a.mode), indent=2))
