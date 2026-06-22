"""Per-run experiment manifests — the tracked, light, mechanical record of a run.

Each experiment records its *mechanical* facts here (config / metrics / artifact
paths / commit / date) so the roadmap can show "this ran, when, at which commit,
with what numbers" without anyone hand-editing roadmap_registry.py. The registry
stays the home for the *narrative* (motivation / verdict / how-to-proceed); the two
join by `eid` and do not duplicate each other.

Manifests are committed (experiments/manifests/<eid>.json); the heavy artifacts
they reference stay in results/ (gitignored) — only relative paths are stored.

Usage in a driver (call once, after a run has its numbers):

    from manifest import write_manifest
    write_manifest("E37", config=vars(args),
                   metrics={"geneval_macro": 0.655, "baseline": 0.644},
                   artifacts=["e37/plots/band.png"])
"""
import datetime
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST_DIR = os.path.join(HERE, "manifests")


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=HERE, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def write_manifest(eid, config=None, metrics=None, artifacts=None,
                   script=None, results_dir=None, source="run"):
    """Write/update experiments/manifests/<eid>.json. Re-running merges (only the
    fields you pass are overwritten), stamping today's date + the current commit."""
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    data = load_manifest(eid) or {"eid": eid}
    data.update({"logged": datetime.date.today().isoformat(),
                 "git_commit": _git_commit(), "source": source})
    if script is not None:
        data["script"] = script
    if results_dir is not None:
        data["results_dir"] = results_dir
    if config is not None:
        data["config"] = config
    if metrics is not None:
        data["metrics"] = metrics
    if artifacts is not None:
        data["artifacts"] = artifacts
    data.setdefault("config", {})
    data.setdefault("metrics", {})
    data.setdefault("artifacts", [])
    path = os.path.join(MANIFEST_DIR, f"{eid}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def load_manifest(eid):
    p = os.path.join(MANIFEST_DIR, f"{eid}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def load_all():
    """All manifests as {eid: dict}."""
    if not os.path.isdir(MANIFEST_DIR):
        return {}
    out = {}
    for fn in os.listdir(MANIFEST_DIR):
        if fn.endswith(".json"):
            out[fn[:-5]] = json.load(open(os.path.join(MANIFEST_DIR, fn)))
    return out
