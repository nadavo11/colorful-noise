"""Generation runner. Groups by model: load once -> run all its jobs -> teardown.

Job routing:
  editing benchmarks (MagicBrush, PIE-Bench)  -> flux_img2img, flux_kontext
  style/leakage pairs (custom_leakage_set)    -> flux_redux, flux_ipadapter, styleid, flux_kontext

Usage:
  python runner.py --phase smoke              # 3 jobs/model
  python runner.py --phase pilot              # full subsets
  python runner.py --phase pilot --models flux_kontext,styleid
"""
from __future__ import annotations
import argparse, json, time, traceback
from pathlib import Path
import config as C
from models import REGISTRY


def _read(p):
    p = Path(p)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def editing_jobs():
    rows = _read(C.BENCH / "magicbrush" / "manifest.jsonl") + _read(C.BENCH / "piebench" / "manifest.jsonl")
    jobs = []
    for r in rows:
        jobs.append(dict(job_id=r["id"], kind="edit", benchmark=r["benchmark"],
                         task_type=r["task_type"], content=str(C.ROOT / r["source"]),
                         instruction=r.get("instruction", ""), prompt=r.get("target_prompt", ""),
                         source_prompt=r.get("source_prompt", ""), gt_target=r.get("gt_target", ""),
                         style=None))
    return jobs


def style_jobs():
    rows = _read(C.LEAK / "manifest.jsonl")
    jobs = []
    for r in rows:
        jobs.append(dict(job_id=r["id"], kind="style", benchmark="custom_leakage",
                         task_type="style", pair_type=r["pair_type"],
                         content=str(C.ROOT / r["content"]), style=str(C.ROOT / r["style"]),
                         style_category=r["style_category"], content_id=r["content_id"],
                         style_id=r["style_id"], instruction=r["instruction"],
                         prompt=r["instruction"], source_prompt=""))
    return jobs


# Kontext runs at its native 1024 bucket (slow); fewer steps keeps the pilot tractable.
STEPS_OVERRIDE = {"flux_kontext": 20}

# which job kinds each model handles
ROUTING = {
    "flux_img2img": ["edit"],
    "flux_kontext": ["edit", "style"],
    "flux_redux": ["style"],
    "flux_ipadapter": ["style"],
    "styleid": ["style"],
}


def run_model(mid, jobs, phase, seed=0):
    out_dir = C.OUT / mid
    out_dir.mkdir(parents=True, exist_ok=True)
    man = out_dir / f"manifest_{phase}.jsonl"
    kinds = ROUTING[mid]
    sel = [j for j in jobs if j["kind"] in kinds]
    if phase == "smoke":
        # 2 edit + 1 style if available
        e = [j for j in sel if j["kind"] == "edit"][:2]
        s = [j for j in sel if j["kind"] == "style"][:1]
        sel = (e + s) or sel[:3]
    print(f"[{mid}] {len(sel)} jobs ({phase})", flush=True)
    model = REGISTRY[mid]().load()
    print(f"[{mid}] loaded", flush=True)
    records = []
    for i, j in enumerate(sel):
        t0 = time.time()
        outp = out_dir / f"{j['job_id']}_{mid}_s{seed}.png"
        try:
            img = model.run(content=j["content"], prompt=j.get("prompt"),
                            instruction=j.get("instruction"), style=j.get("style"),
                            seed=seed, steps=STEPS_OVERRIDE.get(mid, C.INFER_STEPS),
                            guidance=C.GUIDANCE, size=C.GEN_SIZE)
            img.save(outp)
            rec = dict(j, model=mid, seed=seed, output=str(outp.relative_to(C.ROOT)),
                       seconds=round(time.time() - t0, 1), ok=True)
        except Exception as e:
            traceback.print_exc()
            rec = dict(j, model=mid, seed=seed, output="", ok=False, error=repr(e)[:300])
        records.append(rec)
        print(f"  [{mid}] {i+1}/{len(sel)} {j['job_id']} {'ok' if rec['ok'] else 'FAIL'} "
              f"{rec.get('seconds','-')}s", flush=True)
    model.teardown()
    man.write_text("\n".join(json.dumps(r) for r in records))
    n_ok = sum(r["ok"] for r in records)
    print(f"[{mid}] DONE {n_ok}/{len(records)} ok -> {man}", flush=True)
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="smoke", choices=["smoke", "pilot"])
    ap.add_argument("--models", default=",".join(REGISTRY))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    C.ensure_dirs()
    jobs = editing_jobs() + style_jobs()
    mids = [m for m in a.models.split(",") if m in REGISTRY]
    print(f"== phase={a.phase} models={mids} total_jobs={len(jobs)} ==", flush=True)
    for mid in mids:
        try:
            run_model(mid, jobs, a.phase, seed=a.seed)
        except Exception:
            traceback.print_exc()
            print(f"[{mid}] MODEL-LEVEL FAILURE", flush=True)


if __name__ == "__main__":
    main()
