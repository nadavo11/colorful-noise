"""E50 generation: run FLUX.1-Kontext-dev over every E50 job. Run in the uv env (diffusers 0.38).

Reuses the E49 FluxKontext loader (baseline_establishment/lib/models.py) verbatim — same 4-bit NF4
transformer, same pipeline — so E50 differs from the E49 Kontext baseline only in (a) the spectral
op applied to the INPUT image and (b) the instruction wording. Loads the model once, runs all 66
jobs, tears down. Writes outputs + a results manifest with per-job wall-clock and ok flag.
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import config as C

# reuse E49's tested Kontext loader
sys.path.insert(0, str(C.E49 / "lib"))
from models import FluxKontext   # noqa: E402


def load_jobs():
    return [json.loads(l) for l in open(C.MANIFESTS / "e50_jobs.jsonl") if l.strip()]


def main(limit=None):
    jobs = load_jobs()
    if limit:
        jobs = jobs[:limit]
    print(f"[E50] {len(jobs)} Kontext jobs @ {C.GEN_SIZE}px / {C.STEPS} steps / g={C.GUIDANCE}")
    m = FluxKontext().load()
    results, t0 = [], time.time()
    for i, j in enumerate(jobs, 1):
        out_path = C.REPO / j["output"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        inp = C.REPO / j["input_image"]
        t = time.time()
        try:
            img = m.run(content=inp, instruction=j["instruction"], seed=j["seed"],
                        steps=j["steps"], guidance=j["guidance"], size=j["size"])
            img.save(out_path)
            ok, err = True, None
        except Exception as e:
            ok, err = False, repr(e)
            print(f"  !! {j['job_id']} FAILED: {err}")
        dt = round(time.time() - t, 1)
        rec = dict(j, seconds=dt, ok=ok, error=err)
        results.append(rec)
        print(f"  [{i:2d}/{len(jobs)}] {j['job_id']:34s} {j['spectral_op']:24s} {dt:5.1f}s ok={ok}")
        # incremental save so a crash leaves a partial record
        with open(C.MANIFESTS / "e50_results.jsonl", "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
    m.teardown()
    nok = sum(r["ok"] for r in results)
    print(f"[E50] done: {nok}/{len(results)} ok in {round(time.time()-t0)}s")


if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(lim)
