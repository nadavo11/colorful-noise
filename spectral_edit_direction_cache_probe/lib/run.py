"""E51 main run (uv env) — GENERATION ONLY. Per PIE-Bench example:
  1. record the full-compute reference trajectory (v_edit, v_src at each x_t),
  2. compute per-step cacheability diagnostics + per-variant skip schedules,
  3. closed-loop run the 4 cache variants at the primary skip ratio (all examples),
  4. on the Pareto subset, sweep skip ratios for the speed-quality frontier.

Saves all images + a generation manifest (paths, skip schedules, forward counts). Quality
metrics are computed afterwards by evaluate.py in the anaconda env (uv torch<2.6 can't load
the .bin metric checkpoints). Two-env split, same as E49/E50.
Run:  $UVPY lib/run.py            (full)   |   $UVPY lib/run.py --limit 2  (smoke)
"""
from __future__ import annotations
import argparse, json, time, traceback
from pathlib import Path
import numpy as np

import config as C
import data as D
import signals as SG
import flux_engine as FE


def _save_img(img, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="limit #examples (smoke test)")
    ap.add_argument("--steps", type=int, default=C.STEPS)
    ap.add_argument("--skip-pareto", action="store_true", help="skip the Pareto sweep")
    args = ap.parse_args()

    C.ensure_dirs()
    man = D.build()
    examples = man["examples"]
    subset = set(man["pareto_subset"])
    if args.limit:
        examples = examples[:args.limit]

    pipe = FE.load_pipe()
    print(f"[run] pipe loaded; {len(examples)} examples; steps={args.steps}")

    gen_path = C.DIAG / "generation.jsonl"
    rf = open(gen_path, "w")
    t_start = time.time()

    for ei, ex in enumerate(examples):
        eid = ex["id"]
        try:
            t0 = time.time()
            emb_src = FE.encode(pipe, ex["source_prompt"])
            emb_tgt = FE.encode(pipe, ex["target_prompt"])

            def fresh():   # identical seed -> identical init latent + reset scheduler each call
                return FE.prepare(pipe, ex["source_image"], steps=args.steps)

            ref_img, V_edit, V_src, SP_edit, SP_delta = FE.record_reference(pipe, fresh(), emb_src, emb_tgt)
            in_sample = C.SAMP / eid
            _save_img(FE._img(ex["source_image"]), in_sample / "input.png")
            _save_img(ref_img, in_sample / "full_compute_reference.png")

            want_amp = eid in subset
            diag, amp = SG.per_step(V_edit, V_src, SP_edit, SP_delta, want_amp=want_amp)
            diag["id"] = eid; diag["task_type"] = ex["task_type"]; diag["category"] = ex["category"]
            diag["scope"] = ex["scope"]
            C.write_json(C.DIAG / f"trajectory_{eid}.json", diag)
            if amp is not None:
                np.savez_compressed(C.DIAG / f"spectra_{eid}.npz",
                                    amp_edit=amp["amp_edit"].astype(np.float32),
                                    amp_delta=amp["amp_delta"].astype(np.float32))

            n = diag["n"]
            # ---- closed-loop cache variants at the primary skip ratio (all examples)
            ratios = [C.SKIP_PRIMARY] + (C.PARETO_TARGETS if (eid in subset and not args.skip_pareto) else [])
            ratios = sorted(set(ratios))

            common = dict(id=eid, task_type=ex["task_type"], category=ex["category"], scope=ex["scope"],
                          source_image=ex["source_image_rel"], source_prompt=ex["source_prompt"],
                          target_prompt=ex["target_prompt"])
            # reference manifest row
            rf.write(json.dumps(dict(**common, variant="full_compute_reference", skip_ratio=0.0,
                                     realized_skip=0.0, fwd_edit=n, fwd_src=0,
                                     image=C.rel(in_sample / "full_compute_reference.png"),
                                     reference=C.rel(in_sample / "full_compute_reference.png"),
                                     is_primary=True)) + "\n")
            rf.flush()

            for variant in C.CACHE_VARIANTS:
                spec = C.VARIANT_SPEC[variant]
                change = diag[SG.SIGNAL_KEY[variant]]
                for rho in ratios:
                    skip, realized = SG.skip_schedule(change, rho, n)
                    out, fwd = FE.denoise_cached(pipe, fresh(), emb_src, emb_tgt, skip, spec["signal"])
                    primary = abs(rho - C.SKIP_PRIMARY) < 1e-6
                    if primary:
                        img_path = in_sample / f"{variant}.png"
                    else:
                        img_path = in_sample / "sweep" / f"{variant}__r{rho:.2f}.png"
                    _save_img(out, img_path)
                    rf.write(json.dumps(dict(**common, variant=variant, skip_ratio=rho,
                                             realized_skip=realized, fwd_edit=fwd["fwd_edit"],
                                             fwd_src=fwd["fwd_src"], n_steps=n,
                                             image=C.rel(img_path),
                                             reference=C.rel(in_sample / "full_compute_reference.png"),
                                             is_primary=primary)) + "\n")
                    rf.flush()

            dt = time.time() - t0
            print(f"[run] {ei+1}/{len(examples)} {eid} ({ex['task_type']}) done in {dt:.1f}s "
                  f"(elapsed {(time.time()-t_start)/60:.1f}m)")
        except Exception as e:
            print(f"[run] ERROR on {eid}: {e}")
            traceback.print_exc()

    rf.close()
    print(f"[run] all done in {(time.time()-t_start)/60:.1f}m -> {gen_path}")


if __name__ == "__main__":
    main()
