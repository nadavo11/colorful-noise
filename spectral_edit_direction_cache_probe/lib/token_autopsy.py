"""E52 — Text-Token Modulation Autopsy: GENERATION pass (uv env, diffusers 0.38).

Runs ALONGSIDE the E51 cache probe on the same FLUX.1-dev img2img pipeline / PIE-Bench
examples. Two passes per example (autopsy is expensive, so it runs on the Pareto subset):

  1. OBSERVATIONAL  — denoise driven by the target prompt with attention taps installed on a
     depth-spanning set of double-stream blocks. Records, per step × block, image→text
     attention mass / max / value-norm / contribution for every text token; per-token spatial
     attention maps for the role tokens at sampled steps; and per-token causal Δ_edit
     contribution via single-token value ablation at sampled steps.

  2. INTERVENTION   — for the first few examples, sweep the four internal mechanisms
     (embed_scale / attn_logit_bias / attn_prob_reweight / value_scale) × weights × token
     roles, regenerate the edit, and record the resulting Δ_edit trajectory + spectra. Images
     are scored later by token_evaluate.py (anaconda env).

Writes:
  diagnostics/token_obs_<eid>.json         per-token observational stats over steps/blocks
  diagnostics/token_spatial_<eid>.npz      per-token spatial attention maps (role tokens)
  diagnostics/token_align_<eid>.json       tokenization + alignment + role assignment
  token_autopsy/token_generation.jsonl     one row per generated intervention image
  token_autopsy/token_intervention_grids/  the intervention images
Run:  $UVPY lib/token_autopsy.py            |   $UVPY lib/token_autopsy.py --limit 1  (smoke)
"""
from __future__ import annotations
import argparse, json, math, time, traceback
from pathlib import Path
import numpy as np
import torch

import config as C
import data as D
import flux_engine as FE
import spectral as S
import token_attn as TA


def _delta_step(pipe, latents, t, st, emb_src, emb_tgt):
    """v_src, v_edit, Δ_edit spatial at one x_t (Δ_edit recording matches the cache probe)."""
    vsrc = FE._fwd(pipe, latents, t, st, emb_src)
    vedit = FE._fwd(pipe, latents, t, st, emb_tgt)
    sp_delta = FE._spatial(pipe, vedit - vsrc, st["size"])
    return vsrc, vedit, sp_delta


def _spectral_summary(sp_deltas):
    """Per-trajectory Δ_edit spectral summary: mean band energy + temporal smoothness
    (mean absolute adjacent change of the low-pass projection — lower = smoother)."""
    n = len(sp_deltas)
    bands = np.array([S.band_energies(sp_deltas[i], C.LOW, C.HIGH) for i in range(n)])
    lp = [S.lowpass(sp_deltas[i], C.LOWPASS_FRAC) for i in range(n)]
    adj = [float(np.linalg.norm((lp[i] - lp[i - 1]).ravel())) for i in range(1, n)]
    norms = [float(np.linalg.norm(sp_deltas[i].ravel())) for i in range(n)]
    return dict(band_low=float(bands[:, 0].mean()), band_mid=float(bands[:, 1].mean()),
                band_high=float(bands[:, 2].mean()),
                delta_smoothness=float(np.mean(adj)) if adj else float("nan"),
                delta_norm=float(np.mean(norms)))


@torch.no_grad()
def observe(pipe, st, emb_src, emb_tgt, role_tokens, tap):
    """Pass 1. Returns (image, obs, spatial, spectral_summary). obs[block][step] -> stat dict;
    plus per-step per-role-token causal Δ_edit ablation contribution."""
    tap.store = {}; tap.spatial_store = {}          # reset per example (tap is reused)
    latents = st["latents"].clone()
    T = st["timesteps"]; n = len(T); size = st["size"]
    abl_steps = set(np.linspace(0, n - 1, min(C.TOK_ABLATION_STEPS, n)).astype(int).tolist())
    spat_steps = set(np.linspace(0, n - 1, min(C.TOK_SPATIAL_STEPS, n)).astype(int).tolist())
    sp_deltas = []
    ablation = {r: {} for r in role_tokens}            # role -> {step: per-band contrib dict}
    for i, t in enumerate(T):
        tap.step = i
        # base (source) forward: no recording — we want image->TARGET-token attention.
        tap.record = False; tap.spatial = False
        vsrc = FE._fwd(pipe, latents, t, st, emb_src)
        # target forward: taps record image->text attention for this step.
        tap.record = True
        tap.spatial = i in spat_steps
        tap.spatial_tokens = [tok["index"] for tok in role_tokens.values() if tok]
        vedit = FE._fwd(pipe, latents, t, st, emb_tgt)
        tap.record = False; tap.spatial = False
        sp_delta = FE._spatial(pipe, vedit - vsrc, size)
        sp_deltas.append(sp_delta)
        # per-token causal Δ_edit ablation (value-suppress one token, remeasure v_edit)
        if i in abl_steps:
            for r, tok in role_tokens.items():
                if not tok:
                    continue
                tap.intervene = dict(mech="value_scale", token=tok["index"], value=0.0)
                vedit_abl = FE._fwd(pipe, latents, t, st, emb_tgt)
                tap.intervene = None
                d = FE._spatial(pipe, vedit - vedit_abl, size)     # token's marginal effect
                lo, md, hi = S.band_energies(d, C.LOW, C.HIGH)
                ablation[r][i] = dict(norm=float(np.linalg.norm(d.ravel())),
                                      low=float(lo), mid=float(md), high=float(hi))
        latents = pipe.scheduler.step(vedit, t, latents, return_dict=False)[0]
    img = FE.decode(pipe, latents, size)
    # set spatial_hw retroactively from the first recorded block for the npz reshape (already
    # reshaped at record time if hw was known; here we just expose it). Infer from stored maps.
    spec = _spectral_summary(sp_deltas)
    return img, tap.store, tap.spatial_store, spec, ablation


@torch.no_grad()
def intervene_run(pipe, st, emb_src, emb_tgt, tap, mech, token_idx, weight):
    """Pass 2 single run. embed_scale modifies emb_tgt; the other three set tap.intervene for
    every step. Returns (image, spectral_summary)."""
    pe, ppe, tid = emb_tgt
    if mech == "embed_scale":
        pe2 = pe.clone()
        pe2[:, token_idx] = pe2[:, token_idx] * weight              # e_i <- alpha * e_i
        emb_tgt = (pe2, ppe, tid)
        interv = None
    elif mech == "attn_logit_bias":
        interv = dict(mech=mech, token=token_idx, value=C.LOGIT_BIAS_SCALE * math.log(weight))
    else:                                                            # attn_prob_reweight, value_scale
        interv = dict(mech=mech, token=token_idx, value=weight)
    latents = st["latents"].clone()
    T = st["timesteps"]; size = st["size"]
    sp_deltas = []
    for i, t in enumerate(T):
        tap.step = i; tap.intervene = interv
        vsrc, vedit, sp_delta = _delta_step(pipe, latents, t, st, emb_src, emb_tgt)
        tap.intervene = None
        sp_deltas.append(sp_delta)
        latents = pipe.scheduler.step(vedit, t, latents, return_dict=False)[0]
    img = FE.decode(pipe, latents, size)
    return img, _spectral_summary(sp_deltas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="limit #autopsy examples (smoke)")
    ap.add_argument("--steps", type=int, default=C.STEPS)
    ap.add_argument("--no-intervene", action="store_true", help="observational pass only")
    ap.add_argument("--smoke", action="store_true",
                    help="cheap pipeline check: 1 intervention example, 2 mechanisms, 3 weights")
    args = ap.parse_args()
    if args.smoke:                       # shrink the expensive intervention grid for smoke tests
        C.TOK_WEIGHTS = [0.5, 1.0, 2.0]
        C.TOK_MECHANISMS = ["embed_scale", "attn_prob_reweight"]
        C.TOK_INTERVENTION_EXAMPLES = 1
        C.TOK_ABLATION_STEPS = 3
        C.TOK_SPATIAL_STEPS = 3

    C.ensure_dirs()
    man = D.build()
    subset = man["pareto_subset"]
    examples = [e for e in man["examples"] if e["id"] in subset]
    if args.limit:
        examples = examples[:args.limit]

    pipe = FE.load_pipe()
    spatial_hw = None
    # one tap installed on the depth-spanning recorded blocks; intervene on the same set.
    tap, restore = TA.install_taps(pipe, C.TAP_BLOCKS, intervene_blocks=C.TAP_BLOCKS)
    print(f"[token] taps on double-stream blocks {C.TAP_BLOCKS}; {len(examples)} autopsy examples")

    gen_path = C.TOK / "token_generation.jsonl"
    rf = open(gen_path, "w")
    t_start = time.time()
    for ei, ex in enumerate(examples):
        eid = ex["id"]
        try:
            t0 = time.time()
            emb_src = FE.encode(pipe, ex["source_prompt"])
            emb_tgt = FE.encode(pipe, ex["target_prompt"])
            src_tok = TA.tokenize(pipe, ex["source_prompt"])
            tgt_tok = TA.tokenize(pipe, ex["target_prompt"])
            align = TA.align_tokens(src_tok, tgt_tok)
            roles = TA.assign_roles(tgt_tok, align)
            C.write_json(C.DIAG / f"token_align_{eid}.json",
                         dict(id=eid, task_type=ex["task_type"], category=ex["category"],
                              source_prompt=ex["source_prompt"], target_prompt=ex["target_prompt"],
                              source_tokens=src_tok["words"], target_tokens=tgt_tok["words"],
                              align=align, roles=roles))

            def fresh():
                return FE.prepare(pipe, ex["source_image"], steps=args.steps)

            # infer + set the image-token grid for spatial-map reshaping
            if spatial_hw is None:
                st0 = fresh()
                seq_img = st0["latents"].shape[1]
                hw = int(round(math.sqrt(seq_img)))
                spatial_hw = (hw, hw); tap.spatial_hw = spatial_hw

            img, obs, spat, spec, ablation = observe(pipe, fresh(), emb_src, emb_tgt, roles, tap)
            # serialize observational stats (block -> step -> {mass,maxq,valnorm,contrib}).
            # Recorded vectors span the padded txt_len (512); keep only the real tokens [:L]
            # so they line up with `tokens`/alignment indices in the analysis.
            L = tgt_tok["L"]
            obs_ser = {str(b): {str(s): {k: v[:L].tolist() for k, v in rec.items()}
                                for s, rec in steps.items()} for b, steps in obs.items()}
            C.write_json(C.DIAG / f"token_obs_{eid}.json",
                         dict(id=eid, task_type=ex["task_type"], category=ex["category"],
                              tap_blocks=C.TAP_BLOCKS, n_tokens=tgt_tok["L"],
                              tokens=tgt_tok["words"], roles=roles, spatial_hw=spatial_hw,
                              ablation=ablation, spectral=spec, obs=obs_ser))
            if spat:
                np.savez_compressed(
                    C.DIAG / f"token_spatial_{eid}.npz",
                    **{f"b{b}_s{s}_t{t}": m for (b, s, t), m in spat.items()})

            # ---- intervention sweep (first few examples get the full grid)
            if not args.no_intervene and ei < C.TOK_INTERVENTION_EXAMPLES:
                grid = C.TOK_GRIDS / eid
                grid.mkdir(parents=True, exist_ok=True)
                for role, tok in roles.items():
                    if not tok:
                        continue
                    for mech in C.TOK_MECHANISMS:
                        for w in C.TOK_WEIGHTS:
                            out, sp = intervene_run(pipe, fresh(), emb_src, emb_tgt,
                                                    tap, mech, tok["index"], w)
                            fp = grid / f"{role}__{mech}__w{w:.2f}.png"
                            out.save(fp)
                            rf.write(json.dumps(dict(
                                id=eid, task_type=ex["task_type"], category=ex["category"],
                                role=role, token_index=tok["index"], token_word=tok["word"],
                                mechanism=mech, weight=w,
                                source_image=ex["source_image_rel"],
                                source_prompt=ex["source_prompt"], target_prompt=ex["target_prompt"],
                                reference=C.rel(C.SAMP / eid / "full_compute_reference.png"),
                                image=C.rel(fp), **{f"delta_{k}": v for k, v in sp.items()})) + "\n")
                            rf.flush()
            print(f"[token] {ei+1}/{len(examples)} {eid} done in {time.time()-t0:.1f}s "
                  f"(elapsed {(time.time()-t_start)/60:.1f}m)")
        except Exception as e:
            print(f"[token] ERROR on {eid}: {e}"); traceback.print_exc()

    restore(); rf.close()
    print(f"[token] all done in {(time.time()-t_start)/60:.1f}m -> {gen_path}")


if __name__ == "__main__":
    main()
