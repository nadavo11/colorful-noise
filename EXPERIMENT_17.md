# E17 — SD3.5 port: SBN vs CFG-Zero* (true CFG), cross-domain + CompBench

**Status / Verdict:** CODE COMPLETE, RUN-PENDING. The SD3.5-medium backend
(`e17_sd35.py`) and both evaluation drivers (`e17_sd35_compare.py`, `e17_compbench.py`) are
written and reused by E18–E22, but **no `results/e17/` or `results/e17cb/` outputs exist on
disk yet** — no `report.json`/`scores.json` to cite. This doc describes the port and harness;
quantitative findings await a cluster run.

**The direction.** Everything spectral so far (SBN / band-norm, E9–E16) lived on **Flux**,
which has a *distilled* guidance embedding — its "cfg" is not real classifier-free guidance,
which made the high-CFG regime behave oddly (E16). To test the spectral methods on a clean
testbed we **port** them to **Stable Diffusion 3.5-medium**, which uses *true* two-pass CFG
and exposes **already-unpacked** latents — the simplest possible `ClampPSD3` path. E17 is that
port plus its evaluation harness: it pits SBN against **CFG-Zero\*** (a published high-CFG fix)
on fidelity (E17-compare) and on **T2I-CompBench** attribute binding (E17-CB), and asks
whether the two methods *complement*. This backend is the foundation the later experiments
(E18 style transfer, E20 warm-start, E21/E22 editing) all build on.

## Background (plain language)

- **SD3.5-medium.** A 2.5B rectified-flow text-to-image model (transformer + T5-XXL + 2×CLIP
  + 16-channel VAE), 1024px → a `(1, 16, 128, 128)` latent. Fits a 24GB GPU bf16
  GPU-resident; `--mem offload` is the fallback.
- **True CFG vs Flux's distilled guidance.** SD3.5's `guidance_scale` is *real* CFG:
  `do_cfg = guidance_scale > 1`, so `guidance_scale = 1` is the pure conditional flow field
  (the SBN reference and the realism anchor), analogous to Flux cfg=1 but **without** Flux's
  guidance distillation. SD3.5 does one **batched** `[uncond, cond]` transformer forward per
  step, then `chunk(2)` and combines `uncond + w·(cond − uncond)`.
- **Unpacked latents = the simplest ClampPSD3.** SD3.5's step-end callback hands you the
  latent already as `(B, 16, 128, 128)` — no Flux-style pack/unpack — so the spectral ops
  (`radial_psd`, `band_index_map`, `psd_match`) apply directly and the SBN clamp callback is
  much simpler than on Flux.
- **SBN (Spectral Band Normalization).** Clamp the generated latent's per-(channel, band) PSD
  to a **cfg=1 per-step reference**, phase untouched (`ClampPSD3`, mode `band`). `bandnorm_pp`
  adds an E11 saturation × postprocess.
- **CFG-Zero\* (baseline).** A published high-CFG fix: per-sample *optimal-scale* on the
  uncond term plus *zero-init* of the earliest step(s) (`make_cfgzero_step`). Adapted here to
  SD3.5's batched output (split, don't double-call).
- **Rectified-CFG++ (`cfgpp`).** A second guidance baseline (arXiv 2510.07631): a
  predictor/corrector that evaluates guidance at a predicted next point, with optional
  corrector jitter; also composes with SBN.
- **Metrics.** Fidelity = **aesthetic** (LAION predictor) + **ImageReward**; adherence
  guardrail = **CLIP-T**; on CompBench the contest is **B-VQA** (BLIP-VQA attribute binding on
  color/shape/texture). **VQAScore** (`clip-flant5-xxl`) is an extra adherence score in
  E17-compare.

## Method

- **Backend (`e17_sd35.py`).** `load_sd35` / `load_sd35_vae`; `sd3_vae_encode` /
  `sd3_vae_decode` (same `(z − shift)·sf` convention as Flux, so the E10 real-encode recipe
  transfers); `gen_sd3` (one entry point that composes an optional guidance `step_override`
  *and* an optional step-end callback); `record_reference_sd3` (the cfg=1 per-step PSD
  reference); the `RecordPSD3` / `ClampPSD3` callbacks; `make_cfgzero_step`; `gen_rcfgpp_sd3`.
  Also the E20 warm-start helpers (`RecordTraj`, `gen_sd3_warmstart`) live here.
- **E17-compare (`e17_sd35_compare.py`).** 8 conditions on shared seeded init latents per
  seed: `cfg1`, `cfg_hi` (default 4.5, the high-CFG baseline / `BASE`), `bandnorm`,
  `bandnorm_pp`, `cfgzero`, `cfgzero_sbn` (CFG-Zero\* **and** SBN — they compose: guidance
  modifies the velocity, SBN clamps the resulting latent at step end), `cfgpp`, `cfgpp_sbn`.
  Scored on aesthetic / ImageReward / CLIP-T / VQAScore; paired Δ vs `cfg_hi`. **Spectral
  distance** is attempted against an *SD3.5-VAE* real reference (`SD35_REAL_LATENTS`) — the
  E10 Flux real ref is the wrong latent space.
- **E17-CB (`e17_compbench.py`).** Same 8 conditions through **T2I-CompBench** on a balanced
  color/shape/texture subset; primary metric **B-VQA** binding, with aesthetic / ImageReward /
  spectral-dist / CLIP-T as fidelity context; per-category and overall means + paired Δ vs
  `cfg_hi`. This matches how CFG-Zero\* itself was reported, to test whether SBN's spectral
  clamp *preserves or harms* compositional binding, alone and combined.

## Findings

**Run-pending — no quantitative results to report.** `results/e17/` and `results/e17cb/` do
not exist on disk; there is no `scores.json` or `report.json` to cite, so per the
no-fabrication rule no numbers are quoted here. What is **established** (by code review /
diffusers 0.38 verification, recorded in the module docstring) rather than measured:

1. **The port is correct in principle.** SD3.5's `guidance_scale` is true CFG, `guidance=1`
   is the pure conditional field (a valid SBN reference / realism anchor), and the callback
   latents are unpacked `(B,16,128,128)` — so `ClampPSD3` applies the spectral ops directly,
   the cleanest SBN implementation in the repo.
2. **The conditions compose as intended.** A guidance modifier (CFG-Zero\* / CFG++) is a
   `step_override` on `scheduler.step`; SBN is a `callback_on_step_end` that runs *after* the
   Euler step — so `cfgzero_sbn` / `cfgpp_sbn` are genuine combinations, not re-implementations.

The open questions the run must answer: (a) does SBN beat CFG-Zero\* on fidelity in the
high-CFG regime; (b) do they complement (`cfgzero_sbn` > either alone); (c) does the SBN clamp
preserve or harm B-VQA attribute binding on T2I-CompBench.

## Caveats & next

- **The Flux real reference does not transfer.** Spectral-distance-to-real needs an
  **SD3.5-VAE-encoded** real set (`SD35_REAL_LATENTS`); the E10 Flux `real_latents.pt` is the
  wrong latent space and is correctly *not* reused. That SD3.5 real ref must be built (cf.
  `make_sd35_real_ref.py`) for the spectral metric to be meaningful.
- **SBN still pays a per-prompt cfg=1 reference cost.** Each prompt records its own cfg=1
  per-step PSD reference (extra `ref_seeds` generations); a *universal* reference is future
  work (and the E23 real-target line points toward it).
- **CFG-Zero\* / CFG++ are non-trivial monkey-patches** of `transformer.forward` /
  `scheduler.step`; they are restored in a `finally`, but are tied to diffusers internals and
  could break on version bumps.
- **Next:** run `gen,score,analyze` on the cluster for both drivers, build the SD3.5 real PSD
  reference, then fill in this doc's Findings with the measured tables. Downstream, E18–E22
  already depend on this backend.

## Reproduce

```bash
# E17-compare: SBN vs CFG-Zero* vs CFG++ on fidelity (SD3.5-medium, cfg=4.5)
python experiments/e17_sd35_compare.py --part gen,score,analyze \
    --seeds 25 --ref_seeds 3 --cfg 4.5 --steps 28 --n_bins 24
# skip the heavier baselines/scorers if VRAM/time constrained
python experiments/e17_sd35_compare.py --part gen,score,analyze --no_cfgpp --no_vqa

# E17-CB: same 8 conditions through T2I-CompBench (B-VQA binding)
python experiments/e17_compbench.py --part gen,score,analyze \
    --categories color shape texture --per_cat 64 --seeds 4 --ref_seeds 2 --cfg 4.5
```

Code: `experiments/e17_sd35.py` (SD3.5 backend: `load_sd35`, `gen_sd3`,
`record_reference_sd3`, `ClampPSD3`/`RecordPSD3`, `make_cfgzero_step`, `gen_rcfgpp_sd3`,
warm-start helpers), `experiments/e17_sd35_compare.py` (fidelity driver),
`experiments/e17_compbench.py` (T2I-CompBench driver), reusing `experiments/spectral_ops.py`,
`experiments/fidelity_metrics.py` (`SD35_REAL_LATENTS`), `experiments/compbench.py`
(`bvqa_scores`), and the E16 scoring/aggregation helpers. Outputs (when run):
`results/e17/{scores,report}.json` + `summary.md` + grids; `results/e17cb/` likewise.
