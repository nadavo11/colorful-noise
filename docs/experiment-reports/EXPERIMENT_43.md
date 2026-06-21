# E43 — FlowAlign on FLUX + spectral terminal-point variants

## TL;DR

**FlowAlign** (arXiv:2505.23145) edits a real image without inversion: it's **FlowEdit** plus a
**source-consistency terminal-point** regularizer, with CFG run using the **source** prompt as the
negative. We ported it to FLUX and added two spectral twists, then searched for a knob setting
that preserves source **structure** better than plain FlowAlign **without** losing edit adherence.

**Winner: `sbn_phase`** — spectrally locking the *low-band phase* of the CFG velocity to the
source-conditioned reference. Across 3 scenes × `w ∈ {5,7,10}` (28 steps) it **beats plain
FlowAlign on every cell**: DINO structure distance roughly *halves* (e.g. 0.056 vs 0.124) while
CLIP-directional editability *rises* (mean ΔStruct −0.055…−0.061, ΔClip +0.037…+0.085). This is
the first editing lever in the project that preserves structure *more* than the baseline at *no*
editability cost.

## Method

Per step (demo notation, `x_src ≡ qt`, `x_tar ≡ pt = x_src + delta`):

```
qt  = (1−σ)·zsrc + σ·eps                      # forward-diffused source
pt  = xt + qt − zsrc                          # == x_tar
vp  = v(pt,c_src) + w·(v(pt,c_tgt) − v(pt,c_src))   # CFG, SOURCE prompt as the negative
vq  = v(qt,c_src)
xt += (σ_next−σ)·(vp − vq)  +  ζ·(qt−σ·vq − pt+σ·vp)   # ζ = 0.01 terminal-point term
```

The terminal term is `ζ·(E[q0|qt] − E[p0|pt])` = source-clean minus edit-clean point. The core
lives in `invert_core.flowalign` and is **shared verbatim** by the demo's FlowAlign tab and the
`e43_flowalign.py` harness, so they can't drift. Manual CFG (3 velocity forwards/step) keeps `w`
a real, SBN-able knob on the distilled model.

**Two spectral twists** (both reduce to plain FlowAlign at their defaults):

1. **SBN on the CFG reference** — after forming `vp`, clamp its low radial band `[0, cut]` toward
   the source-conditioned reference `v(pt,c_src)`, reusing E37's `velocity_spectral_ops`. Modes:
   `band power` (per-band power match), `mag` (per-bin magnitude transplant), **`phase`** (keep
   `vp` magnitude, lock the low-band phase to the reference), `both`. Idea: keep a high semantic
   `w` while spectrally damping its *structural* over-editing in low frequencies.
2. **Annealed terminal point** — band-limit the consistency vector before adding it, low-pass
   `start_cut → end_cut` over steps (coarse source-consistency early, fine detail later).

## Setup

FLUX.1-dev, 28 steps. Small qualitative sweep: 3 scenes (`cat→dog`, `house→thunderstorm`,
`street→snow`), `w ∈ {5,7,10}`, `cut=0.2`, `ζ=0.01`. Scored vs. the source with
`struct_metrics.py`: **DINO ViT-S/8 self-similarity structure distance** (↓ = structure
preserved), **CLIP-directional** similarity (↑ = edit adherence), LPIPS/DSSIM. Conditions:
`flowalign` (baseline), `sbn_bp`, `sbn_phase`, `term_anneal`, `sbn_bp+term`, plus a `recon`
identity gate (`C_tar==C_src`). Run on RunAI via `cluster_e43_job.sh`
(smoke → identity gate → w-sweep → automatic GOAL gate).

## Results

Identity gate holds: `recon` struct-dist ≈ **0.003–0.005** on every scene/config → the FLUX
port and VAE/packing path are correct.

Mean deltas vs. the FlowAlign baseline (negative ΔStruct and non-negative ΔClip = win):

| variant | w=5 | w=7 | w=10 | all-scene wins |
|---|---|---|---|---|
| **sbn_phase** | ΔStruct −0.061 / ΔClip +0.049 | −0.055 / +0.085 | −0.061 / +0.037 | **3/3 at every w** |
| sbn_bp | −0.027 / +0.059 | −0.020 / +0.059 | −0.028 / +0.043 | 2/3, 2/3, 3/3 |
| sbn_bp+term | −0.022 / +0.043 | −0.018 / +0.069 | −0.023 / +0.022 | 2/3 |
| term_anneal | +0.001 / −0.004 | +0.010 / +0.016 | +0.002 / −0.001 | null |

Example absolute numbers (`sbn_phase` vs baseline): `cat_dog` struct 0.056 vs 0.124, clip 0.566
vs 0.513; `house_storm` 0.040 vs 0.117, clip 0.350 vs 0.296; `street_snow` 0.066 vs 0.105, clip
0.432 vs 0.392 (all at w=5). The job's GOAL gate reports **PASS** with 4 winning config×variant
settings.

**Caveat — step count matters.** The 8-step smoke is misleading: there `sbn_phase` collapses
CLIP-directional to ~0.12 (the edit barely happens). The structure/editability win only emerges
at ≥28 steps.

## Verdict

**KEEP `sbn_phase` (mode=`phase`, cut≈0.2)** as the strong default variant of the FlowAlign tab.
It is the first lever here that preserves structure *more* than the baseline without losing
editability. `sbn_bp` is a secondary KEEP; the annealed terminal point is a KILL.

## Next

- Full 700-image **PIE-Bench** set + masks for the publishable comparison (enables BG-PSNR /
  BG-LPIPS, the masked background-preservation axis deferred here).
- SD3.5 port (true CFG).
- Sweep `sbn_cut` and phase strength to map the structure↔editability frontier.

## Artifacts

- `experiments/e43_flowalign.py` (`--part gen,analyze`), core in `invert_core.flowalign`, FlowAlign
  tab in `experiments/spectral_demo.py`, scoring in `experiments/struct_metrics.py`.
- `experiments/cluster_e43_job.sh` (self-gating: smoke → identity gate → w-sweep → GOAL gate).
- Outputs: `results/e43_w5|w7|w10/` — per-scene strips + `index.html` and `report.json`.
