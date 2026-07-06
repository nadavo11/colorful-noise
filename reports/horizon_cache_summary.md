# HorizonCache (E56) — summary

Status: **PARTIAL**  ·  git `7232adf`

## Runnable paths
sd3_generation=False, flux_generation=True, flowedit=True, flowalign=True

## Best generation (FLUX)
- method: HorizonCache-v0 adaptive_1.25 (τ=0.35)
- achieved speedup: 2.4626×
- PSNR: 28.2951 dB, LPIPS: 0.0704
- ΔPSNR vs SeaCache at **matched achieved speedup** (deck fair rule): +2.36 dB (edges the SeaCache frontier in the 1.7–2.7× sweet spot; overshoots past 2.1×)

## Verdicts
- horizon_cache_v0: **KEEP**
- horizon_cache_v1: **PARK**
- jump_1p25: **KEEP**
- jump_1p5: **PARK**
- jump_2p0: **KILL**
- editing_branch_horizon: **PARK**

## Key findings
- At matched ACHIEVED speedup, the best HorizonCache variant (adaptive_1.25) beats SeaCache by +2.36 dB PSNR at ~2.5× (per-image matched win-rate 100%); jumps convert SeaCache's spare staleness headroom into a bigger step. N=20 images (powered — decision rule applied).
- The surviving primitive is the CONSERVATIVE regrid jump (jump_1.25 / adaptive→1.25). This edges the deck's FLUX prediction of a pure tie: the headroom-driven small stride finds a narrow safe pocket FLUX's bending field still permits.
- The win lives only in a NARROW speed band (~1.7–2.7×) and collapses beyond it — aggressive jumps overshoot. Honest statement: HorizonCache-v0 edges the SeaCache frontier in a conservative-jump regime, not everywhere.
- Reduces to SeaCache exactly when jumps disabled (fair by identity); jump_2.0 = KILL on FLUX.
- The v1 SAFE-HORIZON LABEL is the crux — and it is two-dimensional (metric AND horizon). Strict latent-L2 (2%) labels 0 safe jumps (too pessimistic); short-H=5 decoded-PSNR labels 12 safe jumps but 11/12 are jump_2.0 ({'cache': 24, 'jump_2.0': 11, 'jump_1.25': 1}) — which we KNOW kills end-to-end quality. The short continuation is too OPTIMISTIC for big jumps (it misses compounding — the deck's DP lesson). Neither is right: the correct target is a FRONTIER-IMPROVEMENT label (does this action beat SeaCache end-to-end at matched budget), or a full-trajectory continuation.

## Failure modes
- The win is confined to ~1.7–2.7×; past ~2.1× the longer live stride's Euler truncation dominates and HorizonCache drops below the SeaCache frontier (regrid_1.5 / adaptive→1.5 overshoot first).
- FLUX field bends sooner than SD3 (deck): the safe-jump pocket is small, so only jf≈1.25 survives.
- Absolute accumulated-score jump gates never fire (raw relL1 ~0.1-0.25/step); headroom = 1-acc/τ is the right signal.
- v1 label (decoded_psnr) is mis-specified: latent-L2 gives 0 jump positives, short-H decoded-PSNR over-credits jump_2.0 (compounding blindness). v1 stays PARK until the label is a full-horizon / frontier-improvement target and the rollout dataset is scaled (currently n<=36 states).

## Artifacts
- html_report: reports/horizon_cache.html
- summary_md: reports/horizon_cache_summary.md
- metrics_csv: ../results/horizon_cache/gen_20260707_013626/metrics.csv
- action_dataset: ../metrics/horizon_cache_decoded/action_dataset.csv
- figures_dir: /home/nada/PycharmProjects/colorful-noise/reports/horizon_cache_assets
- samples_dir: ../results/horizon_cache/gen_20260707_013626/samples