# HorizonCache (E56) — summary

Status: **PARTIAL**  ·  git `6b16b7f`

## Runnable paths
sd3_generation=False, flux_generation=True, flowedit=True, flowalign=True

## Best generation (FLUX)
- method: HorizonCache-v0 adaptive_1.25 (τ=0.4)
- achieved speedup: 2.5054×
- PSNR: 27.8551 dB, LPIPS: 0.0785
- ΔPSNR vs SeaCache at **matched achieved speedup** (deck fair rule): +2.11 dB (edges the SeaCache frontier in the 1.7–2.7× sweet spot; overshoots past 2.1×)

## Verdicts
- horizon_cache_v0: **STRONG KEEP**
- horizon_cache_v1: **PARK**
- jump_1p25: **KEEP**
- jump_1p5: **PARK**
- jump_2p0: **KILL**
- editing_branch_horizon: **PARK**

## Key findings
- At matched ACHIEVED speedup, the best HorizonCache variant (adaptive_1.25) beats SeaCache by +2.11 dB PSNR at ~2.5× (per-image matched win-rate 92%); jumps convert SeaCache's spare staleness headroom into a bigger step. N=48 images (powered — decision rule applied).
- The surviving primitive is the CONSERVATIVE regrid jump (jump_1.25 / adaptive→1.25). This edges the deck's FLUX prediction of a pure tie: the headroom-driven small stride finds a narrow safe pocket FLUX's bending field still permits.
- The win lives only in a NARROW speed band (~1.7–2.7×) and collapses beyond it — aggressive jumps overshoot. Honest statement: HorizonCache-v0 edges the SeaCache frontier in a conservative-jump regime, not everywhere.
- Reduces to SeaCache exactly when jumps disabled (fair by identity); jump_2.0 = KILL on FLUX.
- The v1 SAFE-HORIZON LABEL is the crux — and it is two-dimensional (metric AND horizon). Strict latent-L2 (2%) labels 0 safe jumps (too pessimistic); short-H=5 decoded-PSNR labels 12 safe jumps but 11/12 are jump_2.0 ({'cache': 24, 'jump_2.0': 11, 'jump_1.25': 1}) — which we KNOW kills end-to-end quality. The short continuation is too OPTIMISTIC for big jumps (it misses compounding — the deck's DP lesson). Neither is right: the correct target is a FRONTIER-IMPROVEMENT label (does this action beat SeaCache end-to-end at matched budget), or a full-trajectory continuation.
- FRONTIER-IMPROVEMENT label (the fix, built + run, 40 states): 70% of considered-jump states are jump-HELPFUL (adaptive jump preserves final quality vs cache under full rollout to the end) — a real, balanced positive class, unlike latent-L2 (0%) or short-decoded (jump_2.0-contaminated). But the learned binary v1 ties the majority baseline (0.7=0.7): on 40 states the adaptive heuristic is already near-optimal for this feature set (useful negative). Scaling the dataset is the concrete v1 next step.

## Failure modes
- The win is confined to ~1.7–2.7×; past ~2.1× the longer live stride's Euler truncation dominates and HorizonCache drops below the SeaCache frontier (regrid_1.5 / adaptive→1.5 overshoot first).
- FLUX field bends sooner than SD3 (deck): the safe-jump pocket is small, so only jf≈1.25 survives.
- Absolute accumulated-score jump gates never fire (raw relL1 ~0.1-0.25/step); headroom = 1-acc/τ is the right signal.
- v1 label (decoded_psnr) is mis-specified: latent-L2 gives 0 jump positives, short-H decoded-PSNR over-credits jump_2.0 (compounding blindness). v1 stays PARK until the label is a full-horizon / frontier-improvement target and the rollout dataset is scaled (currently n<=36 states).

## Artifacts
- html_report: reports/horizon_cache.html
- summary_md: reports/horizon_cache_summary.md
- metrics_csv: ../results/horizon_cache_sig/gen_20260707_095918/metrics.csv
- action_dataset: ../metrics/horizon_cache_decoded/action_dataset.csv
- figures_dir: /home/nada/PycharmProjects/colorful-noise/reports/horizon_cache_assets
- samples_dir: ../results/horizon_cache_sig/gen_20260707_095918/samples