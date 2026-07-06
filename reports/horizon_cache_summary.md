# HorizonCache (E56) — summary

Status: **PARTIAL**  ·  git `7713f55`

## Runnable paths
sd3_generation=False, flux_generation=True, flowedit=True, flowalign=True

## Best generation (FLUX)
- method: HorizonCache-v0 (regrid, τ=0.3)
- achieved speedup: 2.1223×
- PSNR: 28.5739 dB, LPIPS: 0.1074
- ΔPSNR vs SeaCache at **matched achieved speedup** (deck fair rule): +0.51 dB (edges the SeaCache frontier in the 1.5–2.1× sweet spot; overshoots past 2.1×)

## Verdicts
- horizon_cache_v0: **PARK**
- horizon_cache_v1: **PARK**
- jump_1p25: **KEEP**
- jump_1p5: **PARK**
- jump_2p0: **KILL**
- editing_branch_horizon: **PARK**

## Key findings
- At matched ACHIEVED speedup, HorizonCache-v0 beats SeaCache by +0.51 dB PSNR in the sweet spot (~2.1×); jumps convert SeaCache's spare staleness headroom into a bigger step. (N=4 — promising smoke, not a headline.)
- Regrid jumps FIRE on FLUX (conservative: jump_1.25 dominates). This slightly BEATS the deck's FLUX prediction of a pure tie — the headroom-driven regrid finds small safe strides.
- The win is on the frontier (matched speedup); at the SAME τ HorizonCache trades ~0.2 dB for +0.25× speed — reads as sub-visible in pixels.
- jump_2.0 overshoots past ~2.1× (KILL); reduces to SeaCache exactly when jumps disabled (fair by identity).
- Rollout safe-horizon labels (24 states) found 0 states where a jump stayed under the strict 2% latent-L2 tolerance ({'cache': 20, 'fresh': 4}). On FLUX jumps are 'barely safe' in latent-L2 even where decoded PSNR ties — so v1 degenerates to cache/fresh; the jump win lives in decoded-PSNR, not latent-L2 tolerance. Relax tolerance / label on decoded PSNR next.

## Failure modes
- Past ~2.1× the longer live stride's Euler truncation error dominates: HorizonCache falls -0.97 dB below the SeaCache frontier at 2.5×.
- FLUX field bends sooner than SD3 (deck): headroom for a safe long stride is small, so jumps stay at 1.25.
- Absolute accumulated-score jump gates never fire (raw relL1 ~0.1-0.25/step); headroom = 1-acc/τ is the right signal.
- Rollout labeling at tol=2% latent-L2 yields NO safe-jump positives on FLUX -> v1 can't learn to jump (reinforces E53: jumps are borderline on FLUX). Needs a looser / decoded-PSNR damage metric or SD3.

## Artifacts
- html_report: reports/horizon_cache.html
- summary_md: reports/horizon_cache_summary.md
- metrics_csv: ../results/horizon_cache/gen_20260706_210216/metrics.csv
- action_dataset: ../metrics/horizon_cache/action_dataset.csv
- figures_dir: /home/nada/PycharmProjects/colorful-noise/reports/horizon_cache_assets
- samples_dir: ../results/horizon_cache/gen_20260706_210216/samples