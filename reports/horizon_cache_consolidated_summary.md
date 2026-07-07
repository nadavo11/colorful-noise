# HorizonCache Consolidation (E56) — STRONG KEEP

**Commit** `fc37f45` · FLUX 512px/28 · N=100 paired · cluster H100

**Headline.** FLUX N=100: adaptive_1.25 @ 2.51× = +2.20 dB vs matched SeaCache (95% CI [+1.79, +2.62], win 85%).

**Bounded claim.** A conservative headroom-adaptive stride extension shifts the FLUX SeaCache frontier upward in the ~1.7–2.7× achieved-speedup band, while aggressive jumps still overshoot.

## Per-band significance (adaptive_1.25)

| band | speedup | ΔPSNR | 95% CI | win | verdict |
|---|---|---|---|---|---|
| 1.5-2.0× | 1.70× | +1.77 | [+1.39, +2.13] | 82% | significant gain |
| 2.0-2.5× | 2.12× | +1.28 | [+0.92, +1.66] | 78% | STRONG KEEP |
| 2.5-2.8× | 2.51× | +2.20 | [+1.79, +2.62] | 85% | STRONG KEEP |
| 3.0×+ | 3.40× | -0.16 | [-0.20, -0.12] | 17% | significant loss (overshoot) |

## Verdicts

- **adaptive_1.25**: STRONG KEEP — recommended conservative primitive — narrow safe-band frontier shift
- **adaptive_1.5**: STRONG KEEP — headroom-capped — ties 1.25 in-band (soft cap rarely reached)
- **adaptive_2.0**: STRONG KEEP — headroom-capped — in-band gain like 1.25/1.5; overshoots at 3.4× (≠ uncapped jump_2.0 KILL)
- **regrid_1.25**: STRONG KEEP — fixed-factor baseline the adaptive stride beats at matched speed
- **learned v1**: PARK — label correct, ties heuristic; needs scale
- **SD3 transfer**: PARK — unavailable path (no MMDiT harness)

## What failed

- Aggressive jumps (adaptive_2.0 / high-τ) overshoot the safe band — measured significant loss.
- SD3 replication unavailable: no SD3 generation harness in `experiments/horizon_cache/`.
- Learned v1 still ties the heuristic on the frontier-label set.
