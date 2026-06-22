# E45 — FlowAlign on LTX-Video + spectral phase op (temporal video editing)

Goal: show our low-band PHASE-keep op (E41/E43) improves *temporal coherence* of FlowAlign
video edits over the paper's frame-by-frame approach. One-clip feasibility probe (KEEP/KILL).
Metric bundle: DINO struct-dist + CLIP-directional (per-frame avg) + RAFT warp-error
(global & edited-region-masked). Goal = a phase variant beats baseline on struct + masked-warp
while holding CLIP within 0.01. FlowAlign hyperparams: w=10, zeta=0.01, 24 steps.

## Probe S0 (e45-ltx-smoke) — KEEP
LTX loads; VAE encode->decode round-trip L1 = 0.021 (gate <0.08). Latent (1,128,F,H,W),
F=(frames-1)//8+1, H=W=size//32; latents_mean/std (128,); scaling_factor=1.0. Plumbing shapes
confirmed.

## Probe S1 (e45-ltx-s1) — KEEP
FlowAlign-on-LTX port (velocity/pack/sigma, 3 forwards/step). Identity gate (C_tar=C_src)
recon L1 = 0.0052 -> port is correct.

## Probe S2/S3 (e45-ltx-s2) — PARK (temporal), KEEP (structure/editability)
49 frames, 256px (latent 8x8x7). Identity recon L1 = 0.0041.

| cond          | struct↓ | clip↑   | warpG   | warpM   |
|---------------|---------|---------|---------|---------|
| baseline      | 0.1512  | +0.0292 | 0.00097 | 0.00095 |
| phase2d_c0.2  | 0.1075  | +0.0503 | 0.00122 | 0.00122 |
| phase3d_c0.2  | 0.1064  | +0.0346 | 0.00130 | 0.00132 |
| phase2d_c0.35 | 0.1068  | +0.0176 | 0.00128 | 0.00127 |
| phase3d_c0.35 | 0.1038  | +0.0205 | 0.00122 | 0.00120 |

- Structure+editability: phase op WINS big (struct 0.104-0.108 vs 0.151; phase2d_c0.2 clip
  +0.050 vs +0.029). Reproduces the E41/E43 image result on video.
- Temporal (the goal): INCONCLUSIVE. Baseline warp already ~9.5e-4 and masked≈global -> the
  baseline does not flicker on this clip, so there is no headroom to demonstrate a 3D-phase
  temporal win. Phase ops nudged warp up slightly (~1.2e-3) but all are essentially flicker-free.
- Root cause: at 256px the latent is 8x8 spatial -> too little high-frequency detail for flicker
  to manifest or for the spectral op to have bands to work on.
- Verdict: PARK temporal claim. Next single change -> raise resolution to 512 (16x16 latent) so
  both the flicker and the spectral op have real substrate. Everything else fixed.

## Probe S4 (e45-ltx-s3) — pending
ONE change vs S2: size 256 -> 512. Same sweep/metrics/goal.
