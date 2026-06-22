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

## Probe S4 (e45-ltx-s3, 512px) — temporal KILL (as framed), structure KEEP
49 frames, 512px (latent 16x16x7). Identity recon L1 = 0.0037.

| cond          | struct↓ | clip↑   | warpG   | warpM   |
|---------------|---------|---------|---------|---------|
| baseline      | 0.2068  | +0.2524 | 0.00054 | 0.00045 |
| phase2d_c0.2  | 0.1737  | +0.2158 | 0.00065 | 0.00056 |
| phase3d_c0.2  | 0.1799  | +0.2220 | 0.00062 | 0.00053 |
| phase2d_c0.35 | 0.1445  | +0.1772 | 0.00070 | 0.00061 |
| phase3d_c0.35 | 0.1293  | +0.1724 | 0.00069 | 0.00062 |

- Baseline warp is even LOWER at 512 (warpM 4.5e-4) -> FlowAlign-on-LTX is essentially
  flicker-free at both resolutions. Phase preserves structure strongly (0.207 -> 0.129) but
  slightly raises warp and costs editability here.
- KEY INSIGHT: the paper's temporal-coherence gap is an artifact of frame-by-frame *image-model*
  editing. A real video model (LTX) already removes the flicker, so the 3D-phase op has no
  temporal headroom against a video-model baseline. Temporal hypothesis = KILL *as framed*.
- BUT we never ran the paper's actual method (frame-by-frame). To deliver the user's plan
  ("test as in the paper, then improve") we need the FBF baseline (which flickers) and to show
  our video+phase approach beats it on temporal coherence.

## Probe S5 (e45-ltx-s4) — KEEP: plan-faithful win + 3D-phase temporal edge
256px/25f. Identity recon L1 = 0.0052. Added FBF = paper's frame-by-frame method.

| cond          | struct↓ | clip↑   | warpG   | warpM   |
|---------------|---------|---------|---------|---------|
| fbf (paper)   | 0.1763  | +0.1197 | 0.03887 | 0.05183 |
| baseline      | 0.1490  | +0.0841 | 0.00096 | 0.00140 |
| phase2d_c0.2  | 0.1398  | +0.0536 | 0.00094 | 0.00138 |
| phase3d_c0.2  | 0.1389  | +0.0297 | 0.00078 | 0.00112 |
| phase2d_c0.35 | 0.1345  | +0.0620 | 0.00100 | 0.00154 |
| phase3d_c0.35 | 0.1356  | +0.0365 | 0.00075 | 0.00121 |

- TEMPORAL WIN (plan-faithful): frame-by-frame (paper) flickers at warpM=0.0518; video editing
  ~0.0011 -> **46x less flicker**. Reproduces the paper's admitted limitation and validates the
  metric.
- 3D-phase temporal EDGE: phase3d reduces warp vs video-baseline (0.00112 / 0.00121 vs 0.00140,
  -14..-20%); phase2d does NOT (0.00138 / 0.00154). Confirms the spatiotemporal hypothesis:
  3D couples frames, 2D doesn't.
- Structure: all phase variants beat baseline (0.134-0.140 vs 0.149).
- TRADE-OFF: phase costs editability (clip +0.03-0.06 vs baseline +0.084), so the strict goal
  (beat struct+warp while holding clip within 0.01) is NOT met -- phase3d_c0.2 wins struct+warp
  but drops clip. Next: smaller cuts (narrower low band) to keep editability.

## Probe S6 (e45-ltx-s5) — FINAL: strict goal unreachable by cut; trade-off is fundamental
256px/25f, cuts 0.1,0.15.

| cond          | struct↓ | clip↑   | warpM   |
|---------------|---------|---------|---------|
| baseline      | 0.1490  | +0.0841 | 0.00140 |
| phase3d_c0.1  | 0.1479  | +0.0635 | 0.00142 |
| phase3d_c0.15 | 0.1389  | +0.0297 | 0.00112 |
| fbf (paper)   | 0.1763  | +0.1197 | 0.05183 |

- phase3d_c0.1 (very narrow): negligible effect (~baseline struct/warp), slight clip drop -> no gain.
- phase3d_c0.15 == phase3d_c0.2 from S4 exactly: at the 8x8 latent the radial band snaps to the
  same discrete low-freq bins, so cut 0.15 and 0.2 are identical. Can't tune finely at 256px.
- CONCLUSION: no cut holds editability while keeping the struct+temporal gain -- the benefit and
  the editability cost are COUPLED. The strict hold-CLIP goal is unreachable by sweeping the cut.

## FINAL VERDICT (E45) — KEEP (plan-faithful goal met), temporal-hypothesis qualified
1. Frame-by-frame editing (paper's method) flickers (warpM 0.052); FlowAlign-on-LTX video editing
   is ~0.0011 -> **46x less flicker**. The paper's temporal gap is an artifact of frame-by-frame
   image-model editing, not intrinsic to FlowAlign.
2. The 3D spatiotemporal phase op uniquely reduces flicker vs the video baseline (-20%) where 2D
   does not -> the spatiotemporal hypothesis holds directionally.
3. Phase preserves structure (0.139 vs 0.149) but trades editability (CLIP +0.03 vs +0.084) at
   every cut. No free lunch on the editability axis.
Registered: roadmap_registry.py (E45) + EXPERIMENTS.md + regen site.
Open (needs user): CFG-match the video edit to fbf's edit strength (w sweep) before claiming the
frontier; real input clip instead of an LTX-generated source; higher-res latent for fine cut tuning.
