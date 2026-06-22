# E47 — Geodesic phasor-slerp phase whitening (SDXL, PIE-Bench)

Follow-up to [E46](EXPERIMENT_46.md), which KILLed the seed-phase editing direction (4th E41
frontier-trap confirmation) using a **chord** phase mix. E47 tests whether the mathematically
clean **constant-angular-velocity geodesic** slerp changes the verdict — the user rejects E46's
KILL based on the chord-γ images.

Knob (replaces E46's chord):
    delta = wrap(xi - ph_src) in (-pi, pi];  ph_t = ph_src + t*delta;  seed = ifft(|z| e^{i ph_t})
t=0 -> full source phase, t=1 -> white xi. White amplitude kept (E46 P3: coloring amp -> rainbow).
Self-conjugate FFT bins restored from the white seed (intermediate t breaks Hermitian symmetry
otherwise — 9e-2 imag leak; this is the one real difference from the always-Hermitian chord).

**Goal (= kill criterion):** a geodesic config whose PIE-Bench subset-mean (DINO struct ↓,
CLIP-dir ↑) sits strictly NW of the vanilla SDEdit strength frontier. Loop one knob at a time
until a config beats vanilla (WIN) or the space is exhausted (final KILL = 5th confirmation).

Arms: vanilla SDEdit strength sweep {0.5..0.9} (draws the frontier) · geodesic-global t {0..1} ·
geodesic-band (keep low band [0,cut], whiten highs at t_high). Metric in `struct_metrics.py`.
Harness: `e47_geodesic.py`; cluster: `cluster_e47_job.sh`.

## Unit check (local, pre-cluster) — PASS
Geodesic helper verified: endpoints exact (t=0 -> source phase, t=1 -> xi), amplitude white,
band keeps low-band source phase; imag(ifft) leak 9e-2 -> 2e-7 after restoring self-conjugate
bins. Full run deferred to cluster (PIE-Bench + datasets live in /storage HF cache).

## Config (locked to FlowAlign, 24GB GPUs)
NFE=17 (FlowAlign's plain-sampling budget; inversion methods 17+17, FlowEdit/AlignFlow 33).
Our SDEdit + seed-phase arms and the vanilla baseline are sampling -> 17 steps. Cluster cards:
A5000 (24GB), matching FlowAlign's hardware. SDXL 1024px.

## P0 — variant B (full-gen geodesic seed) vs vanilla frontier, sub20 (cluster, 30 steps)
20 PIE-Bench imgs. Vanilla strength sweep {0.5..0.9} draws the frontier; geo/band = full-gen
from the geodesic seed (no x0). Vanilla frontier: s0.5 0.095/+0.032 .. s0.9 0.159/+0.158.
Every geodesic point sits SE of the curve (e.g. geo_t0.5 0.151/+0.071; band_th1.0 0.155/+0.101).
**WINNERS: NONE.** Verdict: **KILL variant B** — white-magnitude full-gen seed is dominated
(OOD high-freq + no x0 anchor). 5th E41 frontier-trap confirmation for the seed-phase direction.

## P1 — variant A (geodesic noise in SDEdit@0.8) vs vanilla, local 3 imgs (material/object/color)
A = SDEdit@0.8 with the geodesic seed injected as the forward noise (keeps x0). Means vs
vanilla@0.8 (0.127/+0.136): A t=0.5 0.098/+0.010, t=0.75 0.110/+0.016. **KILL as run** — A
double-anchors structure (the √ᾱ·x0 term AND source-phase noise), so structure improves a lot
but **editability collapses** (clip ~+0.01; the dog stays a dog). Confirms E46 Recipe A.

## P2 — spectral-geodesic SDEdit (chair): keep SDEdit energy, set phase by geodesic
Fix B's OOD: keep |x_std| (proper strength-0.8 energy), rebuild phase = phi0 + tau·wrap(xi-phi0).
Chair vs vanilla@0.8 0.107/+0.067: sg t=0.5 0.099/+0.041 (better struct), sg band 0.5/1.0
0.109/+0.099 (better edit, ~tie struct). **Hugs the frontier from both sides** — much closer
than A/B, motivating the apples-to-apples form.

## P3 — apples-to-apples SDEdit-geodesic (chair): tau=0 == vanilla, geodesic perturbs phase
`sdedit_geodesic`: rotate the 0.8 noised-latent's phase tau toward SOURCE (structure-restore)
or WHITE (edit-boost); tau=0 reproduces vanilla. Chair vs vanilla@0.8 0.107/+0.067:
- **SDG src tau=0.25: 0.082/+0.064** — structure +24% at matched edit → **candidate NW-of-frontier WIN.**
- SDG src tau=0.5 0.089/+0.031; tau=0.75 0.106/+0.033.  SDG white tau=0.5 0.130/+0.131 (edit ~2x).
- sanity SDG src tau=0 = 0.107/+0.069 ≡ vanilla (fp match) → construction validated.
Direct proof of energy/phase decoupling: at matched struct ~0.082, **A t=0.5 = +0.022 edit vs
SDG src t=0.25 = +0.064** (3x), because A adds x0 energy (over-locks) while SDG only sharpens
phase at fixed 0.8 energy. **KEEP SDG-source; KILL A.** (1 image — needs PIE-Bench replication.)

## P4 — apples-to-apples PIE-Bench sweep (cluster, 17 steps, in-flight)
Two parallel GPU jobs, n_per_type=2 (~20 imgs), each drawing the vanilla strength frontier +
its tau-sweep at strength 0.8: `e47-sweepa` (method A) and `e47-sweepsdg` (SDG-source + white).
Vanilla frontier: s0.5 0.088/+0.028 .. s0.7 0.109/+0.040 .. s0.8 0.127/+0.100 .. s0.9 0.145/+0.131.
**BOTH light-tau (0.25) arms beat the frontier (first E47 breakout):**
- A_t0.25 = 0.108/+0.065 -> at struct 0.108 vanilla gives only +0.040; **+0.025 margin (solid)**,
  fills the Pareto gap between van s0.7 and s0.8. A_t0.5/0.75 lose.
- sdg_src_t0.25 = 0.113/+0.056 -> beats interp +0.053 by +0.002 (**marginal, within noise**).
  sdg_src_t0.5/0.75 and sdg_wht_t0.5 lose.
Reversal vs the chair (there A over-locked, SDG won): over 20 real imgs A_t0.25 is the stronger.
**KEEP light-tau A (and SDG, tentatively); confirming at n=100.** Conf jobs: `--confA` / `--confSDG`
(taus 0.125/0.25/0.375). (n=100 verdict pending)
