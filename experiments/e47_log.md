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

## P0 — kill test on n_per_type=2 (~20 PIE-Bench imgs)
First read used steps=30 (pre-lock); the locked config is steps=17 for the scale-up and any
re-confirmation. The geodesic-vs-vanilla comparison is a relative frontier, so it transfers.
(pending cluster verdict — `--sub20`)
