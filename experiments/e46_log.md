# E46 — Seed-phase SDEdit (fast, inversion-free editing via low-band seed phase)

Idea: transplant the source image's **low-band FFT phase** into a fresh seed, then
generate toward the target prompt. Precompute is ~0 NFE (FFT only). Structure rides on
the seed's phase (Recipe B) or is added to SDEdit's injected noise (Recipe A).

Key theory settled: averaging n noised copies converges to phase(x_0) exactly (phase is
scale-invariant), so the "noise 5x and average" loop is dominated by just using phase(x_0).
The 100% pass is information-theoretically empty (q(x_T)=N(0,I) ⟂ x_0). Method collapses
to: low-band phase(x_0) -> fresh seed -> generate.

## Probe 0 — does low-band SEED phase control generated layout? (Recipe B, reconstruction)
SDXL, cut=0.2, 3 images x 4 seeds. arm white = N(0,I) seed; arm phaseB = white magnitude
+ source low-band phase. Metric: DINO structure distance to source (lower=closer layout).

Mean struct_dist (white -> phaseB):
  cat_orange     0.408 -> 0.171  (gap +0.237)
  black_panther  0.213 -> 0.161  (gap +0.052)
  savana         0.359 -> 0.245  (gap +0.115)
phaseB beats white on ALL 3 images and ALL 12 seed pairs, no reversals.

**Verdict: KEEP.** Low-band seed phase controls output layout, strongly & consistently.
Grid confirms exact layout transfer (cat pose, giraffe+rhino arrangement, panther stance).
Caveat: on flat/segmentation-like sources (cat, savana) phaseB shows OOD color artifacts
(red/cyan fringing) — white magnitude + strong cartoon phase is off-manifold. Natural photo
(panther) renders clean. -> Probe 1 must use real photos (PIE-Bench) and watch appearance.

## Probe 1 — editing frontier vs vanilla SDEdit (SDXL, 8 SDXL-gen sources, 2 seeds)
NOTE: local proxy — `datasets` not installed / PIE cache empty locally, so sources are
SDXL-generated across PIE edit families. Official PIE-Bench to be run on cluster.
cut=0.2, strength=0.8. vanilla=SDEdit; A=SDEdit w/ phase-structured noise; B=structured seed.

Per-arm means (struct_dist DOWN / clip_dir UP):
  vanilla  struct=0.129  clip_dir=0.222   <- best editability
  A        struct=0.104  clip_dir=0.057   <- best structure, editability COLLAPSED
  B        struct=0.151  clip_dir=0.183   <- worse on BOTH axes (dominated)

Pareto-beats vanilla (struct lower AND clip higher): A 0/8, B 0/8.
Grid: A often fails to edit (dog stays a dog); B edits but layout drifts + color artifacts.

**Verdict: KILL (both recipes, this operating point).** Seed-phase does not beat vanilla
SDEdit for editing: A double-anchors structure (x0 term + source phase) so the edit barely
fires (editability 0.057, several edits go negative); B drops the x0 term and is Pareto-
dominated (worse structure AND editability). Mechanism (P0) is real but redundant with /
weaker than SDEdit's own x0-carry. Mirrors E41 (spectral knob trades along the frontier,
doesn't beat it). PARK option: cut/strength frontier sweep for a matched-editability test
before final KILL, but the editability collapse + E41 precedent make a win unlikely.

## Probe 2 (chair debug) — full-band phase (Cfull) & phase-normalize (Cnorm) vs vanilla
One image (wooden->metal chair), 1 seed. struct DOWN / clip_dir UP:
  vanilla  0.093 / 0.090   (best structure, modest edit)
  Cfull    0.133 / 0.040   (full image phase + white amp: locks structure but OOD fringing, weak edit)
  Cnorm    0.190 / 0.190   (phi_src - phi_z: math -> fresh random seed; no structure, edit fires)

Findings:
- Cfull preserves structure WORSE than low-band, not better: full image phase on a flat (white)
  amplitude is off-distribution in the high band -> color fringing DINO reads as structure loss.
- Cnorm == random seed (subtracting an independent sample's phase re-randomizes: |z|e^{i(ph_src-ph_z)}
  = e^{i ph_src} conj(Z) ~ white Gaussian). Confirms no structure transfer.
- Seed-phase is a monotonic structure<->edit knob whose entire frontier sits INSIDE vanilla SDEdit's
  (vanilla beats Cfull on both axes). x0-carry is a strictly better structural anchor than phase transplant.

**Verdict: KILL the seed-phase EDITING direction (SDXL).** Third confirmation of the E41 pattern
(spectral knob trades along the frontier, never beats vanilla). Mechanism (P0) is real but only useful
where there is NO x0 to carry (pure layout-conditioned generation), not for image editing.

## Probe 3 (chair) — three OOD escapes + the synthesis (1 image, 1 seed)
vanilla = SDEdit @0.8 (struct 0.093 / clip +0.090). struct DOWN / clip_dir UP:

Approach 1 -- gamma phase-whiten (slerp seed phase white<->image, e46_gamma.py):
  g=0(white) 0.190/+0.105 | g0.25 0.179/+0.235 | g0.5 0.124/+0.062 | g0.75 0.133/+0.025 | g1(Cfull) 0.132/+0.037
  -> smooth structure<->edit knob; fringing GROWS with g; best struct (g0.5) still > vanilla.
Approach 2 -- timestep injection (white seed + low-band phase projected during window, e46_inject.py):
  inj0-30 0.082/-0.090 | inj0-50 0.082/-0.079 | inj0-70 0.084/-0.065
  -> CLEAN (on-manifold, NO fringing), structure BEATS vanilla (0.082<0.093), but hard clamp kills edit (clip<0).
Approach 3 -- colored amplitude (image phase + 1/f amp, e46_coloramp.py):
  beta0 0.133/+0.037 | beta-1(pink) 0.125/+0.052 | beta-2(red) 0.156/-0.002
  -> FAIL: coloring amp = rainbow artifacts. Seed amplitude MUST stay white; OOD is purely phase.
Synthesis -- SOFT timestep injection (Approach2 + gamma blend, e46_softinject.py, window 0-50%, cut0.2):
  g0.3 0.081/+0.007 | g0.6 0.082/-0.068 | g1.0 0.083/-0.063
  -> cleanest + best structure (0.081<0.093) but editability caps ~0; cannot reach vanilla +0.090.

**Final verdict: KILL editing direction (4th E41 confirmation).** Seed/phase injection in every form
(seed-bake, gamma, timestep, soft-timestep, colored-amp) traces a structure<->editability frontier that
sits at-or-inside vanilla SDEdit's. x0-carry is a strictly better, cheaper structural anchor. Key derived
facts: (a) averaging noised copies -> phase(x0) exactly, loop is dominated; (b) whitening the phase ==
destroying the structure (they are the same axis; Cnorm proved it); (c) the seed amplitude must be white --
the OOD-ness is the phase COHERENCE (a higher-order stat 2nd-order whitening can't touch); (d) timestep
injection keeps the seed on-manifold and is the only clean route, but over-clamps. Mechanism (P0: seed
low-band phase controls layout) is REAL and KEPT -- useful only where there is NO x0 to carry.
