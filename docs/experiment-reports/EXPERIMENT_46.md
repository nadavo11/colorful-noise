# E46 — Seed-phase fast editing: can a 0-NFE phase prior beat SDEdit? (SDXL)

**Thread:** seed · **Model:** SDXL · **Status:** dead-end (editing) / mechanism KEPT
**Worktree:** `e46-seedphase` (scripts under `experiments/e46_*.py`)

---

## Motivation

Inversion-based editors pay many NFE walking back to the seed; training-free editors
(FlowEdit/FlowAlign) skip inversion but cost >2 NFE/step. SDEdit is cheap but unreliable.
Idea: build a **structural prior for ~0 NFE** by transplanting the source image's **FFT phase**
(where the colored-noise line has repeatedly found structure lives) into a fresh seed, then run a
single fast generation toward the target prompt. If phase carries structure and a fresh
magnitude carries editability, SDEdit could be made reliable for free.

## Derivation (the theory, settled before/through the probes)

1. **Averaging noised copies recovers `phase(x₀)` exactly.** With `x_t = a_t·x₀ + b_t·ε`, the FFT
   is linear: `X_t = a_t·X₀ + b_t·E`. Over draws of ε, `E[X_t] = a_t·X₀`, whose phase **is**
   `phase(x₀)` (phase is scale-invariant). So "noise n times and average" is a *noisier* estimator
   of a quantity available for free — the loop is **dominated**. (Circular-mean-of-angles is the
   one exception: a biased estimator that band-limits phase by SNR — i.e. the low-band cut below.)

2. **The 100% (full-noise) pass is information-theoretically empty.** `q(x_T)=N(0,I)` is independent
   of `x₀`; forward-noising cannot extract an "image-fitting" seed (that *is* inversion). Any usable
   trace lives at partial `t` (the low-band phase that survives at strength ~0.8).

3. **Whitening the phase == destroying the structure.** Structure is the **cross-frequency phase
   coherence** (edges = many frequencies whose phases align spatially). A statistically-white phase
   field is i.i.d.-uniform = no coherence = no structure. So there is *no* transform that whitens
   the phase while keeping structure; they are the same axis (proved empirically by Cnorm below).

4. **The OOD-ness is purely the phase, not the spectrum.** A phase-transplant seed already has white
   *magnitude*; literal whitening (flattening the PSD) is a no-op here. The off-manifold part is a
   higher-order (phase) statistic that 2nd-order whitening can't reach — confirmed by Approach 3.

## Method & probes

All on SDXL (the colored-noise home model). Source latent `x₀ = VAE.encode(src)`; fresh white seed
`z`. Structure scored by **DINO self-similarity distance** (↓ = layout preserved), editability by
**CLIP-directional similarity** (↑). PIE-Bench was unavailable locally (no `datasets`/empty cache),
so the editing probes use **SDXL-generated sources** spanning PIE edit families; official PIE-Bench
is deferred to the cluster.

- **P0 — mechanism (reconstruction).** `phaseB = phase_swap_2d(x₀, z, cut=0.2, mag_from="B")` (source
  low-band phase, white magnitude), full gen with the **source** prompt vs a plain white seed.
  → **phaseB beats white on 12/12 seed pairs**; grid shows exact pose/arrangement transfer.
  **Seed low-band phase controls output layout. KEEP.**
- **P1 — editing frontier.** Recipes vs vanilla SDEdit@0.8 on 8 sources × 2 seeds: `A` = SDEdit with
  phase-structured noise; `B` = structured-seed full gen. → A over-locks (editability collapses,
  edits fail); B is Pareto-dominated. **0/8 Pareto wins. KILL.**
- **P2 — full-band phase & phase-normalize (chair).** `Cfull` (all-frequency image phase) preserves
  structure *worse* than low-band (white amp + high-freq image phase = OOD fringing). `Cnorm`
  (`φ_src − φ_z`) `= e^{iφ_src}·conj(Z) ~` white Gaussian → **a random seed** (structure gone),
  empirically confirming derivation #3.
- **P3 — three OOD escapes + synthesis (chair).**
  - **(1) γ phase-whiten** (slerp seed phase white↔image): a clean smooth structure↔edit knob;
    **fringing grows with γ**; best struct (γ=0.5: 0.124) still > vanilla (0.093).
  - **(2) timestep injection** (white seed, project low-band phase during the structure-forming
    window): **clean / on-manifold, no fringing, structure 0.082 BEATS vanilla** — but the hard
    clamp drives editability negative.
  - **(3) colored amplitude** (image phase + 1/f amp): **rainbow artifacts** — confirms the seed
    amplitude must stay white (derivation #4).
  - **Synthesis — soft timestep injection** (γ-blended low-band phase during 0–50%): γ=0.3 gives the
    cleanest, best-structure point (0.081) but editability caps ~0, never reaching vanilla's +0.090.

## Key result

Every variant — seed-bake, γ, full-band, timestep, soft-timestep, colored-amplitude — traces a
**structure↔editability frontier that sits at-or-inside vanilla SDEdit's**. SDEdit's `x₀`-carry is a
strictly better and cheaper structural anchor than any phase transplant: where you keep `x₀`, the
phase prior is redundant; where you drop it, you give up more than you gain.

## Verdict

**KILL the seed-phase EDITING direction — the 4th independent confirmation of the E41 frontier-trap**
(a spectral/structural knob slides along the SDEdit/RF-inversion frontier, never pushes it outward).
The underlying **mechanism is real and kept** (P0: seed low-band phase controls layout; timestep
injection yields clean structure beating vanilla) — but it only has value where there is **no `x₀`
to carry**: layout-conditioned text-to-image, or cross-modal structure transfer.

## Next / open

- If revisited: matched-editability comparison (sweep vanilla strength to draw its full frontier) to
  put a final nail in, and/or the **no-`x₀` pivot** (seed-phase as a cheap layout prior for plain
  T2I, where SDEdit isn't a competitor).
- Confirm P1 numbers on official PIE-Bench (cluster, where `datasets` is installed).

## Artifacts

Scripts (worktree `e46-seedphase`): `experiments/e46_seedphase.py` (P0),
`experiments/e46_probe1.py` (P1), `experiments/e46_chair.py` (P2),
`experiments/e46_gamma.py` · `e46_inject.py` · `e46_coloramp.py` · `e46_softinject.py` (P3).
Probe log: `experiments/e46_log.md`. Grids/scores under `experiments/results/e46*`.
