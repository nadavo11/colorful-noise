# Handoff — E47 geodesic phasor-slerp phase editing on PIE-Bench (the lead we proceed with)

**Written:** 2026-06-23 (overnight). **Goal for the morning:** read two cluster verdicts and decide
whether geodesic-phase editing **beats SDEdit on a partial PIE-Bench set**, at a *fixed* operating
point (no test-time sweep). **Meeting in 2 days:** use the roadmap + this doc to pitch the idea.

> ⚠️ A **parallel session owns E47 right now** (worktree `e47-geodesic-slerp`, jobs `e47-confa`/
> `e47-confsdg`). It edits `e47_geodesic.py` / `cluster_e47_job.sh` **directly on `/storage`**.
> **Do not rsync stale copies over `/storage`** — I did this session and caused collisions (now
> cleaned up). If you need to change those files, edit the `e47-geodesic-slerp` worktree and let
> that session stage them, or coordinate first.

---

## 1. TL;DR state (honest)

The bare phase-transplant idea is a frontier-trap (as E46 predicted), **but the SDEdit+geodesic
combination is a real winner on the small set:**

| Method (n=20 PIE-Bench, SDXL, NFE=17) | What it is | Verdict |
|---|---|---|
| **B** `geo_t*`, `band_*` | full gen from a geodesic seed, **no x0-carry** | **NONE** beat vanilla → 5th frontier-trap confirmation |
| **A** `A_t0.25` | geodesic phase **noise injected into SDEdit** (x0-carry kept) | **WIN** struct 0.108 / clip **+0.065** |
| **sdg** `sdg_src_t0.25` | spectral-geodesic SDEdit (toward source) | **WIN** struct 0.113 / clip **+0.056** |

Both winners sit at the structure level of vanilla **strength≈0.7** (struct ≈0.11) but deliver
**~40–60% more editability** (+0.056/+0.065 vs vanilla's +0.040 there). Both win at **t/τ = 0.25**.

**Why this makes sense:** E46 proved *x0-carry is a strictly better structural anchor than any phase
transplant*. So the winner **keeps SDEdit's x0-carry and merely adds a small (t=0.25) geodesic phase
nudge** to the injected noise — it does not replace the anchor (method B does, and loses).

### Running now → results by morning
- `e47-confa`  → `results/e47_confA/verdict.md`   (method A, **n_per_type=10 ≈ 100 imgs**, τ∈{0.125,0.25,0.375})
- `e47-confsdg`→ `results/e47_confSDG/verdict.md` (method sdg, n=100, τ∈{0.125,0.25,0.375}, τ_white 0.5)

**Morning check:** `cat /storage/malnick/colorful-noise/experiments/results/e47_conf{A,SDG}/verdict.md`
Look at the `beats_vanilla_frontier` column. If `A_t0.25`/`sdg_src_t0.25` still say **YES at n=100**,
the win generalizes → proceed. If they flip to `no`, the n=20 win was small-sample noise → it's the
6th frontier-trap and we reconsider. (Verify the jobs finished: `runai training standard list -p
avidan | grep conf`; `export KUBECONFIG=/home/shimon/.kube/config_now_working` first.)

---

## 2. The method (for the writeup / meeting)

**Setup.** Source image → SDXL latent `x0`; its spectral phase `ph_src = angle(fft2(x0))`. A white
seed `z` gives white magnitude `|z|`. A white **target phase** `ξ = angle(fft2(gaussian))` (Hermitian).

**Geodesic phasor-slerp (the core op, `geodesic_seed`):**
```
delta = wrap(ξ − ph_src) ∈ (−π, π]      # shortest signed arc, per frequency
ph_t  = ph_src + t · delta               # rotate fraction t of the arc (CONSTANT angular velocity)
seed  = ifft( |z| · exp(i · ph_t) )      # keep WHITE amplitude (coloring it → rainbow artifacts, E46 P3)
```
- `t=0` → full source phase (max structure); `t=1` → white ξ (max editability). `t` is a continuous
  **structure↔editability knob**, scalar or per-frequency (band variant: keep low band, whiten highs).
- Self-conjugate FFT bins (DC/Nyquist) are restored from the white seed; otherwise intermediate `t`
  breaks Hermitian symmetry and leaks a ~9e-2 imaginary part (→ 2e-7 after restore).

**Geodesic vs linear (the key conceptual point).** E46 used a **chord**:
`unit((1−g)·e^{i·ph_z} + g·e^{i·ph_src})` — a *linear* interpolation of two phasors in the complex
plane, then renormalize. On the unit circle that is **lerp-then-project**: variable angular speed and
a **discontinuous flip** when the two phasors are near-antipodal. The **geodesic is slerp**: it moves
along the **shortest arc at constant angular velocity** — smooth, well-defined everywhere, and the
mathematically correct "fraction of the way from source phase to white." (Same lerp-vs-slerp
distinction as quaternion interpolation, applied per Fourier bin.)

**The winning recipe (method A).** Don't generate from the geodesic seed. Instead run **SDEdit**
(SDXL img2img, keeps the x0-carry structural anchor) and **replace its forward noise with the
geodesic seed at t=0.25** (mostly white + 25% source phase). Result: SDEdit's cheap structure anchor
+ a small phase prior → strictly more editability at matched structure. `sdg` is a variant that
applies the geodesic inside the SDEdit noising op toward the source phase.

---

## 3. How this differs from the two papers

- **Colorful-Noise** — *Training-Free Low-Frequency Noise Manipulation for Color-Based Conditional
  Image Generation* (arXiv 2605.00548). They manipulate the **low-frequency MAGNITUDE** of the noise
  to control global **color/structure** in **text-to-image generation**. We manipulate **PHASE**, not
  magnitude (we deliberately keep magnitude white — coloring it gives rainbow artifacts), and our task
  is **editing**, not color-conditioned generation. *(Note: this paper shares our repo's name — call
  out the distinction explicitly so the audience doesn't conflate them.)*
- **Φ-Noise** — *Training-Free Temporal Video Conditioning via Phase-Based Noise Manipulation*
  (arXiv 2605.24509). They inject **low-frequency PHASE from a reference VIDEO** into the noise for
  **motion-conditioned video generation**. Phase-based like us, but: (a) a **hard low-band injection**,
  not a **continuous constant-velocity geodesic** with a structure↔edit knob `t`; (b) **video-motion
  generation**, not **image editing**; (c) no combination with **SDEdit's x0-carry** (our winning
  ingredient). Φ-Noise is also the **closest prior to the E48 temporal-phasor idea** (see §6) — cite
  it there too.
- **Our novelty:** the **geodesic (slerp) phase interpolation as a controllable editing knob**,
  **combined with SDEdit's x0-carry**, giving **fast, inversion-free editing that beats vanilla SDEdit
  on the structure×editability frontier**.

---

## 4. The SDEdit "constant" question (you asked to pin this)

- The SDEdit hyperparameter is **`strength`** (fraction of forward noise / starting timestep).
  E46/E47 method-A use **strength = 0.8 fixed**; the vanilla *baseline* sweeps strength {0.5..0.9}
  only to **draw the frontier curve** — that sweep is a research tool, not a test-time knob.
- **"alignflow" = FlowAlign** (arXiv 2505.23145). Its PIE-Bench eval uses **33 NFE** (following
  FlowEdit) with **SDEdit + DDIB** as the noisy-trajectory baselines. **The exact SDEdit *strength*
  is NOT stated in the FlowAlign paper body** → **OPEN ITEM:** confirm it from **FlowEdit's released
  code** (`github.com/fallenshock/FlowEdit`), since FlowAlign follows FlowEdit's setup.
- **Subtlety that matters:** FlowAlign's SDEdit runs on **SD3 (flow)**; our E47 runs on **SDXL img2img
  (strength 0–1)**. The literal "same number" does **not** transfer across model families. Match the
  **budget (NFE)** and the **philosophy (one fixed operating point, no per-image sweep)**, not the raw
  value. We already match the budget: **NFE=17** (FlowAlign's plain-sampling budget on 24 GB cards).
- **Recommended honest headline (no test-time sweep):** report **our method at fixed t=0.25,
  strength=0.8** vs **vanilla SDEdit at the matched-structure strength (≈0.7)**. Our `{0.5–0.9}` sweep
  already brackets whatever FlowEdit's constant turns out to be, so the matched-structure comparison is
  robust to the open item.

---

## 5. Improvement direction (the pitch)

- **Problem:** inversion-based editors and FlowEdit/FlowAlign are **slow** — inversion = 17+17 NFE;
  FlowEdit/FlowAlign = 33 NFE (>2 model calls/step). **SDEdit is cheap (17 NFE sampling) but weak.**
- **Our bet:** **fast AND good** editing — SDEdit's speed with a **free geodesic-phase prior** that
  lifts it above its own structure×editability frontier. No inversion, no extra NFE.
- **Next direction to explore:** put **inversion methods on top of this**. Many inversion editors use
  **SDEdit-style partial denoising** (they don't traverse the full generation path) when they
  regenerate — our geodesic-phase SDEdit could slot in as their **fast structured base**, potentially
  cutting their NFE while keeping fidelity.

---

## 6. Sibling direction — E48 temporal phasor (video), already KEEP

Separate thread, committed on worktree `e48-temporal-phasor` (`c36b08f`). Probe 0 (pure-math temporal
Fourier-shift sanity on LTX latents) = **KEEP**: operator correct (FFT identities ~1e-6); LTX latent
temporal axis is **shift-equivariant in the interior (37.7 dB)** → temporal phase is a faithful motion
carrier. Reframed (extrapolation dropped — a linear phasor is a *circular* shift): deliverable =
**temporal-only phase preservation for video-edit consistency** vs E45 `phase3d`/vanilla on a
flicker×editability frontier. Files: `experiments/e48_temporal_phasor.py`, `experiments/e48_log.md`,
`results/e48/*.mp4`. **Φ-Noise (§3) is the closest prior here too.**

---

## 7. Roadmap registration (E47 is NOT yet in the registry)

`roadmap_registry.py` ends at E46. **Don't race the parallel session's registry edits** — if it
hasn't added E47 by the time you finalize, paste this (pick `thread="seed"`, set status once the conf
verdict lands: `active` if A wins at n=100, else `dead-end`):

```python
{"id": "E47", "title": "Geodesic phasor-slerp phase editing -- SDEdit+geodesic vs vanilla SDEdit (SDXL, PIE-Bench)",
 "thread": "seed", "models": "SDXL", "status": "active",
 "motivation": "Inversion editors pay 17+17 NFE; FlowEdit/FlowAlign 33 NFE; SDEdit is cheap (17) but weak. "
               "E46 KILLed bare seed-phase transplant (frontier-trap). Test the clean constant-angular-velocity "
               "GEODESIC slerp of source->white phase as a structure<->edit knob, and -- crucially -- COMBINE it "
               "with SDEdit's x0-carry instead of replacing it.",
 "method": "SDXL, NFE=17, PIE-Bench frontier (DINO-struct down x CLIP-dir up). geodesic_seed: ph_t=ph_src+t*delta, "
           "white amplitude, self-conj bins restored. Arms: vanilla SDEdit strength sweep {0.5..0.9} (draws the "
           "frontier); B=full gen from geodesic seed (no x0-carry); A=geodesic phase NOISE in SDEdit@0.8 (x0-carry "
           "kept); sdg=spectral-geodesic SDEdit. Matched-structure (no test-time sweep) comparison.",
 "result": "n=20: B traces a frontier inside vanilla (NONE win, 5th frontier-trap confirmation). A_t0.25 (struct "
           "0.108, clip +0.065) and sdg_src_t0.25 (0.113, +0.056) BEAT the vanilla frontier -- ~40-60% more "
           "editability at the structure of vanilla strength~0.7. Confirmation at n=100 (confA/confSDG) [PENDING].",
 "verdict": "[fill after n=100]: KEEP if A/sdg t=0.25 beat the frontier at n=100; else 6th frontier-trap KILL.",
 "nxt": "Confirm FlowEdit's SDEdit strength constant; scale to full PIE-Bench; then inversion-on-top-of-fast-SDEdit.",
 "script": "experiments/e47_geodesic.py", "doc": "docs/experiment-reports/EXPERIMENT_47.md",
 "results": None, "image": None},
```
Then `python experiments/make_roadmap.py && python experiments/make_roadmap.py --check` (exit 0), commit.

---

## 8. Infra quick ref
- Cluster: `export KUBECONFIG=/home/shimon/.kube/config_now_working`; `~/.runai/bin/runai training
  standard {list,logs,describe} <name> -p avidan`. Submit pattern: `... submit <fresh-name> -p avidan
  -g 1 --large-shm -i pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime --existing-pvc
  claimname=storage,path=/storage --command -- bash /storage/.../experiments/<job>.sh <stage>`.
- `/storage` is SSHFS (df shows 100% but writes fine; flaky listing — retry). **It is the live shared
  copy a parallel session edits directly — treat it as not-yours; don't push stale files over it.**
- E47 stages in `cluster_e47_job.sh`: `--sub20/--sub100` (method B), `--sweepA/--sweepSDG` (n=20),
  `--confA/--confSDG` (n=100). My `cluster_e47_confirm.sh` is a leftover duplicate on `/storage` —
  harmless, ignore or delete.
- PIE-Bench: `piebench.py` loads HF `UB-CVML-Group/PIE_Bench_pp` (cache `/storage/malnick/datasets/
  pie_bench_hf`), `n_per_type` stride-sampled across the 10 edit-type categories.

## 9. Working style reminder
One probe at a time; restate hypothesis + single change before running; KEEP/KILL/PARK per probe in a
durable log; commit per probe; **be honest about negative results** (method B is a clean KILL — say so;
the value is the A/sdg combination, contingent on the n=100 confirmation).
