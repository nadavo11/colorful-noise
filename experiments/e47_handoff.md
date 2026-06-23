# Handoff — E47: geodesic phase-perturbed SDEdit (fast, good editing)

**Date:** 2026-06-23 (~00:15) · **Repo:** colorful-noise · **Branch:** all E47 work on `origin/main`
(through `626ce30`) · **Worktree:** `.claude/worktrees/e47-geodesic-slerp`.

---

## TL;DR (read this first)

E47 found the **first method in the E41→E46→E47 line that beats vanilla SDEdit's frontier** on
PIE-Bench. The method: run SDEdit at a fixed strength, then perturb the noised latent's **phase**
by a small **geodesic** step (constant-angular-velocity slerp) — `τ=0` is exactly vanilla, a
light `τ=0.25` nudge restores structure without paying editability. Two variants both won on the
20-image subset; a **100-image confirmation is running now** (`e47-confa`, `e47-confsdg`).

This is the experiment **we proceed with.** The handoff below sets up the morning so we can answer
**"does this beat SDEdit on a partial PIE-Bench set at a *constant* (non-swept) hyperparameter?"**

---

## Morning goal (the one question to answer in 8h)

**Does geodesic-SDEdit beat plain SDEdit on a partial PIE-Bench set, at a single constant
hyperparameter (no test-time sweep)?** The paper-story requires a *fixed* operating point, not a
swept frontier — so:

1. **Pin the SDEdit hyperparameter to FlowAlign's eval config** (see next section). Use the SAME
   one for our baseline so the comparison is apples-to-apples and reviewer-defensible.
2. **Re-run head-to-head at that single constant** (no strength sweep): `vanilla SDEdit` vs
   `SDEdit + geodesic phase (τ=0.25)`, both at the fixed strength, on a PIE-Bench subset.
3. **Read off the win**: lower DINO structure-distance AND/or higher CLIP-directional at the same
   constant. Report the delta. That is the headline number for the meeting.

The n=100 confirmation finishing overnight (sweeps strength to draw the full frontier) tells us the
win is *real*; the constant-hyperparameter run tells us the win holds at the *fixed* point we'd ship.

---

## The SDEdit hyperparameter to match (FlowAlign / "AlignFlow")

From the official FlowAlign repo on the cluster (`/storage/malnick/flowalign_official`):

- **SDEdit editor** (`diffusion/editing/sd3_edit.py`, `@register_editor('sdedit')`): defaults
  **`n_start=10`, `cfg_scale=7.0`**. It noises `zsrc` at step `i = NFE − n_start` via
  `xt = (1−σ)·zsrc + σ·randn` and denoises the last `n_start` steps. (FlowEdit/AlignFlow-style:
  `n_start=33, cfg=13.5`.)
- **Global eval** (`README.md`, `run_edit.py`): **seed=123, NFE=33 (editing), CFG 13.5, SD3,
  1024px, RTX 3090.**
- **Strength mapping:** SDEdit "strength" ≈ `n_start / NFE`. With `n_start=10`, `NFE=33` →
  **strength ≈ 0.30** (a *light*, structure-preserving SDEdit), at **cfg≈7**. **⚠ CONFIRM** the
  exact `n_start`/`NFE` used for the SDEdit row in the FlowAlign paper's PIE-Bench table before
  locking — the class default is 10 but the eval may override it. Look in the paper appendix /
  any eval script under `flowalign_official`, or just adopt `n_start=10, NFE=33, cfg=7`.

**Important regime note:** our E47 winners so far are at **strength 0.8** (heavy edit). FlowAlign's
SDEdit baseline looks like **~0.30** (light edit). These are different operating points — so the
morning's constant-hyperparameter run should test our geodesic at *their* strength too, not only
0.8. We are on **SDXL** (diffusers img2img, `strength` arg); map their `n_start/NFE` → our
`--strength`. (SDXL vs SD3 is fine for a method comparison; keep both arms on SDXL.)

---

## Full runs + winners (PIE-Bench, what we have)

All on SDXL, 1024px, 17 NFE (FlowAlign's plain-sampling budget), A5000 (24GB). Metric:
DINO structure-distance (↓) × CLIP-directional editability (↑). "Win" = NW of the vanilla
strength frontier {0.5,0.6,0.7,0.8,0.9}.

**P0 `sub20` (B = full-gen geodesic seed, no x0):** WINNERS **NONE** — dominated (white-magnitude
seed is OOD). KILL variant B.

**P4 `sweepA` / `sweepSDG` (n=20, apples-to-apples, τ=0≡vanilla):**
vanilla frontier: s0.5 `0.088/+0.028` · s0.7 `0.109/+0.040` · s0.8 `0.127/+0.100` · s0.9 `0.145/+0.131`
- **A_t0.25 = `0.108/+0.065` → WIN** (vs vanilla's +0.040 at struct 0.108 → **+0.025 margin**,
  fills the Pareto gap between s0.7 and s0.8). A_t0.5/0.75 lose.
- **sdg_src_t0.25 = `0.113/+0.056` → WIN but marginal** (+0.002 over the interpolated frontier).
  sdg_src_t0.5/0.75, sdg_wht_t0.5 lose.
- Reversal vs the synthetic chair (there A over-locked, SDG won); on 20 real images A is stronger.

**P5 n=100 confirmation — DONE (WIN HOLDS, margin shrinks):** vanilla frontier (100 imgs)
s0.5 `0.091/+0.035` · s0.7 `0.112/+0.062` · s0.8 `0.130/+0.098` · s0.9 `0.153/+0.124`.
- **A WINS confirmed:** A_t0.125 `0.118/+0.076` (+0.002), **A_t0.25 `0.110/+0.062` (+0.0045)** —
  A_t0.25 won at both n=20 and n=100 (most robust). A_t0.375 loses.
- **SDG marginal:** sdg_src_t0.125 `0.120/+0.081` (+0.003); sdg_src_t0.25 fell to a **tie**.
- **Margin collapsed +0.025 (n=20) → +0.0045 (n=100)** — vanilla improved more with more data.
  Real + consistent but **modest**, method A > SDG. Verdicts:
  `/storage/.../results/e47_conf{A,SDG}/verdict.md`.
- **Wins sit at struct ~0.11 (≈ vanilla strength 0.65–0.7), NOT FlowAlign's lighter ~0.30 regime**
  — so the morning constant-hyperparameter run must check whether a win exists at *that* fixed
  light operating point (it may not; if so, report where on the strength axis the win lives).

---

## The method (for the meeting / roadmap writeup)

**What is done.** Standard SDEdit forms a partially-noised latent at strength `s`:
`x_std = √ᾱ·x0 + √(1−ᾱ)·ε` (white ε), then denoises toward the target prompt. We **keep x_std's
magnitude** (the correct strength-`s` energy spectrum) and **rotate only its phase** a fraction `τ`
along the **geodesic** between the source phase and a target phase:

```
φ_new(k) = φ_std(k) + τ(k) · wrap( φ_target(k) − φ_std(k) ),     |X_new| = |X_std|
wrap(δ) = ((δ+π) mod 2π) − π        # shortest signed arc per frequency k
```
`τ=0` ≡ vanilla SDEdit (validated: the sanity arm reproduces it). `target = source phase` ⇒
**structure-restore** (push the noised phase back toward the clean layout); `target = white` ⇒
**edit-boost**. `τ` can be **per radial-frequency band** (restore coarse layout, leave/​whiten detail).

**Why a geodesic, and how it differs from "linear".** On the unit circle each frequency's phasor
must keep |·|=1. The **linear/chord** mix `(1−τ)e^{iφ0} + τe^{iφ1}` then renormalize travels the
*chord*: variable angular speed, and when the two phasors are near-antipodal (δ≈±π) the chord
passes close to the origin and the renormalized phase **flips discontinuously** — jitter/fringing.
The **geodesic** `φ0 + τ·wrap(δ)` rotates at **constant angular velocity** along the *arc*, no flip,
smooth and monotone. (E46 used the chord and KILLed; the geodesic is the clean version.)
Equally important is the **energy/phase decoupling**: structure rides on phase, energy on magnitude,
so we can dial structure (`τ`) independently of the edit budget (`s`). That is exactly why the
apples-to-apples win exists where E46's seed-bake did not.

**Difference from the two reference papers** (both training-free, frequency-domain — hence the
confusion; we are distinct on three axes: *phase vs magnitude*, *geodesic vs inject*, *editing vs gen*):
- **Colorful-Noise** (arXiv 2605.00548, this repo's basis): manipulates **low-frequency
  magnitude/noise** with image priors for **color/structure-conditioned generation**. We touch
  **phase** (not magnitude), and we do **editing** via SDEdit, not conditional generation.
- **Φ-Noise** (arXiv 2605.24509): injects **low-frequency phase** from a reference **video** into
  the noise to transfer **motion** for **video generation**. We (a) **geodesically interpolate**
  phase by a controllable `τ` (vs hard inject/replace), (b) operate on the **SDEdit noised latent**
  as a perturbation with `τ=0≡vanilla` (an editing knob, not a generation prior), and (c) target
  **image editing** with an explicit energy/phase decoupling and a matched-strength comparison.

---

## Motivation & directions (for the meeting framing)

**The pitch.** Inversion-based editors (and FlowEdit/FlowAlign-style training-free editors) are
**slow** — inversion costs a full extra pass (17+17 NFE) and FlowEdit/AlignFlow cost >2 NFE/step
(33 NFE). We want **fast AND good** editing: SDEdit is fast (one partial pass) but loses structure;
our geodesic phase-perturbation **adds structure back for ~free** (one FFT, no extra NFE), so it
keeps SDEdit's speed while closing its quality gap toward inversion-quality edits.

**Future direction to explore (note for the meeting).** Many **inversion methods themselves use
SDEdit-style partial noising** when they generate — they do **not** run the full generation path.
So our geodesic phase-perturbation can be **dropped on top of inversion editors** at their
partial-noising step, potentially improving structure preservation for them too. This makes the
contribution a reusable primitive, not a single pipeline.

---

## Roadmap / meeting prep (the user wants the roadmap to explain THIS experiment)

The formal roadmap registration is intentionally deferred until the n=100 verdict + constant-
hyperparameter number are in (so status/verdict are accurate). **Morning task once verdicts land:**

1. `experiments/roadmap_registry.py`: add **E47** (thread `seed`; status `active` if confirmed,
   else `dead-end`). Title e.g. *"Geodesic phase-perturbed SDEdit — fast structure-preserving
   editing"*. Use the **Method / geodesic-vs-linear / vs Colorful-Noise & Φ-Noise / directions**
   text above as the writeup body.
2. `experiments/EXPERIMENTS.md`: add `## E47` (Method / Key result / Verdict / Artifacts).
3. `docs/experiment-reports/EXPERIMENT_47.md`: full report (this is the headline experiment — give
   it the frontier plots + the method/geodesic/decoupling explanation; it's the meeting centerpiece).
4. `experiments/manifests/E47.json`: config + metrics + artifact paths.
5. `python experiments/make_roadmap.py && python experiments/make_roadmap.py --check` (must exit 0),
   commit.

For the **meeting**: lead with the motivation (fast+good editing vs slow inversion/FlowEdit), one
slide of past line (E41 frontier-trap → E46 chord KILL), then E47 = the geodesic + decoupling that
finally clears the frontier, with the PIE-Bench numbers at FlowAlign's constant SDEdit config.
The probe log `experiments/e47_log.md` (P0–P4) has the full chronology.

---

## Environment / how to resume (save yourself time)

- **RunAI CLI:** use **`/home/shimon/.runai/bin/2.116.6/runai`** (v2.116.6 — has `workload`/
  `training` verbs). The `/usr/local/runai/runai` (v2.3.4) is stale; the empty `~/.kube/config`
  blocks it. `export PATH=/home/shimon/.runai/bin/2.116.6:$PATH`. Project = **`avidan`** (4 GPU).
- **Submit pattern:** `runai training submit <name> -p avidan -i pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime
  --gpu-devices-request 1 --large-shm --existing-pvc claimname=storage,path=/storage --command --
  bash /storage/malnick/colorful-noise/experiments/cluster_e47_job.sh <stage>`. Change the job name
  on resubmit (RunAI rejects exact reuse).
- **/storage SSHFS** dies periodically. To remount: `screen -dmS storage runai workspace
  port-forward shimon-myssh --port 2222:22`, then
  `echo root | sshfs -o cache=yes -o password_stdin -o reconnect -p 2222 root@127.0.0.1:/storage /storage`.
  (`shimon-myssh` must be Running: `runai workspace list | grep shimon-myssh`.) The mount-storage
  skill automates this but its `ls imagenet/` probe is stale (dir is `imagenet-datasets`).
- **Staging:** the `/storage` checkout is **not git**; `rsync experiments/*.py experiments/*.sh
  /storage/malnick/colorful-noise/experiments/` after every code change, THEN submit.
- **Cluster job stages** (`experiments/cluster_e47_job.sh`): `--sub20`/`--sub100` (B),
  `--sweepA`/`--sweepSDG` (n=20), `--confA`/`--confSDG` (n=100). For the morning constant-strength
  run, add a stage that fixes `--strengths <one value>` and `--method {A,sdg} --taus 0.25` (the
  harness already supports `--strength` and `--method`).
- **Watching long cluster jobs:** launch a bounded `run_in_background` bash that polls
  `runai training standard describe <job>` for `Phase: Succeeded|Failed` then dumps `logs` +
  `verdict.md` (example in the session; cluster jobs aren't harness-tracked, so poll-then-exit is
  the bridge). Never foreground-`sleep`-loop.
- **stdout buffering / progress:** the harness re-reads logs fine; per-image progress prints
  `[e47] done <key>`. Verdicts: `results/e47_<tag>/verdict.md` (the `WINNERS:` line is the answer).
- **Push rule:** worktree commits go to **`main`** (`git push origin HEAD:main`); `origin/main`
  moves (user pushes from the sandbox in parallel) — `git fetch && git rebase origin/main` first.

## Scripts (all on `main`)

- `experiments/e47_geodesic.py` — harness. Helpers: `geodesic_seed` (B), `sdedit_geodesic`/
  `sdedit_phase_geodesic` (the apples-to-apples op: `toward="source"|"white"`),
  `spectral_geodesic_sdedit` (energy-from-x_std variant), `two_band_t_field`. `main()` takes
  `--method {A,sdg,B} --strength --taus --taus_white --n_per_type --steps --tag`.
- `experiments/cluster_e47_job.sh` — RunAI entrypoint (stages above).
- `experiments/e47_log.md` — append-only probe log P0–P4 (durable record).
- Reused: `piebench.py` (HF `UB-CVML-Group/PIE_Bench_pp`, cache on /storage), `common.py`
  (SDXL `load_pipe`/`generate`/`encode_img_sdxl`), `struct_metrics.py`, `latent_spectral_ops.py`,
  `e46_probe1.py` (`sdedit` helper). E46 context: `docs/experiment-reports/EXPERIMENT_46.md`.
- Local visual grids from this session (job tmp, transient): `e47_chair_apples.png`,
  `e47_sweep{A,SDG}_grid.png`, `e47_sdg_chair.png`.
