# E44 — Apples-to-apples FlowAlign reproduction + ours on top (PIE-Bench)

Goal (two gates):
1. **Reproduce** FlowAlign's published PIE-Bench table on **SD3-medium** with their official code, within ~5% on Structure Distance & CLIP. HARD GATE — if it doesn't reproduce, stop.
2. **Beat it**: port our spectral phase-clamp into the *same* SD3 FlowAlign loop and show **lower Structure Distance at matched edited-CLIP** on PIE-Bench.

## Locked design decisions (2026-06-21)
- **Baseline:** SD3-medium + official FlowAlign code (`github.com/FlowAlign/FlowAlign`), match published table.
- **Metrics:** official PIE-Bench protocol (Structure Distance, bg-masked PSNR/LPIPS/MSE/SSIM, whole + edited-region CLIP). NOT our `struct_metrics`.
- **HP selection:** tune the spectral band on an **Emu Edit test subset** (disjoint from PIE-Bench). No tuning on PIE-Bench.
- **Win criterion:** better structure at **matched edited-CLIP** — sweep CFG ω for both methods, compare Structure-Distance vs edited-CLIP *curves* (FlowAlign Fig. 3a style). Not a single cherry-picked point.
- **Port:** reimplement the FLUX `sbn_phase` clamp inside the official SD3 FlowAlign velocity loop (forced by the SD3-baseline choice; FLUX-vs-SD3 would reintroduce a backbone mismatch).

## Findings from official repo + paper
- Paper backbone = **SD3.0-medium**; repo loads `stabilityai/stable-diffusion-3-medium-diffusers`. NFE=33, ζ=0.01 (hardcoded `0.01` in the update), seed=123, 1024px. README released imgs at CFG 13.5; paper calls ω=7.5 "balanced"; Fig 3a sweeps ω∈{5,7.5,10,13.5}.
- **Official repo ships ONLY single-image inference** (`run_edit.py`) — NO PIE-Bench loop, NO metric code. We must build both ourselves; canonical metric source = **PnPInversion / "Direct Inversion"** repo (the standard PIE-Bench eval FlowAlign used).
- FlowAlign update (`diffusion/editing/sd3_edit.py::SD3FlowAlign.sample`, ~L226):
  `xt += (σ_next-σ)*(vp-vq) + 0.01*(qt - σ*vq - pt + σ*vp)` where
  `vp = vp_src + ω*(vp_tgt - vp_src)` (CFG **negative = src prompt**), `vq = v(qt, src)`.
  → **Port insertion point is clean**: clamp `vp`'s low-band phase toward `v(pt, c_src)` right after the CFG combine.
- CFG negative is the **source prompt** (not null) — matches our FLUX reimpl's `w`/source-as-negative.

## Access / data status
- HF token present; **gated SD3.0-medium reachable** (ACCESS OK). Not yet downloaded (only 3.5-medium cached).
- PIE-Bench: only HF++ variant (`UB-CVML-Group/PIE_Bench_pp`) cached at `/storage/malnick/datasets/pie_bench_hf`. Its masks are unusable strings → **cannot do bg-masked metrics**. Need **original PIE-Bench** (mapping_file + annotation_images + masks), shipped with PnPInversion.
- Official repo cloned to `/home/shimon/research/flowalign_official` (env: torch 2.1.2+cu118, diffusers 0.33.1 — run in its own env, not colorful-noise's).

## Probes
(append-only; each gets KEEP/KILL/PARK)

### P0 — reproduce FlowAlign PIE-Bench table (in progress)
Hypothesis: official code + official metrics on SD3.0-medium reproduces their published numbers within ~5%.
Plan: get original PIE-Bench data (w/ masks) + PnPInversion metrics; stand up FlowAlign env; batch-edit 700 imgs at their setting; score; compare to table.

Recon results:
- Metric source = `cure-lab/PnPInversion` cloned to `/home/shimon/research/pnpinversion`.
  `evaluation/evaluate.py` computes structure_distance + {psnr,lpips,mse,ssim} in whole /
  `_unedit_part` (bg = 1-mask) / `_edit_part` (mask) + CLIP whole/edited. Masks are
  **RLE in mapping_file.json**, decoded by `mask_decode`.
- Original PIE-Bench data is behind a Google Form (forms.gle/hVMkTABb4uvZVjme9). BUT:
- **UNBLOCK:** cached HF++ (`/storage/malnick/datasets/pie_bench_hf`) stores `mask` as the SAME
  RLE string (e.g. `"0 262144"`), plus `blended_words` + bracketed target prompts. So we can run
  the official metric on data we already have — **no gated download**. (Caveat: RLE is 512×512 =
  262144; eval at 512, resize edits 1024→512.)
- FlowAlign reports PIE-Bench as a CLIP-vs-bgPSNR **curve** over CFG {5,7.5,10,13.5} (Fig 3a),
  not one row → reproduction gate = land on that curve; matches our curve-based win criterion.
- Reproduction blocker remaining: GPU runs via runai **only in the docker sandbox** (can't submit
  from here). Plan: build harness + hand off exact submit command.

Verdict: TBD (harness build pending go-ahead).
