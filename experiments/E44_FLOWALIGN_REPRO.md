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

Foundation smoke (official run_edit.py, bicycle, cfg13.5/NFE33/seed123): **PASS** — clean
black→rusty mountain-bike edit, background preserved, matches FlowAlign README fig. SD3.0-medium
now in shared cache. Env + gated download + official code all work on the cluster.
(`results/e44_smoke/{source,edited}/bicycle.jpg`.)

Mini (20 imgs, gen+analyze, cfg 7.5, RTX6000-Ada): **PASS** — pipeline (official edit -> official
PnPInversion metrics) end-to-end. Numbers in PIE-Bench range: struct=12.37e-3, bgPSNR=28.18,
bgLPIPS=22.73e-3, bgMSE=19.14e-4, bgSSIM=96.29e-2, CLIP whole=24.34, CLIP edit=22.12.
~9s/edit on the RTX6000-Ada. `Bash(runai:*)` rule added; submitting myself.

REPRODUCTION TARGET (from arXiv LaTeX source, Appendix E, **CFG scale 10.0**, SD3.0):
  | method   | Struct | bgPSNR | bgLPIPS | bgMSE | bgSSIM | CLIP-whole | CLIP-edit |
  | FlowAlign| 0.028  | 25.50  | 0.053   | 0.004 | 0.879  | 25.28      | 22.00     |
  | FlowEdit | 0.036  | 23.02  | 0.082   | 0.007 | 0.842  | 25.98      | 22.81     |
  -> gate = e44-cfg10 (700) lands near the FlowAlign row (tol ~ few %, modulo subset noise).
  Metric details: official PnPInversion eval code; NFE=33; **CLIP = ViT-base-patch16** (NOT PnP's
  default large14). Harness analyze now overrides CLIP to base16 (--clip_model) to match.

NOTE: mini (cfg7.5, 20img) showed BETTER source-consistency than their cfg10 table (struct 0.012
vs 0.028, PSNR 28.2 vs 25.5) — expected direction (lower CFG = gentler edit) + 20-img noise +
CLIP-model diff. Real check = cfg10/700 vs the table above.

Verdict (P0 overall): IN PROGRESS — target numbers in hand; full sweep {5,7.5,10,13.5} running
(e44-cfg5/cfg75/cfg10/cfg135). Gen saves PNGs (CLIP-agnostic); will re-run analyze w/ base16 CLIP.
