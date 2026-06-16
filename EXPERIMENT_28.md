# E28 — Does biasing the seed RESCUE dropped elements on hard compositional prompts?

**Question.** E25–E27 found seed-biasing is a do-no-harm *palette/appearance* lever with flat
CLIP-T — but those tests were on easy prompts (no headroom) with a metric (CLIP-T) blind to
dropped elements. So: is there a regime where the seed *matters*? The natural candidate is
**hard compositional prompts where the baseline DROPS an element** (missing object / wrong
attribute binding). There the prompt is achievable for *some* seeds but not others, so biasing
a failing seed toward the prompt might "tip the sampler into the mode that renders the missing
element." We test that **on the failing subset only**, with a metric that *sees* dropped
elements: **B-VQA**.

**Metric — B-VQA** (T2I-CompBench attribute binding). spaCy extracts the prompt's noun phrases;
BLIP-VQA answers "{phrase}?" per image; the score is the **product** of P(yes), so one
dropped/mis-bound element tanks it. (We use `Salesforce/blip-vqa-base` — safetensors; the
capfilt-large checkpoint is `.bin`-only and blocked on torch<2.6. Fine for *relative* recovery.)

## Design (`experiments/e28_seedrescue.py`, SDXL 1024px)

Reuses `compbench.{load_compbench_prompts, load_bvqa, noun_phrases, _p_yes}`,
`e26_seedalign_sdxl.{load_sdxl, optimize_seed, ...}`, `clip_sim`, `common`.

1. **Scan**: 30 CompBench prompts (color/shape/texture) × 4 seeds = 120 baseline gens; B-VQA +
   per-phrase P(yes). **FAIL** = B-VQA < τ=0.5. Per-prompt **seed-dependence** = fraction of
   seeds that pass.
2. **Intervene** on the worst 37 failures — bias the failing seed via iterative latent-mode
   optimization (`optimize_seed`, K=8, re-standardized so `‖z‖=√d=256`), regenerate, re-score.
   Two targets: **A** = full prompt; **B** = the single lowest-P(yes) noun phrase (the dropped
   element).
3. **Controls**: **re-roll** = a fresh random seed (no optimization) — does biasing beat a new
   draw? **do-no-harm** = apply arm A to passing pairs.

## Results

Baseline fail rate **0.308** (37/120) — good headroom. On the failing subset:

| (37 failures) | arm A (full prompt) | arm B (dropped phrase) | **re-roll** |
|---|---|---|---|
| mean ΔB-VQA | +0.101 | +0.125 | **+0.230** |
| recovery rate (cross τ) | 0.189 | 0.243 | **0.324** |
| recovery on **seed-dependent** (n=21) | 0.286 | 0.429 | **0.571** |

**do-no-harm: FAILED.** Applying the optimization to *passing* pairs **dropped** B-VQA by
**−0.176** on average — i.e. biasing the seed toward the prompt actively *breaks* compositions
that already worked.

## Verdict (a clean negative)

- **Biasing the seed does not beat simply re-rolling it.** Even in the regime the seed genuinely
  matters (seed-dependent compositional failures), a fresh random seed recovers **0.57** of
  failures vs **0.43 / 0.29** for the gradient-biased seed — re-roll wins on every metric, by a
  wide margin.
- **Always-fail prompts** (passrate 0 — e.g. the 4–5-element `shape_008`, `texture_009`) recover
  with *nothing*: no seed exists in range that the model can render, so seed manipulation can't help.
- **The bias even hurts working cases** (−0.176 on passers), so it is not a safe default.

**Why.** Consistent with E25–E27: the gradient toward CLIP/text moves *palette/appearance*, which
keeps the latent in the **same compositional basin** while degrading it — it does not jump the
sampler to a different "which objects appear" mode. Changing *which* mode renders requires a
genuinely different seed (a re-roll), not a smooth nudge of the current one. The seed's influence
on composition is real (seeds differ a lot — that's the seed-dependence) but **not steerable by
gradient toward the prompt**.

**Practical takeaway:** for compositional adherence, *best-of-N random seeds + a B-VQA picker* beats
optimizing a seed; the gradient-bias direction is a dead end for adherence. This closes the
seed-as-adherence-lever line (E25→E28). The remaining real, positive use of seed-biasing is the
**appearance/palette steering** documented in E27 (Arm B).

## Status

Done. Constraint `‖z‖=√d` held on every edit. Negative result is robust across 37 failures and a
seed-dependence stratification; the decisive comparison (re-roll > arm B > arm A on the
seed-dependent stratum) is unambiguous. No HTML explainer (negative result; plan made it
conditional on positive signal).

## Run

```bash
python experiments/e28_seedrescue.py quick   # smoke
python experiments/e28_seedrescue.py         # full -> results/e28/{grid_recovered.png, summary.png, report.json}
```

Artifacts: `results/e28/grid_recovered.png` (cols: baseline-fail | opt-A | opt-B | re-roll),
`summary.png`, `report.json`. Lineage: `EXPERIMENT_26.md`, `EXPERIMENT_27.md` (E25–E27 seed-bias thread).
