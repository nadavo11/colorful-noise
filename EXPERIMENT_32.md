# E32 — Per-object token-frequency control on two-object prompts

**The direction.** E24 found that token-axis FFT bands of Flux's T5 sequence embedding
are meaningful, and E30 turned that into a *continuous but global* knob (scale a frequency
band of the whole prompt). E32 asks the obvious next question for **editing**: in a prompt
with two objects, can we boost/cut the high or low token-frequencies of **one object**, and
is the effect **selective** to that object — or does it just behave like E30's global gain?
A "yes" would be a genuinely new, object-local text-space editing handle.

## Background (plain language)

- **Text conditioning.** Flux encodes a prompt to a T5 **sequence embedding**
  `E (1, L, 4096)` (L = real tokens) plus a pooled vector. We modify `E` before generation
  and feed it straight to the transformer (the E10 `gen_emb` hook, true-CFG = 1).
- **Token-axis FFT.** Following E24/FNet, we FFT `E` along the **token axis** (a 1-D DFT per
  embedding channel). **Low** token-frequencies = slow variation across the prompt
  (DC = the bag-of-words mean direction); **high** = sharp token-to-token detail.
- **Why "per-object" forces a windowed FFT.** Frequency and token-position are **conjugate**:
  a single FFT over the whole sequence cannot be told to "only touch object A's positions."
  The one coherent per-object operation is a **windowed FFT over the object's contiguous
  token span** `E[:, a:b]` — scale a band there, inverse-FFT, stitch the rest back unchanged
  (`text_spectral_ops.apply_on_subspan` + `band_gain_1d`). This is the per-object
  generalization of E24/E30, which only ever windowed the prefix `[:, :L]`.
- **The cost of windowing — and why the cut is 0.51, not 0.25.** An object phrase is only
  ~5–9 tokens, so its windowed rfft has just `w//2+1` bins (3–5 here) and its lowest non-DC
  normalized frequency is already ≈0.33. E24/E30's `cut = 0.25` (tuned for the full
  512-length sequence) would leave the **low band empty → a silent no-op**. E32 uses the
  **median split `cut = 0.51`**, which keeps both bands non-empty even for a 3-bin
  (5-token) window (the mid-bin at 0.5 → low, Nyquist 1.0 → high) and never scales DC, so an
  object's global level is preserved. This coarseness is intrinsic to per-object frequency
  control and is reported explicitly (bins-per-object).

## Method

- **Model.** FLUX.1-dev (bnb 4-bit transformer, bf16), 28 steps, guidance 3.5,
  true-CFG = 1 — identical stack to E24/E30, reusing `load_flux_preencoded_lens`-style
  loading and `gen_emb`.
- **Prompts.** 10 two-object prompts of the form "*[object A]* and *[object B]*", each
  object a **multi-token phrase** (so its window has >1 bin) that appears verbatim in the
  prompt and doubles as the object's CLIP / B-VQA target (e.g. "a fluffy orange tabby cat"
  + "a sleeping golden retriever dog").
- **Word → token span.** No such utility existed in the repo. We map each object phrase to
  its T5 token span via the fast tokenizer's **char offset mapping** (every token whose
  char span overlaps the phrase's), with a **token-id-subsequence** fallback. Spans are
  computed while the tokenizer is alive (before the encoders are dropped to free GPU) and
  persisted to `results/e32/spans.json`. Validated on all 10 prompts: spans are valid,
  **non-overlapping** (the "and" token sits in the gap), 3–5 bins each.
- **Conditions per prompt** (`CUT0 = 0.51`, gains `{0.5 cut, 2.0 boost}`):
  - `baseline` (unmodified) — 1
  - **targeted**: `obj{A,B}` × band `{low, high}` × gain → 8
  - **global control**: whole-prompt band gain (E30-style), band `{low, high}` × gain → 4
  - **13 conditions × 10 prompts × 3 seeds.**
- **Metrics** (the claim is selectivity, so everything is per-object and paired to the
  same-seed baseline):
  - **Per-object CLIP** — `clip_scores(objA_phrase)` vs `clip_scores(objB_phrase)`
    (`e9_clipt.py`). For a targeted edit: `Δ_target`, `Δ_other`, and
    **selectivity = Δ_target − Δ_other**. For the global control: `Δ(objA)`, `Δ(objB)`,
    and `Δ(objA) − Δ(objB)` (expected ≈ 0 by symmetry).
  - **B-VQA presence** per object phrase (`compbench.bvqa_scores`) — corroborates presence.
  - Secondary context: whole-image `sharpness / hf_frac / colorfulness`
    (`e9_bandnorm_classes.image_metrics`).
  - **Hypothesis:** targeted selectivity > 0 and > the global control's ≈0; i.e.,
    localizing the *same fractional-band edit* to one object's tokens concentrates the
    effect on that object.

## Findings

Ran on runai (FLUX.1-dev, A5000; 10 prompts × 13 conditions × 3 seeds = 390 generations;
per-cell n = 60 targeted / 30 global). Deltas are paired to the same-seed baseline;
**selectivity = Δ_target − Δ_other** (targeted) or Δ(objA) − Δ(objB) (global control).

1. **Per-object editing IS object-selective and directionally steerable (CLIP).** Boosting
   one object's token-frequency band raises *that* object's CLIP and *lowers* the other's;
   cutting reverses it — for **both** bands, with the sign tracking the gain:

   | edit on target | Δtarget | Δother | **selectivity** | t |
   |---|---|---|---|---|
   | cut low (g0.5)  | −0.0041 | +0.0037 | **−0.0077** | −3.1 |
   | boost low (g2.0)| +0.0033 | −0.0029 | **+0.0062** | +2.1 |
   | cut high (g0.5) | −0.0027 | +0.0019 | **−0.0047** | −2.3 |
   | boost high (g2.0)| +0.0028 | −0.0017 | **+0.0045** | +1.6 |

   All four cells move in the predicted direction; three reach |t| ≳ 2. This is the clean
   push-pull signature of a genuine per-object effect.

2. **The global-gain control is a null (CLIP).** Whole-prompt gain gives selectivity
   ≈ −0.003…+0.003 with no consistent sign — so the **localization**, not the gain, produces
   the selectivity. Targeted beats the control on a controllable, sign-correct lever.

3. **Object presence (B-VQA): high band is where it concentrates, but it's noisy.** The
   high-band edit shifts presence in the intended direction (boost target high →
   Δtarget +0.040 / Δother −0.017; cut high → −0.047 / +0.029), echoing E30's "high/mid bands
   = attribute–object binding". But per-image B-VQA variance is large (sd ≈ 0.2–0.3), so at
   n = 60 these are **not** statistically significant (|t| ≤ 1.8); the low band is a null.
   Suggestive, not conclusive.

4. **Effect size is small.** CLIP shifts are ~0.005 on a ~0.22 baseline (sub-1%). Real and
   directional, but a weak handle — consistent with the intrinsic bin-coarseness (3–5
   bins/object).

**Verdict.** Unlike E24-MERGE (negative) and E31 (Δ≈0), per-object token-frequency editing
*is* object-selective and steerable in the intended direction (significant in CLIP, the
control is a null) — the text-freq thread's first **controllable** per-object lever. But the
magnitude is small and the presence (binding) effect, while concentrated in the high band,
is within noise at this N. A real but weak handle.

## Caveats & next

- **Bin coarseness.** 3–5 bins per object means "low vs high" is a coarse 2–3-way split; the
  `cut = 0.51` median split is the most balanced available but cannot be fine-grained. This
  is intrinsic to per-object windowing, not a tuning choice — reported per object.
- **Pooled vector untouched.** Only the sequence embedding `E` is edited; the pooled vector
  stays at baseline (matches E24/E30).
- **Short prompts.** L ≈ 13–16 tokens, so the "global" control already covers few bins;
  it remains a fair locality control (same fractional band, all tokens vs one object).
- **Next (recorded follow-ups, not yet run):**
  1. **Textual inversion → frequency control.** Learn an embedding for a pseudo-token
     `<obj>` from a few images of one object, place it in a two-object prompt, then boost/cut
     *its* span. No TI scaffolding exists in the repo today; diffusers `load_textual_inversion`
     or a custom loop adapted from the E25/E26 seed-optimization loops would be needed, and
     **SDXL/SD1.5 are safer than Flux** for TI tooling.
  2. **Channel-axis interpretability (the hidden D = 4096 axis).** Complementary to E32:
     find which *channels* of the embedding own which attributes (identity / color / texture
     / style) — via attribute probing (per-channel variance/correlation with an attribute
     label) and causal ablation (zero/scale channels, score with CLIP / B-VQA) — then steer
     those channels directly. E24 noted the hidden axis is **not** semantically ordered, so
     this likely needs learned channel *directions*, not raw indices. Composes with E32's
     token-span masking for per-object × per-channel edits.

## Reproduce

```bash
# spans + bins-per-object, no GPU (fails fast if a phrase won't map):
python experiments/e32_object_freq.py --part preflight

# smoke (1 prompt, 1 seed, 8 steps):
python experiments/e32_object_freq.py --part gen,analyze \
    --num_prompts 1 --seeds 1 --steps 8 --no_vqa --out_tag smoke

# full sweep (cluster; self-gating preflight -> smoke -> CLIP gate -> full):
bash experiments/cluster_e32_job.sh
```

Code: `experiments/e32_object_freq.py` (driver), `experiments/text_spectral_ops.py`
(`apply_on_subspan`, `band_gain_1d`), reuses `e10_cfg_spectral.gen_emb`,
`e9_clipt`, `e9_bandnorm_classes.image_metrics`, `compbench` (B-VQA), `common.save_grid`.
Cluster: `experiments/cluster_e32_job.sh` (ship via `kubectl cp`; `/storage` is not git).
