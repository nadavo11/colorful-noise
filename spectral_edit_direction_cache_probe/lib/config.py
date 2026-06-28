"""E51 — Spectral Edit-Direction Cache Probe: shared config, paths, variant defs.

Diagnostic of the hypothesis that the *edit direction*
    delta_edit(t) = v_edit(t) - v_src(t)
is more temporally/spectrally stable (cacheable) than the full edited prediction
    v_edit(t) = T(x_t, target_prompt, t).

Pipeline: FLUX.1-dev img2img (SDEdit) on PIE-Bench, 512px, 4-bit NF4. GPU code runs in the
uv env (diffusers 0.38); analysis/figures/report run in the anaconda env (matplotlib).
"""
from __future__ import annotations
import json
from pathlib import Path

LIB = Path(__file__).resolve().parent
ROOT = LIB.parent                                  # spectral_edit_direction_cache_probe/
REPO = ROOT.parent
OUT = REPO / "outputs" / "spectral_edit_direction_cache_probe"
FIG = OUT / "figures"
SAMP = OUT / "samples"
DIAG = OUT / "diagnostics"
LOGS = OUT / "logs"

# ---- Text-Token Modulation Autopsy (E52) sub-module output tree --------------------
# Lives in the SAME output dir as the cache probe (one integrated report) but keeps its
# heavy artifacts under token_autopsy/ so the two diagnostics don't collide.
TOK = OUT / "token_autopsy"
TOK_TABLES = TOK / "token_tables"
TOK_HEAT = TOK / "token_heatmaps"
TOK_SPATIAL = TOK / "token_spatial_maps"
TOK_GRIDS = TOK / "token_intervention_grids"
TOK_CURVES = TOK / "token_weight_curves"
TOK_CACHE = TOK / "token_cache_correlation"

PIEBENCH = REPO / "baseline_establishment" / "data" / "benchmark_subsets" / "piebench"
BASE_LIB = REPO / "baseline_establishment" / "lib"

PHASE_VERSION = "e51-spectral-edit-direction-cache-probe-v1"
SEED = 0
SIZE = 512
STEPS = 24            # SDEdit denoising steps after strength truncation
GUIDANCE = 2.5
STRENGTH = 0.7       # SDEdit noise level; high enough to permit real edits, low enough to keep structure

# Radial spectral band edges as a fraction of the Nyquist frequency.
LOW = 0.15           # low band  : r < LOW
HIGH = 0.45          # high band : r > HIGH ; mid band is in-between
LOWPASS_FRAC = 0.25  # SEA-style low-pass cutoff used for the *cache decision* of spectral variants

# 8 PIE-Bench task types -> reporting category + edit scope (local vs global).
CATEGORY = {
    "object_replace": "object change",
    "object_add":     "object change",
    "object_remove":  "object change",
    "attribute":      "attribute edit",
    "color":          "color/style",
    "material":       "color/style",
    "global":         "global edit",
    "style":          "color/style",
}
SCOPE = {
    "object_replace": "local", "object_add": "local", "object_remove": "local",
    "attribute": "local", "color": "local", "material": "local",
    "global": "global", "style": "global",
}

VARIANTS = [
    "full_compute_reference",
    "raw_full_prediction_cache",
    "spectral_full_prediction_cache",
    "raw_edit_delta_cache",
    "spectral_edit_delta_cache",
]
# (signal cached, decision filter) per variant. reference recomputes every step.
VARIANT_SPEC = {
    "full_compute_reference":         dict(signal="full",  filt="none",     cache=False),
    "raw_full_prediction_cache":      dict(signal="full",  filt="raw",      cache=True),
    "spectral_full_prediction_cache": dict(signal="full",  filt="spectral", cache=True),
    "raw_edit_delta_cache":           dict(signal="delta", filt="raw",      cache=True),
    "spectral_edit_delta_cache":      dict(signal="delta", filt="spectral", cache=True),
}
CACHE_VARIANTS = [v for v in VARIANTS if VARIANT_SPEC[v]["cache"]]

# Closed-loop operating points (target fraction of denoising steps skipped/reused).
SKIP_PRIMARY = 0.5                       # single point used for the full 24-example qualitative + quality run
PARETO_TARGETS = [0.2, 0.33, 0.5, 0.66, 0.8]  # threshold sweep, on a representative subset
# Representative subset (one per category family, spanning local/global) for the Pareto sweep
# + dense per-example diagnostics. Filled at data-build time; this is the fallback ordering.
PARETO_SUBSET_TASKS = ["object_replace", "object_add", "object_remove", "attribute",
                       "color", "material", "global", "style"]

# Method-card metadata required by the report (model / data / supervision / one-line insight).
METHODS = {
    "full_compute_reference": dict(
        model="FLUX.1-dev img2img (4-bit NF4), full denoise",
        data="PIE-Bench source image + target prompt",
        supervision="none (training-free); gold reference trajectory",
        insight="Every step recomputes v_edit(t); the quality ceiling all caches are scored against."),
    "raw_full_prediction_cache": dict(
        model="FLUX.1-dev img2img; reuse stale v_edit on skipped steps",
        data="PIE-Bench source image + target prompt",
        supervision="none; skips chosen where raw ||v_edit(t)-v_edit(t-1)|| is smallest",
        insight="The SeaCache-style move without spectral filtering: cache the whole prediction by raw stability."),
    "spectral_full_prediction_cache": dict(
        model="FLUX.1-dev img2img; reuse stale v_edit on skipped steps",
        data="PIE-Bench source image + target prompt",
        supervision="none; skips chosen by low-pass-filtered v_edit change (SEA-style)",
        insight="SeaCache-like baseline: decide reuse on the perceptually-dominant low-frequency band of v_edit."),
    "raw_edit_delta_cache": dict(
        model="FLUX.1-dev img2img; freeze stale delta_edit, keep base v_src live",
        data="PIE-Bench source + target prompt (both branches)",
        supervision="none; skips chosen where raw ||delta_edit(t)-delta_edit(t-1)|| is smallest",
        insight="Cache the *edit direction*, not the whole prediction — the proposed object, raw-gated."),
    "spectral_edit_delta_cache": dict(
        model="FLUX.1-dev img2img; freeze stale delta_edit, keep base v_src live",
        data="PIE-Bench source + target prompt (both branches)",
        supervision="none; skips chosen by low-pass-filtered delta_edit change",
        insight="The main hypothesis: edit direction is smoothest in its low-frequency band — cache that."),
}

VARIANT_SHORT = {
    "full_compute_reference": "Full (ref)",
    "raw_full_prediction_cache": "Raw full-cache",
    "spectral_full_prediction_cache": "Spectral full-cache",
    "raw_edit_delta_cache": "Raw delta-cache",
    "spectral_edit_delta_cache": "Spectral delta-cache",
}
VARIANT_COLOR = {
    "full_compute_reference": "#222222",
    "raw_full_prediction_cache": "#e07b39",
    "spectral_full_prediction_cache": "#c0392b",
    "raw_edit_delta_cache": "#2e86c1",
    "spectral_edit_delta_cache": "#1e8449",
}


# ===========================================================================
# Text-Token Modulation Autopsy (E52) — config
# ===========================================================================
# Where text enters FLUX (MMDiT). Stated verbatim in the report; see token_attn.py.
TEXT_ENTRY = dict(
    representation="T5-XXL token sequence (encoder_hidden_states, 512 tokens × 4096-d) + a "
                   "global pooled CLIP-text vector (pooled_projections, 768-d).",
    mechanism="MMDiT joint attention. The T5 token sequence is concatenated with the image "
              "tokens and they attend JOINTLY — there is no separate U-Net cross-attention. In "
              "both the 19 double-stream (FluxTransformerBlock) and 38 single-stream "
              "(FluxSingleTransformerBlock) blocks the text tokens occupy the FIRST txt_len "
              "positions of the joint key/value sequence, so 'image→text attention' is the "
              "image-query rows attending to those leading text-key columns.",
    pooled="The pooled CLIP vector is added to the timestep embedding and drives AdaLayerNorm "
           "modulation (scale/shift/gate) — a GLOBAL conditioning path, not per-token.",
    note="Token-level control therefore lives entirely in the joint-attention columns of the "
         "T5 tokens; that is the surface this autopsy instruments and intervenes on.",
)

# Blocks tapped for attention recording (a depth-spanning subset of the 19 double-stream
# blocks — recording every block × step × example is too heavy and redundant). 'd<i>' =
# double-stream block i; the recorder maps these to transformer.transformer_blocks[i].
TAP_BLOCKS = [0, 4, 9, 14, 18]
# Coarser block-coverage scan (one cheap pass) so the token×layer heatmap spans full depth.
TAP_BLOCKS_LAYER_SCAN = [0, 2, 4, 6, 9, 12, 14, 16, 18]

# Causal token-intervention grid (Section C/D). Four internal mechanisms + an embedding one.
TOK_WEIGHTS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
TOK_MECHANISMS = ["embed_scale", "attn_logit_bias", "attn_prob_reweight", "value_scale"]
# Token roles probed per example (Section C). Assigned by token_attn.assign_roles().
TOK_ROLES = ["edited_noun", "attribute", "style", "background", "control"]
# attn_logit_bias is additive in logit space; map a multiplicative "weight" w to a bias
# beta = LOGIT_BIAS_SCALE * ln(w) so w<1 suppresses and w>1 amplifies, matching the others.
LOGIT_BIAS_SCALE = 1.0

# Autopsy is expensive (extra attention recompute + per-token ablations + intervention
# sweeps). Run it on the Pareto subset only by default; cap the dense intervention sweep
# to a few examples and a few timesteps.
TOK_INTERVENTION_EXAMPLES = 3          # #examples that get the full mechanism×weight×role grid
TOK_ABLATION_STEPS = 6                 # #timesteps sampled for per-token Δ_edit ablation
TOK_SPATIAL_STEPS = 6                  # #timesteps kept for per-token spatial attention maps

TOK_PHASE_VERSION = "e52-text-token-modulation-autopsy-v1"


def ensure_dirs():
    for d in (OUT, FIG, SAMP, DIAG, LOGS,
              TOK, TOK_TABLES, TOK_HEAT, TOK_SPATIAL, TOK_GRIDS, TOK_CURVES, TOK_CACHE):
        d.mkdir(parents=True, exist_ok=True)


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2))


def rel(p):
    """Repo-relative path string for manifests/reports."""
    try:
        return str(Path(p).resolve().relative_to(REPO))
    except Exception:
        return str(p)
