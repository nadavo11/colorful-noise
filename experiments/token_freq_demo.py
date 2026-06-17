"""Interactive browser demo for spectral image editing — TWO modes (E24-E36 toolkit).

A Gradio app with two tabs that both run on one Flux model:

  • TOKEN modulation  — FFT Flux's T5 token-sequence embedding and edit the spectrum ONCE
    before generation (E24/E30/E32/E35: low/high-pass, band gain, notch, phase/mag,
    two-prompt swap/blend/lerp, per-object band gain).
  • LATENT modulation — FFT the 2D diffusion latent and edit its radial spectrum DURING
    generation via a step-end callback (E8-E23/E36). The extra knob here is the SCHEDULE —
    *when* the op fires (every step / early / late / last step). Operators: low/high-pass,
    band gain, notch, phase-only/mag-only, phase band-keep, quantize phase, SBN→cfg1 /
    SBN→real / band-modulate / global-power, colored-noise init, restyle→B, SBN-blend A+B,
    and the offline two-latent hybrid / phase-swap.

Both tabs are thin wrappers around the ops in `text_spectral_ops.py` (token axis) and
`latent_spectral_ops.py` / `spectral_ops.py` / `style_ops.py` (latent axis); the only new
machinery is keeping the text encoders + VAE loaded so arbitrary prompts encode on the fly.

Run:  python experiments/token_freq_demo.py   (then ssh -L 7860:localhost:7860 <host>)
Model: FLUX.1-dev (bnb 4-bit transformer), single A5000 fits encoders + transformer + VAE.

Run with uv (auto-builds/caches an env from the inline deps below, incl. a CUDA torch):
    uv run experiments/token_freq_demo.py
"""
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch==2.5.1",
#     "diffusers==0.38.0",
#     "transformers==4.57.6",
#     "accelerate",
#     "bitsandbytes",
#     "sentencepiece",
#     "protobuf",
#     "huggingface-hub==0.35.3",
#     "gradio==5.9.1",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cu124"
# url = "https://download.pytorch.org/whl/cu124"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu124" }
# ///
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import text_spectral_ops as TS          # token axis (light: torch only)
import latent_spectral_ops as L         # latent axis (torch only; pulls spectral_ops/style_ops)
from spectral_ops import band_index_map, band_power
from style_ops import restyle_latent, color_noise, blend_references

REAL_LATENTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "results", "e10", "real_latents.pt")   # Flux-VAE real photos (E10)
N_BINS = 24

# Model registry -- only ONE entry is ever loaded (chosen by --model), so a single model
# sits on the GPU. flux-schnell is the lighter/faster option: same Flux architecture (so
# every op works unchanged) but timestep-distilled to ~4 steps, and ungated (no HF token).
MODELS = {
    "flux-dev":     {"repo": "black-forest-labs/FLUX.1-dev",     "max_seq": 512, "steps": 16, "guidance": 3.5},
    "flux-schnell": {"repo": "black-forest-labs/FLUX.1-schnell", "max_seq": 256, "steps": 4,  "guidance": 3.5},
}
MODEL = MODELS["flux-dev"]              # overridden in __main__ from --model (flux-dev == e7_flux_phase.REPO)
REPO = MODEL["repo"]

OBJ_CUT = 0.51   # E32 per-object median split (short windows need >0.25; see e32_object_freq)


def phrase_span(tokenizer, prompt, phrase, L):
    """Map object `phrase` -> (a, b) token indices in `prompt` (E32). Offset mapping with
    token-id-subsequence fallback. Inlined from e32_object_freq to keep the demo's imports
    light (no e9/e10/common chain)."""
    try:
        enc = tokenizer(prompt, max_length=512, truncation=True, return_offsets_mapping=True)
        offs = enc["offset_mapping"]
        c0 = prompt.index(phrase); c1 = c0 + len(phrase)
        idx = [i for i, (s, e) in enumerate(offs) if i < L and e > s and s < c1 and e > c0]
        if idx:
            return min(idx), min(max(idx) + 1, L)
    except (ValueError, KeyError, TypeError):
        pass
    pid = tokenizer(prompt, max_length=512, truncation=True)["input_ids"]
    ph = tokenizer(phrase, add_special_tokens=False)["input_ids"]
    for i in range(len(pid) - len(ph) + 1):
        if pid[i:i + len(ph)] == ph:
            return i, min(i + len(ph), L)
    raise ValueError(f"could not locate phrase {phrase!r} in prompt {prompt!r}")

OPS = [
    "baseline", "low-pass", "high-pass", "band gain", "notch",
    "phase-only", "mag-only", "phase band-keep", "phase gain",
    "two-prompt band-swap", "two-prompt band-blend", "two-prompt lerp",
    "per-object band gain",
]
TWO_PROMPT = {"two-prompt band-swap", "two-prompt band-blend", "two-prompt lerp"}
NEEDS_CUT = {"low-pass", "high-pass", "two-prompt band-swap", "two-prompt band-blend"}
NEEDS_RANGE = {"band gain", "notch", "phase band-keep", "phase gain", "per-object band gain"}
NEEDS_GAIN = {"band gain", "phase gain", "per-object band gain"}
# ops whose token-axis transform can ALSO be applied to CLIP's pre-pool sequence (then
# re-pooled at EOS) to push the same change into the global pooled vector (ppe).
CLIP_POOLABLE = {"low-pass", "high-pass", "band gain", "notch",
                 "phase-only", "mag-only", "phase band-keep", "phase gain"}

HELP = {
    "baseline":
        "**Baseline.** Prompt A, unmodified — the reference image (shown on the left).",
    "low-pass":
        "**Low-pass** *(uses `cut`)*. Keep token-frequencies in `[0, cut]` (incl. DC), zero "
        "the rest. Keeps the slow, global **gist** of the prompt and drops token-to-token "
        "detail. `cut→0` collapses toward a generic average prompt; `cut→1` ≈ the full prompt.",
    "high-pass":
        "**High-pass** *(uses `cut`)*. Keep `[cut, 1]` **plus DC**, zero the low band. Keeps "
        "sharp token-to-token **detail** and the overall level, drops the coarse structure. "
        "(DC is kept so the result isn't degenerate.)",
    "band gain":
        "**Band gain** *(uses `cut`, `band`, `gain`)*. Multiply **one** side of the `cut` by "
        "`gain` (>1 amplify, <1 attenuate); **DC is always left at unity**. `cut` sets *where* "
        "the split is; `band` picks *which* side is scaled — `high` boosts detail/texture/style, "
        "`low` boosts the coarse 'gist'. E30's continuous knob.",
    "notch":
        "**Notch** *(uses `cut`, `band`)*. Zero out *exactly* the chosen band (knockout), keep "
        "everything else incl. DC. A diagnostic: what does removing this band cost?",
    "phase-only":
        "**Phase-only.** Keep the token-axis **phase**, set every magnitude to 1. E30 found "
        "phase-only ≈ full → phase carries the content.",
    "mag-only":
        "**Mag-only.** Keep **magnitude**, set phase to 0. The complement of phase-only; E30 "
        "found this collapses → phase, not magnitude, is the semantic carrier. Expect a "
        "degenerate image.",
    "phase band-keep":
        "**Phase band-keep** *(uses `cut`, `band`)*. Band-limited phase-only: keep the original "
        "**phase** only on the chosen side of `cut` and flatten it to 0 elsewhere, while keeping "
        "**magnitude everywhere**. `band=low` = phase low-pass (phase kept in `[0,cut]`); "
        "`band=high` = phase high-pass. Since E30 showed phase carries the content, this asks "
        "*which frequency bands' phase* carries it.",
    "phase gain":
        "**Phase gain** *(uses `cut`, `band`, `gain`)*. Scale the **phase angle** by `gain` in the "
        "chosen band (magnitude kept): `gain=0` removes phase there (→ mag-only in that band), "
        "`1` = identity, `>1` amplifies the rotation. The phase analogue of band-gain. Note phase "
        "is circular, so `gain` near ±π wraps around (branch-cut effect); DC/Nyquist are left "
        "unscaled to keep magnitude valid.",
    "two-prompt band-swap":
        "**Band-swap** *(needs Prompt B; uses `cut`)*. Low band `[0,cut]` (incl. DC) from **A** "
        "+ high band `(cut,1]` from **B**, recombined: 'A's subject/gist + B's detail'. "
        "(E24/E30 merge.)",
    "two-prompt band-blend":
        "**Band-blend** *(needs Prompt B; uses `cut`, `width`)*. A *soft* cosine crossover at "
        "`cut` of half-width `width`: below → A, above → B, smoothly ramped. Gentler than the "
        "hard swap. A flat blend = lerp, so the real knob is *where* the crossover sits (`cut`).",
    "two-prompt lerp":
        "**Lerp** *(needs Prompt B; uses `alpha`)*. Plain `(1-alpha)·A + alpha·B` in token space "
        "(no FFT) — the non-spectral **baseline** the spectral merges aim to beat.",
    "per-object band gain":
        "**Per-object band gain** *(needs Object phrase; uses `band`, `gain`)*. E32: find the "
        "object phrase's token span in Prompt A, FFT **only those tokens**, scale the chosen "
        f"half by `gain` (median split {OBJ_CUT}; DC untouched), stitch back. Edits **one "
        "object's** frequencies in isolation — boost raises its prominence, cut lowers it. "
        "The phrase must appear verbatim in Prompt A; the caption shows its span + bins.",
}

INTRO_MD = """\
### How it works
A prompt is encoded into a **token-sequence embedding** `E` of shape `(1, L, 4096)`
(`L` real tokens × 4096 dims). Every operator takes the FFT **along the token axis**
(one 1-D transform per channel), edits the spectrum, inverts it, and feeds the result to
Flux. Normalised frequency runs `0 → 1`:
- **DC (0)** = the average across tokens — the prompt's overall meaning / level.
- **low** = slow variation across tokens → coarse, global structure ("gist").
- **high** = fast token-to-token variation → sharp local detail / style.

Left image = **baseline** (unmodified), right image = your **edit**, at the same seed, so
any difference is purely the operation. Pick an operation to see what it does and which
knobs it uses.
"""

KNOBS_MD = """\
- **cut** — normalised frequency `0..1` (`1` = Nyquist). Sets the single split/crossover
  point; only low/high-pass and the two-prompt band-swap/blend ops use it.
- **band low / band high** — the two edges of the `[lo,hi]` frequency range an operator
  acts on (gain / notch / phase-keep / phase-gain / per-object). Pick any sub-band directly:
  `[0,0.3]` coarse, `[0.7,1]` fine, or a middle band — no low/high toggle needed.
- **gain** — multiplier for the selected band: `>1` amplify, `1` identity, `<1` attenuate,
  `0` = remove. DC is never scaled, so the prompt's global level is preserved.
- **blend width** — half-width of the soft A↔B crossover (band-blend only).
- **lerp alpha** — interpolation weight toward Prompt B (lerp only).
- **seed** — fix it to compare edits fairly (baseline is cached per seed).
- **steps / guidance / size** — generation quality vs speed; smaller `size` is faster and
  uses less VRAM.
"""

# ---------------------------------------------------------------------------
# model (loaded once at startup)
# ---------------------------------------------------------------------------

PIPE = None


def load_pipe():
    from diffusers import (FluxPipeline, FluxTransformer2DModel, BitsAndBytesConfig)
    qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.bfloat16)
    tr = FluxTransformer2DModel.from_pretrained(
        REPO, subfolder="transformer", quantization_config=qc,
        torch_dtype=torch.bfloat16)
    pipe = FluxPipeline.from_pretrained(REPO, transformer=tr, torch_dtype=torch.bfloat16)
    pipe.set_progress_bar_config(disable=True)
    # keep text encoders + VAE on GPU (unlike the experiment loaders, which drop encoders)
    pipe.text_encoder.to("cuda")
    pipe.text_encoder_2.to("cuda")
    pipe.vae.to("cuda")
    print(f"[demo] {REPO} loaded (encoders kept on GPU)", flush=True)
    return pipe


# ---------------------------------------------------------------------------
# encode / generate
# ---------------------------------------------------------------------------

def encode(prompt):
    """prompt -> (pe_cpu (1,512,4096), ppe_cpu (1,4096), L real tokens)."""
    with torch.no_grad():
        pe, ppe, _ = PIPE.encode_prompt(
            prompt=prompt, prompt_2=prompt, device="cuda",
            num_images_per_prompt=1, max_sequence_length=MODEL["max_seq"])
    tok = PIPE.tokenizer_2(prompt, max_length=MODEL["max_seq"], truncation=True, return_tensors="pt")
    L = int(tok.attention_mask.sum())
    return pe.cpu(), ppe.cpu(), L


def generate(pe, ppe, seed, steps, guidance, size):
    """One true-CFG=1 generation from (possibly edited) embeddings -> PIL."""
    with torch.no_grad():
        img = PIPE(prompt_embeds=pe.cuda(), pooled_prompt_embeds=ppe.cuda(),
                   height=size, width=size, guidance_scale=guidance,
                   true_cfg_scale=1.0, num_inference_steps=int(steps),
                   generator=torch.Generator("cuda").manual_seed(int(seed))).images[0]
    return img


def _phase_only(E):
    F = torch.fft.rfft(E.float(), dim=1)
    return torch.fft.irfft(torch.polar(torch.ones_like(F.abs()), torch.angle(F)),
                           n=E.shape[1], dim=1).to(E.dtype)


def _mag_only(E):
    F = torch.fft.rfft(E.float(), dim=1)
    return torch.fft.irfft(torch.polar(F.abs(), torch.zeros_like(F.abs())),
                           n=E.shape[1], dim=1).to(E.dtype)


# ---------------------------------------------------------------------------
# op dispatch -> (edited prompt_embeds, edited pooled, description)
# ---------------------------------------------------------------------------

def clip_pooled_op(prompt, fn):
    """Apply a token-axis op `fn` to CLIP's last hidden state (the layer feeding the pool),
    then re-pool at the EOS token -> a globally-modified pooled embedding. Reproduces Flux's
    pooler_output exactly when fn is identity. `fn` takes/returns a (1, L, 768) tensor."""
    te, tok = PIPE.text_encoder, PIPE.tokenizer
    ids = tok(prompt, padding="max_length", max_length=PIPE.tokenizer_max_length,
              truncation=True, return_tensors="pt").input_ids.to(te.device)
    with torch.no_grad():
        lhs = te(ids, output_hidden_states=False).last_hidden_state    # (1,77,768) post-final-LN
        eos = int(ids.to(torch.int).argmax(dim=-1)[0])                 # EOS = highest token id
        mod = TS.apply_on_span(fn, lhs.float(), eos + 1).to(lhs.dtype)  # op over real tokens only
        pooled = mod[torch.arange(mod.shape[0], device=mod.device), eos]
    return pooled.cpu()


def _range(p):
    """Read the [lo,hi] band-edge sliders, ordered so lo<=hi even if dragged past."""
    lo, hi = float(p["lo"]), float(p["hi"])
    return (hi, lo) if hi < lo else (lo, hi)


def apply_op(op, promptA, peA, ppeA, LA, peB, ppeB, LB, p):
    on_span = lambda fn, L=LA: TS.apply_on_span(fn, peA, L)
    def _ppe(fn):
        # CLIP-pooled toggle: run the SAME token-axis op on CLIP's pre-pool sequence and
        # re-pool at EOS, so the global pooled vector gets a matching change (else unchanged).
        return clip_pooled_op(promptA, fn) if p.get("clip_pool") else ppeA
    if op == "baseline":
        return peA, ppeA, "baseline (unmodified)"
    if op == "low-pass":
        c = p["cut"]; fn = lambda x: TS.band_filter_1d(x, 0.0, c)
        return on_span(fn), _ppe(fn), f"low-pass keep [0,{c:.2f}]"
    if op == "high-pass":
        c = p["cut"]; fn = lambda x: TS.band_filter_1d(x, c, 1.0, keep_dc=True)
        return on_span(fn), _ppe(fn), f"high-pass keep [{c:.2f},1]+DC"
    if op == "band gain":
        lo, hi = _range(p); g = p["gain"]; fn = lambda x: TS.band_gain_1d(x, lo, hi, g)
        return on_span(fn), _ppe(fn), f"band [{lo:.2f},{hi:.2f}] x{g:.2f}"
    if op == "notch":
        lo, hi = _range(p); fn = lambda x: TS.band_notch_1d(x, lo, hi)
        return on_span(fn), _ppe(fn), f"notch band [{lo:.2f},{hi:.2f}]"
    if op == "phase-only":
        return on_span(_phase_only), _ppe(_phase_only), "phase-only (magnitude=1)"
    if op == "mag-only":
        return on_span(_mag_only), _ppe(_mag_only), "mag-only (phase=0)"
    if op == "phase band-keep":
        lo, hi = _range(p); fn = lambda x: TS.band_phase_filter_1d(x, lo, hi)
        return on_span(fn), _ppe(fn), \
            f"phase kept in [{lo:.2f},{hi:.2f}], =0 elsewhere (mag kept)"
    if op == "phase gain":
        lo, hi = _range(p); g = p["gain"]; fn = lambda x: TS.band_phase_gain_1d(x, lo, hi, g)
        return on_span(fn), _ppe(fn), \
            f"phase angle x{g:.2f} in band [{lo:.2f},{hi:.2f}] (mag kept)"
    if op in TWO_PROMPT:
        if peB is None:
            raise ValueError("enter prompt B for a two-prompt op")
        Lm = min(LA, LB)
        if op == "two-prompt band-swap":
            c = p["cut"]
            return TS.apply_on_span(lambda x: TS.band_swap_1d(x, peB[:, :Lm], c), peA, Lm), \
                ppeA, f"low(A)+high(B) swap @cut {c:.2f}"
        if op == "two-prompt band-blend":
            c = p["cut"]; w = p["width"]
            return TS.apply_on_span(lambda x: TS.band_blend_1d(x, peB[:, :Lm], c, w), peA, Lm), \
                ppeA, f"A<->B blend @cut {c:.2f} width {w:.2f}"
        a = p["alpha"]  # lerp
        peN = TS.apply_on_span(lambda x: TS.lerp_embeds(x, peB[:, :Lm], a), peA, Lm)
        return peN, TS.lerp_embeds(ppeA, ppeB, a), f"lerp alpha {a:.2f} (A->B)"
    if op == "per-object band gain":
        phrase = (p["object_phrase"] or "").strip()
        if not phrase:
            raise ValueError("enter an object phrase (must appear verbatim in prompt A)")
        a, b = phrase_span(PIPE.tokenizer_2, promptA, phrase, LA)
        lo, hi = _range(p)
        g = p["gain"]; bins = (b - a) // 2 + 1
        return TS.apply_on_subspan(lambda x: TS.band_gain_1d(x, lo, hi, g), peA, a, b), \
            ppeA, f"object '{phrase}' span [{a},{b}) {bins} bins · band [{lo:.2f},{hi:.2f}] x{g:.2f}"
    raise ValueError(f"unknown op {op}")


# ===========================================================================
# LATENT modulation (E36): 2D-radial FFT ops on the Flux latent, during generation
# ===========================================================================

LAT_OPS = [
    "baseline", "low-pass", "high-pass", "band gain", "notch",
    "phase-only", "mag-only", "phase band-keep", "quantize phase",
    "SBN→cfg1", "SBN→real", "band modulate", "global power", "colored-noise init",
    "restyle→B", "SBN blend A+B", "hybrid (low A / high B)", "phase-swap (A phase / B mag)",
]
LAT_SCHEDULES = ["every", "early", "late", "last"]
LAT_PERSTEP = {"low-pass", "high-pass", "band gain", "notch", "phase-only", "mag-only",
               "phase band-keep", "quantize phase", "SBN→cfg1", "SBN→real", "band modulate",
               "global power", "restyle→B", "SBN blend A+B"}
LAT_TWO_PROMPT = {"restyle→B", "SBN blend A+B", "hybrid (low A / high B)",
                  "phase-swap (A phase / B mag)"}
LAT_OFFLINE = {"hybrid (low A / high B)", "phase-swap (A phase / B mag)"}
LAT_NEEDS_CUT = {"low-pass", "high-pass", "band modulate",
                 "hybrid (low A / high B)", "phase-swap (A phase / B mag)"}
LAT_NEEDS_RANGE = {"band gain", "notch", "phase band-keep"}
LAT_NEEDS_GAIN = {"band gain", "band modulate"}
LAT_NEEDS_QK = {"quantize phase"}
LAT_NEEDS_STRENGTH = {"restyle→B"}
LAT_NEEDS_SCALE = {"global power"}

LAT_INTRO_MD = """\
### How latent modulation works
Each generation step the model holds a **latent image** `(16, H/8, W/8)`. We FFT its two
**spatial** axes and edit the **radial** spectrum (DC at the centre = the slow, global
structure; outer rings = fine detail/texture), then continue denoising. Normalised radial
frequency runs `0 → 1`. Unlike the token tab (edit once, up front), here the op runs *inside*
the loop — so the **schedule** matters:
- **every** — clamp/edit at every step (strongest, e.g. classic SBN).
- **early / late** — only the first / last third of steps.
- **last** — only the final step (gentle; the E23 real-SBN regime).

Left = baseline, right = your edit, same seed. `SBN→…` clamps the latent's per-band power
toward a target spectrum (a cfg=1 proxy, or real photos); `restyle/blend/hybrid/phase-swap`
mix a **second** prompt B in frequency space.
"""

LAT_HELP = {
    "baseline": "**Baseline** — unmodified generation (left image).",
    "low-pass": "**Low-pass** *(cut, schedule)*. Keep radial freqs `[0,cut]` (+DC), zero the rest — slow global structure only.",
    "high-pass": "**High-pass** *(cut, schedule)*. Keep `[cut,1]`+DC — fine detail/texture only.",
    "band gain": "**Band gain** *(cut, band, gain, schedule)*. Scale one radial side by `gain` (DC kept at 1).",
    "notch": "**Notch** *(cut, band, schedule)*. Zero one radial band, keep the rest.",
    "phase-only": "**Phase-only** *(schedule)*. Keep the latent's spatial phase, flatten magnitude (Oppenheim-Lim) — layout, no spectral envelope.",
    "mag-only": "**Mag-only** *(schedule)*. Keep magnitude, randomise phase — texture/palette, no layout.",
    "phase band-keep": "**Phase band-keep** *(cut, band, schedule)*. Keep phase only in one radial band (magnitude kept everywhere).",
    "quantize phase": "**Quantize phase** *(k, schedule)*. Snap the spatial phase to `k` levels (magnitude kept).",
    "SBN→cfg1": "**SBN → cfg=1** *(schedule)*. Clamp each radial band's power toward a cfg=1 reference of the SAME prompt — the E8/E16 de-saturation lever. (Records the reference on first use.)",
    "SBN→real": "**SBN → real** *(schedule)*. Clamp band power toward REAL-photo statistics (E23, +aesthetic). Needs results/e10/real_latents.pt.",
    "band modulate": "**Band modulate** *(cut, gain, schedule)*. Low/high-band power tilt (E9) split at **cut**: boosts the high band by **gain** and cuts the low band by 1/gain (gain<1 reverses it). gain=1 is a no-op.",
    "global power": "**Global power** *(scale, schedule)*. Scale the whole latent (Parseval) — changes contrast/total power, not spectral shape.",
    "colored-noise init": "**Colored-noise init**. Shape the INITIAL noise's radial spectrum toward natural/real statistics, then denoise normally (E20).",
    "restyle→B": "**Restyle → B** *(strength, schedule; needs Prompt B)*. Drive A's per-band power toward prompt B's spectrum, keeping A's phase/layout — B's palette/texture on A's content.",
    "SBN blend A+B": "**SBN blend A+B** *(schedule; needs Prompt B)*. Clamp toward the geometric-mean spectrum of A's and B's references (E22 two-prompt SBN).",
    "hybrid (low A / high B)": "**Hybrid** *(cut; needs Prompt B)*. OFFLINE: generate both, take A's full spectrum below `cut` + B's above — coarse A, fine B (Oliva hybrid).",
    "phase-swap (A phase / B mag)": "**Phase-swap** *(cut; needs Prompt B)*. OFFLINE: A's phase (layout) with B's magnitude (style), swapped at `cut` (E18).",
}

_REF_CACHE = {}            # (prompt, steps, size) -> {"band": (steps,16,nb)}
_REAL_BAND = {}            # nb -> (16,nb) real-photo band power (or None)


def _flux_unpack(size):
    from diffusers import FluxPipeline
    return lambda packed: FluxPipeline._unpack_latents(packed, size, size, PIPE.vae_scale_factor)


def _flux_pack():
    from diffusers import FluxPipeline
    return lambda lat: FluxPipeline._pack_latents(lat, 1, lat.shape[1], lat.shape[2], lat.shape[3])


def decode_latent(lat):
    """Decode an unpacked (1,16,Hl,Wl) Flux latent to a PIL image (for offline ops)."""
    v = PIPE.vae
    with torch.no_grad():                        # VAE params require grad; without this the
        z = (lat.to(v.dtype) / v.config.scaling_factor) + v.config.shift_factor
        img = v.decode(z, return_dict=False)[0]  # decoded tensor carries grad and postprocess's
    return PIPE.image_processor.postprocess(img, output_type="pil")[0]  # .numpy() then errors


class _RecLast:
    def __call__(self, p, i, t, kw):
        self.last = kw["latents"]; return {}


class _RecBand:
    def __init__(self, unpack, idx_holder, steps):
        self.unpack, self.idx_holder, self.band = unpack, idx_holder, [None] * steps
    def __call__(self, p, i, t, kw):
        lat = self.unpack(kw["latents"]).float()
        if self.idx_holder[0] is None:
            self.idx_holder[0] = band_index_map(lat.shape[-2], lat.shape[-1], N_BINS, lat.device)
        F2 = (torch.fft.fft2(lat).abs() ** 2).mean(0)
        self.band[i] = band_power(F2, self.idx_holder[0], N_BINS).cpu()
        return {}


def _final_latent(pe, ppe, seed, steps, guidance, size):
    """Generate and return the final unpacked (1,16,Hl,Wl) fp32 latent (for offline ops)."""
    rec = _RecLast()
    with torch.no_grad():
        PIPE(prompt_embeds=pe.cuda(), pooled_prompt_embeds=ppe.cuda(), height=size, width=size,
             guidance_scale=guidance, true_cfg_scale=1.0, num_inference_steps=int(steps),
             generator=torch.Generator("cuda").manual_seed(int(seed)),
             callback_on_step_end=rec).images
    return _flux_unpack(size)(rec.last).float()


def flux_reference(prompt, steps, size, ref_seeds=2):
    """cfg=1 (guidance=1.0) per-step band-power reference for `prompt`, cached."""
    key = (prompt, int(steps), int(size))
    if key in _REF_CACHE:
        return _REF_CACHE[key]
    pe, ppe, _ = _encode_cached(prompt)
    unpack = _flux_unpack(size)
    acc = None
    for s in range(ref_seeds):
        idx_holder = [None]
        rec = _RecBand(unpack, idx_holder, int(steps))
        with torch.no_grad():
            PIPE(prompt_embeds=pe.cuda(), pooled_prompt_embeds=ppe.cuda(), height=size, width=size,
                 guidance_scale=1.0, true_cfg_scale=1.0, num_inference_steps=int(steps),
                 generator=torch.Generator("cuda").manual_seed(7000 + s),
                 callback_on_step_end=rec).images
        st = torch.stack([b for b in rec.band])
        acc = st if acc is None else acc + st
    ref = {"band": acc / ref_seeds}
    _REF_CACHE[key] = ref
    return ref


def flux_real_band(nb=N_BINS):
    """Real-photo (16, nb) band power from E10's Flux latents, or None if unavailable."""
    if nb in _REAL_BAND:
        return _REAL_BAND[nb]
    rb = None
    if os.path.exists(REAL_LATENTS):
        lats = torch.load(REAL_LATENTS, weights_only=True)
        idx = band_index_map(lats.shape[-2], lats.shape[-1], nb, "cuda")
        F2 = (torch.fft.fft2(lats.cuda().float()).abs() ** 2).mean(0)
        rb = band_power(F2, idx, nb).cpu()
    _REAL_BAND[nb] = rb
    return rb


def _latent_op_fn(op, p, ctx):
    """op_fn(lat (B,16,Hl,Wl), step_i) -> lat for the per-step / init ops. idx_map is built
    from the latent shape (Flux size varies)."""
    nb = N_BINS

    def idx_of(lat):
        return band_index_map(lat.shape[-2], lat.shape[-1], nb, lat.device)

    if op == "low-pass":
        c = p["cut"]; return lambda lat, i: L.band_filter_2d(lat, 0.0, c)
    if op == "high-pass":
        c = p["cut"]; return lambda lat, i: L.band_filter_2d(lat, c, 1.0, keep_dc=True)
    if op == "band gain":
        lo, hi = _range(p); g = p["gain"]
        return lambda lat, i: L.band_gain_2d(lat, lo, hi, g)
    if op == "notch":
        lo, hi = _range(p)
        return lambda lat, i: L.band_notch_2d(lat, lo, hi)
    if op == "phase-only":
        return lambda lat, i: L.phase_only_2d(lat)
    if op == "mag-only":
        return lambda lat, i: L.mag_only_2d(lat)
    if op == "phase band-keep":
        lo, hi = _range(p)
        return lambda lat, i: L.band_phase_filter_2d(lat, lo, hi)
    if op == "quantize phase":
        k = int(p["qk"]); return lambda lat, i: L.quantize_phase_2d(lat, k)
    if op == "global power":
        s = p["scale"]; return lambda lat, i: L.global_power(lat, s)
    if op == "band modulate":
        cb = max(1, min(nb - 1, int(round(p["cut"] * nb))))   # split band index from cut slider
        g = max(float(p["gain"]), 1e-3)                       # tilt: hi*g, lo/g (g=2 == old 0.5,2.0)
        return lambda lat, i: L.band_modulate(lat, idx_of(lat), nb, cb, 1.0 / g, g)
    if op == "SBN→cfg1":
        ref = ctx["ref_band"]
        return lambda lat, i: L.sbn_clamp(lat, ref[min(i, ref.shape[0] - 1)].to(lat.device), idx_of(lat), nb)
    if op == "SBN→real":
        rb = ctx["real_band"]
        return lambda lat, i: L.sbn_clamp(lat, rb.to(lat.device), idx_of(lat), nb)
    if op == "SBN blend A+B":
        bref = ctx["blend_band"]
        return lambda lat, i: L.sbn_clamp(lat, bref[min(i, bref.shape[0] - 1)].to(lat.device), idx_of(lat), nb)
    if op == "restyle→B":
        sb = ctx["style_band"]; st = p.get("strength", 1.0)
        return lambda lat, i: L._batched(lambda x: restyle_latent(x, sb.to(x.device), idx_of(x), nb, strength=st), lat)
    if op == "colored-noise init":
        tgt = ctx["color_band"]
        return lambda lat, i: L._batched(lambda x: color_noise(x, tgt.to(x.device), idx_of(x), nb), lat)
    raise ValueError(f"unknown latent op {op}")


def generate_latent(pe, ppe, seed, steps, guidance, size, op_fn=None, schedule="every",
                    init_fn=None):
    """One Flux generation applying op_fn on the schedule (or init_fn to pre-set the
    initial noise)."""
    unpack, pack = _flux_unpack(size), _flux_pack()
    cb = None
    latents = None
    if init_fn is not None:
        hl = size // PIPE.vae_scale_factor
        noise = torch.randn((1, 16, hl, hl), generator=torch.Generator("cuda").manual_seed(int(seed)),
                            device="cuda", dtype=torch.float32)
        latents = pack(init_fn(noise, 0)).to(torch.bfloat16)
    elif op_fn is not None:
        cb = L.LatentOpCallback(op_fn, schedule, int(steps), unpack=unpack, pack=pack)
    with torch.no_grad():
        img = PIPE(prompt_embeds=pe.cuda(), pooled_prompt_embeds=ppe.cuda(), height=size, width=size,
                   guidance_scale=guidance, true_cfg_scale=1.0, num_inference_steps=int(steps),
                   generator=torch.Generator("cuda").manual_seed(int(seed)),
                   latents=latents, callback_on_step_end=cb).images[0]
    return img


def run_latent(promptA, promptB, op, cut, gain, lo, hi, qk, schedule, strength, scale,
               seed, steps, guidance, size):
    if not (promptA or "").strip():
        return None, None, "enter prompt A"
    size = int(size); seed = int(seed); steps = int(steps)
    peA, ppeA, LA = _encode_cached(promptA)
    bkey = (promptA, seed, steps, float(guidance), size)
    if bkey not in _BASE_CACHE:
        _BASE_CACHE[bkey] = generate(peA, ppeA, seed, steps, guidance, size)
    base = _BASE_CACHE[bkey]
    if op == "baseline":
        return base, base, "baseline (same as left)"

    need_B = op in LAT_TWO_PROMPT
    if need_B and not (promptB or "").strip():
        return base, None, "enter prompt B for a two-prompt op"
    p = dict(cut=cut, gain=gain, lo=lo, hi=hi, qk=qk, strength=strength, scale=scale)
    try:
        if op in LAT_OFFLINE:
            peB, ppeB, LB = _encode_cached(promptB)
            latA = _final_latent(peA, ppeA, seed, steps, guidance, size)
            latB = _final_latent(peB, ppeB, seed, steps, guidance, size)
            if op.startswith("hybrid"):
                rec = L.hybrid_split_2d(latA, latB, cut)
                desc = f"hybrid: A spectrum < cut {cut:.2f}, B above"
            else:
                rec = L.phase_swap_2d(latA, latB, cut, mag_from="B")
                desc = f"phase-swap: A phase < cut {cut:.2f}, B magnitude"
            return base, decode_latent(rec), desc

        ctx = {}
        if op == "SBN→cfg1":
            ctx["ref_band"] = flux_reference(promptA, steps, size)["band"]
        elif op == "SBN→real":
            rb = flux_real_band()
            if rb is None:
                return base, None, "no results/e10/real_latents.pt — SBN→real unavailable"
            ctx["real_band"] = rb
        elif op == "restyle→B":
            ctx["style_band"] = flux_reference(promptB, steps, size)["band"][-1]
        elif op == "SBN blend A+B":
            refA = flux_reference(promptA, steps, size)
            refB = flux_reference(promptB, steps, size)
            ctx["blend_band"] = blend_references(refA, refB, 0.5)["band"]
        elif op == "colored-noise init":
            rb = flux_real_band()
            ctx["color_band"] = rb if rb is not None else flux_reference(promptA, steps, size)["band"][-1]

        op_fn = _latent_op_fn(op, p, ctx)
        if op == "colored-noise init":
            edited = generate_latent(peA, ppeA, seed, steps, guidance, size, init_fn=op_fn)
            desc = "colored-noise init (" + ("real" if flux_real_band() is not None else "cfg=1") + " target)"
        else:
            edited = generate_latent(peA, ppeA, seed, steps, guidance, size, op_fn=op_fn,
                                     schedule=schedule)
            desc = f"{op} · schedule={schedule}"
    except Exception as e:
        return base, None, f"error: {e}"
    return base, edited, desc


def _latent_visibility(op):
    import gradio as gr
    return [
        gr.update(visible=op in LAT_TWO_PROMPT),              # prompt B
        gr.update(visible=op in LAT_NEEDS_CUT),               # cut
        gr.update(visible=op in LAT_NEEDS_RANGE),             # band low edge
        gr.update(visible=op in LAT_NEEDS_RANGE),             # band high edge
        gr.update(visible=op in LAT_NEEDS_GAIN),              # gain
        gr.update(visible=op in LAT_NEEDS_QK),                # quant k
        gr.update(visible=op in LAT_PERSTEP),                 # schedule
        gr.update(visible=op in LAT_NEEDS_STRENGTH),          # strength
        gr.update(visible=op in LAT_NEEDS_SCALE),             # scale
        gr.update(value=LAT_HELP.get(op, "")),               # help
    ]


# ---------------------------------------------------------------------------
# Gradio handler (with baseline caching)
# ---------------------------------------------------------------------------

_BASE_CACHE = {}
_ENC_CACHE = {}


def _encode_cached(prompt):
    if prompt not in _ENC_CACHE:
        _ENC_CACHE[prompt] = encode(prompt)
    return _ENC_CACHE[prompt]


def run(promptA, promptB, object_phrase, op, cut, gain, lo, hi, width, alpha, clip_pool,
        seed, steps, guidance, size):
    if not (promptA or "").strip():
        return None, None, "enter prompt A"
    size = int(size); seed = int(seed); steps = int(steps)
    peA, ppeA, LA = _encode_cached(promptA)

    key = (promptA, seed, steps, float(guidance), size)
    if key not in _BASE_CACHE:
        _BASE_CACHE[key] = generate(peA, ppeA, seed, steps, guidance, size)
    base = _BASE_CACHE[key]

    peB = ppeB = LB = None
    if op in TWO_PROMPT:
        if not (promptB or "").strip():
            return base, None, "enter prompt B for a two-prompt op"
        peB, ppeB, LB = _encode_cached(promptB)

    p = dict(cut=cut, gain=gain, lo=lo, hi=hi, width=width, alpha=alpha,
             object_phrase=object_phrase, clip_pool=clip_pool)
    try:
        peN, ppeN, desc = apply_op(op, promptA, peA, ppeA, LA, peB, ppeB, LB, p)
    except Exception as e:
        return base, None, f"error: {e}"

    if op == "baseline":
        return base, base, "baseline (same as left)"
    if clip_pool and op in CLIP_POOLABLE:
        desc += " · +CLIP-pooled (global)"
    edited = generate(peN, ppeN, seed, steps, guidance, size)
    return base, edited, desc


def _visibility(op):
    import gradio as gr
    return [
        gr.update(visible=op in TWO_PROMPT),                 # prompt B
        gr.update(visible=op == "per-object band gain"),     # object phrase
        gr.update(visible=op in NEEDS_CUT),                  # cut
        gr.update(visible=op in NEEDS_RANGE),                # band low edge
        gr.update(visible=op in NEEDS_RANGE),                # band high edge
        gr.update(visible=op in NEEDS_GAIN),                 # gain
        gr.update(visible=op == "two-prompt band-blend"),    # width
        gr.update(visible=op == "two-prompt lerp"),          # alpha
        gr.update(visible=op in CLIP_POOLABLE),              # clip-pooled toggle
        gr.update(value=HELP.get(op, "")),                   # help text
    ]


def _token_tab():
    import gradio as gr
    with gr.Accordion("How it works  ·  what the knobs mean", open=False):
        gr.Markdown(INTRO_MD)
        gr.Markdown("**Controls**\n" + KNOBS_MD)
    with gr.Row():
        with gr.Column(scale=1):
            promptA = gr.Textbox(label="Prompt A", value="a fluffy orange tabby cat and a sleeping golden retriever dog")
            promptB = gr.Textbox(label="Prompt B (two-prompt ops)", visible=False,
                                 info="Second prompt; its band is mixed into A.",
                                 value="a red sports car on a mountain road")
            object_phrase = gr.Textbox(label="Object phrase (must appear verbatim in Prompt A)",
                                       visible=False,
                                       info="Edits only this object's token span.",
                                       value="a fluffy orange tabby cat")
            op = gr.Dropdown(OPS, value="baseline", label="Operation")
            helpbox = gr.Markdown(HELP["baseline"])
            cut = gr.Slider(0.0, 1.0, value=0.25, step=0.01, label="cut (low/high split)",
                            info="0=DC … 1=Nyquist. WHERE the spectrum splits (not which side).",
                            visible=False)
            lo = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="band low edge",
                           info="Low edge of the [lo,hi] band to act on (0=DC … 1=Nyquist).",
                           visible=False)
            hi = gr.Slider(0.0, 1.0, value=1.0, step=0.01, label="band high edge",
                           info="High edge of the band. Operator acts on frequencies in [lo,hi].",
                           visible=False)
            gain = gr.Slider(0.0, 3.0, value=2.0, step=0.05, label="gain (x)",
                             info=">1 amplify, 1 identity, <1 attenuate, 0 remove. DC kept at 1.",
                             visible=False)
            width = gr.Slider(0.0, 0.5, value=0.15, step=0.01, label="blend width",
                              info="Half-width of the soft A↔B crossover around cut.",
                              visible=False)
            alpha = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="lerp alpha (A->B)",
                              info="0 = all A, 1 = all B (plain token-space mix).",
                              visible=False)
            clip_pool = gr.Checkbox(value=False, visible=False,
                                    label="also apply to CLIP pooled (global)",
                                    info="Run the same op on CLIP's pre-pool sequence and "
                                         "re-pool at EOS → matching change to the global vector.")
            with gr.Row():
                seed = gr.Number(value=0, precision=0, label="seed",
                                 info="Fix to compare fairly; baseline cached per seed.")
                steps = gr.Slider(4, 28, value=MODEL["steps"], step=1, label="steps",
                                  info="Denoising steps: more = better, slower.")
            with gr.Row():
                guidance = gr.Slider(1.0, 7.0, value=3.5, step=0.1, label="guidance",
                                     info="Flux distilled guidance (~3.5).")
                size = gr.Dropdown([512, 768, 1024], value=768, label="size",
                                   info="Pixels; smaller = faster, less VRAM.")
            go = gr.Button("Generate", variant="primary")
        with gr.Column(scale=2):
            with gr.Row():
                out_base = gr.Image(label="baseline", type="pil")
                out_edit = gr.Image(label="edited", type="pil")
            desc = gr.Markdown()

    op.change(_visibility, op,
              [promptB, object_phrase, cut, lo, hi, gain, width, alpha, clip_pool, helpbox])
    go.click(run,
             [promptA, promptB, object_phrase, op, cut, gain, lo, hi, width, alpha, clip_pool,
              seed, steps, guidance, size],
             [out_base, out_edit, desc])


def _latent_tab():
    import gradio as gr
    with gr.Accordion("How it works  ·  what the knobs mean", open=False):
        gr.Markdown(LAT_INTRO_MD)
    with gr.Row():
        with gr.Column(scale=1):
            promptA = gr.Textbox(label="Prompt A", value="a fluffy orange tabby cat sitting on a windowsill")
            promptB = gr.Textbox(label="Prompt B (two-prompt ops)", visible=False,
                                 info="Second prompt for restyle / blend / hybrid / phase-swap.",
                                 value="a stained glass window, vivid colors")
            op = gr.Dropdown(LAT_OPS, value="baseline", label="Operation")
            helpbox = gr.Markdown(LAT_HELP["baseline"])
            schedule = gr.Radio(LAT_SCHEDULES, value="every", label="schedule (when it fires)",
                                info="every / early / late third / last step only.", visible=False)
            cut = gr.Slider(0.0, 1.0, value=0.3, step=0.01, label="cut (radial low/high split)",
                            info="0=DC (centre) … 1=corner. WHERE the radial spectrum splits.",
                            visible=False)
            lo = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="band low edge",
                           info="Low edge of the radial [lo,hi] band (0=DC/centre … 1=corner).",
                           visible=False)
            hi = gr.Slider(0.0, 1.0, value=1.0, step=0.01, label="band high edge",
                           info="High edge of the radial band. Acts on radii in [lo,hi].",
                           visible=False)
            gain = gr.Slider(0.0, 3.0, value=2.0, step=0.05, label="gain (x)",
                             info=">1 amplify, 1 identity, <1 attenuate. DC kept at 1.", visible=False)
            qk = gr.Slider(2, 16, value=8, step=1, label="phase levels k", visible=False,
                           info="Quantize spatial phase to k levels (fewer = harsher).")
            strength = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="restyle strength",
                                 info="0 = keep A's spectrum, 1 = fully adopt B's per-band power.",
                                 visible=False)
            scale = gr.Slider(0.2, 2.0, value=0.7, step=0.05, label="power scale",
                              info="Multiply the whole latent (contrast / total power).", visible=False)
            with gr.Row():
                seed = gr.Number(value=0, precision=0, label="seed",
                                 info="Fix to compare fairly; baseline cached per seed.")
                steps = gr.Slider(4, 28, value=MODEL["steps"], step=1, label="steps",
                                  info="Denoising steps: more = better, slower.")
            with gr.Row():
                guidance = gr.Slider(1.0, 7.0, value=3.5, step=0.1, label="guidance",
                                     info="Flux distilled guidance (~3.5).")
                size = gr.Dropdown([512, 768, 1024], value=768, label="size",
                                   info="Pixels; smaller = faster, less VRAM.")
            go = gr.Button("Generate", variant="primary")
        with gr.Column(scale=2):
            with gr.Row():
                out_base = gr.Image(label="baseline", type="pil")
                out_edit = gr.Image(label="edited", type="pil")
            desc = gr.Markdown()

    op.change(_latent_visibility, op,
              [promptB, cut, lo, hi, gain, qk, schedule, strength, scale, helpbox])
    go.click(run_latent,
             [promptA, promptB, op, cut, gain, lo, hi, qk, schedule, strength, scale,
              seed, steps, guidance, size],
             [out_base, out_edit, desc])


def build_ui():
    import gradio as gr
    with gr.Blocks(title="Spectral image editing — token & latent") as demo:
        gr.Markdown("# Spectral image editing (E24–E36)\n"
                    "Two ways to steer Flux in frequency space. **Token modulation** edits the "
                    "text embedding's token-axis spectrum once up front; **Latent modulation** edits "
                    "the image latent's 2D radial spectrum *during* generation. Left = baseline, "
                    "right = edit, same seed.")
        with gr.Tabs():
            with gr.TabItem("Token modulation"):
                _token_tab()
            with gr.TabItem("Latent modulation"):
                _latent_tab()
    return demo


def _patch_gradio_schema_bug():
    """Work around gradio_client's 'argument of type bool is not iterable' crash on
    bool JSON-schemas (additionalProperties: true/false). It breaks /config, which makes
    launch()'s localhost self-check 500 and wrongly report 'localhost not accessible'."""
    import gradio_client.utils as u
    _gt = u.get_type
    def get_type(schema):
        return "Any" if not isinstance(schema, dict) else _gt(schema)
    u.get_type = get_type
    _js = u._json_schema_to_python_type
    def _json_schema_to_python_type(schema, defs=None):
        return "Any" if isinstance(schema, bool) else _js(schema, defs)
    u._json_schema_to_python_type = _json_schema_to_python_type


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Spectral image-editing demo (token + latent).")
    ap.add_argument("--model", choices=list(MODELS), default="flux-dev",
                    help="which model to load; only this one goes on the GPU "
                         "(flux-schnell = lighter/faster, ~4 steps, ungated).")
    ap.add_argument("--port", type=int, default=7860, help="Gradio server port.")
    ap.add_argument("--share", action="store_true", help="create a public Gradio link.")
    a = ap.parse_args()
    MODEL = MODELS[a.model]; REPO = MODEL["repo"]
    print(f"[demo] model={a.model} ({REPO})", flush=True)
    _patch_gradio_schema_bug()
    PIPE = load_pipe()
    build_ui().launch(server_name="0.0.0.0", server_port=a.port, share=a.share,
                      show_api=False, show_error=True)
