"""Interactive browser demo for spectral image editing — THREE modes (E24-E37 toolkit).

A Gradio app; ONE model is loaded (chosen by --model). The default sd3.5-medium drives the
Velocity tab (true CFG); the Token/Latent tabs need a Flux model (--model flux-dev).

  • VELOCITY modulation — edit the CFG velocity `v_w = v_∅ + w(v_c − v_∅)` toward the
    unconditional `v_∅` DURING generation, via an SD3.5 scheduler.step override (E37):
    per-bin magnitude transplant / per-band power match (normalize→cfg1) or band amplify/
    reduce, on a band [lo,hi] and a step interval. Needs real CFG → SD3.5 (not Flux).
  • TOKEN modulation  — FFT Flux's T5 token-sequence embedding and edit the spectrum ONCE
    before generation (E24/E30/E32/E35: low/high-pass, band gain, notch, phase/mag,
    two-prompt swap/blend/lerp, per-object band gain).
  • LATENT modulation — FFT the 2D diffusion latent and edit its radial spectrum DURING
    generation via a step-end callback (E8-E23/E36). The extra knob here is the SCHEDULE —
    *when* the op fires (every step / early / late / last step). Operators: low/high-pass,
    band gain, notch, phase-only/mag-only, phase band-keep, quantize phase, SBN→cfg1 /
    SBN→real / SBN→universal / band-modulate / global-power, colored-noise init, restyle→B, SBN-blend A+B,
    and the offline two-latent hybrid / phase-swap.

The tabs are thin wrappers around the ops in `velocity_spectral_ops.py` (velocity axis),
`text_spectral_ops.py` (token axis) and `latent_spectral_ops.py` / `spectral_ops.py` /
`style_ops.py` (latent axis); the only new machinery is keeping the text encoders + VAE
loaded so arbitrary prompts encode on the fly.

Run:  python experiments/spectral_demo.py   (then ssh -L 7860:localhost:7860 <host>)
Model: SD3.5-medium (bf16, ~24GB) by default; --model flux-dev for the Flux tabs.

Run with uv (auto-builds/caches an env from the inline deps below, incl. a CUDA torch):
    uv run experiments/spectral_demo.py
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
#     "gradio_rangeslider==0.0.8",
#     "imageio",
#     "imageio-ffmpeg",
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
import velocity_spectral_ops as VEL     # velocity axis (E37; SD3.5 real-CFG step override)
from spectral_ops import band_index_map, band_power
from style_ops import restyle_latent, color_noise, blend_references

REAL_LATENTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "results", "e10", "real_latents.pt")   # Flux-VAE real photos (E10)
UNIVERSAL_REF = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "results", "e9", "universal_ref.pt")  # E9 universal cfg=1 band ref
N_BINS = 24

# In-memory cache for each tab's LEFT/baseline image, so tweaking only the method knobs
# reuses the unchanged baseline. Key on the baseline-determining inputs (prompt/image/seed/
# steps/guidance), NOT the method knobs (op/cut/strength/...). See cached_baseline callers.
_BASE_CACHE = {}
def _img_key(pil):
    return None if pil is None else hash(pil.convert("RGB").resize((256, 256)).tobytes())
def cached_baseline(key, compute):
    img = _BASE_CACHE.get(key)
    if img is None:
        img = compute()
        _BASE_CACHE[key] = img
    return img

# Model registry -- only ONE entry is ever loaded (chosen by --model), so a single model
# sits on the GPU. `kind` selects the backend: "flux" = guidance-distilled Flux (Token /
# Latent tabs), "sd3" = Stable Diffusion 3.5 with TRUE classifier-free guidance (the
# Velocity tab needs a real v_uncond, which Flux's distillation does not expose). flux-schnell
# is the lighter/faster Flux (distilled to ~4 steps, ungated).
MODELS = {
    "sd3.5-medium": {"repo": "stabilityai/stable-diffusion-3.5-medium", "kind": "sd3",
                     "max_seq": 256, "steps": 28, "guidance": 4.5},
    "flux-dev":     {"repo": "black-forest-labs/FLUX.1-dev",     "kind": "flux", "max_seq": 512, "steps": 16, "guidance": 3.5},
    "flux-schnell": {"repo": "black-forest-labs/FLUX.1-schnell", "kind": "flux", "max_seq": 256, "steps": 4,  "guidance": 3.5},
    "ltx":          {"repo": "Lightricks/LTX-Video", "kind": "ltx", "max_seq": 128, "steps": 24, "guidance": 10.0},
}
MODEL = MODELS["sd3.5-medium"]         # overridden in __main__ from --model
MODEL_NAME = "sd3.5-medium"            # the --model key, recorded in saved runs
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
- **timesteps to intervene** — fraction of the denoising schedule `[start, end]` on which
  the token edit is active (`[0,1]` = the whole run, identical to editing once up front;
  shrink it to apply the edit early-only or late-only). Implemented by swapping the edited
  vs. unedited embedding mid-loop; the pooled/global vector stays edited throughout.
- **seed** — fix it to compare edits fairly (baseline is cached per seed).
- **steps / guidance / size** — generation quality vs speed; smaller `size` is faster and
  uses less VRAM.
"""

# ---------------------------------------------------------------------------
# model (loaded once at startup)
# ---------------------------------------------------------------------------

PIPE = None


def load_pipe():
    if MODEL["kind"] == "ltx":
        # LTX-Video (E45): FlowAlign on a real video model. Reuses the experiment's LTX core.
        import e45_ltx_flowalign as ltxcore
        pipe = ltxcore.load_ltx()
        pipe.set_progress_bar_config(disable=True)
        print(f"[demo] {REPO} loaded (LTX-Video, FlowAlign video tab)", flush=True)
        return pipe
    if MODEL["kind"] == "sd3":
        # SD3.5-medium (2.5B) fits a 24GB GPU bf16 GPU-resident (transformer + T5-XXL +
        # CLIPx2 + VAE); real CFG, so v_uncond is available for the Velocity tab. (e17_sd35.)
        from diffusers import StableDiffusion3Pipeline
        pipe = StableDiffusion3Pipeline.from_pretrained(REPO, torch_dtype=torch.bfloat16)
        pipe.to("cuda")
        pipe.set_progress_bar_config(disable=True)
        print(f"[demo] {REPO} loaded (SD3.5, true CFG)", flush=True)
        return pipe
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


def gen_sd3_demo(prompt, seed, steps, guidance, size, step_override=None):
    """One SD3.5 generation (real CFG) -> PIL. With `step_override` the e17_sd35.gen_sd3
    interception is installed: `transformer.forward` is patched to record the batched
    [uncond, cond] output, and `scheduler.step` to run
    step_override(records, model_output, sample) -> new velocity BEFORE the Euler step."""
    orig_fwd = PIPE.transformer.forward
    orig_step = PIPE.scheduler.step
    if step_override is not None:
        records = []

        def fwd(*a, **k):
            o = orig_fwd(*a, **k)
            s = o[0] if isinstance(o, (tuple, list)) else o.sample
            records.append(s.detach())
            return o

        def step(model_output, timestep, sample, *a, **k):
            mo = step_override(records, model_output, sample)
            return orig_step(mo, timestep, sample, *a, **k)

        PIPE.transformer.forward = fwd
        PIPE.scheduler.step = step
    try:
        with torch.no_grad():
            img = PIPE(prompt=prompt, height=int(size), width=int(size),
                       guidance_scale=float(guidance), num_inference_steps=int(steps),
                       negative_prompt="",
                       generator=torch.Generator("cuda").manual_seed(int(seed))).images[0]
    finally:
        PIPE.transformer.forward = orig_fwd
        PIPE.scheduler.step = orig_step
    return img


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


def generate(pe, ppe, seed, steps, guidance, size, pe_base=None, interval=None):
    """One true-CFG=1 generation from (possibly edited) embeddings -> PIL.

    With `pe_base` + `interval=(i_lo,i_hi)` the token edit is TIMESTEP-GATED: a
    TS.EmbedIntervalCallback swaps prompt_embeds between the edited `pe` (inside the
    inclusive step window) and `pe_base` (outside it). Step 0 uses the edited embeds iff
    i_lo == 0; the callback sets each subsequent step. pooled stays as passed (not gated)."""
    cb = None
    pe_step0 = pe
    if pe_base is not None and interval is not None:
        i_lo, i_hi = interval
        pe_g, pb_g = pe.cuda(), pe_base.cuda()
        cb = TS.EmbedIntervalCallback(pb_g, pe_g, i_lo, i_hi)
        pe_step0 = pe_g if i_lo <= 0 <= i_hi else pb_g
    with torch.no_grad():
        img = PIPE(prompt_embeds=pe_step0.cuda(), pooled_prompt_embeds=ppe.cuda(),
                   height=size, width=size, guidance_scale=guidance,
                   true_cfg_scale=1.0, num_inference_steps=int(steps),
                   generator=torch.Generator("cuda").manual_seed(int(seed)),
                   callback_on_step_end=cb).images[0]
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
    "SBN→cfg1", "SBN→real", "SBN→universal", "band modulate", "global power", "colored-noise init",
    "restyle→B", "SBN blend A+B", "hybrid (low A / high B)", "phase-swap (A phase / B mag)",
]
LAT_SCHEDULES = ["every", "early", "late", "last", "interval"]
LAT_PERSTEP = {"low-pass", "high-pass", "band gain", "notch", "phase-only", "mag-only",
               "phase band-keep", "quantize phase", "SBN→cfg1", "SBN→real", "SBN→universal",
               "band modulate", "global power", "restyle→B", "SBN blend A+B"}
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
- **interval** — a custom inclusive step window via the *timesteps to intervene* slider
  (the free-form version of early/late; mirrors the Velocity tab).

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
    "SBN→universal": "**SBN → universal** *(schedule)*. Clamp each band's power toward the E9 **universal** cfg=1 reference (one per-step profile averaged across prompt classes) — a precomputed magnitude target, so NO per-prompt cfg=1 pass is needed. Needs results/e9/universal_ref.pt.",
    "band modulate": "**Band modulate** *(cut, gain, schedule)*. Low/high-band power tilt (E9) split at **cut**: boosts the high band by **gain** and cuts the low band by 1/gain (gain<1 reverses it). gain=1 is a no-op.",
    "global power": "**Global power** *(scale, schedule)*. Scale the whole latent (Parseval) — changes contrast/total power, not spectral shape.",
    "colored-noise init": "**Colored-noise init**. Shape the INITIAL noise's radial spectrum toward natural/real statistics, then denoise normally (E20).",
    "restyle→B": "**Restyle → B** *(strength, schedule; needs Prompt B)*. Drive A's per-band power toward prompt B's spectrum, keeping A's phase/layout — B's palette/texture on A's content.",
    "SBN blend A+B": "**SBN blend A+B** *(schedule; needs Prompt B)*. Clamp toward the geometric-mean spectrum of A's and B's references (E22 two-prompt SBN).",
    "hybrid (low A / high B)": "**Hybrid** *(cut; needs Prompt B)*. OFFLINE: generate both, take A's full spectrum below `cut` + B's above — coarse A, fine B (Oliva hybrid).",
    "phase-swap (A phase / B mag)": "**Phase-swap** *(cut; needs Prompt B)*. OFFLINE: A's phase (layout) with B's magnitude (style), swapped at `cut` (E18).",
}


# ===========================================================================
# VELOCITY modulation (E37): edit the CFG velocity v_w toward v_uncond, SD3.5 real CFG
# ===========================================================================

VEL_OPS = ["baseline", "normalize→cfg1 (mag)", "normalize→cfg1 (band power)", "band amplify/reduce"]
VEL_MODE = {"normalize→cfg1 (mag)": "mag", "normalize→cfg1 (band power)": "band power",
            "band amplify/reduce": "gain"}
VEL_NEEDS_RANGE = set(VEL_OPS) - {"baseline"}                       # all ops act on a band
VEL_NEEDS_STRENGTH = {"normalize→cfg1 (mag)", "normalize→cfg1 (band power)"}
VEL_NEEDS_GAIN = {"band amplify/reduce"}

VEL_INTRO_MD = """\
### How velocity modulation works
Stable Diffusion 3.5 is a **flow-matching** model sampled by Euler: `z_{t+1} = z_t + Δt·v`,
where the model output **`v`** is a *velocity* (the direction the latent moves this step).
Classifier-free guidance combines two velocities into

  **`v_w = v_∅ + w·(v_c − v_∅)`** — `v_∅` = *unconditional* (the natural, w=1 / "cfg=1"
  flow field), `v_c` = *conditional* (prompt-driven), `w` = guidance scale.

Higher `w` buys prompt adherence but over-amplifies certain frequency **magnitudes** (the
oversaturated, over-contrasty CFG look). The **phase** of `v_w` carries layout/composition.
So we keep `v_w`'s phase and pull its **amplitude** toward `v_∅`'s — surgically undoing CFG's
magnitude over-shoot without losing adherence. Both velocities are already computed each step,
so the only cost is a pair of FFTs (no extra model forward). This is the one-pass, *scale-correct*
cousin of the latent-tab SBN (`v_∅` is the same-step field, so its amplitude is already at the
right scale — unlike a fixed clean-image target).

Two knobs beyond the operator:
- **band [lo, hi]** — the normalised radial frequency range to act on (`0` = DC/global tone …
  `1` = corner/fine texture).
- **timesteps to intervene** — fire only on a contiguous window of steps (`[0,1]` = all steps;
  shrink it to intervene early-only or late-only).

Left = plain CFG baseline, right = your edit, same seed. Needs **real CFG** → run with
`--model sd3.5-medium` (Flux is guidance-distilled and has no `v_∅`).
"""

VEL_HELP = {
    "baseline": "**Baseline** — plain CFG generation (left image), no velocity edit.",
    "normalize→cfg1 (mag)": "**Normalize → cfg1 (mag)** *(band, strength, interval)*. Per-frequency-bin "
        "**magnitude transplant**: inside the band, replace `|v_w|` with the unconditional `|v_∅|` "
        "(keeping `v_w`'s phase), blended by **strength** (1 = full, 0 = identity). The literal "
        "\"force `v_w` to have `v_∅`'s amplitude\".",
    "normalize→cfg1 (band power)": "**Normalize → cfg1 (band power)** *(band, strength, interval)*. "
        "Gentler/coarser: match `v_w`'s **mean power per radial band** to `v_∅`'s (the E8/E16/E23 SBN "
        "operator, `psd_match`), only for bands inside the range. **strength** blends the target.",
    "band amplify/reduce": "**Band amplify/reduce** *(band, gain, interval)*. Scale `v_w`'s magnitude "
        "inside the band by **gain** (>1 amplify, <1 attenuate, DC kept) — independent of `v_∅`.",
}

_REF_CACHE = {}            # (prompt, steps, size) -> {"band": (steps,16,nb)}
_REAL_BAND = {}            # nb -> (16,nb) real-photo band power (or None)
_UNI_BAND = {}             # nb -> (steps,16,nb) E9 universal cfg=1 band ref (or None)


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


def flux_universal_band(nb=N_BINS):
    """E9 universal (prompt-agnostic) per-step cfg=1 band ref (steps,16,nb), or None.

    Averaged across prompt classes by e9_universal_ref.py -> universal_ref.pt. Clamping
    toward it skips the per-prompt cfg=1 reference pass entirely (the whole point of the
    universal ref). None if the file is missing or its bin count != nb."""
    if nb in _UNI_BAND:
        return _UNI_BAND[nb]
    band = None
    if os.path.exists(UNIVERSAL_REF):
        uni = torch.load(UNIVERSAL_REF, weights_only=True)
        b = uni["band"] if isinstance(uni, dict) else uni
        if b.shape[-1] == nb:
            band = b.float().cpu()
    _UNI_BAND[nb] = band
    return band


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
    if op == "SBN→universal":
        ub = ctx["uni_band"]
        return lambda lat, i: L.sbn_clamp(lat, ub[min(i, ub.shape[0] - 1)].to(lat.device), idx_of(lat), nb)
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
                    init_fn=None, interval=None):
    """One Flux generation applying op_fn on the schedule (or init_fn to pre-set the
    initial noise). `interval=(i_lo,i_hi)` is the inclusive step window used by
    schedule='interval'."""
    unpack, pack = _flux_unpack(size), _flux_pack()
    cb = None
    latents = None
    if init_fn is not None:
        hl = size // PIPE.vae_scale_factor
        noise = torch.randn((1, 16, hl, hl), generator=torch.Generator("cuda").manual_seed(int(seed)),
                            device="cuda", dtype=torch.float32)
        latents = pack(init_fn(noise, 0)).to(torch.bfloat16)
    elif op_fn is not None:
        cb = L.LatentOpCallback(op_fn, schedule, int(steps), unpack=unpack, pack=pack,
                                interval=interval)
    with torch.no_grad():
        img = PIPE(prompt_embeds=pe.cuda(), pooled_prompt_embeds=ppe.cuda(), height=size, width=size,
                   guidance_scale=guidance, true_cfg_scale=1.0, num_inference_steps=int(steps),
                   generator=torch.Generator("cuda").manual_seed(int(seed)),
                   latents=latents, callback_on_step_end=cb).images[0]
    return img


def run_latent(promptA, promptB, op, cut, gain, band, qk, schedule, interval, strength, scale,
               seed, steps, guidance, size):
    if MODEL["kind"] != "flux":
        return None, None, _FLUX_ONLY_NOTE
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
    p = dict(cut=cut, gain=gain, lo=band[0], hi=band[1], qk=qk, strength=strength, scale=scale)
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
        elif op == "SBN→universal":
            ub = flux_universal_band()
            if ub is None:
                return base, None, "no results/e9/universal_ref.pt — SBN→universal unavailable"
            ctx["uni_band"] = ub
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
            iv = None
            sched_desc = f"schedule={schedule}"
            if schedule == "interval":
                t_lo, t_hi = _ordered(interval)
                i_lo = int(round(t_lo * (steps - 1)))
                i_hi = int(round(t_hi * (steps - 1)))
                iv = (i_lo, i_hi)
                sched_desc = f"steps {i_lo}–{i_hi}/{steps - 1}"
            edited = generate_latent(peA, ppeA, seed, steps, guidance, size, op_fn=op_fn,
                                     schedule=schedule, interval=iv)
            desc = f"{op} · {sched_desc}"
    except Exception as e:
        return base, None, f"error: {e}"
    return base, edited, desc


def _latent_visibility(op):
    import gradio as gr
    return [
        gr.update(visible=op in LAT_TWO_PROMPT),              # prompt B
        gr.update(visible=op in LAT_NEEDS_CUT),               # cut
        gr.update(visible=op in LAT_NEEDS_RANGE),             # band range [lo,hi]
        gr.update(visible=op in LAT_NEEDS_GAIN),              # gain
        gr.update(visible=op in LAT_NEEDS_QK),                # quant k
        gr.update(visible=op in LAT_PERSTEP),                 # schedule
        gr.update(visible=op in LAT_PERSTEP),                 # interval (timesteps)
        gr.update(visible=op in LAT_NEEDS_STRENGTH),          # strength
        gr.update(visible=op in LAT_NEEDS_SCALE),             # scale
        gr.update(value=LAT_HELP.get(op, "")),               # help
    ]


# ---------------------------------------------------------------------------
# VELOCITY modulation handler (E37; SD3.5 only)
# ---------------------------------------------------------------------------

def _ordered(pair):
    """Read a RangeSlider (lo, hi), ordered so lo<=hi even if handles cross."""
    a, b = float(pair[0]), float(pair[1])
    return (b, a) if b < a else (a, b)


def run_velocity(promptA, op, band, strength, gain, interval, seed, steps, guidance, size):
    if MODEL["kind"] != "sd3":
        return None, None, ("Velocity modulation needs a real-CFG model — relaunch with "
                            "`--model sd3.5-medium`. (Flux is guidance-distilled: no `v_∅`.)")
    if not (promptA or "").strip():
        return None, None, "enter prompt A"
    size = int(size); seed = int(seed); steps = int(steps); guidance = float(guidance)
    bkey = (promptA, seed, steps, guidance, size, "vel")
    if bkey not in _BASE_CACHE:
        _BASE_CACHE[bkey] = gen_sd3_demo(promptA, seed, steps, guidance, size)
    base = _BASE_CACHE[bkey]
    if op == "baseline":
        return base, base, "baseline (plain CFG, same as left)"

    lo, hi = _ordered(band)
    t_lo, t_hi = _ordered(interval)
    i_lo = int(round(t_lo * (steps - 1)))
    i_hi = int(round(t_hi * (steps - 1)))
    mode = VEL_MODE[op]
    try:
        ov = VEL.make_velocity_override(mode, lo, hi, float(strength), float(gain),
                                        i_lo, i_hi, n_bins=N_BINS)
        edited = gen_sd3_demo(promptA, seed, steps, guidance, size, step_override=ov)
    except Exception as e:
        return base, None, f"error: {e}"
    knob = f"gain {gain:g}" if mode == "gain" else f"strength {strength:g}"
    desc = f"{op} · band [{lo:.2f},{hi:.2f}] · {knob} · steps {i_lo}–{i_hi}/{steps - 1} · w={guidance:g}"
    return base, edited, desc


def _velocity_visibility(op):
    import gradio as gr
    return [
        gr.update(visible=op in VEL_NEEDS_RANGE),             # band [lo,hi]
        gr.update(visible=op in VEL_NEEDS_STRENGTH),          # strength
        gr.update(visible=op in VEL_NEEDS_GAIN),              # gain
        gr.update(visible=op != "baseline"),                  # interval
        gr.update(value=VEL_HELP.get(op, "")),                # help
    ]


# ---------------------------------------------------------------------------
# Gradio handler (with baseline caching)
# ---------------------------------------------------------------------------

_ENC_CACHE = {}


def _encode_cached(prompt):
    if prompt not in _ENC_CACHE:
        _ENC_CACHE[prompt] = encode(prompt)
    return _ENC_CACHE[prompt]


_FLUX_ONLY_NOTE = ("This tab needs a **Flux** model — relaunch with `--model flux-dev`. "
                   "(The current model is SD3.5, loaded for the Velocity tab.)")


def run(promptA, promptB, object_phrase, op, cut, gain, band, width, alpha, clip_pool,
        interval, seed, steps, guidance, size):
    if MODEL["kind"] != "flux":
        return None, None, _FLUX_ONLY_NOTE
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

    p = dict(cut=cut, gain=gain, lo=band[0], hi=band[1], width=width, alpha=alpha,
             object_phrase=object_phrase, clip_pool=clip_pool)
    try:
        peN, ppeN, desc = apply_op(op, promptA, peA, ppeA, LA, peB, ppeB, LB, p)
    except Exception as e:
        return base, None, f"error: {e}"

    if op == "baseline":
        return base, base, "baseline (same as left)"
    if clip_pool and op in CLIP_POOLABLE:
        desc += " · +CLIP-pooled (global)"
    # timestep gate: apply the token edit only on the inclusive step window [i_lo,i_hi];
    # the full range [0,1] degenerates to the plain "edit throughout" path (no callback).
    t_lo, t_hi = _ordered(interval)
    i_lo = int(round(t_lo * (steps - 1)))
    i_hi = int(round(t_hi * (steps - 1)))
    if (i_lo, i_hi) != (0, steps - 1):
        edited = generate(peN, ppeN, seed, steps, guidance, size, pe_base=peA,
                          interval=(i_lo, i_hi))
        desc += f" · steps {i_lo}–{i_hi}/{steps - 1}"
    else:
        edited = generate(peN, ppeN, seed, steps, guidance, size)
    return base, edited, desc


def _visibility(op):
    import gradio as gr
    return [
        gr.update(visible=op in TWO_PROMPT),                 # prompt B
        gr.update(visible=op == "per-object band gain"),     # object phrase
        gr.update(visible=op in NEEDS_CUT),                  # cut
        gr.update(visible=op in NEEDS_RANGE),                # band range [lo,hi]
        gr.update(visible=op in NEEDS_GAIN),                 # gain
        gr.update(visible=op == "two-prompt band-blend"),    # width
        gr.update(visible=op == "two-prompt lerp"),          # alpha
        gr.update(visible=op in CLIP_POOLABLE),              # clip-pooled toggle
        gr.update(visible=op != "baseline"),                 # interval (timesteps)
        gr.update(value=HELP.get(op, "")),                   # help text
    ]


def _token_tab():
    import gradio as gr
    from gradio_rangeslider import RangeSlider
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
            band = RangeSlider(minimum=0.0, maximum=1.0, value=(0.5, 1.0), step=0.01,
                               label="band [low, high]", visible=False,
                               info="Drag both handles: the [lo,hi] frequency range to act on (0=DC … 1=Nyquist).")
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
            interval = RangeSlider(minimum=0.0, maximum=1.0, value=(0.0, 1.0), step=0.01,
                                   label="timesteps to intervene [start, end]", visible=False,
                                   info="Fraction of the schedule (0=first step … 1=last). [0,1] = "
                                        "edit throughout (= edit once up front); shrink to apply the "
                                        "token edit only early / late. Pooled (CLIP) stays edited "
                                        "throughout. Like the Velocity tab's interval.")
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
            out_base, out_edit = _image_pair("baseline", "edited", "pair_token")
            desc = gr.Markdown()

    op.change(_visibility, op,
              [promptB, object_phrase, cut, band, gain, width, alpha, clip_pool, interval, helpbox])
    names = ["promptA", "promptB", "object_phrase", "op", "cut", "gain", "band", "width",
             "alpha", "clip_pool", "interval", "seed", "steps", "guidance", "size"]
    inputs = [promptA, promptB, object_phrase, op, cut, gain, band, width, alpha, clip_pool,
              interval, seed, steps, guidance, size]
    go.click(run, inputs, [out_base, out_edit, desc])
    _save_load_ui("token", names, inputs, out_base, out_edit,
                  vis=(_visibility, op, [promptB, object_phrase, cut, band, gain, width, alpha,
                                         clip_pool, interval, helpbox]))


def _latent_tab():
    import gradio as gr
    from gradio_rangeslider import RangeSlider
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
                                info="every / early / late third / last step / custom interval.",
                                visible=False)
            interval = RangeSlider(minimum=0.0, maximum=1.0, value=(0.0, 1.0), step=0.01,
                                   label="timesteps to intervene [start, end]", visible=False,
                                   info="Used when schedule = interval. Fraction of the schedule "
                                        "(0=first step … 1=last); the free-form version of early/late. "
                                        "Like the Velocity tab's interval.")
            cut = gr.Slider(0.0, 1.0, value=0.3, step=0.01, label="cut (radial low/high split)",
                            info="0=DC (centre) … 1=corner. WHERE the radial spectrum splits.",
                            visible=False)
            band = RangeSlider(minimum=0.0, maximum=1.0, value=(0.5, 1.0), step=0.01,
                               label="radial band [low, high]", visible=False,
                               info="Drag both handles: the radial [lo,hi] range (0=DC/centre … 1=corner).")
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
            out_base, out_edit = _image_pair("baseline", "edited", "pair_latent")
            desc = gr.Markdown()

    op.change(_latent_visibility, op,
              [promptB, cut, band, gain, qk, schedule, interval, strength, scale, helpbox])
    names = ["promptA", "promptB", "op", "cut", "gain", "band", "qk", "schedule", "interval",
             "strength", "scale", "seed", "steps", "guidance", "size"]
    inputs = [promptA, promptB, op, cut, gain, band, qk, schedule, interval, strength, scale,
              seed, steps, guidance, size]
    go.click(run_latent, inputs, [out_base, out_edit, desc])
    _save_load_ui("latent", names, inputs, out_base, out_edit,
                  vis=(_latent_visibility, op, [promptB, cut, band, gain, qk, schedule, interval,
                                                strength, scale, helpbox]))


def _velocity_tab():
    import gradio as gr
    from gradio_rangeslider import RangeSlider
    with gr.Accordion("How it works  ·  what the knobs mean", open=False):
        gr.Markdown(VEL_INTRO_MD)
    with gr.Row():
        with gr.Column(scale=1):
            promptA = gr.Textbox(label="Prompt", value="a fluffy orange tabby cat sitting on a windowsill")
            op = gr.Dropdown(VEL_OPS, value="baseline", label="Operation")
            helpbox = gr.Markdown(VEL_HELP["baseline"])
            band = RangeSlider(minimum=0.0, maximum=1.0, value=(0.0, 1.0), step=0.01,
                               label="band [low, high]", visible=False,
                               info="Radial freq range to act on (0=DC/global tone … 1=corner/fine texture).")
            strength = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="strength (→ v∅)",
                                 info="0 = keep v_w's amplitude, 1 = fully adopt v∅'s.", visible=False)
            gain = gr.Slider(0.0, 3.0, value=1.5, step=0.05, label="gain (x)",
                             info=">1 amplify, 1 identity, <1 attenuate. DC kept at 1.", visible=False)
            interval = RangeSlider(minimum=0.0, maximum=1.0, value=(0.0, 1.0), step=0.01,
                                   label="timesteps to intervene [start, end]", visible=False,
                                   info="Fraction of the denoising schedule (0=first step … 1=last). "
                                        "[0,1] = every step; shrink for early-only / late-only.")
            with gr.Row():
                seed = gr.Number(value=0, precision=0, label="seed",
                                 info="Fix to compare fairly; baseline cached per seed.")
                steps = gr.Slider(10, 50, value=MODEL["steps"], step=1, label="steps",
                                  info="Denoising steps: more = better, slower.")
            with gr.Row():
                guidance = gr.Slider(1.0, 10.0, value=MODEL["guidance"], step=0.1, label="guidance (CFG w)",
                                     info="SD3.5 real CFG; w=1 = no guidance (op becomes a no-op).")
                size = gr.Dropdown([512, 768, 1024], value=768, label="size",
                                   info="Pixels; smaller = faster, less VRAM.")
            go = gr.Button("Generate", variant="primary")
        with gr.Column(scale=2):
            out_base, out_edit = _image_pair("baseline (plain CFG)", "velocity-edited",
                                              "pair_velocity")
            desc = gr.Markdown()

    op.change(_velocity_visibility, op, [band, strength, gain, interval, helpbox])
    names = ["promptA", "op", "band", "strength", "gain", "interval", "seed", "steps",
             "guidance", "size"]
    inputs = [promptA, op, band, strength, gain, interval, seed, steps, guidance, size]
    go.click(run_velocity, inputs, [out_base, out_edit, desc])
    _save_load_ui("velocity", names, inputs, out_base, out_edit,
                  vis=(_velocity_visibility, op, [band, strength, gain, interval, helpbox]))


# ===========================================================================
# SPECTRAL AdaIN (E39): a single-pass, picked-params self-AdaIN on the latent
# ===========================================================================
# The operator lives in spectral_adain.py; this tab is a thin wrapper. During the ONE
# generation pass, each step it FFTs the latent, splits the 2D spectrum into soft radial
# bands (a partition of unity), and rewrites each band's MAGNITUDE to a user-picked AdaIN
# affine |V~_k| = g_k*(|V|-mu_k)/sig_k + b_k (mu_k, sig_k measured from the current latent),
# keeping the content phase. Absolute targets in raw latent units; identity is g=sig, b=mu.

ADAIN_OPS = ["global", "3-band"]

ADAIN_INTRO_MD = """\
### Spectral AdaIN — a single-pass frequency knob (E39)
The network's internal **AdaLN** is a *semantic / timestep* knob. This is the orthogonal
*frequency* knob: during the **one** generation pass, each step it FFTs the latent, splits the 2D
spectrum into **soft radial bands**, and rewrites each band's **magnitude** to an AdaIN affine you
choose — `|V~_k| = g_k·(|V|−μ_k)/σ_k + b_k` — keeping the **content phase**. μ_k, σ_k are measured
from the *current* latent (no reference image, no second generation).

- **global** — one band over the whole spectrum: target std `g`, target mean magnitude `b`.
- **3-band** — three soft radial groups (low / mid / high), each with its own `g, b`.

`g, b` are **absolute, raw latent units** (not multipliers): identity is `g≈σ, b≈μ`. The run
**reports the measured μ_k, σ_k per band** — read them, then set targets near those for a gentle
edit or far for a strong one. One (g,b) is shared across the 16 latent channels.

Left = baseline (vanilla), right = edit, same seed. Needs **Flux** (`--model flux-dev`).
"""

ADAIN_HELP = {
    "global": "**Global AdaIN** *(g, b)*. Each step, reset the whole spectrum's magnitude to mean "
        "`b`, std `g` (phase kept). Absolute raw units — see the reported μ, σ; g≈σ & b≈μ ≈ identity.",
    "3-band": "**3-band AdaIN** *(g/b × low/mid/high)*. Same affine per soft radial band: low = "
        "global tone/layout, high = fine texture. Absolute units; read the reported per-band μ_k, σ_k.",
}


def _adain_band_masks(hl, wl, mode, device):
    """global -> one all-ones band (K=1); 3-band -> 3 soft radial rings (partition of unity)."""
    from spectral_adain import soft_band_masks
    if mode == "global":
        return torch.ones(1, hl, wl, device=device)
    return soft_band_masks(hl, wl, centers=[0.0, 0.5, 1.0], widths=[0.22, 0.22, 0.22], device=device)


def run_adain(prompt, op, g_all, b_all, g_lo, b_lo, g_mid, b_mid, g_hi, b_hi,
              seed, steps, guidance, size):
    if MODEL["kind"] != "flux":
        return None, None, _FLUX_ONLY_NOTE
    if not (prompt or "").strip():
        return None, None, "enter a prompt"
    seed, steps, size = int(seed), int(steps), int(size)
    try:
        import torch.fft as _tfft
        from spectral_adain import adain_affine, band_moments
        peA, ppeA, _ = _encode_cached(prompt)
        bkey = (prompt, seed, steps, float(guidance), size)
        if bkey not in _BASE_CACHE:
            _BASE_CACHE[bkey] = generate(peA, ppeA, seed, steps, guidance, size)
        base = _BASE_CACHE[bkey]

        g = [float(g_all)] if op == "global" else [float(g_lo), float(g_mid), float(g_hi)]
        b = [float(b_all)] if op == "global" else [float(b_lo), float(b_mid), float(b_hi)]
        M_holder, meas = [None], [None]

        def op_fn(lat, i):
            if M_holder[0] is None:
                M_holder[0] = _adain_band_masks(lat.shape[-2], lat.shape[-1], op, lat.device)
            M = M_holder[0]
            if meas[0] is None:
                mu, sig = band_moments(_tfft.fft2(lat.float(), norm="ortho").abs(), M)
                meas[0] = (mu.mean(dim=(1, 2)).tolist(), sig.mean(dim=(1, 2)).tolist())
            return adain_affine(lat.float(), M, g, b).to(lat.dtype)

        edited = generate_latent(peA, ppeA, seed, steps, guidance, size, op_fn=op_fn,
                                 schedule="every")
        names = ["all"] if op == "global" else ["low", "mid", "high"]
        mu_m, sig_m = meas[0]
        rows = "; ".join(f"{n}: measured μ={m:.3g} σ={s:.3g} → target b={bb:.3g} g={gg:.3g}"
                         for n, m, s, bb, gg in zip(names, mu_m, sig_m, b, g))
        return base, edited, f"AdaIN {op} (absolute targets) — {rows}"
    except Exception as e:
        return None, None, f"error: {e}"


def _adain_visibility(op):
    import gradio as gr
    glob = (op == "global")
    return [gr.update(visible=glob), gr.update(visible=glob),          # g_all, b_all
            gr.update(visible=not glob), gr.update(visible=not glob),  # g_lo, b_lo
            gr.update(visible=not glob), gr.update(visible=not glob),  # g_mid, b_mid
            gr.update(visible=not glob), gr.update(visible=not glob),  # g_hi, b_hi
            gr.update(value=ADAIN_HELP.get(op, ""))]


def _adain_tab():
    import gradio as gr
    with gr.Accordion("How it works  ·  what the knobs mean", open=False):
        gr.Markdown(ADAIN_INTRO_MD)
    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(label="Prompt",
                                value="a fluffy orange tabby cat sitting on a windowsill")
            op = gr.Dropdown(ADAIN_OPS, value="global", label="Operation")
            helpbox = gr.Markdown(ADAIN_HELP["global"])
            with gr.Row():
                g_all = gr.Number(value=1.0, label="g (target std)")
                b_all = gr.Number(value=1.0, label="b (target mean)")
            with gr.Row():
                g_lo = gr.Number(value=1.0, label="g low", visible=False)
                b_lo = gr.Number(value=1.0, label="b low", visible=False)
            with gr.Row():
                g_mid = gr.Number(value=1.0, label="g mid", visible=False)
                b_mid = gr.Number(value=1.0, label="b mid", visible=False)
            with gr.Row():
                g_hi = gr.Number(value=1.0, label="g high", visible=False)
                b_hi = gr.Number(value=1.0, label="b high", visible=False)
            with gr.Row():
                seed = gr.Number(value=0, precision=0, label="seed")
                steps = gr.Slider(4, 28, value=MODEL["steps"], step=1, label="steps")
            with gr.Row():
                guidance = gr.Slider(1.0, 7.0, value=3.5, step=0.1, label="guidance")
                size = gr.Dropdown([512, 768, 1024], value=768, label="size")
            go = gr.Button("Generate", variant="primary")
        with gr.Column(scale=2):
            out_base, out_edit = _image_pair("baseline", "edited", "pair_adain")
            desc = gr.Markdown()

    op.change(_adain_visibility, op,
              [g_all, b_all, g_lo, b_lo, g_mid, b_mid, g_hi, b_hi, helpbox])
    names = ["prompt", "op", "g_all", "b_all", "g_lo", "b_lo", "g_mid", "b_mid", "g_hi", "b_hi",
             "seed", "steps", "guidance", "size"]
    inputs = [prompt, op, g_all, b_all, g_lo, b_lo, g_mid, b_mid, g_hi, b_hi,
              seed, steps, guidance, size]
    go.click(run_adain, inputs, [out_base, out_edit, desc])
    _save_load_ui("adain", names, inputs, out_base, out_edit,
                  vis=(_adain_visibility, op,
                       [g_all, b_all, g_lo, b_lo, g_mid, b_mid, g_hi, b_hi, helpbox]))


# ---------------------------------------------------------------------------
# RF inversion + trajectory-matched low-band spectral clamp (E40)
# ---------------------------------------------------------------------------
#
# Invert a real image to noise on Flux (reverse Euler under the source prompt), recording the
# latent at every sigma node -> the inversion *trajectory*. Regenerate under the edit prompt
# and clamp the LOW band [0, cut] of each step's latent back to traj[i] (same sigma). Low bands
# carry coarse layout, so this preserves structure while the high band follows the edit.
# Three modes: sbn (power), phase (power + low-band phase lock), adain (per-band mean+std).

INV_ADAIN_K = 8
_FH = _FW = 128                 # Flux latent dims (real image is encoded at 1024px -> 128x128)
_INV_SIZE = 1024                # px the real image is VAE-encoded at (-> _FH x _FW latent)
_INV_SEQLEN = 512               # Flux txt_ids length


# Flux denoising plumbing + spectral clamp + RF-inversion edit now live in invert_core
# (shared verbatim with the e41 calibration harness). Imported as thin aliases so every
# existing call site -- and the demo's light startup -- is unchanged.
import invert_core as _ic
from invert_core import flux_sigmas, flux_velocity, pack, unpack, vae_encode
_gids = _ic.gids


_INV_INTRO_MD = (
    "**RF inversion + spectral clamp.** Upload an image and give its caption (**source**) plus an "
    "**edit** prompt. We RF-invert the image to noise under the source prompt, recording the latent "
    "spectrum at every step, then regenerate under the edit prompt while pinning the **low band "
    "[0, cut]** back to that recorded trajectory. **Left** = plain inversion edit (no clamp); "
    "**right** = clamped edit. Modes — **sbn**: match low-band power; **phase**: sbn power **plus** "
    "lock the source phase on an independent **phase band [lo, hi]** (its own slider); "
    "**adain**: match low-band mean+std. Smaller `cut` preserves only the coarsest structure; "
    "`strength` blends the clamp; the time window limits which steps clamp.")


_inv_clamp = _ic.inv_clamp


@torch.no_grad()
def _rf_invert(pe, ppe, x0_packed, sig, gids):
    return _ic.rf_invert(PIPE, pe, ppe, x0_packed, sig, gids)


@torch.no_grad()
def _forward_edit(pe, ppe, x_noise, sig, gids, traj=None, mode=None, cut=0.25,
                  strength=1.0, window=None, idx=None, M=None, cen_k=None,
                  phase_band=(0.0, 0.25)):
    lat = _ic.forward_edit(PIPE, pe, ppe, x_noise, sig, gids, traj=traj, mode=mode,
                           cut=cut, strength=strength, window=window, idx=idx, M=M,
                           cen_k=cen_k, phase_band=phase_band)
    return decode_latent(lat)


# Single-slot caches for the invert tab: the inversion (x_noise, traj) and the no-clamp
# baseline depend only on the source/edit prompts, image, steps and guidance -- not on the
# method knobs -- so tweaking mode/cut/strength/interval reuses both and only re-runs `edited`.
_INV_CACHE = {"key": None, "val": None}
_INV_BASE_CACHE = {"key": None, "val": None}


def run_invert(src_prompt, edit_prompt, real_img, mode, cut, strength, interval,
               phase_band, seed, steps, guidance, inv_guidance):
    import gradio as gr
    if MODEL["kind"] != "flux":
        return None, None, _FLUX_ONLY_NOTE
    if real_img is None:
        return None, None, "upload a real image"
    if not (src_prompt or "").strip():
        return None, None, "enter the source caption"
    if not (edit_prompt or "").strip():
        return None, None, "enter the edit prompt"
    try:
        from spectral_ops import band_index_map
        from spectral_adain import soft_band_masks
        seed, steps, cut = int(seed), int(steps), float(cut)
        peS, ppeS, _ = _encode_cached(src_prompt)
        peE, ppeE, _ = _encode_cached(edit_prompt)
        peS, ppeS, peE, ppeE = peS.cuda(), ppeS.cuda(), peE.cuda(), ppeE.cuda()
        sig = flux_sigmas(PIPE, steps)
        gids_inv = _gids(PIPE, float(inv_guidance))
        gids_gen = _gids(PIPE, float(guidance))
        idx = band_index_map(_FH, _FW, N_BINS, "cuda")
        cen = torch.linspace(0, 1, INV_ADAIN_K)
        M = soft_band_masks(_FH, _FW, cen.tolist(), [1.0 / INV_ADAIN_K] * INV_ADAIN_K, "cuda")
        cen_k = cen.tolist()

        # 1) inversion (source prompt, image, steps, inv-guidance) -> noise + trajectory
        img_hash = hash(real_img.tobytes())
        inv_key = (src_prompt, img_hash, steps, float(inv_guidance))
        if _INV_CACHE["key"] != inv_key:
            x0 = pack(PIPE, vae_encode(PIPE.vae, real_img))
            _INV_CACHE.update(key=inv_key, val=_rf_invert(peS, ppeS, x0, sig, gids_inv))
        x_noise, traj = _INV_CACHE["val"]
        nstd = float(unpack(PIPE, x_noise).std())

        t_lo, t_hi = _ordered(interval)
        window = (int(round(t_lo * (steps - 1))), int(round(t_hi * (steps - 1))))

        # 2) no-clamp baseline (inversion + edit prompt + guidance), independent of the knobs
        base_key = (inv_key, edit_prompt, float(guidance))
        if _INV_BASE_CACHE["key"] != base_key:
            _INV_BASE_CACHE.update(key=base_key,
                                   val=_forward_edit(peE, ppeE, x_noise, sig, gids_gen))
        base = _INV_BASE_CACHE["val"]

        # 3) method output -- the only pass that depends on mode/cut/strength/interval
        pband = _ordered(phase_band)
        edited = _forward_edit(peE, ppeE, x_noise, sig, gids_gen, traj=traj, mode=mode,
                               cut=cut, strength=float(strength), window=window,
                               idx=idx, M=M, cen_k=cen_k, phase_band=pband)
        pdesc = f" · phase band [{pband[0]:.2f},{pband[1]:.2f}]" if mode == "phase" else ""
        desc = (f"RF-inversion edit · mode **{mode}** · cut {cut:.2f} · strength {float(strength):.2f}"
                f"{pdesc} · clamp steps {window[0]}–{window[1]} · inv-guidance {float(inv_guidance):.1f} "
                f"· inverted-noise std {nstd:.3f} (≈1.0 = clean inversion)")
        return base, edited, desc
    except Exception as e:
        return None, None, f"error: {e}"


def _invert_tab():
    import gradio as gr
    from gradio_rangeslider import RangeSlider
    with gr.Accordion("How it works  ·  what the knobs mean", open=False):
        gr.Markdown(_INV_INTRO_MD)
    with gr.Row():
        with gr.Column(scale=1):
            src_prompt = gr.Textbox(label="Source caption (describes the image)",
                                    value="a photograph of a cat sitting on a sofa")
            edit_prompt = gr.Textbox(label="Edit prompt",
                                     value="a photograph of a dog sitting on a sofa")
            real_img = gr.Image(label="Real image", type="pil")
            mode = gr.Dropdown(["sbn", "phase", "adain"], value="sbn", label="Clamp mode")
            cut = gr.Slider(0.05, 0.95, value=0.25, step=0.01, label="cut (low-band cutoff)",
                            info="0 = DC/global … 1 = corner/fine. Lower = preserve only coarse structure.")
            strength = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="strength",
                                 info="0 = no clamp (= baseline), 1 = full clamp to the source trajectory.")
            phase_band = RangeSlider(minimum=0.0, maximum=1.0, value=(0.0, 0.25), step=0.01,
                                     label="phase band [lo, hi]  (phase mode only)",
                                     info="Radial band whose PHASE is taken from the source (magnitude kept). "
                                          "Independent of cut; 0 = DC … 1 = corner.")
            interval = RangeSlider(minimum=0.0, maximum=1.0, value=(0.0, 1.0), step=0.01,
                                   label="clamp window [start, end]",
                                   info="Fraction of the schedule on which the low-band clamp fires.")
            with gr.Row():
                seed = gr.Number(value=0, precision=0, label="seed")
                steps = gr.Slider(4, 28, value=MODEL["steps"], step=1, label="steps")
            with gr.Row():
                guidance = gr.Slider(1.0, 7.0, value=3.5, step=0.1, label="guidance (edit)")
                inv_guidance = gr.Slider(1.0, 7.0, value=1.0, step=0.1, label="guidance (inversion)",
                                         info="Flux inversion is usually most faithful at 1.0.")
            go = gr.Button("Generate", variant="primary")
        with gr.Column(scale=2):
            out_base, out_edit = _image_pair("edit · no clamp", "edit · low-band clamped",
                                              "pair_invert")
            desc = gr.Markdown()
    names = ["src_prompt", "edit_prompt", "real_img", "mode", "cut", "strength", "interval",
             "phase_band", "seed", "steps", "guidance", "inv_guidance"]
    inputs = [src_prompt, edit_prompt, real_img, mode, cut, strength, interval,
              phase_band, seed, steps, guidance, inv_guidance]
    go.click(run_invert, inputs, [out_base, out_edit, desc])
    _save_load_ui("invert", names, inputs, out_base, out_edit)


# ---------------------------------------------------------------------------
# FlowEdit (inversion-free) with frequency-annealed smoothing of V_delta
# ---------------------------------------------------------------------------
#
# FlowEdit integrates the editing-direction field V_delta = V_tar - V_src directly (no
# inversion) and adds the accumulated delta to the source latent. Here we low-pass V_delta
# in the 2D radial frequency domain with a GAUSSIAN mask whose cutoff is ANNEALED from
# `start_cut` (heavy smoothing, early/high-sigma steps) up to `end_cut` (light/none, late
# steps): early steps commit only the coarse/low-frequency part of the edit, high-frequency
# detail enters progressively -- a coarse-to-fine edit. Left = plain FlowEdit (no smoothing),
# right = annealed-smoothed.

_FE_INTRO_MD = (
    "**FlowEdit.** Inversion-free text editing: integrate the difference between the "
    "**target**- and **source**-prompted velocity fields, `V_delta = V_tar - V_src`, and add "
    "it to the source latent. Upload a real image (or leave empty to generate the source from "
    "its prompt). **Experimental knob:** low-pass `V_delta` with a gaussian whose cutoff is "
    "annealed `start_cut → end_cut` over the steps, so early steps make only coarse edits and "
    "fine detail enters later. **Left** = plain FlowEdit; **right** = annealed-smoothed. "
    "`skip` drops the noisiest early steps; `renorm` rescales the filtered direction back to "
    "full magnitude (annealing then changes only *which* frequencies move, not the edit speed).")


def _gauss_lowpass(vd_packed, cutoff, renorm):
    """Gaussian radial low-pass of a packed velocity-delta (shared with the FlowAlign tab)."""
    return _ic.gauss_lowpass(PIPE, vd_packed, cutoff, renorm)


@torch.no_grad()
def flowedit_annealed(x0_packed, C_src, C_tar, sig, skip, seed, gids,
                      start_cut, end_cut, renorm):
    """FlowEdit with a gaussian low-pass on V_delta, cutoff annealed start_cut->end_cut."""
    steps = len(sig) - 1
    eps = torch.randn(x0_packed.shape, generator=torch.Generator("cuda").manual_seed(seed),
                      device="cuda").float()
    delta = torch.zeros_like(x0_packed)
    i0 = int(skip * steps)
    for i in range(i0, steps):
        frac = (i - i0) / max(steps - 1 - i0, 1)            # 0 at first active step -> 1 at last
        cutoff = start_cut + (end_cut - start_cut) * frac  # linear anneal low -> high
        s_hi, s_lo = float(sig[i]), float(sig[i + 1])
        x_src = (1 - s_hi) * x0_packed + s_hi * eps
        x_tar = x_src + delta
        v_src = flux_velocity(PIPE, x_src, s_hi, C_src[0], C_src[1], gids)
        v_tar = flux_velocity(PIPE, x_tar, s_hi, C_tar[0], C_tar[1], gids)
        vd = _gauss_lowpass((s_lo - s_hi) * (v_tar - v_src), cutoff, renorm)
        delta = delta + vd
    return x0_packed + delta


def run_flowedit(src_prompt, tar_prompt, real_img, skip, start_cut, end_cut, renorm,
                 seed, steps, guidance):
    if MODEL["kind"] != "flux":
        return None, None, _FLUX_ONLY_NOTE
    if not (src_prompt or "").strip():
        return None, None, "enter the source prompt"
    if not (tar_prompt or "").strip():
        return None, None, "enter the target prompt"
    try:
        seed, steps = int(seed), int(steps)
        skip, start_cut, end_cut = float(skip), float(start_cut), float(end_cut)
        guidance = float(guidance)
        peS, ppeS, _ = _encode_cached(src_prompt)
        peT, ppeT, _ = _encode_cached(tar_prompt)
        C_src = (peS.cuda(), ppeS.cuda())
        C_tar = (peT.cuda(), ppeT.cuda())
        sig = flux_sigmas(PIPE, steps)
        gids = _gids(PIPE, guidance)
        if real_img is not None:
            x0 = pack(PIPE, vae_encode(PIPE.vae, real_img))
        else:                                            # 1024px source from the prompt itself
            x0 = pack(PIPE, _final_latent(C_src[0], C_src[1], seed, steps, guidance, 1024))
        base_key = ("flowedit", src_prompt, tar_prompt, _img_key(real_img), seed, steps,
                    round(guidance, 3), round(skip, 3))
        base = cached_baseline(base_key, lambda: decode_latent(unpack(PIPE, flowedit_annealed(
            x0, C_src, C_tar, sig, skip, seed, gids, 1.0, 1.0, False)).float()))   # passthrough = plain FlowEdit
        edited = decode_latent(unpack(PIPE, flowedit_annealed(
            x0, C_src, C_tar, sig, skip, seed, gids, start_cut, end_cut, bool(renorm))).float())
        desc = (f"FlowEdit · skip {skip:.2f} · guidance {guidance:.1f} · gaussian low-pass on "
                f"V_delta, cutoff {start_cut:.2f}→{end_cut:.2f} · "
                f"energy renorm {'on' if renorm else 'off'}")
        return base, edited, desc
    except Exception as e:
        return None, None, f"error: {e}"


def _flowedit_tab():
    import gradio as gr
    with gr.Accordion("How it works  ·  what the knobs mean", open=False):
        gr.Markdown(_FE_INTRO_MD)
    with gr.Row():
        with gr.Column(scale=1):
            src_prompt = gr.Textbox(label="Source prompt (describes the image)",
                                    value="a photograph of a cat sitting on a sofa")
            tar_prompt = gr.Textbox(label="Target prompt",
                                    value="a photograph of a dog sitting on a sofa")
            real_img = gr.Image(label="Real image (optional)", type="pil")
            skip = gr.Slider(0.0, 0.5, value=0.0, step=0.01, label="skip (drop early steps)",
                             info="Fraction of the noisiest early steps to skip before editing.")
            with gr.Row():
                start_cut = gr.Slider(0.02, 1.0, value=0.1, step=0.01, label="start cutoff",
                                      info="Low-pass cutoff at the FIRST step (small = heavy smoothing).")
                end_cut = gr.Slider(0.02, 1.0, value=1.0, step=0.01, label="end cutoff",
                                    info="Cutoff at the LAST step (1.0 = no smoothing).")
            renorm = gr.Checkbox(value=False, label="preserve V_delta energy",
                                 info="Rescale the filtered direction to full magnitude.")
            with gr.Row():
                seed = gr.Number(value=0, precision=0, label="seed")
                steps = gr.Slider(4, 28, value=MODEL["steps"], step=1, label="steps")
            guidance = gr.Slider(1.0, 7.0, value=3.5, step=0.1, label="guidance")
            go = gr.Button("Generate", variant="primary")
        with gr.Column(scale=2):
            out_base, out_edit = _image_pair("FlowEdit · no smoothing",
                                              "FlowEdit · annealed low-pass", "pair_flowedit")
            desc = gr.Markdown()
    names = ["src_prompt", "tar_prompt", "real_img", "skip", "start_cut", "end_cut", "renorm",
             "seed", "steps", "guidance"]
    inputs = [src_prompt, tar_prompt, real_img, skip, start_cut, end_cut, renorm,
              seed, steps, guidance]
    go.click(run_flowedit, inputs, [out_base, out_edit, desc])
    _save_load_ui("flowedit", names, inputs, out_base, out_edit)


# ---------------------------------------------------------------------------
# FlowAlign (Kim et al. 2025, arXiv:2505.23145): FlowEdit + a source-consistency
# TERMINAL-POINT regularizer. Per step (x_src==qt, x_tar==pt):
#   qt = (1-σ)·x0 + σ·eps ;  pt = xt + qt - x0
#   vp = v(pt,c_src) + w·(v(pt,c_tgt) - v(pt,c_src))    # CFG, SOURCE prompt as negative
#   vq = v(qt,c_src)
#   xt += (σ_next-σ)·(vp - vq)  +  ζ·(E[q0|qt] - E[p0|pt])      with E[·|x]=x-σ·v
# Two experimental spectral twists over plain FlowAlign (both = identity at defaults):
#   (1) SBN on the CFG reference: clamp vp's low radial band toward v(pt,c_src), taming
#       high-w structural over-editing while keeping the semantic push (idea: reduce the
#       effective w spectrally instead of globally).  -> _vel_sbn
#   (2) spectral / annealed terminal point: band-limit the consistency vector before
#       adding it (low bands early -> high later), reusing _gauss_lowpass.
# Left = plain FlowAlign; right = your variant.

_FA_INTRO_MD = (
    "**FlowAlign.** Inversion-free editing = FlowEdit **plus** a source-consistency term at the "
    "*terminal* (clean) point. Per step it forms the CFG velocity `vp = v(pt,c_src) + w·(v(pt,c_tgt) "
    "− v(pt,c_src))` (the **source** prompt is the CFG negative), integrates `vp − v(qt,c_src)`, and "
    "adds `ζ·(E[q0|qt] − E[p0|pt])` where `E[·|x]=x−σ·v` is the predicted clean latent — pulling the "
    "edit's clean estimate toward the source's. Upload a real image (or leave empty to generate the "
    "source from its prompt).\n\n"
    "**Experimental knobs (both off ⇒ plain FlowAlign):** "
    "**(1) SBN on the CFG reference** — spectrally clamp `vp`'s low radial band `[0,cut]` toward the "
    "reference `v(pt,c_src)` (`band power`/`mag` power-match, `phase` low-band phase-lock, `both`), "
    "reducing the effective `w` only where structure lives. "
    "**(2) annealed terminal point** — low-pass the consistency vector with a gaussian whose cutoff "
    "anneals `start→end` over steps (coarse source-consistency early, fine detail freed later). "
    "**Left** = plain FlowAlign; **right** = your variant.")

FA_SBN_MODES = _ic.FA_SBN_MODES   # ["off", "band power", "mag", "phase", "both"]


def run_flowalign(src_prompt, tar_prompt, real_img, w, zeta, sbn_mode, sbn_cut, sbn_strength,
                  term_start_cut, term_end_cut, term_renorm, seed, steps, gbase):
    if MODEL["kind"] != "flux":
        return None, None, _FLUX_ONLY_NOTE
    if not (src_prompt or "").strip():
        return None, None, "enter the source prompt"
    if not (tar_prompt or "").strip():
        return None, None, "enter the target prompt"
    try:
        seed, steps = int(seed), int(steps)
        w, zeta, gbase = float(w), float(zeta), float(gbase)
        sbn_cut, sbn_strength = float(sbn_cut), float(sbn_strength)
        term_start_cut, term_end_cut = float(term_start_cut), float(term_end_cut)
        peS, ppeS, _ = _encode_cached(src_prompt)
        peT, ppeT, _ = _encode_cached(tar_prompt)
        C_src = (peS.cuda(), ppeS.cuda())
        C_tar = (peT.cuda(), ppeT.cuda())
        sig = flux_sigmas(PIPE, steps)
        gids = _gids(PIPE, gbase)
        if real_img is not None:
            x0 = pack(PIPE, vae_encode(PIPE.vae, real_img))
        else:                                            # 1024px source from the prompt itself
            x0 = pack(PIPE, _final_latent(C_src[0], C_src[1], seed, steps, gbase, 1024))
        # baseline (left) = plain FlowAlign; depends on the editing-determining inputs only
        base_key = ("flowalign", src_prompt, tar_prompt, _img_key(real_img), seed, steps,
                    round(w, 3), round(zeta, 4), round(gbase, 3))
        base = cached_baseline(base_key, lambda: decode_latent(unpack(PIPE, _ic.flowalign(
            PIPE, x0, C_src, C_tar, sig, seed, gids, w, zeta)).float()))
        edited = decode_latent(unpack(PIPE, _ic.flowalign(
            PIPE, x0, C_src, C_tar, sig, seed, gids, w, zeta,
            sbn_mode, sbn_cut, sbn_strength, term_start_cut, term_end_cut,
            bool(term_renorm))).float())
        desc = (f"FlowAlign · w {w:.1f} · ζ {zeta:.3f} · flux-guidance {gbase:.1f} · "
                f"SBN[{sbn_mode}] cut {sbn_cut:.2f} str {sbn_strength:.2f} · "
                f"terminal cutoff {term_start_cut:.2f}→{term_end_cut:.2f} "
                f"renorm {'on' if term_renorm else 'off'}")
        return base, edited, desc
    except Exception as e:
        return None, None, f"error: {e}"


def _flowalign_tab():
    import gradio as gr
    with gr.Accordion("How it works  ·  what the knobs mean", open=False):
        gr.Markdown(_FA_INTRO_MD)
    with gr.Row():
        with gr.Column(scale=1):
            src_prompt = gr.Textbox(label="Source prompt (describes the image)",
                                    value="a photograph of a cat sitting on a sofa")
            tar_prompt = gr.Textbox(label="Target prompt",
                                    value="a photograph of a dog sitting on a sofa")
            real_img = gr.Image(label="Real image (optional)", type="pil")
            with gr.Row():
                w = gr.Slider(1.0, 15.0, value=5.0, step=0.5, label="w (CFG, src negative)",
                              info="Edit strength; source prompt is the CFG negative.")
                zeta = gr.Slider(0.0, 0.05, value=0.01, step=0.001, label="ζ (terminal consistency)",
                                 info="Source-consistency weight at the clean point (0 = FlowEdit).")
            gr.Markdown("**Twist 1 — SBN on the CFG reference** *(tame structural over-editing)*")
            with gr.Row():
                sbn_mode = gr.Dropdown(FA_SBN_MODES, value="off", label="SBN mode",
                                       info="Clamp vp's low band toward v(pt,c_src).")
                sbn_cut = gr.Slider(0.0, 0.6, value=0.2, step=0.01, label="SBN cut (low band)",
                                    info="Upper edge of the clamped low radial band.")
                sbn_strength = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="SBN strength",
                                         info="Blend vp's power toward the reference (band/mag).")
            gr.Markdown("**Twist 2 — annealed terminal point** *(coarse-to-fine consistency)*")
            with gr.Row():
                term_start_cut = gr.Slider(0.02, 1.0, value=1.0, step=0.01, label="terminal start cut",
                                           info="Low-pass cutoff of the consistency vector at step 0.")
                term_end_cut = gr.Slider(0.02, 1.0, value=1.0, step=0.01, label="terminal end cut",
                                         info="Cutoff at the last step (both 1.0 = plain FlowAlign).")
            term_renorm = gr.Checkbox(value=False, label="preserve terminal-vector energy",
                                      info="Rescale the filtered consistency vector to full magnitude.")
            with gr.Row():
                seed = gr.Number(value=0, precision=0, label="seed")
                steps = gr.Slider(4, 28, value=MODEL["steps"], step=1, label="steps")
            gbase = gr.Slider(1.0, 4.0, value=1.0, step=0.5, label="flux guidance (base embed)",
                              info="Distilled guidance embed for the base velocity; w is applied on top.")
            go = gr.Button("Generate", variant="primary")
        with gr.Column(scale=2):
            out_base, out_edit = _image_pair("FlowAlign · plain",
                                              "FlowAlign · spectral variant", "pair_flowalign")
            desc = gr.Markdown()
    names = ["src_prompt", "tar_prompt", "real_img", "w", "zeta", "sbn_mode", "sbn_cut",
             "sbn_strength", "term_start_cut", "term_end_cut", "term_renorm", "seed", "steps", "gbase"]
    inputs = [src_prompt, tar_prompt, real_img, w, zeta, sbn_mode, sbn_cut, sbn_strength,
              term_start_cut, term_end_cut, term_renorm, seed, steps, gbase]
    go.click(run_flowalign, inputs, [out_base, out_edit, desc])
    _save_load_ui("flowalign", names, inputs, out_base, out_edit)


# ---------------------------------------------------------------------------
# LTX-Video FlowAlign tab (E45): inversion-free FlowAlign on a real video model,
# with the low-band phase op (2D per-frame vs 3D spatiotemporal). Left = plain
# FlowAlign; right = + phase op. Reuses the experiment's LTX core.
# ---------------------------------------------------------------------------
def run_ltx_video(mode, up_video, src_prompt, src_caption, edit_prompt, w, zeta, sbn_cut,
                  phase, frames, width, height, steps, seed):
    import tempfile
    import imageio.v3 as iio
    import e45_ltx_flowalign as L
    if MODEL["kind"] != "ltx":
        return None, None, "**Relaunch with `--model ltx`** — this tab needs LTX-Video."
    try:
        pipe = PIPE
        W, H = int(width), int(height)               # LTX-native, non-square (round to /32)
        W, H = (W // 32) * 32, (H // 32) * 32
        frames = max(((int(frames) - 1) // 8) * 8 + 1, 9)
        seed, steps = int(seed), int(steps)
        n_max = steps - max(2, round(0.15 * steps))   # skip the highest-noise early steps (canonical)
        if mode == "upload":
            if not up_video:
                return None, None, "Upload a clip, or switch Source to 'generate'."
            src_frames = L.ltx_conform(up_video, W, frames, H)
            caption = src_caption.strip() or src_prompt
        else:
            g = pipe(prompt=src_prompt, num_frames=frames, height=H, width=W,
                     num_inference_steps=steps, guidance_scale=3.0,
                     generator=torch.Generator("cuda").manual_seed(seed), output_type="np")
            src_frames = (np.asarray(g.frames[0]) * 255).round().astype(np.uint8)
            caption = src_prompt
        F = len(src_frames)
        Fl = (F - 1) // pipe.vae_temporal_compression_ratio + 1
        Hl, Wl = H // pipe.vae_spatial_compression_ratio, W // pipe.vae_spatial_compression_ratio
        sig, ts = L.ltx_schedule(pipe, steps, Fl * Hl * Wl)
        C_src, C_tar = L.ltx_encode_prompt(pipe, caption), L.ltx_encode_prompt(pipe, edit_prompt)
        x0 = L.ltx_pack(pipe, L.ltx_encode(pipe, src_frames))

        def edit(m, c):
            xe = L.flowalign_video(pipe, x0, C_src, C_tar, sig, ts, seed, float(w), float(zeta),
                                   Fl, Hl, Wl, sbn_mode=m, sbn_cut=float(c), n_max=n_max)
            fr = L.ltx_decode(pipe, L.ltx_unpack(pipe, xe, Fl, Hl, Wl))
            p = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
            iio.imwrite(p, fr, fps=8)
            return p

        base = edit("off", 0.0)
        var = edit(phase, sbn_cut) if phase != "off" else base
        desc = (f"FlowAlign·LTX · {F}f@{H}px · w {float(w):.1f} · ζ {float(zeta):.3f} · "
                f"phase {phase}" + (f" cut {float(sbn_cut):.2f}" if phase != "off" else ""))
        return base, var, desc
    except Exception as e:
        return None, None, f"error: {e}"


def _ltx_video_tab():
    import gradio as gr
    gr.Markdown("**FlowAlign on LTX-Video (E45).** Inversion-free FlowAlign on a real video model. "
                "The low-band **phase** op — `phase2d` (per-frame, ~the paper) vs `phase3d` "
                "(spatiotemporal) — preserves structure; 3D also helps temporal coherence. "
                "Left = plain FlowAlign · right = + phase op.")
    if MODEL["kind"] != "ltx":
        gr.Markdown("⚠️ This tab needs **LTX-Video** — relaunch with `--model ltx`.")
    with gr.Row():
        with gr.Column(scale=1):
            mode = gr.Radio(["generate", "upload"], value="generate", label="Source")
            up_video = gr.Video(label="Upload a clip (Source = 'upload')")
            src_prompt = gr.Textbox(label="Source prompt (generate) / its caption",
                                    value="a small toy car driving across a wooden table")
            src_caption = gr.Textbox(label="Source caption (for an uploaded clip)", value="")
            edit_prompt = gr.Textbox(label="Edit prompt",
                                     value="a small toy tank driving across a wooden table")
            with gr.Row():
                w = gr.Slider(1.0, 18.0, value=6.0, step=0.5, label="w (CFG, src negative)")
                zeta = gr.Slider(0.0, 0.05, value=0.01, step=0.001, label="ζ (terminal consistency)")
            with gr.Row():
                phase = gr.Dropdown(["off", "phase2d", "phase3d"], value="phase3d", label="phase op",
                                    info="3D = spatiotemporal (couples frames); 2D = per-frame.")
                sbn_cut = gr.Slider(0.0, 0.6, value=0.2, step=0.01, label="phase cut (low band)")
            with gr.Row():
                frames = gr.Slider(9, 97, value=49, step=8, label="frames (8k+1)")
                steps = gr.Slider(8, 40, value=30, step=1, label="steps")
            with gr.Row():
                width = gr.Dropdown([512, 640, 704, 768, 832], value=704, label="width (px, /32)",
                                    info="LTX wants larger, non-square frames; square low-res distorts.")
                height = gr.Dropdown([320, 384, 448, 480, 512], value=480, label="height (px, /32)")
            seed = gr.Number(value=0, precision=0, label="seed")
            go = gr.Button("Generate", variant="primary")
        with gr.Column(scale=2):
            with gr.Row():
                out_base = gr.Video(label="FlowAlign · plain")
                out_var = gr.Video(label="FlowAlign · + phase")
            desc = gr.Markdown()
    go.click(run_ltx_video,
             [mode, up_video, src_prompt, src_caption, edit_prompt, w, zeta, sbn_cut, phase,
              frames, width, height, steps, seed],
             [out_base, out_var, desc])


SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_runs")


def _list_runs(tab):
    """Saved-run folder names for a tab, newest first."""
    if not os.path.isdir(SAVE_DIR):
        return []
    return sorted((n for n in os.listdir(SAVE_DIR)
                   if n.startswith(tab + "_") and os.path.isdir(os.path.join(SAVE_DIR, n))),
                  reverse=True)


def _save_run(tab, names, base, edit, *vals):
    """Write a timestamped folder with the output images + a config.json that load can replay.
    Image-valued inputs are saved as PNGs and referenced by filename; tuples (RangeSliders) as lists."""
    import json, time
    import gradio as gr
    d = os.path.join(SAVE_DIR, f"{tab}_{time.strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(d, exist_ok=True)
    cfg = {}
    for n, v in zip(names, vals):
        if isinstance(v, Image.Image):
            v.save(os.path.join(d, f"input_{n}.png"))
            cfg[n] = {"__image__": f"input_{n}.png"}
        else:
            cfg[n] = list(v) if isinstance(v, tuple) else v
    if base is not None:
        base.save(os.path.join(d, "baseline.png"))
    if edit is not None:
        edit.save(os.path.join(d, "edited.png"))
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump({"tab": tab, "model": MODEL_NAME, "config": cfg}, f, indent=2, default=str)
    return f"Saved → {os.path.basename(d)}", gr.update(choices=_list_runs(tab))


def _load_run(tab, names, run_name):
    """Replay a saved run: value-updates for every input (in `names` order) + the saved images."""
    import json
    import gradio as gr
    if not run_name:
        return [gr.update() for _ in names] + [gr.update(), gr.update(), "pick a saved run"]
    d = os.path.join(SAVE_DIR, run_name)
    with open(os.path.join(d, "config.json")) as f:
        cfg = json.load(f)["config"]
    updates = []
    for n in names:
        v = cfg.get(n)
        if isinstance(v, dict) and "__image__" in v:
            v = Image.open(os.path.join(d, v["__image__"]))
        elif isinstance(v, list):
            v = tuple(v)
        updates.append(gr.update(value=v))
    def _png(name):
        p = os.path.join(d, name)
        return Image.open(p) if os.path.exists(p) else None
    return updates + [_png("baseline.png"), _png("edited.png"), f"loaded {run_name}"]


def _save_load_ui(tab, names, inputs, out_base, out_edit, vis=None):
    """Add a Save/Load accordion under a tab. `names`/`inputs` are parallel; `vis`, if given, is
    (visibility_fn, op_component, vis_outputs) re-run after load so dependent controls re-show."""
    import gradio as gr
    with gr.Accordion("Save / load run", open=False):
        with gr.Row():
            save_btn = gr.Button("💾 Save this run")
            runs = gr.Dropdown(_list_runs(tab), label="Saved runs", scale=2)
            load_btn = gr.Button("Load")
        status = gr.Markdown()

    def do_save(base, edit, *vals):
        return _save_run(tab, names, base, edit, *vals)

    def do_load(run_name):
        return _load_run(tab, names, run_name)

    save_btn.click(do_save, [out_base, out_edit] + inputs, [status, runs])
    ev = load_btn.click(do_load, runs, inputs + [out_base, out_edit, status])
    if vis is not None:
        fn, op_comp, vis_out = vis
        ev.then(fn, op_comp, vis_out)


COMPARE_CSS = """
.img-pair:fullscreen { background:#000; padding:0; gap:4px; align-items:center; }
.img-pair:fullscreen > * { flex:1 1 0; min-width:0; height:100vh; }
.img-pair:fullscreen img { height:100%; width:100%; object-fit:contain; }
"""


def _image_pair(label_base, label_edit, elem_id):
    """A baseline/edit image Row plus a button that opens the pair side by side
    in browser fullscreen. Returns (out_base, out_edit)."""
    import gradio as gr
    with gr.Row(elem_id=elem_id, elem_classes="img-pair"):
        out_base = gr.Image(label=label_base, type="pil")
        out_edit = gr.Image(label=label_edit, type="pil")
    gr.Button("⛶ View side by side (fullscreen)", size="sm").click(
        fn=None, inputs=None, outputs=None,
        js=f"() => {{ const el = document.getElementById('{elem_id}'); "
           f"if (el) el.requestFullscreen(); }}")
    return out_base, out_edit


def build_ui():
    import gradio as gr
    with gr.Blocks(title="Spectral image editing — velocity, token, latent & AdaIN",
                   css=COMPARE_CSS) as demo:
        gr.Markdown("# Spectral image editing (E24–E39)\n"
                    "Four ways to steer a diffusion model in frequency space. **Velocity modulation** "
                    "(SD3.5, real CFG) edits the CFG *velocity* `v_w` toward the unconditional `v_∅` "
                    "*during* generation; **Token modulation** (Flux) edits the text embedding's token-axis "
                    "spectrum once up front; **Latent modulation** (Flux) edits the image latent's 2D radial "
                    "spectrum during generation; **Spectral AdaIN** (Flux) is a single-pass self-AdaIN that resets each radial "
                    "band's magnitude (mean+std) to absolute targets you choose (global or 3-band); "
                    "**RF inversion** (Flux) inverts a real image to noise, records its per-step spectrum, "
                    "and clamps the low band back to that trajectory while an edit prompt regenerates. "
                    "Left = baseline, right = edit, same seed.")
        with gr.Tabs():
            with gr.TabItem("Velocity modulation"):
                _velocity_tab()
            with gr.TabItem("Token modulation"):
                _token_tab()
            with gr.TabItem("Latent modulation"):
                _latent_tab()
            with gr.TabItem("Spectral AdaIN"):
                _adain_tab()
            with gr.TabItem("RF inversion"):
                _invert_tab()
            with gr.TabItem("FlowEdit"):
                _flowedit_tab()
            with gr.TabItem("FlowAlign"):
                _flowalign_tab()
            with gr.TabItem("LTX Video FlowAlign"):
                _ltx_video_tab()
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
    ap = argparse.ArgumentParser(description="Spectral image-editing demo (velocity + token + latent).")
    ap.add_argument("--model", choices=list(MODELS), default="sd3.5-medium",
                    help="which model to load; only this one goes on the GPU. "
                         "sd3.5-medium (default) = real CFG, for the Velocity tab; "
                         "flux-dev / flux-schnell = the Token/Latent tabs; "
                         "ltx = LTX-Video, for the LTX Video FlowAlign tab.")
    ap.add_argument("--port", type=int, default=7860, help="Gradio server port.")
    ap.add_argument("--share", action="store_true", help="create a public Gradio link.")
    a = ap.parse_args()
    MODEL = MODELS[a.model]; MODEL_NAME = a.model; REPO = MODEL["repo"]
    print(f"[demo] model={a.model} ({REPO})", flush=True)
    _patch_gradio_schema_bug()
    PIPE = load_pipe()
    build_ui().launch(server_name="0.0.0.0", server_port=a.port, share=a.share,
                      show_api=False, show_error=True)
