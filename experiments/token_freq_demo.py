"""Interactive browser demo for token-axis text-frequency editing (E24/E30/E32 toolkit).

A Gradio app that lets you type prompts, pick a token-axis FFT operation on Flux's T5
sequence embedding, turn the knobs, and see the edited image next to the unmodified
baseline. It is a thin wrapper around the ops already in `text_spectral_ops.py` plus the
per-object span finder from `e32_object_freq.py`; the only new machinery is keeping the
text encoders loaded so arbitrary prompts can be encoded on the fly (the experiment loaders
drop them to save memory).

Operations exposed:
  baseline                 -- unmodified prompt (the reference)
  low-pass / high-pass     -- keep only low / high token-frequencies (E30 band_filter)
  band gain                -- amplify/attenuate one band, continuous knob (E30 band_gain)
  notch                    -- zero one band (E30 band_notch)
  phase-only / mag-only    -- E30 probe: which carries the content?
  two-prompt band-swap     -- low(A)+high(B) merge (E24/E30 band_swap)
  two-prompt band-blend    -- soft crossover merge (E24/E30 band_blend)
  two-prompt lerp          -- plain token-space interpolation (the merge baseline)
  per-object band gain     -- E32: windowed band gain on ONE object's token span

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
import text_spectral_ops as TS          # light (torch only)

REPO = "black-forest-labs/FLUX.1-dev"   # == e7_flux_phase.REPO (hardcoded to avoid its
                                        # matplotlib/heavy import chain in the demo)

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
NEEDS_CUT = {"low-pass", "high-pass", "band gain", "notch", "phase band-keep", "phase gain",
             "two-prompt band-swap", "two-prompt band-blend"}
NEEDS_BAND = {"band gain", "notch", "phase band-keep", "phase gain", "per-object band gain"}
NEEDS_GAIN = {"band gain", "phase gain", "per-object band gain"}

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
- **cut** — normalised frequency `0..1` (`1` = Nyquist, the fastest token-to-token
  alternation) that splits the spectrum into a low side `[0,cut]` and a high side `[cut,1]`.
  It sets *where* the boundary is, not which side is acted on.
- **band (low / high)** — *which* side of `cut` an operator scales/zeroes. Needed because
  one `cut` makes two bands and `gain`/notch only touch one of them.
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
    print("[demo] Flux loaded (encoders kept on GPU)", flush=True)
    return pipe


# ---------------------------------------------------------------------------
# encode / generate
# ---------------------------------------------------------------------------

def encode(prompt):
    """prompt -> (pe_cpu (1,512,4096), ppe_cpu (1,4096), L real tokens)."""
    with torch.no_grad():
        pe, ppe, _ = PIPE.encode_prompt(
            prompt=prompt, prompt_2=prompt, device="cuda",
            num_images_per_prompt=1, max_sequence_length=512)
    tok = PIPE.tokenizer_2(prompt, max_length=512, truncation=True, return_tensors="pt")
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

def apply_op(op, promptA, peA, ppeA, LA, peB, ppeB, LB, p):
    on_span = lambda fn, L=LA: TS.apply_on_span(fn, peA, L)
    if op == "baseline":
        return peA, ppeA, "baseline (unmodified)"
    if op == "low-pass":
        c = p["cut"]
        return on_span(lambda x: TS.band_filter_1d(x, 0.0, c)), ppeA, f"low-pass keep [0,{c:.2f}]"
    if op == "high-pass":
        c = p["cut"]
        return on_span(lambda x: TS.band_filter_1d(x, c, 1.0, keep_dc=True)), ppeA, \
            f"high-pass keep [{c:.2f},1]+DC"
    if op == "band gain":
        c = p["cut"]; lo, hi = (0.0, c) if p["band"] == "low" else (c, 1.0); g = p["gain"]
        return on_span(lambda x: TS.band_gain_1d(x, lo, hi, g)), ppeA, \
            f"{p['band']} band x{g:.2f} (cut {c:.2f})"
    if op == "notch":
        c = p["cut"]; lo, hi = (0.0, c) if p["band"] == "low" else (c, 1.0)
        return on_span(lambda x: TS.band_notch_1d(x, lo, hi)), ppeA, \
            f"notch {p['band']} band [{lo:.2f},{hi:.2f}]"
    if op == "phase-only":
        return on_span(_phase_only), ppeA, "phase-only (magnitude=1)"
    if op == "mag-only":
        return on_span(_mag_only), ppeA, "mag-only (phase=0)"
    if op == "phase band-keep":
        c = p["cut"]; lo, hi = (0.0, c) if p["band"] == "low" else (c, 1.0)
        return on_span(lambda x: TS.band_phase_filter_1d(x, lo, hi)), ppeA, \
            f"phase {p['band']}-pass: phase kept in [{lo:.2f},{hi:.2f}], =0 elsewhere (mag kept)"
    if op == "phase gain":
        c = p["cut"]; lo, hi = (0.0, c) if p["band"] == "low" else (c, 1.0); g = p["gain"]
        return on_span(lambda x: TS.band_phase_gain_1d(x, lo, hi, g)), ppeA, \
            f"phase angle x{g:.2f} in {p['band']} band [{lo:.2f},{hi:.2f}] (mag kept)"
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
        lo, hi = (0.0, OBJ_CUT) if p["band"] == "low" else (OBJ_CUT, 1.0)
        g = p["gain"]; bins = (b - a) // 2 + 1
        return TS.apply_on_subspan(lambda x: TS.band_gain_1d(x, lo, hi, g), peA, a, b), \
            ppeA, f"object '{phrase}' span [{a},{b}) {bins} bins · {p['band']} band x{g:.2f}"
    raise ValueError(f"unknown op {op}")


# ---------------------------------------------------------------------------
# Gradio handler (with baseline caching)
# ---------------------------------------------------------------------------

_BASE_CACHE = {}
_ENC_CACHE = {}


def _encode_cached(prompt):
    if prompt not in _ENC_CACHE:
        _ENC_CACHE[prompt] = encode(prompt)
    return _ENC_CACHE[prompt]


def run(promptA, promptB, object_phrase, op, cut, gain, band, width, alpha,
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

    p = dict(cut=cut, gain=gain, band=band, width=width, alpha=alpha,
             object_phrase=object_phrase)
    try:
        peN, ppeN, desc = apply_op(op, promptA, peA, ppeA, LA, peB, ppeB, LB, p)
    except Exception as e:
        return base, None, f"error: {e}"

    if op == "baseline":
        return base, base, "baseline (same as left)"
    edited = generate(peN, ppeN, seed, steps, guidance, size)
    return base, edited, desc


def _visibility(op):
    import gradio as gr
    return [
        gr.update(visible=op in TWO_PROMPT),                 # prompt B
        gr.update(visible=op == "per-object band gain"),     # object phrase
        gr.update(visible=op in NEEDS_CUT),                  # cut
        gr.update(visible=op in NEEDS_BAND),                 # band
        gr.update(visible=op in NEEDS_GAIN),                 # gain
        gr.update(visible=op == "two-prompt band-blend"),    # width
        gr.update(visible=op == "two-prompt lerp"),          # alpha
        gr.update(value=HELP.get(op, "")),                   # help text
    ]


def build_ui():
    import gradio as gr
    with gr.Blocks(title="Token-frequency text editing") as demo:
        gr.Markdown("# Token-axis text-frequency editing (E24 / E30 / E32)\n"
                    "Edit Flux's T5 token-sequence embedding, then compare to the baseline. "
                    "Same seed left vs right.")
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
                band = gr.Radio(["low", "high"], value="high", label="band",
                                info="WHICH side of cut to act on: high=detail/style, low=coarse gist.",
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
                with gr.Row():
                    seed = gr.Number(value=0, precision=0, label="seed",
                                     info="Fix to compare fairly; baseline cached per seed.")
                    steps = gr.Slider(4, 28, value=16, step=1, label="steps",
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
                  [promptB, object_phrase, cut, band, gain, width, alpha, helpbox])
        go.click(run,
                 [promptA, promptB, object_phrase, op, cut, gain, band, width, alpha,
                  seed, steps, guidance, size],
                 [out_base, out_edit, desc])
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
    _patch_gradio_schema_bug()
    PIPE = load_pipe()
    build_ui().launch(server_name="0.0.0.0", server_port=7860, share=False,
                      show_api=False, show_error=True)
