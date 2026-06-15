"""E24: spectral surgery on the TEXT conditioning -- can token-frequency bands
merge or edit images?

Motivation. FNet (Lee-Thorp et al. 2021) replaced Transformer self-attention with a
parameter-free DFT over the token-sequence axis and still reached ~92-97% of BERT,
evidence that the token-axis FFT is a meaningful token-mixing basis: low
token-frequencies carry slow / global meaning (DC = the bag-of-words mean), high
token-frequencies carry sharp token-to-token detail. This is the text-side analogue
of the project's E18 image-latent Fourier band-swap ("AdaIN-in-Fourier"). E24 FFTs
Flux's T5 sequence embedding along the token axis, swaps/blends/filters bands between
prompts, and asks whether images can be merged (prompt A's subject + prompt B's
detail) or edited (inject a style prompt's high band) purely through the conditioning.

The hook. Flux pre-encodes a prompt to (prompt_embeds (1,512,4096), pooled (1,4096))
and feeds them straight to the transformer's cross-attention (see e10's `gen_emb`).
We modify `prompt_embeds` before generation. T5 pads to 512 tokens, so every spectral
op runs only on the real-token span [:L] (captured from the T5 tokenizer) and the
padding is reattached -- otherwise the content->padding cliff pollutes the high band.

Risk this tests first. Recombined embeddings can leave the text-encoder manifold and
produce garbage / ignored conditioning. Hence `--part probe` runs BEFORE any merge to
confirm band-filtered embeddings still yield coherent images; `--renorm` optionally
rescales each recombined token to its source per-token norm as a mitigation.

Parts (--part, comma list):
  probe   -- single prompt: full / dc_only / low / high / high+dc variants -> what
             does each token-frequency band control? (manifold sanity + which band is
             the subject).
  merge   -- prompt A + B: low_A+high_B, low_B+high_A, soft blend over a cut grid,
             token-axis phase/magnitude swap, plus baselines lerp@0.5, pooled-swap,
             FNet 2D swap.
  edit    -- base prompt + style prompt: inject the style's high band into the base
             over a cut grid; compare to the full style prompt and a pooled-only inject.
  analyze -- CLIP-T attribution (sim to A vs B), aesthetic, ImageReward; token-axis
             band-power plot; montage grids; report.json; self-contained index.html.

Memory: same Flux load as E10 (bnb4). Per the flux-gen-ops note, if memory-contended
run the generation parts one prompt-set at a time in separate processes.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS
from e7_flux_phase import REPO, SIZE
from e10_cfg_spectral import gen_emb
from e9_clipt import agg, load_clip, clip_scores
from fidelity_metrics import (load_aesthetic, aesthetic_scores,
                              load_imagereward, imagereward_scores)
import text_spectral_ops as TS

OUT = os.path.join(RESULTS, "e24")

# Prompt sets chosen for clean CLIP attribution (distinct subjects / styles).
PROBE_PROMPTS = [
    ("cat", "a photograph of an orange tabby cat sitting on a windowsill"),
    ("car", "a photograph of a red sports car on a mountain road"),
    ("castle", "an oil painting of a medieval castle on a green hill at sunset"),
]
MERGE_PAIRS = [  # (key, prompt A, prompt B)
    ("cat_car", "a photograph of an orange tabby cat",
     "a photograph of a red sports car"),
    ("castle_forest", "an oil painting of a medieval castle",
     "a dense pine forest in thick fog"),
]
EDIT_PAIRS = [  # (key, base/content, style/attribute)
    ("house_vangogh", "a house by a lake",
     "in the style of Van Gogh, thick swirling brushstrokes, impasto"),
    ("portrait_cyber", "a portrait of a woman",
     "cyberpunk neon city at night, glowing lights, futuristic"),
]
CUTS = [0.15, 0.25, 0.40]  # normalised token-frequency crossover sweep


# ---------------------------------------------------------------------------
# Flux loader that also records the real-token span L per prompt
# ---------------------------------------------------------------------------

def load_flux_preencoded_lens(prompts):
    """Like e10.load_flux_preencoded but also returns lens[txt] = real T5 token
    count (non-pad), captured from the T5 tokenizer BEFORE the encoders are dropped.
    Returns (pipe, embeds, lens) with embeds[txt]=(pe_cpu (1,512,4096), ppe_cpu)."""
    import gc
    from diffusers import (FluxPipeline, FluxTransformer2DModel,
                           BitsAndBytesConfig)
    qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.bfloat16)
    tr = FluxTransformer2DModel.from_pretrained(
        REPO, subfolder="transformer", quantization_config=qc,
        torch_dtype=torch.bfloat16)
    pipe = FluxPipeline.from_pretrained(REPO, transformer=tr,
                                        torch_dtype=torch.bfloat16)
    pipe.set_progress_bar_config(disable=True)
    pipe.text_encoder.to("cuda")
    pipe.text_encoder_2.to("cuda")
    embeds, lens = {}, {}
    with torch.no_grad():
        for txt in dict.fromkeys(prompts):  # dedup, keep order
            pe, ppe, _ = pipe.encode_prompt(
                prompt=txt, prompt_2=txt, device="cuda",
                num_images_per_prompt=1, max_sequence_length=512)
            embeds[txt] = (pe.cpu(), ppe.cpu())
            tok = pipe.tokenizer_2(txt, max_length=512, truncation=True,
                                   return_tensors="pt")
            lens[txt] = int(tok.attention_mask.sum())
    pipe.text_encoder = pipe.text_encoder_2 = None
    pipe.tokenizer = pipe.tokenizer_2 = None
    gc.collect()
    torch.cuda.empty_cache()
    pipe.vae.to("cuda")
    print(f"[e24] pre-encoded {len(embeds)} prompts "
          f"(spans {sorted(set(lens.values()))}); text encoders dropped", flush=True)
    return pipe, embeds, lens


def _gen(pipe, pe, ppe, seed, args, png, lp=None):
    """Cached single generation from (possibly modified) embeddings. true_cfg=1 ->
    single pass (the trained field), so no negative embeds needed."""
    if os.path.exists(png) and (lp is None or os.path.exists(lp)):
        return
    img, lat = gen_emb(pipe, (pe.cpu(), ppe.cpu()), None, seed, 1.0,
                       args.guidance, args.steps)
    img.save(png)
    if lp:
        torch.save(lat, lp)
    print(f"[e24] gen {os.path.relpath(png, OUT)} s{seed}", flush=True)


def _span_pe(pe, fn, L):
    """Apply token-axis op `fn` on the real span of a (1,512,D) embedding."""
    return TS.apply_on_span(fn, pe, L)


def _maybe_renorm(args, new_pe, ref_pe, L):
    if not args.renorm:
        return new_pe
    out = new_pe.clone()
    out[:, :L] = TS.renorm_per_token(new_pe[:, :L], ref_pe[:, :L])
    return out


# ---------------------------------------------------------------------------
# Part: probe (single prompt -> what each band controls)
# ---------------------------------------------------------------------------

def probe_variants(pe, L, cut):
    """Dict name -> modified prompt_embeds for a single prompt."""
    return {
        "full": pe,
        "dc_only": _span_pe(pe, lambda x: TS.band_filter_1d(x, 0.0, 0.0, keep_dc=True), L),
        "low": _span_pe(pe, lambda x: TS.band_filter_1d(x, 0.0, cut, keep_dc=True), L),
        "high": _span_pe(pe, lambda x: TS.band_filter_1d(x, cut, 1.0, keep_dc=False), L),
        "high_dc": _span_pe(pe, lambda x: TS.band_filter_1d(x, cut, 1.0, keep_dc=True), L),
    }


def run_probe(args, pipe, embeds, lens):
    for key, prompt in PROBE_PROMPTS[: args.num_prompts]:
        d = os.path.join(OUT, "probe", key)
        os.makedirs(d, exist_ok=True)
        pe, ppe = embeds[prompt]
        for name, mpe in probe_variants(pe, lens[prompt], args.cut).items():
            for s in range(args.seeds):
                _gen(pipe, mpe, ppe, s, args,
                     os.path.join(d, f"{name}_s{s}.png"))
        print(f"[e24] probe {key} done", flush=True)


# ---------------------------------------------------------------------------
# Part: merge (prompt A + B)
# ---------------------------------------------------------------------------

def merge_conditions(peA, ppeA, peB, ppeB, L, args):
    """Dict cond -> (prompt_embeds, pooled) for merging A and B on a shared span L.
    pooled (global subject) follows the low-band source unless noted."""
    cut = args.cut
    conds = {
        "pure_A": (peA, ppeA),
        "pure_B": (peB, ppeB),
        "lowA_highB": (_maybe_renorm(args,
            _span_pe(peA, lambda x: TS.band_swap_1d(x, peB[:, :L], cut), L), peA, L), ppeA),
        "lowB_highA": (_maybe_renorm(args,
            _span_pe(peB, lambda x: TS.band_swap_1d(x, peA[:, :L], cut), L), peB, L), ppeB),
        "phaseA_magB": (_span_pe(peA,
            lambda x: TS.phase_mag_split_1d(x, peB[:, :L])["phaseA_magB"], L), ppeA),
        "magA_phaseB": (_span_pe(peA,
            lambda x: TS.phase_mag_split_1d(x, peB[:, :L])["magA_phaseB"], L), ppeA),
        # baselines
        "lerp": (_span_pe(peA, lambda x: TS.lerp_embeds(x, peB[:, :L], 0.5), L),
                 TS.lerp_embeds(ppeA, ppeB, 0.5)),
        "pooled_swap": (peA, ppeB),          # seq from A, global from B
        "fnet_swap": (_span_pe(peA, lambda x: TS.fnet_swap_2d(x, peB[:, :L], cut), L), ppeA),
    }
    # soft-blend crossover sweep over cut location
    for c in CUTS:
        conds[f"blend_c{c:g}"] = (
            _span_pe(peA, lambda x, c=c: TS.band_blend_1d(x, peB[:, :L], c), L), ppeA)
    return conds


def run_merge(args, pipe, embeds, lens):
    for key, pA, pB in MERGE_PAIRS[: args.num_prompts]:
        d = os.path.join(OUT, "merge", key)
        os.makedirs(d, exist_ok=True)
        peA, ppeA = embeds[pA]
        peB, ppeB = embeds[pB]
        L = min(lens[pA], lens[pB])
        for cond, (mpe, mppe) in merge_conditions(peA, ppeA, peB, ppeB, L, args).items():
            for s in range(args.seeds):
                _gen(pipe, mpe, mppe, s, args, os.path.join(d, f"{cond}_s{s}.png"))
        print(f"[e24] merge {key} done (span L={L})", flush=True)


# ---------------------------------------------------------------------------
# Part: edit (base + style)
# ---------------------------------------------------------------------------

def edit_conditions(peB, ppeB, peS, ppeS, L, args):
    """Inject the style prompt's high band into the base over a cut grid (lower cut
    -> more style high-frequency content)."""
    conds = {
        "base": (peB, ppeB),
        "full_style": (peS, ppeS),
        "pooled_inject": (peB, ppeS),  # base seq + style global
    }
    for c in CUTS:
        conds[f"inject_c{c:g}"] = (_maybe_renorm(args,
            _span_pe(peB, lambda x, c=c: TS.band_swap_1d(x, peS[:, :L], c), L), peB, L), ppeB)
    return conds


def run_edit(args, pipe, embeds, lens):
    for key, base, style in EDIT_PAIRS[: args.num_prompts]:
        d = os.path.join(OUT, "edit", key)
        os.makedirs(d, exist_ok=True)
        peB, ppeB = embeds[base]
        peS, ppeS = embeds[style]
        L = min(lens[base], lens[style])
        for cond, (mpe, mppe) in edit_conditions(peB, ppeB, peS, ppeS, L, args).items():
            for s in range(args.seeds):
                _gen(pipe, mpe, mppe, s, args, os.path.join(d, f"{cond}_s{s}.png"))
        print(f"[e24] edit {key} done (span L={L})", flush=True)


# ---------------------------------------------------------------------------
# generation driver (loads Flux once for all requested gen parts)
# ---------------------------------------------------------------------------

def _prompts_for(parts, n):
    p = []
    if "probe" in parts:
        p += [pr for _, pr in PROBE_PROMPTS[:n]]
    if "merge" in parts:
        for _, a, b in MERGE_PAIRS[:n]:
            p += [a, b]
    if "edit" in parts:
        for _, a, b in EDIT_PAIRS[:n]:
            p += [a, b]
    return list(dict.fromkeys(p))


def run_gen(args, parts):
    prompts = _prompts_for(parts, args.num_prompts)
    if not prompts:
        return
    pipe, embeds, lens = load_flux_preencoded_lens(prompts)
    if "probe" in parts:
        run_probe(args, pipe, embeds, lens)
    if "merge" in parts:
        run_merge(args, pipe, embeds, lens)
    if "edit" in parts:
        run_edit(args, pipe, embeds, lens)


# ---------------------------------------------------------------------------
# Part: analyze (metrics + plots + grids + site)
# ---------------------------------------------------------------------------

def _imgs(d):
    """{cond: ([seed,...],[PIL,...])} from a directory of <cond>_s<seed>.png."""
    out = {}
    if not os.path.isdir(d):
        return out
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".png"):
            continue
        base = fn[:-4]
        if "_s" not in base:        # skip non-condition files (e.g. grid.png)
            continue
        cond, s = base.rsplit("_s", 1)
        if not s.isdigit():
            continue
        out.setdefault(cond, ([], []))
        out[cond][0].append(int(s))
        out[cond][1].append(Image.open(os.path.join(d, fn)).convert("RGB"))
    return out


def _score(clip, aes, ir, prompt, imgs, paths):
    cm, cp = clip
    return {
        "clip": agg(clip_scores(cm, cp, prompt, imgs)),
        "aesthetic": agg(aesthetic_scores(aes, cm, cp, imgs)),
        "imagereward": agg(imagereward_scores(ir, prompt, paths)),
    }


def run_analyze(args):
    clip = load_clip(args.clip_model)
    aes = load_aesthetic()
    ir = load_imagereward()
    report = {"params": vars(args), "probe": {}, "merge": {}, "edit": {}}

    # probe: each variant scored vs its own prompt
    for key, prompt in PROBE_PROMPTS[: args.num_prompts]:
        d = os.path.join(OUT, "probe", key)
        ents = {}
        for cond, (seeds, imgs) in _imgs(d).items():
            paths = [os.path.join(d, f"{cond}_s{s}.png") for s in seeds]
            ents[cond] = _score(clip, aes, ir, prompt, imgs, paths)
        report["probe"][key] = {"prompt": prompt, "conds": ents}
        _grid(d, key, "probe")

    # merge: every condition scored vs A AND vs B (attribution)
    for key, pA, pB in MERGE_PAIRS[: args.num_prompts]:
        d = os.path.join(OUT, "merge", key)
        ents = {}
        for cond, (seeds, imgs) in _imgs(d).items():
            paths = [os.path.join(d, f"{cond}_s{s}.png") for s in seeds]
            ents[cond] = {
                "clip_A": agg(clip_scores(*clip, pA, imgs)),
                "clip_B": agg(clip_scores(*clip, pB, imgs)),
                "aesthetic": agg(aesthetic_scores(aes, *clip, imgs)),
            }
        report["merge"][key] = {"A": pA, "B": pB, "conds": ents}
        _grid(d, key, "merge")

    # edit: scored vs base content AND vs style
    for key, base, style in EDIT_PAIRS[: args.num_prompts]:
        d = os.path.join(OUT, "edit", key)
        ents = {}
        for cond, (seeds, imgs) in _imgs(d).items():
            ents[cond] = {
                "clip_base": agg(clip_scores(*clip, base, imgs)),
                "clip_style": agg(clip_scores(*clip, style, imgs)),
                "aesthetic": agg(aesthetic_scores(aes, *clip, imgs)),
            }
        report["edit"][key] = {"base": base, "style": style, "conds": ents}
        _grid(d, key, "edit")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    _band_power_plot(args)
    _site(report)
    print(f"[e24] wrote report.json + plots + index.html", flush=True)


def _grid(d, key, part, seed=0):
    """Labeled montage of all conditions at one seed -> <d>/grid.png."""
    if not os.path.isdir(d):  # part not generated (e.g. merge/edit during a probe-only smoke)
        return
    items = []
    for fn in sorted(os.listdir(d)):
        if fn.endswith(f"_s{seed}.png"):
            items.append((fn[:-len(f"_s{seed}.png")], os.path.join(d, fn)))
    if not items:
        return
    thumb = 256
    cols = min(len(items), 6)
    rows = (len(items) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * thumb, rows * (thumb + 18)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (label, p) in enumerate(items):
        r, c = divmod(i, cols)
        im = Image.open(p).convert("RGB").resize((thumb, thumb))
        canvas.paste(im, (c * thumb, r * (thumb + 18) + 18))
        draw.text((c * thumb + 3, r * (thumb + 18) + 3), label, fill="black")
    canvas.save(os.path.join(d, "grid.png"))


def _band_power_plot(args):
    """Illustrate the token-axis 'PSD' on a representative latent-free example: the
    band-power of each probe prompt's full embedding (shows the spectrum being cut)."""
    # uses cached latents? no -- this is conceptual on the embedding spectrum, which
    # requires the encoder. Skip silently if a cached spectra file isn't present.
    sp = os.path.join(OUT, "token_psd.pt")
    if not os.path.exists(sp):
        return
    data = torch.load(sp, weights_only=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, (c, p) in data.items():
        ax.semilogy(c, p, marker="o", label=name)
    ax.axvline(args.cut, ls="--", color="k", lw=1, label=f"cut={args.cut}")
    ax.set(xlabel="normalised token frequency", ylabel="power",
           title="Token-axis band power (text-conditioning 'PSD')")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "token_psd.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# self-contained HTML explainer
# ---------------------------------------------------------------------------

def _fmt(a):
    return f"{a['mean']:.3f}" if a else "—"


# Inline schematic of the idea (self-contained -- no external asset). Stage 1:
# decompose a prompt's token embeddings by frequency along the token axis. Stage 2:
# recombine bands from two prompts (merge) or a base+style prompt (edit).
_SCHEMATIC_SVG = '''
<figure style="margin:1.2em 0">
<svg viewBox="0 0 760 300" xmlns="http://www.w3.org/2000/svg" role="img"
     aria-label="E24 schematic"
     style="width:100%;max-width:760px;border:1px solid #ddd;border-radius:6px;background:#fff">
  <defs><marker id="ar" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto">
    <path d="M0,0 L7,3 L0,6 Z" fill="#555"/></marker></defs>
  <style>
    .b{font:12px system-ui,sans-serif;fill:#1b1b1b}
    .s{font:10px system-ui,sans-serif;fill:#666}
    .t{font:bold 12px system-ui,sans-serif;fill:#111}
    .box{fill:#f3f6fb;stroke:#9fb6d6}
    line{stroke:#555;stroke-width:1.4;marker-end:url(#ar)}
  </style>
  <text x="14" y="20" class="t">1 · Decompose: FFT the token embeddings along the TOKEN axis</text>
  <rect class="box" x="14" y="32" width="70" height="28"/><text x="22" y="50" class="b">Prompt</text>
  <line x1="84" y1="46" x2="106" y2="46"/>
  <rect class="box" x="108" y="32" width="40" height="28"/><text x="116" y="50" class="b">T5</text>
  <line x1="148" y1="46" x2="170" y2="46"/>
  <rect x="174" y="30" width="9" height="32" fill="#cfe0f6" stroke="#9fb6d6"/>
  <rect x="184" y="30" width="9" height="32" fill="#bcd3f0" stroke="#9fb6d6"/>
  <rect x="194" y="30" width="9" height="32" fill="#cfe0f6" stroke="#9fb6d6"/>
  <rect x="204" y="30" width="9" height="32" fill="#bcd3f0" stroke="#9fb6d6"/>
  <rect x="214" y="30" width="9" height="32" fill="#cfe0f6" stroke="#9fb6d6"/>
  <text x="174" y="74" class="s">token embeds E (L x 4096)</text>
  <line x1="226" y1="46" x2="248" y2="46"/>
  <rect class="box" x="250" y="32" width="66" height="28"/><text x="258" y="50" class="b">FFT(tokens)</text>
  <line x1="316" y1="46" x2="338" y2="46"/>
  <rect x="340" y="32" width="120" height="28" fill="#dbe9ff" stroke="#5b8bd0"/>
  <rect x="340" y="32" width="46" height="28" fill="#8fc0ff" stroke="#5b8bd0"/>
  <text x="350" y="50" class="b">low</text><text x="424" y="50" class="b">high</text>
  <text x="476" y="44" class="s">low (DC+low freq) = PHASE -> subject / identity</text>
  <text x="476" y="58" class="s">high = per-token detail / style</text>
  <text x="14" y="108" class="t">2 · Recombine bands from two prompts</text>
  <rect x="14" y="120" width="60" height="26" fill="#8fc0ff" stroke="#5b8bd0"/><text x="22" y="137" class="b">low A</text>
  <rect x="74" y="120" width="62" height="26" fill="#ffce8f" stroke="#d09a5b"/><text x="82" y="137" class="b">high B</text>
  <line x1="136" y1="133" x2="158" y2="133"/>
  <rect class="box" x="160" y="120" width="56" height="26"/><text x="168" y="137" class="b">IFFT</text>
  <line x1="216" y1="133" x2="238" y2="133"/>
  <rect class="box" x="240" y="120" width="50" height="26"/><text x="248" y="137" class="b">Flux</text>
  <line x1="290" y1="133" x2="312" y2="133"/>
  <rect x="314" y="118" width="28" height="30" fill="#eee" stroke="#999"/>
  <text x="350" y="137" class="s">MERGE -> snaps to A (the low-band owner); no clean blend</text>
  <rect x="14" y="158" width="60" height="26" fill="#8fc0ff" stroke="#5b8bd0"/><text x="18" y="175" class="b">low base</text>
  <rect x="74" y="158" width="62" height="26" fill="#ffce8f" stroke="#d09a5b"/><text x="80" y="175" class="b">high style</text>
  <line x1="136" y1="171" x2="158" y2="171"/>
  <rect class="box" x="160" y="158" width="56" height="26"/><text x="168" y="175" class="b">IFFT</text>
  <line x1="216" y1="171" x2="238" y2="171"/>
  <rect class="box" x="240" y="158" width="50" height="26"/><text x="248" y="175" class="b">Flux</text>
  <line x1="290" y1="171" x2="312" y2="171"/>
  <rect x="314" y="156" width="28" height="30" fill="#eee" stroke="#999"/>
  <text x="350" y="175" class="s">EDIT -> high-band of style acts as a style-strength knob (partial)</text>
  <text x="14" y="214" class="s">Baseline to beat = token-space lerp(A,B). Basis: FNet (2021) showed the token-axis DFT is an effective token-mixing transform.</text>
</svg>
<figcaption style="font:11px system-ui;color:#777">Schematic — E24 token-axis spectral surgery on the text conditioning.</figcaption>
</figure>
'''


def _site(report):
    h = ["<!doctype html><meta charset=utf-8><title>E24 text-spectral</title>",
         "<style>body{font:14px/1.5 system-ui;max-width:1100px;margin:2rem auto;"
         "padding:0 1rem;color:#222}h1,h2{line-height:1.2}code{background:#f0f0f0;"
         "padding:1px 4px;border-radius:3px}table{border-collapse:collapse;margin:1em 0}"
         "td,th{border:1px solid #ccc;padding:3px 8px;text-align:right}th:first-child,"
         "td:first-child{text-align:left}img{max-width:100%;border:1px solid #ddd;"
         "margin:.5em 0}.note{background:#fff8e1;padding:.6em 1em;border-left:3px solid "
         "#f0c000}</style>",
         "<h1>E24 — Spectral surgery on the text conditioning</h1>",
         "<p><b>Question.</b> Can the <i>high frequencies of text-token embeddings</i> "
         "be used to merge or edit images? <b>Basis.</b> FNet (Lee-Thorp et al. 2021) "
         "showed a parameter-free DFT over the token axis is an effective token-mixing "
         "transform — low token-frequencies = global meaning (DC = bag-of-words mean), "
         "high = sharp token-to-token detail. This is the text-side analogue of E18's "
         "image-latent Fourier band-swap.</p>",
         "<p><b>Method.</b> Flux feeds a prompt's T5 sequence embedding "
         "<code>(1,512,4096)</code> to cross-attention. We FFT it along the token axis "
         "(only over the real-token span; T5 pads to 512), swap/blend/filter frequency "
         "bands between prompts, and regenerate at a fixed seed. <code>cut</code> is the "
         "normalised crossover frequency (0=DC … 1=Nyquist).</p>",
         _SCHEMATIC_SVG,
         "<div class=note><b>Manifold caveat.</b> Recombined embeddings can leave the "
         "encoder manifold. The <b>probe</b> section is the sanity check: if band-filtered "
         "embeddings already produce garbage, the merge results are uninterpretable. "
         f"<code>--renorm={report['params'].get('renorm')}</code>.</div>"]

    # probe
    h.append("<h2>Probe — what each token-frequency band controls</h2>")
    for key, e in report["probe"].items():
        h.append(f"<h3>{key}: <code>{e['prompt']}</code></h3>")
        h.append(f"<img src='probe/{key}/grid.png'>")
        h.append("<table><tr><th>variant</th><th>CLIP↑</th><th>aesthetic↑</th>"
                 "<th>ImageReward↑</th></tr>")
        for cond, sc in e["conds"].items():
            h.append(f"<tr><td>{cond}</td><td>{_fmt(sc['clip'])}</td>"
                     f"<td>{_fmt(sc['aesthetic'])}</td><td>{_fmt(sc['imagereward'])}</td></tr>")
        h.append("</table>")

    # merge
    h.append("<h2>Merge — combine prompt A + prompt B</h2>")
    h.append("<p>CLIP_A / CLIP_B = similarity of the merged image to prompt A / B. "
             "If a low-band-from-A condition scores higher on A than the <code>lerp</code> "
             "baseline at equal B, the token spectrum disentangles better than plain "
             "interpolation.</p>")
    for key, e in report["merge"].items():
        h.append(f"<h3>{key}</h3><p>A=<code>{e['A']}</code> · B=<code>{e['B']}</code></p>")
        h.append(f"<img src='merge/{key}/grid.png'>")
        h.append("<table><tr><th>condition</th><th>CLIP_A</th><th>CLIP_B</th>"
                 "<th>aesthetic↑</th></tr>")
        for cond, sc in e["conds"].items():
            h.append(f"<tr><td>{cond}</td><td>{_fmt(sc['clip_A'])}</td>"
                     f"<td>{_fmt(sc['clip_B'])}</td><td>{_fmt(sc['aesthetic'])}</td></tr>")
        h.append("</table>")

    # edit
    h.append("<h2>Edit — inject a style prompt's high band into a base</h2>")
    for key, e in report["edit"].items():
        h.append(f"<h3>{key}</h3><p>base=<code>{e['base']}</code> · "
                 f"style=<code>{e['style']}</code></p>")
        h.append(f"<img src='edit/{key}/grid.png'>")
        h.append("<table><tr><th>condition</th><th>CLIP_base</th><th>CLIP_style</th>"
                 "<th>aesthetic↑</th></tr>")
        for cond, sc in e["conds"].items():
            h.append(f"<tr><td>{cond}</td><td>{_fmt(sc['clip_base'])}</td>"
                     f"<td>{_fmt(sc['clip_style'])}</td><td>{_fmt(sc['aesthetic'])}</td></tr>")
        h.append("</table>")

    with open(os.path.join(OUT, "index.html"), "w") as f:
        f.write("\n".join(h))


# ---------------------------------------------------------------------------

def main(args):
    global OUT
    if args.out_tag:
        OUT = os.path.join(RESULTS, f"e24_{args.out_tag}")
    os.makedirs(OUT, exist_ok=True)
    parts = [p.strip() for p in args.part.split(",") if p.strip()]
    gen_parts = [p for p in parts if p in ("probe", "merge", "edit")]
    if gen_parts:
        run_gen(args, gen_parts)
    if "analyze" in parts:
        run_analyze(args)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="probe,merge,edit,analyze")
    ap.add_argument("--cut", type=float, default=0.25,
                    help="normalised token-frequency low/high split")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--num_prompts", type=int, default=99,
                    help="cap on prompts/pairs per part (for smoke runs)")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=3.5,
                    help="Flux distilled guidance_scale")
    ap.add_argument("--renorm", action="store_true",
                    help="rescale recombined tokens to source per-token norm")
    ap.add_argument("--clip_model", default="openai/clip-vit-large-patch14")
    ap.add_argument("--out_tag", default="",
                    help="write to results/e24_<tag> instead of results/e24 "
                         "(keeps smoke runs from polluting the full sweep's cache)")
    main(ap.parse_args())
