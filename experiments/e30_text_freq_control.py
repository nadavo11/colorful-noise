"""E30: continuous text-frequency control & extraction (follow-up to E24).

E24 showed token-axis frequency bands of Flux's T5 sequence embedding are meaningful
and on-manifold, that merging two prompts snaps to the low-band/phase owner, and that
high-band injection is a style-strength knob. E30 goes deeper and makes the control
*continuous*, and asks what frequency filtering does to long / compositional prompts.

The signal: a prompt's T5 sequence embedding E (1, L, 4096). We FFT along the TOKEN
axis (1-D DFT per embedding channel, real-token span only) and filter/scale/swap bands,
then regenerate via the e10 `gen_emb` hook (true-CFG=1, guidance 3.5).

Parts (--part, comma list):
  probe_deep    -- one prompt: per-band knockout (notch), phase-only vs magnitude-only,
                   low/high families. Metrics: CLIP + image stats (sharpness, hf_frac,
                   colorfulness). What does each band actually control?
  continuous    -- the headline visual: image STRIPS as one knob varies --
                   (a) low-pass cutoff sweep, (b) high-band gain sweep,
                   (c) two-prompt A<->B morph (band-swap cut sweep).
  concat        -- blend vs one big prompt: compare band-swap / band-blend / lerp merges
                   against a single concatenated prompt "A and B" (CLIP_A/CLIP_B + B-VQA:
                   are BOTH objects present?).
  longprompt    -- DPG-Bench long prompts: low-pass / high-pass / per-band knockout;
                   does dropping high freq drop the tail objects while low keeps the gist?
                   Measured with VQAScore + B-VQA.
  compositional -- T2I-CompBench (color/shape/texture): band filtering vs per-object
                   attribute binding (B-VQA).
  analyze       -- report.json, strips/grids, self-contained index.html WITH a schematic.

Cluster: ship via kubectl cp (storage is not a git repo); self-gating job runs a smoke
subset, sanity-gates on CLIP, then the full set. Heavy scorers (VQAScore ~11GB, B-VQA)
load in analyze only, after Flux is freed.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, save_grid
from e10_cfg_spectral import gen_emb
from e24_text_spectral import load_flux_preencoded_lens, PROBE_PROMPTS, MERGE_PAIRS
from e9_clipt import agg, load_clip, clip_scores
from e9_bandnorm_classes import image_metrics
from fidelity_metrics import (load_aesthetic, aesthetic_scores,
                              load_imagereward, imagereward_scores)
import text_spectral_ops as TS

OUT = os.path.join(RESULTS, "e30")
CUT0 = 0.25                      # default low/high split (matches E24)
N_BANDS = 6                      # per-band knockout resolution
CUT_SWEEP = [round(0.1 + 0.8 * i / 7, 3) for i in range(8)]   # 0.1 .. 0.9
GAIN_SWEEP = [0.0, 0.5, 1.0, 1.5, 2.0]
IMG_KEYS = ["sharpness", "hf_frac", "rms_contrast", "colorfulness", "saturation"]


# ---------------------------------------------------------------------------
# generation helper (own OUT; reuse e24's loader + e10's gen_emb)
# ---------------------------------------------------------------------------

def _gen(pipe, pe, ppe, seed, args, png):
    """Cached single generation from (modified) embeddings -> PIL (also returned)."""
    if os.path.exists(png):
        return Image.open(png).convert("RGB")
    img, _ = gen_emb(pipe, (pe.cpu(), ppe.cpu()), None, seed, 1.0,
                     args.guidance, args.steps)
    os.makedirs(os.path.dirname(png), exist_ok=True)
    img.save(png)
    print(f"[e30] gen {os.path.relpath(png, OUT)}", flush=True)
    return img


def _span(pe, fn, L):
    return TS.apply_on_span(fn, pe, L)


def _phase_only(E):
    F = torch.fft.rfft(E.float(), dim=1)
    out = torch.fft.irfft(torch.polar(torch.ones_like(F.abs()), torch.angle(F)),
                          n=E.shape[1], dim=1)
    return out.to(E.dtype)


def _mag_only(E):
    F = torch.fft.rfft(E.float(), dim=1)
    out = torch.fft.irfft(torch.polar(F.abs(), torch.zeros_like(F.abs())),
                          n=E.shape[1], dim=1)
    return out.to(E.dtype)


# ---------------------------------------------------------------------------
# Part: probe_deep
# ---------------------------------------------------------------------------

def _probe_variants(pe, L):
    v = {"full": pe,
         "phase_only": _span(pe, _phase_only, L),
         "mag_only": _span(pe, _mag_only, L)}
    for i in range(N_BANDS):
        lo, hi = i / N_BANDS, (i + 1) / N_BANDS
        v[f"notch_b{i}"] = _span(pe, lambda x, lo=lo, hi=hi: TS.band_notch_1d(x, lo, hi), L)
    return v


def run_probe_deep(args, pipe, emb, lens):
    for key, prompt in PROBE_PROMPTS[: args.num_prompts]:
        d = os.path.join(OUT, "probe_deep", key)
        pe, ppe = emb[prompt]
        names = list(_probe_variants(pe, lens[prompt]))
        for name, mpe in _probe_variants(pe, lens[prompt]).items():
            _gen(pipe, mpe, ppe, 0, args, os.path.join(d, f"{name}_s0.png"))
        row = [Image.open(os.path.join(d, f"{n}_s0.png")).convert("RGB") for n in names]
        save_grid([row], [key], names, os.path.join(d, "strip.png"), thumb=200)
        print(f"[e30] probe_deep {key} done", flush=True)


# ---------------------------------------------------------------------------
# Part: continuous (the headline strips)
# ---------------------------------------------------------------------------

def run_continuous(args, pipe, emb, lens):
    # (a) low-pass cutoff sweep + (b) high-band gain sweep -- single prompt
    for key, prompt in PROBE_PROMPTS[: args.num_prompts]:
        pe, ppe = emb[prompt]
        L = lens[prompt]
        d = os.path.join(OUT, "continuous", key)
        lp = []
        for c in CUT_SWEEP:
            p = os.path.join(d, f"lowpass_c{c}_s0.png")
            lp.append(_gen(pipe, _span(pe, lambda x, c=c: TS.band_filter_1d(x, 0.0, c), L),
                           ppe, 0, args, p))
        save_grid([lp], ["low-pass"], [f"cut={c}" for c in CUT_SWEEP],
                  os.path.join(d, "lowpass_sweep.png"), thumb=200)
        gn = []
        for g in GAIN_SWEEP:
            p = os.path.join(d, f"gain_g{g}_s0.png")
            gn.append(_gen(pipe, _span(pe, lambda x, g=g: TS.band_gain_1d(x, CUT0, 1.0, g), L),
                           ppe, 0, args, p))
        save_grid([gn], [f"high-band gain (cut={CUT0})"], [f"g={g}" for g in GAIN_SWEEP],
                  os.path.join(d, "gain_sweep.png"), thumb=200)
        print(f"[e30] continuous (single) {key} done", flush=True)

    # (c) two-prompt A<->B morph: band_swap low from A grows with cut (cut->0 ~ B, cut->1 ~ A)
    for key, pA, pB in MERGE_PAIRS[: args.num_prompts]:
        peA, ppeA = emb[pA]
        peB = emb[pB][0]
        L = min(lens[pA], lens[pB])
        d = os.path.join(OUT, "continuous", f"morph_{key}")
        row = []
        for c in CUT_SWEEP:
            p = os.path.join(d, f"morph_c{c}_s0.png")
            row.append(_gen(pipe, _span(peA, lambda x, c=c: TS.band_swap_1d(x, peB[:, :L], c), L),
                            ppeA, 0, args, p))
        save_grid([row], [f"{key}: B<-cut->A"], [f"cut={c}" for c in CUT_SWEEP],
                  os.path.join(d, "morph_sweep.png"), thumb=200)
        print(f"[e30] continuous (morph) {key} done", flush=True)


# ---------------------------------------------------------------------------
# Part: concat (blend vs one big prompt)
# ---------------------------------------------------------------------------

def concat_text(pA, pB):
    return f"{pA} and {pB}"


def run_concat(args, pipe, emb, lens):
    for key, pA, pB in MERGE_PAIRS[: args.num_prompts]:
        peA, ppeA = emb[pA]
        peB, ppeB = emb[pB]
        L = min(lens[pA], lens[pB])
        d = os.path.join(OUT, "concat", key)
        conds = {
            "band_swap": (_span(peA, lambda x: TS.band_swap_1d(x, peB[:, :L], CUT0), L), ppeA),
            "band_blend": (_span(peA, lambda x: TS.band_blend_1d(x, peB[:, :L], CUT0), L), ppeA),
            "lerp": (_span(peA, lambda x: TS.lerp_embeds(x, peB[:, :L], 0.5), L),
                     TS.lerp_embeds(ppeA, ppeB, 0.5)),
            "concat": emb[concat_text(pA, pB)],   # the "one big prompt" baseline
        }
        names = list(conds)
        for name, (mpe, mppe) in conds.items():
            _gen(pipe, mpe, mppe, 0, args, os.path.join(d, f"{name}_s0.png"))
        row = [Image.open(os.path.join(d, f"{n}_s0.png")).convert("RGB") for n in names]
        save_grid([row], [key], names, os.path.join(d, "strip.png"), thumb=220)
        print(f"[e30] concat {key} done", flush=True)


# ---------------------------------------------------------------------------
# Parts: longprompt (DPG-Bench) and compositional (T2I-CompBench)
# ---------------------------------------------------------------------------

def _filter_variants(pe, L):
    """Band-filtering families shared by long/compositional extraction tests."""
    return {
        "full": pe,
        "low": _span(pe, lambda x: TS.band_filter_1d(x, 0.0, CUT0), L),
        "high": _span(pe, lambda x: TS.band_filter_1d(x, CUT0, 1.0, keep_dc=True), L),
        "notch_lo": _span(pe, lambda x: TS.band_notch_1d(x, 0.0, CUT0), L),  # remove low, keep high+DC
    }


def _run_extract(args, pipe, emb, lens, prompts, sub):
    """Shared runner for longprompt / compositional: per prompt, filter-variant grid."""
    for pid, prompt in prompts:
        if prompt not in emb:
            continue
        d = os.path.join(OUT, sub, str(pid))
        pe, ppe = emb[prompt]
        variants = _filter_variants(pe, lens[prompt])
        names = list(variants)
        for name, mpe in variants.items():
            _gen(pipe, mpe, ppe, 0, args, os.path.join(d, f"{name}_s0.png"))
        row = [Image.open(os.path.join(d, f"{n}_s0.png")).convert("RGB") for n in names]
        save_grid([row], [str(pid)], names, os.path.join(d, "strip.png"), thumb=220)
        print(f"[e30] {sub} {pid} done", flush=True)


def _dpg_prompts(args):
    from dpg_bench import load_dpg_prompts
    return [(pid, txt) for pid, txt in load_dpg_prompts(n=args.n_dpg)]


def _comp_prompts(args):
    from compbench import load_compbench_prompts
    return [(pid, txt) for pid, txt, _cat in
            load_compbench_prompts(per_cat=args.per_cat)]


def run_longprompt(args, pipe, emb, lens):
    _run_extract(args, pipe, emb, lens, _dpg_prompts(args), "longprompt")


def run_compositional(args, pipe, emb, lens):
    _run_extract(args, pipe, emb, lens, _comp_prompts(args), "compositional")


# ---------------------------------------------------------------------------
# generation driver
# ---------------------------------------------------------------------------

def _all_prompts(args, parts):
    p = []
    if "probe_deep" in parts or "continuous" in parts:
        p += [pr for _, pr in PROBE_PROMPTS[: args.num_prompts]]
    if "continuous" in parts or "concat" in parts:
        for _, a, b in MERGE_PAIRS[: args.num_prompts]:
            p += [a, b]
    if "concat" in parts:
        for _, a, b in MERGE_PAIRS[: args.num_prompts]:
            p.append(concat_text(a, b))
    if "longprompt" in parts:
        p += [t for _, t in _dpg_prompts(args)]
    if "compositional" in parts:
        p += [t for _, t in _comp_prompts(args)]
    return list(dict.fromkeys(p))


def run_gen(args, parts):
    prompts = _all_prompts(args, parts)
    if not prompts:
        return
    pipe, emb, lens = load_flux_preencoded_lens(prompts)
    runners = {"probe_deep": run_probe_deep, "continuous": run_continuous,
               "concat": run_concat, "longprompt": run_longprompt,
               "compositional": run_compositional}
    for part in parts:
        if part in runners:
            runners[part](args, pipe, emb, lens)


# ---------------------------------------------------------------------------
# Part: analyze
# ---------------------------------------------------------------------------

def _variant_imgs(d):
    """{variant: PIL} for *_s0.png in a dir (skips strip.png)."""
    out = {}
    if not os.path.isdir(d):
        return out
    for fn in sorted(os.listdir(d)):
        if fn.endswith("_s0.png"):
            out[fn[:-len("_s0.png")]] = Image.open(os.path.join(d, fn)).convert("RGB")
    return out


def run_analyze(args):
    parts = [p.strip() for p in args.part.split(",") if p.strip()]
    clip = load_clip(args.clip_model)
    aes = load_aesthetic()
    bvqa = load_bvqa_safe()
    vqa = load_vqa_safe(args)
    report = {"params": vars(args)}

    # probe_deep: per-variant CLIP + image stats
    if "probe_deep" in parts:
        report["probe_deep"] = {}
        for key, prompt in PROBE_PROMPTS[: args.num_prompts]:
            d = os.path.join(OUT, "probe_deep", key)
            ents = {}
            for v, im in _variant_imgs(d).items():
                ents[v] = {"clip": agg(clip_scores(*clip, prompt, [im])),
                           **{k: image_metrics(im)[k] for k in IMG_KEYS}}
            report["probe_deep"][key] = {"prompt": prompt, "variants": ents}

    # concat: CLIP_A / CLIP_B + B-VQA (both objects present?)
    if "concat" in parts:
        report["concat"] = {}
        for key, pA, pB in MERGE_PAIRS[: args.num_prompts]:
            d = os.path.join(OUT, "concat", key)
            ents = {}
            for v, im in _variant_imgs(d).items():
                ents[v] = {"clip_A": agg(clip_scores(*clip, pA, [im])),
                           "clip_B": agg(clip_scores(*clip, pB, [im])),
                           "bvqa": agg(bvqa_safe(bvqa, concat_text(pA, pB), [im]))}
            report["concat"][key] = {"A": pA, "B": pB, "variants": ents}

    # longprompt / compositional: object retention (VQAScore + B-VQA + CLIP) per filter
    for sub, getp in (("longprompt", _dpg_prompts), ("compositional", _comp_prompts)):
        if sub not in parts:
            continue
        report[sub] = {}
        for pid, prompt in getp(args):
            d = os.path.join(OUT, sub, str(pid))
            ents = {}
            for v, im in _variant_imgs(d).items():
                pth = os.path.join(d, f"{v}_s0.png")
                ents[v] = {"clip": agg(clip_scores(*clip, prompt, [im])),
                           "bvqa": agg(bvqa_safe(bvqa, prompt, [im])),
                           "vqa": agg(vqa_safe(vqa, prompt, [pth]))}
            report[sub][str(pid)] = {"prompt": prompt, "variants": ents}

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    _site(report, parts)
    print("[e30] wrote report.json + index.html", flush=True)


# graceful scorer wrappers (heavy/optional deps) ----------------------------

def load_bvqa_safe():
    try:
        from compbench import load_bvqa
        return load_bvqa()
    except Exception as e:
        print(f"[e30] B-VQA unavailable: {e}", flush=True)
        return None


def bvqa_safe(bvqa, prompt, imgs):
    if bvqa is None:
        return [None] * len(imgs)
    from compbench import bvqa_scores
    return bvqa_scores(bvqa, prompt, imgs)


def load_vqa_safe(args):
    if args.no_vqa:
        return None
    try:
        from vqascore import load_vqascore
        return load_vqascore(args.vqa_model)
    except Exception as e:
        print(f"[e30] VQAScore unavailable: {e}", flush=True)
        return None


def vqa_safe(vqa, prompt, paths):
    if vqa is None:
        return [None] * len(paths)
    from vqascore import vqa_scores_paths
    return vqa_scores_paths(vqa, prompt, paths)


# ---------------------------------------------------------------------------
# self-contained HTML explainer (with inline-SVG schematic)
# ---------------------------------------------------------------------------

_SCHEMATIC_SVG = '''
<figure style="margin:1.2em 0">
<svg viewBox="0 0 760 220" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="E30 schematic"
     style="width:100%;max-width:760px;border:1px solid #ddd;border-radius:6px;background:#fff">
  <defs><marker id="ar" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto">
    <path d="M0,0 L7,3 L0,6 Z" fill="#555"/></marker></defs>
  <style>.b{font:12px system-ui;fill:#1b1b1b}.s{font:10px system-ui;fill:#666}
    .t{font:bold 12px system-ui;fill:#111}.box{fill:#f3f6fb;stroke:#9fb6d6}
    line{stroke:#555;stroke-width:1.4;marker-end:url(#ar)}</style>
  <text x="14" y="20" class="t">Continuous text-frequency control: turn ONE knob, watch the image morph</text>
  <rect class="box" x="14" y="36" width="92" height="30"/><text x="22" y="55" class="b">Prompt -> T5</text>
  <line x1="106" y1="51" x2="128" y2="51"/>
  <rect x="132" y="36" width="120" height="30" fill="#dbe9ff" stroke="#5b8bd0"/>
  <rect x="132" y="36" width="46" height="30" fill="#8fc0ff" stroke="#5b8bd0"/>
  <text x="140" y="55" class="b">low</text><text x="216" y="55" class="b">high</text>
  <text x="132" y="80" class="s">FFT over tokens (per channel)</text>
  <line x1="252" y1="51" x2="274" y2="51"/>
  <rect class="box" x="278" y="36" width="150" height="30"/>
  <text x="286" y="55" class="b">knob: cutoff / gain / swap-cut</text>
  <line x1="428" y1="51" x2="450" y2="51"/>
  <rect class="box" x="454" y="36" width="60" height="30"/><text x="462" y="55" class="b">IFFT</text>
  <line x1="514" y1="51" x2="536" y2="51"/>
  <rect class="box" x="540" y="36" width="50" height="30"/><text x="548" y="55" class="b">Flux</text>
  <line x1="590" y1="51" x2="612" y2="51"/>
  <rect x="616" y="34" width="34" height="34" fill="#eee" stroke="#999"/>
  <text x="14" y="118" class="t">Three knobs</text>
  <text x="14" y="138" class="s">low-pass cutoff (cut 0->1): keep DC..cut -> blurry gist -> full prompt</text>
  <text x="14" y="154" class="s">high-band gain (0->2): attenuate / amplify per-token detail</text>
  <text x="14" y="170" class="s">A<->B morph: band_swap(A,B,cut) -- low(A) grows with cut</text>
  <text x="14" y="196" class="s">Extraction: low-pass / high-pass / band-knockout on long (DPG) &amp; compositional (CompBench) prompts, scored by B-VQA + VQAScore.</text>
</svg>
<figcaption style="font:11px system-ui;color:#777">Schematic — E30 continuous text-frequency control &amp; extraction.</figcaption>
</figure>
'''


_CSS = """
body{font:15px/1.65 -apple-system,Segoe UI,Roboto,sans-serif;margin:24px auto;max-width:1000px;color:#1a1a1a;padding:0 16px}
h1{font-size:25px;line-height:1.25} h2{font-size:20px;margin-top:38px;border-bottom:1px solid #ddd;padding-bottom:5px}
h3{font-size:16px;margin:22px 0 4px} h4{font-size:14px;margin:16px 0 2px;color:#333}
.tldr{background:#eef4ff;border:1px solid #c7d9ff;border-radius:6px;padding:14px 16px;margin:14px 0}
.look{background:#f6f8fa;border-left:4px solid #0969da;padding:8px 13px;border-radius:4px;margin:8px 0;font-size:14px}
.read{margin:8px 0 4px} .win{background:#eafaf0;border-left:4px solid #2da44e;padding:8px 13px;border-radius:4px;margin:8px 0}
.cav{background:#fff8f0;border-left:4px solid #d4a017;padding:10px 14px;border-radius:4px;margin:12px 0}
dl{margin:10px 0} dt{font-weight:700;margin-top:11px} dd{margin:2px 0 2px 18px;color:#333}
table{border-collapse:collapse;margin:10px 0;font-variant-numeric:tabular-nums;font-size:14px}
th,td{border:1px solid #d0d7de;padding:4px 9px;text-align:right}
th{background:#f6f8fa;text-align:center} td.v{text-align:left;font-weight:600;white-space:nowrap}
td.pos{background:#dafbe1;font-weight:600} td.neg{background:#ffebe9}
.cap{color:#555;font-size:13px;margin:2px 0 14px}
img{width:100%;border:1px solid #d0d7de;border-radius:4px;margin:6px 0}
code{background:#eff1f3;padding:1px 5px;border-radius:3px;font-size:13px}
"""


def _emb(data_uri, rel):
    """<img> with the strip embedded as base64 (portable), or a note if missing."""
    p = os.path.join(OUT, rel)
    if not os.path.exists(p):
        return f"<p class=cap>(missing <code>{rel}</code>)</p>"
    src = data_uri(p) if data_uri else rel
    return f"<img src='{src}' alt='{rel}'>"


def _table(rows, cols, headers, best="max", best_cols=None):
    """rows = list of (label, {col: float|None}); highlight the best cell per best_col."""
    best_cols = best_cols or cols
    best_val = {}
    for c in best_cols:
        vals = [r[1].get(c) for r in rows if r[1].get(c) is not None]
        if vals:
            best_val[c] = max(vals) if best == "max" else min(vals)
    out = ["<table><tr><th>variant</th>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>"]
    for label, vals in rows:
        cells = []
        for c in cols:
            v = vals.get(c)
            if v is None:
                cells.append("<td>—</td>")
            else:
                hot = c in best_val and abs(v - best_val[c]) < 1e-9
                cells.append(f"<td class=pos>{v:.3f}</td>" if hot else f"<td>{v:.3f}</td>")
        out.append(f"<tr><td class=v>{label}</td>{''.join(cells)}</tr>")
    out.append("</table>")
    return "".join(out)


def _g(sc, k):
    """Pull a mean from a report cell that may be a {'mean':..} dict, a float, or None."""
    v = sc.get(k)
    if isinstance(v, dict):
        return v.get("mean")
    return v


def _site(report, parts=None):
    """Self-contained explainer: TL;DR -> glossary -> method -> visuals-then-numbers.

    Pure templating from `report` (= results/e30/report.json) + the saved strips on disk;
    loads no model and re-scores nothing, so the page rebuilds anywhere (`--part site`)."""
    try:
        from common import data_uri  # base64 embed -> portable single file
    except Exception:
        data_uri = None

    h = ["<!doctype html><meta charset=utf-8><title>E30 — text-frequency control</title>",
         f"<style>{_CSS}</style>",
         "<h1>E30 — Continuous text-frequency control &amp; extraction</h1>"]

    # ---- TL;DR ----
    h.append(
        "<div class=tldr><b>In one paragraph.</b> A text-to-image model first turns the prompt "
        "into a <b>sequence of token embeddings</b> (Flux's T5 encoder: a <code>(1, L, 4096)</code> "
        "array — <code>L</code> word-pieces, each a 4096-dim vector). E24 showed you can take a "
        "<b>1-D Fourier transform along the token axis</b> (independently per channel) and the "
        "resulting low/high frequency <b>bands</b> are meaningful. E30 (a) makes that a "
        "<b>continuous knob</b> (sweep a cutoff / gain / two-prompt crossover and watch the image "
        "morph), and (b) asks <b>what each band carries</b>. <b>Headline:</b> the token spectrum is "
        "genuinely structured — <b>phase carries the content</b>, the <b>low band is a coarse gist</b> "
        "and the <b>mid/high bands carry attribute–object binding</b> — but <b>no single band is "
        "load-bearing</b> and <b>spectral blending still loses to literally writing \"A and B\"</b>. "
        "So it is a clean <i>map</i> of the spectrum, not a new control lever.</div>")
    h.append(_SCHEMATIC_SVG)

    # ---- glossary ----
    h.append("<h2>0 · Background (plain language)</h2><dl>"
             "<dt>Token-axis FFT</dt><dd>The prompt embedding is <code>(1, L, 4096)</code>. We run a "
             "<b>1-D real FFT along the <i>token</i> axis</b> (length <code>L</code>), <b>separately "
             "for each of the 4096 channels</b>. This is <i>not</i> a 2-D image FFT and <i>not</i> the "
             "single pooled vector — it asks how each embedding dimension <i>varies from word to word</i>.</dd>"
             "<dt>Token-frequency, DC, bands</dt><dd>Frequencies are normalized to <code>[0, 1]</code>. "
             "<b>DC</b> (freq 0) = the per-channel average across tokens (the prompt's bag-of-words gist). "
             "<b>Low</b> frequencies = slow drift across the sequence; <b>high</b> = fast token-to-token "
             "change (fine detail). The default split is the cut <code>0.25</code> (low = "
             "<code>[0, 0.25]</code>, high = <code>[0.25, 1]</code>); the per-band probe slices "
             f"<code>[0,1]</code> into {N_BANDS} equal bands <code>b0..b{N_BANDS-1}</code>.</dd>"
             "<dt>The single-prompt variants</dt><dd>"
             "<code>full</code> = the unmodified embedding (baseline). "
             "<code>low</code> = <b>low-pass</b>: keep DC..0.25, zero the rest (coarse gist only). "
             "<code>high</code> = <b>high-pass</b>: keep 0.25..1 (+DC) (fine token detail only). "
             "<code>notch_lo</code> = <b>remove only the low band</b> <code>[0,0.25]</code>, keep "
             "everything above it (tests whether the low band is <i>necessary</i>, vs. the mid/high). "
             "<code>phase_only</code> = keep the spectrum's <b>phase</b>, set every magnitude to 1. "
             "<code>mag_only</code> = keep the <b>magnitude</b>, set every phase to 0. "
             f"<code>notch_b0..b{N_BANDS-1}</code> = knock out exactly one of the {N_BANDS} equal "
             "bands, keep the other five.</dd>"
             "<dt>The two-prompt merge variants</dt><dd>For prompts A and B: "
             "<code>band_swap</code> = low band from A + high band from B (hard cut at 0.25). "
             "<code>band_blend</code> = same, but a soft cosine crossover (no hard edge). "
             "<code>lerp</code> = plain 50/50 average of the two embeddings (the time-domain baseline). "
             "<code>concat</code> = the <b>gold-standard baseline</b>: just encode the literal text "
             "<code>\"A and B\"</code>.</dd>"
             "<dt>The metrics</dt><dd>"
             "<b>CLIP-T</b> (0–1, higher = better): cosine similarity between the image and the prompt "
             "text in CLIP space — how well the image matches the words. "
             "<b>B-VQA</b> (0–1, higher = better): a visual-question-answering check that each named "
             "object appears <i>with its correct attribute</i> (color/shape/texture) — i.e. attribute "
             "binding. <b>sharpness / hf_frac / colorful</b>: image statistics (edge energy, "
             "high-spatial-frequency fraction, colorfulness) used as sanity signals.</dd>"
             "</dl>")

    # ---- method ----
    h.append("<h2>1 · Method</h2><dl>"
             "<dt>probe_deep</dt><dd>One prompt at a time: regenerate under "
             "<code>full / phase_only / mag_only / notch_b0..b5</code>. <i>Which part of the spectrum "
             "carries the content?</i></dd>"
             "<dt>continuous</dt><dd>The headline visuals: regenerate while sweeping <b>one</b> knob — a "
             "low-pass cutoff (0.1→0.9), a high-band gain (0→2×), and a two-prompt A↔B crossover. "
             "<i>Does the image move smoothly?</i></dd>"
             "<dt>concat</dt><dd>Merge two prompts via <code>band_swap / band_blend / lerp</code> and "
             "compare to the literal <code>concat</code> (\"A and B\"). <i>Does any spectral merge keep "
             "<b>both</b> objects (B-VQA) better than just writing them?</i></dd>"
             "<dt>longprompt / compositional</dt><dd>Apply <code>low / high / notch_lo</code> to long "
             "(DPG-Bench) and compositional (T2I-CompBench) prompts. <i>Which band carries the gist, and "
             "which carries the attribute–object binding?</i></dd>"
             "</dl>")

    h.append("<h2>2 · Results</h2>")

    # --- probe_deep ---
    if report.get("probe_deep"):
        h.append("<h3>probe_deep — what each band carries</h3>")
        h.append("<div class=look><b>What to look for.</b> Compare <code>phase_only</code> and "
                 "<code>mag_only</code> to <code>full</code>: if <code>phase_only</code> still looks "
                 "on-prompt but <code>mag_only</code> is junk, the <b>phase</b> carries the meaning. "
                 "Across <code>notch_b0..b5</code>, if every image stays on-prompt, <b>no single band "
                 "is essential</b>.</div>")
        h.append("<div class=read><b>Reading.</b> <code>phase_only</code> ≈ <code>full</code> in CLIP "
                 "while <code>mag_only</code> collapses for the object prompts (cat/car) — the "
                 "<b>token-axis phase carries the content; magnitude is near-discardable</b>. And no "
                 "single-band knockout moves CLIP by more than ~0.02, so the content is "
                 "<b>redundantly spread across bands</b>.</div>")
        for key, e in report["probe_deep"].items():
            h.append(f"<h4>{key}: <code>{e['prompt']}</code></h4>")
            h.append(_emb(data_uri, f"probe_deep/{key}/strip.png"))
            h.append("<p class=cap>columns: full · phase_only · mag_only · notch_b0…b5</p>")
            rows = [(v, {"CLIP": _g(sc, "clip"), "sharpness": sc.get("sharpness"),
                         "hf_frac": sc.get("hf_frac"), "colorful": sc.get("colorfulness")})
                    for v, sc in e["variants"].items()]
            h.append(_table(rows, ["CLIP", "sharpness", "hf_frac", "colorful"],
                            ["CLIP-T ↑", "sharpness", "hf_frac", "colorful"], best_cols=["CLIP"]))

    # --- continuous ---
    if "continuous" in (parts or []) or any(
            os.path.exists(os.path.join(OUT, "continuous", k, "lowpass_sweep.png"))
            for k, _ in PROBE_PROMPTS):
        h.append("<h3>continuous — turn one knob, watch it morph</h3>")
        h.append("<div class=look><b>What to look for.</b> Left→right each strip is one knob moving. "
                 "Low-pass cutoff: at a low cut only the coarse gist survives, growing into the full "
                 "prompt. High-band gain: 0× strips fine detail, 2× over-emphasizes it. A↔B morph: the "
                 "image should slide from prompt B toward prompt A as the cut rises.</div>")
        for key, _p in PROBE_PROMPTS:
            for fn, cap in (("lowpass_sweep.png", "low-pass cutoff sweep (cut 0.1 → 0.9)"),
                            ("gain_sweep.png", f"high-band gain sweep (g 0 → 2×, cut={CUT0})")):
                rel = f"continuous/{key}/{fn}"
                if os.path.exists(os.path.join(OUT, rel)):
                    h.append(f"<h4>{key} — {cap}</h4>{_emb(data_uri, rel)}")
        for key, _a, _b in MERGE_PAIRS:
            rel = f"continuous/morph_{key}/morph_sweep.png"
            if os.path.exists(os.path.join(OUT, rel)):
                h.append(f"<h4>{key} — A↔B morph (band_swap, cut 0.1 → 0.9; B → A)</h4>"
                         f"{_emb(data_uri, rel)}")

    # --- concat ---
    if report.get("concat"):
        h.append("<h3>concat — spectral merge vs. writing \"A and B\"</h3>")
        h.append("<div class=look><b>What to look for.</b> Only the <code>concat</code> panel should "
                 "show <b>both</b> objects; the spectral merges tend to keep one and drop the other. "
                 "The number that matters is <b>B-VQA</b> (both present?).</div>")
        h.append("<div class=read><b>Reading.</b> <code>concat</code> wins decisively on B-VQA; every "
                 "spectral merge collapses toward whichever prompt owns the low band. Spectral blending "
                 "offers nothing over just writing the two objects.</div>")
        for key, e in report["concat"].items():
            h.append(f"<h4>{key}: <code>A = {e.get('A')}</code> · <code>B = {e.get('B')}</code></h4>")
            h.append(_emb(data_uri, f"concat/{key}/strip.png"))
            h.append("<p class=cap>columns: band_swap · band_blend · lerp · concat</p>")
            rows = [(v, {"clip_A": _g(sc, "clip_A"), "clip_B": _g(sc, "clip_B"),
                         "bvqa": _g(sc, "bvqa")}) for v, sc in e["variants"].items()]
            h.append(_table(rows, ["clip_A", "clip_B", "bvqa"],
                            ["CLIP→A ↑", "CLIP→B ↑", "B-VQA ↑"], best_cols=["bvqa"]))

    # --- longprompt / compositional ---
    for sub, blurb in (
        ("longprompt", "Long (DPG-Bench) prompts. B-VQA is sparse on ~80-word prompts, so read CLIP: "
                       "removing the low band (<code>notch_lo</code>) tends to hurt most — the low band "
                       "carries the gist."),
        ("compositional", "Compositional (T2I-CompBench) prompts. <b>low-pass destroys attribute "
                          "binding</b> (B-VQA → ~0) while <code>notch_lo</code> (drop only the low band) "
                          "often keeps it — so the <b>binding lives in the mid/high bands</b>, the gist "
                          "in the low band.")):
        if not report.get(sub):
            continue
        h.append(f"<h3>{sub} — which band carries what</h3>")
        h.append("<div class=look><b>What to look for.</b> columns are "
                 "<code>full · low · high · notch_lo</code>. Watch whether the named objects and their "
                 "attributes survive each filter.</div>")
        h.append(f"<div class=read><b>Reading.</b> {blurb}</div>")
        for key, e in report[sub].items():
            h.append(f"<h4>{key}: <code>{e.get('prompt')}</code></h4>")
            h.append(_emb(data_uri, f"{sub}/{key}/strip.png"))
            rows = [(v, {"clip": _g(sc, "clip"), "bvqa": _g(sc, "bvqa"), "vqa": _g(sc, "vqa")})
                    for v, sc in e["variants"].items()]
            h.append(_table(rows, ["clip", "bvqa", "vqa"],
                            ["CLIP-T ↑", "B-VQA ↑", "VQAScore ↑"], best_cols=["clip", "bvqa"]))

    h.append("<h2>3 · Caveats &amp; next</h2><div class=cav>"
             "Single seed per cell (read directions, not third decimals); VQAScore was deferred "
             "(<code>--no_vqa</code>) so those columns are blank — B-VQA already carries the binding "
             "story. The structure is descriptive: it explains the spectrum but does not beat writing "
             "the prompt, so the thread is mapped rather than a live control lever.</div>")

    with open(os.path.join(OUT, "index.html"), "w") as f:
        f.write("\n".join(h))


# ---------------------------------------------------------------------------

def main(args):
    global OUT
    if args.out_tag:
        OUT = os.path.join(RESULTS, f"e30_{args.out_tag}")
    os.makedirs(OUT, exist_ok=True)
    parts = [p.strip() for p in args.part.split(",") if p.strip()]
    gen_parts = [p for p in parts if p in
                 ("probe_deep", "continuous", "concat", "longprompt", "compositional")]
    if gen_parts:
        run_gen(args, gen_parts)
    if "analyze" in parts:
        run_analyze(args)
    if "site" in parts:   # model-free: rebuild index.html from report.json + cached strips
        rp = os.path.join(OUT, "report.json")
        if not os.path.exists(rp):
            raise SystemExit(f"[e30] --part site needs {rp} (run analyze first / fetch it)")
        with open(rp) as f:
            report = json.load(f)
        _site(report, [k for k in report if k != "params"])
        print(f"[e30] rebuilt {os.path.join(OUT, 'index.html')} (no model loaded)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="probe_deep,continuous,concat,longprompt,compositional,analyze")
    ap.add_argument("--num_prompts", type=int, default=3, help="cap on probe prompts / merge pairs")
    ap.add_argument("--n_dpg", type=int, default=6, help="DPG-Bench long prompts")
    ap.add_argument("--per_cat", type=int, default=4, help="CompBench prompts per category")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=3.5)
    ap.add_argument("--clip_model", default="openai/clip-vit-large-patch14")
    ap.add_argument("--vqa_model", default="clip-flant5-xxl")
    ap.add_argument("--no_vqa", action="store_true", help="skip the heavy VQAScore model")
    ap.add_argument("--out_tag", default="")
    main(ap.parse_args())
