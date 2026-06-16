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


def _strip_imgs(report, parts):
    """Collect saved strip.png / *_sweep.png paths (relative to OUT) for the page."""
    found = []
    for root, _dirs, files in os.walk(OUT):
        for fn in sorted(files):
            if fn == "strip.png" or fn.endswith("_sweep.png"):
                found.append(os.path.relpath(os.path.join(root, fn), OUT))
    return sorted(found)


def _site(report, parts):
    try:
        from e27_site import data_uri  # self-contained base64 embedding
    except Exception:
        data_uri = None

    def emb_img(rel):
        p = os.path.join(OUT, rel)
        if data_uri and os.path.exists(p):
            return f"<img src='{data_uri(p)}' style='max-width:100%'>"
        return f"<img src='{rel}' style='max-width:100%'>"

    h = ["<!doctype html><meta charset=utf-8><title>E30 text-frequency control</title>",
         "<style>body{font:14px/1.5 system-ui;max-width:1100px;margin:2rem auto;"
         "padding:0 1rem;color:#222}code{background:#f0f0f0;padding:1px 4px;border-radius:3px}"
         "h1,h2{line-height:1.2}img{border:1px solid #ddd;margin:.4em 0}"
         "figure{margin:1em 0}</style>",
         "<h1>E30 — Continuous text-frequency control &amp; extraction</h1>",
         "<p><b>Follow-up to E24.</b> We FFT a prompt's T5 <b>sequence</b> embedding "
         "<code>(1,L,4096)</code> along the <b>token axis</b> (a 1-D DFT per embedding "
         "channel — not 2-D, and not the pooled vector), then turn a single knob — "
         "low-pass cutoff, high-band gain, or two-prompt swap-cut — and regenerate. The "
         "goal is continuous control plus understanding what frequency filtering does to "
         "long and compositional prompts.</p>",
         _SCHEMATIC_SVG,
         "<h2>Continuous sweeps</h2>"]
    for rel in _strip_imgs(report, parts):
        h.append(f"<p><code>{rel}</code><br>{emb_img(rel)}</p>")

    # compact metric tables
    if report.get("probe_deep"):
        h.append("<h2>Probe — what each band controls (CLIP + image stats)</h2>")
        for key, e in report["probe_deep"].items():
            h.append(f"<h3>{key}: <code>{e['prompt']}</code></h3><table border=1 "
                     "cellpadding=4 style='border-collapse:collapse'><tr><th>variant</th>"
                     "<th>CLIP</th><th>sharpness</th><th>hf_frac</th><th>colorful</th></tr>")
            for v, sc in e["variants"].items():
                cl = f"{sc['clip']['mean']:.3f}" if sc.get('clip') else '—'
                h.append(f"<tr><td>{v}</td><td>{cl}</td><td>{sc['sharpness']:.1f}</td>"
                         f"<td>{sc['hf_frac']:.3f}</td><td>{sc['colorfulness']:.3f}</td></tr>")
            h.append("</table>")
    for sub, cols in (("concat", ["clip_A", "clip_B", "bvqa"]),
                      ("longprompt", ["clip", "bvqa", "vqa"]),
                      ("compositional", ["clip", "bvqa", "vqa"])):
        if not report.get(sub):
            continue
        h.append(f"<h2>{sub}</h2>")
        for key, e in report[sub].items():
            cap = e.get("prompt") or f"A={e.get('A')} · B={e.get('B')}"
            h.append(f"<h4>{key}: <code>{cap}</code></h4><table border=1 cellpadding=4 "
                     "style='border-collapse:collapse'><tr><th>variant</th>"
                     + "".join(f"<th>{c}</th>" for c in cols) + "</tr>")
            for v, sc in e["variants"].items():
                cells = "".join(f"<td>{sc[c]['mean']:.3f}</td>" if sc.get(c) else "<td>—</td>"
                                for c in cols)
                h.append(f"<tr><td>{v}</td>{cells}</tr>")
            h.append("</table>")

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
