"""E43: FlowAlign on FLUX + two spectral twists, scored against the plain-FlowAlign baseline.

FlowAlign (arXiv:2505.23145) = inversion-free FlowEdit + a source-consistency TERMINAL-POINT
term. This experiment runs, per scene, the same FlowAlign integration under several conditions
that differ only in the spectral knobs, then scores each edit vs. the source with PIE-Bench-style
metrics (DINO structure distance, CLIP-directional editability, LPIPS/DSSIM). Goal: find a knob
setting that matches/beats plain FlowAlign on structure preservation without losing editability.

The two twists (both = plain FlowAlign at their defaults):
  - SBN on the CFG reference: clamp the CFG velocity vp's low band toward v(pt,c_src).
  - annealed terminal point: low-pass the consistency vector (coarse early -> fine late).

The editing math is `invert_core.flowalign` -- the SAME code the demo's FlowAlign tab calls, so
this harness and the interactive tab can't drift. Parts (--part): gen (edits) ; analyze (metrics).

Identity gate: the `recon` condition sets C_tar=C_src, so FlowAlign must reproduce the source
(structure-dist ~ 0, LPIPS ~ 0) -- validates the FLUX plumbing independent of the schedule.

Run:  uv run experiments/e43_flowalign.py --part gen,analyze --num 3 --steps 28
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
#     "numpy",
#     "pillow",
#     "matplotlib",
#     "lpips",
#     "scikit-image",
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
import argparse
import json
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, save_grid
from e7_flux_phase import flux_vae_decode, SIZE
from e10_cfg_spectral import gen_emb
from e24_text_spectral import load_flux_preencoded_lens
import invert_core as IC

OUT = os.path.join(RESULTS, "e43")

# (key, source prompt, edit/target prompt)
SOURCES = [
    ("cat_dog", "a photograph of a cat sitting on a sofa",
     "a photograph of a dog sitting on a sofa"),
    ("house_storm", "a house by a lake on a sunny day",
     "a house by a lake during a dramatic thunderstorm"),
    ("street_snow", "a city street with shops in summer",
     "a city street with shops covered in deep snow"),
]


def conditions(cut, anneal_start):
    """name -> kwargs for IC.flowalign (recon handled separately). All run at the same w/zeta."""
    return {
        "flowalign": {},                                                       # plain (baseline)
        "sbn_bp": {"sbn_mode": "band power", "sbn_cut": cut, "sbn_strength": 1.0},
        "sbn_phase": {"sbn_mode": "phase", "sbn_cut": cut},
        "term_anneal": {"term_start_cut": anneal_start, "term_end_cut": 1.0},
        "sbn_bp+term": {"sbn_mode": "band power", "sbn_cut": cut,
                        "term_start_cut": anneal_start, "term_end_cut": 1.0},
    }


# ---------------------------------------------------------------------------
# Part: gen
# ---------------------------------------------------------------------------

def run_gen(args):
    prompts = []
    for _, s, t in SOURCES[: args.num]:
        prompts += [s, t]
    pipe, smap, _ = load_flux_preencoded_lens(list(dict.fromkeys(prompts)))
    sig = IC.flux_sigmas(pipe, args.steps)
    g_edit = IC.gids(pipe, args.gbase)            # base embed for the editing velocities
    conds = conditions(args.cut, args.anneal_start)

    for key, src, tgt in SOURCES[: args.num]:
        d = os.path.join(OUT, key)
        os.makedirs(d, exist_ok=True)
        C_src = (smap[src][0].cuda(), smap[src][1].cuda())
        C_tar = (smap[tgt][0].cuda(), smap[tgt][1].cuda())

        # source latent x0: generated from the source caption, or VAE-encoded from a real image.
        srcp = os.path.join(d, "source.png")
        real = os.path.join(args.real_dir, f"{key}.png") if args.real_dir else ""
        if real and os.path.exists(real):
            x0 = IC.pack(pipe, IC.vae_encode(pipe.vae, Image.open(real)))
            if not os.path.exists(srcp):
                Image.open(real).convert("RGB").resize((SIZE, SIZE)).save(srcp)
        else:
            img, lat = gen_emb(pipe, (smap[src][0].cpu(), smap[src][1].cpu()), None,
                               args.seed, 1.0, args.gen_guidance, args.steps)
            img.save(srcp)
            x0 = IC.pack(pipe, lat)

        # identity gate first (C_tar = C_src must reproduce the source)
        outp = os.path.join(d, "recon.png")
        if not os.path.exists(outp):
            xe = IC.flowalign(pipe, x0, C_src, C_src, sig, args.seed, g_edit, args.w, args.zeta)
            flux_vae_decode(pipe.vae, IC.unpack(pipe, xe)).save(outp)
            print(f"[e43] {key}/recon done", flush=True)

        for cond, kw in conds.items():
            outp = os.path.join(d, f"{cond}.png")
            if os.path.exists(outp):
                continue
            xe = IC.flowalign(pipe, x0, C_src, C_tar, sig, args.seed, g_edit,
                              args.w, args.zeta, **kw)
            flux_vae_decode(pipe.vae, IC.unpack(pipe, xe)).save(outp)
            print(f"[e43] {key}/{cond} done", flush=True)

        names = ["source", "recon"] + list(conds)
        row = [Image.open(os.path.join(d, f"{n}.png")).convert("RGB") for n in names
               if os.path.exists(os.path.join(d, f"{n}.png"))]
        save_grid([row], [key], names, os.path.join(d, "strip.png"), thumb=240)
        print(f"[e43] {key} grid done", flush=True)


# ---------------------------------------------------------------------------
# Part: analyze
# ---------------------------------------------------------------------------

def run_analyze(args):
    from struct_metrics import load_metrics, structure_distance, clip_directional, image_metrics
    m = load_metrics()
    report = {"params": vars(args), "sources": {}}
    for key, src, tgt in SOURCES[: args.num]:
        d = os.path.join(OUT, key)
        srcp = os.path.join(d, "source.png")
        if not os.path.exists(srcp):
            continue
        src_img = Image.open(srcp).convert("RGB")
        ents = {}
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".png") or fn in ("source.png", "strip.png"):
                continue
            cond = fn[:-4]
            im = Image.open(os.path.join(d, fn)).convert("RGB")
            img_m = image_metrics(im, src_img, m["lpips"], m["ssim"])
            ents[cond] = {
                "struct_dist": structure_distance(m["dino"], im, src_img),
                "clip_dir": clip_directional(m["clip"], src_img, im, src, tgt),
                "lpips": img_m.get("lpips"),
                "dssim": img_m.get("dssim"),
            }
        report["sources"][key] = {"src": src, "tgt": tgt, "conds": ents}
        r = ents.get("recon", {}).get("struct_dist")
        print(f"[e43] {key}: recon struct-dist={r}", flush=True)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    _site(report)
    print("[e43] wrote report.json + index.html", flush=True)


def _site(report):
    """Compact per-scene strip + metric table; FlowAlign-baseline is the reference row."""
    def fmt(v):
        return "—" if v is None else f"{v:.4f}"
    h = ["<!doctype html><meta charset=utf-8><title>E43 — FlowAlign + spectral twists</title>",
         "<style>body{font:14px system-ui;margin:24px auto;max-width:1100px}"
         "table{border-collapse:collapse;margin:8px 0;font-variant-numeric:tabular-nums}"
         "th,td{border:1px solid #ccc;padding:4px 9px;text-align:right}"
         "td.v{text-align:left;font-weight:600}th{background:#f4f6fa}"
         "img{width:100%;border:1px solid #ccc;margin:6px 0}.best{background:#dafbe1}</style>",
         "<h1>E43 — FlowAlign on FLUX + spectral twists</h1>",
         "<p>Per scene: <code>source · recon · flowalign(baseline) · sbn_bp · sbn_phase · "
         "term_anneal · sbn_bp+term</code>. Goal: a twist with <b>struct-dist ↓ / LPIPS ↓</b> "
         "vs the <code>flowalign</code> row while keeping <b>clip-dir ↑</b>.</p>"]
    for key, e in report["sources"].items():
        h.append(f"<h3>{key}: <code>{e['src']}</code> → <code>{e['tgt']}</code></h3>")
        h.append(f"<img src='{key}/strip.png'>")
        rows = e["conds"]
        base = rows.get("flowalign", {})
        h.append("<table><tr><th>condition</th><th>struct-dist ↓</th><th>clip-dir ↑</th>"
                 "<th>LPIPS ↓</th><th>DSSIM ↓</th></tr>")
        for c, sc in rows.items():
            sd = sc.get("struct_dist")
            better = (c not in ("flowalign", "recon") and sd is not None
                      and base.get("struct_dist") is not None and sd < base["struct_dist"])
            cls = " class=best" if better else ""
            h.append(f"<tr><td class=v>{c}</td><td{cls}>{fmt(sd)}</td>"
                     f"<td>{fmt(sc.get('clip_dir'))}</td><td>{fmt(sc.get('lpips'))}</td>"
                     f"<td>{fmt(sc.get('dssim'))}</td></tr>")
        h.append("</table>")
    with open(os.path.join(OUT, "index.html"), "w") as fh:
        fh.write("\n".join(h))


# ---------------------------------------------------------------------------

def main(args):
    global OUT
    if args.out_tag:
        OUT = os.path.join(RESULTS, f"e43_{args.out_tag}")
    os.makedirs(OUT, exist_ok=True)
    parts = [p.strip() for p in args.part.split(",") if p.strip()]
    if "gen" in parts:
        run_gen(args)
    if "analyze" in parts:
        run_analyze(args)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="gen,analyze")
    ap.add_argument("--num", type=int, default=3, help="number of source scenes")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--w", type=float, default=5.0, help="CFG (source prompt as negative)")
    ap.add_argument("--zeta", type=float, default=0.01, help="terminal source-consistency weight")
    ap.add_argument("--cut", type=float, default=0.2, help="SBN low-band cutoff")
    ap.add_argument("--anneal_start", type=float, default=0.15,
                    help="terminal-point start cutoff (anneals to 1.0)")
    ap.add_argument("--gbase", type=float, default=1.0, help="flux distilled-guidance base embed")
    ap.add_argument("--gen_guidance", type=float, default=3.5, help="guidance for source generation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--real_dir", default="", help="dir of <key>.png real images (else generate)")
    ap.add_argument("--out_tag", default="")
    main(ap.parse_args())
