"""E31: real-image editing via FlowEdit, with a *frequency-surgery* target conditioning.

FlowEdit (Kulikov et al. 2024) edits images with a flow model WITHOUT inversion: it
integrates the difference between the target- and source-conditioned velocity fields and
adds the resulting delta to the source latent. E31's twist: the target conditioning is a
**token-frequency surgery** of the source conditioning (E24/E28 ops) -- e.g. inject an
edit prompt's high band into the source's low band -- instead of a plain different prompt.

Pipeline (per source):
  x0  = clean source latent (generated from the source prompt, or VAE-encoded from a real
        image via --real_dir).
  C_src = source T5 sequence+pooled embeds; C_tar = freq_surgery(C_src, C_style).
  FlowEdit:  delta = 0; for sigma high->low:  x_src=(1-s)x0+s*eps;  x_tar=x_src+delta;
             delta += (s_next - s)*(v(x_tar,s,C_tar) - v(x_src,s,C_src)).
  edited = x0 + delta  ->  unpack -> VAE decode.

Identity property (the safety net): if C_tar == C_src the velocity difference is exactly
zero, so `recon` reproduces the source EXACTLY by construction -- the reconstruction gate
therefore validates the VAE/packing plumbing, independent of the sigma schedule.

The only new model-level piece is `flux_velocity`: a manual Flux transformer forward
(packed latents + img/txt ids + guidance embed), mirroring FluxPipeline's denoising call
and the SD3.5 `velocity()` in e21_spectral_edit.py.

Parts (--part): gen (sources + edits) ; analyze (metrics + index.html w/ schematic).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, save_grid
from e7_flux_phase import flux_vae_decode, SIZE
from e10_cfg_spectral import gen_emb
from e24_text_spectral import load_flux_preencoded_lens
from e9_clipt import agg, load_clip, clip_scores
from fidelity_metrics import load_aesthetic, aesthetic_scores
import text_spectral_ops as TS

OUT = os.path.join(RESULTS, "e31")
H = W = 128                     # unpacked latent spatial dims (1024px / 8)
SEQ_LEN = 512                  # T5 padded sequence length (txt_ids length)
PACK_TOKENS = (H // 2) * (W // 2)   # 4096

# (key, source prompt, edit/style prompt)
SOURCES = [
    ("house_storm", "a house by a lake on a sunny day",
     "a dramatic thunderstorm with dark clouds and lightning"),
    ("cat_paint", "a photograph of a cat on a sofa",
     "an oil painting with thick visible brushstrokes, impasto"),
    ("street_snow", "a city street with shops in summer",
     "a city street covered in deep snow during a blizzard"),
]
CUTS = [0.25, 0.4]


# ---------------------------------------------------------------------------
# Flux velocity accessor + sigma schedule (mirror FluxPipeline denoising call)
# ---------------------------------------------------------------------------

def flux_sigmas(pipe, steps, seq_len=PACK_TOKENS):
    """Flux's resolution-shifted sigma grid (steps+1, decreasing, [-1]=0)."""
    from diffusers.pipelines.flux.pipeline_flux import (retrieve_timesteps,
                                                        calculate_shift)
    cfg = pipe.scheduler.config
    sigmas = np.linspace(1.0, 1.0 / steps, steps)
    try:
        mu = calculate_shift(seq_len, cfg.get("base_image_seq_len", 256),
                             cfg.get("max_image_seq_len", 4096),
                             cfg.get("base_shift", 0.5), cfg.get("max_shift", 1.15))
        retrieve_timesteps(pipe.scheduler, steps, "cuda", sigmas=sigmas, mu=mu)
    except Exception as e:
        print(f"[e31] shift schedule failed ({e}); plain set_timesteps", flush=True)
        pipe.scheduler.set_timesteps(steps, device="cuda")
    return pipe.scheduler.sigmas.float()


def _gids(pipe, guidance):
    """Per-run constants: guidance embed, txt_ids (zeros), img_ids (positional)."""
    img_ids = pipe._prepare_latent_image_ids(1, H // 2, W // 2, "cuda", pipe.dtype)
    txt_ids = torch.zeros(SEQ_LEN, 3, device="cuda", dtype=pipe.dtype)
    g = torch.full([1], float(guidance), device="cuda", dtype=torch.float32)
    return g, txt_ids, img_ids


@torch.no_grad()
def flux_velocity(pipe, packed_x, sigma, pe, ppe, gids):
    """Flow-matching velocity v(x, sigma | conditioning) in PACKED latent space.
    timestep passed = sigma (FluxPipeline feeds timestep/1000 with timestep=sigma*1000)."""
    guidance, txt_ids, img_ids = gids
    t = torch.full((packed_x.shape[0],), float(sigma), device="cuda", dtype=pipe.dtype)
    v = pipe.transformer(hidden_states=packed_x.to(pipe.dtype), timestep=t,
                         guidance=guidance, pooled_projections=ppe.to(pipe.dtype),
                         encoder_hidden_states=pe.to(pipe.dtype),
                         txt_ids=txt_ids, img_ids=img_ids, return_dict=False)[0]
    return v.float()


@torch.no_grad()
def flowedit(pipe, x0_packed, C_src, C_tar, sig, skip, seed, gids):
    """Inversion-free FlowEdit: return the edited PACKED latent x0 + delta."""
    steps = len(sig) - 1
    gen = torch.Generator("cuda").manual_seed(seed)
    eps = torch.randn(x0_packed.shape, generator=gen, device="cuda").float()
    delta = torch.zeros_like(x0_packed)
    for i in range(int(skip * steps), steps):
        s_hi, s_lo = float(sig[i]), float(sig[i + 1])
        x_src = (1 - s_hi) * x0_packed + s_hi * eps
        x_tar = x_src + delta
        v_src = flux_velocity(pipe, x_src, s_hi, C_src[0], C_src[1], gids)
        v_tar = flux_velocity(pipe, x_tar, s_hi, C_tar[0], C_tar[1], gids)
        delta = delta + (s_lo - s_hi) * (v_tar - v_src)
    return x0_packed + delta


# ---------------------------------------------------------------------------
# latent helpers
# ---------------------------------------------------------------------------

def pack(pipe, lat):
    return pipe._pack_latents(lat.cuda().float(), 1, 16, H, W)


def unpack(pipe, packed):
    return pipe._unpack_latents(packed, SIZE, SIZE, pipe.vae_scale_factor)


def vae_encode(vae, pil):
    """Real image -> generation-space latent (1,16,128,128), inverting flux_vae_decode."""
    img = pil.convert("RGB").resize((SIZE, SIZE))
    x = torch.from_numpy(np.asarray(img).copy()).float() / 255.0
    x = (x.permute(2, 0, 1)[None] * 2 - 1).to(vae.dtype).cuda()
    with torch.no_grad():
        z = vae.encode(x).latent_dist.mean
    return ((z - vae.config.shift_factor) * vae.config.scaling_factor).float().cpu()


# ---------------------------------------------------------------------------
# conditions: target = frequency-surgery of the source conditioning
# ---------------------------------------------------------------------------

def edit_conditions(src, style, lens, smap, args):
    """dict cond -> (C_tar=(pe,ppe)). src/style are prompt strings; smap=embeds."""
    src_pe, src_ppe = smap[src]
    sty_pe, sty_ppe = smap[style]
    L = min(lens[src], lens[style])
    conds = {"recon": (src_pe, src_ppe),                 # identity -> gate
             "full": (sty_pe, sty_ppe)}                  # plain FlowEdit (prompt swap)
    for c in CUTS:                                       # low(src) + high(style)
        conds[f"swap_c{c}"] = (
            TS.apply_on_span(lambda x, c=c: TS.band_swap_1d(x, sty_pe[:, :L], c), src_pe, L),
            src_ppe)
    return conds


# ---------------------------------------------------------------------------
# Part: gen
# ---------------------------------------------------------------------------

def run_gen(args):
    prompts = []
    for _, s, st in SOURCES[: args.num]:
        prompts += [s, st]
    pipe, smap, lens = load_flux_preencoded_lens(list(dict.fromkeys(prompts)))
    sig = flux_sigmas(pipe, args.steps)
    gids = _gids(pipe, args.guidance)

    for key, src, style in SOURCES[: args.num]:
        d = os.path.join(OUT, key)
        os.makedirs(d, exist_ok=True)
        # source latent x0: generate from the source prompt (exact caption), unless a
        # real image is supplied.
        srcp = os.path.join(d, "source.png")
        real = os.path.join(args.real_dir, f"{key}.png") if args.real_dir else ""
        if real and os.path.exists(real):
            x0 = pack(pipe, vae_encode(pipe.vae, Image.open(real)))
            if not os.path.exists(srcp):
                Image.open(real).convert("RGB").resize((SIZE, SIZE)).save(srcp)
        else:
            img, lat = gen_emb(pipe, (smap[src][0].cpu(), smap[src][1].cpu()), None,
                               args.seed, 1.0, args.guidance, args.steps)
            img.save(srcp)
            x0 = pack(pipe, lat)
        # edits
        for cond, C_tar in edit_conditions(src, style, lens, smap, args).items():
            outp = os.path.join(d, f"{cond}.png")
            if os.path.exists(outp):
                continue
            xe = flowedit(pipe, x0, (smap[src][0].cuda(), smap[src][1].cuda()),
                          (C_tar[0].cuda(), C_tar[1].cuda()), sig, args.skip, args.seed, gids)
            flux_vae_decode(pipe.vae, unpack(pipe, xe)).save(outp)
            print(f"[e31] {key}/{cond} done", flush=True)
        # before/after strip
        names = ["source"] + list(edit_conditions(src, style, lens, smap, args))
        row = [Image.open(os.path.join(d, f"{n}.png")).convert("RGB") for n in names
               if os.path.exists(os.path.join(d, f"{n}.png"))]
        save_grid([row], [key], names, os.path.join(d, "strip.png"), thumb=240)
        print(f"[e31] {key} grid done", flush=True)


# ---------------------------------------------------------------------------
# Part: analyze
# ---------------------------------------------------------------------------

def run_analyze(args):
    clip = load_clip(args.clip_model)
    aes = load_aesthetic()
    report = {"params": vars(args), "sources": {}}
    for key, src, style in SOURCES[: args.num]:
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
            # pixel L2 distance to source (content preservation; ~0 for recon)
            a = np.asarray(im).astype(np.float32) / 255
            b = np.asarray(src_img).astype(np.float32) / 255
            ents[cond] = {
                "clip_style": agg(clip_scores(*clip, style, [im])),     # edit adherence
                "clip_src": agg(clip_scores(*clip, src, [im])),         # content kept
                "px_dist_to_src": float(np.sqrt(((a - b) ** 2).mean())),
                "aesthetic": agg(aesthetic_scores(aes, *clip, [im])),
            }
        report["sources"][key] = {"src": src, "style": style, "conds": ents}
        r = ents.get("recon", {}).get("px_dist_to_src")
        print(f"[e31] {key}: recon px-dist={r}", flush=True)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    _site(report)
    print("[e31] wrote report.json + index.html", flush=True)


# ---------------------------------------------------------------------------
# HTML explainer (with inline-SVG schematic)
# ---------------------------------------------------------------------------

_SCHEMATIC_SVG = '''
<figure style="margin:1.2em 0">
<svg viewBox="0 0 760 230" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="E31 schematic"
     style="width:100%;max-width:760px;border:1px solid #ddd;border-radius:6px;background:#fff">
  <defs><marker id="ar" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto">
    <path d="M0,0 L7,3 L0,6 Z" fill="#555"/></marker></defs>
  <style>.b{font:12px system-ui;fill:#1b1b1b}.s{font:10px system-ui;fill:#666}
    .t{font:bold 12px system-ui;fill:#111}.box{fill:#f3f6fb;stroke:#9fb6d6}
    line{stroke:#555;stroke-width:1.4;marker-end:url(#ar)}</style>
  <text x="14" y="20" class="t">FlowEdit (inversion-free) with a frequency-surgery target conditioning</text>
  <rect x="14" y="36" width="64" height="34" fill="#eee" stroke="#999"/><text x="20" y="57" class="b">source</text>
  <line x1="78" y1="53" x2="100" y2="53"/>
  <rect class="box" x="104" y="36" width="74" height="34"/><text x="112" y="57" class="b">x0 latent</text>
  <line x1="178" y1="53" x2="200" y2="53"/>
  <rect class="box" x="204" y="30" width="270" height="48"/>
  <text x="212" y="49" class="b">FlowEdit ODE: delta += dsigma * [ v(x_tar, C_tar) - v(x_src, C_src) ]</text>
  <text x="212" y="66" class="s">no inversion; identity when C_tar = C_src</text>
  <line x1="474" y1="53" x2="496" y2="53"/>
  <rect class="box" x="500" y="36" width="60" height="34"/><text x="508" y="57" class="b">decode</text>
  <line x1="560" y1="53" x2="582" y2="53"/>
  <rect x="586" y="36" width="64" height="34" fill="#eee" stroke="#999"/><text x="592" y="57" class="b">edited</text>
  <text x="14" y="116" class="t">C_tar = frequency-surgery of C_src</text>
  <rect x="14" y="128" width="80" height="24" fill="#8fc0ff" stroke="#5b8bd0"/><text x="20" y="144" class="b">low: source</text>
  <rect x="96" y="128" width="90" height="24" fill="#ffce8f" stroke="#d09a5b"/><text x="102" y="144" class="b">high: style</text>
  <text x="200" y="144" class="s">band_swap(C_src, C_style, cut) on the T5 token-frequency spectrum</text>
  <text x="14" y="182" class="s">recon (C_tar=C_src) reproduces the source exactly -> validates the VAE/packing path (the gate).</text>
  <text x="14" y="200" class="s">Metrics: CLIP-to-style (edit adherence) vs CLIP-to-source + pixel distance (content kept).</text>
</svg>
<figcaption style="font:11px system-ui;color:#777">Schematic — E31 FlowEdit with frequency-surgery conditioning.</figcaption>
</figure>
'''


def _site(report):
    try:
        from e27_site import data_uri
    except Exception:
        data_uri = None

    def emb(rel):
        p = os.path.join(OUT, rel)
        if data_uri and os.path.exists(p):
            return f"<img src='{data_uri(p)}' style='max-width:100%'>"
        return f"<img src='{rel}' style='max-width:100%'>"

    def f(a):
        return f"{a['mean']:.3f}" if a else "—"

    h = ["<!doctype html><meta charset=utf-8><title>E31 FlowEdit + freq</title>",
         "<style>body{font:14px/1.5 system-ui;max-width:1100px;margin:2rem auto;"
         "padding:0 1rem;color:#222}code{background:#f0f0f0;padding:1px 4px;border-radius:3px}"
         "table{border-collapse:collapse;margin:.5em 0}td,th{border:1px solid #ccc;"
         "padding:3px 8px;text-align:right}td:first-child,th:first-child{text-align:left}"
         "img{border:1px solid #ddd;margin:.4em 0}</style>",
         "<h1>E31 — Real-image editing via FlowEdit + frequency-surgery conditioning</h1>",
         "<p>FlowEdit edits a flow model's output without inversion by integrating the "
         "difference between target- and source-conditioned velocities. Here the <b>target "
         "conditioning is a token-frequency surgery</b> of the source conditioning "
         "(low band from the source prompt, high band from the edit/style prompt). "
         "<code>recon</code> (target = source) reproduces the source exactly, validating "
         "the pipeline.</p>",
         _SCHEMATIC_SVG]
    for key, e in report["sources"].items():
        h.append(f"<h2>{key}</h2><p>source=<code>{e['src']}</code> · "
                 f"style=<code>{e['style']}</code></p>")
        h.append(emb(f"{key}/strip.png"))
        h.append("<table><tr><th>condition</th><th>CLIP→style↑</th><th>CLIP→source</th>"
                 "<th>px-dist→source</th><th>aesthetic↑</th></tr>")
        for cond, sc in e["conds"].items():
            h.append(f"<tr><td>{cond}</td><td>{f(sc['clip_style'])}</td>"
                     f"<td>{f(sc['clip_src'])}</td><td>{sc['px_dist_to_src']:.3f}</td>"
                     f"<td>{f(sc['aesthetic'])}</td></tr>")
        h.append("</table>")
    with open(os.path.join(OUT, "index.html"), "w") as fh:
        fh.write("\n".join(h))


# ---------------------------------------------------------------------------
# model-free preflight (FlowEdit math on a synthetic linear field)
# ---------------------------------------------------------------------------

def preflight():
    torch.manual_seed(0)
    sig = torch.linspace(1, 0, 9)
    x0 = torch.randn(1, PACK_TOKENS, 64)
    eps = torch.randn_like(x0)
    A_src, A_tar = torch.randn_like(x0), torch.randn_like(x0)

    def run(Asrc, Atar, skip=0.0):
        delta = torch.zeros_like(x0)
        steps = len(sig) - 1
        for i in range(int(skip * steps), steps):
            delta = delta + (float(sig[i + 1]) - float(sig[i])) * (Atar - Asrc)
        return x0 + delta
    # identity: C_tar == C_src -> exact reconstruction
    assert torch.allclose(run(A_src, A_src), x0, atol=1e-5), "identity not exact"
    # constant-field delta integrates to (sig_end - sig_start)*(A_tar - A_src)
    out = run(A_src, A_tar)
    expect = x0 + (float(sig[-1]) - float(sig[0])) * (A_tar - A_src)
    assert torch.allclose(out, expect, atol=1e-4), "delta accumulation wrong"
    print("[e31] preflight OK (FlowEdit identity + linear-field accumulation)")


# ---------------------------------------------------------------------------

def main(args):
    global OUT
    if args.out_tag:
        OUT = os.path.join(RESULTS, f"e31_{args.out_tag}")
    os.makedirs(OUT, exist_ok=True)
    parts = [p.strip() for p in args.part.split(",") if p.strip()]
    if "preflight" in parts:
        preflight()
    if "gen" in parts:
        run_gen(args)
    if "analyze" in parts:
        run_analyze(args)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="gen,analyze")
    ap.add_argument("--num", type=int, default=3, help="number of source scenes")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--skip", type=float, default=0.33,
                    help="fraction of top (noisy) steps to skip -> edit strength")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--guidance", type=float, default=3.5)
    ap.add_argument("--real_dir", default="", help="dir of <key>.png real images (else generate source)")
    ap.add_argument("--clip_model", default="openai/clip-vit-large-patch14")
    ap.add_argument("--out_tag", default="")
    main(ap.parse_args())
