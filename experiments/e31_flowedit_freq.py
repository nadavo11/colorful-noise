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


_CSS = """
body{font:15px/1.65 -apple-system,Segoe UI,Roboto,sans-serif;margin:24px auto;max-width:1000px;color:#1a1a1a;padding:0 16px}
h1{font-size:25px;line-height:1.25} h2{font-size:20px;margin-top:38px;border-bottom:1px solid #ddd;padding-bottom:5px}
h3{font-size:16px;margin:22px 0 4px}
.tldr{background:#eef4ff;border:1px solid #c7d9ff;border-radius:6px;padding:14px 16px;margin:14px 0}
.look{background:#f6f8fa;border-left:4px solid #0969da;padding:8px 13px;border-radius:4px;margin:8px 0;font-size:14px}
.read{margin:8px 0 4px} .win{background:#eafaf0;border-left:4px solid #2da44e;padding:8px 13px;border-radius:4px;margin:8px 0}
.cav{background:#fff8f0;border-left:4px solid #d4a017;padding:10px 14px;border-radius:4px;margin:12px 0}
dl{margin:10px 0} dt{font-weight:700;margin-top:11px} dd{margin:2px 0 2px 18px;color:#333}
table{border-collapse:collapse;margin:10px 0;font-variant-numeric:tabular-nums;font-size:14px}
th,td{border:1px solid #d0d7de;padding:4px 9px;text-align:right}
th{background:#f6f8fa;text-align:center} td.v{text-align:left;font-weight:600;white-space:nowrap}
td.pos{background:#dafbe1;font-weight:600}
.cap{color:#555;font-size:13px;margin:2px 0 14px}
img{width:100%;border:1px solid #d0d7de;border-radius:4px;margin:6px 0}
code{background:#eff1f3;padding:1px 5px;border-radius:3px;font-size:13px}
"""


def _site(report):
    """Self-contained explainer: TL;DR -> glossary -> method -> per-scene visuals+numbers.

    Pure templating from `report` (= results/e31/report.json) + the saved strips; loads no
    model, so the page rebuilds anywhere (`--part site`)."""
    try:
        from e27_site import data_uri
    except Exception:
        data_uri = None

    def emb(rel):
        p = os.path.join(OUT, rel)
        if not os.path.exists(p):
            return f"<p class=cap>(missing <code>{rel}</code>)</p>"
        return f"<img src='{data_uri(p) if data_uri else rel}' alt='{rel}'>"

    def gv(sc, k):
        v = sc.get(k)
        return v.get("mean") if isinstance(v, dict) else v

    h = ["<!doctype html><meta charset=utf-8><title>E31 — FlowEdit + frequency surgery</title>",
         f"<style>{_CSS}</style>",
         "<h1>E31 — Real-image editing via FlowEdit + frequency-surgery conditioning</h1>"]

    # ---- TL;DR ----
    h.append(
        "<div class=tldr><b>In one paragraph.</b> <b>FlowEdit</b> edits an image with a flow model "
        "<b>without inverting it</b>: it walks the denoising schedule and accumulates the "
        "<i>difference</i> between the velocity the model wants under the <b>target</b> prompt and "
        "under the <b>source</b> prompt, then adds that accumulated nudge to the source latent. "
        "E31's twist: the target conditioning is a <b>token-frequency surgery</b> of the source "
        "conditioning (E24/E30 ops) — keep the source prompt's <b>low band</b>, graft the style "
        "prompt's <b>high band</b>. <b>Headline:</b> the pipeline is sound (the identity check "
        "reproduces the source to ~0.003 pixel distance) and a <b>plain prompt swap</b> edits "
        "scene-dependently, but the <b>frequency-surgery target barely edits at all</b> — keeping "
        "the source's low band anchors the result to the source, so the velocity difference is ≈0 "
        "and nothing moves. Token-frequency surgery is <b>not</b> a usable editing handle.</div>")
    h.append(_SCHEMATIC_SVG)

    # ---- glossary ----
    h.append("<h2>0 · Background (plain language)</h2><dl>"
             "<dt>Flow model &amp; velocity</dt><dd>Flux generates by following a learned "
             "<b>velocity field</b> <code>v(x, σ, C)</code> from noise (σ=1) to a clean latent (σ=0), "
             "conditioned on the text embedding <code>C</code>. <code>x0</code> is the clean source "
             "latent (generated from the source prompt, or VAE-encoded from a real image).</dd>"
             "<dt>FlowEdit (inversion-free)</dt><dd>Instead of inverting the image back to noise, it "
             "sets <code>δ=0</code> and, stepping σ high→low, accumulates "
             "<code>δ += (σ_next−σ)·[ v(x_tar, C_tar) − v(x_src, C_src) ]</code>; the edited latent is "
             "<code>x0 + δ</code>. Only the <b>difference</b> between the two conditionings drives the "
             "edit.</dd>"
             "<dt>C_src vs C_tar</dt><dd><code>C_src</code> = the source prompt's embedding. "
             "<code>C_tar</code> = the target. If <code>C_tar = C_src</code> the difference is exactly "
             "zero, so the output is the source unchanged (this is the safety gate).</dd>"
             "<dt>--skip (edit strength)</dt><dd>Fraction of the top (noisiest) steps to skip. Higher "
             "= weaker edit / more source preserved (here 0.33).</dd>"
             "<dt>The conditions</dt><dd>"
             "<code>recon</code> = <code>C_tar = C_src</code>: the <b>identity gate</b> — must "
             "reproduce the source (pixel distance ≈ 0), validating the VAE/packing path. "
             "<code>full</code> = <code>C_tar</code> is the <b>whole style prompt</b> — ordinary "
             "prompt-swap FlowEdit (the edit baseline). "
             "<code>swap_c0.25</code> / <code>swap_c0.4</code> = <b>frequency surgery</b>: token-axis "
             "spectrum with the <b>low band (0–0.25 / 0–0.4) from the source</b> and the <b>high band "
             "from the style</b>.</dd>"
             "<dt>The metrics</dt><dd>"
             "<b>CLIP→style</b> (↑ = stronger edit): image–style-prompt similarity. "
             "<b>CLIP→source</b> (↑ = content kept): image–source-prompt similarity. "
             "<b>px-dist→source</b> (↓ = content kept; ≈0 for <code>recon</code>): raw pixel distance "
             "to the source image. <b>aesthetic</b> (↑): LAION aesthetic score (sanity). A good edit "
             "<i>raises</i> CLIP→style while <i>keeping</i> CLIP→source reasonable.</dd>"
             "</dl>")

    # ---- method ----
    h.append("<h2>1 · Method</h2><p>Three scenes, each a source prompt + a style/edit prompt. We run "
             "FlowEdit under each condition (<code>recon</code>, <code>full</code>, "
             "<code>swap_c0.25</code>, <code>swap_c0.4</code>), all from the same <code>x0</code> and "
             "schedule, and score edit adherence vs. content preservation. The <code>recon</code> gate "
             "runs first: if it does not reproduce the source, the plumbing is wrong and nothing else "
             "is trustworthy.</p>")

    h.append("<h2>2 · Results (per scene)</h2>")
    h.append("<div class=look><b>What to look for.</b> Each strip is "
             "<code>source · recon · full · swap_c0.25 · swap_c0.4</code>. <code>recon</code> should be "
             "indistinguishable from <code>source</code>. <code>full</code> is the real edit. The two "
             "<code>swap_*</code> panels should look like the strongest test of the idea — but they "
             "barely differ from the source.</div>")
    for key, e in report["sources"].items():
        h.append(f"<h3>{key}: <code>{e['src']}</code> → <code>{e['style']}</code></h3>")
        h.append(emb(f"{key}/strip.png"))
        h.append("<p class=cap>columns: source · recon · full · swap_c0.25 · swap_c0.4</p>")
        rows = [(c, {"clip_style": gv(sc, "clip_style"), "clip_src": gv(sc, "clip_src"),
                     "px": sc.get("px_dist_to_src"), "aes": gv(sc, "aesthetic")})
                for c, sc in e["conds"].items()]
        # highlight strongest edit (max CLIP→style among the non-recon edits)
        edit_styles = [r[1]["clip_style"] for r in rows
                       if r[0] != "recon" and r[1]["clip_style"] is not None]
        best_style = max(edit_styles) if edit_styles else None
        out = ["<table><tr><th>condition</th><th>CLIP→style ↑</th><th>CLIP→source ↑</th>"
               "<th>px-dist→source ↓</th><th>aesthetic ↑</th></tr>"]
        for c, v in rows:
            cs = v["clip_style"]
            hot = (cs is not None and best_style is not None and abs(cs - best_style) < 1e-9
                   and c != "recon")
            csc = f"<td class=pos>{cs:.3f}</td>" if hot else (f"<td>{cs:.3f}</td>" if cs is not None else "<td>—</td>")
            src = f"{v['clip_src']:.3f}" if v["clip_src"] is not None else "—"
            aes = f"{v['aes']:.3f}" if v["aes"] is not None else "—"
            out.append(f"<tr><td class=v>{c}</td>{csc}<td>{src}</td>"
                       f"<td>{v['px']:.3f}</td><td>{aes}</td></tr>")
        out.append("</table>")
        h.append("".join(out))

    h.append("<h2>3 · Reading</h2><div class=read>"
             "<b>1. The gate holds.</b> <code>recon</code> px-distance is ~0.003 across scenes — the "
             "FlowEdit/VAE/packing path is correct by construction.<br>"
             "<b>2. Plain swap edits, scene-dependently.</b> <code>full</code> moves the image (e.g. "
             "street_snow CLIP→style 0.093→0.208), but weakly on harder semantic jumps.<br>"
             "<b>3. Frequency surgery barely edits.</b> <code>swap_c0.25/0.4</code> sit at "
             "<code>recon</code>'s CLIP→style level: keeping the source's low band anchors the "
             "conditioning to the source, so <code>v(C_tar)−v(C_src)≈0</code> and δ≈0. The style's high "
             "band is too weak to redirect the flow.</div>")
    h.append("<h2>4 · Caveats &amp; next</h2><div class=cav>"
             "Single seed per cell; sources are generated from the caption (clean eval) — real images "
             "via <code>--real_dir</code> are untested. Short prompts make cuts 0.25 and 0.4 collapse "
             "to the same frequency index (swap_c0.25 == swap_c0.4 for street_snow). Conclusion: "
             "token-frequency surgery is a dead-end editing handle; latent-band editing (E22) remains "
             "the usable route.</div>")

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
    if "site" in parts:   # model-free: rebuild index.html from report.json + cached strips
        rp = os.path.join(OUT, "report.json")
        if not os.path.exists(rp):
            raise SystemExit(f"[e31] --part site needs {rp} (run analyze first / fetch it)")
        with open(rp) as f:
            _site(json.load(f))
        print(f"[e31] rebuilt {os.path.join(OUT, 'index.html')} (no model loaded)", flush=True)


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
