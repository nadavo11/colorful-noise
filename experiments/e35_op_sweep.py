"""E35: systematic sweep of the token-frequency operator toolkit on SD1.5.

We have ~13 token-axis FFT operators on the text conditioning (low/high-pass, band gain,
notch, phase-only/mag-only, phase band-keep, phase gain, two-prompt swap/blend/lerp,
per-object band gain). E24/E30/E32 validated pieces on Flux. E35 characterises the WHOLE
toolkit on a fast model (SD1.5): for each operator x parameter level x prompt category,
what happens to prompt ADHERENCE and image FIDELITY, and how much the edit moves the image.

Signal: SD1.5's CLIP text embedding E (1, 77, 768). Ops FFT the 77-token axis on the real
span [:L] (L = attention_mask.sum(); padding left untouched), then we generate from the
edited prompt_embeds (CFG with an empty-string negative). 5 seeds per condition, batched.

Metrics per image (all reuse repo utils):
  - clip      : CLIP-T adherence to the prompt (+ vs A/B for pairs, + vs object phrase)
  - aesthetic : LAION aesthetic predictor (no-ref fidelity, reuses the same CLIP)
  - sharpness/hf_frac/colorfulness : cheap image stats (e9_bandnorm_classes)
  - drift     : 1 - cosine(CLIP_img(baseline), CLIP_img(edit))  -- how far the edit moved it

Parts (--part): preflight (no GPU; prints prompts/spans/counts/ETA), gen (SD1.5 batched
generation, cached), analyze (score + aggregate + per-operator plots + grids + index.html).

Cluster: ship via kubectl cp; self-gating smoke -> CLIP gate -> full (cluster_e35_job.sh).
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, save_grid
from e9_clipt import agg, load_clip, clip_scores
from e9_bandnorm_classes import image_metrics
from fidelity_metrics import load_aesthetic, aesthetic_scores
import text_spectral_ops as TS

OUT = os.path.join(RESULTS, "e35")
SD15_IDS = ["sd-legacy/stable-diffusion-v1-5", "runwayml/stable-diffusion-v1-5"]
DTYPE = torch.float16
SIZE = 512
GUIDANCE = 7.5
OBJ_CUT = 0.51            # E32 per-object median split
IMG_KEYS = ["sharpness", "hf_frac", "colorfulness"]

# ---------------------------------------------------------------------------
# prompt set: 5 categories x 5 prompts; object/two-object carry an `obj` phrase
# ---------------------------------------------------------------------------
PROMPTS = [
    # short / simple
    {"id": "short_cat", "cat": "short", "text": "a cat"},
    {"id": "short_bike", "cat": "short", "text": "a red bicycle"},
    {"id": "short_mtn", "cat": "short", "text": "a mountain landscape"},
    {"id": "short_coffee", "cat": "short", "text": "a cup of coffee"},
    {"id": "short_chair", "cat": "short", "text": "a wooden chair"},
    # long / detailed
    {"id": "long_market", "cat": "long",
     "text": "a bustling medieval marketplace at dawn, wooden stalls piled with fruit, "
             "merchants in colorful robes, cobblestone street, soft golden light, fog"},
    {"id": "long_diner", "cat": "long",
     "text": "the interior of a 1950s american diner at night, red vinyl booths, chrome "
             "stools, neon sign reflected in rain on the window, a half-empty coffee cup"},
    {"id": "long_lab", "cat": "long",
     "text": "a cluttered alchemist's laboratory, glass flasks bubbling with green liquid, "
             "leather books stacked on an oak desk, candlelight, dust motes in the air"},
    {"id": "long_beach", "cat": "long",
     "text": "a tropical beach at sunset, turquoise water, palm trees bending in the wind, "
             "a small wooden boat on white sand, distant storm clouds on the horizon"},
    {"id": "long_space", "cat": "long",
     "text": "an astronaut floating outside a space station above earth, sunlight glinting "
             "off the visor, solar panels, the blue curve of the planet below, stars"},
    # art / style
    {"id": "style_vangogh", "cat": "style",
     "text": "a wheat field with cypress trees, in the style of Van Gogh, thick swirling brushstrokes"},
    {"id": "style_cubism", "cat": "style",
     "text": "a portrait of a man, in the style of Picasso cubism, fragmented geometric planes"},
    {"id": "style_cyber", "cat": "style",
     "text": "a city street at night, cyberpunk neon, blade runner aesthetic, rain"},
    {"id": "style_watercolor", "cat": "style",
     "text": "a still life of fruit in a bowl, soft watercolor painting, pastel washes"},
    {"id": "style_ukiyoe", "cat": "style",
     "text": "a great wave and a mountain, ukiyo-e japanese woodblock print"},
    # single object (per-object applies; obj appears verbatim)
    {"id": "obj_cat", "cat": "object", "text": "a fluffy orange tabby cat sitting on a windowsill",
     "obj": "a fluffy orange tabby cat"},
    {"id": "obj_car", "cat": "object", "text": "a shiny red sports car on a mountain road",
     "obj": "a shiny red sports car"},
    {"id": "obj_castle", "cat": "object", "text": "a tall stone medieval castle on a green hill",
     "obj": "a tall stone medieval castle"},
    {"id": "obj_teapot", "cat": "object", "text": "a white ceramic teapot on a wooden table",
     "obj": "a white ceramic teapot"},
    {"id": "obj_cactus", "cat": "object", "text": "a small green potted cactus by a window",
     "obj": "a small green potted cactus"},
    # two-object / compositional (per-object on the first object)
    {"id": "two_catdog", "cat": "twoobj",
     "text": "a fluffy orange tabby cat and a sleeping golden retriever dog",
     "obj": "a fluffy orange tabby cat", "objB": "a sleeping golden retriever dog"},
    {"id": "two_carbike", "cat": "twoobj",
     "text": "a shiny red sports car and a rusty old bicycle",
     "obj": "a shiny red sports car", "objB": "a rusty old bicycle"},
    {"id": "two_cactusrose", "cat": "twoobj",
     "text": "a tall green cactus and a single red rose",
     "obj": "a tall green cactus", "objB": "a single red rose"},
    {"id": "two_guitarpiano", "cat": "twoobj",
     "text": "a wooden acoustic guitar and a black grand piano",
     "obj": "a wooden acoustic guitar", "objB": "a black grand piano"},
    {"id": "two_owlfox", "cat": "twoobj",
     "text": "a white snowy owl and a small red fox",
     "obj": "a white snowy owl", "objB": "a small red fox"},
]
PAIRS = [  # two-prompt merge ops: distinct A,B
    ("pair_cat_car", "a photograph of an orange tabby cat", "a photograph of a red sports car"),
    ("pair_castle_forest", "an oil painting of a medieval castle", "a dense pine forest in fog"),
    ("pair_house_vangogh", "a house by a lake", "in the style of Van Gogh, swirling brushstrokes"),
    ("pair_portrait_cyber", "a portrait of a woman", "cyberpunk neon city at night"),
    ("pair_teapot_cup", "a silver teapot", "a porcelain teacup"),
    ("pair_owl_fox", "a snowy owl", "a red fox"),
]


# ---------------------------------------------------------------------------
# parameter grids per coverage level
# ---------------------------------------------------------------------------
def grids(cov):
    if cov == "quick":
        return dict(filter_cuts=[0.15, 0.4], mag_gains=[0.5, 2.0], phase_gains=[0.5, 2.0],
                    perobj_gains=[0.5, 2.0], gain_cut=0.25, notch_cuts=[0.25],
                    keep_cuts=[0.25, 0.5], swap_cuts=[0.25], blend=(0.25, 0.15), alpha=0.5)
    if cov == "max":
        return dict(filter_cuts=[0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8],
                    mag_gains=[0.25, 0.5, 0.75, 1.5, 2, 3], phase_gains=[0.25, 0.5, 1.5, 2],
                    perobj_gains=[0.25, 0.5, 1.5, 2], gain_cut=0.25, notch_cuts=[0.2, 0.35, 0.5],
                    keep_cuts=[0.2, 0.35, 0.5, 0.65], swap_cuts=[0.15, 0.25, 0.4, 0.6],
                    blend=(0.25, 0.15), alpha=0.5)
    return dict(filter_cuts=[0.1, 0.2, 0.3, 0.45, 0.65], mag_gains=[0.25, 0.5, 1.5, 2, 3],
                phase_gains=[0.5, 2.0], perobj_gains=[0.5, 2.0], gain_cut=0.25,
                notch_cuts=[0.25, 0.5], keep_cuts=[0.2, 0.35, 0.5], swap_cuts=[0.15, 0.25, 0.4],
                blend=(0.25, 0.15), alpha=0.5)


def single_conditions(g):
    """[(cond_name, op, params)] for single-prompt operators."""
    c = [("baseline", "baseline", {})]
    for x in g["filter_cuts"]:
        c.append((f"lowpass_c{x}", "lowpass", {"cut": x}))
    for x in g["filter_cuts"]:
        c.append((f"highpass_c{x}", "highpass", {"cut": x}))
    for b in ("low", "high"):
        for gg in g["mag_gains"]:
            c.append((f"bandgain_{b}_g{gg}", "bandgain", {"band": b, "gain": gg, "cut": g["gain_cut"]}))
    for b in ("low", "high"):
        for x in g["notch_cuts"]:
            c.append((f"notch_{b}_c{x}", "notch", {"band": b, "cut": x}))
    c.append(("phaseonly", "phaseonly", {}))
    c.append(("magonly", "magonly", {}))
    for b in ("low", "high"):
        for x in g["keep_cuts"]:
            c.append((f"phasekeep_{b}_c{x}", "phasekeep", {"band": b, "cut": x}))
    for b in ("low", "high"):
        for gg in g["phase_gains"]:
            c.append((f"phasegain_{b}_g{gg}", "phasegain", {"band": b, "gain": gg, "cut": g["gain_cut"]}))
    return c


def object_conditions(g):
    c = []
    for b in ("low", "high"):
        for gg in g["perobj_gains"]:
            c.append((f"perobj_{b}_g{gg}", "perobject", {"band": b, "gain": gg}))
    return c


def pair_conditions(g):
    c = [("baseline", "baseline", {})]
    for x in g["swap_cuts"]:
        c.append((f"swap_c{x}", "swap", {"cut": x}))
    bc, bw = g["blend"]
    c.append((f"blend_c{bc}", "blend", {"cut": bc, "width": bw}))
    c.append((f"lerp_a{g['alpha']}", "lerp", {"alpha": g["alpha"]}))
    return c


# ---------------------------------------------------------------------------
# token span (CLIP slow tokenizer -> id-subsequence fallback)
# ---------------------------------------------------------------------------
def phrase_span(tokenizer, prompt, phrase, L):
    try:
        enc = tokenizer(prompt, max_length=77, truncation=True, return_offsets_mapping=True)
        offs = enc["offset_mapping"]
        c0 = prompt.index(phrase); c1 = c0 + len(phrase)
        idx = [i for i, (s, e) in enumerate(offs) if i < L and e > s and s < c1 and e > c0]
        if idx:
            return min(idx), min(max(idx) + 1, L)
    except (ValueError, KeyError, TypeError, NotImplementedError):
        pass
    pid = tokenizer(prompt, max_length=77, truncation=True)["input_ids"]
    ph = tokenizer(phrase, add_special_tokens=False)["input_ids"]
    for i in range(len(pid) - len(ph) + 1):
        if pid[i:i + len(ph)] == ph:
            return i, min(i + len(ph), L)
    raise ValueError(f"could not locate phrase {phrase!r} in prompt {prompt!r}")


def _phase_only(E):
    F = torch.fft.rfft(E.float(), dim=1)
    return torch.fft.irfft(torch.polar(torch.ones_like(F.abs()), torch.angle(F)),
                           n=E.shape[1], dim=1).to(E.dtype)


def _mag_only(E):
    F = torch.fft.rfft(E.float(), dim=1)
    return torch.fft.irfft(torch.polar(F.abs(), torch.zeros_like(F.abs())),
                           n=E.shape[1], dim=1).to(E.dtype)


def apply_op(op, params, pe, L, tok, prompt, peB=None, LB=None):
    """Return edited prompt_embeds (1,77,768). Reuses text_spectral_ops on the [:L] span."""
    span = lambda fn: TS.apply_on_span(fn, pe, L)
    if op == "baseline":
        return pe
    if op == "lowpass":
        c = params["cut"]; return span(lambda x: TS.band_filter_1d(x, 0.0, c))
    if op == "highpass":
        c = params["cut"]; return span(lambda x: TS.band_filter_1d(x, c, 1.0, keep_dc=True))
    if op == "bandgain":
        c = params["cut"]; lo, hi = (0.0, c) if params["band"] == "low" else (c, 1.0)
        g = params["gain"]; return span(lambda x: TS.band_gain_1d(x, lo, hi, g))
    if op == "notch":
        c = params["cut"]; lo, hi = (0.0, c) if params["band"] == "low" else (c, 1.0)
        return span(lambda x: TS.band_notch_1d(x, lo, hi))
    if op == "phaseonly":
        return span(_phase_only)
    if op == "magonly":
        return span(_mag_only)
    if op == "phasekeep":
        c = params["cut"]; lo, hi = (0.0, c) if params["band"] == "low" else (c, 1.0)
        return span(lambda x: TS.band_phase_filter_1d(x, lo, hi))
    if op == "phasegain":
        c = params["cut"]; lo, hi = (0.0, c) if params["band"] == "low" else (c, 1.0)
        g = params["gain"]; return span(lambda x: TS.band_phase_gain_1d(x, lo, hi, g))
    if op == "perobject":
        a, b = params["_span"]
        lo, hi = (0.0, OBJ_CUT) if params["band"] == "low" else (OBJ_CUT, 1.0)
        g = params["gain"]; return TS.apply_on_subspan(lambda x: TS.band_gain_1d(x, lo, hi, g), pe, a, b)
    Lm = min(L, LB)
    if op == "swap":
        c = params["cut"]; return TS.apply_on_span(lambda x: TS.band_swap_1d(x, peB[:, :Lm], c), pe, Lm)
    if op == "blend":
        c = params["cut"]; w = params["width"]
        return TS.apply_on_span(lambda x: TS.band_blend_1d(x, peB[:, :Lm], c, w), pe, Lm)
    if op == "lerp":
        a = params["alpha"]; return TS.apply_on_span(lambda x: TS.lerp_embeds(x, peB[:, :Lm], a), pe, Lm)
    raise ValueError(f"unknown op {op}")


# ---------------------------------------------------------------------------
# Part: preflight (no GPU)
# ---------------------------------------------------------------------------
def run_preflight(args):
    from transformers import CLIPTokenizer
    tok = None
    for mid in SD15_IDS:
        try:
            tok = CLIPTokenizer.from_pretrained(mid, subfolder="tokenizer"); break
        except Exception:
            continue
    g = grids(args.coverage)
    prompts = PROMPTS[: args.num_prompts]
    n_single = len(single_conditions(g))
    n_obj = len(object_conditions(g))
    n_objp = sum(1 for p in prompts if "obj" in p)
    n_pair = len(pair_conditions(g)) - 1  # baseline shared with single? pairs separate dir
    pairs = PAIRS[: args.num_prompts]
    total_cond = len(prompts) * n_single + n_objp * n_obj + len(pairs) * len(pair_conditions(g))
    total_imgs = total_cond * args.seeds
    eta_h = total_cond * args.batch_eta_s / 3600.0
    print(f"[e35] coverage={args.coverage}  prompts={len(prompts)} pairs={len(pairs)} "
          f"seeds={args.seeds}", flush=True)
    print(f"[e35] conditions: {n_single}/prompt single, {n_obj}/obj-prompt object "
          f"({n_objp} obj prompts), {len(pair_conditions(g))}/pair", flush=True)
    print(f"[e35] TOTAL conditions={total_cond}  images={total_imgs}  "
          f"ETA~{eta_h:.1f}h @ {args.batch_eta_s}s/batch", flush=True)
    if tok is not None:
        print("[e35] per-object span check:", flush=True)
        for p in prompts:
            if "obj" not in p:
                continue
            ids = tok(p["text"], max_length=77, truncation=True)["input_ids"]
            L = len(ids)
            try:
                a, b = phrase_span(tok, p["text"], p["obj"], L)
                ok = "OK"
            except Exception as e:
                a = b = -1; ok = f"FAIL ({e})"
            print(f"[e35]   {p['id']:16s} L={L:2d} obj span=({a},{b}) {ok}", flush=True)
    else:
        print("[e35] (tokenizer unavailable offline; span check skipped)", flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "plan.json"), "w") as f:
        json.dump({"coverage": args.coverage, "total_conditions": total_cond,
                   "total_images": total_imgs, "seeds": args.seeds}, f, indent=2)


# ---------------------------------------------------------------------------
# SD1.5 load / encode / batched generate
# ---------------------------------------------------------------------------
def load_sd15():
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    pipe = None
    for mid in SD15_IDS:
        try:
            pipe = StableDiffusionPipeline.from_pretrained(
                mid, torch_dtype=DTYPE, safety_checker=None, requires_safety_checker=False)
            break
        except Exception as e:
            print(f"[e35] load {mid} failed: {e}", flush=True)
    if pipe is None:
        raise RuntimeError("could not load SD1.5")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def encode(pipe, text):
    with torch.no_grad():
        pe, _ = pipe.encode_prompt(text, "cuda", 1, False)
    ids = pipe.tokenizer(text, max_length=77, truncation=True)["input_ids"]
    return pe.detach(), len(ids)


@torch.no_grad()
def gen_batch(pipe, pe, neg, seeds, steps):
    """Generate len(seeds) images from one edited prompt_embeds (1,77,768)."""
    n = len(seeds)
    peb = pe.to("cuda").expand(n, -1, -1)
    negb = neg.to("cuda").expand(n, -1, -1)
    gens = [torch.Generator("cuda").manual_seed(int(s)) for s in seeds]
    try:
        imgs = pipe(prompt_embeds=peb, negative_prompt_embeds=negb,
                    num_inference_steps=int(steps), guidance_scale=GUIDANCE,
                    height=SIZE, width=SIZE, generator=gens).images
    except Exception:  # fallback: per-seed loop
        imgs = []
        for s in seeds:
            im = pipe(prompt_embeds=pe.to("cuda"), negative_prompt_embeds=neg.to("cuda"),
                      num_inference_steps=int(steps), guidance_scale=GUIDANCE,
                      height=SIZE, width=SIZE,
                      generator=torch.Generator("cuda").manual_seed(int(s))).images[0]
            imgs.append(im)
    return imgs


def _gen_condition(pipe, d, conds, pe, L, neg, tok, text, args, peB=None, LB=None):
    seeds = list(range(args.seeds))
    for name, op, params in conds:
        paths = [os.path.join(d, f"{name}_s{s}.png") for s in seeds]
        if all(os.path.exists(p) for p in paths):
            continue
        try:
            mpe = apply_op(op, params, pe, L, tok, text, peB, LB)
        except Exception as e:
            print(f"[e35]   skip {name}: {e}", flush=True)
            continue
        imgs = gen_batch(pipe, mpe, neg, seeds, args.steps)
        os.makedirs(d, exist_ok=True)
        for im, p in zip(imgs, paths):
            im.save(p)
    print(f"[e35] gen {os.path.relpath(d, OUT)} done", flush=True)


def run_gen(args):
    g = grids(args.coverage)
    pipe = load_sd15()
    neg, _ = encode(pipe, "")
    single = single_conditions(g)
    objconds = object_conditions(g)
    enc_cache = {}
    for p in PROMPTS[: args.num_prompts]:
        pe, L = enc_cache.setdefault(p["text"], encode(pipe, p["text"]))
        d = os.path.join(OUT, p["id"])
        conds = list(single)
        if "obj" in p:
            try:
                a, b = phrase_span(pipe.tokenizer, p["text"], p["obj"], L)
                conds = conds + [(n, op, {**pr, "_span": (a, b)}) for n, op, pr in objconds]
            except Exception as e:
                print(f"[e35] {p['id']} per-object skipped: {e}", flush=True)
        _gen_condition(pipe, d, conds, pe, L, neg, pipe.tokenizer, p["text"], args)
    # two-prompt pairs
    pconds = pair_conditions(g)
    for pid, A, B in PAIRS[: args.num_prompts]:
        peA, LA = enc_cache.setdefault(A, encode(pipe, A))
        peB, LB = enc_cache.setdefault(B, encode(pipe, B))
        _gen_condition(pipe, os.path.join(OUT, pid), pconds, peA, LA, neg, pipe.tokenizer, A,
                       args, peB=peB, LB=LB)


# ---------------------------------------------------------------------------
# Part: analyze (score + aggregate + plots + grids + html)
# ---------------------------------------------------------------------------
SWEEP = {  # op -> (x-key, has_band)
    "lowpass": ("cut", False), "highpass": ("cut", False),
    "notch": ("cut", True), "phasekeep": ("cut", True),
    "bandgain": ("gain", True), "phasegain": ("gain", True),
    "swap": ("cut", False),
}


def _img_feat(clip, imgs):
    from clip_sim import clip_image_features
    return clip_image_features(clip[0], clip[1], imgs)


def run_analyze(args):
    import clip_sim
    g = grids(args.coverage)
    clip = load_clip(args.clip_model)               # (model, proc) -- shared by all CLIP metrics
    mlp = load_aesthetic()
    prompts = {p["id"]: p for p in PROMPTS[: args.num_prompts]}
    pair_map = {pid: (A, B) for pid, A, B in PAIRS[: args.num_prompts]}
    raw = {}

    def score_dir(pid, conds, main_prompt, extra_prompts):
        d = os.path.join(OUT, pid)
        if not os.path.isdir(d):
            return
        ent = {}
        # baseline image features per seed (for drift)
        base_feat = {}
        for s in range(args.seeds):
            bp = os.path.join(d, f"baseline_s{s}.png")
            if os.path.exists(bp):
                base_feat[s] = _img_feat(clip, [Image.open(bp).convert("RGB")])[0]
        for name, op, params in conds:
            per_seed = {}
            for s in range(args.seeds):
                p = os.path.join(d, f"{name}_s{s}.png")
                if not os.path.exists(p):
                    continue
                im = Image.open(p).convert("RGB")
                m = {"clip": clip_scores(*clip, main_prompt, [im])[0]}
                aes = aesthetic_scores(mlp, clip[0], clip[1], [im])[0]
                m["aesthetic"] = aes
                im_metrics = image_metrics(im)
                m.update({k: im_metrics[k] for k in IMG_KEYS})
                for label, pr in extra_prompts.items():
                    m[label] = clip_scores(*clip, pr, [im])[0]
                if s in base_feat:
                    f = _img_feat(clip, [im])[0]
                    m["drift"] = 1.0 - float(clip_sim.cosine(base_feat[s], f))
                per_seed[str(s)] = m
            ent[name] = {"op": op, "params": {k: v for k, v in params.items() if k != "_span"},
                         "seeds": per_seed}
        raw[pid] = {"cat": cat_of(pid, prompts, pair_map), "main": main_prompt, "conds": ent}
        print(f"[e35] scored {pid}", flush=True)

    single = single_conditions(g)
    objconds = object_conditions(g)
    for p in PROMPTS[: args.num_prompts]:
        conds = list(single)
        extra = {}
        if "obj" in p:
            conds = conds + objconds
            extra["clip_obj"] = p["obj"]
        if "objB" in p:
            extra["clip_objB"] = p["objB"]
        score_dir(p["id"], conds, p["text"], extra)
    for pid, A, B in PAIRS[: args.num_prompts]:
        score_dir(pid, pair_conditions(g), A, {"clip_B": B})

    summary = aggregate(raw)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "report.json"), "w") as f:
        json.dump({"params": vars(args), "summary": summary, "raw": raw}, f, indent=2)
    plots = make_plots(raw)
    grids_png = make_grids(g)
    _site(summary, plots, grids_png)
    print("[e35] wrote report.json, plots, grids, index.html", flush=True)


def cat_of(pid, prompts, pair_map):
    if pid in prompts:
        return prompts[pid]["cat"]
    return "pair"


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def aggregate(raw):
    """summary[op][band|'-'][xval][cat][metric] = mean across prompts+seeds.
    Also summary['_overall'][op][cat][metric] for discrete ops."""
    curves, overall = {}, {}
    for pid, e in raw.items():
        cat = e["cat"]
        for name, c in e["conds"].items():
            op, params = c["op"], c["params"]
            metrics = {}
            for mk in ("clip", "aesthetic", "drift", *IMG_KEYS):
                metrics[mk] = _mean([sd.get(mk) for sd in c["seeds"].values()])
            overall.setdefault(op, {}).setdefault(cat, {}).setdefault("_n", 0)
            o = overall[op][cat]
            o["_n"] += 1
            for mk, v in metrics.items():
                if v is not None:
                    o.setdefault(mk, []).append(v)
            if op in SWEEP:
                xkey, has_band = SWEEP[op]
                band = params.get("band", "-") if has_band else "-"
                xv = params.get(xkey)
                node = curves.setdefault(op, {}).setdefault(band, {}).setdefault(str(xv), {}) \
                            .setdefault(cat, {})
                for mk, v in metrics.items():
                    if v is not None:
                        node.setdefault(mk, []).append(v)
    # reduce overall lists -> means
    for op in overall:
        for cat in overall[op]:
            for mk in list(overall[op][cat]):
                if mk != "_n":
                    overall[op][cat][mk] = round(_mean(overall[op][cat][mk]), 4)
    # reduce curves lists -> means
    for op in curves:
        for band in curves[op]:
            for xv in curves[op][band]:
                for cat in curves[op][band][xv]:
                    for mk in list(curves[op][band][xv][cat]):
                        curves[op][band][xv][cat][mk] = round(_mean(curves[op][band][xv][cat][mk]), 4)
    return {"curves": curves, "overall": overall}


def make_plots(raw):
    """Per swept op (and band), a 1x3 panel: clip / aesthetic / drift vs param, line per cat."""
    summary = aggregate(raw)["curves"]
    cats = ["short", "long", "style", "object", "twoobj", "pair"]
    out = []
    pdir = os.path.join(OUT, "plots"); os.makedirs(pdir, exist_ok=True)
    for op, bands in summary.items():
        xkey = SWEEP[op][0]
        for band, xs in bands.items():
            xvals = sorted(xs, key=lambda v: float(v))
            fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
            for ax, mk in zip(axes, ("clip", "aesthetic", "drift")):
                for cat in cats:
                    ys = [xs[xv].get(cat, {}).get(mk) for xv in xvals]
                    xy = [(float(xv), y) for xv, y in zip(xvals, ys) if y is not None]
                    if xy:
                        ax.plot([a for a, _ in xy], [b for _, b in xy], marker="o", label=cat)
                ax.set_title(mk); ax.set_xlabel(xkey); ax.grid(alpha=0.3)
            axes[0].legend(fontsize=7)
            ttl = f"{op}" + (f" [{band} band]" if band != "-" else "")
            fig.suptitle(ttl); fig.tight_layout()
            fn = f"{op}_{band}.png".replace("-", "all")
            fig.savefig(os.path.join(pdir, fn), dpi=90); plt.close(fig)
            out.append(("plots/" + fn, ttl))
    return out


def make_grids(g):
    """Contact sheets: rows = one prompt per category, cols = a param sweep, seed 0."""
    out = []
    cat_prompt = {}
    for p in PROMPTS:
        cat_prompt.setdefault(p["cat"], p)
    # headline sweeps to visualise
    headline = [
        ("lowpass", [("baseline", "base")] + [(f"lowpass_c{c}", f"c{c}") for c in g["filter_cuts"]]),
        ("highpass", [("baseline", "base")] + [(f"highpass_c{c}", f"c{c}") for c in g["filter_cuts"]]),
        ("bandgain_high", [("baseline", "base")] + [(f"bandgain_high_g{gg}", f"g{gg}") for gg in g["mag_gains"]]),
        ("phasekeep", [("baseline", "base"), ("phaseonly", "phaseonly"), ("magonly", "magonly")]
         + [(f"phasekeep_low_c{c}", f"low_c{c}") for c in g["keep_cuts"]]
         + [(f"phasekeep_high_c{c}", f"high_c{c}") for c in g["keep_cuts"]]),
    ]
    gdir = os.path.join(OUT, "grids"); os.makedirs(gdir, exist_ok=True)
    for title, cols in headline:
        rows, rlabels = [], []
        for cat, p in cat_prompt.items():
            d = os.path.join(OUT, p["id"])
            row = []
            for name, _lab in cols:
                fp = os.path.join(d, f"{name}_s0.png")
                row.append(Image.open(fp).convert("RGB") if os.path.exists(fp)
                           else Image.new("RGB", (256, 256), "gray"))
            rows.append(row); rlabels.append(f"{cat}:{p['id']}")
        fn = os.path.join(gdir, f"{title}.png")
        save_grid(rows, rlabels, [l for _, l in cols], fn, thumb=160)
        out.append(("grids/" + os.path.basename(fn), title))
    return out


def _site(summary, plots, grids_png):
    try:
        from common import data_uri
    except Exception:
        data_uri = None

    def img(rel):
        p = os.path.join(OUT, rel)
        if data_uri and os.path.exists(p):
            return f"<img src='{data_uri(p)}' style='max-width:100%'>"
        return f"<img src='{rel}' style='max-width:100%'>"

    h = ["<!doctype html><meta charset=utf-8><title>E35 operator sweep (SD1.5)</title>",
         "<style>body{font:14px/1.5 system-ui;max-width:1200px;margin:2rem auto;padding:0 1rem;color:#222}"
         "table{border-collapse:collapse;margin:.5em 0}td,th{border:1px solid #bbb;padding:3px 7px;font-size:12px}"
         "img{border:1px solid #ddd}h2{margin-top:1.6em}</style>",
         "<h1>E35 — token-frequency operator sweep (SD1.5)</h1>",
         "<p>For each operator × parameter × prompt-category: CLIP adherence, LAION aesthetic "
         "(fidelity), and baseline-drift (how far the edit moved the image). 5 seeds/condition.</p>",
         "<h2>Per-operator sweeps</h2>"]
    for rel, ttl in plots:
        h.append(f"<p><b>{ttl}</b><br>{img(rel)}</p>")
    h.append("<h2>Contact sheets (seed 0)</h2>")
    for rel, ttl in grids_png:
        h.append(f"<p><b>{ttl}</b><br>{img(rel)}</p>")
    # overall table
    h.append("<h2>Overall means by operator × category</h2>")
    ov = summary["overall"]
    h.append("<table><tr><th>operator</th><th>category</th><th>CLIP</th><th>aesthetic</th>"
             "<th>drift</th><th>sharpness</th><th>colorful</th><th>n</th></tr>")
    for op in sorted(ov):
        for cat in sorted(ov[op]):
            e = ov[op][cat]
            h.append("<tr>" + "".join(f"<td>{x}</td>" for x in [
                op, cat, e.get("clip", "—"), e.get("aesthetic", "—"), e.get("drift", "—"),
                e.get("sharpness", "—"), e.get("colorfulness", "—"), e.get("_n", "—")]) + "</tr>")
    h.append("</table>")
    with open(os.path.join(OUT, "index.html"), "w") as f:
        f.write("\n".join(h))


# ---------------------------------------------------------------------------
def main(args):
    global OUT
    if args.out_tag:
        OUT = os.path.join(RESULTS, f"e35_{args.out_tag}")
    os.makedirs(OUT, exist_ok=True)
    parts = [p.strip() for p in args.part.split(",") if p.strip()]
    if "preflight" in parts:
        run_preflight(args)
    if "gen" in parts:
        run_gen(args)
    if "analyze" in parts:
        run_analyze(args)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="preflight,gen,analyze")
    ap.add_argument("--coverage", default="thorough", choices=["quick", "thorough", "max"])
    ap.add_argument("--num_prompts", type=int, default=99, help="cap on prompts/pairs (smoke)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=GUIDANCE)
    ap.add_argument("--clip_model", default="openai/clip-vit-large-patch14")
    ap.add_argument("--batch_eta_s", type=float, default=8.0, help="sec/batch for ETA estimate")
    ap.add_argument("--out_tag", default="")
    a = ap.parse_args()
    GUIDANCE = a.guidance
    main(a)
