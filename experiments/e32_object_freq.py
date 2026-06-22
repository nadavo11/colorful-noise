"""E32: PER-OBJECT token-frequency control on two-object prompts (follow-up to E24/E30).

E24 showed token-axis FFT bands of Flux's T5 sequence embedding are meaningful and
on-manifold; E30 made the control continuous but GLOBAL (it scales a band of the whole
prompt). E32 asks the next question: take a prompt with TWO objects and increase/decrease
the high or low token-frequencies of ONE object's own token span -- is the effect
SELECTIVE to that object, or does it just behave like E30's global gain?

The signal & op. A prompt's T5 sequence embedding E (1, L, 4096). FFT is global over the
token axis (frequency and token-position are conjugate), so a "per-object frequency" edit
can only be a WINDOWED FFT over the object's contiguous token span E[:, a:b]: scale the
band there, IFFT, stitch back (TS.apply_on_subspan + TS.band_gain_1d). Consequence: an
object phrase is short, so its windowed rfft has few bins -- "low vs high" inside a span
is coarse. Mitigation: every prompt uses multi-token object phrases, and `preflight`
prints bins-per-object so the coarseness is explicit.

Word -> token span. No such utility existed; we map an object phrase to its T5 token
indices via the fast tokenizer's char offset mapping (fallback: token-id subsequence),
computed while the tokenizer is still loaded (it gets dropped to free GPU like E24).

Conditions per prompt (median split CUT0=0.51 -- short windows need it; see below):
  baseline                                            (1)
  obj{A,B} x band{low,high} x gain{0.5 cut, 2.0 boost} (8, targeted)
  global   x band{low,high} x gain{0.5, 2.0}           (4, the E30-style null control)
= 13 conditions x 10 prompts x args.seeds.

Metric (the claim is selectivity): per-object CLIP -- objA_phrase vs objB_phrase -- and,
relative to baseline (paired by seed): Delta_target, Delta_other,
SELECTIVITY = Delta_target - Delta_other. A targeted edit should give selectivity > 0
while the global control's stays ~0. B-VQA per object phrase corroborates presence.

Parts (--part, comma list):
  preflight  -- tokenizer only (no GPU): print each object's span + bins-per-object,
                assert spans are valid / non-overlapping. Run this first.
  gen        -- pre-encode prompts, compute spans, generate the 13 conditions x seeds.
  analyze    -- per-object CLIP + B-VQA, selectivity table, strips, self-contained html.

Cluster: ship via kubectl cp (storage is not a git repo); the job self-gates a smoke
subset on CLIP, then runs the full set. Heavy B-VQA loads in analyze only, after Flux is
freed; VQAScore is scored offline (no_vqa here), as in E30.
"""
import argparse
import json
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, save_grid
from e7_flux_phase import REPO
from e10_cfg_spectral import gen_emb
from e9_clipt import agg, load_clip, clip_scores
from e9_bandnorm_classes import image_metrics
import text_spectral_ops as TS

OUT = os.path.join(RESULTS, "e32")
# Median split (NOT E24/E30's 0.25): an object window is only ~5-9 tokens, so its rfft
# has few bins and the lowest non-DC normalised frequency is already ~0.33. A 0.25 cut
# would leave the "low" band EMPTY -> a silent no-op. 0.51 splits the available bins in
# half AND keeps both bands non-empty even for 3-bin (5-token) windows: the single
# mid-bin at 0.5 falls in "low", Nyquist (1.0) in "high". DC is never scaled (keep_dc).
CUT0 = 0.51
GAINS = [0.5, 2.0]                # cut (attenuate) / boost (amplify)
BANDS = {"low": (0.0, CUT0), "high": (CUT0, 1.0)}
IMG_KEYS = ["sharpness", "hf_frac", "colorfulness"]

# 10 two-object prompts. Each object phrase is multi-token (so its windowed span has >1
# frequency bin) and appears VERBATIM in the prompt (so offset mapping can locate it).
# The phrases double as the per-object CLIP / B-VQA targets, so they are kept distinct.
PROMPTS_2OBJ = [
    ("cat_dog",        "a fluffy orange tabby cat and a sleeping golden retriever dog",
                       "a fluffy orange tabby cat", "a sleeping golden retriever dog"),
    ("car_bike",       "a shiny red sports car and a rusty old bicycle",
                       "a shiny red sports car", "a rusty old bicycle"),
    ("castle_forest",  "a tall stone medieval castle and a dense green pine forest",
                       "a tall stone medieval castle", "a dense green pine forest"),
    ("teapot_cup",     "an ornate silver teapot and a small blue porcelain teacup",
                       "an ornate silver teapot", "a small blue porcelain teacup"),
    ("owl_fox",        "a white snowy owl and a small red fox",
                       "a white snowy owl", "a small red fox"),
    ("guitar_piano",   "a wooden acoustic guitar and a black grand piano",
                       "a wooden acoustic guitar", "a black grand piano"),
    ("lighthouse_boat","a striped red lighthouse and a small wooden fishing boat",
                       "a striped red lighthouse", "a small wooden fishing boat"),
    ("cactus_rose",    "a tall green cactus and a single red rose",
                       "a tall green cactus", "a single red rose"),
    ("robot_teddy",    "a shiny metal robot and a soft brown teddy bear",
                       "a shiny metal robot", "a soft brown teddy bear"),
    ("mountain_lake",  "a snowy mountain peak and a calm blue lake",
                       "a snowy mountain peak", "a calm blue lake"),
]


# ---------------------------------------------------------------------------
# word -> token span
# ---------------------------------------------------------------------------

def phrase_span(tokenizer, prompt, phrase, L):
    """Map an object `phrase` to its (a, b) token indices in `prompt`'s T5 token
    sequence (0 <= a < b <= L). Prefers the fast tokenizer's char offset mapping;
    falls back to matching the phrase's own token-id subsequence. Raises if neither
    locates the phrase (caught early by `preflight`)."""
    # 1) char offset mapping: take every token whose char span overlaps the phrase's.
    try:
        enc = tokenizer(prompt, max_length=512, truncation=True,
                        return_offsets_mapping=True)
        offs = enc["offset_mapping"]
        c0 = prompt.index(phrase)
        c1 = c0 + len(phrase)
        idx = [i for i, (s, e) in enumerate(offs)
               if i < L and e > s and s < c1 and e > c0]
        if idx:
            return min(idx), min(max(idx) + 1, L)
    except (ValueError, KeyError, TypeError):
        pass
    # 2) fallback: locate the phrase's token-id subsequence in the prompt's token ids.
    pid = tokenizer(prompt, max_length=512, truncation=True)["input_ids"]
    ph = tokenizer(phrase, add_special_tokens=False)["input_ids"]
    for i in range(len(pid) - len(ph) + 1):
        if pid[i:i + len(ph)] == ph:
            return i, min(i + len(ph), L)
    raise ValueError(f"could not locate phrase {phrase!r} in prompt {prompt!r}")


def _bins(span):
    """Number of rfft frequency bins for a token span of width (b-a)."""
    w = span[1] - span[0]
    return w // 2 + 1


# ---------------------------------------------------------------------------
# Part: preflight (tokenizer only, no GPU) -- validate spans before any compute
# ---------------------------------------------------------------------------

def run_preflight(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(REPO, subfolder="tokenizer_2")
    print(f"[e32] preflight: tokenizer={type(tok).__name__} fast={tok.is_fast}", flush=True)
    spans = {}
    n_warn = 0
    for key, prompt, oa, ob in PROMPTS_2OBJ[: args.num_prompts]:
        L = len(tok(prompt, max_length=512, truncation=True)["input_ids"])
        sA = phrase_span(tok, prompt, oa, L)
        sB = phrase_span(tok, prompt, ob, L)
        spans[key] = {"L": L, "objA": list(sA), "objB": list(sB),
                      "binsA": _bins(sA), "binsB": _bins(sB)}
        # assertions: valid, within prompt, non-overlapping
        for s in (sA, sB):
            assert 0 <= s[0] < s[1] <= L, f"{key}: bad span {s} (L={L})"
        assert sA[1] <= sB[0] or sB[1] <= sA[0], f"{key}: spans overlap {sA} {sB}"
        warn = ""
        if min(_bins(sA), _bins(sB)) < 2:
            warn = "  <-- WARN: <2 bins, low/high split is degenerate"
            n_warn += 1
        print(f"[e32] {key:14s} L={L:3d}  objA={sA} ({_bins(sA)} bins)  "
              f"objB={sB} ({_bins(sB)} bins){warn}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "spans.json"), "w") as f:
        json.dump(spans, f, indent=2)
    print(f"[e32] preflight OK: {len(spans)} prompts, {n_warn} bin-warnings; "
          f"wrote {os.path.relpath(os.path.join(OUT, 'spans.json'), OUT)}", flush=True)
    return spans


# ---------------------------------------------------------------------------
# loader: pre-encode + capture object token spans (before tokenizers are dropped)
# ---------------------------------------------------------------------------

def load_flux_preencoded_spans(prompt_phrases):
    """Like e24.load_flux_preencoded_lens but also returns spans[prompt] = [(a,b),...]
    (one token span per object phrase, in the order given). Spans are computed while
    tokenizer_2 is alive; encoders are then dropped to free GPU."""
    import gc
    from diffusers import (FluxPipeline, FluxTransformer2DModel, BitsAndBytesConfig)
    qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.bfloat16)
    tr = FluxTransformer2DModel.from_pretrained(
        REPO, subfolder="transformer", quantization_config=qc,
        torch_dtype=torch.bfloat16)
    pipe = FluxPipeline.from_pretrained(REPO, transformer=tr, torch_dtype=torch.bfloat16)
    pipe.set_progress_bar_config(disable=True)
    pipe.text_encoder.to("cuda")
    pipe.text_encoder_2.to("cuda")
    embeds, lens, spans = {}, {}, {}
    with torch.no_grad():
        for txt in dict.fromkeys(prompt_phrases):
            pe, ppe, _ = pipe.encode_prompt(
                prompt=txt, prompt_2=txt, device="cuda",
                num_images_per_prompt=1, max_sequence_length=512)
            embeds[txt] = (pe.cpu(), ppe.cpu())
            tok = pipe.tokenizer_2(txt, max_length=512, truncation=True,
                                   return_tensors="pt")
            L = int(tok.attention_mask.sum())
            lens[txt] = L
            spans[txt] = [phrase_span(pipe.tokenizer_2, txt, ph, L)
                          for ph in prompt_phrases[txt]]
    pipe.text_encoder = pipe.text_encoder_2 = None
    pipe.tokenizer = pipe.tokenizer_2 = None
    gc.collect()
    torch.cuda.empty_cache()
    pipe.vae.to("cuda")
    print(f"[e32] pre-encoded {len(embeds)} prompts; text encoders dropped", flush=True)
    return pipe, embeds, lens, spans


def _gen(pipe, pe, ppe, seed, args, png):
    """Cached single generation from (modified) embeddings -> PIL (also returned)."""
    if os.path.exists(png):
        return Image.open(png).convert("RGB")
    img, _ = gen_emb(pipe, (pe.cpu(), ppe.cpu()), None, seed, 1.0,
                     args.guidance, args.steps)
    os.makedirs(os.path.dirname(png), exist_ok=True)
    img.save(png)
    print(f"[e32] gen {os.path.relpath(png, OUT)}", flush=True)
    return img


# ---------------------------------------------------------------------------
# condition enumeration (shared by gen + analyze so they never drift)
# ---------------------------------------------------------------------------

def targeted_conditions():
    """(name, target in {A,B}, band, gain) for the 8 per-object edits."""
    for tgt in ("A", "B"):
        for band in BANDS:
            for g in GAINS:
                yield f"obj{tgt}_{band}_g{g}", tgt, band, g


def global_conditions():
    """(name, band, gain) for the 4 whole-prompt (E30-style) control edits."""
    for band in BANDS:
        for g in GAINS:
            yield f"global_{band}_g{g}", band, g


def _build_conditions(pe, L, spanA, spanB):
    """{cond_name: modified prompt_embeds} for one prompt (pooled stays baseline)."""
    conds = {"baseline": pe}
    for name, tgt, band, g in targeted_conditions():
        a, b = spanA if tgt == "A" else spanB
        lo, hi = BANDS[band]
        conds[name] = TS.apply_on_subspan(
            lambda x, lo=lo, hi=hi, g=g: TS.band_gain_1d(x, lo, hi, g), pe, a, b)
    for name, band, g in global_conditions():
        lo, hi = BANDS[band]
        conds[name] = TS.apply_on_span(
            lambda x, lo=lo, hi=hi, g=g: TS.band_gain_1d(x, lo, hi, g), pe, L)
    return conds


# ---------------------------------------------------------------------------
# Part: gen
# ---------------------------------------------------------------------------

def run_gen(args):
    items = PROMPTS_2OBJ[: args.num_prompts]
    prompt_phrases = {p: [oa, ob] for _, p, oa, ob in items}
    pipe, emb, lens, spans = load_flux_preencoded_spans(prompt_phrases)

    # persist spans (analyze runs without the tokenizer) + bins for the writeup
    span_meta = {}
    for key, prompt, _oa, _ob in items:
        (a0, b0), (a1, b1) = spans[prompt]
        span_meta[key] = {"L": lens[prompt], "objA": [a0, b0], "objB": [a1, b1],
                          "binsA": _bins((a0, b0)), "binsB": _bins((a1, b1))}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "spans.json"), "w") as f:
        json.dump(span_meta, f, indent=2)

    for key, prompt, _oa, _ob in items:
        d = os.path.join(OUT, key)
        pe, ppe = emb[prompt]
        conds = _build_conditions(pe, lens[prompt], *spans[prompt])
        for name, mpe in conds.items():
            for s in range(args.seeds):
                _gen(pipe, mpe, ppe, s, args, os.path.join(d, f"{name}_s{s}.png"))
        _save_strip(d, key)
        print(f"[e32] gen {key} done", flush=True)


def _save_strip(d, key):
    """Visual strip (seed 0): rows = objA / objB / global, cols = baseline + the four
    band x gain edits. baseline is repeated each row as a fixed reference."""
    cols = [("baseline", "baseline")]
    for band in BANDS:
        for g in GAINS:
            cols.append((band, g))
    col_labels = ["baseline"] + [f"{b} g{g}" for b in BANDS for g in GAINS]
    rows, row_labels = [], []
    for scope in ("objA", "objB", "global"):
        row = []
        for band, g in cols:
            name = "baseline" if band == "baseline" else (
                f"{scope}_{band}_g{g}" if scope != "global" else f"global_{band}_g{g}")
            p = os.path.join(d, f"{name}_s0.png")
            row.append(Image.open(p).convert("RGB") if os.path.exists(p)
                       else Image.new("RGB", (256, 256), "gray"))
        rows.append(row)
        row_labels.append(scope)
    save_grid(rows, row_labels, col_labels, os.path.join(d, "strip.png"), thumb=200)


# ---------------------------------------------------------------------------
# Part: analyze -- per-object CLIP / B-VQA, selectivity
# ---------------------------------------------------------------------------

def load_bvqa_safe():
    try:
        from compbench import load_bvqa
        return load_bvqa()
    except Exception as e:
        print(f"[e32] B-VQA unavailable: {e}", flush=True)
        return None


def bvqa_one(bvqa, phrase, img):
    if bvqa is None:
        return None
    from compbench import bvqa_scores
    return bvqa_scores(bvqa, phrase, [img])[0]


def run_analyze(args):
    clip = load_clip(args.clip_model)
    bvqa = load_bvqa_safe()
    spans = {}
    sp = os.path.join(OUT, "spans.json")
    if os.path.exists(sp):
        spans = json.load(open(sp))

    # 1) raw per (prompt, condition, seed): CLIP_A/B, presence_A/B, image stats
    raw = {}
    for key, prompt, oa, ob in PROMPTS_2OBJ[: args.num_prompts]:
        d = os.path.join(OUT, key)
        if not os.path.isdir(d):
            continue
        per_cond = {}
        for name in ["baseline"] + [n for n, *_ in targeted_conditions()] \
                + [n for n, *_ in global_conditions()]:
            seeds = {}
            for s in range(args.seeds):
                p = os.path.join(d, f"{name}_s{s}.png")
                if not os.path.exists(p):
                    continue
                im = Image.open(p).convert("RGB")
                ent = {"clipA": agg(clip_scores(*clip, oa, [im]))["mean"],
                       "clipB": agg(clip_scores(*clip, ob, [im]))["mean"]}
                if bvqa is not None:
                    ent["presA"] = bvqa_one(bvqa, oa, im)
                    ent["presB"] = bvqa_one(bvqa, ob, im)
                m = image_metrics(im)
                ent.update({k: m[k] for k in IMG_KEYS})
                seeds[str(s)] = ent
            per_cond[name] = seeds
        raw[key] = {"prompt": prompt, "objA": oa, "objB": ob,
                    "spans": spans.get(key, {}), "conditions": per_cond}

    # 2) summary: deltas vs baseline (paired per seed), pooled over prompts+seeds
    summary = _summarize(raw, args)

    report = {"params": vars(args), "raw": raw, "summary": summary}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    _site(report)
    print("[e32] wrote report.json + index.html", flush=True)
    _print_verdict(summary)


def _summarize(raw, args):
    """Pool, per (band, gain): targeted Delta_target / Delta_other / selectivity, and
    global Delta_A / Delta_B / (A-B). Deltas are paired to the same-seed baseline.
    Computed for CLIP ('clip') and B-VQA presence ('pres')."""
    metrics = {"clip": ("clipA", "clipB"), "pres": ("presA", "presB")}
    acc = {}  # acc[scope][band][gain][metric][quantity] -> list

    def push(scope, band, g, metric, q, v):
        if v is None:
            return
        acc.setdefault(scope, {}).setdefault(band, {}).setdefault(str(g), {}) \
           .setdefault(metric, {}).setdefault(q, []).append(v)

    for key, e in raw.items():
        conds = e["conditions"]
        base = conds.get("baseline", {})
        for mkey, (ka, kb) in metrics.items():
            # targeted: relabel target/other so A- and B-targeted edits pool together
            for name, tgt, band, g in targeted_conditions():
                for s, ent in conds.get(name, {}).items():
                    b = base.get(s)
                    if b is None or ka not in ent or ka not in b:
                        continue
                    if ent.get(ka) is None or b.get(ka) is None:
                        continue
                    dA = ent[ka] - b[ka]
                    dB = ent[kb] - b[kb]
                    d_tgt, d_oth = (dA, dB) if tgt == "A" else (dB, dA)
                    push("targeted", band, g, mkey, "d_target", d_tgt)
                    push("targeted", band, g, mkey, "d_other", d_oth)
                    push("targeted", band, g, mkey, "selectivity", d_tgt - d_oth)
            # global control: no target -> record d_A, d_B and their difference
            for name, band, g in global_conditions():
                for s, ent in conds.get(name, {}).items():
                    b = base.get(s)
                    if b is None or ent.get(ka) is None or b.get(ka) is None:
                        continue
                    dA = ent[ka] - b[ka]
                    dB = ent[kb] - b[kb]
                    push("global", band, g, mkey, "d_A", dA)
                    push("global", band, g, mkey, "d_B", dB)
                    push("global", band, g, mkey, "selectivity", dA - dB)

    # reduce lists -> agg dicts
    def reduce(node):
        if isinstance(node, dict):
            return {k: reduce(v) for k, v in node.items()}
        return agg(node)
    return reduce(acc)


def _print_verdict(summary):
    print("[e32] --- selectivity (CLIP), targeted vs global control ---", flush=True)
    for band in BANDS:
        for g in GAINS:
            t = _dig(summary, "targeted", band, str(g), "clip", "selectivity")
            gl = _dig(summary, "global", band, str(g), "clip", "selectivity")
            print(f"[e32]  {band:4s} g{g}: targeted sel="
                  f"{_fmt(t)}  global sel={_fmt(gl)}", flush=True)


def _dig(d, *ks):
    for k in ks:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _fmt(a):
    return f"{a['mean']:+.4f}+/-{a['std']:.4f}(n{a['n']})" if a else "NA"


# ---------------------------------------------------------------------------
# self-contained HTML explainer
# ---------------------------------------------------------------------------

_SCHEMATIC_SVG = '''
<figure style="margin:1.2em 0">
<svg viewBox="0 0 760 200" xmlns="http://www.w3.org/2000/svg" role="img"
     style="width:100%;max-width:760px;border:1px solid #ddd;border-radius:6px;background:#fff">
  <defs><marker id="ar" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto">
    <path d="M0,0 L7,3 L0,6 Z" fill="#555"/></marker></defs>
  <style>.b{font:12px system-ui;fill:#1b1b1b}.s{font:10px system-ui;fill:#666}
    .t{font:bold 12px system-ui;fill:#111}.box{fill:#f3f6fb;stroke:#9fb6d6}
    line{stroke:#555;stroke-width:1.4;marker-end:url(#ar)}</style>
  <text x="14" y="20" class="t">Per-object token-frequency edit: window the FFT on ONE object's tokens</text>
  <rect class="box" x="14" y="40" width="150" height="40"/>
  <text x="22" y="58" class="b">"[objA] and [objB]"</text><text x="22" y="73" class="s">-> T5 seq embedding</text>
  <line x1="164" y1="60" x2="186" y2="60"/>
  <rect x="190" y="40" width="200" height="40" fill="#eef" stroke="#9fb6d6"/>
  <rect x="190" y="40" width="86" height="40" fill="#8fc0ff" stroke="#5b8bd0"/>
  <text x="198" y="64" class="b">objA tokens</text><text x="300" y="64" class="b">objB tokens</text>
  <text x="190" y="96" class="s">windowed rfft over objA's span only -> scale band -> irfft -> stitch back</text>
  <line x1="390" y1="60" x2="412" y2="60"/>
  <rect class="box" x="416" y="40" width="60" height="40"/><text x="424" y="64" class="b">Flux</text>
  <line x1="476" y1="60" x2="498" y2="60"/>
  <rect x="502" y="38" width="44" height="44" fill="#eee" stroke="#999"/>
  <text x="14" y="140" class="t">Is the effect selective?</text>
  <text x="14" y="158" class="s">targeted: selectivity = Delta(CLIP target object) - Delta(CLIP other object), vs baseline</text>
  <text x="14" y="174" class="s">global control = same band gain on the WHOLE prompt (E30). targeted should beat its ~0 selectivity.</text>
</svg>
<figcaption style="font:11px system-ui;color:#777">Schematic — E32 per-object token-frequency control.</figcaption>
</figure>
'''


def _site(report):
    try:
        from common import data_uri
    except Exception:
        data_uri = None

    def emb_img(rel):
        p = os.path.join(OUT, rel)
        if data_uri and os.path.exists(p):
            return f"<img src='{data_uri(p)}' style='max-width:100%'>"
        return f"<img src='{rel}' style='max-width:100%'>"

    h = ["<!doctype html><meta charset=utf-8><title>E32 per-object text-frequency</title>",
         "<style>body{font:14px/1.5 system-ui;max-width:1100px;margin:2rem auto;"
         "padding:0 1rem;color:#222}code{background:#f0f0f0;padding:1px 4px;border-radius:3px}"
         "h1,h2{line-height:1.2}img{border:1px solid #ddd;margin:.4em 0}"
         "table{border-collapse:collapse;margin:.6em 0}td,th{border:1px solid #bbb;padding:3px 7px}"
         "</style>",
         "<h1>E32 — Per-object token-frequency control on two-object prompts</h1>",
         "<p><b>Follow-up to E24/E30.</b> E30 scales a token-frequency band of the "
         "<i>whole</i> prompt. E32 windows the FFT onto <b>one object's token span</b> and "
         "asks whether the high/low band edit is <b>selective</b> to that object, against a "
         "global-gain control. FFT is global over the token axis, so the only coherent "
         "per-object op is a <b>windowed FFT over the object's contiguous span</b> — short "
         "spans mean few bins, so we use multi-token object phrases (see spans below).</p>",
         _SCHEMATIC_SVG]

    # selectivity tables (the headline)
    summ = report.get("summary", {})
    for mkey, mlabel in (("clip", "CLIP"), ("pres", "B-VQA presence")):
        h.append(f"<h2>Selectivity — {mlabel}</h2>")
        h.append("<table><tr><th>band × gain</th><th>targeted Δtarget</th>"
                 "<th>targeted Δother</th><th>targeted <b>selectivity</b></th>"
                 "<th>global Δ(objA)</th><th>global Δ(objB)</th><th>global selectivity</th></tr>")
        for band in BANDS:
            for g in GAINS:
                cells = [f"{band} g{g}",
                         _fmt(_dig(summ, "targeted", band, str(g), mkey, "d_target")),
                         _fmt(_dig(summ, "targeted", band, str(g), mkey, "d_other")),
                         _fmt(_dig(summ, "targeted", band, str(g), mkey, "selectivity")),
                         _fmt(_dig(summ, "global", band, str(g), mkey, "d_A")),
                         _fmt(_dig(summ, "global", band, str(g), mkey, "d_B")),
                         _fmt(_dig(summ, "global", band, str(g), mkey, "selectivity"))]
                h.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
        h.append("</table>")

    # per-prompt strips + spans
    h.append("<h2>Per-prompt strips (seed 0)</h2>")
    h.append("<p>Rows: objA-targeted / objB-targeted / global. Columns: baseline + "
             "band×gain edits.</p>")
    for key in report.get("raw", {}):
        e = report["raw"][key]
        sp = e.get("spans", {})
        cap = (f"objA={e['objA']!r} span {sp.get('objA')} ({sp.get('binsA')} bins) · "
               f"objB={e['objB']!r} span {sp.get('objB')} ({sp.get('binsB')} bins)")
        rel = os.path.join(key, "strip.png")
        h.append(f"<h3>{key}: <code>{e['prompt']}</code></h3>"
                 f"<p style='font:11px system-ui;color:#666'>{cap}</p>{emb_img(rel)}")

    with open(os.path.join(OUT, "index.html"), "w") as f:
        f.write("\n".join(h))


# ---------------------------------------------------------------------------

def main(args):
    global OUT
    if args.out_tag:
        OUT = os.path.join(RESULTS, f"e32_{args.out_tag}")
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
    ap.add_argument("--num_prompts", type=int, default=10, help="cap on two-object prompts")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=3.5)
    ap.add_argument("--clip_model", default="openai/clip-vit-large-patch14")
    ap.add_argument("--no_vqa", action="store_true",
                    help="(accepted for cluster-job parity; E32 scores VQAScore offline)")
    ap.add_argument("--out_tag", default="")
    main(ap.parse_args())
