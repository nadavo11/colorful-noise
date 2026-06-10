"""Build the Spectral Band Normalization (SBN) explainer page.

A standalone, self-contained interactive page (vanilla JS, no build, no CDNs,
works over file://). Reads the analysis JSONs and inlines them; images/plots are
referenced by relative path so re-running after more renders land just refreshes
them. The rendered page contains no internal experiment naming.

    python make_e9_site.py
"""
import argparse
import base64
import glob
import json
import os
import sys
from io import BytesIO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS
from e9_bandnorm_classes import CLASSES, METRICS

# Saturation is the recommended practical (cheap, single-pass) color correction;
# pick the factor whose colorfulness lands closest to full guidance without
# over-saturating. Chosen empirically from results/e11/report.json (sat 1.4 vs
# 1.8): 1.4 has the smaller mean gap to the cfg=3.5 colorfulness and overshoots
# far less, so it reads natural rather than garish.
CC_BEST = "sat1.4"

OUT = os.path.join(RESULTS, "e9")   # SBN data source (report/clip/universal/...)
SITE = os.path.join(RESULTS, "site")  # multi-experiment site (E9-E12), not e9-local

# Neutral, externally-framed per-class context.
CLASS_META = {
    "animal":      {"label": "Animal",      "domain": "photo (high-power)",       "refstd": 0.89, "blurb": "Fur and misty foliage — fine detail spread broadly across the frame. SBN nudges texture UP."},
    "portrait":    {"label": "Portrait",    "domain": "photo (high-power)",       "refstd": 0.82, "blurb": "Weathered skin and beard — broadly distributed micro-detail. SBN nudges texture UP."},
    "landscape":   {"label": "Landscape",   "domain": "low-power photo",          "refstd": 0.72, "blurb": "Rock and water texture is broad but the scene is naturally soft. Small positive texture nudge."},
    "urban_night": {"label": "Urban night", "domain": "photo (high-power)",       "refstd": 0.90, "blurb": "Neon highlights on dark — detail concentrated in a few high-power structures. SBN REMOVES texture here."},
    "abstract":    {"label": "Abstract",    "domain": "illustration (low-power)", "refstd": 0.65, "blurb": "Bold painterly strokes — concentrated high-power detail. SBN REMOVES texture here."},
    "watercolor":  {"label": "Watercolor",  "domain": "illustration (low-power)", "refstd": 0.67, "blurb": "Soft washes — little fine detail to begin with. Texture nudge is marginal."},
}

# Display labels for the generation conditions (no internal names).
CONDS = [
    ("cfg1.0",   "Weak guidance",  "cfg 1.0", "Low guidance — soft, calm contrast and palette. The reference look."),
    ("cfg3.5",   "Full guidance",  "cfg 3.5", "Standard guidance — crisp, punchy, saturated. The composition we keep."),
    ("bandnorm", "SBN",            "band-normalized", "Full-guidance trajectory with per-band power clamped to the weak-guidance reference every step."),
]


def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def strip_internal(obj):
    """Drop internal-only keys (e.g. the legacy 'xfer' condition) so even the
    inlined JSON source carries no experiment naming."""
    drop = lambda k: ("xfer" in k) or ("cat" in k)
    if isinstance(obj, dict):
        return {k: strip_internal(v) for k, v in obj.items() if not drop(k)}
    if isinstance(obj, list):
        return [strip_internal(v) for v in obj]
    return obj


def _data_uri(abspath, max_px, quality):
    """Downscale to max_px on the long side and re-encode as base64 JPEG."""
    from PIL import Image
    im = Image.open(abspath).convert("RGB")
    if max(im.size) > max_px:
        r = max_px / max(im.size)
        im = im.resize((round(im.width * r), round(im.height * r)))
    buf = BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def build_img_map(data, max_px=512, quality=72):
    """Embed every image the page can reference, keyed by results-relative path
    (matching the JS S(...) keys). Walks only the referenced subtrees and filters
    out conditions/dirs the page never shows (xfer/cat conds, e11 grids)."""
    img, root = {}, RESULTS

    def add(abspath):
        rel = os.path.relpath(abspath, root).replace(os.sep, "/")
        if rel not in img:
            img[rel] = _data_uri(abspath, max_px, quality)

    # e9 plots (the four referenced) + e10/e9 plot dir
    for f in ("cfg_power", "cfg_psd", "ref_std_curves", "metrics_delta"):
        p = os.path.join(root, "e9", "plots", f"{f}.png")
        if os.path.exists(p):
            add(p)
    # e9 per-class condition images shown in Compare / Universal
    keep = ("cfg1.0_", "cfg3.5_", "bandnorm_", "uninorm_")
    for c in data["classes"]:
        for p in glob.glob(os.path.join(root, "e9", c["key"], "images", "*.png")):
            if os.path.basename(p).startswith(keep):
                add(p)
    # e9 frequency-control sweep images
    for p in glob.glob(os.path.join(root, "e9", "freqctrl", "*", "images", "*.png")):
        add(p)
    # e11 corrected frames (every method dir, skip the grids contact sheets)
    cc = data.get("colorcorr") or {}
    for k in cc:
        for p in glob.glob(os.path.join(root, "e11", k, "*", "*.png")):
            if os.sep + "grids" + os.sep not in p:
                add(p)
    # phase thread (E12-E15) plots + representative grids
    for f in ("e12/plots/coherence_radial.png", "e12/plots/global_hist.png",
              "e13/plots/identity_phase_vs_mag.png", "e13/grid_animal.png",
              "e13/grid_abstract.png",
              "e14/plots/identity_vs_eps.png", "e14/grid_landscape_noise.png",
              "e15/plots/proj_by_manipulation.png", "e15/plots/proj_by_class.png"):
        p = os.path.join(root, f)
        if os.path.exists(p):
            add(p)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--standalone", action="store_true",
                    help="also write index_standalone.html with images inlined")
    ap.add_argument("--max-px", type=int, default=512)
    ap.add_argument("--quality", type=int, default=72)
    args = ap.parse_args()

    os.makedirs(SITE, exist_ok=True)
    report = strip_internal(load_json(os.path.join(OUT, "report.json")) or {})
    data = {
        "report": report,
        "cfgspec": load_json(os.path.join(RESULTS, "e10", "cfg_spectral.json")),
        "clipt": strip_internal(load_json(os.path.join(OUT, "clip_t.json"))),
        "universal": load_json(os.path.join(OUT, "universal.json")),
        "freqctrl": load_json(os.path.join(OUT, "freqctrl.json")),
        "cost": load_json(os.path.join(OUT, "cost.json")),
        # E11: cheap image-level color correction of SBN frames toward the
        # full-guidance palette (keyed by bare class name, no internal naming).
        "colorcorr": load_json(os.path.join(RESULTS, "e11", "report.json")),
        # Phase thread (E12-E15): the complement of the band-norm power story.
        "phaseDist":  load_json(os.path.join(RESULTS, "e12", "report.json")),
        "phaseSwap":  load_json(os.path.join(RESULTS, "e13", "report.json")),
        "phaseFn":    load_json(os.path.join(RESULTS, "e14", "report.json")),
        "phaseClust": load_json(os.path.join(RESULTS, "e15", "report.json")),
        "classes": [{"key": k, "prompt": p, **CLASS_META.get(k, {"label": k})}
                    for k, p in CLASSES],
        "conds": [{"key": k, "label": l, "tag": t, "desc": d}
                  for k, l, t, d in CONDS],
        "metrics": METRICS,
        "pairs": report.get("params", {}).get("pairs", 25),
    }
    base = (TEMPLATE.replace("/*__DATA__*/", json.dumps(data))
                    .replace("/*__CC_BEST__*/", CC_BEST))
    # Lightweight build: IMG empty, S() falls back to relative ../eN paths.
    out_path = os.path.join(SITE, "index.html")
    with open(out_path, "w") as f:
        f.write(base)
    flags = [n for n in ("cfgspec", "clipt", "universal", "freqctrl", "cost",
                         "colorcorr") if data[n]]
    print(f"[site] wrote {out_path}  (data: {', '.join(flags) or 'base only'})",
          flush=True)

    if args.standalone:
        img = build_img_map(data, args.max_px, args.quality)
        # Substitute the populated map for the empty placeholder. json.dumps the
        # map alone and drop the trailing "{}" the placeholder carried.
        html = base.replace("/*__IMG__*/{}", "/*__IMG__*/" + json.dumps(img))
        sa_path = os.path.join(SITE, "index_standalone.html")
        with open(sa_path, "w") as f:
            f.write(html)
        mb = os.path.getsize(sa_path) / 1e6
        print(f"[site] wrote {sa_path}  ({len(img)} images inlined, {mb:.1f} MB)",
              flush=True)


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Spectral Band Normalization</title>
<style>
  :root{
    --bg:#0e1116; --panel:#161b22; --panel2:#1c2330; --ink:#e6edf3;
    --muted:#9aa7b4; --line:#2a3340; --accent:#58a6ff; --good:#3fb950;
    --bad:#f85149; --warn:#d29922; --lo:#d29922; --hi:#a371f7;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
    font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
  header{position:sticky;top:0;z-index:10;background:rgba(14,17,22,.92);
    backdrop-filter:blur(6px);border-bottom:1px solid var(--line);padding:10px 22px}
  header h1{margin:0;font-size:16px;font-weight:600;letter-spacing:.2px}
  header .sub{color:var(--muted);font-size:12.5px;margin-top:2px}
  nav{display:flex;gap:3px;flex-wrap:wrap;margin-top:10px}
  nav button{background:transparent;color:var(--muted);border:1px solid transparent;
    padding:6px 12px;border-radius:7px;cursor:pointer;font-size:13px;font-weight:500}
  nav button:hover{color:var(--ink);background:var(--panel)}
  nav button.active{color:#fff;background:var(--accent);border-color:var(--accent)}
  main{max-width:1080px;margin:0 auto;padding:26px 22px 90px}
  section{display:none;animation:fade .25s ease}
  section.active{display:block}
  @keyframes fade{from{opacity:0;transform:translateY(4px)}to{opacity:1}}
  h2{font-size:23px;margin:.2em 0 .5em;font-weight:650}
  h3{font-size:16px;margin:1.5em 0 .5em;color:var(--accent)}
  p,li{color:#d3dce6}
  .lede{font-size:16.5px;color:var(--ink)}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:11px;padding:18px 20px;margin:14px 0}
  .thesis{border-left:3px solid var(--accent);background:var(--panel2)}
  .grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
  .grid3{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px}
  .pill{display:inline-block;background:var(--panel2);border:1px solid var(--line);border-radius:999px;padding:2px 10px;font-size:12px;color:var(--muted);margin:2px 4px 2px 0}
  code,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
  code{background:var(--panel2);padding:1px 6px;border-radius:5px;color:#a9d1ff;font-size:13px}
  .eq{background:#0b0e13;border:1px solid var(--line);border-radius:9px;padding:13px 16px;text-align:center;font-family:ui-monospace,monospace;font-size:14.5px;color:#cfe6ff;margin:12px 0}
  img.plot{max-width:100%;border-radius:9px;border:1px solid var(--line);background:#fff}
  table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}
  th,td{border:1px solid var(--line);padding:6px 9px;text-align:right}
  th:first-child,td:first-child{text-align:left}
  th{background:var(--panel2);color:var(--muted);font-weight:600}
  td.pos{color:var(--good)} td.neg{color:var(--bad)}
  .controls{display:flex;gap:18px;align-items:center;flex-wrap:wrap;margin:8px 0 16px}
  .controls label{font-size:13px;color:var(--muted)}
  select{background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:7px;padding:6px 10px;font-size:14px}
  input[type=range]{accent-color:var(--accent)}
  .seedval{font-family:ui-monospace,monospace;color:var(--accent);min-width:2.4em;display:inline-block}
  .triptych{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px}
  .tile{background:var(--panel);border:1px solid var(--line);border-radius:11px;overflow:hidden}
  .tile .cap{padding:9px 12px}
  .tile .cap b{font-size:14px}
  .tile .cap .d{color:var(--muted);font-size:12px;margin-top:2px}
  .tile .cap .clip{font-family:ui-monospace,monospace;font-size:12px;color:var(--accent);margin-top:4px}
  .tile .imgwrap{position:relative;aspect-ratio:1/1;background:#0b0e13;display:flex;align-items:center;justify-content:center}
  .tile img{width:100%;height:100%;object-fit:cover;display:block}
  .ph{color:var(--muted);font-size:12.5px;text-align:center;padding:20px}
  .badge{display:inline-block;font-size:11px;padding:1px 7px;border-radius:5px;font-weight:600}
  .b-up{background:rgba(63,185,80,.16);color:var(--good)}
  .b-down{background:rgba(248,81,73,.16);color:var(--bad)}
  .note{color:var(--warn);font-size:13px}
  .muted{color:var(--muted)} .small{font-size:13px}
  .bars{margin:10px 0}
  .barrow{display:grid;grid-template-columns:130px 1fr;gap:10px;align-items:center;margin:5px 0}
  .barrow .lbl{font-size:12.5px;color:var(--muted);text-align:right}
  .bartrack{background:var(--panel2);border-radius:6px;height:22px;position:relative;overflow:hidden}
  .barfill{height:100%;border-radius:6px;display:flex;align-items:center;justify-content:flex-end;padding-right:7px;font-size:11.5px;color:#06210d;font-weight:700}
  ul{padding-left:20px}
  hr{border:none;border-top:1px solid var(--line);margin:22px 0}
  .metric{border-left:3px solid var(--accent)}
  .kv{display:grid;grid-template-columns:auto 1fr;gap:4px 12px;font-size:13px;margin-top:6px}
  .kv .k{color:var(--muted)}
  .knob{font-family:ui-monospace,monospace;font-size:15px}
  .tag-lo{color:var(--lo)} .tag-hi{color:var(--hi)}
</style>
</head>
<body>
<header>
  <h1>Spectral Band Normalization <span class="muted" style="font-weight:400">(SBN)</span></h1>
  <div class="sub">Keep full-guidance composition, normalize the latent's spectral power to a calm reference — a frequency-domain control for diffusion images</div>
  <nav id="nav"></nav>
</header>
<main id="main"></main>

<script>
const DATA = /*__DATA__*/;
// Image map for the self-contained build: results-relative path -> data URI.
// Empty in the lightweight build, so S() falls back to the relative path.
const IMG = /*__IMG__*/{};
const S = p => IMG[p] || ("../" + p);
const {report, cfgspec, clipt, universal, freqctrl, cost, colorcorr, classes, conds, metrics, pairs} = DATA;
const {phaseDist, phaseSwap, phaseFn, phaseClust} = DATA;
const fmt = (x,d=4)=> (x==null||isNaN(x))?"—":(x>=0?"+":"")+Number(x).toFixed(d);
const fmt2 = (x,d=3)=> (x==null||isNaN(x))?"—":Number(x).toFixed(d);
const sem = a => a && a.std!=null ? a.std/Math.sqrt(Math.max(a.n,1)) : null;
const centry = k => report["class/"+k] || {};
const condLabel = {"cfg1.0":"Weak guidance","cfg3.5":"Full guidance","bandnorm":"SBN"};

function hfBadge(k){
  const d = (centry(k).delta_hf_frac||{}).mean;
  if(d==null) return "";
  return d>=0 ? `<span class="badge b-up">texture ▲ ${fmt(d)}</span>`
              : `<span class="badge b-down">texture ▼ ${fmt(d)}</span>`;
}

const TABS = [
  ["overview","Overview", renderOverview],
  ["motivation","Why normalize?", renderMotivation],
  ["method","How it works", renderMethod],
  ["pipeline","Pipeline", renderPipeline],
  ["metrics","Metrics", renderMetrics],
  ["compare","Compare", renderCompare],
  ["colorcorr","Color correction", renderColorCorr],
  ["quant","Quantitative", renderQuant],
  ["universal","Universal reference", renderUniversal],
  ["freq","Frequency control", renderFreq],
  ["cost","Compute cost", renderCost],
  ["phase","Phase & identity", renderPhase],
  ["takeaways","Takeaways", renderTakeaways],
];
let current = "overview";
function buildNav(){
  document.getElementById("nav").innerHTML = TABS.map(([id,label])=>
    `<button data-id="${id}" class="${id==current?'active':''}">${label}</button>`).join("");
  document.querySelectorAll("nav button").forEach(b=>
    b.onclick=()=>{current=b.dataset.id;buildNav();renderMain();window.scrollTo(0,0)});
}
function renderMain(){
  document.getElementById("main").innerHTML =
    `<section class="active">${TABS.find(t=>t[0]===current)[2]()}</section>`;
  if(current==="compare") wireCompare();
  if(current==="colorcorr") wireColorCorr();
  if(current==="universal") wireUniversal();
  if(current==="freq") wireFreq();
}

/* ===================== OVERVIEW ===================== */
function renderOverview(){
  return `
  <h2>Spectral Band Normalization</h2>
  <p class="lede">A diffusion model rendered with strong classifier guidance produces
  crisp, well-composed images — but also punchy, often over-saturated contrast.
  Drop the guidance and the contrast calms, but composition and prompt-fidelity
  weaken. <b>SBN</b> separates the two: it runs the full-guidance trajectory (so the
  composition is intact) and, after every denoising step, rescales the latent's
  spectral <i>power</i> back to the level a weak-guidance pass would have — keeping
  only the magnitudes, never the phase.</p>

  <div class="card thesis">
    <b>Effect.</b> SBN ≈ <i>full-guidance composition, with weak-guidance's calmer
    contrast &amp; palette, plus a fine-texture nudge on broadly-textured (photographic)
    subjects.</i> The texture nudge reverses for content whose detail is concentrated
    in a few high-power structures.
  </div>

  <h3>What you can do with the frequency split</h3>
  <p>Because the normalization is per radial frequency band, you can target just one
  range. Scaling the <span class="tag-lo">low bands</span> is a clean
  <b>structure / contrast</b> control. Scaling the <span class="tag-hi">high bands</span>,
  though, is <i>not</i> a detail control — amplifying high-frequency latent power makes
  the decoder produce granular artifacts and <b>reduces</b> real image detail. That
  asymmetry is itself a result. See <b>Frequency control</b>.</p>

  <h3>The three baseline conditions</h3>
  <div class="grid3">
    ${conds.map(c=>`<div class="card"><b>${c.label}</b> <span class="muted small">(${c.tag})</span><div class="muted small">${c.desc}</div></div>`).join("")}
  </div>

  <h3>Test content (6 classes)</h3>
  <div class="card">
    ${classes.map(c=>`<div style="margin:7px 0">
      <span class="pill">${c.label}</span>
      <span class="muted small">"${c.prompt}"</span> ${hfBadge(c.key)}</div>`).join("")}
  </div>`;
}

/* ===================== MOTIVATION ===================== */
function renderMotivation(){
  const pending = !cfgspec || !cfgspec.per_cfg;
  const theory = `
  <h2>Guidance, spectral power, and where real images sit</h2>
  <p class="lede">Classifier-free guidance (CFG) is an inference-time extrapolation the
  model was never trained on. Turning it up steadily <b>inflates the latent's spectral
  power</b> — and it turns out real photographs sit high on that same scale, near
  standard guidance, well above the unguided field.</p>

  <div class="card thesis">
    <b>Why CFG moves the spectral scale at all.</b> FLUX is a flow model: the transformer
    predicts a velocity <code>v<sub>θ</sub>(x,t)</code> trained by flow-matching to fit the
    data velocity, with <b>no CFG term</b> in the loss:
    <div class="eq">L = 𝔼 ‖ v<sub>θ</sub>(x<sub>t</sub>, t, c) − (x<sub>1</sub> − x<sub>0</sub>) ‖²</div>
    CFG appears only at inference, as an extrapolation between the conditional and
    unconditional velocities:
    <div class="eq">ṽ = v<sub>u</sub> + w · ( v<sub>c</sub> − v<sub>u</sub> )</div>
    At <b>w = 1</b> the sampler integrates exactly the trained field — no extrapolation.
    Each step of <b>w &gt; 1</b> pushes the velocity further along
    <code>v<sub>c</sub> − v<sub>u</sub></code>, and that compounding extrapolation shows up
    as steadily higher spectral power in the resulting latent.</p>
  </div>`;

  if(pending) return theory +
    `<div class="card note">⏳ CFG spectral-sweep data pending — run the sweep to populate
     the figures and table below.</div>`;

  const pc = cfgspec.per_cfg, real = cfgspec.real;
  const cell = (e,d=3)=> e ? `${fmt2(e.mean,d)}<span class="muted"> ±${fmt2(sem(e),d)}</span>` : "—";
  const w1 = pc[cfgspec.cfgs[0]];
  const wMax = cfgspec.cfgs[cfgspec.cfgs.length-1];
  const ratio = (pc[wMax].power.mean / w1.power.mean);
  // which guidance scale matches real-image power most closely?
  let nearestW = cfgspec.cfgs[0], best = Infinity;
  cfgspec.cfgs.forEach(w=>{const dp=Math.abs(pc[w].power.mean - (real?real.power.mean:0));
    if(real && dp<best){best=dp; nearestW=w;}});
  const realMark = real ? `<span class="badge b-up">real photos ≈ w ${nearestW}</span>` : "";

  const rows = cfgspec.cfgs.map((w)=>{
    const e=pc[w];
    const tag = (w==="1"||w==="1.0") ? ' <span class="muted small">(trained field)</span>'
              : (w===nearestW ? ' <span class="muted small">(≈ real)</span>' : '');
    return `<tr><td>w = ${w}${tag}</td>
      <td>${cell(e.power)}</td><td>${cell(e.lat_std)}</td>
      <td>${cell(e.spec_norm,1)}</td><td>${cell(e.low_power,3)}</td></tr>`;
  }).join("");
  const realRow = real ? `<tr style="background:rgba(63,185,80,.08)">
      <td><b>real images</b> <span class="muted small">(${real.n} VAE-encoded photos)</span></td>
      <td>${cell(real.power)}</td><td>${cell(real.lat_std)}</td>
      <td>${cell(real.spec_norm,1)}</td><td>${cell(real.low_power,3)}</td></tr>` : "";

  return theory + `
  <h3>Spectral power climbs monotonically with the guidance scale ${realMark}</h3>
  <p>Sweeping the true-CFG scale <code>w</code> (two-pass guidance, with the distilled
  guidance held neutral at ${fmt2(cfgspec.guidance_scale,1)}) over
  ${cfgspec.num_classes} prompt classes × ${cfgspec.seeds} seeds. Every spectral
  intensity measure — Fourier power, latent std, the latent's spectral norm, and
  low-band power — rises monotonically with <code>w</code>: from the unguided field to the
  strongest setting the latent gains <b>${fmt2(ratio,2)}×</b> in Fourier power.</p>
  <img class="plot" src="${S('e9/plots/cfg_power.png')}" alt="spectral intensity vs CFG"
    onerror="this.replaceWith(Object.assign(document.createElement('div'),{className:'ph',innerText:'cfg_power.png not found'}))">
  <p class="muted small">Spectral intensity measures vs the true-CFG scale. The green
  dashed line / band is the real-image level${real?` — it falls near <b>w ≈ ${nearestW}</b>,
  well above the unguided field (w = 1)`:""}. Guidance raises the latent <i>toward</i> real-image
  statistics, overshooting past them at high <code>w</code>.</p>

  <h3>The whole radial spectrum lifts — not just the slope</h3>
  <img class="plot" src="${S('e9/plots/cfg_psd.png')}" alt="radial PSD vs CFG"
    onerror="this.replaceWith(Object.assign(document.createElement('div'),{className:'ph',innerText:'cfg_psd.png not found'}))">
  <p class="muted small">Radially-averaged latent PSD per guidance scale, with real photos
  (dashed). Higher <code>w</code> ⇒ more power across the whole band; the real-image curve
  sits up among the mid-to-high guidance curves, not at <code>w = 1</code>.</p>

  <h3>The numbers</h3>
  <table>
    <tr><th>guidance</th><th>Fourier power</th><th>latent std</th>
        <th>spectral norm</th><th>low-band power</th></tr>
    ${rows}${realRow}
  </table>
  <p class="small muted">Anchored against ${real?real.n:"n"} VAE-encoded natural photographs.
  The unguided trained field (w = 1) is spectrally <i>weaker</i> than real data; standard
  guidance (w ≈ ${nearestW}) is where the latent's power matches the real-image scale.</p>

  <div class="card thesis" style="margin-top:16px">
    <b>What this means for the method.</b> The elevated spectral power that guidance produces
    is not an artifact — it is roughly where natural images live, so a generation <i>should</i>
    carry that spectral scale. Band normalization operates on exactly this axis: it sets the
    latent's per-band power to a chosen <i>reference</i> level every step. The rest of this page
    clamps toward the calmer weak-guidance look as a deliberate contrast / palette control
    (intentionally below the real-image scale); the very same machinery can target any level —
    including the real-image scale itself.
    <span class="muted small">(The distilled <code>cfg 1.0</code> ↔ <code>cfg 3.5</code>
    references the rest of this page uses are the same inflation along Flux's distilled
    guidance axis.)</span>
  </div>`;
}

/* ===================== METHOD ===================== */
function renderMethod(){
  return `
  <h2>How the band operation works</h2>
  <p>SBN runs as a hook at the end of every denoising step. The model and scheduler
  are untouched; only the latent that gets carried to the next step is adjusted.</p>

  <h3>Per step</h3>
  <ol>
    <li>Take the latent (16 channels × 128×128) <b>after</b> the scheduler has applied
      the model's update.</li>
    <li>FFT each channel; sort the Fourier coefficients into <b>24 radial frequency
      bands</b> (band 0 = DC + lowest frequencies, band 23 = finest).</li>
    <li>Measure each band's mean power and compare to the reference target for this
      step. Multiply every coefficient in the band by
      <code>gain = √(ref / current)</code>.</li>
    <li>Inverse FFT. The band map is radially symmetric, so the result stays real.</li>
  </ol>
  <div class="eq">F = FFT(latent) → |F| ×= √( ref<sub>band</sub> / current<sub>band</sub> ) → latent = IFFT(F)</div>

  <div class="card">
    <b>Two things stay fixed.</b>
    <ul style="margin:.4em 0">
      <li><b>Phase is never touched</b> — only magnitudes are rescaled. The phase of
        each Fourier coefficient carries <i>where</i> structures sit (the composition),
        which is exactly what we want to preserve.</li>
      <li>The gains are mild — typically in <code>[0.84, 1.12]</code> — so the model
        barely "fights" the correction on the next step.</li>
    </ul>
  </div>

  <h3 id="latent">The latent is normalized — not the velocity</h3>
  <p>FLUX is a flow model: the transformer predicts a <b>velocity</b>, and the scheduler
  integrates it to step the latent forward. SBN acts <b>after</b> that integration, on
  the resulting latent state — it does not modify the predicted velocity. In pipeline
  terms the order each step is:</p>
  <div class="eq">latent = scheduler.step(<span class="muted">velocity</span>, latent) &nbsp;→&nbsp; <b>latent = SBN(latent)</b> &nbsp;→&nbsp; next step</div>
  <p>So we renormalize the running image state <code>x<sub>t</sub></code>, and the model
  sees the corrected state when it predicts the next velocity. (A fixed gain applied to
  the velocity, or applied every step without re-measuring, would compound across the
  28 steps and blow up — the <code>√(ref/cur)</code> form instead targets a power
  <i>level</i> each step and is self-correcting.)</p>

  <h3>Where the reference comes from</h3>
  <p>The per-step, per-band power target is recorded from a few weak-guidance (cfg 1.0)
  runs of the prompt. Each kind of content settles at its own natural power level —
  photos high, illustration/abstract lower — which is why the reference is mildly
  content-specific (see <b>Universal reference</b> for using one general profile instead).</p>
  <img class="plot" src="${S('e9/plots/ref_std_curves.png')}" alt="per-step reference power curves"
       onerror="this.replaceWith(Object.assign(document.createElement('div'),{className:'ph',innerText:'reference curves not found'}))">
  <p class="muted small">Per-step latent power (std) of each content type's weak-guidance reference.</p>`;
}

/* ===================== PIPELINE (SVG) ===================== */
function renderPipeline(){
  return `
  <h2>The full pipeline</h2>
  <p>Two passes. A short <b>reference pass</b> records the calm power profile; the
  <b>guided pass</b> renders at full guidance with the per-step band correction. The
  reference can be the prompt's own, or a single general profile reused for everything.</p>
  <div class="card" style="overflow-x:auto">
  ${PIPELINE_SVG}
  </div>
  <p class="small muted">The correction block (FFT → rescale bands → IFFT) is the only
  addition to an ordinary sampling loop; the transformer and scheduler are unchanged.</p>`;
}
const PIPELINE_SVG = `
<svg viewBox="0 0 920 360" width="100%" font-family="ui-monospace,monospace" font-size="13">
  <defs>
    <marker id="ar" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 Z" fill="#58a6ff"/>
    </marker>
    <style>
      .box{fill:#161b22;stroke:#2a3340;rx:8}
      .lbl{fill:#e6edf3} .mut{fill:#9aa7b4} .acc{fill:#58a6ff}
      .ed{stroke:#58a6ff;stroke-width:1.6;fill:none;marker-end:url(#ar)}
      .ed2{stroke:#3fb950;stroke-width:1.6;fill:none;marker-end:url(#ar);stroke-dasharray:4 3}
    </style>
  </defs>

  <!-- reference pass -->
  <text x="20" y="30" class="acc">1 · Reference pass (weak guidance, a few seeds)</text>
  <rect class="box" x="20" y="42" width="150" height="46"/>
  <text x="95" y="62" text-anchor="middle" class="lbl">cfg 1.0 sampling</text>
  <text x="95" y="79" text-anchor="middle" class="mut">28 steps</text>
  <rect class="box" x="210" y="42" width="190" height="46"/>
  <text x="305" y="62" text-anchor="middle" class="lbl">record per-step</text>
  <text x="305" y="79" text-anchor="middle" class="mut">per-band power target</text>
  <path class="ed" d="M170,65 L208,65"/>
  <rect class="box" x="440" y="42" width="150" height="46" style="stroke:#3fb950"/>
  <text x="515" y="62" text-anchor="middle" class="lbl">reference</text>
  <text x="515" y="79" text-anchor="middle" class="mut">(28 × ch × 24)</text>
  <path class="ed" d="M400,65 L438,65"/>
  <text x="650" y="60" class="mut">or one</text>
  <text x="650" y="78" class="mut" fill="#3fb950">universal profile</text>
  <path class="ed2" d="M735,65 L735,150 L592,150" />

  <!-- guided pass -->
  <text x="20" y="135" class="acc">2 · Guided pass (full guidance + SBN)</text>
  <rect class="box" x="20" y="150" width="120" height="50"/>
  <text x="80" y="172" text-anchor="middle" class="lbl">latent xₜ</text>
  <text x="80" y="189" text-anchor="middle" class="mut">step t</text>
  <rect class="box" x="180" y="150" width="150" height="50"/>
  <text x="255" y="172" text-anchor="middle" class="lbl">transformer</text>
  <text x="255" y="189" text-anchor="middle" class="mut">→ velocity</text>
  <path class="ed" d="M140,175 L178,175"/>
  <rect class="box" x="370" y="150" width="150" height="50"/>
  <text x="445" y="172" text-anchor="middle" class="lbl">scheduler.step</text>
  <text x="445" y="189" text-anchor="middle" class="mut">integrate → xₜ₊₁</text>
  <path class="ed" d="M330,175 L368,175"/>
  <rect class="box" x="560" y="142" width="200" height="66" style="stroke:#a371f7"/>
  <text x="660" y="163" text-anchor="middle" class="lbl">SBN correction</text>
  <text x="660" y="180" text-anchor="middle" class="mut">FFT → |F|×=√(ref/cur)</text>
  <text x="660" y="195" text-anchor="middle" class="mut">→ IFFT  (phase kept)</text>
  <path class="ed" d="M520,175 L558,175"/>
  <!-- loop back -->
  <path class="ed" d="M660,208 L660,250 L80,250 L80,202"/>
  <text x="370" y="245" text-anchor="middle" class="mut">next step (28×)</text>

  <!-- decode -->
  <rect class="box" x="560" y="280" width="150" height="46"/>
  <text x="635" y="300" text-anchor="middle" class="lbl">VAE decode</text>
  <text x="635" y="317" text-anchor="middle" class="mut">→ image</text>
  <path class="ed" d="M660,208 L660,278"/>
</svg>`;

/* ===================== METRICS ===================== */
function renderMetrics(){
  const M = [
    ["Texture — high-frequency fraction (hf_frac)", "var(--good)",
     "The headline detail measure. Convert the image to grayscale, take its 2-D FFT, drop the DC term, and report the <b>share of total power at radial frequencies above 0.25</b> (fine detail). Because it is a power <i>ratio</i>, it is contrast-invariant: it measures how detailed the image is independently of how punchy it is.",
     "P(|f| &gt; 0.25) / P(total)"],
    ["Laplacian sharpness", "var(--muted)",
     "Variance of the Laplacian (a second-derivative edge filter) over the grayscale image. Intuitive, but it scales with <b>contrast²</b> — so it can fall even when an image gains fine texture, if contrast drops. This is exactly why we report hf_frac alongside it.",
     "var( ∇² gray )"],
    ["RMS contrast", "var(--muted)",
     "Standard deviation of grayscale pixel values — global contrast / tonal spread.",
     "std(gray)"],
    ["Colorfulness (Hasler–Süsstrunk)", "var(--muted)",
     "A perceptual color-richness score from the red–green and yellow–blue opponent channels (their spread plus a mean term). Higher = more vivid.",
     "√(σ²rg+σ²yb) + 0.3√(μ²rg+μ²yb)"],
    ["Saturation", "var(--muted)",
     "Mean per-pixel (max−min)/max over RGB — average colorimetric saturation.",
     "mean((max−min)/max)"],
    ["CLIP-T (prompt fidelity)", "var(--accent)",
     "Cosine similarity between a CLIP image embedding and the prompt's text embedding (CLIP ViT-L/14). Higher = more on-prompt. Used to check that calming the palette does not cost semantic adherence.",
     "cos( CLIP_img , CLIP_text )"],
    ["Low-band power (structure proxy)", "var(--lo)",
     "Fraction of the latent's spectral power held in the low radial bands (&lt; 0.25). Used in Frequency control as the structure counterpart to hf_frac.",
     "P_low / P_total  (latent)"],
  ];
  return `
  <h2>How each metric is computed</h2>
  <p>The texture story hinges on separating <b>detail</b> from <b>contrast</b> — hence
  two detail metrics that behave differently.</p>
  ${M.map(([t,c,d,f])=>`<div class="card metric" style="border-left-color:${c}">
     <b>${t}</b><p class="small" style="margin:.4em 0 .3em">${d}</p>
     <div class="eq" style="text-align:left;font-size:13px">${f}</div></div>`).join("")}`;
}

/* ===================== COMPARE ===================== */
function renderCompare(){
  return `
  <h2>Compare — weak / full / SBN</h2>
  <p>Same seed, three conditions. SBN holds full-guidance composition while the palette
  and contrast calm toward the weak-guidance reference.</p>
  <div class="controls card">
    <div><label>Content</label><br>
      <select id="cls">${classes.map(c=>`<option value="${c.key}">${c.label}</option>`).join("")}</select></div>
    <div style="flex:1;min-width:240px"><label>Seed <span class="seedval" id="sv">0</span></label><br>
      <input type="range" id="seed" min="0" max="${pairs-1}" value="0" style="width:100%"></div>
  </div>
  <div id="trip" class="triptych"></div>
  <div id="clsblurb" class="card muted small"></div>`;
}
function tile(key, cond, seed){
  const src = S(`e9/${key}/images/${cond.key}_s${seed}.png`);
  let clip = "";
  if(clipt){
    const arr = ((clipt["class/"+key]||{}).per_seed||{})[cond.key];
    if(arr && arr[seed]!=null) clip = `<div class="clip">CLIP-T ${fmt2(arr[seed],3)}</div>`;
  }
  return `<div class="tile"><div class="imgwrap">
      <img loading="lazy" src="${src}" alt="${cond.label} seed ${seed}"
        onerror="this.style.display='none';this.parentNode.innerHTML='<div class=ph>image generating…<br><span class=mono>${cond.key}_s${seed}</span></div>'">
    </div><div class="cap"><b>${cond.label}</b> <span class="muted small">${cond.tag}</span>
      <div class="d">${cond.desc}</div>${clip}</div></div>`;
}
function wireCompare(){
  const cls=document.getElementById("cls"), seed=document.getElementById("seed"),
        sv=document.getElementById("sv"), trip=document.getElementById("trip"),
        blurb=document.getElementById("clsblurb");
  function draw(){
    const k=cls.value, s=+seed.value; sv.textContent=s;
    trip.innerHTML = conds.map(c=>tile(k,c,s)).join("");
    const c = classes.find(x=>x.key===k);
    blurb.innerHTML = `<b>${c.label}</b> — ${c.domain}, reference power std ≈ ${c.refstd}. ${c.blurb} ${hfBadge(k)}`;
  }
  cls.onchange=draw; seed.oninput=draw; draw();
}

/* ===================== COLOR CORRECTION ===================== */
const CC_LABELS = {hist_match:"Histogram match", autocontrast:"Auto-contrast",
  lum_eq:"Luminance equalize", "contrast1.2":"Contrast ×1.2",
  "sat1.4":"Saturation ×1.4", "sat1.8":"Saturation ×1.8"};
const CC_ORDER = ["sat1.4","sat1.8","contrast1.2","autocontrast","lum_eq","hist_match"];
const CC_BEST = "/*__CC_BEST__*/";  // chosen practical (saturation) method, filled at build
const ccLabel = m => CC_LABELS[m] || m;
function ccMethods(){
  if(!colorcorr) return [];
  const any = Object.values(colorcorr)[0] || {};
  const ms = Object.keys(any.methods || {});
  return CC_ORDER.filter(m=>ms.includes(m)).concat(ms.filter(m=>!CC_ORDER.includes(m)));
}
function ccSeeds(){
  if(!colorcorr) return 1;
  return Math.max(...Object.values(colorcorr).map(c=>c.n_seeds||0), 1);
}
function renderColorCorr(){
  if(!colorcorr) return `<h2>Color correction</h2>
    <div class="card note">⏳ Color-correction results pending.</div>`;
  const methods = ccMethods();
  return `
  <h2>Recovering the palette without re-baking</h2>
  <p class="lede">Band normalization deliberately calms contrast and saturation — good for
  composition and texture, but the frame can read as washed-out next to full guidance. Because
  that is a global palette shift, it can be put back as <b>post-processing on the already-rendered
  frame</b> — no regeneration, no GPU. The goal: lift colorfulness / contrast back toward the
  full-guidance look while keeping the contrast-invariant detail (<code>hf_frac</code>) that
  band-norm bought.</p>

  <div class="card thesis"><b>The practical fix is a saturation boost.</b> A single
  <code>${ccLabel(CC_BEST)}</code> multiply on the rendered frame puts most of the lost color back —
  with <b>no extra generation, no reference image, no GPU</b> — and leaves detail
  (<code>hf_frac</code>) and contrast essentially untouched. It is the recommended, deployable
  correction.</div>

  <div class="card note"><b>Why not histogram matching?</b> Matching each channel's tone curve to the
  paired full-guidance frame does hit the palette target most precisely — but it <b>requires a second,
  full-guidance (cfg 3.5) generation pass per image</b> to produce that reference frame. That roughly
  <b>doubles generation cost</b> and needs the very full-guidance output band-norm is meant to avoid.
  So histogram match is an <i>upper-bound oracle</i> for "how close can correction get," not a
  practical, cheap correction. The saturation boost is what you would actually ship.</div>

  <div class="controls card">
    <div><label>Content</label><br>
      <select id="ccls">${classes.map(c=>`<option value="${c.key}">${c.label}</option>`).join("")}</select></div>
    <div><label>Correction</label><br>
      <select id="cmeth">${methods.map(m=>`<option value="${m}"${m===CC_BEST?" selected":""}>${ccLabel(m)}</option>`).join("")}</select></div>
    <div style="flex:1;min-width:220px"><label>Seed <span class="seedval" id="csv">0</span></label><br>
      <input type="range" id="cseed" min="0" max="${ccSeeds()-1}" value="0" style="width:100%"></div>
  </div>
  <div id="ctrip" class="triptych"></div>
  <div id="cctbl"></div>`;
}
function ccTile(label, desc, src, tag){
  return `<div class="tile"><div class="imgwrap">
      <img loading="lazy" src="${src}" alt="${label}"
        onerror="this.style.display='none';this.parentNode.innerHTML='<div class=ph>image missing<br><span class=mono>${tag}</span></div>'">
    </div><div class="cap"><b>${label}</b><div class="d">${desc}</div></div></div>`;
}
function ccTable(k, sel){
  const ce = colorcorr[k] || {}; const M = ce.methods || {};
  const cols = ["colorfulness","rms_contrast","saturation","hf_frac"];
  const head = `<tr><th>correction</th>${cols.map(c=>`<th>Δ ${c}</th>`).join("")}<th>gap to full</th></tr>`;
  const rows = ccMethods().map(m=>{
    const e = M[m]; if(!e) return "";
    const ds = e.delta_vs_sbn || {}, dc = e["delta_vs_cfg3.5"] || {};
    const cells = cols.map(c=>{
      const v = ds[c];
      const cl = c==="hf_frac" ? "" : (v>0?"pos":(v<0?"neg":""));
      return `<td class="${cl}">${fmt(v)}</td>`;
    }).join("");
    const hl = m===sel ? ' style="background:rgba(88,166,255,.10)"' : '';
    const star = m===CC_BEST ? ' ★' : '';
    const tagCost = m==="hist_match" ? ' <span class="muted small">(needs extra cfg 3.5 pass)</span>' : '';
    return `<tr${hl}><td>${ccLabel(m)}${star}${tagCost}</td>${cells}<td>${fmt2(Math.abs(dc.colorfulness),4)}</td></tr>`;
  }).join("");
  return `<h3>How each correction moves the metrics — ${classes.find(c=>c.key===k).label}</h3>
  <table>${head}${rows}</table>
  <p class="small muted">Δ vs the band-normalized frame, averaged over ${ce.n_seeds||"n"} seeds.
  Want <b>colorfulness / contrast / saturation up</b> and <b>Δ hf_frac ≈ 0</b> (detail kept).
  "Gap to full" is the remaining |colorfulness − full guidance|; smaller = closer to the
  full-guidance palette. The saturation boost (★) is the recommended cheap, single-pass fix;
  histogram match closes the gap slightly more but only by paying a second full cfg 3.5 pass per
  image, so it is a reference upper bound, not a practical correction.</p>`;
}
function wireColorCorr(){
  if(!colorcorr) return;
  const cls=document.getElementById("ccls"); if(!cls) return;
  const meth=document.getElementById("cmeth"), seed=document.getElementById("cseed"),
        sv=document.getElementById("csv"), trip=document.getElementById("ctrip"),
        tbl=document.getElementById("cctbl");
  function draw(){
    const k=cls.value, m=meth.value, s=+seed.value; sv.textContent=s;
    trip.innerHTML =
      ccTile("Band-normalized","the washed-out starting frame",
             S(`e9/${k}/images/bandnorm_s${s}.png`), `bandnorm_s${s}`) +
      ccTile(ccLabel(m),"palette recovered on the rendered frame",
             S(`e11/${k}/${m}/${m}_s${s}.png`), `${m}_s${s}`) +
      ccTile("Full guidance","the palette target",
             S(`e9/${k}/images/cfg3.5_s${s}.png`), `cfg3.5_s${s}`);
    tbl.innerHTML = ccTable(k, m);
  }
  cls.onchange=draw; meth.onchange=draw; seed.oninput=draw; draw();
}

/* ===================== QUANTITATIVE ===================== */
function renderQuant(){
  const names = classes.map(c=>c.key);
  const delRows = names.map(k=>{
    const e=centry(k);
    const cells = metrics.map(m=>{
      const a=e["delta_"+m]; if(!a) return `<td>—</td>`;
      const cl = a.mean>0?"pos":(a.mean<0?"neg":"");
      return `<td class="${cl}">${fmt(a.mean)}<span class="muted"> ±${fmt2(sem(a),4)}</span></td>`;
    }).join("");
    return `<tr><td>${classes.find(c=>c.key===k).label}</td>${cells}</tr>`;
  }).join("");
  return `
  <h2>Quantitative results</h2>
  <p>All deltas are <b>SBN − full guidance</b>, averaged over ${pairs} paired seeds per
  content type (± standard error).</p>
  <h3>Δ image metrics</h3>
  <table><tr><th>content</th>${metrics.map(m=>`<th>Δ ${m}</th>`).join("")}</tr>${delRows}</table>
  <p class="small muted">Universal: <b>contrast, colorfulness and saturation all drop</b>
  (SBN tames the palette). <b>hf_frac</b> (contrast-invariant detail) is the
  content-dependent one — positive for broadly-textured photos, negative where detail
  is concentrated in highlights (urban night, abstract). Laplacian sharpness scales with
  contrast², so it falls even when fine texture rises — only hf_frac isolates detail.</p>
  <img class="plot" src="${S('e9/plots/metrics_delta.png')}" alt="metric deltas"
    onerror="this.replaceWith(Object.assign(document.createElement('div'),{className:'ph',innerText:'delta plot not found'}))">
  <hr>
  <h3>CLIP-T — does SBN stay on-prompt?</h3>
  ${renderClip()}`;
}
function renderClip(){
  if(!clipt) return `<div class="card note">⏳ CLIP-T pending.</div>`;
  const names = classes.map(c=>c.key);
  const ck = ["cfg1.0","cfg3.5","bandnorm"];
  const colors = {"cfg1.0":"#8b949e","cfg3.5":"#58a6ff","bandnorm":"#3fb950"};
  let rows = names.map(k=>{
    const ce = clipt["class/"+k]||{};
    const cells = ck.map(c=>`<td>${ce[c]?fmt2(ce[c].mean,3):"—"}</td>`).join("");
    const d = ce.delta_bandnorm_cfg35, dc = d?(d.mean>0?"pos":(d.mean<0?"neg":"")):"";
    return `<tr><td>${classes.find(c=>c.key===k).label}</td>${cells}<td class="${dc}">${d?fmt(d.mean,4):"—"}</td></tr>`;
  }).join("");
  const overall={};
  ck.forEach(c=>{const v=names.map(k=>((clipt["class/"+k]||{})[c]||{}).mean).filter(x=>x!=null);overall[c]=v.length?v.reduce((a,b)=>a+b)/v.length:0;});
  const mx=Math.max(...ck.map(c=>overall[c]))*1.08;
  const bars=ck.map(c=>`<div class="barrow"><div class="lbl">${condLabel[c]}</div>
     <div class="bartrack"><div class="barfill" style="width:${overall[c]/mx*100}%;background:${colors[c]}">${fmt2(overall[c],3)}</div></div></div>`).join("");
  return `<p class="small muted">Text–image cosine (${clipt.model||"CLIP ViT-L/14"}), higher = more on-prompt.</p>
  <div class="card"><b>Mean CLIP-T across all content</b><div class="bars">${bars}</div></div>
  <table><tr><th>content</th><th>Weak</th><th>Full</th><th>SBN</th><th>Δ (SBN−Full)</th></tr>${rows}</table>
  <p class="small muted">SBN's biggest fidelity cost lands on urban night — the same content
  where its texture effect reverses — so an independent semantic metric flags the same
  worst-fit case.</p>`;
}

/* ===================== UNIVERSAL ===================== */
function renderUniversal(){
  if(!universal) return `<h2>Universal reference</h2><div class="card note">⏳ Universal-reference data pending.</div>`;
  const pc = universal.per_class||{};
  const rows = classes.map(c=>{
    const e=pc[c.key]||{};
    return `<tr><td>${c.label}</td><td>${fmt2(e.own_std_final,3)}</td>
      <td>${e.mean_reldev!=null?(100*e.mean_reldev).toFixed(1)+"%":"—"}</td></tr>`;
  }).join("");
  return `
  <h2>One general reference instead of a per-prompt one</h2>
  <p>The per-step band targets can come from the prompt's own weak-guidance pass, or from
  a single <b>general profile</b> — here the average of all six content references. If the
  general profile is close enough, the per-prompt reference pass can be skipped entirely
  (its cost amortizes to ~0).</p>
  <div class="card">
    <b>How general is "general"?</b>
    <div class="kv">
      <div class="k">Universal reference power (std)</div><div>${fmt2(universal.uni_std_final,3)}</div>
      <div class="k">Mean band-power deviation, prompt-own vs universal</div><div>${(100*universal.overall_mean_reldev).toFixed(1)}%</div>
    </div>
    <p class="small muted" style="margin-top:8px">Per-prompt references span std 0.65–0.90;
    the universal profile (0.77) is within a few percent of each. Abstract is the biggest
    outlier — the lowest-power content.</p>
  </div>
  <table><tr><th>content</th><th>own power std</th><th>dev vs universal</th></tr>${rows}</table>

  <h3>Per-prompt SBN vs universal SBN</h3>
  <div class="controls card">
    <div><label>Content</label><br>
      <select id="ucls">${classes.map(c=>`<option value="${c.key}">${c.label}</option>`).join("")}</select></div>
    <div style="flex:1;min-width:200px"><label>Seed <span class="seedval" id="usv">0</span></label><br>
      <input type="range" id="useed" min="0" max="4" value="0" style="width:100%"></div>
  </div>
  <div id="utrip" class="triptych"></div>`;
}
function wireUniversal(){
  const cls=document.getElementById("ucls"); if(!cls) return;
  const seed=document.getElementById("useed"), sv=document.getElementById("usv"),
        trip=document.getElementById("utrip");
  function utile(key,cond,label,desc,seed){
    return `<div class="tile"><div class="imgwrap">
      <img loading="lazy" src="${S(`e9/${key}/images/${cond}_s${seed}.png`)}"
        onerror="this.style.display='none';this.parentNode.innerHTML='<div class=ph>image generating…<br><span class=mono>${cond}_s${seed}</span></div>'">
      </div><div class="cap"><b>${label}</b><div class="d">${desc}</div></div></div>`;
  }
  function draw(){
    const k=cls.value, s=+seed.value; sv.textContent=s;
    trip.innerHTML =
      utile(k,"cfg3.5","Full guidance","cfg 3.5 baseline",s) +
      utile(k,"bandnorm","SBN (own reference)","prompt's own weak-guidance profile",s) +
      utile(k,"uninorm","SBN (universal)","one general profile, no per-prompt pass",s);
  }
  cls.onchange=draw; seed.oninput=draw; draw();
}

/* ===================== FREQUENCY CONTROL ===================== */
function renderFreq(){
  if(!freqctrl) return `<h2>Frequency control</h2><div class="card note">⏳ Frequency-control experiment pending.</div>`;
  const pk = Object.keys(freqctrl.prompts||{});
  return `
  <h2>Selective high / low frequency control</h2>
  <p>Instead of clamping every band to the reference, scale just one frequency range of
  the target before clamping. Gain 1.0 = plain SBN. The two ranges behave very differently:</p>
  <div class="grid2">
    <div class="card" style="border-left:3px solid var(--lo)"><b class="tag-lo">Low bands → structure (works)</b>
      <p class="small" style="margin:.4em 0 0">Raising the low-band target monotonically increases
      large-scale structure / contrast, and the image stays coherent through the mild range.
      A usable structure dial.</p></div>
    <div class="card" style="border-left:3px solid var(--hi)"><b class="tag-hi">High bands → detail (does not work)</b>
      <p class="small" style="margin:.4em 0 0">Amplifying high-frequency latent power does <b>not</b>
      sharpen the image. The VAE renders the off-distribution energy as a granular stipple
      artifact and real detail (hf_frac) <b>falls</b>; strong gains corrupt the subject.
      Image detail is not a high-frequency latent dial you can turn up.</p></div>
  </div>
  <p class="small muted">Move the slider to the extremes (×0.55, ×1.3) to see the effect — the
  high-band side visibly degrades into stipple, the low-band side mostly holds together.
  Each result is shown next to the matched cfg3.5 full-guidance baseline.</p>
  <div class="controls card">
    <div><label>Content</label><br>
      <select id="fcls">${pk.map(k=>`<option value="${k}">${k}</option>`).join("")}</select></div>
    <div><label>Target band</label><br>
      <select id="ftgt"><option value="high">high (detail)</option><option value="low">low (structure)</option></select></div>
    <div style="flex:1;min-width:220px"><label>Gain <span class="knob" id="fgv">1.0</span></label><br>
      <input type="range" id="fg" min="0" max="${(freqctrl.gains.length-1)}" value="${freqctrl.gains.indexOf(1.0)}" style="width:100%"></div>
    <div><label>Seed <span class="seedval" id="fsv">0</span></label><br>
      <input type="range" id="fseed" min="0" max="${freqctrl.seeds-1}" value="0"></div>
  </div>
  <div class="grid2">
    <div id="fimg" style="display:flex;gap:10px;flex-wrap:wrap"></div>
    <div class="card"><b>Effect of the gain sweep</b><div id="fchart"></div>
      <p class="small muted" id="fnote"></p></div>
  </div>`;
}
function wireFreq(){
  if(!freqctrl) return;
  const cls=document.getElementById("fcls"); if(!cls) return;
  const tgt=document.getElementById("ftgt"), g=document.getElementById("fg"),
        gv=document.getElementById("fgv"), seed=document.getElementById("fseed"),
        sv=document.getElementById("fsv"), img=document.getElementById("fimg"),
        chart=document.getElementById("fchart"), note=document.getElementById("fnote");
  const gains = freqctrl.gains;
  function tag(target,gain){ return gain===1.0 ? "baseline" : `${target}_g${gain}`; }
  function cells(){ return (freqctrl.prompts[cls.value]||{}).cells||{}; }
  function draw(){
    const gn=gains[+g.value]; gv.textContent=gn.toFixed(2); sv.textContent=seed.value;
    const t=tgt.value, s=+seed.value, tg=tag(t,gn);
    const src=S(`e9/freqctrl/${cls.value}/images/${tg}_s${s}.png`);
    const c=(cells()[tg]||{});
    const metric = t==="high" ? c.hf_frac : c.lowband_power;
    const mlabel = t==="high" ? "hf_frac (detail)" : "low-band power (structure)";
    // cfg3.5 full-guidance baseline for this prompt (same metric + matched seed image)
    const base=(freqctrl.prompts[cls.value]||{}).cfg35||{};
    const bmetric = t==="high" ? base.hf_frac : base.lowband_power;
    const bsrc=S(`e9/${cls.value}/images/cfg3.5_s${s}.png`);
    const dlt = (metric&&bmetric) ? metric.mean-bmetric.mean : null;
    const dstr = dlt==null ? "" : ` · Δ vs cfg3.5 <b>${dlt>=0?"+":""}${fmt2(dlt,4)}</b>`;
    img.innerHTML = `
      <div class="tile" style="flex:1;min-width:200px"><div class="imgwrap"><img loading="lazy" src="${src}"
          onerror="this.style.display='none';this.parentNode.innerHTML='<div class=ph>image generating…<br><span class=mono>${tg}_s${s}</span></div>'">
        </div><div class="cap"><b>${cls.value}</b> · <span class="tag-${t==='high'?'hi':'lo'}">${t} ×${gn.toFixed(2)}</span>
        <div class="d">${mlabel}: <b>${metric?fmt2(metric.mean,4):"—"}</b>${dstr}</div></div></div>
      <div class="tile" style="flex:1;min-width:200px"><div class="imgwrap"><img loading="lazy" src="${bsrc}"
          onerror="this.style.display='none';this.parentNode.innerHTML='<div class=ph>cfg3.5 baseline<br><span class=mono>cfg3.5_s${s}</span></div>'">
        </div><div class="cap"><b>cfg3.5</b> · <span class="muted">full guidance baseline</span>
        <div class="d">${mlabel}: <b>${bmetric?fmt2(bmetric.mean,4):"—"}</b></div></div></div>`;
    // mini bar chart of metric vs gain for this target, with a cfg3.5 reference row
    const vals = gains.map(gg=>{const cc=cells()[tag(t,gg)]||{}; const mm=t==="high"?cc.hf_frac:cc.lowband_power; return mm?mm.mean:null;});
    const bval = bmetric?bmetric.mean:null;
    const good=vals.concat(bval).filter(v=>v!=null); const lo=Math.min(...good), hi=Math.max(...good), rng=(hi-lo)||1;
    const baseRow = bval==null ? "" :
      `<div class="barrow"><div class="lbl">cfg3.5</div>
        <div class="bartrack"><div class="barfill" style="width:${8+(bval-lo)/rng*92}%;background:var(--accent)">${fmt2(bval,4)}</div></div></div>`;
    chart.innerHTML = `<div class="bars">`+gains.map((gg,i)=>{
      const v=vals[i]; const w=v==null?0:(8+(v-lo)/rng*92);
      const cur = gg===gn ? `style="outline:2px solid var(--accent)"`:"";
      return `<div class="barrow"><div class="lbl">×${gg.toFixed(2)}</div>
        <div class="bartrack" ${cur}><div class="barfill" style="width:${w}%;background:${t==='high'?'var(--hi)':'var(--lo)'}">${v!=null?fmt2(v,4):"—"}</div></div></div>`;
    }).join("")+baseRow+`</div>`;
    note.textContent = t==="high"
      ? "Inverse / destructive: as the high-band gain rises, real image detail (hf_frac) falls and the decoder produces a granular stipple artifact — amplifying high-frequency latent power does not sharpen the image."
      : "Clean knob: raising the low-band gain monotonically increases large-scale structure / contrast; the image stays coherent through the mild range.";
  }
  [cls,tgt].forEach(e=>e.onchange=draw); [g,seed].forEach(e=>e.oninput=draw); draw();
}

/* ===================== COMPUTE COST ===================== */
function renderCost(){
  const c = cost||{};
  const have = cost && cost.plain_ms_per_step!=null;
  const costline = have
    ? `<div class="kv">
        <div class="k">Plain full-guidance step</div><div>${fmt2(c.plain_ms_per_step,1)} ms</div>
        <div class="k">SBN step (with FFT correction)</div><div>${fmt2(c.clamp_ms_per_step,1)} ms</div>
        <div class="k">Per-step overhead</div><div>${c.overhead_pct!=null?"+"+fmt2(c.overhead_pct,1)+"%":"—"}</div>
       </div>`
    : `<p class="note">⏳ Per-step timing being measured…</p>`;
  return `
  <h2>Is SBN more expensive than normal generation?</h2>
  <p>Two cost components, and only one of them is real:</p>

  <h3>1 · Per-step overhead during generation — negligible</h3>
  <p>SBN adds one <code>FFT</code> + one <code>IFFT</code> on a 16×128×128 latent per step,
  plus a cheap per-band reduction. That is tiny next to a transformer forward pass.</p>
  <div class="card">${costline}</div>
  <p class="small muted">So an SBN generation runs at essentially the same speed as an
  ordinary full-guidance generation of the same step count.</p>

  <h3>2 · The reference pass — the only meaningful extra, and avoidable</h3>
  <p>Recording a per-prompt reference costs a few weak-guidance generations <i>once per
  prompt</i>. But the reference is barely content-specific (see
  <b>Universal reference</b>: prompt-own profiles sit within ~${universal?(100*universal.overall_mean_reldev).toFixed(0):"4"}%
  of one general profile). Precompute a single universal profile <b>once, ever</b>, and the
  per-prompt cost drops to zero — SBN then costs the same as a normal generation, full stop.</p>

  <div class="card thesis">
    <b>Bottom line.</b> With a universal reference, SBN ≈ one ordinary generation. With a
    per-prompt reference, add a small one-time reference pass for that prompt; the per-step
    FFT overhead is negligible either way.
  </div>`;
}

/* ===================== PHASE & IDENTITY ===================== */
const phaseKeys = rep => rep ? Object.keys(rep).filter(k=>k.startsWith("class/")).map(k=>k.slice(6)) : [];
const pmean = a => a.length ? a.reduce((s,x)=>s+(+x||0),0)/a.length : null;
const phImg = (p,cap) => `<div class="tile"><div class="imgwrap"><img loading="lazy" src="${S(p)}"
    onerror="this.style.display='none';this.parentNode.innerHTML='<div class=ph>figure pending<br><span class=mono>${p}</span></div>'">
  </div><div class="cap">${cap}</div></div>`;

function renderPhase(){
  const head = `
  <h2>Phase: where latent identity lives</h2>
  <p>The rest of this site is about spectral <b>power</b> — band-norm moves it and freezes the
  Fourier <b>phase</b>. This companion thread asks the opposite question: what does the phase
  carry? Across four experiments the answer is consistent — <b>a latent's identity lives in its
  low-band FFT phase</b>, the complement of the power story.</p>`;

  // 1 — phase vs magnitude swap (E13)
  let swap = `<div class="card note">⏳ Phase↔magnitude swap pending.</div>`;
  if(phaseSwap && phaseKeys(phaseSwap).length){
    const ks = phaseKeys(phaseSwap);
    const rows = ks.map(k=>{const d=phaseSwap["class/"+k];
      return `<tr><td>${k}</td><td><b>${fmt2(d.Aphase_Bmag.clip_to_A)}</b></td>
        <td>${fmt2(d.Aphase_Bmag.clip_to_B)}</td><td>${fmt2(d.phaseonly_A.clip_to_A)}</td>
        <td>${fmt2(d.magonly_A.clip_to_A)}</td></tr>`;}).join("");
    const mPh=pmean(ks.map(k=>phaseSwap["class/"+k].Aphase_Bmag.clip_to_A));
    const mMg=pmean(ks.map(k=>phaseSwap["class/"+k].Aphase_Bmag.clip_to_B));
    const mPo=pmean(ks.map(k=>phaseSwap["class/"+k].phaseonly_A.clip_to_A));
    const mMo=pmean(ks.map(k=>phaseSwap["class/"+k].magonly_A.clip_to_A));
    swap = `
    <p>Build a latent from one image's phase and another's magnitude (full spectrum), decode, and
    measure CLIP cosine to each source. <b>Magnitude alone carries no layout</b> (mag-only ≈
    ${fmt2(mMo)}, a textured swatch); <b>phase alone stays recognizable</b> but flat / desaturated
    (${fmt2(mPo)}). In the swap, identity follows the phase donor — by a modest, content-graded
    margin, since CLIP also reads palette from the magnitude.</p>
    <div class="grid2">
      ${phImg('e13/plots/identity_phase_vs_mag.png','A-phase + B-mag: CLIP to phase donor vs magnitude donor, per class.')}
      <div class="card"><b>Oppenheim–Lim in the Flux latent</b>
        <table><tr><th>class</th><th>swap→phase</th><th>swap→mag</th><th>phase-only</th><th>mag-only</th></tr>${rows}
        <tr style="border-top:2px solid var(--line)"><td><b>mean</b></td><td><b>${fmt2(mPh)}</b></td>
        <td>${fmt2(mMg)}</td><td>${fmt2(mPo)}</td><td>${fmt2(mMo)}</td></tr></table>
        <p class="small muted">The clean signal is the gap between phase-only (~0.75) and
        mag-only (~0.51); the swap margin itself is small.</p></div>
    </div>
    ${phImg('e13/grid_animal.png','Per pair: A, B, A-phase+B-mag, B-phase+A-mag, phase-only(A), mag-only(A). The swaps carry the phase donor\'s composition; mag-only is a structureless swatch.')}`;
  }

  // 2 — which bands carry identity (E14)
  let fn = `<div class="card note">⏳ Phase-function sweeps pending.</div>`;
  if(phaseFn && phaseFn._noise_curves){
    const eps=phaseFn._eps, nc=phaseFn._noise_curves, ks=Object.keys(nc.low||{});
    const row=b=>eps.map((e,i)=>`<td>${fmt2(pmean(ks.map(k=>nc[b][k][i])))}</td>`).join("");
    fn = `
    <p>Add Hermitian phase noise to a band and decode: <b>low-band phase noise destroys identity,
    high-band phase noise is almost free</b> — identity is concentrated in the low band. Scaling
    φ→αφ collapses identity at both α=0 and α=2; a frequency-linear ramp is just a spatial shift,
    while a constant phase offset genuinely distorts.</p>
    <div class="grid2">
      ${phImg('e14/plots/identity_vs_eps.png','CLIP-to-original vs phase-noise amplitude, low vs high band.')}
      <div class="card"><b>Identity erosion (CLIP to unmodified)</b>
        <table><tr><th>band</th>${eps.map(e=>`<th>ε=${e}</th>`).join("")}</tr>
        <tr><td class="tag-lo">low</td>${row('low')}</tr>
        <tr><td class="tag-hi">high</td>${row('high')}</tr></table>
        <p class="small muted">Low-band noise already erodes at ε=0.25 and saturates near 0.67;
        high-band barely moves.</p></div>
    </div>
    ${phImg('e14/grid_landscape_noise.png','Graded phase noise — low band (top row) vs high band (bottom): low-band loses the scene fast, high-band stays recognizable.')}`;
  }

  // 3 — classify by manipulation (E15)
  let clust = `<div class="card note">⏳ Clustering pending.</div>`;
  if(phaseClust && phaseClust.centroid_dist_to_orig){
    const cd=phaseClust.centroid_dist_to_orig, cl=phaseClust.clip||{};
    const rows=Object.keys(cd).map(k=>`<tr><td>${k}</td><td>${fmt2(cd[k])}</td></tr>`).join("");
    clust = `
    <p>Embed every manipulated output (CLIP) and ask whether edits form consistent groups. They do
    <i>not</i> cluster into discrete classes — CLIP space is dominated by image content (KMeans
    purity vs manipulation ${fmt2(cl.kmeans_purity_vs_manipulation)} vs vs-class
    ${fmt2(cl.kmeans_purity_vs_class)}). Instead the manipulation→output map is a clean
    <b>magnitude-of-effect axis</b>: distance from the unmodified centroid is monotone in how much
    the edit touches identity-bearing structure.</p>
    <div class="grid2">
      ${phImg('e15/plots/proj_by_manipulation.png','CLIP PCA colored by manipulation: content/class dominates; identity-destroying edits spread to the periphery.')}
      <div class="card"><b>CLIP distance to unmodified</b>
        <table><tr><th>manipulation</th><th>dist→orig</th></tr>${rows}</table>
        <p class="small muted">High-band edits collapse onto <i>orig</i>; mag-only is the far outlier.</p></div>
    </div>`;
  }

  // 4 — phase distributions baseline (E12)
  let dist = `<div class="card note">⏳ Phase-distribution baseline (E12) running…</div>`;
  if(phaseDist && phaseKeys(phaseDist).length){
    const ks=phaseKeys(phaseDist);
    const rows=ks.map(k=>{const d=phaseDist["class/"+k];
      return `<tr><td>${k}</td><td>${fmt2(d.R_lowest_band)}</td><td>${fmt2(d.R_midhigh_mean)}</td>
        <td>${fmt2(d.coherence_lowest_band)}</td><td>${fmt2(d.coherence_null)}</td></tr>`;}).join("");
    dist = `
    <p>The baseline that motivates all of the above: the phase <b>marginal</b> is uniform (the
    white-noise null) everywhere except the very lowest band — a flat phase histogram is expected
    and is <i>not</i> evidence that phase is uninformative. The signal is in cross-frequency
    <b>structure</b>, and cross-seed coherence is elevated only in the low band.</p>
    <div class="grid2">
      ${phImg('e12/plots/coherence_radial.png','Cross-seed phase coherence vs radial frequency: above the null only at low frequency.')}
      <div class="card"><b>Phase marginal vs joint structure</b>
        <table><tr><th>class</th><th>R low</th><th>R mid/high</th><th>coh low</th><th>null</th></tr>${rows}</table>
        <p class="small muted">Resultant length R≈0 (uniform) at mid/high; coherence rises above the
        null only in the lowest band.</p></div>
    </div>`;
  }

  return head
    + `<h2 style="margin-top:1.4em">1 · Phase ↔ magnitude swap <span class="muted">(Oppenheim–Lim)</span></h2>` + swap
    + `<h2 style="margin-top:1.4em">2 · Which bands carry identity</h2>` + fn
    + `<h2 style="margin-top:1.4em">3 · Classifying by manipulation</h2>` + clust
    + `<h2 style="margin-top:1.4em">4 · Phase distributions <span class="muted">(baseline)</span></h2>` + dist;
}

/* ===================== TAKEAWAYS ===================== */
function renderTakeaways(){
  return `
  <h2>Takeaways</h2>
  <div class="card thesis"><b>SBN = full-guidance composition, with weak-guidance's calmer
  contrast/palette, plus a texture nudge on broadly-textured subjects.</b></div>
  <ul>
    <li><b>It universally tames the palette</b> — contrast, colorfulness and saturation drop
      for every content type, most on the most saturated content.</li>
    <li><b>The texture effect is content-dependent</b> — fine detail rises where it is broadly
      distributed (fur, skin, foliage) and falls where it is concentrated in a few high-power
      structures (neon on dark, bold strokes).</li>
    <li><b>Prompt fidelity is preserved</b> (CLIP-T) except on the same concentrated-highlight
      content where the texture effect reverses.</li>
    <li><b>The frequency split is an asymmetric control surface</b> — low bands are a clean
      structure/contrast dial, but high bands are not a detail dial: amplifying high-frequency
      latent power degrades the image into stipple instead of sharpening it.</li>
    <li><b>Cheap</b> — negligible per-step overhead; a single universal reference removes the only
      meaningful extra cost.</li>
  </ul>`;
}

buildNav(); renderMain();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
