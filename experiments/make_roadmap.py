"""Generate the interactive research roadmap site from roadmap_registry.py.

Writes a LIGHTWEIGHT multi-page HTML site to docs/roadmap/ (shared style.css; no
base64-embedded images -- figures, if any, are referenced from experiments/results
by relative path, so the committed site is KB not MB):

  index.html        project thesis + an SVG map of the threads over time + legend
  thread-<id>.html  one per thread: the narrative, how-to-proceed, experiment cards
  experiments.html  filterable table of every experiment E0-E31
  glossary.html     central definitions so the pages can stay short

To update: edit roadmap_registry.py, then run  python experiments/make_roadmap.py
Honors CN_RESULTS for locating result figures / per-experiment index.html links.

    python experiments/make_roadmap.py
"""
import html
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(REPO, "docs", "roadmap")
RESULTS = os.environ.get("CN_RESULTS") or os.path.join(HERE, "results")

sys.path.insert(0, HERE)
from roadmap_registry import THREADS, EXPERIMENTS, STATUSES  # noqa: E402
from manifest import load_all  # noqa: E402

THREAD_BY_ID = {t["id"]: t for t in THREADS}
MANIFESTS = load_all()


def manifest_note(e):
    """One-line provenance from the per-run manifest. Drift = a scripted experiment
    with no manifest at all (added but never logged). Backfilled manifests are the
    accepted historical record, not drift. Returns (text, is_drift)."""
    m = MANIFESTS.get(e["id"])
    if not m:
        if e.get("script"):
            return ("run not logged (no manifest)", True)
        return (None, False)
    if m.get("source") == "backfill":
        return ("backfilled — no run metrics", False)
    bits = [f"logged {m.get('logged')}"]
    if m.get("git_commit"):
        bits.append(f"@ {m['git_commit']}")
    if m.get("metrics"):
        bits.append(", ".join(f"{k}={v}" for k, v in list(m["metrics"].items())[:4]))
    return (" · ".join(bits), False)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def esc(s):
    return html.escape(str(s if s is not None else ""))


def enum(eid):
    """E10 -> 10 (for chronological x-placement)."""
    try:
        return int(eid[1:])
    except (ValueError, IndexError):
        return 0


def rel(target_repo_path, outfile="index.html"):
    """Relative link from an output page to a repo-relative path."""
    return os.path.relpath(os.path.join(REPO, target_repo_path),
                           os.path.dirname(os.path.join(OUT, outfile)))


def results_link(slug, outfile):
    """Relative link to experiments/results/<slug>/index.html, or None if absent."""
    if not slug:
        return None
    p = os.path.join(RESULTS, slug, "index.html")
    if not os.path.exists(p):
        return None
    return os.path.relpath(p, os.path.dirname(os.path.join(OUT, outfile)))


def image_tag(img, outfile):
    """<img> referencing a results figure by relative path (light), or '' if absent."""
    if not img:
        return ""
    p = os.path.join(RESULTS, img)
    if not os.path.exists(p):
        return f"<p class=cap>(figure <code>{esc(img)}</code> — generate results to view)</p>"
    src = os.path.relpath(p, os.path.dirname(os.path.join(OUT, outfile)))
    return f"<img class=fig src='{esc(src)}' loading=lazy alt='{esc(img)}'>"


def pill(status):
    label, color = STATUSES.get(status, (status, "#666"))
    return f"<span class=pill style='--c:{color}'>{esc(label)}</span>"


def report_outfile(exp):
    """report-<eid>.html for an experiment whose .md writeup exists on disk, else None."""
    doc = exp.get("doc")
    if doc and doc.endswith(".md") and os.path.exists(os.path.join(REPO, doc)):
        return f"report-{exp['id']}.html"
    return None


def doc_link(exp, outfile):
    """Link to the RENDERED report page (nice HTML), else the chronological log fallback."""
    rof = report_outfile(exp)
    if rof:
        return os.path.relpath(os.path.join(OUT, rof),
                               os.path.dirname(os.path.join(OUT, outfile))), "report"
    return rel("experiments/EXPERIMENTS.md", outfile), "EXPERIMENTS.md"


def _img_data_uri(path, max_px=1400, quality=82):
    """Downscaled JPEG data: URI for a figure, or None if it can't be read."""
    try:
        import base64
        from io import BytesIO
        from PIL import Image
        im = Image.open(path).convert("RGB")
        if max(im.size) > max_px:
            r = max_px / max(im.size)
            im = im.resize((max(1, round(im.width * r)), max(1, round(im.height * r))))
        buf = BytesIO()
        im.save(buf, "JPEG", quality=quality)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def _inline_images(html, base_dir):
    """Replace <img src=...> with a downscaled data-URI resolved relative to base_dir; a
    missing file (e.g. a heavy grid that lives only on /storage) degrades to a caption."""
    import re

    def repl(m):
        src = m.group(1)
        if src.startswith(("data:", "http")):
            return m.group(0)
        p = src if os.path.isabs(src) else os.path.normpath(os.path.join(base_dir, src))
        uri = _img_data_uri(p)
        if uri:
            return m.group(0).replace(src, uri)
        return (f"<p class=cap>(figure <code>{esc(os.path.basename(src))}</code> — "
                f"full-res under <code>/storage/.../roadmap_results/</code>)</p>")

    return re.sub(r'<img[^>]*\bsrc="([^"]+)"[^>]*>', repl, html)


def build_report(exp):
    """Render EXPERIMENT_<n>.md to a styled, self-contained HTML page (tables/code/figures)."""
    import markdown
    doc = exp["doc"]
    src = os.path.join(REPO, doc)
    of = report_outfile(exp)
    html = markdown.markdown(open(src, encoding="utf-8").read(),
                             extensions=["tables", "fenced_code", "sane_lists", "attr_list"])
    html = _inline_images(html, os.path.dirname(src))
    nav_links = (f"<p class=rnav><a href='thread-{esc(exp['thread'])}.html'>← "
                 f"{esc(THREAD_BY_ID[exp['thread']]['title'])}</a> · "
                 f"<a href='experiments.html'>all experiments</a> · "
                 f"<a href='{esc(rel(doc, of))}'>source .md</a>"
                 + (f" · <a href='{esc(rel(exp['script'], of))}'>driver</a>" if exp.get('script') else "")
                 + "</p>")
    body = f"{nav_links}<article class=report>{html}</article>{nav_links}"
    return of, page(f"{exp['id']} — {exp.get('title', '')}", body, "experiments.html", of)


def nav(active, outfile="index.html"):
    items = [("index.html", "Overview")]
    items += [(f"thread-{t['id']}.html", t["title"]) for t in THREADS]
    items += [("experiments.html", "All experiments"), ("glossary.html", "Glossary")]
    out = ["<nav>"]
    for href, label in items:
        cls = " class=on" if href == active else ""
        rl = os.path.relpath(os.path.join(OUT, href), os.path.dirname(os.path.join(OUT, outfile)))
        out.append(f"<a{cls} href='{esc(rl)}'>{esc(label)}</a>")
    out.append("</nav>")
    return "".join(out)


def page(title, body, active, outfile):
    css = os.path.relpath(os.path.join(OUT, "style.css"),
                          os.path.dirname(os.path.join(OUT, outfile)))
    return (f"<!doctype html><html lang=en><head><meta charset=utf-8>"
            f"<meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>{esc(title)}</title><link rel=stylesheet href='{esc(css)}'></head>"
            f"<body>{nav(active, outfile)}<main>{body}</main>"
            f"<footer>Generated by <code>experiments/make_roadmap.py</code> from "
            f"<code>experiments/roadmap_registry.py</code> — edit the registry and re-run "
            f"to update.</footer></body></html>")


# --------------------------------------------------------------------------- #
# style.css  (small, shared)
# --------------------------------------------------------------------------- #
STYLE = """
:root{--fg:#1a1a1a;--mut:#586069;--line:#d8dee4;--bg:#fff;--soft:#f6f8fa}
*{box-sizing:border-box}
body{font:15px/1.6 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--fg);
  margin:0;background:var(--bg)}
main{max-width:980px;margin:0 auto;padding:8px 18px 48px}
nav{position:sticky;top:0;z-index:5;background:#fffffff2;backdrop-filter:blur(6px);
  border-bottom:1px solid var(--line);padding:9px 18px;display:flex;flex-wrap:wrap;gap:4px 14px;
  font-size:13px;max-width:100%}
nav a{color:var(--mut);text-decoration:none;white-space:nowrap}
nav a:hover{color:#0969da} nav a.on{color:var(--fg);font-weight:700}
h1{font-size:25px;line-height:1.25;margin:22px 0 6px}
h2{font-size:19px;margin:30px 0 8px;border-bottom:1px solid var(--line);padding-bottom:5px}
h3{font-size:15px;margin:0 0 4px}
p{margin:8px 0} a{color:#0969da}
.lede{font-size:16px;color:#24292f;background:var(--soft);border:1px solid var(--line);
  border-radius:8px;padding:14px 16px;margin:14px 0}
.pill{display:inline-block;font-size:11px;font-weight:700;color:#fff;background:var(--c);
  border-radius:999px;padding:1px 9px;vertical-align:middle;letter-spacing:.02em}
.legend{display:flex;flex-wrap:wrap;gap:8px 16px;margin:10px 0;font-size:13px;color:var(--mut)}
.legend span{display:inline-flex;align-items:center;gap:6px}
.dot{width:11px;height:11px;border-radius:50%;display:inline-block}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px;margin:14px 0}
.tcard{border:1px solid var(--line);border-left:5px solid var(--c);border-radius:8px;
  padding:12px 14px;background:#fff}
.tcard h3 a{color:var(--fg);text-decoration:none} .tcard h3 a:hover{color:#0969da}
.tcard .sum{color:var(--mut);font-size:13.5px;margin:6px 0 0}
.proceed{border-left:4px solid #2da44e;background:#eafaf0;border-radius:4px;padding:11px 14px;margin:14px 0}
.proceed.dead{border-color:#cf222e;background:#ffeef0}
.proceed.warn{border-color:#9a6700;background:#fff8e6}
.exp{border:1px solid var(--line);border-radius:8px;padding:13px 15px;margin:13px 0;background:#fff}
.exp .hd{display:flex;align-items:baseline;gap:9px;flex-wrap:wrap}
.exp .eid{font-weight:800;font-variant-numeric:tabular-nums}
.exp .ti{font-weight:600} .exp .mdl{color:var(--mut);font-size:12px;margin-left:auto}
.exp dl{display:grid;grid-template-columns:auto 1fr;gap:2px 12px;margin:9px 0 0}
.exp dt{font-weight:700;color:var(--mut);font-size:12.5px;text-transform:uppercase;letter-spacing:.03em}
.exp dd{margin:0}
.exp .links{margin-top:9px;font-size:13px;display:flex;gap:14px;flex-wrap:wrap}
.fig{max-width:100%;border:1px solid var(--line);border-radius:6px;margin:8px 0}
.cap{color:var(--mut);font-size:12.5px;margin:4px 0}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:13.5px}
th,td{border:1px solid var(--line);padding:6px 9px;text-align:left;vertical-align:top}
th{background:var(--soft);position:sticky;top:42px;cursor:default}
td.c,th.c{text-align:center;white-space:nowrap}
code{background:#eff1f3;padding:1px 5px;border-radius:3px;font-size:12.5px}
.controls{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0;font-size:13px}
.controls select{padding:4px 8px;border:1px solid var(--line);border-radius:6px}
.svgwrap{overflow-x:auto;border:1px solid var(--line);border-radius:8px;margin:14px 0;background:#fff}
.gloss dt{font-weight:700;margin-top:12px} .gloss dd{margin:2px 0 2px 0;color:#24292f}
footer{max-width:980px;margin:0 auto;padding:18px;color:var(--mut);font-size:12px;
  border-top:1px solid var(--line)}
.rnav{font-size:13px;color:var(--mut);margin:10px 0}
.report{font-size:15px;line-height:1.65}
.report img{max-width:100%;border:1px solid var(--line);border-radius:6px;margin:12px 0;display:block}
.report h1{font-size:24px;margin:18px 0 8px} .report h2{margin-top:26px}
.report ul,.report ol{margin:8px 0;padding-left:24px} .report li{margin:3px 0}
.report blockquote{border-left:3px solid var(--line);margin:10px 0;padding:2px 14px;color:var(--mut)}
.report pre{background:var(--soft);border:1px solid var(--line);border-radius:6px;padding:10px 12px;overflow-x:auto}
.report pre code{background:none;padding:0;font-size:12.5px}
.report table{font-size:13.5px} .report hr{border:none;border-top:1px solid var(--line);margin:22px 0}
"""


# --------------------------------------------------------------------------- #
# index: thesis + SVG research map + thread cards
# --------------------------------------------------------------------------- #
def svg_map():
    nums = [enum(e["id"]) for e in EXPERIMENTS]
    nmin, nmax = min(nums), max(nums)
    L, R, top = 224, 34, 54
    lane_h = 52
    W = 940
    H = top + lane_h * len(THREADS) + 24
    x = lambda n: L + (n - nmin) / max(1, (nmax - nmin)) * (W - L - R)
    s = [f"<svg viewBox='0 0 {W} {H}' width='{W}' role=img "
         "aria-label='research threads over experiment number'>"]
    # time axis ticks
    s.append(f"<text x='{L}' y='26' font-size='11' fill='#586069'>← experiment number "
             "(chronological) →</text>")
    for n in range(0, nmax + 1, 5):
        s.append(f"<text x='{x(n):.0f}' y='44' font-size='10' fill='#9aa3ad' "
                 f"text-anchor='middle'>E{n}</text>")
    for i, t in enumerate(THREADS):
        cy = top + lane_h * i + lane_h / 2
        _, lane_color = STATUSES.get(t["status"], ("", "#666"))
        # lane label + status pill
        s.append(f"<text x='10' y='{cy - 3:.0f}' font-size='12.5' font-weight='700' "
                 f"fill='#1a1a1a'>{esc(t['title'])}</text>")
        s.append(f"<rect x='10' y='{cy + 4:.0f}' rx='6' width='{8 + 6.4 * len(STATUSES[t['status']][0]):.0f}' "
                 f"height='14' fill='{lane_color}'/>")
        s.append(f"<text x='14' y='{cy + 14:.0f}' font-size='10' font-weight='700' "
                 f"fill='#fff'>{esc(STATUSES[t['status']][0])}</text>")
        s.append(f"<line x1='{L}' y1='{cy:.0f}' x2='{W - R}' y2='{cy:.0f}' "
                 f"stroke='#e7ebef' stroke-width='1'/>")
        for e in [e for e in EXPERIMENTS if e["thread"] == t["id"]]:
            _, c = STATUSES.get(e["status"], ("", "#666"))
            ex = x(enum(e["id"]))
            s.append(f"<a href='thread-{esc(t['id'])}.html'>"
                     f"<circle cx='{ex:.0f}' cy='{cy:.0f}' r='11' fill='{c}' "
                     f"stroke='#fff' stroke-width='1.5'><title>{esc(e['id'])} — "
                     f"{esc(e['title'])} [{esc(STATUSES[e['status']][0])}]</title></circle>"
                     f"<text x='{ex:.0f}' y='{cy + 3:.0f}' font-size='8.5' fill='#fff' "
                     f"font-weight='700' text-anchor='middle' pointer-events='none'>"
                     f"{esc(e['id'])}</text></a>")
    s.append("</svg>")
    return "".join(s)


def build_index():
    of = "index.html"
    legend = "<div class=legend>" + "".join(
        f"<span><i class=dot style='background:{c}'></i>{esc(lbl)}</span>"
        for lbl, c in STATUSES.values()) + "</div>"
    cards = ["<div class=cards>"]
    for t in THREADS:
        _, c = STATUSES.get(t["status"], ("", "#666"))
        n = sum(1 for e in EXPERIMENTS if e["thread"] == t["id"])
        cards.append(
            f"<div class=tcard style='--c:{c}'>"
            f"<h3><a href='thread-{esc(t['id'])}.html'>{esc(t['title'])}</a> {pill(t['status'])}</h3>"
            f"<p class=sum>{esc(t['summary'])}</p>"
            f"<p class=sum><b>{n}</b> experiments · "
            f"<a href='thread-{esc(t['id'])}.html'>open thread →</a></p></div>")
    cards.append("</div>")
    body = (
        "<h1>Colorful-Noise — research roadmap</h1>"
        "<div class=lede>This project asks what you can <b>control in a diffusion model by "
        "working in the Fourier domain</b>. The starting observation: a latent's 2-D FFT "
        "splits into <b>phase</b> (which carries image structure/layout) and per-band "
        "<b>magnitude / power</b> (which carries texture and palette). From that one split "
        "grew several research vectors — re-leveling spectral power toward better targets "
        "(<b>SBN</b>), mapping where structure is committed (<b>phase</b>), transferring "
        "style and editing images in frequency space, steering the initial <b>seed</b>, and "
        "moving the whole idea onto the <b>text conditioning</b>. The map below places every "
        "experiment on its thread and timeline; colour = current status.</div>"
        + legend
        + "<div class=svgwrap>" + svg_map() + "</div>"
        "<p class=cap>Click any node or card to open that thread. Each thread page tells the "
        "story end-to-end and says explicitly how to proceed.</p>"
        "<h2>The threads</h2>" + "".join(cards)
        + "<h2>How this site is maintained</h2>"
        "<p>It is generated from a single registry. To add or update an experiment, edit "
        f"<code>experiments/roadmap_registry.py</code> and run "
        f"<code>python experiments/make_roadmap.py</code>. Deep per-experiment writeups live in "
        f"the root <code>EXPERIMENT_*.md</code> files; the chronological log is "
        f"<a href='{esc(rel('experiments/EXPERIMENTS.md', of))}'>experiments/EXPERIMENTS.md</a>.</p>")
    return page("Colorful-Noise — research roadmap", body, "index.html", of)


# --------------------------------------------------------------------------- #
# thread pages
# --------------------------------------------------------------------------- #
def exp_card(e, of):
    dlink, dname = doc_link(e, of)
    rlink = results_link(e.get("results"), of)
    links = [f"<a href='{esc(rel(e['script'], of))}'>driver</a>" if e.get("script") else "",
             f"<a href='{esc(dlink)}'>{esc(dname)}</a>"]
    if rlink:
        links.append(f"<a href='{esc(rlink)}'>results ↗</a>")
    if e["id"] in MANIFESTS:
        mpath = "experiments/manifests/" + e["id"] + ".json"
        links.append(f"<a href='{esc(rel(mpath, of))}'>manifest</a>")
    note, drift = manifest_note(e)
    rows = [("Asks", e.get("motivation")), ("Method", e.get("method")),
            ("Result", e.get("result")), ("Verdict", e.get("verdict")),
            ("Next", e.get("nxt"))]
    if note:
        rows.append(("⚠ Drift" if drift else "Logged", note))
    dl = "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in rows if v)
    fig = image_tag(e.get("image"), of)
    return (f"<div class=exp style='--c:{STATUSES.get(e['status'],('','#666'))[1]}'>"
            f"<div class=hd><span class=eid>{esc(e['id'])}</span>"
            f"<span class=ti>{esc(e['title'])}</span>{pill(e['status'])}"
            f"<span class=mdl>{esc(e.get('models'))}</span></div>"
            f"{fig}<dl>{dl}</dl>"
            f"<div class=links>{''.join(x for x in links if x)}</div></div>")


def build_thread(t):
    of = f"thread-{t['id']}.html"
    exps = sorted([e for e in EXPERIMENTS if e["thread"] == t["id"]], key=lambda e: enum(e["id"]))
    pcls = {"dead-end": " dead", "pending": " warn"}.get(t["status"], "")
    body = (
        f"<h1>{esc(t['title'])} {pill(t['status'])}</h1>"
        f"<p class=lede>{esc(t['narrative'])}</p>"
        f"<div class='proceed{pcls}'><b>How to proceed.</b> {esc(t['proceed'])}</div>"
        f"<h2>Experiments in this thread ({len(exps)})</h2>"
        + "".join(exp_card(e, of) for e in exps))
    return of, page(f"{t['title']} — roadmap", body, of, of)


# --------------------------------------------------------------------------- #
# experiments table
# --------------------------------------------------------------------------- #
def build_table():
    of = "experiments.html"
    threads_opt = "".join(f"<option value='{esc(t['id'])}'>{esc(t['title'])}</option>"
                          for t in THREADS)
    status_opt = "".join(f"<option value='{esc(k)}'>{esc(v[0])}</option>"
                         for k, v in STATUSES.items())
    rows = []
    for e in sorted(EXPERIMENTS, key=lambda e: enum(e["id"])):
        dlink, dname = doc_link(e, of)
        rlink = results_link(e.get("results"), of)
        t = THREAD_BY_ID[e["thread"]]
        links = []
        if e.get("script"):
            links.append(f"<a href='{esc(rel(e['script'], of))}'>code</a>")
        links.append(f"<a href='{esc(dlink)}'>doc</a>")
        if rlink:
            links.append(f"<a href='{esc(rlink)}'>↗</a>")
        if e["id"] in MANIFESTS:
            links.append(f"<a href='{esc(rel('experiments/manifests/' + e['id'] + '.json', of))}'>m</a>")
        _, drift = manifest_note(e)
        drift_tag = " <span class=cap>⚠ run not logged</span>" if drift else ""
        rows.append(
            f"<tr data-thread='{esc(e['thread'])}' data-status='{esc(e['status'])}'>"
            f"<td class=c><b>{esc(e['id'])}</b></td>"
            f"<td>{esc(e['title'])}<br><span class=cap>{esc(e.get('verdict'))}</span>{drift_tag}</td>"
            f"<td><a href='thread-{esc(e['thread'])}.html'>{esc(t['title'])}</a></td>"
            f"<td class=c>{esc(e.get('models'))}</td>"
            f"<td class=c>{pill(e['status'])}</td>"
            f"<td class=c>{' · '.join(links)}</td></tr>")
    js = ("<script>function flt(){var th=document.getElementById('ft').value,"
          "st=document.getElementById('fs').value;"
          "document.querySelectorAll('tbody tr').forEach(function(r){"
          "r.style.display=((!th||r.dataset.thread==th)&&(!st||r.dataset.status==st))?'':'none';});}"
          "</script>")
    body = (
        "<h1>All experiments</h1>"
        "<div class=controls>"
        f"<label>Thread <select id=ft onchange=flt()><option value=''>all</option>{threads_opt}</select></label>"
        f"<label>Status <select id=fs onchange=flt()><option value=''>all</option>{status_opt}</select></label>"
        "</div>"
        "<table><thead><tr><th class=c>#</th><th>Title / verdict</th><th>Thread</th>"
        "<th class=c>Model</th><th class=c>Status</th><th class=c>Links</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table>" + js)
    return page("All experiments — roadmap", body, of, of)


# --------------------------------------------------------------------------- #
# glossary
# --------------------------------------------------------------------------- #
GLOSSARY = [
    ("Latent", "Diffusion models denoise in a compressed array (e.g. Flux 16×128×128, "
     "SD1.5 4×64×64) that a VAE decodes to the image. All spectral analysis here is on latents."),
    ("FFT phase vs magnitude", "The 2-D Fourier transform gives every spatial frequency a "
     "magnitude (how strong that ripple is — texture/palette power) and a phase (where the "
     "ripples line up). Oppenheim–Lim: phase carries most recognisable structure/layout."),
    ("Radial band / PSD", "Group Fourier coefficients into rings by distance from DC (low = "
     "coarse, high = fine). The power-spectral-density (PSD) is power vs band — an image's "
     "coarse-vs-fine fingerprint."),
    ("SBN (Spectral Band Normalization)", "Re-level a generated latent's per-(channel,band) "
     "power toward a reference spectrum, leaving phase (layout) untouched. Originally targeted "
     "a cfg=1 reference; real-SBN (E23) targets the spectrum of real photos."),
    ("psd_match / AdaIN-in-Fourier", "The operator behind SBN/style transfer: multiply each "
     "band's FFT magnitude by √(target/current), keep phase. It is AdaIN applied to the radial "
     "power spectrum (style = per-band moments)."),
    ("CFG (classifier-free guidance)", "The knob for how hard the prompt steers generation. "
     "cfg=1 ≈ unsteered (soft, low-contrast); higher overrides more of the seed and inflates "
     "low-frequency power above natural levels (E10)."),
    ("CFG-Zero*", "A training-free guidance variant used as a baseline in the SD3.5 port (E17)."),
    ("Seed (z_T) vs output (z_0)", "z_T is the initial Gaussian noise latent; deterministic "
     "DDIM maps it to the final latent z_0. E29 measures how much of z_0 the seed pre-commits."),
    ("Circular correlation", "Correlation for angles (−π ≡ +π): Jammalamadaka–SenGupta. Used "
     "in E29 to measure seed-phase vs output-phase inheritance per band against a permutation null."),
    ("CLIP-T / CLIP-I", "CLIP cosine similarity of an image to the prompt text (CLIP-T, "
     "prompt adherence) or to another image (CLIP-I, content/identity preservation)."),
    ("Aesthetic / ImageReward", "Learned image-quality predictors (LAION aesthetic MLP; "
     "ImageReward). Note the aesthetic predictor rewards over-sharpening past human preference."),
    ("B-VQA", "BLIP-VQA attribute-binding score from T2I-CompBench — the proper compositional "
     "adherence metric (does the image actually contain the prompt's objects/attributes)."),
    ("VQAScore", "A VQA-entailment compositional score; its dependency stack (t2v-metrics) "
     "conflicts with the BLIP torch.load path, so it is run selectively / sometimes skipped."),
    ("DDIM / RF inversion", "Run the sampler backward to recover the noise that produces an "
     "image, enabling real-image editing. DDIM inversion (eps-models, SDXL) is reliable (E22); "
     "rectified-flow inversion (SD3.5) drifted (E21)."),
    ("FlowEdit", "Inversion-free real-image editing: integrate the ODE delta between source and "
     "target conditioning directly (E31)."),
    ("Token-axis FFT", "1-D FFT along the token sequence of the text embedding (FNet-motivated, "
     "E24/E30): low token-frequency ≈ subject/identity, high ≈ style/detail."),
    ("Golden noise / seed steering", "The (here, negative) idea that optimising the initial seed "
     "improves prompt adherence. E25–E28: it loses to best-of-N re-rolling."),
]


def build_glossary():
    of = "glossary.html"
    dl = "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in GLOSSARY)
    body = ("<h1>Glossary</h1><p>Terms used across the roadmap and the "
            "<code>EXPERIMENT_*.md</code> writeups.</p><dl class=gloss>" + dl + "</dl>")
    return page("Glossary — roadmap", body, of, of)


# --------------------------------------------------------------------------- #
# coverage check (drift detector)
# --------------------------------------------------------------------------- #
def _scripts_on_disk():
    """Experiment numbers that have at least one e<N>_*.py driver in experiments/."""
    import re
    nums = {}
    for fn in os.listdir(HERE):
        m = re.match(r"e(\d+)_.*\.py$", fn)
        if m:
            nums.setdefault(int(m.group(1)), []).append(fn)
    return nums


def check_coverage():
    """Report drift between the registry, the scripts on disk, and the manifests.
    Returns the number of actionable problems (unregistered drivers / stale paths)."""
    reg_nums = {enum(e["id"]): e for e in EXPERIMENTS}
    disk = _scripts_on_disk()
    problems = 0

    unregistered = sorted(n for n in disk if n not in reg_nums)
    if unregistered:
        problems += len(unregistered)
        print("DRIFT — driver(s) on disk with no registry entry:")
        for n in unregistered:
            print(f"  E{n}: {', '.join(sorted(disk[n]))}")

    stale = [(e["id"], e["script"]) for e in EXPERIMENTS
             if e.get("script") and not os.path.exists(os.path.join(REPO, e["script"]))]
    if stale:
        problems += len(stale)
        print("DRIFT — registry script path does not exist:")
        for eid, sp in stale:
            print(f"  {eid}: {sp}")

    no_manifest = [e["id"] for e in EXPERIMENTS if e.get("script") and e["id"] not in MANIFESTS]
    if no_manifest:
        print(f"NOTE — scripted experiments with no manifest: {', '.join(no_manifest)}")

    if not problems:
        print(f"[roadmap] coverage OK — {len(reg_nums)} registry entries, "
              f"{len(disk)} numbered drivers on disk, no drift.")
    return problems


# --------------------------------------------------------------------------- #
def main():
    os.makedirs(OUT, exist_ok=True)
    written = []
    with open(os.path.join(OUT, "style.css"), "w") as f:
        f.write(STYLE)
    written.append("style.css")
    with open(os.path.join(OUT, "index.html"), "w") as f:
        f.write(build_index())
    written.append("index.html")
    for t in THREADS:
        of, htmltext = build_thread(t)
        with open(os.path.join(OUT, of), "w") as f:
            f.write(htmltext)
        written.append(of)
    for e in EXPERIMENTS:                       # rendered .md report pages (nice HTML + figures)
        if report_outfile(e):
            of, htmltext = build_report(e)
            with open(os.path.join(OUT, of), "w") as f:
                f.write(htmltext)
            written.append(of)
    with open(os.path.join(OUT, "experiments.html"), "w") as f:
        f.write(build_table())
    written.append("experiments.html")
    with open(os.path.join(OUT, "glossary.html"), "w") as f:
        f.write(build_glossary())
    written.append("glossary.html")
    total = sum(os.path.getsize(os.path.join(OUT, w)) for w in written)
    print(f"[roadmap] wrote {len(written)} files to {OUT}  ({total // 1024} KB total)")
    for w in written:
        print(f"          {w}  ({os.path.getsize(os.path.join(OUT, w))} B)")


if __name__ == "__main__":
    if "--check" in sys.argv:
        sys.exit(1 if check_coverage() else 0)
    main()
