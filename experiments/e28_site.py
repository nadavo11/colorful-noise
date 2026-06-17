"""Build a self-contained HTML explainer for E28 (does biasing the seed RESCUE dropped
elements on hard compositional prompts?).

Reads results/e28/{report.json, grid_recovered.png, grid_nochange.png, summary.png} and
EMBEDS every image as base64 so the page is fully portable (open results/e28/index.html
anywhere). Honors CN_RESULTS.

The page STANDS ALONE to the project explainer standard (experiments/EXPLAINER_STANDARD.md):
TL;DR (seed-bias is a dead end for adherence; re-rolling wins) -> a glossary that DEFINES
EVERY TERM (seed bias / latent-mode optimization, B-VQA + how it is computed, the FAIL
threshold tau, per-prompt seed-dependence, recovery rate, the intervention-vs-re-roll
comparison) -> Method (scan + intervene parts and the question each answers) -> Results,
figures FIRST grouped by part (summary, recovered grid, no-change grid) -> "what to look
for" -> interpretation -> a numbers table with the best cell highlighted (re-roll wins).
(memory: experiment-documentation-standard.)

    python e28_site.py                              # rebuild results/e28/index.html
    python experiments/e28_seedrescue.py --part site   # same, model-free, the canonical path
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS
from e27_site import data_uri  # base64 embed -> portable single file

OUT = os.path.join(RESULTS, "e28")

# Reuse the E29/E30 explainer CSS verbatim (.tldr/.look/.read/.win/.cav, glossary dl, td.pos).
CSS = """
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


def img_tag(name, **kw):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        return f"<p class=cap>(missing <code>{name}</code>)</p>"
    return f"<img src='{data_uri(p, **kw)}' alt='{name}'>"


def fmt(x, n=3, sign=False):
    if x is None:
        return "—"
    return f"{x:+.{n}f}" if sign else f"{x:.{n}f}"


ARM_LABEL = {"A": "opt-A (full prompt)", "B": "opt-B (dropped phrase)", "reroll": "re-roll"}


def _row(label, cells, best_idx=None, signs=None):
    """A table row; highlight cell best_idx as .pos. cells = list of (value, is_signed)."""
    out = [f"<tr><td class=v>{label}</td>"]
    vals = [c[0] for c in cells]
    for i, (v, signed) in enumerate(cells):
        cls = "pos" if (best_idx is not None and i == best_idx) else ""
        out.append(f"<td class={cls}>{fmt(v, sign=signed)}</td>")
    out.append("</tr>")
    return "".join(out), vals


def _argmax(vals):
    best, bi = None, None
    for i, v in enumerate(vals):
        if v is not None and (best is None or v > best):
            best, bi = v, i
    return bi


def render(rep):
    cfg = rep["config"]
    tau = cfg["tau"]
    arms = ["A", "B", "reroll"]
    fail_rate = rep.get("fail_rate")
    dbvqa = rep.get("mean_dbvqa_fails", {})
    recov = rep.get("recovery_rate", {})
    sd = rep.get("seed_dependent", {})
    af = rep.get("always_fail", {})
    sd_recov = sd.get("recovery", {})
    sd_dbvqa = sd.get("dbvqa", {})
    harm = rep.get("do_no_harm_dbvqa_passers")
    n_scan = cfg.get("n_scan")
    n_fail = cfg.get("n_fail")
    n_int = cfg.get("n_intervened")

    h = ["<!doctype html><meta charset=utf-8><title>E28 — can the seed rescue dropped elements?</title>",
         f"<style>{CSS}</style>",
         "<h1>E28 — does biasing the <em>seed</em> rescue dropped elements on hard "
         "compositional prompts?</h1>"]

    # ---- TL;DR ----
    h.append(
        "<div class=tldr><b>In one paragraph.</b> A diffusion model starts from a random "
        "<b>seed</b> (a Gaussian noise array). On <b>hard compositional prompts</b> (e.g. "
        "“a green bench and a blue bowl”) it often <b>drops or mis-binds an element</b> — and "
        "<i>which</i> element it drops depends on the seed. E25–E27 showed seed-biasing is a "
        "do-no-harm <b>palette/appearance</b> lever on easy prompts; here we test the regime where "
        "the seed might actually matter, using a metric that <i>sees</i> a dropped element "
        "(<b>B-VQA</b>). We scan compositional prompts × seeds, find the failures, and on the worst "
        "ones <b>bias the failing seed toward the prompt</b> (iterative latent-mode optimization) — "
        "then compare to simply <b>re-rolling a fresh random seed</b>. <b>Verdict (a clean "
        "negative):</b> biasing the seed <b>loses to plain re-rolling</b> "
        f"(seed-dependent recovery {fmt(sd_recov.get('reroll'), 2)} for re-roll vs "
        f"{fmt(sd_recov.get('A'), 2)} / {fmt(sd_recov.get('B'), 2)} for the biased seed) and it "
        f"even <b>breaks compositions that already worked</b> (do-no-harm "
        f"{fmt(harm, 3, sign=True)} on passers). <b>Seed-as-adherence is a dead end; best-of-N "
        "random seeds + a B-VQA picker wins.</b> This closes the seed-as-adherence line "
        "(E25→E28).</div>")

    # ---- background / glossary ----
    h.append("<h2>0 · Background (plain language)</h2><dl>"
             "<dt>Seed / ‖z‖=√d</dt><dd>SDXL denoises a random <b>seed</b> (a "
             f"<code>4×128×128</code> Gaussian latent, ‖z‖≈√d=256) into the "
             f"<code>{cfg['size']}×{cfg['size']}</code> image. The same seed gives the same image; "
             "<i>which</i> objects the model renders depends on the seed.</dd>"
             "<dt>Seed bias / latent-mode optimization</dt><dd>“Biasing the seed” = nudging the "
             "seed toward a text target by <b>gradient on the seed</b>: we backprop "
             "<code>cosine(CLIP_image(decode(z)), CLIP_text(target))</code> to the seed and take "
             f"<code>K={cfg['K']}</code> small steps (lr <code>{cfg['lr']}</code>), "
             "<b>re-standardizing</b> after each step so the seed stays on the ‖z‖=√d sphere "
             "(E25/E26 latent-mode, reused here as <code>optimize_seed</code>). It is a smooth "
             "nudge of the <i>current</i> seed, not a new draw.</dd>"
             "<dt>B-VQA (↑, 0–1)</dt><dd>The metric — T2I-CompBench attribute binding. spaCy "
             "extracts the prompt's <b>noun phrases</b> (e.g. “a green bench”, “a blue bowl”); "
             "BLIP-VQA answers “{phrase}?” per image, giving <code>P(yes)</code>; the B-VQA score "
             "is the <b>product</b> of those P(yes). Because it is a product, <b>one dropped or "
             "mis-bound element tanks it</b> — unlike CLIP-T, B-VQA <i>sees</i> a missing object. "
             "(We use <code>Salesforce/blip-vqa-base</code> — safetensors; the capfilt-large "
             "checkpoint is .bin-only and blocked on torch&lt;2.6. Fine for <i>relative</i> "
             "recovery.)</dd>"
             f"<dt>FAIL threshold τ={tau}</dt><dd>A (prompt, seed) generation <b>fails</b> if its "
             f"B-VQA is below <code>τ={tau}</code> — i.e. at least one element is dropped or "
             "mis-bound. We only intervene on failures (the cases with headroom).</dd>"
             "<dt>Per-prompt seed-dependence</dt><dd>For each prompt, the <b>fraction of seeds that "
             "pass</b> (B-VQA≥τ). A prompt with passrate&gt;0 is <b>seed-dependent</b>: some seed "
             "<i>can</i> render it, so a seed change <i>could</i> help. A prompt with "
             "passrate=0 is <b>always-fail</b>: no seed in range works, so seed manipulation can't "
             "help. We stratify the verdict by this.</dd>"
             "<dt>Recovery rate (↑)</dt><dd>Of the intervened failures, the <b>fraction that cross "
             f"τ={tau}</b> after the treatment (i.e. now pass). Computed per arm; reported overall "
             "and on the seed-dependent stratum (the fair comparison).</dd>"
             "<dt>The treatments — intervention vs re-roll</dt><dd>On each failure we compare three "
             "things. <b>opt-A</b> = bias the failing seed toward the <i>full prompt</i>. "
             "<b>opt-B</b> = bias it toward the <i>single lowest-P(yes) noun phrase</i> (the "
             "dropped element specifically). <b>re-roll</b> = the control: just draw a <i>fresh "
             "random seed</i> (no optimization). The whole question is whether the gradient bias "
             "(A/B) beats luck (re-roll).</dd>"
             "<dt>Do-no-harm (↑, want ≈0)</dt><dd>A safety check: apply opt-A to <i>passing</i> "
             "pairs and measure ΔB-VQA. If biasing toward the prompt is safe it should be ≈0; if "
             "negative, the bias <b>breaks compositions that already worked</b>.</dd>"
             "</dl>")

    # ---- method ----
    h.append("<h2>1 · Method</h2><dl>"
             "<dt>scan (Stage 1)</dt><dd>Generate the baseline over "
             f"{cfg.get('per_cat', '?')}×{len(cfg.get('cats', []))} CompBench prompts "
             f"(<code>{', '.join(cfg.get('cats', []))}</code>) × {len(cfg.get('seeds', []))} seeds "
             f"(<code>{cfg.get('seeds')}</code>); score B-VQA + per-phrase P(yes); locate the "
             f"<b>failures</b> (B-VQA&lt;τ={tau}) and each prompt's seed-dependence. <i>How often "
             "does SDXL drop an element, and is it seed-dependent?</i></dd>"
             "<dt>intervene (Stage 2)</dt><dd>On the worst failures, run <b>opt-A</b> / "
             "<b>opt-B</b> (bias the failing seed) and the <b>re-roll</b> control; re-score. "
             "<i>Does biasing the seed toward the prompt recover the dropped element — and does it "
             "beat just drawing a new seed?</i></dd>"
             "<dt>do-no-harm</dt><dd>Apply opt-A to a sample of <b>passing</b> pairs. <i>Is biasing "
             "the seed at least safe on cases that already work?</i></dd>"
             "</dl>")

    h.append("<h2>2 · Results</h2>")
    h.append(f"<p class=cap>Scan = {fmt(n_scan, 0)} baseline gens; "
             f"baseline fail rate {fmt(fail_rate, 3)} ({fmt(n_fail, 0)} fails); intervened on "
             f"{fmt(n_int, 0)}. Metric = B-VQA (↑), Δ = treated − baseline.</p>")

    # --- summary figure FIRST ---
    h.append("<h3>summary — recovery by arm</h3>")
    h.append(img_tag("summary.png"))
    h.append("<p class=cap>left: mean ΔB-VQA on failures, by arm; right: recovery rate "
             f"(fraction that cross τ={tau}) for all fails vs the seed-dependent stratum.</p>")
    h.append("<div class=look><b>What to look for.</b> Compare the <code>re-roll</code> bar to "
             "<code>opt-A</code>/<code>opt-B</code>. If <b>re-roll is highest</b>, a fresh random "
             "seed recovers more failures than gradient-biasing the failing one — i.e. the bias "
             "does not beat luck.</div>")
    h.append("<div class='read'><b>Reading.</b> Re-roll wins on every metric. Even on the "
             "<b>seed-dependent</b> stratum (where some seed genuinely can render the prompt, so "
             f"the seed matters most), a fresh random seed recovers "
             f"{fmt(sd_recov.get('reroll'), 2)} of failures vs "
             f"{fmt(sd_recov.get('B'), 2)} / {fmt(sd_recov.get('A'), 2)} for the biased seed — "
             "re-roll wins by a wide margin.</div>")

    # numbers table
    h.append(f"<table><tr><th>treatment</th><th>mean ΔB-VQA ↑</th><th>recovery rate "
             f"(cross τ={tau}) ↑</th><th>recovery on<br>seed-dependent (n={sd.get('n', '?')}) ↑</th></tr>")
    # find winners per column
    col_dbvqa = [dbvqa.get(a) for a in arms]
    col_recov = [recov.get(a) for a in arms]
    col_sd = [sd_recov.get(a) for a in arms]
    bi_d, bi_r, bi_s = _argmax(col_dbvqa), _argmax(col_recov), _argmax(col_sd)
    for i, a in enumerate(arms):
        c_d = "pos" if i == bi_d else ""
        c_r = "pos" if i == bi_r else ""
        c_s = "pos" if i == bi_s else ""
        h.append(f"<tr><td class=v>{ARM_LABEL[a]}</td>"
                 f"<td class={c_d}>{fmt(dbvqa.get(a), sign=True)}</td>"
                 f"<td class={c_r}>{fmt(recov.get(a))}</td>"
                 f"<td class={c_s}>{fmt(sd_recov.get(a))}</td></tr>")
    h.append("</table>")
    if af.get("n"):
        h.append(f"<p class=cap><b>Always-fail stratum</b> (n={af['n']}, passrate=0): recovery "
                 f"A={fmt(af.get('recovery', {}).get('A'))} · "
                 f"B={fmt(af.get('recovery', {}).get('B'))} · "
                 f"reroll={fmt(af.get('recovery', {}).get('reroll'))} — these recover with "
                 "<i>nothing</i>: no seed in range works, so seed manipulation can't help.</p>")

    # --- recovered grid ---
    h.append("<h3>grid_recovered — failures we intervened on</h3>")
    h.append(img_tag("grid_recovered.png"))
    h.append("<p class=cap>rows = a failing (prompt, seed), labelled with the dropped phrase; "
             "columns = <b>baseline (fail) · opt-A (full) · opt-B (dropped phrase) · re-roll</b>.</p>")
    h.append("<div class=look><b>What to look for.</b> In which column does the dropped element "
             "<b>reappear</b>? If the re-roll column most often shows <i>both</i> objects bound "
             "correctly while opt-A/opt-B mostly keep the same (failing) composition with a "
             "different palette, the bias is moving <b>appearance, not which objects appear</b>.</div>")
    h.append("<div class='read'><b>Reading.</b> The gradient toward the prompt shifts "
             "palette/appearance while keeping the latent in the <b>same compositional basin</b> — "
             "it degrades or recolors the existing composition rather than jumping to the mode that "
             "renders the missing element. Changing <i>which</i> mode renders requires a genuinely "
             "different seed (a re-roll), not a smooth nudge of the current one.</div>")

    # --- no-change / passers grid ---
    h.append("<h3>grid_nochange — do-no-harm on passers</h3>")
    h.append(img_tag("grid_nochange.png"))
    h.append("<p class=cap>passing pairs before/after applying opt-A (the seed bias toward the "
             "full prompt).</p>")
    h.append("<div class=look><b>What to look for.</b> Do the already-correct compositions survive "
             "the bias, or does an object get dropped / a binding break?</div>")
    h.append(f"<div class='read'><b>Reading.</b> <b>do-no-harm FAILED</b>: applying the "
             f"optimization to <i>passing</i> pairs dropped B-VQA by "
             f"<b>{fmt(harm, 3, sign=True)}</b> on average — biasing the seed toward the prompt "
             "actively <b>breaks compositions that already worked</b>, so it is not a safe "
             "default.</div>")

    # ---- verdict / caveats ----
    h.append("<h2>3 · Verdict (a clean negative)</h2>")
    h.append(
        "<div class='win'><b>Biasing the seed does not beat simply re-rolling it.</b> Even in the "
        "regime the seed genuinely matters (seed-dependent compositional failures), a fresh random "
        f"seed recovers <b>{fmt(sd_recov.get('reroll'), 2)}</b> of failures vs "
        f"<b>{fmt(sd_recov.get('B'), 2)} / {fmt(sd_recov.get('A'), 2)}</b> for the gradient-biased "
        "seed — re-roll wins on every metric, by a wide margin. <b>Always-fail</b> prompts recover "
        "with <i>nothing</i> (no seed in range works). And the bias <b>even hurts working cases</b> "
        f"({fmt(harm, 3, sign=True)} on passers). <b>Why:</b> consistent with E25–E27, the gradient "
        "toward CLIP/text moves <b>palette/appearance</b>, keeping the latent in the same "
        "compositional basin while degrading it — it does not jump the sampler to a different "
        "“which objects appear” mode. Changing <i>which</i> mode renders requires a genuinely "
        "different seed (a re-roll), not a smooth nudge of the current one. <b>Seed-as-adherence is "
        "a dead end; best-of-N random seeds + a B-VQA picker wins.</b> This closes the "
        "seed-as-adherence line (E25→E28); the remaining real, positive use of seed-biasing is the "
        "appearance/palette steering documented in E27 (Arm B).</div>")
    h.append("<h2>4 · Caveats &amp; next</h2>"
             "<div class=cav>(1) B-VQA uses <code>blip-vqa-base</code> (the larger capfilt "
             "checkpoint is blocked on torch&lt;2.6); fine for <i>relative</i> recovery, less so "
             "for absolute scores. (2) The negative is robust across the failures and a "
             "seed-dependence stratification, but n per stratum is modest — read the direction "
             "(re-roll &gt; opt-B &gt; opt-A), not third decimals. (3) The constraint ‖z‖=√d was "
             "held on every edit. <b>Next:</b> the practical lever for compositional adherence is "
             "best-of-N random seeds + a B-VQA/VQA picker (not gradient on the seed); seed-biasing "
             "stays an appearance tool.</div>")

    h.append("<h2>5 · Reproduce</h2>"
             "<pre><code>python experiments/e28_seedrescue.py quick   # smoke\n"
             "python experiments/e28_seedrescue.py          # full -> results/e28/{grids, summary, report.json}\n"
             "python experiments/e28_seedrescue.py --part site  # model-free rebuild of index.html\n"
             "</code></pre>")
    h.append("<p class=cap>Generated by <code>e28_site.py</code> from "
             "<code>results/e28/report.json</code> + figures. Method: "
             "<code>e28_seedrescue.py</code> (reuses <code>compbench.py</code>, "
             "<code>e26_seedalign_sdxl.py</code>, <code>clip_sim.py</code>). See also "
             "<code>EXPERIMENT_28.md</code>, and E25/E26/E27 for the seed-bias thread.</p>")
    return "".join(h)


def build():
    """Load report.json, render, write index.html. Returns the dest path or None if no data."""
    rpath = os.path.join(OUT, "report.json")
    if not os.path.exists(rpath):
        print(f"[e28-site] no {rpath}; run e28_seedrescue.py (the generator) first — "
              "nothing to template, leaving index.html untouched")
        return None
    rep = json.load(open(rpath))
    html = render(rep)
    os.makedirs(OUT, exist_ok=True)
    dest = os.path.join(OUT, "index.html")
    with open(dest, "w") as f:
        f.write(html)
    print(f"[e28-site] wrote {dest}  ({len(html) // 1024} KB, no model loaded)")
    return dest


def main():
    build()


if __name__ == "__main__":
    main()
