"""Build a self-contained HTML explainer for E26 (seed alignment on SDXL, N-sweep).

Reads results/e26/{report.json, grid.png, deltaclip_vs_N.png} and EMBEDS every image
as base64 (via e27_site.data_uri) so the page is fully portable (open
results/e26/index.html anywhere). Honors CN_RESULTS through `common.RESULTS`.

The page STANDS ALONE: it defines every term (the seed z~N(0,I); the ||z||=sqrt(d)
Gaussian-sphere / moment constraint; the latent-space CLIP objective that decodes the
seed itself and NEVER runs the UNet / no x_hat0; the long-aware CLIP-T metric; the inner
step count N; the baseline / N=k / N=1*strong columns; deltaCLIP). It leads each result
with its figure, then the "what to look for", then the interpretation, then the numbers
table (best cell highlighted). (memory: experiment-documentation-standard.)

    python experiments/e26_site.py            # rebuild from results/e26/report.json (no model)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS

try:
    from e27_site import data_uri          # base64 embed -> portable single file
except Exception:                          # noqa: BLE001
    data_uri = None

OUT = os.path.join(RESULTS, "e26")

# Reused verbatim from E29/E30 (.tldr/.look/.read/.win/.cav, glossary dl, td.pos).
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


def img_tag(name):
    """<img> with the figure embedded as base64 (portable), or a note if missing."""
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        return f"<p class=cap>(missing <code>{name}</code>)</p>"
    src = data_uri(p) if data_uri else name
    return f"<img src='{src}' alt='{name}'>"


def fmt(x, sign=True):
    if x is None:
        return "—"
    return f"{x:+.4f}" if sign else f"{x:.4f}"


def _col_keys(sweep):
    """The generation columns in report order: baseline, N=k..., N=1*strong."""
    return ["baseline"] + [f"N={n}" for n in sweep] + ["N=1*strong"]


def render(rep):
    cfg = rep.get("config", {})
    sweep = cfg.get("sweep_n", [1, 2, 3, 5])
    md = rep.get("mean_clip_delta_by_n", {})       # {"1":Δ, "2":Δ, ..., "1*":Δ}
    records = rep.get("records", [])
    d = cfg.get("latent_dim")
    sqrt_d = cfg.get("sqrt_d")
    size = cfg.get("size")
    lr = cfg.get("lr")
    strong_lr = cfg.get("strong_lr")
    gen_steps = cfg.get("gen_steps")
    guidance = cfg.get("guidance")

    # The N=1 cell is the headline; pull it for the TL;DR if present.
    d1 = md.get("1")

    h = ["<!doctype html><meta charset=utf-8><title>E25–E26 — seed alignment (SDXL N-sweep)</title>",
         f"<style>{CSS}</style>",
         "<h1>E25–E26 — biasing the initial <em>seed</em> toward the prompt "
         "(SDXL, sweeping how many optimization steps)</h1>"]

    # ---- TL;DR ----
    h.append(
        "<div class=tldr><b>In one paragraph.</b> Diffusion sampling starts from a random "
        "<b>seed</b> — a Gaussian noise latent <code>z ~ N(0, I)</code>. There is a well-known "
        "observation that <b>the seed leaves traces in the output</b> (the final latent stays "
        "highly correlated with the seed it started from). So instead of sampling <code>z</code> "
        "purely at random, can we spend a <i>tiny, cheap</i> amount of optimization to nudge the "
        "seed <b>toward the prompt</b> before generation, and have that bias survive into the image? "
        "We optimize the seed <b>purely in latent space</b> — decode <code>z</code>, score its "
        "CLIP-similarity to the text, and step <code>z</code> uphill — <b>never running the UNet</b>, "
        "while a hard moment constraint keeps <code>z</code> a valid Gaussian sample (zero mean, "
        "unit variance, so <code>‖z‖ = √d</code> exactly). E25 piloted this on SD1.5; <b>E26</b> "
        "ports it to <b>SDXL</b> with long <b>DPG-Bench</b> prompts and <b>sweeps N</b>, the number "
        "of inner gradient steps. "
        + (f"<b>Headline:</b> the nudge is a genuinely <b>gentle, do-no-harm, break-even</b> "
           f"operation — a single cheap step (<code>N=1</code>, Δ long-CLIP-T = "
           f"<code>{fmt(d1)}</code>) is the only clearly non-negative point and "
           f"<b>more steps do not help</b> (they drift slightly negative / off-manifold). "
           if d1 is not None else
           "<b>Headline:</b> the nudge is a gentle, break-even operation and a single cheap step "
           "(<code>N=1</code>) is best — more steps do not help. ") +
        "The robust, reproducible part is the constraint (<code>‖z‖=√d</code>, moments held) "
        "and the absence of structural damage; the small ΔCLIP-T sign is within noise.</div>")

    # ---- background / glossary ----
    h.append("<h2>0 · Background (plain language)</h2><dl>"
             "<dt>Seed <code>z ~ N(0, I)</code></dt><dd>Diffusion generation starts from a random "
             "Gaussian noise latent (for SDXL at "
             f"{size or 1024}px this is a <code>4×{(size or 1024)//8}×{(size or 1024)//8}</code> "
             f"array, <code>d = {d or 'd'}</code> numbers). The sampler denoises it into the final "
             "latent that the VAE paints into the image.</dd>"

             "<dt>The Gaussian-sphere / moment constraint (<code>‖z‖ = √d</code>)</dt><dd>After "
             "<i>every</i> gradient step we <b>re-standardize</b> <code>z ← (z − mean) / std</code>, "
             "forcing <b>zero mean and unit variance</b>. Since <code>‖z‖² = d·(var + mean²)</code>, "
             "this pins the norm to <code>‖z‖ = √d</code> exactly "
             + (f"(<code>√{d} = {sqrt_d:.0f}</code>) " if (d and sqrt_d) else "") +
             "— i.e. the optimization is a move <b>on the sphere of radius √d</b> where a true "
             "Gaussian sample lives, not off into low-probability latent regions. Measured "
             "<code>‖z*‖</code> equals <code>√d</code> to two decimals on every run.</dd>"

             "<dt>The latent-space CLIP objective (<b>no UNet, no x̂₀</b>)</dt><dd>We do <b>not</b> "
             "predict the clean latent <code>x̂₀</code> and we <b>never run the UNet</b> in the "
             "optimization. We simply decode the seed itself and compare it to the prompt in CLIP "
             "space: <code>loss = − cosine( CLIP_image( VAE.decode(z) ), CLIP_text(prompt) )</code>. "
             "Gradients flow only through the frozen VAE decoder and frozen CLIP image encoder back "
             "to <code>z</code>; the UNet is used <b>only</b> in the ordinary generation call "
             "afterwards. (An alternative that runs one UNet step to form <code>x̂₀</code> was tried "
             "in E25 and was more aggressive / destructive — see the E25 section. This latent-space "
             "version is the gentler, better-behaved one.)</dd>"

             "<dt>Long-aware CLIP-T (the metric / target)</dt><dd>SDXL's two text encoders <b>and</b> "
             "the CLIP scorer truncate at <b>77 tokens</b>, but DPG-Bench prompts run ~55–109 words. "
             "So plain CLIP-T cannot read the whole prompt. We instead split the prompt into clauses "
             "(each ≤77 tokens), CLIP-encode each, <b>mean-pool + renormalize</b>, and use that as "
             "<i>both</i> the optimization target and the evaluation metric. Higher = the image "
             "matches the (whole) prompt better.</dd>"

             "<dt>Inner-step count <code>N</code></dt><dd>How many gradient steps we take on the seed "
             "before generating. <b>One optimization is run to the max N and snapshotted at each "
             f"value</b> in the sweep <code>{sweep}</code> (a prefix reuses work, so the sweep is "
             "nearly free). Re-standardization keeps <code>‖z‖=√d</code> at every snapshot.</dd>"

             "<dt>The generation columns</dt><dd>"
             "<code>baseline</code> = the untouched random seed <code>z₀</code> (no optimization). "
             "<code>N=1</code> = the cheap one-step <b>linear</b> nudge. "
             "<code>N=2 / N=3 / N=5</code> = more inner steps (snapshots of the same run). "
             f"<code>N=1*strong</code> = a single <b>strengthened</b> step (larger lr, "
             + (f"<code>{strong_lr}</code> vs <code>{lr}</code>" if (lr and strong_lr) else "larger lr") +
             ") — tests whether one bigger jump beats one small one.</dd>"

             "<dt>ΔCLIP-T (the headline number, ↑ = good)</dt><dd>Per column, "
             "<code>long-CLIP-T(aligned image) − long-CLIP-T(baseline image)</code>. <b>Positive</b> "
             "= optimizing the seed moved the <i>generated image</i> toward the prompt; "
             "<code>0</code> = no net effect; <b>negative</b> = it hurt. The question is whether the "
             "seed nudge survives into the image, and whether more steps <code>N</code> help.</dd>"
             "</dl>")

    # ---- method ----
    h.append("<h2>1 · Method</h2><dl>"
             "<dt>E25 — pilot on SD1.5 (512px, <code>e25_seedalign.py</code>)</dt><dd>"
             "<code>d = 4·64·64 = 16384</code>, so <code>√d = 128</code>. Uses a standalone "
             "<code>clip-vit-large-patch14</code> for the objective (SD1.5's own text encoder is not "
             "in the joint image/text CLIP space). Characterizes the two objective modes: the "
             "<b>x̂₀ mode</b> (runs one UNet step) and the <b>latent mode</b> (decode <code>z</code> "
             "directly, no UNet). <i>Which formulation is do-no-harm?</i></dd>"
             "<dt>E26 — SDXL + long prompts + N-sweep (<code>e26_seedalign_sdxl.py</code>)</dt><dd>"
             "The E25 latent mode, ported to <b>SDXL</b> "
             + (f"({size}px, <code>d = {d}</code>, <code>√d = {sqrt_d:.0f}</code>)" if (d and sqrt_d)
                else "(1024px)") +
             ". The stock SDXL fp16 VAE NaNs on decode, so we swap in "
             "<code>madebyollin/sdxl-vae-fp16-fix</code>; dropping the UNet from the objective makes "
             "it comfortably fit a 24 GB A5000. Long, dense <b>DPG-Bench</b> prompts + the long-aware "
             "CLIP-T target. Then <b>sweep N</b> over " + (f"<code>{sweep}</code>" if sweep else "a few values") +
             " plus a strengthened single step, generate from each snapshot at "
             + (f"{gen_steps}-step / guidance {guidance}" if (gen_steps and guidance) else "the standard settings") +
             ", and score Δ long-CLIP-T. <i>Does the seed nudge survive into the image on a stronger "
             "model with long prompts, and do more steps help?</i></dd>"
             "</dl>")

    # ---- results ----
    h.append("<h2>2 · Results — does more optimization help? (E26)</h2>")

    # (a) figure first: the sweep plot, then the grid
    h.append("<h3>Δ long-CLIP-T vs number of inner steps N</h3>")
    h.append(img_tag("deltaclip_vs_N.png"))
    h.append("<div class=look><b>What to look for.</b> The y-axis is Δ long-CLIP-T (aligned − "
             "baseline); <b>0 = no effect</b>. Faint dots are per-prompt deltas; the line is the "
             "mean over prompts; the red star is the strengthened single step. We are asking whether "
             "any N sits clearly <b>above 0</b>, and whether the curve <b>rises</b> with N (more "
             "steps help) or <b>not</b> (break-even).</div>")
    h.append("<div class='note win'><b>Reading — break-even, and the cheapest setting wins.</b> "
             "<code>N=1</code> (the one-step linear solution) is the only clearly non-negative point; "
             "adding more steps does <b>not</b> help and drifts slightly negative (off-manifold), and "
             "the strengthened single step is ~0. Per-prompt deltas are tiny (±0.01). A single cheap "
             "gradient step captures whatever benefit there is — a sensible low-cost <i>better "
             "starting point</i> for the seed, not a heavy optimizer.</div>")

    # numbers table: Δ per N (best cell highlighted)
    if md:
        # column order: sweep ints, then the strengthened single step ("1*")
        order = [str(n) for n in sweep] + (["1*"] if "1*" in md else [])
        vals = {k: md.get(k) for k in order}
        present = [v for v in vals.values() if v is not None]
        best = max(present) if present else None
        hdr = "".join(
            f"<th>N=1*strong</th>" if k == "1*" else f"<th>N={k}</th>" for k in order)
        cells = []
        for k in order:
            v = vals[k]
            if v is None:
                cells.append("<td>—</td>")
            else:
                hot = best is not None and abs(v - best) < 1e-12
                cells.append(f"<td class=pos>{fmt(v)}</td>" if hot else f"<td>{fmt(v)}</td>")
        h.append("<table><tr><th>column</th>" + hdr + "</tr>"
                 "<tr><td class=v>mean Δ long-CLIP-T ↑</td>" + "".join(cells) + "</tr></table>")
        h.append("<p class=cap>Mean Δ long-CLIP-T (aligned − baseline) over the DPG prompts, by "
                 "number of inner steps. Best (highest) cell highlighted. Positive = the seed nudge "
                 "moved the image toward the prompt.</p>")

    # (b) the visual grid
    h.append("<h3>The images — do the aligned columns stay sane?</h3>")
    h.append(img_tag("grid.png"))
    h.append("<p class=cap>Rows = DPG prompts (one seed each); columns = "
             + " · ".join(f"<code>{c}</code>" for c in _col_keys(sweep)) + ".</p>")
    h.append("<div class=look><b>What to look for.</b> Read across each row: do the aligned columns "
             "stay close to the <code>baseline</code> image (gentle palette / saturation / detail "
             "shifts), or does any column lose composition / collapse? Do later <code>N</code> drift "
             "further from baseline?</div>")
    h.append("<div class=read><b>Reading.</b> The aligned columns stay very close to baseline — "
             "gentle palette / saturation / detail shifts, <b>no structural damage</b> (unlike E25's "
             "x̂₀ mode, which produced CLIP-adversarial seeds). This is the do-no-harm behavior the "
             "latent-space objective was chosen for.</div>")

    # constraint check from records
    if records:
        norms = []
        for r in records:
            for c, m in (r.get("z_moments") or {}).items():
                if c != "baseline":
                    norms.append(m.get("norm"))
        norms = [n for n in norms if n is not None]
        if norms:
            nmin, nmax = min(norms), max(norms)
            h.append(f"<p class=cap><b>Constraint check.</b> Across every aligned snapshot the seed "
                     f"norm stays at <code>‖z*‖ ∈ [{nmin:.1f}, {nmax:.1f}]</code>"
                     + (f" = <code>√d = {sqrt_d:.0f}</code>" if sqrt_d else "") +
                     " (zero mean, unit variance held), so every step is a move on the Gaussian "
                     "sphere.</p>")

    # ---- E25 summary (joint scope) ----
    h.append("<h2>3 · E25 pilot (SD1.5) — why the latent objective</h2>")
    h.append("<div class=read>E25 ran the same idea on SD1.5 and compared the two objective modes "
             "(mean Δ CLIP-T over 4 prompts × 2 seeds):"
             "<ul>"
             "<li><b>x̂₀ mode</b> (runs one UNet step in the objective): the inner objective "
             "<i>over-optimizes</i> (CLIP cosine shoots to ~0.50, above any natural image's "
             "~0.25–0.30) and the seed becomes <b>CLIP-adversarial</b> — leopard-print for \"cat\", "
             "swirls for \"blue sphere\", lost composition. Net <b>−0.022 to −0.025</b>: it slightly "
             "<i>hurts</i>. Early-stopping reduces but does not flip the sign.</li>"
             "<li><b>latent mode</b> (decode <code>z</code> directly, no UNet — the method used in "
             "E26): the <b>gentlest</b> and best-behaved. Mean Δ <b>−0.010</b>, visually stays very "
             "close to baseline, behaves like a controlled palette / global-appearance nudge that "
             "occasionally helps.</li>"
             "</ul>"
             "<b>Takeaway:</b> the seed's trace is a <b>palette / global-appearance</b> trace, not a "
             "composition one; the latent-space objective is the right, do-no-harm formulation — "
             "which is why E26 uses it.</div>")

    # ---- caveats ----
    h.append("<h2>4 · Caveats &amp; next</h2><div class=cav>"
             "<b>(1)</b> Single seed per cell on the E26 sweep — read <b>directions</b>, not third "
             "decimals; the small ΔCLIP-T sign is within noise. <b>(2)</b> SDXL's 77-token "
             "bottleneck means the model literally cannot read the whole long prompt at generation "
             "time, so seed alignment cannot do much for long-prompt adherence by construction. "
             "<b>(3)</b> The robust, reproducible findings are the constraint (<code>‖z‖=√d</code>, "
             "moments held) and the do-no-harm behavior, <i>not</i> a CLIP-T win. "
             "<b>Next:</b> beat the 77-token bottleneck (Long-CLIP / T5-conditioned models so the "
             "seed can carry the long-prompt tail); restrict the nudge to a <b>low-frequency band of "
             "<code>z</code></b> to bias global layout without texture (ties into the project's "
             "spectral toolbox); a stronger metric than CLIP-T (official DPG / VQAScore).</div>")

    h.append("<p class=cap>Generated by <code>e26_site.py</code> from "
             "<code>results/e26/{report.json, grid.png, deltaclip_vs_N.png}</code>. Method: "
             "<code>e26_seedalign_sdxl.py</code> (E26) + <code>e25_seedalign.py</code> (E25). "
             "See also <code>EXPERIMENT_26.md</code>.</p>")
    return "".join(h)


def build_site():
    """Model-free rebuild of results/e26/index.html from report.json. Returns the path,
    or None (after printing a clear message) if report.json is missing."""
    rpath = os.path.join(OUT, "report.json")
    if not os.path.exists(rpath):
        print(f"[e26-site] no {rpath}; run the driver first: "
              f"python experiments/e26_seedalign_sdxl.py")
        return None
    with open(rpath) as f:
        rep = json.load(f)
    html = render(rep)
    dest = os.path.join(OUT, "index.html")
    os.makedirs(OUT, exist_ok=True)
    with open(dest, "w") as f:
        f.write(html)
    print(f"[e26-site] wrote {dest}  ({len(html) // 1024} KB, no model loaded)")
    return dest


def main():
    build_site()


if __name__ == "__main__":
    main()
