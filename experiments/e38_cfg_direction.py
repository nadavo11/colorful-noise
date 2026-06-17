"""E38: the FREQUENCY DIRECTION of CFG (Flux-dev guidance) -- magnitude AND
phase, along the whole denoising trajectory.

Motivation
----------
E7 (correlational) compared FLUX.1-dev output latents at cfg=1 vs cfg=3.5 and
found they differ mainly in POWER (low-freq pumped ~3x, steeper slope) while
the *marginal* phase statistics (per-coefficient phase histogram, phase-mag
correlation, cross-seed coherence) were indistinguishable. E8 then showed the
power difference is causal (clamping it back removes guidance's look).

But "marginal phase histogram is uniform at every cfg" does NOT mean guidance
leaves phase untouched. With the SAME prompt+seed the initial noise is
identical (guidance is only an embedding input), so the latents at two cfg
values are POINT-WISE comparable. The question E38 asks is the *paired* one:
does raising cfg rotate each Fourier coefficient's phase in a COHERENT,
band-specific way (a real "phase direction") -- which a marginal histogram
would completely miss -- or are the per-coefficient rotations random?

We answer this for magnitude and phase, at three cfg points (1.0, 3.5, 7.0),
across the same 10 prompts, and we record the FULL per-step latent trajectory
so the change can be tracked early -> late in denoising.

Definitions (all per radial frequency band b, see band_index_map)
  log-power(b, cfg)   = log mean |F|^2 over the coefficients in band b
  magnitude direction = d log-power(b) / d cfg  (least-squares slope over the
                        three cfg points). >0: guidance amplifies that band.
  paired phase ratio  r_k = (F_b[k]/|F_b[k]|) * conj(F_a[k]/|F_a[k]|)  -- the
                        unit complex rotation coefficient k undergoes going
                        from cfg=a to cfg=b (same prompt, seed, step, channel).
  phase coherence(b)  = |sum_k w_k r_k| / sum_k w_k,  w_k = |F_a[k]|*|F_b[k]|
                        (magnitude-weighted). 1 = every coefficient in the band
                        rotates by the SAME angle (a coherent phase direction);
                        ~1/sqrt(N_b) = random (no direction). N_b printed so the
                        reader can compare to the random null.
  mean rotation(b)    = angle(sum_k w_k r_k)  -- the dominant rotation angle.

Parts (--part, comma list)
  gen      -- generate `n_prompts` prompts x cfgs (--cfgs), same seed per
              prompt across cfgs. Save final image PNG + the per-step unpacked
              latent trajectory (steps,16,128,128) fp16 to results/e38/traj/.
  analyze  -- CPU-only. Load trajectories, compute magnitude direction + phase
              coherence/rotation per band, per cfg-pair, binned early/mid/late.
              Write summary.json, plots (PNG) and a self-contained index.html.

    # cluster (1 GPU):
    python e38_cfg_direction.py --part gen --n_prompts 10 --cfgs 1.0,3.5,7.0
    # anywhere (no GPU):
    python e38_cfg_direction.py --part analyze
"""
import argparse
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS
from spectral_ops import band_index_map

OUT = os.path.join(RESULTS, "e38")
TRAJ = os.path.join(OUT, "traj")
N_CH, H, W = 16, 128, 128
N_BINS = 24
DEFAULT_CFGS = [1.0, 3.5, 7.0]   # 3.5 = Flux-dev default guidance

# Fixed, diverse "random" prompt set (same across all cfgs so the latents are
# point-wise comparable). Deliberately varied: animals, scenes, objects, people.
PROMPTS = [
    "a red fox sitting in a snowy forest at dawn",
    "a bustling tokyo street at night with neon signs",
    "a still life of lemons and a ceramic jug on a table",
    "a lighthouse on a rocky cliff during a storm",
    "a portrait of an old fisherman with a weathered face",
    "a hot air balloon over a patchwork of green fields",
    "a steaming bowl of ramen with chopsticks",
    "a vintage motorcycle parked by a brick wall",
    "a coral reef teeming with tropical fish",
    "a wooden cabin in a pine forest under the milky way",
]


# ---------------------------------------------------------------------------
# Generation (GPU)
# ---------------------------------------------------------------------------

class RecordTraj:
    """Step-end callback: store the unpacked latent (16,H,W) fp16 cpu at every
    step. No modification (returns {})."""

    def __init__(self, pipe, steps):
        self.pipe, self.size = pipe, None
        self.lat = [None] * steps

    def __call__(self, p, i, t, kw):
        from diffusers import FluxPipeline
        lat = FluxPipeline._unpack_latents(kw["latents"], 1024, 1024,
                                           p.vae_scale_factor)
        self.lat[i] = lat[0].float().cpu().half()   # (16,H,W) fp16
        return {}


def gen(args):
    from e7_flux_phase import load_flux, SIZE
    os.makedirs(TRAJ, exist_ok=True)
    os.makedirs(os.path.join(OUT, "images"), exist_ok=True)
    cfgs = [float(c) for c in args.cfgs.split(",")]
    pipe = load_flux(args.mem)
    prompts = PROMPTS[:args.n_prompts]
    meta = {"prompts": prompts, "cfgs": cfgs, "steps": args.steps,
            "seed_base": args.seed, "size": SIZE}
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    for pi, prompt in enumerate(prompts):
        seed = args.seed + pi                       # same seed across cfgs
        for cfg in cfgs:
            tp = os.path.join(TRAJ, f"p{pi}_cfg{cfg:g}.pt")
            ip = os.path.join(OUT, "images", f"p{pi}_cfg{cfg:g}.png")
            if os.path.exists(tp) and os.path.exists(ip):
                print(f"[e38] skip p{pi} cfg{cfg:g} (exists)", flush=True)
                continue
            rec = RecordTraj(pipe, args.steps)
            img = pipe(prompt=prompt, height=SIZE, width=SIZE,
                       guidance_scale=cfg, num_inference_steps=args.steps,
                       generator=torch.Generator("cuda").manual_seed(seed),
                       callback_on_step_end=rec).images[0]
            img.save(ip)
            torch.save(torch.stack(rec.lat), tp)    # (steps,16,H,W) fp16
            print(f"[e38] p{pi} cfg{cfg:g} done -> {tp}", flush=True)
    print("[e38] gen complete", flush=True)


# ---------------------------------------------------------------------------
# Analysis (CPU)
# ---------------------------------------------------------------------------

def _band_logpower(F2_chmean, idx_flat, counts, n_bins):
    """mean |F|^2 per band over (channels, coefficients) -> (n_bins,) log."""
    out = torch.zeros(n_bins).scatter_add_(0, idx_flat, F2_chmean)
    return torch.log((out / counts.clamp(min=1)).clamp(min=1e-20))


def _phase_dir(Fa, Fb, idx_flat, counts, n_bins):
    """Magnitude-weighted paired phase coherence + mean rotation per band for
    one (channel-summed) cfg pair. Fa, Fb: (C,H,W) complex. Returns
    (coherence (n_bins,), mean_rot (n_bins,))."""
    a = Fa.reshape(Fa.shape[0], -1)
    b = Fb.reshape(Fb.shape[0], -1)
    ua = a / a.abs().clamp(min=1e-20)
    ub = b / b.abs().clamp(min=1e-20)
    w = (a.abs() * b.abs())                      # (C, HW) magnitude weight
    r = ub * ua.conj() * w                       # weighted unit rotation
    # accumulate over channels and coefficients into bands
    idx = idx_flat.unsqueeze(0).expand(r.shape[0], -1).reshape(-1)
    rr = r.reshape(-1)
    ww = w.reshape(-1)
    Rre = torch.zeros(n_bins).scatter_add_(0, idx, rr.real)
    Rim = torch.zeros(n_bins).scatter_add_(0, idx, rr.imag)
    Wsum = torch.zeros(n_bins).scatter_add_(0, idx, ww)
    R = torch.complex(Rre, Rim)
    coh = R.abs() / Wsum.clamp(min=1e-20)
    rot = torch.angle(R)
    return coh, rot


def analyze(args):
    meta = json.load(open(os.path.join(OUT, "meta.json")))
    cfgs = meta["cfgs"]
    steps = meta["steps"]
    prompts = meta["prompts"]
    n_bins = N_BINS
    idx_map = band_index_map(H, W, n_bins, "cpu")
    idx_flat = idx_map.flatten()
    counts = torch.zeros(n_bins).scatter_add_(
        0, idx_flat, torch.ones_like(idx_flat, dtype=torch.float))
    cfg_t = torch.tensor(cfgs)
    cfg_c = cfg_t - cfg_t.mean()
    # cfg pairs to report phase direction for
    pairs = [(0, len(cfgs) - 1)]                 # lowest -> highest
    if len(cfgs) >= 3:
        pairs = [(0, 1), (1, len(cfgs) - 1), (0, len(cfgs) - 1)]
    thirds = [("early", 0, steps // 3),
              ("mid", steps // 3, 2 * steps // 3),
              ("late", 2 * steps // 3, steps)]

    # accumulators over prompts: magnitude slope per (third, band);
    # phase coherence/rotation per (pair, third, band)
    n_p = len(prompts)
    mag_slope = {t[0]: [] for t in thirds}
    phase_coh = {(a, b): {t[0]: [] for t in thirds} for a, b in pairs}
    phase_rot = {(a, b): {t[0]: [] for t in thirds} for a, b in pairs}

    for pi in range(n_p):
        # load trajectories for all cfgs: list of (steps,16,H,W) complex FFTs
        trajs = []
        ok = True
        for cfg in cfgs:
            tp = os.path.join(TRAJ, f"p{pi}_cfg{cfg:g}.pt")
            if not os.path.exists(tp):
                print(f"[e38] WARN missing {tp}; skip prompt {pi}", flush=True)
                ok = False
                break
            lat = torch.load(tp, weights_only=True).float()     # (steps,16,H,W)
            trajs.append(torch.fft.fft2(lat))                   # complex
        if not ok:
            continue
        for tname, lo, hi in thirds:
            # ---- magnitude direction: logpower(band, cfg) averaged over steps
            logP = []   # per cfg: (n_bins,)
            for F in trajs:
                # mean over steps and channels of |F|^2 per coefficient
                F2cm = (F[lo:hi].abs() ** 2).mean(dim=(0, 1)).reshape(-1)  # (HW,)
                logP.append(_band_logpower(F2cm, idx_flat, counts, n_bins))
            logP = torch.stack(logP)                            # (n_cfg, n_bins)
            # least-squares slope of logP vs cfg per band
            slope = (cfg_c[:, None] * (logP - logP.mean(0, keepdim=True))).sum(0) \
                / (cfg_c ** 2).sum()
            mag_slope[tname].append(slope)
            # ---- phase direction per pair (step-averaged complex accumulation)
            for (a, b) in pairs:
                cohs, rots, ws = [], [], []
                for s in range(lo, hi):
                    coh, rot = _phase_dir(trajs[a][s], trajs[b][s],
                                          idx_flat, counts, n_bins)
                    cohs.append(coh); rots.append(rot)
                phase_coh[(a, b)][tname].append(torch.stack(cohs).mean(0))
                phase_rot[(a, b)][tname].append(torch.stack(rots).mean(0))

    def agg(lst):
        s = torch.stack(lst)
        return s.mean(0), s.std(0)

    centers = (torch.arange(n_bins) + 0.5) / n_bins
    summary = {"meta": meta, "n_bins": n_bins,
               "band_centers_norm": centers.tolist(),
               "band_coeff_counts": counts.int().tolist(),
               "random_phase_null_per_band": (1.0 / counts.clamp(min=1).sqrt()).tolist(),
               "magnitude_direction": {}, "phase_direction": {}}
    for tname, _, _ in thirds:
        if not mag_slope[tname]:
            continue
        m, sd = agg(mag_slope[tname])
        summary["magnitude_direction"][tname] = {
            "dlogpower_dcfg_mean": m.tolist(), "std": sd.tolist()}
    for (a, b) in pairs:
        key = f"cfg{cfgs[a]:g}->cfg{cfgs[b]:g}"
        summary["phase_direction"][key] = {}
        for tname, _, _ in thirds:
            if not phase_coh[(a, b)][tname]:
                continue
            cm, cs = agg(phase_coh[(a, b)][tname])
            rm, rs = agg(phase_rot[(a, b)][tname])
            summary["phase_direction"][key][tname] = {
                "coherence_mean": cm.tolist(), "coherence_std": cs.tolist(),
                "mean_rotation_rad": rm.tolist(), "rotation_std": rs.tolist()}
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    _plots(summary, cfgs, pairs)
    _html(summary, cfgs, pairs)
    _print_headline(summary, cfgs, pairs)
    print("[e38] analyze complete ->", OUT, flush=True)


def _plots(summary, cfgs, pairs):
    centers = summary["band_centers_norm"]
    null = summary["random_phase_null_per_band"]
    # magnitude direction
    plt.figure(figsize=(7, 4))
    for tname, d in summary["magnitude_direction"].items():
        plt.plot(centers, d["dlogpower_dcfg_mean"], marker=".", label=tname)
    plt.axhline(0, color="k", lw=0.5)
    plt.xlabel("normalized radial frequency"); plt.ylabel("d log-power / d cfg")
    plt.title("E38 magnitude direction of CFG"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "mag_direction.png"), dpi=110)
    plt.close()
    # phase coherence
    plt.figure(figsize=(7, 4))
    for (a, b) in pairs:
        key = f"cfg{cfgs[a]:g}->cfg{cfgs[b]:g}"
        d = summary["phase_direction"].get(key, {})
        if "late" in d:
            plt.plot(centers, d["late"]["coherence_mean"], marker=".", label=key + " (late)")
    plt.plot(centers, null, "k--", lw=1, label="random null ~1/sqrt(N)")
    plt.xlabel("normalized radial frequency"); plt.ylabel("phase coherence")
    plt.title("E38 phase direction of CFG (coherence)"); plt.legend()
    plt.ylim(0, 1)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "phase_coherence.png"), dpi=110)
    plt.close()


def _print_headline(summary, cfgs, pairs):
    md = summary["magnitude_direction"].get("late")
    if md:
        m = torch.tensor(md["dlogpower_dcfg_mean"])
        print(f"[e38] mag direction (late): low-band d logP/dcfg={m[:4].mean():+.3f} "
              f"high-band={m[-6:].mean():+.3f}", flush=True)
    key = f"cfg{cfgs[0]:g}->cfg{cfgs[-1]:g}"
    pd = summary["phase_direction"].get(key, {}).get("late")
    null = torch.tensor(summary["random_phase_null_per_band"])
    if pd:
        c = torch.tensor(pd["coherence_mean"])
        print(f"[e38] phase {key} (late): low-band coh={c[:4].mean():.3f} "
              f"(null~{null[:4].mean():.3f}) high-band coh={c[-6:].mean():.3f} "
              f"(null~{null[-6:].mean():.3f})", flush=True)


def _html(summary, cfgs, pairs):
    meta = summary["meta"]
    rows = "".join(
        f"<tr><td>{c:g}</td><td>{'Flux-dev default' if abs(c-3.5)<1e-6 else ('de-saturated reference' if abs(c-1.0)<1e-6 else 'over-guided')}</td></tr>"
        for c in cfgs)
    pr = "".join(f"<li>{p}</li>" for p in meta["prompts"])
    html = f"""<!doctype html><html><head><meta charset=utf-8>
<title>E38 — the frequency direction of CFG</title>
<style>body{{font-family:system-ui,sans-serif;max-width:900px;margin:2rem auto;line-height:1.5;color:#222}}
code{{background:#f3f3f3;padding:1px 4px;border-radius:3px}}img{{max-width:100%;border:1px solid #ddd}}
table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:4px 8px}}</style></head><body>
<h1>E38 — the frequency direction of CFG (Flux-dev guidance)</h1>
<p><b>Question.</b> Raising classifier-free-guidance strength changes a Flux-dev
image. <i>Where in the frequency spectrum</i> does that change live, in
<b>magnitude</b> and in <b>phase</b>, and how does it build up across the
denoising trajectory? E7 showed cfg moves <b>power</b> (low frequencies pumped)
while the marginal phase <i>histogram</i> stays uniform. E38 adds the missing
piece: because the same prompt+seed gives an identical starting latent, we can
measure the <b>paired</b> per-coefficient phase rotation between cfg levels — a
coherent rotation is a real "phase direction" that a histogram cannot see.</p>

<h2>Setup</h2>
<p>{len(meta['prompts'])} prompts, identical seed per prompt across cfg levels
({meta['steps']} steps, {meta['size']}px, FLUX.1-dev). cfg levels:</p>
<table><tr><th>guidance_scale</th><th>regime</th></tr>{rows}</table>
<p>Per generation we store the full per-step unpacked latent trajectory
(steps×16×128×128) and FFT it offline.</p>

<h2>Definitions</h2>
<ul>
<li><b>radial band</b> — coefficients are binned by distance from the FFT
origin into {summary['n_bins']} rings; band 0 ≈ DC/global, last band ≈ finest
detail.</li>
<li><b>magnitude direction</b> <code>d log-power / d cfg</code> — least-squares
slope of each band's log mean power against cfg over the three cfg points. &gt;0
means guidance amplifies that band.</li>
<li><b>phase coherence(band)</b> — magnitude-weighted resultant length of the
per-coefficient rotations <code>F_high/F_low</code> in that band. <b>1</b> = every
coefficient rotates by the same angle (a coherent phase direction); a value near
the printed <b>random null ~1/√N</b> means the rotations are directionless
(consistent with E7's uniform marginal phase).</li>
<li><b>mean rotation</b> — the dominant signed rotation angle of a band.</li>
<li><b>early/mid/late</b> — the {meta['steps']} steps split into thirds, to see
when the direction emerges.</li>
</ul>

<h2>Results</h2>
<img src="mag_direction.png"><br><img src="phase_coherence.png">
<p>See <code>summary.json</code> for the full per-band arrays (mean ± std over
prompts), band coefficient counts and the random-phase null per band.</p>

<h2>Prompts</h2><ol>{pr}</ol>
</body></html>"""
    with open(os.path.join(OUT, "index.html"), "w") as f:
        f.write(html)


def main(args):
    os.makedirs(OUT, exist_ok=True)
    for part in filter(None, (p.strip() for p in args.part.split(","))):
        if part == "gen":
            gen(args)
        elif part == "analyze":
            analyze(args)
        else:
            raise ValueError(f"unknown part {part}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="gen,analyze")
    ap.add_argument("--n_prompts", type=int, default=10)
    ap.add_argument("--cfgs", default="1.0,3.5,7.0")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--mem", default="bnb4", choices=["bnb4", "seq_offload"])
    main(ap.parse_args())
