"""E47 -- geodesic phasor-slerp phase whitening vs chord-gamma vs vanilla SDEdit.

E46 whitened the seed phase with a CHORD mix unit((1-g)e^{i ph_z} + g e^{i ph_src}),
which renormalizes a sum of phasors -> variable angular speed and a discontinuous flip
when the two phasors are near-antipodal. E47 replaces it with the constant-angular-velocity
GEODESIC:

    delta = wrap(xi - ph_src) in (-pi, pi]          # shortest signed arc to a white target xi
    ph_t  = ph_src + t * delta                       # rotate a fraction t of the arc
    seed  = ifft( |z| * exp(i ph_t) )                # white amplitude kept (E46: coloring amp -> rainbow)

t=0 -> full source phase (structure), t=1 -> white xi. `t` may be a scalar (geodesic-global)
or a per-frequency field (geodesic-band: keep source phase in the low band [0,cut], whiten
the high band at t_high -> kills the high-freq OOD fringing that held E46's Cfull/gamma
inside the vanilla frontier).

xi is drawn as angle(fft2(real gaussian)) so it is Hermitian; with |z| from a real seed the
spectrum stays Hermitian and ifft(.).real is exact up to the ~1e-6 self-conjugate-bin residue
(same convention as e46_gamma).

Frontier test (the matched-editability comparison E46 never ran): sweep vanilla SDEdit
strength to draw its struct<->clip curve, then check whether any geodesic point sits strictly
NW of it (lower DINO structure distance AND higher CLIP-directional editability).

Run on the cluster (PIE-Bench lives in the /storage HF cache):
    python experiments/e47_geodesic.py --n_per_type 2 --tag sub20
"""
import argparse
import json
import math
import os

import torch
from diffusers import StableDiffusionXLImg2ImgPipeline

import common
from latent_spectral_ops import radial_norm
from piebench import load_piebench
from spectral_ops import _restore_self_conj
from struct_metrics import load_dino, structure_distance, clip_directional, load_clip
from e46_probe1 import sdedit


def geodesic_seed(x0, Fz, xi_phase, t):
    """Constant-angular-velocity slerp of the source phase toward white target `xi_phase`.

    x0: source latent (1,C,H,W). Fz: white seed spectrum fft2(z) (white magnitude + the
    Hermitian self-conjugate bins). t: scalar or (H,W) per-frequency fraction in [0,1].

    An intermediate t lands the self-conjugate bins (DC/Nyquist) off {0,pi}, which breaks
    Hermitian symmetry and leaks a large imaginary part; restoring those 4 bins from the
    (real) white seed keeps ifft(.).real exact. Returns a real latent in x0.dtype."""
    H, W = x0.shape[-2:]
    phi = torch.fft.fft2(x0.float()).angle()
    delta = torch.remainder(xi_phase - phi + math.pi, 2 * math.pi) - math.pi   # (-pi, pi]
    t_ = t if torch.is_tensor(t) else torch.as_tensor(float(t), device=x0.device)
    Fout = torch.polar(Fz.abs(), phi + t_ * delta)
    _restore_self_conj(Fout, Fz, H, W)
    return torch.fft.ifft2(Fout).real.to(x0.dtype)


def band_t_field(H, W, cut, t_high, device):
    """Per-frequency t: 0 inside the low band [0,cut] (keep source phase), t_high outside."""
    r = radial_norm(H, W, device)
    return torch.where(r <= cut, torch.zeros((), device=device),
                       torch.as_tensor(float(t_high), device=device))


def frontier_clip_at(vanilla_pts, struct):
    """Linear-interpolate the vanilla (struct->clip) frontier at a given struct level.
    vanilla_pts: list of (struct, clip). Returns the vanilla clip you'd get at `struct`."""
    pts = sorted(vanilla_pts)
    if struct <= pts[0][0]:
        return pts[0][1]
    if struct >= pts[-1][0]:
        return pts[-1][1]
    for (s0, c0), (s1, c1) in zip(pts, pts[1:]):
        if s0 <= struct <= s1:
            w = (struct - s0) / (s1 - s0 + 1e-12)
            return c0 + w * (c1 - c0)
    return pts[-1][1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_type", type=int, default=2)
    ap.add_argument("--strengths", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8, 0.9])
    ap.add_argument("--ts", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--cut", type=float, default=0.2)
    ap.add_argument("--t_high", type=float, nargs="+", default=[0.5, 1.0])
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guid", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--img_size", type=int, default=1024)
    ap.add_argument("--tag", default="sub")
    args = ap.parse_args()

    out = os.path.join(common.RESULTS, f"e47_{args.tag}")
    os.makedirs(out, exist_ok=True)
    pipe = common.load_pipe()
    img2img = StableDiffusionXLImg2ImgPipeline.from_pipe(pipe)
    img2img.set_progress_bar_config(disable=True)
    dino, clip = load_dino(), load_clip()
    items = load_piebench(args.n_per_type)

    # arm spec: (name, kind, param)
    arms = [(f"van_s{s}", "vanilla", s) for s in args.strengths]
    arms += [(f"geo_t{t}", "global", t) for t in args.ts]
    arms += [(f"band_c{args.cut}_th{th}", "band", th) for th in args.t_high]

    recs, grid_rows, grid_labels = [], [], []
    grid_cols = ["source"] + [a[0] for a in arms]
    for it in items:
        src = it["src_img"].resize((args.img_size, args.img_size))
        x0 = common.encode_img_sdxl(pipe, src)                       # (1,C,H,W) fp16 cuda
        H, W = x0.shape[-2:]
        g = torch.Generator("cuda").manual_seed(args.seed)
        z = torch.randn(x0.shape, generator=g, device="cuda", dtype=x0.dtype)
        Fz = torch.fft.fft2(z.float())
        gx = torch.Generator("cuda").manual_seed(args.seed + 777)
        xi = torch.fft.fft2(torch.randn(x0.shape, generator=gx, device="cuda").float()).angle()

        row = [src]
        for name, kind, p in arms:
            if kind == "vanilla":
                im = sdedit(img2img, src, it["edit_prompt"], p, args.steps, args.guid,
                            torch.Generator("cuda").manual_seed(args.seed))
            else:
                t = p if kind == "global" else band_t_field(H, W, args.cut, p, x0.device)
                seed = geodesic_seed(x0, Fz, xi, t)
                im = common.generate(pipe, it["edit_prompt"], seed, steps=args.steps,
                                     guidance=args.guid)
            sd = structure_distance(dino, im, src)
            cd = clip_directional(clip, src, im, it["src_prompt"], it["edit_prompt"])
            recs.append({"key": it["key"], "arm": name, "kind": kind, "param": p,
                         "edit_type": it["edit_type"], "struct_dist": sd, "clip_dir": cd})
            row.append(im)
        grid_rows.append(row)
        grid_labels.append(it["key"].split("/")[-1][:18])
        print(f"[e47] done {it['key']}", flush=True)

    common.save_grid(grid_rows, grid_labels, grid_cols, os.path.join(out, "grid.png"))
    json.dump(recs, open(os.path.join(out, "scores.json"), "w"), indent=2)
    verdict(recs, arms, out)


def verdict(recs, arms, out):
    """Per-arm subset means + Pareto test vs the vanilla SDEdit frontier."""
    def mean(arm, k):
        v = [r[k] for r in recs if r["arm"] == arm]
        return sum(v) / len(v)

    means = {a[0]: (mean(a[0], "struct_dist"), mean(a[0], "clip_dir")) for a in arms}
    vanilla = [means[a[0]] for a in arms if a[1] == "vanilla"]

    print("\n=== E47 per-arm subset means (struct DOWN / clip_dir UP) ===", flush=True)
    lines = ["# E47 results", "", "arm | struct | clip_dir | beats_vanilla_frontier",
             "--- | --- | --- | ---"]
    winners = []
    for name, kind, _ in arms:
        s, c = means[name]
        beat = ""
        if kind != "vanilla":
            # NW of the frontier: at this arm's struct level, its editability beats what
            # vanilla achieves at the same structure (i.e. the point sits above the curve).
            won = c > frontier_clip_at(vanilla, s)
            beat = "YES" if won else "no"
            if won:
                winners.append((name, s, c))
        print(f"  {name:16s} struct={s:.4f}  clip_dir={c:+.4f}  {beat}", flush=True)
        lines.append(f"{name} | {s:.4f} | {c:+.4f} | {beat}")

    print(f"\nvanilla frontier: {sorted(vanilla)}", flush=True)
    print(f"WINNERS (geodesic point NW of vanilla frontier): {winners or 'NONE'}", flush=True)
    lines += ["", f"vanilla frontier: {sorted(vanilla)}",
              f"winners: {winners or 'NONE'}"]
    open(os.path.join(out, "verdict.md"), "w").write("\n".join(lines) + "\n")
    print(f"\nartifacts: {out}/grid.png  {out}/scores.json  {out}/verdict.md", flush=True)


if __name__ == "__main__":
    main()
