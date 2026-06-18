"""E39: spectral band-AdaIN -- non-interactive demos for the soft-band frequency operator.

The operator itself lives in spectral_adain.py; the interactive demos (latent mixing,
in-sampler x0 correction, real-image SDEdit) are a tab in spectral_demo.py. This script
holds the two demos that do not fit a Gradio tab:

  demo0  PIXEL SANITY (no model). spectral_adain on two real RGB images: low band from B,
         high band + phase from A. Catches FFT / ringing / reality bugs before any FLUX
         dynamics. Prints the imaginary-residue reality check.

  demo3  LEARNED SCHEDULE. Fit a tiny BandSchedule table {g_k(t), b_k(t)} so a low-guidance
         velocity is reshaped, per band and per timestep, into the high-guidance (distilled-
         CFG) velocity -- "distill the frequency correction" as a few-hundred-parameter
         table. The fitted g/b heatmaps are the per-(band,time) frequency-shaping schedule;
         driving cells toward attenuation is the candidate concept-erasure probe.

Usage:
  python experiments/e39_spectral_adain.py demo0 A.png B.png [--out out.png]
  python experiments/e39_spectral_adain.py demo3 [--bands 4] [--tbins 4] [--steps 200]
"""
import argparse
import os
import sys

import torch
import torch.fft as fft

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spectral_adain import soft_band_masks, band_moments, spectral_adain, BandSchedule
from spectral_ops import _restore_self_conj


# ---------------------------------------------------------------------------
# Demo 0 -- pixel sanity (no model)
# ---------------------------------------------------------------------------

def _load_rgb(path, size=None):
    from PIL import Image
    import numpy as np
    if not os.path.exists(path):
        sys.exit(f"[e39] missing image: {path}")
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize(size)
    x = torch.from_numpy(np.asarray(img).copy()).float() / 255.0
    return x.permute(2, 0, 1)[None]                     # (1,3,H,W)


def _imag_residue(A, B, M):
    """The imaginary energy the operator discards (must be ~0 -> output is real).
    Re-derives the reassembled complex spectrum independently of spectral_adain."""
    V = fft.fft2(A.float(), norm="ortho")
    magV, phase = V.abs(), V.angle()
    muc, sigc = band_moments(magV, M)
    srcs = [B, A]
    per = []
    for k in range(M.shape[0]):
        Sk = fft.fft2(srcs[k].float(), norm="ortho").abs()
        mus, sigs = band_moments(Sk, M[k:k + 1])
        mus, sigs = mus[0], sigs[0]
        nrm = (magV - muc[k][..., None, None]) / (sigc[k][..., None, None] + 1e-6)
        per.append((sigs[..., None, None] * nrm + mus[..., None, None]).clamp_min(0))
    Fmix = torch.polar(sum(M[k] * per[k] for k in range(M.shape[0])), phase)
    _restore_self_conj(Fmix, V, Fmix.shape[-2], Fmix.shape[-1])
    return float(fft.ifft2(Fmix, norm="ortho").imag.abs().max())


def demo0(args):
    from PIL import Image
    import numpy as np
    A = _load_rgb(args.A)
    H, W = A.shape[-2:]
    B = _load_rgb(args.B, size=(W, H))                  # PIL size = (W, H)
    M = soft_band_masks(H, W, [0.0, 0.3], [0.12, 0.2], device="cpu")  # low, high
    out = spectral_adain(A, sources=[B, A], M=M)        # low band from B, high+phase from A

    res = _imag_residue(A, B, M)
    print(f"[e39] imag residue {res:.2e}  (reality check, want < 1e-3)")
    assert res < 1e-3, "operator output is not real -- bands not radially symmetric?"
    assert torch.isfinite(out).all()
    print(f"[e39] partition-of-unity sum range "
          f"[{float(M.sum(0).min()):.4f}, {float(M.sum(0).max()):.4f}]")
    print(f"[e39] output range [{float(out.min()):.3f}, {float(out.max()):.3f}]; "
          f"changed from A by {float((out - A).abs().mean()):.4f} (mean abs)")

    def to_pil(x):
        arr = (x[0].clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy()
        return Image.fromarray(arr)

    strip = Image.fromarray(np.concatenate(
        [np.asarray(to_pil(A)), np.asarray(to_pil(B)), np.asarray(to_pil(out))], axis=1))
    strip.save(args.out)
    print(f"[e39] saved A | B | adain(A; low<-B)  ->  {args.out}")


# ---------------------------------------------------------------------------
# Demo 3 -- learned per-band/per-time schedule (needs FLUX + GPU)
# ---------------------------------------------------------------------------

def _band_centers_widths(n):
    """n evenly spaced radial-band centers in [0,1] with overlapping widths."""
    centers = [i / max(n - 1, 1) for i in range(n)]
    w = max(1.0 / max(n - 1, 1), 0.12)
    return centers, [w] * n


def demo3(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from e7_flux_phase import load_flux, flux_generate
    from e31_flowedit_freq import flux_velocity, flux_sigmas, _gids, pack, unpack

    pipe = load_flux(mem="bnb4")
    pe, ppe, _ = pipe.encode_prompt(prompt=args.prompt, prompt_2=args.prompt, device="cuda",
                                    num_images_per_prompt=1, max_sequence_length=512)
    # one clean latent to noise; sample a few sigmas across the trajectory
    _, x0 = flux_generate(pipe, args.prompt, seed=0, guidance=args.guidance, steps=args.gen_steps)
    x0 = x0.cuda()
    sig = flux_sigmas(pipe, args.gen_steps)
    idxs = torch.linspace(0, len(sig) - 2, args.batch).round().long().tolist()
    g1, gW = _gids(pipe, 1.0), _gids(pipe, args.guidance)
    gen = torch.Generator("cuda").manual_seed(0)

    batch = []                                          # (v_lowg, v_ref, sigma) unpacked fp32
    for i in idxs:
        s = float(sig[i])
        eps = torch.randn((1, 16, 128, 128), generator=gen, device="cuda")
        x_t = pack(pipe, (1 - s) * x0 + s * eps)
        v_lowg = unpack(pipe, flux_velocity(pipe, x_t, s, pe, ppe, g1)).float()
        v_ref = unpack(pipe, flux_velocity(pipe, x_t, s, pe, ppe, gW)).float()
        batch.append((v_lowg, v_ref, s))

    centers, widths = _band_centers_widths(args.bands)
    M = soft_band_masks(128, 128, centers, widths, device="cuda")
    sch = BandSchedule(args.bands, args.tbins).cuda()
    opt = torch.optim.Adam(sch.parameters(), lr=args.lr)
    print(f"[e39] demo3: fitting {args.bands}x{args.tbins} table on {len(batch)} (x_t, sigma) "
          f"pairs (v@g=1 -> v@g={args.guidance})", flush=True)
    for step in range(args.steps):
        opt.zero_grad()
        loss = sum(((sch(vw, M, s) - vref) ** 2).mean() for vw, vref, s in batch) / len(batch)
        loss.backward()
        opt.step()
        if step % 20 == 0 or step == args.steps - 1:
            print(f"  step {step:4d}  loss {float(loss):.6f}", flush=True)

    g, b = sch.g.detach().cpu(), sch.b.detach().cpu()
    print(f"[e39] g range [{float(g.min()):.3f}, {float(g.max()):.3f}]  "
          f"b range [{float(b.min()):.3f}, {float(b.max()):.3f}]")
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    for a, tbl, name in ((ax[0], g, "g (gain)"), (ax[1], b, "b (bias)")):
        im = a.imshow(tbl, aspect="auto", cmap="coolwarm")
        a.set_title(name); a.set_xlabel("t bin (0=noisy)"); a.set_ylabel("band (0=low)")
        fig.colorbar(im, ax=a)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"[e39] saved schedule heatmaps -> {args.out}")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d0 = sub.add_parser("demo0", help="pixel sanity (no model)")
    d0.add_argument("A"); d0.add_argument("B")
    d0.add_argument("--out", default="e39_demo0.png")
    d0.set_defaults(func=demo0)

    d3 = sub.add_parser("demo3", help="fit the learned per-band/per-time schedule (FLUX)")
    d3.add_argument("--prompt", default="a fluffy orange tabby cat sitting on a windowsill")
    d3.add_argument("--bands", type=int, default=4)
    d3.add_argument("--tbins", type=int, default=4)
    d3.add_argument("--batch", type=int, default=4, help="number of (x_t, sigma) samples")
    d3.add_argument("--gen-steps", type=int, default=16, dest="gen_steps")
    d3.add_argument("--steps", type=int, default=200, help="fit iterations")
    d3.add_argument("--lr", type=float, default=0.05)
    d3.add_argument("--guidance", type=float, default=3.5)
    d3.add_argument("--out", default="e39_demo3_schedule.png")
    d3.set_defaults(func=demo3)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
