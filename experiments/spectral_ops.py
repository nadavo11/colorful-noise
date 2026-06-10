"""Spectral operations for exploring noise-PSD manipulation in diffusion latents.

Extends the Colorful-Noise (Cohen et al., SIGGRAPH 2026) frequency swap with:
  - colored noise generation (PSD ~ f^beta)
  - a generalized low-band conditioning that decouples phase / magnitude / DC
  - radial PSD measurement + the paper's band-wise whiteness metric
  - FFT phase surgery: Hermitian uniform-phase sampling, phase quantization
    with optional level omission (E6)
"""
import math

import torch


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def radial_bins(H, W, device):
    """Normalized radial frequency grid (DC at [0,0], unshifted)."""
    fy = torch.fft.fftfreq(H, device=device)
    fx = torch.fft.fftfreq(W, device=device)
    return torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)  # max ~ sqrt(0.5)


def radial_psd(x, n_bins=24):
    """Radially averaged power spectral density.

    x: (B, C, H, W) real tensor.
    Returns (centers, psd) where psd is (C, n_bins), averaged over batch.
    """
    B, C, H, W = x.shape
    power = torch.fft.fft2(x.float()).abs() ** 2 / (H * W)
    rr = radial_bins(H, W, x.device)
    edges = torch.linspace(0, rr.max() + 1e-6, n_bins + 1, device=x.device)
    idx = torch.bucketize(rr.flatten(), edges) - 1
    idx = idx.clamp(0, n_bins - 1)
    psd = torch.zeros(C, n_bins, device=x.device)
    counts = torch.zeros(n_bins, device=x.device).scatter_add_(
        0, idx, torch.ones_like(idx, dtype=torch.float))
    flat = power.mean(0).reshape(C, -1)  # batch mean
    for c in range(C):
        psd[c] = torch.zeros(n_bins, device=x.device).scatter_add_(0, idx, flat[c])
    psd = psd / counts.clamp(min=1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers.cpu(), psd.cpu()


def whiteness(x, n_bins=24):
    """Paper-style whiteness: std of band-wise power across bands, per channel,
    normalized by mean band power (0 = perfectly flat/white)."""
    _, psd = radial_psd(x, n_bins)
    return (psd.std(dim=1) / psd.mean(dim=1)).tolist()


# ---------------------------------------------------------------------------
# Colored noise (E1)
# ---------------------------------------------------------------------------

def colored_noise(shape, beta, device="cuda", normalize="global", generator=None):
    """Gaussian noise with PSD ~ f^beta.

    beta: -2 red/brown, -1 pink, 0 white, +1 blue, +2 violet.
    normalize: 'global' rescales each sample to zero mean / unit variance,
               'none' leaves raw filtered noise.
    """
    B, C, H, W = shape
    white = torch.randn(shape, device=device, generator=generator)
    if beta == 0 and normalize == "global":
        return white
    fft = torch.fft.fft2(white)
    rr = radial_bins(H, W, device)
    r_min = rr[rr > 0].min()
    filt = rr.clamp(min=r_min) ** (beta / 2)  # amplitude ~ f^(beta/2) => PSD ~ f^beta
    x = torch.fft.ifft2(fft * filt[None, None]).real
    if normalize == "global":
        x = (x - x.mean(dim=(1, 2, 3), keepdim=True)) / x.std(dim=(1, 2, 3), keepdim=True)
    return x


# ---------------------------------------------------------------------------
# Generalized low-band conditioning (E2)
# ---------------------------------------------------------------------------

def paper_low_mask(H, W, p, device):
    """Exact reproduction of the paper's radial-percentile mask (fftshifted
    coordinates, lowest p fraction of bins by area)."""
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing="ij",
    )
    rr = torch.sqrt(xx ** 2 + yy ** 2)
    cutoff = torch.quantile(rr.flatten(), p)
    return (rr <= cutoff).float()


def condition_latent(noise, img_latent, p=0.015, gamma=0.05,
                     phase="image", mag="image", dc="image", mag_scale=1.0):
    """Generalized version of fft_radial_frequency_swap.

    Inside the lowest-p radial band, builds coefficients M * exp(i*phi) from:
      phase: 'image' | 'noise'   -- where the spatial arrangement comes from
      mag:   'image' (gamma-scaled, the paper's peak) | 'noise' (white PSD)
      dc:    'image' (gamma-scaled) | 'noise' | 'zero' -- per-channel mean

    (phase='image', mag='image', dc='image')  == the paper's method.
    (phase='image', mag='noise', dc='noise')  == PSD-whitened conditioning.
    (phase='noise', mag='noise', dc='image')  == DC-only (channel means).
    """
    assert noise.shape == img_latent.shape
    B, C, H, W = noise.shape
    fft_n = torch.fft.fftshift(torch.fft.fft2(noise.float()), dim=(-2, -1))
    fft_i = torch.fft.fftshift(torch.fft.fft2(img_latent.float()), dim=(-2, -1))

    low = paper_low_mask(H, W, p, noise.device)[None, None]

    M = mag_scale * (gamma * fft_i.abs() if mag == "image" else fft_n.abs())
    phi = fft_i.angle() if phase == "image" else fft_n.angle()
    low_coeffs = M * torch.exp(1j * phi)

    mix = low_coeffs * low + fft_n * (1.0 - low)

    # DC bin sits at the fftshift center
    cy, cx = H // 2, W // 2
    if dc == "image":
        mix[..., cy, cx] = gamma * fft_i[..., cy, cx]
    elif dc == "noise":
        mix[..., cy, cx] = fft_n[..., cy, cx]
    elif dc == "zero":
        mix[..., cy, cx] = 0.0

    mix = torch.fft.ifftshift(mix, dim=(-2, -1))
    return torch.fft.ifft2(mix).real


# Named E2 conditions: (phase, mag, dc)
E2_CONDITIONS = {
    "white":      ("noise", "noise", "noise"),  # control = plain white noise
    "paper":      ("image", "image", "image"),  # Cohen et al.
    "paper_nodc": ("image", "image", "noise"),
    "phase_only": ("image", "noise", "noise"),  # USER: whitened PSD, image phase
    "phase_dc":   ("image", "noise", "image"),
    "mag_only":   ("noise", "image", "noise"),  # spectral shape, no layout
    "mag_dc":     ("noise", "image", "image"),
    "dc_only":    ("noise", "noise", "image"),  # channel means alone
}


# ---------------------------------------------------------------------------
# FFT phase surgery (E6)
# ---------------------------------------------------------------------------

def random_hermitian_phase(C, H, W, device="cuda", generator=None):
    """Uniform iid phase field with exact Hermitian (conjugate) symmetry.

    The FFT phase of a real white-Gaussian field is exactly uniform on
    (-pi, pi] with the required phi(-f) = -phi(f) antisymmetry, so the
    simplest correct construction is to take the angle of one.
    Returns a (1, C, H, W) tensor (unshifted, DC at [0, 0]).
    """
    g = torch.randn(1, C, H, W, device=device, generator=generator)
    return torch.fft.fft2(g).angle()


def quantize_phase(x, k, omit=None, mode="zero", preserve_dc=True,
                   return_stats=False):
    """Quantize the FFT phase of x to k uniform levels, keeping magnitudes.

    Works unshifted (fft2 convention). torch.round of the antisymmetric
    phase stays antisymmetric (half-to-even is an odd function), so the
    quantized spectrum remains Hermitian and ifft2(.).real discards only
    ~1e-7 residue.

    omit: int level index -- zero every bin whose quantized level falls in
          {omit % k, (-omit) % k}; conjugate +/- pairs are zeroed together so
          the drop mask is Hermitian-symmetric and the ifft stays real. For
          k=8 the distinct pairs are levels {0, 1, 2, 3, 4(=pi)}; 0 and 4 are
          self-conjugate singletons.
    mode: 'zero' leaves the power hole (std drops ~ sqrt(kept_fraction));
          'renorm' rescales the kept non-DC magnitudes per channel by
          sqrt(P_total / P_remaining) (Parseval) so unit variance is
          restored and only the phase hole differs from the control.
    preserve_dc: restore the original DC bin and the 3 real Nyquist bins
          after quantization/omission -- E0 showed the model is acutely
          sensitive to the DC level, so it is held fixed by default.
    k=None: early-return x unchanged (the k=inf control).
    """
    if k is None:
        if return_stats:
            return x, {"kept_fraction": 1.0, "power_ratio": 1.0,
                       "imag_residue": 0.0}
        return x
    B, C, H, W = x.shape
    F = torch.fft.fft2(x.float())
    mag, phi = F.abs(), F.angle()
    delta = 2 * math.pi / k
    lvl = torch.round(phi / delta)
    Fq = mag * torch.exp(1j * lvl * delta)

    kept_fraction = 1.0
    if omit is not None:
        lv = lvl.long() % k
        drop = (lv == omit % k) | (lv == (-omit) % k)
        Fq = torch.where(drop, torch.zeros_like(Fq), Fq)
        kept_fraction = float(1.0 - drop.float().mean())

    if mode == "renorm":
        pw = Fq.abs() ** 2
        p_total = (mag ** 2).sum(dim=(-2, -1)) - mag[..., 0, 0] ** 2
        p_rem = pw.sum(dim=(-2, -1)) - pw[..., 0, 0]
        Fq = Fq * torch.sqrt(p_total / p_rem.clamp(min=1e-12))[..., None, None]

    if preserve_dc:
        for (i, j) in ((0, 0), (H // 2, 0), (0, W // 2), (H // 2, W // 2)):
            Fq[..., i, j] = F[..., i, j]

    out = torch.fft.ifft2(Fq)
    if return_stats:
        stats = {
            "kept_fraction": kept_fraction,
            "power_ratio": float((Fq.abs() ** 2).sum() / (F.abs() ** 2).sum()),
            "imag_residue": float(out.imag.abs().max()),
        }
        return out.real, stats
    return out.real


# ---------------------------------------------------------------------------
# Output-latent phase analysis + band-split phase hybrids (E7)
# ---------------------------------------------------------------------------

def phase_coherence(phis, n_bins=24):
    """Cross-sample phase coherence, radially averaged.

    phis: (N, C, H, W) stacked FFT phases (unshifted), one per sample/seed.
    Per bin, the resultant length R = |mean_n exp(i*phi_n)|: 1 = identical
    phase across samples, ~R_null for independent uniform phases.
    Returns (centers, R[C, n_bins], R_null) with
    R_null = sqrt(pi)/(2*sqrt(N)), the Rayleigh mean for N uniform phases.
    """
    N, C, H, W = phis.shape
    R = torch.exp(1j * phis.float()).mean(dim=0).abs()  # (C, H, W)
    rr = radial_bins(H, W, phis.device)
    edges = torch.linspace(0, rr.max() + 1e-6, n_bins + 1, device=phis.device)
    idx = (torch.bucketize(rr.flatten(), edges) - 1).clamp(0, n_bins - 1)
    counts = torch.zeros(n_bins, device=phis.device).scatter_add_(
        0, idx, torch.ones_like(idx, dtype=torch.float))
    prof = torch.zeros(C, n_bins, device=phis.device)
    flat = R.reshape(C, -1)
    for c in range(C):
        prof[c] = torch.zeros(n_bins, device=phis.device).scatter_add_(
            0, idx, flat[c])
    prof = prof / counts.clamp(min=1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    r_null = math.sqrt(math.pi) / (2 * math.sqrt(N))
    return centers.cpu(), prof.cpu(), r_null


def flatness(phi, bins=32):
    """std/mean of the FFT-phase histogram over [-pi, pi] (0 = uniform).

    The marginal phase-uniformity metric used by E6/E7 (white noise -> ~0).
    `phase_histogram` returns the same quantity resolved per (channel, band)."""
    h = torch.histc(phi.detach().flatten().float().cpu(), bins=bins,
                    min=-math.pi, max=math.pi)
    return float(h.std() / h.mean())


def phase_histogram(phi, idx_map, n_bins_band=24, n_bins_hist=64):
    """Per-(channel, band) FFT-phase distribution on [-pi, pi] + circular stats.

    phi: (C, H, W) FFT phase (unshifted, DC at [0,0]), e.g. fft2(lat).angle().
    idx_map: (H, W) long radial-band map from band_index_map (same H, W), so
             bands match radial_psd / psd_match exactly (DC -> band 0).

    Returns a dict (all tensors on cpu):
      hist:  (C, n_bins_band, n_bins_hist) -- each row is the phase histogram of
             one (channel, band), L1-normalized over [-pi, pi]; empty bands are
             left all-zero.
      edges: (n_bins_hist + 1,) shared histogram bin edges spanning [-pi, pi].
      flat:  (C, n_bins_band) std/mean of each histogram row (0 = uniform;
             empty/degenerate rows -> 0). Same definition as `flatness`.
      R:     (C, n_bins_band) mean resultant length |mean exp(i*phi)| per band
             (0 = uniform marginal, 1 = a single concentrated phase).

    A uniform marginal (flat ~ 0, R ~ 0) is the white-noise null; deviations
    (typically only the DC / lowest band) flag structured phase. Note marginal
    uniformity does NOT mean phase is uninformative -- image structure lives in
    cross-frequency phase *relationships* (see phase_coherence, E7)."""
    C, H, W = phi.shape
    dev = phi.device
    flat_idx = idx_map.reshape(-1).to(dev)
    counts = torch.zeros(n_bins_band, device=dev).scatter_add_(
        0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float))
    edges = torch.linspace(-math.pi, math.pi, n_bins_hist + 1, device=dev)
    phi_f = phi.reshape(C, -1).float()

    hist = torch.zeros(C, n_bins_band, n_bins_hist, device=dev)
    R = torch.zeros(C, n_bins_band, device=dev)
    flat_stat = torch.zeros(C, n_bins_band, device=dev)
    for c in range(C):
        pb = (torch.bucketize(phi_f[c], edges) - 1).clamp(0, n_bins_hist - 1)
        comb = flat_idx * n_bins_hist + pb
        h = torch.zeros(n_bins_band * n_bins_hist, device=dev).scatter_add_(
            0, comb, torch.ones_like(comb, dtype=torch.float))
        h = h.view(n_bins_band, n_bins_hist)
        ex = torch.zeros(n_bins_band, device=dev).scatter_add_(
            0, flat_idx, torch.cos(phi_f[c]))
        ey = torch.zeros(n_bins_band, device=dev).scatter_add_(
            0, flat_idx, torch.sin(phi_f[c]))
        R[c] = torch.sqrt(ex ** 2 + ey ** 2) / counts.clamp(min=1)
        row_sum = h.sum(dim=1, keepdim=True)
        hist[c] = h / row_sum.clamp(min=1)
        m = hist[c].mean(dim=1)
        flat_stat[c] = torch.where(
            (m > 0) & (row_sum.squeeze(1) > 0),
            hist[c].std(dim=1) / m.clamp(min=1e-12), torch.zeros_like(m))
    return {"hist": hist.cpu(), "edges": edges.cpu(),
            "flat": flat_stat.cpu(), "R": R.cpu()}


def band_phase_swap(lat_a, lat_b, c, mag_from="A", dc_from=None):
    """Hybrid spectrum: phase from A inside the lowest-c radial band, phase
    from B outside; magnitude everywhere from `mag_from` ('A' or 'B').

    c=0 -> pure-B phase, c=1.0 -> pure-A phase (the interpolation knob).
    Works unshifted with a radial-quantile mask built on fftfreq coordinates
    (same lowest-c-fraction-by-area semantics as paper_low_mask, but exactly
    Hermitian-symmetric -- the shifted linspace grid is not, for even sizes
    -- so conjugate bins never straddle the cutoff and the ifft stays real).
    The DC bin (magnitude AND phase, i.e. the per-channel mean) comes from
    `dc_from or mag_from`. Returns the complex ifft -- callers take .real
    and may record .imag residue (same convention as e6's phase_rerand).
    """
    assert lat_a.shape == lat_b.shape
    B, C, H, W = lat_a.shape
    fft_a = torch.fft.fft2(lat_a.float())
    fft_b = torch.fft.fft2(lat_b.float())

    if c > 0:
        rr = radial_bins(H, W, lat_a.device)
        low = (rr <= torch.quantile(rr.flatten(), c)).float()[None, None]
    else:
        low = torch.zeros(1, 1, H, W, device=lat_a.device)
    phi = fft_a.angle() * low + fft_b.angle() * (1.0 - low)
    M = fft_a.abs() if mag_from == "A" else fft_b.abs()
    mix = M * torch.exp(1j * phi)

    # DC bin at [0,0] (unshifted); take it wholesale from one image
    dc_src = fft_a if (dc_from or mag_from) == "A" else fft_b
    mix[..., 0, 0] = dc_src[..., 0, 0]

    return torch.fft.ifft2(mix)


def spectral_slope(centers, psd, fmin=0.02, fmax=0.4):
    """Per-channel log-log linear fit of a radial_psd output over
    fmin <= f <= fmax. Returns a list of slopes (0 = white)."""
    centers, psd = centers.float(), psd.float()
    sel = (centers >= fmin) & (centers <= fmax) & (psd > 0).all(dim=0)
    x = torch.log(centers[sel])
    slopes = []
    for c in range(psd.shape[0]):
        y = torch.log(psd[c, sel])
        xc = x - x.mean()
        slopes.append(float((xc * (y - y.mean())).sum() / (xc ** 2).sum()))
    return slopes


# ---------------------------------------------------------------------------
# Per-band PSD matching (E8)
# ---------------------------------------------------------------------------

def band_index_map(H, W, n_bins=24, device="cuda"):
    """(H, W) long map: unshifted FFT bin -> radial band id in [0, n_bins).

    Same binning recipe as radial_psd (linspace edges over radial_bins,
    bucketize-1, clamped), so power measured with it and gains applied
    through it land on identical bins. DC at [0, 0] -> band 0. The map is
    Hermitian-symmetric (rr(-f) = rr(f)), so per-band gains preserve
    spectrum symmetry."""
    rr = radial_bins(H, W, device)
    edges = torch.linspace(0, rr.max() + 1e-6, n_bins + 1, device=device)
    idx = (torch.bucketize(rr.flatten(), edges) - 1).clamp(0, n_bins - 1)
    return idx.view(H, W)


def band_power(F2, idx_map, n_bins):
    """Counts-weighted mean power per (channel, band).

    F2: (C, H, W) real tensor of |fft2(x)|**2 values. Returns (C, n_bins)
    float32. Same reduction as radial_psd (scatter_add / counts) without
    the /(H*W) normalization -- gain ratios cancel it."""
    C = F2.shape[0]
    flat_idx = idx_map.flatten()
    counts = torch.zeros(n_bins, device=F2.device).scatter_add_(
        0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float))
    out = torch.zeros(C, n_bins, device=F2.device)
    src = F2.float().reshape(C, -1)
    for c in range(C):
        out[c] = torch.zeros(n_bins, device=F2.device).scatter_add_(
            0, flat_idx, src[c])
    return out / counts.clamp(min=1)


def psd_match(lat, ref_band_power, idx_map, n_bins, eps=1e-8,
              return_stats=False):
    """Scale the FFT magnitudes of lat per (channel, radial band) so each
    band's mean power matches ref_band_power; phase untouched.

    lat: (1, C, H, W), any float dtype (math in fp32, cast back on return).
    ref_band_power: (C, n_bins) as produced by band_power.
    gain[c, b] = sqrt(ref[c, b] / cur[c, b]); the gain map gain[c, idx_map]
    is real and radially symmetric, hence Hermitian-preserving: the ifft
    stays real (~1e-6 residue). The DC bin lives in band 0 and is scaled
    with it -- channel means are part of the power story (E7), so they are
    matched too."""
    dt = lat.dtype
    F = torch.fft.fft2(lat.float())
    cur = band_power((F.abs() ** 2)[0], idx_map, n_bins)
    gain = torch.sqrt(ref_band_power.to(cur.device) / cur.clamp(min=eps))
    F = F * gain[:, idx_map][None]
    out = torch.fft.ifft2(F)
    if return_stats:
        return out.real.to(dt), {
            "imag_residue": float(out.imag.abs().max()),
            "gain_min": float(gain.min()),
            "gain_max": float(gain.max()),
        }
    return out.real.to(dt)


# ---------------------------------------------------------------------------
# Parametric phase surgery on output latents (E13 / E14)
# ---------------------------------------------------------------------------
#
# All operators below work UNSHIFTED (DC at [0, 0], torch.fft convention) and
# are Hermitian-preserving so ifft2(.).real loses only ~1e-6 residue. The four
# self-conjugate bins -- DC (0,0) and the Nyquist axes (H/2,0), (0,W/2),
# (H/2,W/2) -- carry a real coefficient (phase in {0, pi}); any edit that would
# rotate them off the real axis is undone by restoring them from the source.

_SELF_CONJ = lambda H, W: ((0, 0), (H // 2, 0), (0, W // 2), (H // 2, W // 2))


def _restore_self_conj(F_new, F_src, H, W):
    """In-place restore the 4 self-conjugate bins from F_src (keeps realness +
    the per-channel DC mean)."""
    for (i, j) in _SELF_CONJ(H, W):
        F_new[..., i, j] = F_src[..., i, j]
    return F_new


def odd_sign_mask(H, W, device="cuda"):
    """(H, W) tensor in {-1, 0, +1}, odd under frequency negation: +1 on the
    lexicographically-first bin of each conjugate pair, -1 on its partner, 0 on
    self-conjugate bins. Adding `delta * mask` to the FFT phase is the
    Hermitian-preserving *constant phase offset* (NOT a spatial shift -- that is
    `phase_ramp`). Demonstrates why a naive constant added to every phase breaks
    realness: only the antisymmetric version stays real."""
    iy = torch.arange(H, device=device)
    ix = torch.arange(W, device=device)
    lin = iy[:, None] * W + ix[None, :]
    conj_lin = ((-iy) % H)[:, None] * W + ((-ix) % W)[None, :]
    s = torch.zeros(H, W, device=device)
    s[lin < conj_lin] = 1.0
    s[lin > conj_lin] = -1.0
    return s


def _band_mask(idx_map, bands, device):
    """(H, W) {0,1} mask selecting the union of radial `bands` (iterable of band
    ids) in a band_index_map. `bands=None` -> all-ones (whole spectrum)."""
    H, W = idx_map.shape
    if bands is None:
        return torch.ones(H, W, device=device)
    m = torch.zeros(H, W, device=device)
    for b in bands:
        m = m + (idx_map == b).float()
    return (m > 0).float()


def phase_only(lat, eps=1e-12):
    """Oppenheim-Lim phase-only latent: keep the FFT phase, flatten every
    non-DC magnitude to a per-channel constant chosen to preserve total power
    (hence latent std); the DC bin (per-channel mean) is held fixed.
    Hermitian-preserving -> real. Expected decode: recognizable layout, flat /
    desaturated palette (magnitude envelope erased)."""
    B, C, H, W = lat.shape
    F = torch.fft.fft2(lat.float())
    mag = F.abs()
    p_total = (mag ** 2).sum(dim=(-2, -1))
    p_dc = mag[..., 0, 0] ** 2
    const = torch.sqrt(((p_total - p_dc) / (H * W - 1)).clamp(min=eps))
    newmag = const[..., None, None].expand_as(mag).clone()
    Fp = newmag * torch.exp(1j * F.angle())
    _restore_self_conj(Fp, F, H, W)   # DC + Nyquist back to source (real)
    return torch.fft.ifft2(Fp).real


def magnitude_only(lat, generator=None):
    """Keep the FFT magnitude, replace the phase with a fresh Hermitian-uniform
    phase field (random_hermitian_phase). DC + Nyquist kept from source.
    Expected decode: textured palette swatch (no layout)."""
    B, C, H, W = lat.shape
    F = torch.fft.fft2(lat.float())
    phi = random_hermitian_phase(C, H, W, device=lat.device, generator=generator)
    Fm = F.abs() * torch.exp(1j * phi)              # phi: (1, C, H, W)
    _restore_self_conj(Fm, F, H, W)
    return torch.fft.ifft2(Fm).real


def scale_phase(lat, alpha):
    """phi -> alpha * phi, magnitude kept. alpha scaling preserves antisymmetry
    (alpha*phi(-f) = -alpha*phi(f)); self-conjugate bins are restored so the
    result is real. alpha=1 is identity; alpha=0 -> zero-phase real-even latent;
    alpha=2 doubles every phase angle."""
    B, C, H, W = lat.shape
    F = torch.fft.fft2(lat.float())
    Fs = F.abs() * torch.exp(1j * alpha * F.angle())
    _restore_self_conj(Fs, F, H, W)
    return torch.fft.ifft2(Fs).real


def phase_offset(lat, delta):
    """Hermitian-preserving *constant phase offset*: F(f) *= exp(i*delta*s(f))
    with s the odd_sign_mask. This is NOT a spatial shift (cf. phase_ramp) and
    is the correct way to 'add a constant to the phase' while staying real."""
    B, C, H, W = lat.shape
    F = torch.fft.fft2(lat.float())
    s = odd_sign_mask(H, W, lat.device)
    return torch.fft.ifft2(F * torch.exp(1j * delta * s)[None, None]).real


def phase_ramp(lat, dy, dx):
    """Linear phase ramp = spatial (sub-pixel) shift by (dy, dx) via the Fourier
    shift theorem: F(f) *= exp(-2pi i (fy*dy + fx*dx)). For integer (dy, dx)
    this equals torch.roll(lat, (dy, dx), dims=(-2,-1)) exactly. Included to
    *demonstrate* that a frequency-linear phase ramp is a translation."""
    B, C, H, W = lat.shape
    F = torch.fft.fft2(lat.float())
    fy = torch.fft.fftfreq(H, device=lat.device)
    fx = torch.fft.fftfreq(W, device=lat.device)
    ramp = torch.exp(-2j * math.pi * (fy[:, None] * dy + fx[None, :] * dx))
    return torch.fft.ifft2(F * ramp[None, None]).real


def rotate_band_phase(lat, idx_map, bands, delta):
    """Add a constant phase rotation `delta` only inside the selected radial
    `bands`, applied antisymmetrically (odd_sign_mask) so the latent stays real.
    Magnitude untouched. `bands=None` rotates the whole spectrum (== phase_offset)."""
    B, C, H, W = lat.shape
    F = torch.fft.fft2(lat.float())
    s = odd_sign_mask(H, W, lat.device) * _band_mask(idx_map, bands, lat.device)
    return torch.fft.ifft2(F * torch.exp(1j * delta * s)[None, None]).real


def add_band_phase_noise(lat, idx_map, bands, eps, generator=None):
    """phi -> phi + eps*eta inside the selected radial `bands`, eta a Hermitian
    uniform phase field (random_hermitian_phase, antisymmetric -> stays real).
    DC + Nyquist restored. Magnitude untouched. Sweeping eps per band localizes
    where latent identity lives: low-band noise is expected to destroy identity
    at small eps, high-band noise to be near-free (cf. E6 quantization)."""
    B, C, H, W = lat.shape
    F = torch.fft.fft2(lat.float())
    eta = random_hermitian_phase(C, H, W, device=lat.device, generator=generator)
    m = _band_mask(idx_map, bands, lat.device)
    Fn = F.abs() * torch.exp(1j * (F.angle() + eps * eta * m))
    _restore_self_conj(Fn, F, H, W)
    return torch.fft.ifft2(Fn).real
