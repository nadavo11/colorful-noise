"""E24: 1D spectral surgery on the TOKEN axis of a text-conditioning tensor.

The project's spectral tooling (`spectral_ops.py`) is hard-wired for 2D image
latents `(C,H,W)`. This module is the token-axis analogue: it FFTs a prompt's
sequence embedding `E` of shape `(1, L, D)` along the **token axis** (`dim=1`),
exactly the basis FNet (Lee-Thorp et al. 2021) showed is an effective token-mixing
transform. Low token-frequencies = slow / global meaning across the prompt (DC =
the bag-of-words mean direction); high token-frequencies = sharp token-to-token
detail. E24 swaps / blends / filters these bands between two prompts to test whether
images can be merged or edited through the text conditioning.

Everything is real-in/real-out via `rfft`/`irfft` along `dim=1`, so there is no
Hermitian-symmetry bookkeeping. Frequencies are normalised to `[0, 1]`
(0 = DC, 1 = Nyquist) so a `cut` fraction means the same thing for any prompt
length `L`.

IMPORTANT — padding: Flux's T5 pads every prompt to 512 tokens. FFTing the full
512 mixes the content->padding cliff into the high band. Always operate on the
**real-token span** `E[:, :L]` and reattach the untouched padding with
`apply_on_span`.
"""
import torch

SEQ = 1  # token axis of a (1, L, D) conditioning tensor


# ---------------------------------------------------------------------------
# frequency bookkeeping
# ---------------------------------------------------------------------------

def _norm_freqs(L, device):
    """Normalised rfft frequencies in [0,1] (0=DC, 1=Nyquist) for length L."""
    f = torch.fft.rfftfreq(L, device=device)  # 0 .. 0.5
    fmax = float(f[-1]) if L > 1 else 1.0
    return f / fmax if fmax > 0 else f


def band_map_1d(L, n_bands, device="cpu"):
    """Assign each rfft bin (n_freq = L//2+1) to one of n_bands equal-width radial
    bands in normalised-frequency space. Returns a long tensor of bin->band ids."""
    f = _norm_freqs(L, device)
    ids = (f * n_bands).long().clamp(max=n_bands - 1)
    return ids


@torch.no_grad()
def band_power_1d(E_span, n_bands=12):
    """Channel-pooled power per token-frequency band for `E_span` (1,L,D).
    Returns (centers (n_bands,), power (n_bands,)) -- the token-axis 'PSD'."""
    L = E_span.shape[SEQ]
    F = torch.fft.rfft(E_span.float(), dim=SEQ)          # (1, n_freq, D)
    p = (F.abs() ** 2).mean(dim=(0, 2))                  # (n_freq,) channel+batch mean
    ids = band_map_1d(L, n_bands, E_span.device)
    power = torch.zeros(n_bands, device=E_span.device)
    cnt = torch.zeros(n_bands, device=E_span.device)
    power.scatter_add_(0, ids, p)
    cnt.scatter_add_(0, ids, torch.ones_like(p))
    power = power / cnt.clamp(min=1)
    centers = (torch.arange(n_bands, device=E_span.device) + 0.5) / n_bands
    return centers, power


# ---------------------------------------------------------------------------
# band filters / splits (single prompt -- the 'probe')
# ---------------------------------------------------------------------------

def band_filter_1d(E_span, lo, hi, keep_dc=True):
    """Keep only token-frequencies in the normalised range [lo, hi]; zero the rest.
    DC (freq 0) is kept whenever keep_dc (so a high-pass image isn't pitch black)."""
    L = E_span.shape[SEQ]
    F = torch.fft.rfft(E_span.float(), dim=SEQ)
    f = _norm_freqs(L, E_span.device)
    mask = (f >= lo) & (f <= hi)
    if keep_dc:
        mask = mask | (f == 0)
    F = F * mask.to(F.dtype)[None, :, None]
    return torch.fft.irfft(F, n=L, dim=SEQ).to(E_span.dtype)


def split_bands_1d(E_span, cut):
    """Split into (low, high) reconstructions at normalised cut. low keeps DC..cut,
    high keeps the rest. low + high == E_span (linearity), so this is a clean
    decomposition, not a lossy filter pair."""
    low = band_filter_1d(E_span, 0.0, cut, keep_dc=True)
    high = E_span - low
    return low, high


def band_gain_1d(E_span, lo, hi, gain, keep_dc=True):
    """Multiply token-frequencies in the normalised range [lo, hi] by `gain` -- the
    continuous-control knob: gain<1 attenuates, gain>1 amplifies, gain=1 is identity.
    With keep_dc (default) DC is left at unity so amplifying the high band doesn't
    rescale the prompt's global (bag-of-words) level."""
    L = E_span.shape[SEQ]
    F = torch.fft.rfft(E_span.float(), dim=SEQ)
    f = _norm_freqs(L, E_span.device)
    sel = (f >= lo) & (f <= hi)
    if keep_dc:
        sel = sel & (f != 0)
    g = torch.where(sel, torch.as_tensor(float(gain), device=f.device),
                    torch.ones((), device=f.device))
    F = F * g.to(F.dtype)[None, :, None]
    return torch.fft.irfft(F, n=L, dim=SEQ).to(E_span.dtype)


def band_notch_1d(E_span, lo, hi):
    """Zero a single token-frequency band [lo, hi] (per-band knockout); keeps the
    complement, including DC. notch + band_filter(lo,hi) reconstructs the input."""
    return E_span - band_filter_1d(E_span, lo, hi, keep_dc=False)


def band_phase_filter_1d(E_span, lo, hi, keep_dc=True, randomize=False):
    """PHASE band-pass: keep the token-axis phase only inside the normalised band
    [lo, hi]; outside it flatten the phase to 0 (the 'no-phase' baseline used by the
    mag_only probe) while KEEPING the magnitude at every frequency. Isolates which
    frequency bands' PHASE carries the content -- the band-limited version of phase_only,
    motivated by E30 (phase carries the meaning, magnitude does not).

    randomize=True instead replaces the stopband phase with uniform noise in (-pi, pi]
    -- a stronger ablation that destroys phase info rather than aligning it to 0."""
    L = E_span.shape[SEQ]
    F = torch.fft.rfft(E_span.float(), dim=SEQ)            # (1, n_freq, D)
    f = _norm_freqs(L, E_span.device)
    keep = (f >= lo) & (f <= hi)
    if keep_dc:
        keep = keep | (f == 0)
    # DC (and, for even L, Nyquist) must stay real for irfft to be magnitude-preserving,
    # so always keep their original phase regardless of the band.
    real_bin = torch.zeros_like(keep)
    real_bin[0] = True
    if L % 2 == 0:
        real_bin[-1] = True
    keep = (keep | real_bin)[None, :, None]
    ph = torch.angle(F)
    if randomize:
        noise = (torch.rand_like(ph) * 2 - 1) * torch.pi
        new_ph = torch.where(keep, ph, noise)
    else:
        new_ph = torch.where(keep, ph, torch.zeros_like(ph))   # stopband phase -> 0
    out = torch.fft.irfft(torch.polar(F.abs(), new_ph), n=L, dim=SEQ)
    return out.to(E_span.dtype)


def band_phase_gain_1d(E_span, lo, hi, gain, keep_dc=True):
    """PHASE gain: scale the token-axis phase angle by `gain` inside band [lo, hi], keeping
    the magnitude everywhere. The phase analogue of band_gain_1d (which scales magnitude):
      gain<1 shrinks the phase toward 0 (gain=0 removes phase in the band -> mag-only there),
      gain=1 identity, gain>1 amplifies the per-bin rotation.
    NOTE phase is circular -- angles live in (-pi, pi], so scaling near +/-pi wraps around
    (a branch-cut effect, not a bug). DC and (for even L) Nyquist are never scaled so their
    bins stay real and irfft remains magnitude-preserving."""
    L = E_span.shape[SEQ]
    F = torch.fft.rfft(E_span.float(), dim=SEQ)
    f = _norm_freqs(L, E_span.device)
    sel = (f >= lo) & (f <= hi)
    if keep_dc:
        sel = sel & (f != 0)
    sel[0] = False                       # DC stays real
    if L % 2 == 0:
        sel[-1] = False                  # even-L Nyquist stays real
    sel = sel[None, :, None]
    ph = torch.angle(F)
    new_ph = torch.where(sel, ph * float(gain), ph)
    out = torch.fft.irfft(torch.polar(F.abs(), new_ph), n=L, dim=SEQ)
    return out.to(E_span.dtype)


# ---------------------------------------------------------------------------
# two-prompt recombination (the 'merge' / 'edit')
# ---------------------------------------------------------------------------

def band_swap_1d(E_a, E_b, cut, mag_from=None):
    """Low band (DC..cut) from A, high band from B, recombined into one embedding.

    mag_from in {None,'A','B'}:
      None -> straight complex swap (low spectrum of A, high spectrum of B).
      'A'/'B' -> keep magnitude entirely from that prompt and take only the *phase*
                 per band (low phase from A, high phase from B) -- isolates whether
                 token-axis phase, not magnitude, carries the swap.
    """
    L = E_a.shape[SEQ]
    Fa = torch.fft.rfft(E_a.float(), dim=SEQ)
    Fb = torch.fft.rfft(E_b.float(), dim=SEQ)
    f = _norm_freqs(L, E_a.device)
    low = ((f <= cut) | (f == 0)).to(Fa.dtype)[None, :, None]
    high = 1.0 - low
    if mag_from is None:
        F = Fa * low + Fb * high
    else:
        phase = torch.angle(Fa) * low.real + torch.angle(Fb) * high.real
        mag = (Fa if mag_from == "A" else Fb).abs()
        F = torch.polar(mag, phase)
    return torch.fft.irfft(F, n=L, dim=SEQ).to(E_a.dtype)


def band_blend_1d(E_a, E_b, cut, width=0.15):
    """Soft crossover swap: a cosine ramp of half-width `width` around `cut`, low
    side -> A, high side -> B. Gentler than band_swap_1d's hard mask (no ringing).
    NOTE: a *flat* per-band convex blend equals time-domain lerp_embeds, so the
    meaningful spectral knob is the crossover location `cut`, not a global alpha."""
    L = E_a.shape[SEQ]
    Fa = torch.fft.rfft(E_a.float(), dim=SEQ)
    Fb = torch.fft.rfft(E_b.float(), dim=SEQ)
    f = _norm_freqs(L, E_a.device)
    # weight on A: 1 below cut-width, 0 above cut+width, cosine in between
    t = ((f - (cut - width)) / (2 * width)).clamp(0, 1)
    wA = (0.5 * (1 + torch.cos(torch.pi * t)))[None, :, None]
    F = Fa * wA + Fb * (1 - wA)
    return torch.fft.irfft(F, n=L, dim=SEQ).to(E_a.dtype)


def phase_mag_split_1d(E_a, E_b):
    """Token-axis analogue of the image-side phase/magnitude swap. Returns
    {'phaseA_magB', 'magA_phaseB'} -- which one preserves the subject tells us
    whether phase or magnitude along the token axis is the semantic carrier."""
    L = E_a.shape[SEQ]
    Fa = torch.fft.rfft(E_a.float(), dim=SEQ)
    Fb = torch.fft.rfft(E_b.float(), dim=SEQ)
    pa, pb = torch.angle(Fa), torch.angle(Fb)
    ma, mb = Fa.abs(), Fb.abs()
    out = {
        "phaseA_magB": torch.fft.irfft(torch.polar(mb, pa), n=L, dim=SEQ),
        "magA_phaseB": torch.fft.irfft(torch.polar(ma, pb), n=L, dim=SEQ),
    }
    return {k: v.to(E_a.dtype) for k, v in out.items()}


def lerp_embeds(E_a, E_b, alpha):
    """Plain token-space interpolation (1-alpha)*A + alpha*B. The BASELINE the
    spectral merges must beat on disentanglement."""
    return ((1 - alpha) * E_a.float() + alpha * E_b.float()).to(E_a.dtype)


# ---------------------------------------------------------------------------
# FNet-style 2D (seq x hidden) variant -- secondary comparison
# ---------------------------------------------------------------------------

def fnet_swap_2d(E_a, E_b, cut):
    """FNet-flavoured merge: low 2D band (over BOTH the token and hidden axes,
    a la FNet's 2D DFT) from A, high band from B, taken back to the real embedding.
    The hidden axis isn't semantically ordered, so this is the 'is the 2D mixing
    transform itself enough?' control against the interpretable 1D token-axis swap.
    """
    dims = (1, 2)
    Fa = torch.fft.fft2(E_a.float(), dim=dims)
    Fb = torch.fft.fft2(E_b.float(), dim=dims)
    L, D = E_a.shape[1], E_a.shape[2]
    fy = (torch.fft.fftfreq(L, device=E_a.device).abs() * 2)
    fx = (torch.fft.fftfreq(D, device=E_a.device).abs() * 2)
    r = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2) / (2 ** 0.5)  # 0..1
    low = (r <= cut).to(Fa.dtype)[None]
    F = Fa * low + Fb * (1 - low)
    return torch.fft.ifft2(F, dim=dims).real.to(E_a.dtype)


# ---------------------------------------------------------------------------
# manifold mitigation + padding plumbing
# ---------------------------------------------------------------------------

def renorm_per_token(E_new, E_ref):
    """Rescale each token vector of E_new (over the hidden dim) to E_ref's per-token
    L2 norm. Pulls a recombined embedding back toward the encoder's manifold without
    changing its direction."""
    nn = E_new.float().norm(dim=-1, keepdim=True).clamp(min=1e-8)
    nr = E_ref.float().norm(dim=-1, keepdim=True)
    return (E_new.float() / nn * nr).to(E_new.dtype)


def apply_on_span(fn, E_full, L):
    """Run `fn` on the real-token span E_full[:, :L] and stitch the untouched
    padding E_full[:, L:] back on. `fn` takes and returns a (1, L, D) tensor."""
    out = E_full.clone()
    out[:, :L] = fn(E_full[:, :L])
    return out


def apply_on_subspan(fn, E_full, a, b):
    """Run `fn` on an arbitrary contiguous token sub-span E_full[:, a:b] and stitch
    everything outside it (tokens before `a`, the rest of the prompt, and the padding)
    back unchanged. This is the per-OBJECT generalisation of apply_on_span: `fn` does a
    windowed FFT over just one object's tokens (E24/E30 only ever windowed the prefix
    [:, :L]). `fn` takes and returns a (1, b-a, D) tensor.

    NOTE: the window is short (an object phrase is a few tokens), so the windowed rfft
    has only (b-a)//2+1 bins -- 'low vs high' inside a span is coarse. Use multi-token
    object phrases and check bins-per-object (the e32 preflight prints this)."""
    out = E_full.clone()
    out[:, a:b] = fn(E_full[:, a:b])
    return out
