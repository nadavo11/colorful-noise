# E48 — Temporal-axis Fourier phasor on LTX latents (follow-on to E45)

Reframe (after grilling the user): a linear temporal phasor is a **circular shift**, it cannot
extrapolate genuinely-new frames. So we only ever test **re-timing / interpolation**. Fractional
interpolation = a **diagnostic** (is temporal phase a faithful, manipulable motion carrier?). The
**deliverable** is temporal-only phase preservation for **edit consistency** vs E45 `phase3d`/vanilla,
judged on a flicker x editability frontier (must *beat* it, not sit on it).

## P0 — pure-math temporal-phasor sanity (no model judgement)
Hypothesis: `fft_F(z) * exp(-2pi j k Δ/F) <=> circular frame shift by Δ` holds in LTX latent space,
and decodes to coherent video — so temporal phase is a usable motion carrier.
Single change vs nothing: this is the foundational probe. Script `e48_temporal_phasor.py`
(reuses E45 `ltx_encode/decode/conform`). Real clip `imageio:cockatoo.mp4`, native **704x480**, 49f.
Cluster job `cluster_e48_phasor.sh` (RunAI `e48-phasor`, A5000).

Latent shape `(1,128,7,15,22)` — **F_lat=7**, tcr=8. VAE round-trip L1 = 0.0187 (clean baseline).

Results:
- **Operator correctness** (latent-space FFT identities): integer-shift==`torch.roll` max|err|=1.2e-6;
  roundtrip(+.5,-.5) max|err|=1.4e-6; half+half==roll max|err|=1.9e-6. All at float32 FFT precision — **correct**.
- **VAE temporal shift-equivariance** (Δ=1): `decode(shift1)` vs `roll(decode, 8)` —
  PSNR all=19.80 dB, **interior[16:33]=37.67 dB**. The interior number is the real one; the low
  all-frames PSNR is the expected boundary corruption (circular wrap frame6→0 vs pixel roll, plus the
  LTX VAE's frame0=1-pixel / rest=8-pixel asymmetry). **37.7 dB interior ⇒ shifting the latent by 1
  frame ≈ a clean 8-pixel-frame shift: the latent F-axis is a faithful, shift-equivariant time axis.**
- **Fractional Δ=0.5 coherence**: PSNR(phasor, latent-lerp)=12.54 dB — phasor differs substantially
  from a naive each-frame-with-next average (it's band-limited sinc interp, not a blend). Eyeball pending
  (`shift_frac0.5.mp4` vs `lerp0.5.mp4`).

Artifacts: `results/e48/{source,recon,shift_int1,shift_frac0.5,lerp0.5}.mp4`.

### Verdict: **KEEP**
The foundation holds: the operator is correct and the LTX latent temporal axis is shift-equivariant in
the interior (37.7 dB). Temporal phase is a faithful, manipulable motion carrier → the deliverable
(temporal-only phase preservation for edit consistency) is worth building.

Honest caveats carried to P1:
1. Equivariance only holds **interior** — boundaries are corrupted by circular wrap + the 1+8k VAE
   asymmetry. P1 preserves phase (no shift), so wrap isn't directly hit, but the boundary asymmetry is
   a real property of the LTX temporal VAE.
2. F_lat=7 (3 oscillatory bins) is fine for interpolation (lossless invertible transform) but leaves no
   room for spectral *modeling* — consistent with dropping extrapolation.

## P1 — temporal-only phase preservation for edit consistency (NEXT)
`mode="phaseT"` in `band_phase_keep` (`fft` over F axis only) inside `flowalign_video`, vs vanilla +
E45 `phase3d`, native non-square dims + real footage. Frontier knob = phase-transplant strength.
Metric: flicker (unconfounded — background-masked or output-self-consistency warped-LPIPS, TBD) x CLIP
editability. KEEP iff it **beats** the vanilla/`phase3d` frontier; KILL if it lands on it (E41/E46 trap).
