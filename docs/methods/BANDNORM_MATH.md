# Spectral Band Normalization: A Formal Description

This note formalizes the *band-normalization* intervention used in E8/E9 — the per-step
clamp of a diffusion latent's radial power spectrum onto a `cfg = 1.0` reference. It is
written to answer three concrete questions precisely:

> **See also** [`VELOCITY_SPECTRAL_MATH.md`](VELOCITY_SPECTRAL_MATH.md) (E37): the same FFT /
> radial-band machinery applied to the CFG *velocity* instead of the latent, clamping toward
> the *same-step* unconditional velocity `v_∅` (scale-correct by construction, one pass).

- **Is the reference a scalar, a scalar per band, or a scalar per band per channel?**
  It is a *per-step, per-channel, per-band* table of mean powers — a tensor of shape
  $(T, C, B) = (28, 16, 24)$. Not a scalar.
- **Do we normalize the magnitude by dividing by that constant in each band?**
  Not a plain division. Each band's magnitude is scaled by the *ratio* $\sqrt{R/P}$ of
  the reference power $R$ to the *current* band power $P$, so the band's mean power is
  driven onto $R$. A division by a fixed constant would be a special case where $P=1$.
- The math below is given paper-style, with each object cross-referenced to the source
  function that implements it.

Source files: `spectral_ops.py` (`radial_bins`, `band_index_map`, `band_power`, `psd_match`),
`bandnorm.py` (`record_reference`, `band_centers`, `modulate_reference`),
`e8_psd_clamp.py` (`RecordPSD`, `ClampPSD`).

---

## 1. Setup and notation

At each denoising step $t \in \{0, \dots, T-1\}$ (with $T = 28$) the pipeline holds a
working latent which, after unpacking, is a real tensor

$$
x \;=\; x_t \;\in\; \mathbb{R}^{C \times H \times W},
\qquad C = 16,\; H = W = 128 .
$$

We operate **per channel**, on the two spatial axes, with the **unshifted** 2-D discrete
Fourier transform (DC at index $[0,0]$, *no* `fftshift`):

$$
F_c[u,v] \;=\; \sum_{h=0}^{H-1}\sum_{w=0}^{W-1}
x_c[h,w]\,\exp\!\Big(-2\pi i\big(\tfrac{uh}{H} + \tfrac{vw}{W}\big)\Big),
\qquad F = \operatorname{fft2}(x) \in \mathbb{C}^{C \times H \times W}.
$$

This is `torch.fft.fft2` on the float latent (`spectral_ops.py:357`). Because $x$ is real,
$F$ is **Hermitian-symmetric**: $F_c[-u,-v] = \overline{F_c[u,v]}$ (indices mod $H,W$).

We will use Parseval's identity for the unnormalized DFT,

$$
\sum_{h,w} x_c[h,w]^2 \;=\; \frac{1}{HW}\sum_{u,v} \big|F_c[u,v]\big|^2 .
\tag{1}
$$

---

## 2. Radial frequency grid and band partition

Define normalized frequency coordinates with `fftfreq` (so they lie in $[-\tfrac12, \tfrac12)$):

$$
f^{(y)}_u = \operatorname{fftfreq}(H)_u,\qquad
f^{(x)}_v = \operatorname{fftfreq}(W)_v,
$$

and the **radial frequency** of each grid point (`radial_bins`, `spectral_ops.py:19`):

$$
r[u,v] \;=\; \sqrt{\big(f^{(y)}_u\big)^2 + \big(f^{(x)}_v\big)^2},
\qquad r_{\max} = \max_{u,v} r[u,v] \approx \sqrt{0.5} \approx 0.707 .
$$

We partition $[0, r_{\max}]$ into $B = 24$ uniform intervals (radial annuli). With edges

$$
e_k \;=\; \frac{k}{B}\,(r_{\max} + \varepsilon_0),\quad k = 0,\dots,B,
\qquad \varepsilon_0 = 10^{-6},
$$

each grid point is assigned a **band index** (`band_index_map`, `spectral_ops.py:312`):

$$
b[u,v] \;=\; \operatorname{clamp}\big(\operatorname{bucketize}(r[u,v];\,e) - 1,\; 0,\; B-1\big)
\;\in\; \{0,\dots,B-1\}.
$$

This defines $B$ disjoint index sets (annuli) that tile the frequency plane:

$$
\mathcal{B}_k \;=\; \big\{ (u,v) : b[u,v] = k \big\},
\qquad N_k \;=\; |\mathcal{B}_k|,
\qquad \bigsqcup_{k=0}^{B-1}\mathcal{B}_k = \{0,\dots,H-1\}\times\{0,\dots,W-1\}.
$$

Two facts we rely on later:

- **DC lives in band 0**: $r[0,0]=0 \Rightarrow b[0,0]=0$, so the channel mean (the DC term)
  is part of band 0's energy and is matched along with it (`psd_match` docstring,
  `spectral_ops.py:353`).
- **The band map is even**: $r[-u,-v] = r[u,v] \Rightarrow b[-u,-v]=b[u,v]$. Conjugate
  partner frequencies always share a band — this is what makes the clamp Hermitian-preserving
  (§6.3).

The band *center* frequencies, used by the frequency-control variant, are the interval
midpoints $\tfrac12(e_k + e_{k+1})$ (`band_centers`, `bandnorm.py:53`).

---

## 3. The per-band power operator

For a latent $x$ define, for each channel $c$ and band $k$, the **counts-weighted mean
power** (`band_power`, `spectral_ops.py:326`):

$$
\boxed{\;
P_{c,k}(x) \;=\; \frac{1}{N_k}\sum_{(u,v)\in\mathcal{B}_k} \big|F_c[u,v]\big|^2
\;}
\tag{2}
$$

with $F = \operatorname{fft2}(x)$. Two implementation notes:

- The quantity averaged is $|F|^2$ — **power** (squared magnitude), not magnitude.
- There is **no** $1/(HW)$ factor here (unlike `radial_psd`, `spectral_ops.py:33`). It is
  omitted deliberately: the clamp uses only the *ratio* of two such quantities (§5), in which
  any common scale cancels. Up to that constant, $P_{c,k}$ is exactly the radially-averaged
  power spectral density of channel $c$ at radius band $k$.

The full measurement is the map $x \mapsto P(x) \in \mathbb{R}^{C \times B}$, i.e. a
$(16, 24)$ table per latent.

---

## 4. The cfg = 1.0 reference

The reference is built once, before the guided run, by generating $S$ images at $cfg = 1.0$
(seeds $s = 1, \dots, S$, default $S = 3$) and recording the per-band power at **every step**
of each (`record_reference`, `bandnorm.py:29`; `RecordPSD`, `e8_psd_clamp.py:51`). Averaging
over seeds gives

$$
\boxed{\;
R_{t,c,k} \;=\; \frac{1}{S}\sum_{s=1}^{S} P_{c,k}\big(x_t^{(s)}\big)
\;}
\qquad
R \in \mathbb{R}^{\,T \times C \times B} = \mathbb{R}^{\,28 \times 16 \times 24}.
\tag{3}
$$

**This directly answers the "scalar or not" question.** The reference is *not* a scalar and
*not* one scalar per band. It is indexed by three coordinates:

| index | range | meaning |
|-------|-------|---------|
| $t$   | $28$  | denoising step (the spectrum is non-stationary in $t$) |
| $c$   | $16$  | latent channel (each channel has its own profile) |
| $k$   | $24$  | radial frequency band |

so $R$ contains $28 \times 16 \times 24 = 10{,}752$ numbers — one target mean-power per
(step, channel, band). The accompanying scalars $R^{\text{tot}}_t = \frac1S\sum_s \sum |x_t^{(s)}|^2$
and $\sigma_t = \frac1S\sum_s \operatorname{std}(x_t^{(s)})$ are recorded too but are used only
by the global variant (Appendix A) and for logging.

---

## 5. The per-step clamp (band normalization)

During a *guided* generation ($cfg = 3.5$), at the end of step $t$ we replace the working
latent $x_t$ by a spectrally-corrected $x_t'$ whose per-band power equals the reference
$R_{t,\cdot,\cdot}$. This is `psd_match` (`spectral_ops.py:344`), invoked each step by
`ClampPSD` (`e8_psd_clamp.py:90`). The procedure is exactly three operations:

**(i) Measure** the current spectrum, $F = \operatorname{fft2}(x_t)$, and its per-band power
$P_{c,k} = P_{c,k}(x_t)$ via Eq. (2).

**(ii) Form the per-band gain** as the square root of the power ratio, with a floor
$\varepsilon = 10^{-8}$ on the denominator:

$$
\boxed{\;
g_{c,k} \;=\; \sqrt{\dfrac{R_{t,c,k}}{\max\!\big(P_{c,k},\,\varepsilon\big)}}
\;}
\qquad g \in \mathbb{R}^{C \times B}.
\tag{4}
$$

**(iii) Apply** the gain to every coefficient, broadcasting the per-band scalar across all
frequencies in its band (the gain is *constant within a band*), then invert:

$$
F'_c[u,v] \;=\; g_{c,\,b[u,v]}\; F_c[u,v],
\qquad
x_t' \;=\; \operatorname{Re}\big(\operatorname{ifft2}(F')\big).
\tag{5}
$$

In code, step (iii) is the single line `F = F * gain[:, idx_map][None]` followed by
`ifft2(...).real` (`spectral_ops.py:360–361`). The map `idx_map` is $b[u,v]$, so
`gain[:, idx_map]` expands the $(C, B) = (16, 24)$ gain table into a $(C, H, W) = (16,128,128)$
per-coefficient gain field.

Note the gain is **real and positive** and depends on $(u,v)$ only through the band index
$b[u,v]$.

> **Relation to "dividing by the constant."** Writing $F_c[u,v] = |F_c[u,v]|\,e^{i\phi_c[u,v]}$,
> Eq. (5) leaves the phase $\phi$ untouched and rescales the magnitude:
> $|F'_c[u,v]| = g_{c,k}\,|F_c[u,v]|$. The factor $g_{c,k}=\sqrt{R/P}$ is *not* a fixed
> constant — it adapts to the current band power $P$ so the band lands exactly on $R$. It is
> a **renormalization to a target level**, not a division by a precomputed number. (A division
> by a fixed constant would be the degenerate case $g_{c,k}=\sqrt{R_{t,c,k}}$, i.e. $P\equiv 1$.)

---

## 6. Properties

Throughout, fix a step $t$, write $R = R_{t,\cdot,\cdot}$, and let $x' = x_t'$ be the output
of Eqs. (4)–(5).

### 6.1 Exact per-band power matching

For each $(c,k)$ with $P_{c,k} > \varepsilon$,

$$
P_{c,k}(x') \;=\; \frac{1}{N_k}\sum_{(u,v)\in\mathcal{B}_k} |F'_c[u,v]|^2
\;=\; \frac{1}{N_k}\sum_{(u,v)\in\mathcal{B}_k} g_{c,k}^2\,|F_c[u,v]|^2
\;=\; g_{c,k}^2\,P_{c,k}
\;=\; \frac{R_{c,k}}{P_{c,k}}\,P_{c,k}
\;=\; R_{c,k}.
$$

The gain is constant over the band, so it pulls out of the sum; the ratio then cancels the
current power exactly. The output's radial power spectrum *is* the reference, by construction.
(The preflight in `e8_psd_clamp.py:166` asserts this to relative error $<10^{-3}$.)

### 6.2 Phase preservation

Since $g_{c,k} \ge 0$ is real, $\arg F'_c[u,v] = \arg F_c[u,v]$ for all $(u,v)$. The clamp
changes only the *magnitude* spectrum (and only its per-band *average* level — see §6.4).
All Fourier phase, which carries the bulk of the image's geometric structure, is left intact.
This is the design intent: E7 found the cfg-vs-content signal lives in power, so we move power
and freeze phase.

### 6.3 Real output (Hermitian preservation)

Because the band map is even, $b[-u,-v] = b[u,v]$ (§2), conjugate-partner frequencies receive
the **same** gain: $g_{c,b[-u,-v]} = g_{c,b[u,v]}$. Combined with the input Hermitian symmetry
$F_c[-u,-v] = \overline{F_c[u,v]}$,

$$
F'_c[-u,-v] = g_{c,b[-u,-v]}F_c[-u,-v]
= g_{c,b[u,v]}\,\overline{F_c[u,v]}
= \overline{g_{c,b[u,v]}F_c[u,v]}
= \overline{F'_c[u,v]},
$$

so $F'$ is Hermitian and $\operatorname{ifft2}(F')$ is real up to floating-point error. The
`.real` in Eq. (5) discards only a numerical residue; measured $\max|\operatorname{Im}| \sim 10^{-6}$
(asserted $<10^{-4}$, `e8_psd_clamp.py:165`).

### 6.4 What is and isn't changed inside a band

The gain $g_{c,k}$ multiplies *all* coefficients in band $k$ by one number. Therefore:

- **Magnitude** of each coefficient scales by $g_{c,k}$; **power** by $g_{c,k}^2$.
- The **relative shape** of the spectrum *within* a band (the ratio of any two coefficients in
  $\mathcal{B}_k$) is preserved, as is each coefficient's phase. Only the band's overall *mean
  power level* is retargeted.

So band-norm matches the radially-averaged PSD while preserving fine within-band structure and
all phase — it is the minimal magnitude edit that achieves §6.1.

### 6.5 Non-compounding (stability across steps)

The gain in Eq. (4) targets an *absolute level* $R_{t,c,k}$ derived from the current latent's
own power $P_{c,k}(x_t)$. It does not multiply onto whatever was applied at step $t-1$; it
re-solves for the level afresh. Hence the clamp cannot run away: if the latent already sits on
the reference, $P = R \Rightarrow g = 1 \Rightarrow x' = x$ (identity, modulo the `.real`
residue — see the preflight identity check, `e8_psd_clamp.py:169`). Empirically the per-step
gains stay in $[0.84, 1.12]$ (E8), i.e. the model barely fights the clamp.

---

## 7. Algorithm summary

$$
\begin{aligned}
&\textbf{Offline (reference):}\\
&\quad \text{for } s=1..S:\ \text{generate at } cfg{=}1,\ \text{record } P_{c,k}(x_t^{(s)})\ \forall t \\
&\quad R_{t,c,k} \leftarrow \tfrac1S\textstyle\sum_s P_{c,k}(x_t^{(s)}) \qquad // \ (T,C,B)=(28,16,24) \\[4pt]
&\textbf{Online (guided, each step } t \text{):}\\
&\quad F \leftarrow \operatorname{fft2}(x_t) \\
&\quad P_{c,k} \leftarrow \tfrac{1}{N_k}\textstyle\sum_{(u,v)\in\mathcal{B}_k}|F_c[u,v]|^2 \\
&\quad g_{c,k} \leftarrow \sqrt{R_{t,c,k} / \max(P_{c,k},\varepsilon)} \\
&\quad F'_c[u,v] \leftarrow g_{c,\,b[u,v]}\,F_c[u,v] \\
&\quad x_t' \leftarrow \operatorname{Re}\big(\operatorname{ifft2}(F')\big) \quad // \ \text{feed back to the pipeline}
\end{aligned}
$$

---

## 8. Equivalent spatial-domain convolution

The clamp (Eqs. 4–5) multiplies every Fourier coefficient by a real, per-band gain.
Pointwise multiplication in the frequency domain *is* convolution in the spatial domain, so
band-norm can be read as a filter applied to the latent — with one important caveat, namely
that the gain is data-dependent (§8.3).

### 8.1 The equivalent kernel

Collect the per-band gain into a frequency-domain field $G_c[u,v] = g_{c,\,b[u,v]}$ — the
$(C,B)$ table of Eq. (4) broadcast over the plane, i.e. exactly `gain[:, idx_map]` from §5.
Then Eq. (5) is the Hadamard product $F'_c = G_c \odot F_c$, and the (cyclic) convolution
theorem gives

$$
\boxed{\;
x'_c \;=\; x_c \;\circledast\; h_c,
\qquad
h_c \;=\; \operatorname{ifft2}\big(G_c\big) \;=\; \operatorname{ifft2}\big(g_{c,\,b[\cdot,\cdot]}\big)
\;}
\tag{6}
$$

a **circular** convolution whose kernel $h_c$ (one per channel) is the inverse transform of the
radial gain profile.

### 8.2 Kernel properties

Each property of $h_c$ is inherited directly from the gain field $G_c$:

- **Zero-phase (real, even).** $G_c$ is real, non-negative, and *even*: $b[-u,-v]=b[u,v]$ (§2)
  gives $G_c[-u,-v]=G_c[u,v]$. A real even spectrum inverts to a real even kernel,
  $h_c[-h,-w]=h_c[h,w]$. A symmetric kernel is **zero-phase** — this is precisely §6.2:
  convolving with $h_c$ cannot move any Fourier phase.
- **Isotropic.** $G_c$ depends on $(u,v)$ only through the radius $r[u,v]$, so $h_c$ is
  (discretely) radially symmetric — the 2-D inverse Hankel transform of the 1-D radial gain
  $g_{c,\cdot}$, with continuum analogue
  $h_c(\rho) = \int_0^\infty g_c(r)\,J_0(2\pi r\rho)\,2\pi r\,dr.$
- **Ringing.** The gain is *piecewise constant* in $r$ (one value per annulus), i.e. a sum of
  ideal ring band-passes. Each annulus $[e_k, e_{k+1})$ inverts to a difference of jinc
  functions ($\propto J_1(2\pi e\rho)/\rho$), so $h_c$ is a broad isotropic kernel with
  oscillatory side-lobes — the Gibbs cost of hard band edges. Its spatial sum equals the DC
  gain, $\sum_{h,w} h_c[h,w] = G_c[0,0] = g_{c,0}$ (the kernel's response to a constant input).
- **Circular, not linear.** Because Eq. (6) comes from the DFT, $\circledast$ is *cyclic* —
  energy wraps at the latent borders. A linear-convolution reading would require zero-padding.

### 8.3 Why it is not one fixed filter (adaptive equalization)

The kernel $h_c$ is only defined once the gain is, and the gain
$g_{c,k}=\sqrt{R_{t,c,k}/P_{c,k}(x_t)}$ (Eq. 4) reads the *current* latent's own band power. So
$h_c$ is recomputed for every input and every step, and the map $x\mapsto x'$ is **nonlinear**.
Concretely it is scale-invariant: under $x\to\alpha x$ we have $F\to\alpha F$, $P\to\alpha^2 P$,
$g\to g/\alpha$, hence $F' = gF \to F'$ *unchanged* — band-norm drives power onto $R$ regardless
of input amplitude. It is therefore best read as an **adaptive spectral equalizer** (a
zero-phase, isotropic cousin of a Wiener filter) that acts, on each *individual* latent, as an
exact circular convolution.

The only literally *fixed* convolution in this family is the global variant (Appendix A): there
$G_c \equiv \lambda_t$ is flat, so $h_c = \lambda_t\,\delta$ — a scaled Dirac, i.e. the pure
scalar multiply, the degenerate $B=1$ case.

---

## Appendix A — Global-power variant

`ClampPSD(mode='global')` (`e8_psd_clamp.py:100`) is the degenerate case where the whole latent
is matched with a **single scalar per step** — no banding, no FFT. Using the recorded total
power $R^{\text{tot}}_t = \frac1S\sum_s\sum|x_t^{(s)}|^2$,

$$
\lambda_t \;=\; \sqrt{\dfrac{R^{\text{tot}}_t}{\max\big(\sum_{c,h,w} x_t[c,h,w]^2,\ 10^{-12}\big)}},
\qquad x_t' \;=\; \lambda_t\, x_t .
$$

By Parseval (Eq. 1), scaling the latent by $\lambda_t$ scales *every* band's power by
$\lambda_t^2$ identically, so this matches total energy exactly in latent space without any
transform. It is the spectrally-flat baseline against which band-norm is compared: band-norm
reshapes the *profile* across bands, global-norm only sets the *total*.

## Appendix B — Frequency control

`modulate_reference` (`bandnorm.py:60`) steers the method by editing the *target* before the
clamp, scaling the reference power over a selected radial range by a factor $g^2$:

$$
R'_{t,c,k} \;=\;
\begin{cases}
g^2\, R_{t,c,k}, & \text{$k$ selected} \\
R_{t,c,k}, & \text{otherwise,}
\end{cases}
\qquad
\text{selected} =
\begin{cases}
\text{center}_k \ge f_{\text{cut}} & (\texttt{target='high'})\\
\text{center}_k < f_{\text{cut}} & (\texttt{target='low'})
\end{cases}
$$

with default cut frequency $f_{\text{cut}} = 0.25$ and $\text{center}_k = \tfrac12(e_k+e_{k+1})$.
Power is scaled by $g^2$ precisely so that, through Eq. (4), the *magnitude* target in those
bands becomes $g\times$ its plain band-norm level. Because the clamp re-targets an absolute
level each step (§6.5), this is stable and does not compound; $g=1$ recovers plain band-norm.
Setting $g>1$ on the high bands boosts fine detail, $g<1$ smooths it, etc.

---

### Cross-reference table

| Symbol | Definition | Source |
|--------|-----------|--------|
| $F = \operatorname{fft2}(x)$ | per-channel unshifted 2-D DFT | `spectral_ops.py:357` |
| $r[u,v]$ | radial frequency grid | `radial_bins`, `spectral_ops.py:19` |
| $b[u,v],\ \mathcal{B}_k$ | band index / annulus, $B=24$ | `band_index_map`, `spectral_ops.py:312` |
| $P_{c,k}(x)$ | per-band mean power, Eq. (2) | `band_power`, `spectral_ops.py:326` |
| $R_{t,c,k}$ | cfg=1 reference, $(28,16,24)$, Eq. (3) | `record_reference`, `bandnorm.py:29` |
| $g_{c,k}$, clamp | Eqs. (4)–(5) | `psd_match`, `spectral_ops.py:344` |
| $h_c=\operatorname{ifft2}(g_{c,b[\cdot]})$ | equivalent conv. kernel, Eq. (6) | §8 (this note) |
| per-step driver | calls `psd_match` each step | `ClampPSD`, `e8_psd_clamp.py:74` |
| $\lambda_t$ | global-power variant | `e8_psd_clamp.py:100` |
| $R'$ | frequency control | `modulate_reference`, `bandnorm.py:60` |
