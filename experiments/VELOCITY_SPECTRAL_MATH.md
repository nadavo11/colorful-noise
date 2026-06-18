# Velocity Spectral Normalization: A Formal Description (E37)

This note formalizes the **velocity-modulation** intervention — editing the
classifier-free-guidance (CFG) velocity of a flow-matching sampler *during* generation so
that its Fourier **amplitude** is pulled toward the **unconditional** velocity, while its
**phase** is preserved. It is the one-pass, scale-correct analogue of the band-normalization
clamp of [`BANDNORM_MATH.md`](BANDNORM_MATH.md) (E8/E9/E16/E23): there we clamped the
*latent*'s radial power onto a recorded `cfg = 1` reference; here we clamp the *velocity*'s
amplitude onto the *same-step* unconditional velocity, which is computed for free by CFG.

It answers three questions precisely:

- **What exactly is normalized — the latent or the model output?** The model output: the
  flow-matching **velocity** `v` that the Euler step integrates. Not the latent.
- **What is the reference?** The **unconditional velocity** `v_∅` at the *current* step — not
  a recorded table and not a clean-image statistic. Because it is the same-step field, its
  amplitude is already at the right scale, so the edit is on-manifold at every step.
- **Magnitude or power?** Two variants are provided: a per-**bin** magnitude transplant
  (`|V_w| ← |V_∅|`) and a per-**band** mean-power match (the `psd_match`/SBN operator).

Source file: `velocity_spectral_ops.py` (`cfg_velocity`, `mag_transplant_band`,
`bandpower_match_band`, `band_gain_velocity`, `make_velocity_override`), reusing
`spectral_ops.py` (`band_index_map`, `band_power`, `psd_match`, `_restore_self_conj`) and
`latent_spectral_ops.py` (`radial_norm`, `_band_sel`). Interception plumbing:
`spectral_demo.gen_sd3_demo` (after `e17_sd35.gen_sd3`).

---

## 1. Setup: flow matching, the Euler step, and CFG

Stable Diffusion 3.5 is a **rectified-flow / flow-matching** model. The transformer predicts
a **velocity** field $v_\theta(z, t, c)$, and a finished image is obtained by integrating the
ODE $\dot z = v$ from noise to data with the explicit (Euler) `FlowMatchEulerDiscreteScheduler`:

$$
z_{t+1} \;=\; z_t \;+\; \Delta_t\, v(z_t, t, c),
\qquad \Delta_t = \sigma_{t+1} - \sigma_t < 0 ,
$$

where the scheduler's `step` receives `model_output` $= v$ as its argument
(`scheduling_flow_match_euler_discrete.py`, the line `prev_sample = sample + (sigma_next - sigma) * model_output`).

**Classifier-free guidance** runs the transformer on a batched `[uncond, cond]` input and
combines the two outputs:

$$
v_\varnothing = v_\theta(z_t, t, \varnothing), \qquad
v_c = v_\theta(z_t, t, c), \qquad
\boxed{\,v_w \;=\; v_\varnothing + w\,(v_c - v_\varnothing)\,}
\tag{1}
$$

with guidance scale $w$ (`pipeline_stable_diffusion_3.py`:
`noise_pred = noise_pred_uncond + guidance_scale*(noise_pred_text - noise_pred_uncond)`).
`cfg_velocity(v_uncond, v_cond, w)` implements (1). At $w=1$, $v_w = v_\varnothing$ — the pure
conditional-free flow field, which lies on the model's natural manifold (the analogue of the
Flux "cfg = 1" reference, but here from *true* CFG, the reason for moving off the distilled
Flux). Larger $w$ improves prompt adherence but over-amplifies certain frequency **magnitudes**
— the oversaturated/over-contrasty CFG look.

Each tensor is real, shape $z, v \in \mathbb{R}^{C\times H\times W}$, $C = 16$. We use the
**unshifted** 2-D DFT $V = \operatorname{fft2}(v)$, per channel, exactly as in
[`BANDNORM_MATH.md`](BANDNORM_MATH.md) §1–§2; because $v$ is real, $V$ is Hermitian
($V_c[-u,-v] = \overline{V_c[u,v]}$). The **radial frequency** $r[u,v]$, its normalization to
$[0,1]$ (`radial_norm`), and the band partition into $B=24$ annuli $\mathcal{B}_k$
(`band_index_map`) are identical to that note; a normalised band $[\ell, h] \subseteq [0,1]$
selects the bins $\mathcal{S} = \{(u,v) : \ell \le \hat r[u,v] \le h\}$ (`_band_sel`).

The four **self-conjugate** bins — DC $(0,0)$ and the Nyquist axes $(\tfrac H2,0),(0,\tfrac W2),
(\tfrac H2,\tfrac W2)$ — carry real coefficients (phase in $\{0,\pi\}$); any edit that would
rotate them off the real axis is undone by restoring them from the source
(`_restore_self_conj`), keeping $\operatorname{ifft2}(\cdot)$ real and preserving $v_w$'s
global level (the DC term).

---

## 2. The intervention

We edit only $v_w$, leaving the Euler step (and hence the scheduler) untouched. Two normalize
modes plus a band gain; all act only inside the band $\mathcal{S}$ and only on a chosen step
window (§4).

### 2.1 Per-bin magnitude transplant — `mag_transplant_band`

Keep $v_w$'s **phase** everywhere; inside $\mathcal{S}$ replace the **magnitude** with the
unconditional magnitude, blended by a strength $s \in [0,1]$:

$$
|V_w'|[u,v] =
\begin{cases}
(1-s)\,|V_w[u,v]| + s\,|V_\varnothing[u,v]| & (u,v)\in\mathcal{S}\setminus\{\text{DC}\}\\[2pt]
|V_w[u,v]| & \text{otherwise}
\end{cases}
\qquad
V_w'[u,v] = |V_w'|[u,v]\; e^{\,i\,\angle V_w[u,v]} .
\tag{2}
$$

then restore the self-conjugate bins from $V_w$ and set $v_w' = \operatorname{ifft2}(V_w').\mathrm{real}$.

**Realness is exact (up to $\sim 10^{-6}$).** $V_w$ is Hermitian, so $|V_w|$ and $\angle V_w$
are respectively even and odd under $(u,v)\mapsto(-u,-v)$. $V_\varnothing$ is Hermitian too, so
$|V_\varnothing|$ is even. The band mask $\mathcal{S}$ is radially defined, hence even. Thus
$|V_w'|$ is even and $\angle V_w$ is odd, so $V_w' = |V_w'|e^{i\angle V_w}$ is Hermitian and its
inverse transform is real (verified empirically: residue $\approx 2.4\times10^{-7}$; the only
bins where $|V_w'|\neq$ Eq. (2) are the four restored self-conjugate bins, by construction).
At $s=1$ this is the literal "force $v_w$ to have $v_\varnothing$'s amplitude"; at $s=0$ it is
the identity (verified: $\lVert v_w' - v_w\rVert_\infty \approx 7\times10^{-7}$).

### 2.2 Per-band mean-power match — `bandpower_match_band`

The gentler, coarser variant: match $v_w$'s **mean power per radial band** to $v_\varnothing$'s,
for bands whose centre lies in $[\ell,h]$, leaving other bands at identity. With the per-band
mean power $P_{c,k}(\cdot)$ of [`BANDNORM_MATH.md`](BANDNORM_MATH.md) Eq. (2) (`band_power`) and
the band-membership indicator $\mathbb{1}[k\in[\ell,h]]$ (band centre $(k+\tfrac12)/B$):

$$
R_{c,k} = (1-s)\,P_{c,k}(v_w) + s\,P_{c,k}(v_\varnothing)\ \ \text{if } k\in[\ell,h],
\qquad R_{c,k} = P_{c,k}(v_w)\ \ \text{otherwise,}
\tag{3}
$$

then $v_w' = \texttt{psd\_match}(v_w, R)$, i.e. scale band $k$ of channel $c$ by the gain
$g_{c,k} = \sqrt{R_{c,k}/P_{c,k}(v_w)}$ (phase untouched; identical operator to the latent SBN
clamp, [`BANDNORM_MATH.md`](BANDNORM_MATH.md) Eqs. (4)–(5)). For $k\notin[\ell,h]$,
$R_{c,k}=P_{c,k}(v_w)\Rightarrow g_{c,k}=1$ (identity), so only in-band power is retargeted.

### 2.3 Band amplify/reduce — `band_gain_velocity`

Independent of $v_\varnothing$: scale $v_w$'s magnitude inside $\mathcal{S}$ by a constant gain
$g$ (DC kept at unity), $V_w'[u,v] = g\,V_w[u,v]$ on $\mathcal{S}$, else $V_w$. The E9 band-gain
lever applied to the velocity.

---

## 3. Why same-step, and why it is scale-correct

The latent `SBN→real` operator (E23/E36) clamps the latent's band power toward a **fixed,
clean-image** target $R$ at every step. But the intermediate latent $z_t$ is a noise–data
interpolant whose power is dominated by the noise level $\sigma_t$; clamping it to a clean
target is only scale-correct at the final step (the empirically observed bug: every-step
`SBN→real` over-clamps the noisy early steps and produces artifacts).

Here the reference is $v_\varnothing$ **at the same step $t$** — it is produced by the same
forward pass, at the same $\sigma_t$, so its amplitude is already the right scale. Pulling
$|V_w|$ toward $|V_\varnothing|$ therefore stays on-manifold throughout. Intuitively, Eq. (1)
says $v_w = v_\varnothing + w\,\delta$ with $\delta = v_c-v_\varnothing$ the adherence
direction; CFG inflates the magnitude of $v_w$ relative to $v_\varnothing$, and §2 deflates it
back **band-selectively** while keeping $v_w$'s phase (hence keeping the adherence-bearing
*direction/layout* and only undoing the magnitude over-shoot).

---

## 4. Step-window gating and the interception point

`callback_on_step_end` fires **after** the Euler step and is not given the velocity, so it
cannot be used. Instead we use the `e17_sd35.gen_sd3` pattern (`gen_sd3_demo`): monkeypatch
`transformer.forward` to record the batched `[uncond, cond]` output, and `scheduler.step` to
run a closure on `model_output` ($=v_w$) *before* the Euler update.

`make_velocity_override(op, ℓ, h, s, g, i_lo, i_hi)` returns
$\texttt{override}(\textit{records}, \textit{model\_output}, \textit{sample})$ that tracks the
step index $i$ and fires only on the inclusive window $i_{lo}\le i\le i_{hi}$ (outside it the
plain CFG velocity passes through unchanged). The demo maps a normalised interval
$[\tau_{lo},\tau_{hi}]\subseteq[0,1]$ to indices $i_\bullet=\operatorname{round}(\tau_\bullet
(T-1))$, so $[0,1]$ = every step. $v_\varnothing$ is read from `records[-1].chunk(2)[0]`; if the
recorded batch is not $2B$ (i.e. $w\le1$, no CFG), the override passes through (no crash).

**Cost.** Per fired step: two `fft2` and one `ifft2` (mag mode) — $O(CHW\log HW)$, no extra
transformer forward, since $v_\varnothing,v_c$ are already computed by CFG.

---

### Cross-reference table

| Symbol | Definition | Source |
|--------|-----------|--------|
| $v_\varnothing, v_c$ | uncond / cond velocity (batched transformer output) | `gen_sd3_demo` `records[-1].chunk(2)` |
| $v_w = v_\varnothing + w(v_c-v_\varnothing)$ | CFG velocity (= `model_output`) | `cfg_velocity`; Eq. (1) |
| $V=\operatorname{fft2}(v)$, $\hat r$, $\mathcal{S}$ | unshifted DFT, normalised radius, band mask | `radial_norm`, `_band_sel` (cf. BANDNORM §1–2) |
| Eq. (2) mag transplant | $|V_w|\!\leftarrow\!|V_\varnothing|$ in-band, keep phase | `mag_transplant_band` |
| Eq. (3) power match | `psd_match` toward $P(v_\varnothing)$ in-band | `bandpower_match_band` (cf. BANDNORM Eqs. 4–5) |
| band gain $g$ | $V_w\!\leftarrow\! gV_w$ in-band | `band_gain_velocity` |
| self-conj restore | DC + 3 Nyquist bins kept real | `_restore_self_conj` |
| step window $[i_{lo},i_{hi}]$ | gate; `scheduler.step` override | `make_velocity_override`, `gen_sd3_demo` |

---

## 5. Status

Operators implemented and validated off-GPU (realness, $s=0$ identity, power-match exactness,
gating). Exposed as the **Velocity modulation** tab of `spectral_demo.py` (default model
`stabilityai/stable-diffusion-3.5-medium`). Quantitative evaluation (GenEval, DPG-Bench) and
RL-style tuning of the band/strength/interval against aesthetic + CLIP scores are **deferred**
(see `EXPERIMENTS.md` §E37).
