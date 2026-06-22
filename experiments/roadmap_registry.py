"""Single source of truth for the research roadmap (docs/roadmap/).

This file describes the research *vectors* (threads) and every experiment E0-E43
(E3 was never run; E33/E34 are proposed-only):
what each one asked, what we found, whether the direction is alive or a dead end,
and how to proceed. `make_roadmap.py` reads this and regenerates the HTML site.

To add an experiment: append a dict to EXPERIMENTS (and, if it opens a new line of
work, a dict to THREADS), then run `python experiments/make_roadmap.py`.

Field conventions
-----------------
THREADS[i]:
  id        short slug, used in filenames (thread-<id>.html) and to link experiments
  title     human title
  status    one of STATUSES keys (drives the colour/label)
  summary   one-line gloss for the index map
  narrative the arc of the thread (what question it chases, how its experiments build)
  proceed   explicit "how to proceed / open questions" for this thread

EXPERIMENTS[i]:
  id        "E10" etc.
  title     human title
  thread    a THREADS id
  models    model(s) used
  status    one of STATUSES keys
  motivation what question it asks
  method    how it was done (1-2 sentences)
  result    what was measured/seen
  verdict   the one-line takeaway
  nxt       what it sets up / what to try next ("next" is a builtin-ish name)
  script    repo-relative path to the driver (or None)
  doc       repo-relative path to the deep writeup (or None -> falls back to the log)
  results   results-dir slug under experiments/results/ (or None); if an index.html
            exists there the page links to it
  image     OPTIONAL results-relative image path (e.g. "e23/plots/gap.png") to show a
            small thumbnail; missing files degrade to a caption, so the site stays light
"""

# label + colour for each status (colour is used in the SVG map + legend)
STATUSES = {
    "active":   ("Active",        "#2da44e"),  # green  - live, paying off
    "mapped":   ("Mapped",        "#0969da"),  # blue   - understood, characterised
    "paused":   ("Paused",        "#8250df"),  # purple - works, parked / partial
    "dead-end": ("Dead end",      "#cf222e"),  # red    - tried, does not win
    "pending":  ("Run pending",   "#9a6700"),  # amber  - code done, awaiting cluster
    "done":     ("Foundational",  "#57606a"),  # grey   - early scaffolding
}

THREADS = [
    {
        "id": "foundations",
        "title": "Foundations & diagnostics",
        "status": "done",
        "summary": "What the colored-noise prior actually does to the input noise.",
        "narrative":
            "The project starts from the Colorful-Noise paper's trick of swapping a "
            "low-frequency band of the initial latent noise. E0-E6 dissect that move: "
            "what `fft_radial_frequency_swap` bundles together (DC / low-band magnitude / "
            "phase), how amplitude acts as a conditioning-strength knob, and how white "
            "Gaussian noise factorises exactly into independent phase and magnitude. This "
            "is the scaffolding the later threads stand on.",
        "proceed":
            "Closed as a foundation. Its two durable handles feed everything downstream: "
            "(1) phase vs magnitude is a real, manipulable split in latent Fourier space; "
            "(2) per-band magnitude amplitude = conditioning strength.",
    },
    {
        "id": "spectral-power",
        "title": "Spectral-power control (SBN)",
        "status": "active",
        "summary": "Re-level a latent's per-band power toward a target spectrum.",
        "narrative":
            "The main line. E7 found that high-CFG output latents carry inflated power; "
            "E8 turned that into a causal, per-step PSD clamp; E9 packaged it as a method "
            "-- Spectral Band Normalization (SBN) -- across prompt classes. E10 diagnosed "
            "*why*: CFG inflates low-frequency power above where real photos sit. E11 added "
            "cheap colour/contrast cleanup, E16/E17 benchmarked SBN against training-free "
            "guidance on Flux and SD3.5. E23 is the payoff: stop clamping toward the weak "
            "cfg=1 proxy and clamp toward the spectrum of *real photographs* (real-SBN).",
        "proceed":
            "ALIVE. real-SBN (E23) gives the biggest aesthetic gain of any condition at "
            "~zero prompt-adherence cost and beats the old cfg-1 SBN. Next: bake a single "
            "fixed per-channel real/gen correction curve into a free, deterministic "
            "post-generation step (no per-image matching); confirm the B-VQA adherence "
            "story on T2I-CompBench; port the real target to SD3.5's VAE space (E17 harness).",
    },
    {
        "id": "phase",
        "title": "Phase & structure",
        "status": "mapped",
        "summary": "Phase carries layout; how much of it the seed pre-commits.",
        "narrative":
            "Where does image *structure* live, and how early is it fixed? E6/E7 located "
            "structure in the FFT phase (esp. low bands); E12-E15 mapped latent phase "
            "distributions, the Oppenheim-Lim phase<->magnitude swap, which bands carry "
            "identity, and a classifier over phase manipulations. E29 closes the loop on "
            "the diffusion map itself: measuring (and causally transplanting) how much of "
            "the output latent's phase is inherited from the seed's phase.",
        "proceed":
            "MAPPED. Headline correction: at low guidance the seed fixes the *whole* output "
            "spectrum (magnitude >= phase, pixel r ~0.76), not a phase-specific channel; CFG "
            "erodes inheritance preferentially in low-freq composition bands. Next: repeat "
            "E29 on Flux/SD3.5 (16-ch rectified flow) for architecture-independence, and use "
            "low-band phase as the highest-leverage seed edit at low CFG.",
    },
    {
        "id": "style",
        "title": "Spectral style transfer & editing",
        "status": "paused",
        "summary": "AdaIN-in-Fourier: drive generation/editing with two spectra.",
        "narrative":
            "If phase = content and per-band power = style, then re-leveling per-band power "
            "is AdaIN on the radial power spectrum. E18 recombined two *real* images offline "
            "(phase A + power B); E19 moved it generation-time (content prompt clamped toward "
            "a style image's envelope); E20 asked whether locked-in low-band phase lets us "
            "skip early denoising steps (warm-start). E21/E22 pushed it to *real-image "
            "editing*: invert a photo, regenerate under a new prompt while locking source "
            "frequency bands.",
        "proceed":
            "PARTIAL / PARKED. The transferable quantity is tone/palette/spectral-energy, "
            "NOT oriented brushwork (radial bands are isotropic) -- and it is VAE-dependent "
            "(real on SD3.5, near-inert on Flux). E21's RF inversion on SD3.5 fails the "
            "reconstruction gate; E22's SDXL DDIM-inversion pivot reconstructs (CLIP-I ~0.94) "
            "and confirms low-band-phase-lock preserves composition at an edit-strength cost. "
            "Next: anisotropic (oriented) bands for real strokes; tune the E22 lock/strength "
            "frontier; finish the E19 generation-time gen/score run on SD3.5.",
    },
    {
        "id": "seed",
        "title": "Seed steering (“golden noise”)",
        "status": "dead-end",
        "summary": "Bias the initial seed toward the prompt. It loses to re-rolling.",
        "narrative":
            "Can we optimise the initial seed to improve prompt adherence while keeping it a "
            "valid Gaussian (||z||=sqrt(d))? E25 (SD1.5) found a gentle latent-mode lever; "
            "E26 (SDXL) swept the cost on DPG-Bench; E27 distilled it to a single additive "
            "concept-direction via CLIP-grad x decoder-Jacobian; E28 ran the decisive regime "
            "test on hard compositional CompBench failures with B-VQA.",
        "proceed":
            "DEAD END (documented). On compositional prompts, gradient seed-biasing LOSES to "
            "a plain re-roll (seed-dependent recovery .57 vs .43/.29) and breaks prompts that "
            "already passed. Seed-as-adherence does not win; best-of-N + a picker does. The "
            "useful residue is diagnostic, not generative -- it confirmed E29's seed->output "
            "determinism. Do not invest further in seed optimisation for adherence.",
    },
    {
        "id": "text-freq",
        "title": "Text-frequency conditioning",
        "status": "mapped",
        "summary": "FFT along the token axis of the text embedding.",
        "narrative":
            "Move the spectral idea off the image latent and onto the *text conditioning*. "
            "E24 (FNet-motivated) takes a 1-D FFT along the token axis of the T5 embedding: "
            "low band ~ subject/identity, high band ~ style/detail, with band swaps and "
            "blends on Flux. E30 turns that into a continuous attenuate/amplify knob and "
            "characterises each band; E31 integrates frequency-surgery target conditioning "
            "into FlowEdit for inversion-free real-image editing.",
        "proceed":
            "CHARACTERISED, but no new control lever. E30 mapped the token spectrum (ran on "
            "runai): PHASE carries the content (phase_only~full, mag_only collapses); LOW band = "
            "coarse gist, MID+HIGH bands = attribute-object binding (low-pass kills B-VQA, notch-lo "
            "keeps it); no single band is load-bearing. But spectral MERGE/BLEND still loses to "
            "literally writing 'A and B' (concat B-VQA 0.85 vs merges ~0), confirming E24. E31 then "
            "showed token-frequency surgery does NOT drive inversion-free FlowEdit -- the kept low "
            "band anchors to the source so the velocity delta ~0; it can't out-edit a plain prompt "
            "swap. Net: the spectrum is structured and interpretable, but neither blending nor "
            "frequency-surgery editing beats the trivial baseline. Latent-band editing (E22) remains "
            "the usable image-editing handle. E32 reopened one untested angle -- localising the band "
            "edit to a SINGLE object's token span (windowed FFT) -- and FOUND the thread's first "
            "controllable per-object lever: targeted band gain is object-selective and steerable in "
            "CLIP (boost->target up/other down, t up to 3.1) while the global-gain control is a null, "
            "though the effect is small and the presence/binding effect (high band) is noisy. Two "
            "follow-ups are queued: textual-inversion of an object then frequency-control of its span, "
            "and channel-axis (D=4096) interpretability for direct attribute steering.",
    },
]

EXPERIMENTS = [
    # ---- foundations -------------------------------------------------------
    {"id": "E0", "title": "PSD diagnostics of the colored-noise mix", "thread": "foundations",
     "models": "SDXL", "status": "done",
     "motivation": "What does the paper's fft_radial_frequency_swap actually do to the noise?",
     "method": "Decompose the low-band swap and read out the per-band PSD it imposes.",
     "result": "The swap bundles DC, low-band magnitude and low-band phase together.",
     "verdict": "Established the radial-PSD diagnostic the whole project reuses.",
     "nxt": "Separate the three bundled ingredients (E2).",
     "script": "experiments/e0_diagnostics.py", "doc": None, "results": "e0", "image": None},
    {"id": "E1", "title": "Generation from full-spectrum colored noise", "thread": "foundations",
     "models": "SDXL", "status": "done",
     "motivation": "If a tiny low-band tweak conditions SDXL, what does full-spectrum colour do?",
     "method": "Drive SDXL from noise colored across the whole spectrum, not just one band.",
     "result": "Full-spectrum coloring over-conditions; the low band does the heavy lifting.",
     "verdict": "Conditioning power is concentrated in the low bands.",
     "nxt": "Build the controlled 8-way ingredient matrix (E2).",
     "script": "experiments/e1_colored.py", "doc": None, "results": "e1", "image": None},
    {"id": "E2", "title": "8-way phase/magnitude/DC conditioning matrix", "thread": "foundations",
     "models": "SDXL", "status": "done",
     "motivation": "Disentangle the three ingredients the low-band swap bundles.",
     "method": "A 2x2x2 matrix over {DC, low-band magnitude, low-band phase} donors.",
     "result": "Amplitude (magnitude) behaves as a conditioning-strength dial.",
     "verdict": "Magnitude amplitude = strength; phase = structure. Core split.",
     "nxt": "Sweep the strength dial (E5); test zero-SNR regime (E4).",
     "script": "experiments/e2_matrix.py", "doc": None, "results": "e2", "image": None},
    {"id": "E4", "title": "Zero terminal SNR control (Playground v2.5)", "thread": "foundations",
     "models": "Playground v2.5", "status": "done",
     "motivation": "Is the photoreal-yet-conditioned regime a property of zero-terminal-SNR?",
     "method": "Repeat the conditioning probe on a zero-terminal-SNR model.",
     "result": "The conditioning behaviour is not specific to zero-SNR training.",
     "verdict": "Effect generalises beyond the SNR schedule.",
     "nxt": "Quantify the strength dial (E5).",
     "script": "experiments/e4_zero_snr.py", "doc": None, "results": "e4", "image": None},
    {"id": "E5", "title": "Conditioning-strength sweep (flat low-band magnitude)", "thread": "foundations",
     "models": "SDXL", "status": "done",
     "motivation": "E2 showed amplitude = strength; map the dial quantitatively.",
     "method": "Sweep a flat low-band magnitude scale (mag_scale) and read conditioning.",
     "result": "A smooth monotone strength response; natural-image amplitude is a sweet spot.",
     "verdict": "The strength dial is continuous and predictable.",
     "nxt": "Turn to phase surgery on the input noise (E6).",
     "script": "experiments/e5_strength.py", "doc": None, "results": "e5", "image": None},
    {"id": "E6", "title": "FFT-phase surgery on the input noise (SDXL)", "thread": "phase",
     "models": "SDXL", "status": "done",
     "motivation": "White noise factorises into independent phase & magnitude -- exploit it.",
     "method": "Phase re-randomisation, image-phase transplant, phase quantisation, level omission.",
     "result": "Phase (esp. low bands) carries the conditioned layout; magnitude carries power.",
     "verdict": "Locates structure in phase -- seed for the whole phase thread.",
     "nxt": "Flip to the *output* latent and to Flux (E7).",
     "script": "experiments/e6_phase.py", "doc": None, "results": "e6", "image": None},
    # ---- spectral-power / phase pivot --------------------------------------
    {"id": "E7", "title": "Flux output-latent phase & spectrum (cfg 1.0 vs 3.5)", "thread": "phase",
     "models": "FLUX.1-dev", "status": "done",
     "motivation": "E0-E6 probed *input* noise; what do *output* latents look like spectrally?",
     "method": "Compare cfg=1 vs cfg=3.5 Flux output-latent phase stats + band-split phase interpolation.",
     "result": "Higher CFG carries more spectral power; identity follows the low-band phase donor.",
     "verdict": "Output latents inherit structure from low-band phase; CFG inflates power.",
     "nxt": "Test causally with a per-step PSD clamp (E8); diagnose the CFG inflation (E10).",
     "script": "experiments/e7_flux_phase.py", "doc": None, "results": "e7", "image": None},
    {"id": "E8", "title": "Per-step PSD clamping during generation", "thread": "spectral-power",
     "models": "FLUX.1-dev", "status": "done",
     "motivation": "E7 is correlational -- causally test whether re-leveling power changes output.",
     "method": "Clamp the per-band PSD at every denoising step toward a reference.",
     "result": "Clamping the power spectrum causally alters texture/detail without moving layout.",
     "verdict": "PSD clamping works causally -- becomes the SBN operator.",
     "nxt": "Package it as a method across prompt classes (E9).",
     "script": "experiments/e8_psd_clamp.py", "doc": None, "results": "e8", "image": None},
    {"id": "E9", "title": "Band-normalized generation (SBN) across prompt classes", "thread": "spectral-power",
     "models": "FLUX.1-dev", "status": "done",
     "motivation": "Make per-step PSD clamping a usable method and test it broadly.",
     "method": "SBN = clamp cfg=3.5 latent power to a cfg=1 reference; 6 prompt classes; +CLIP-T, "
               "cost, universal reference, selective high/low frequency control (E9b add-ons).",
     "result": "Band-norm detail effect is content-dependent; CLIP-T held; cost characterised.",
     "verdict": "SBN is a real, cheap method -- but the cfg=1 target is a proxy (see E10/E23).",
     "nxt": "Explain WHY (E10); clean up colour (E11); benchmark vs baselines (E16).",
     "script": "experiments/e9_bandnorm_classes.py", "doc": None, "results": "e9", "image": None},
    {"id": "E10", "title": "CFG inflates spectral power (the SBN motivation)", "thread": "spectral-power",
     "models": "FLUX.1-dev", "status": "mapped",
     "motivation": "Why re-level power at all? Show where CFG puts the spectrum vs real photos.",
     "method": "True-CFG sweep w in {1..5}; compare generated PSD to real-photo (picsum/COCO) PSD.",
     "result": "Latent power rises ~3x over w=1->5; real photos sit at standard guidance (w~3); "
               "the unguided field is spectrally *weaker* than real.",
     "verdict": "CFG inflates low-freq power above natural -- the fact SBN clamps back.",
     "nxt": "Target the REAL spectrum instead of the cfg=1 proxy (E23).",
     "script": "experiments/e10_cfg_spectral.py", "doc": "docs/experiment-reports/EXPERIMENT_10.md", "results": "e10", "image": None},
    {"id": "E11", "title": "Cheap colour/contrast correction of SBN outputs", "thread": "spectral-power",
     "models": "FLUX.1-dev", "status": "done",
     "motivation": "SBN clamps power but can shift palette/contrast; fix it cheaply post-hoc.",
     "method": "Image-level autocontrast / contrast / hist-match / luminance-eq / saturation variants.",
     "result": "Simple image-space corrections recover palette without touching the latent.",
     "verdict": "Colour drift is a cheap post-process, not a method blocker.",
     "nxt": "Benchmark SBN fidelity properly (E16).",
     "script": "experiments/e11_color_correct.py", "doc": None, "results": "e11", "image": None},
    # ---- phase mapping -----------------------------------------------------
    {"id": "E12", "title": "Latent FFT phase distributions across classes", "thread": "phase",
     "models": "FLUX.1-dev", "status": "done",
     "motivation": "Baseline for the phase line: is latent phase uniform, or class-structured?",
     "method": "Measure per-band latent FFT phase distributions across image classes.",
     "result": "Phase is broadly uniform per band; structure is in the *cross-frequency* pattern.",
     "verdict": "Sets the null the E13-E15 manipulations are read against.",
     "nxt": "Swap phase vs magnitude wholesale (E13).",
     "script": "experiments/e12_phase_dist.py", "doc": None, "results": "e12", "image": None},
    {"id": "E13", "title": "Full-spectrum phase <-> magnitude swap (Oppenheim-Lim)", "thread": "phase",
     "models": "FLUX.1-dev", "status": "done",
     "motivation": "Does the classic 'phase carries structure' hold in the Flux latent?",
     "method": "Swap phase vs magnitude wholesale between two latents and decode.",
     "result": "Decoded identity follows the phase donor -- Oppenheim-Lim holds in latent space.",
     "verdict": "Phase = structure confirmed in the latent, not just pixels.",
     "nxt": "Find WHICH phase bands carry identity (E14).",
     "script": "experiments/e13_phase_mag_swap.py", "doc": None, "results": "e13", "image": None},
    {"id": "E14", "title": "Functions on phase: which bands carry identity", "thread": "phase",
     "models": "FLUX.1-dev", "status": "done",
     "motivation": "E13 swapped wholesale; localise identity to specific phase bands.",
     "method": "Deform the phase per band (quantise, rerandomise, omit) and watch identity.",
     "result": "Low-band phase carries most recognisable identity; high bands = detail.",
     "verdict": "Identity lives in low-band phase -- the lever for editing/warm-start.",
     "nxt": "Cluster the manipulated outputs to quantify (E15).",
     "script": "experiments/e14_phase_functions.py", "doc": None, "results": "e14", "image": None},
    {"id": "E15", "title": "Classify outputs by phase manipulation", "thread": "phase",
     "models": "FLUX.1-dev", "status": "done",
     "motivation": "Turn the E13/E14 battery into a quantitative read.",
     "method": "Cluster/classify the phase-manipulated decodes by manipulation type.",
     "result": "Manipulations separate cleanly -- the phase effects are systematic, not noise.",
     "verdict": "Closes the descriptive phase mapping; E29 takes it to the seed->output map.",
     "nxt": "Ask whether the seed pre-commits output phase (E29).",
     "script": "experiments/e15_phase_clusters.py", "doc": None, "results": "e15", "image": None},
    # ---- benchmarking + port ----------------------------------------------
    {"id": "E16", "title": "SBN fidelity vs training-free guidance baselines (Flux)", "thread": "spectral-power",
     "models": "FLUX.1-dev", "status": "pending",
     "motivation": "Practice uses high CFG; benchmark SBN fidelity against training-free guidance.",
     "method": "Compare SBN vs guidance baselines on fidelity + prompt-adherence metrics.",
     "result": "Flux's distilled guidance makes the high-CFG regime odd; full scored run pending.",
     "verdict": "The contest is FIDELITY, not adherence -- motivates the SD3.5 port.",
     "nxt": "Re-run on a true-CFG model, SD3.5 (E17).",
     "script": "experiments/e16_baselines.py", "doc": "docs/experiment-reports/EXPERIMENT_16.md", "results": None, "image": None},
    {"id": "E17", "title": "SD3.5 port (true CFG): SBN vs CFG-Zero* + CompBench harness", "thread": "spectral-power",
     "models": "SD3.5-medium", "status": "pending",
     "motivation": "Flux's distilled guidance is odd; port the methods to a true-CFG model.",
     "method": "SD3.5-medium VAE encode/decode + gen helpers (reused by E18-E22); 8-condition "
               "fidelity + T2I-CompBench B-VQA drivers.",
     "result": "Backend + harness written and reused downstream; results/e17 run pending.",
     "verdict": "The SD3.5 base camp for the style + benchmark work.",
     "nxt": "Run the scored conditions; feed real-SBN target into SD3.5 VAE space (E23).",
     "script": "experiments/e17_sd35_compare.py", "doc": "docs/experiment-reports/EXPERIMENT_17.md", "results": "e17", "image": None},
    # ---- style -------------------------------------------------------------
    {"id": "E18", "title": "Offline two-image spectral recombination (AdaIN-in-Fourier)", "thread": "style",
     "models": "SD3.5 / Flux VAE", "status": "mapped",
     "motivation": "Before generation: can phase A + power B recombine two *real* images?",
     "method": "VAE-encode A (content) & B (style), recombine spectra (restyle/swap/hybrid), decode.",
     "result": "Restyle keeps A's layout (clip->A 0.90-0.97) and moves palette toward B; on SD3.5 "
               "it ~halves the spectral distance to a painting -- but transfers tone, not strokes.",
     "verdict": "AdaIN-in-Fourier = real spectral *tone/palette* transfer; VAE-dependent (SD3.5).",
     "nxt": "Do it generation-time (E19).",
     "script": "experiments/e18_spectral_recombine.py", "doc": "docs/experiment-reports/EXPERIMENT_18.md", "results": "e18", "image": None},
    {"id": "E19", "title": "Generation-time spectral style transfer", "thread": "style",
     "models": "SD3.5-medium", "status": "pending",
     "motivation": "Generate a content prompt while clamping its spectrum toward a style image.",
     "method": "ClampPSD3 with a style-band reference (content phase + per-step energy, style envelope).",
     "result": "Model-free preflight passes (strength=0 == SBN); gen/score needs the SD3.5 run.",
     "verdict": "Headline of the style thread; code-complete, awaiting cluster.",
     "nxt": "Run gen/score; add hybrid / morph / two-prompt modes (operators exist).",
     "script": "experiments/e19_spectral_style.py", "doc": "docs/experiment-reports/EXPERIMENT_18.md", "results": "e19", "image": None},
    {"id": "E20", "title": "Spectral warm-start (skip the beginning of generation)", "thread": "style",
     "models": "SD3.5-medium", "status": "pending",
     "motivation": "If low-band phase locks in early, can we inject it and skip early steps?",
     "method": "Profile within-trajectory per-band phase convergence (lock-in); oracle re-entry "
               "via Img2Img from a band-pre-set intermediate latent.",
     "result": "Phase-convergence lock-in metric + oracle ceiling built; full gen run pending.",
     "verdict": "Warm-start is plausible (low band locks first); needs the timing run.",
     "nxt": "Measure real step savings vs the oracle ceiling.",
     "script": "experiments/e20_warmstart.py", "doc": "docs/experiment-reports/EXPERIMENT_20.md", "results": "e20", "image": None},
    {"id": "E21", "title": "RF-inversion frequency-band editing (SD3.5) -- gate fails", "thread": "style",
     "models": "SD3.5-medium", "status": "dead-end",
     "motivation": "Edit a real photo: invert to noise, regenerate under a new prompt, lock source bands.",
     "method": "Rectified-flow ODE inversion (naive + fixed-point), then band-lock + new prompt.",
     "result": "Reconstruction GATE fails: RF inversion on SD3.5 drifts; editing is moot until it holds.",
     "verdict": "RF inversion on SD3.5 is unreliable -- pivot to a model where DDIM inversion works.",
     "nxt": "Redo on SDXL with DDIM inversion (E22).",
     "script": "experiments/e21_spectral_edit.py", "doc": "docs/experiment-reports/EXPERIMENT_21.md", "results": "e21", "image": None},
    {"id": "E22", "title": "DDIM-inversion frequency-band editing (SDXL pivot)", "thread": "style",
     "models": "SDXL", "status": "mapped",
     "motivation": "E21 stalled on inversion; SDXL (eps-pred) inverts reliably with DDIM.",
     "method": "DDIM-invert a photo, regenerate under a new prompt while locking source phase/power bands.",
     "result": "Recon CLIP-I ~0.94 (gate passes); low-band phase-lock holds composition (struct ~0.90) "
               "but trades down edit strength; power-lock fails to hold layout.",
     "verdict": "A real structure<->edit frontier; phase-lock = composition, power-lock != layout.",
     "nxt": "Tune the lock/strength dials; try anisotropic bands for true strokes.",
     "script": "experiments/e22_ddim_edit.py", "doc": "docs/experiment-reports/EXPERIMENT_22.md", "results": "e22", "image": None},
    # ---- spectral-power payoff --------------------------------------------
    {"id": "E23", "title": "Real-image spectral target (“real-SBN”)", "thread": "spectral-power",
     "models": "FLUX.1-dev", "status": "active",
     "motivation": "Stop clamping toward the weak cfg=1 proxy; clamp toward the spectrum of real photos.",
     "method": "Build a per-channel real-PSD target from 500 MS-COCO photos; psd_match generated latents "
               "toward it (phase kept), offline / last-step / init-noise.",
     "result": "Gap is bimodal (low-freq excess + broad high-freq deficit); real-SBN gives the biggest "
               "aesthetic gain at ~0 adherence cost and beats cfg-1 SBN; s~0.25 is the sweet spot; "
               "init-noise shaping fails.",
     "verdict": "The live payoff of the SBN line -- real-photo target is the right one.",
     "nxt": "Bake a fixed per-channel correction curve; confirm B-VQA adherence; port to SD3.5 VAE.",
     "script": "experiments/e23_real_sbn.py", "doc": "docs/experiment-reports/EXPERIMENT_23.md", "results": "e23", "image": None},
    # ---- text-frequency ----------------------------------------------------
    {"id": "E24", "title": "Token-axis FFT on the TEXT conditioning (FNet-motivated)", "thread": "text-freq",
     "models": "FLUX.1-dev", "status": "mapped",
     "motivation": "Move the spectral idea onto the text embedding: FFT along the token axis.",
     "method": "1-D FFT over T5 tokens; isolate / swap / blend low vs high token-frequency bands on Flux.",
     "result": "Bands meaningful & on-manifold (low ~ subject, high ~ style); MERGE is negative (snaps "
               "to the low-band/phase owner, doesn't beat a lerp); EDIT partial (high-band style knob); "
               "token phase ~ identity.",
     "verdict": "Token-frequency bands are real and editable; merging two spectra is not the win.",
     "nxt": "Make it a continuous knob and characterise each band (E30).",
     "script": "experiments/e24_text_spectral.py", "doc": "docs/experiment-reports/EXPERIMENT_24.md", "results": None, "image": None},
    # ---- seed steering (dead end) -----------------------------------------
    {"id": "E25", "title": "Seed-alignment pilot: bias the seed toward the prompt (SD1.5)", "thread": "seed",
     "models": "SD1.5", "status": "dead-end",
     "motivation": "Can a gentle seed optimisation improve adherence while holding Gaussian moments?",
     "method": "Latent-mode CLIP objective on the seed, re-standardised to ||z||=sqrt(d).",
     "result": "A gentle, do-no-harm palette/composition lever -- but only a lever, not a fix.",
     "verdict": "Latent-mode is the gentlest variant; effect is mild.",
     "nxt": "Scale up on SDXL + DPG-Bench and sweep cost (E26).",
     "script": "experiments/e25_seedalign.py", "doc": "docs/experiment-reports/EXPERIMENT_26.md", "results": "e25", "image": None},
    {"id": "E26", "title": "Seed-alignment on SDXL + DPG-Bench + step sweep", "thread": "seed",
     "models": "SDXL", "status": "dead-end",
     "motivation": "Does the seed lever hold up at 1024px on a real adherence benchmark?",
     "method": "SDXL port of E25 + DPG-Bench scoring + an N-optimisation-steps sweep.",
     "result": "Break-even at best -- N=1 (barely touched) is as good as heavier optimisation.",
     "verdict": "More seed optimisation does not buy more adherence.",
     "nxt": "Distil to a single reusable direction (E27).",
     "script": "experiments/e26_seedalign_sdxl.py", "doc": "docs/experiment-reports/EXPERIMENT_26.md", "results": "e26", "image": None},
    {"id": "E27", "title": "A single “concept direction” in the seed (CLIP->latent pullback)", "thread": "seed",
     "models": "SDXL", "status": "dead-end",
     "motivation": "Replace per-prompt optimisation with one additive seed direction.",
     "method": "Two-stage pullback: CLIP gradient x decoder Jacobian (= chain rule); anchor sweep.",
     "result": "Anchor-independent but too blunt; iterative use shifts palette, not composition.",
     "verdict": "A single direction is too coarse for compositional control.",
     "nxt": "Run the decisive regime test on hard compositional failures (E28).",
     "script": "experiments/e27_seeddir.py", "doc": "docs/experiment-reports/EXPERIMENT_27.md", "results": "e27", "image": None},
    {"id": "E28", "title": "Does seed-biasing RESCUE dropped compositional elements?", "thread": "seed",
     "models": "SDXL", "status": "dead-end",
     "motivation": "The decisive test: on CompBench failures, does seed-bias recover missing elements?",
     "method": "T2I-CompBench B-VQA on failing prompts; gradient seed-bias vs a plain re-roll control.",
     "result": "Seed-bias LOSES to re-roll (seed-dependent recovery .57 vs .43/.29) and breaks passers.",
     "verdict": "Seed-as-adherence is a DEAD END; best-of-N + a picker wins.",
     "nxt": "Stop; the residue (seed->output determinism) feeds E29.",
     "script": "experiments/e28_seedrescue.py", "doc": "docs/experiment-reports/EXPERIMENT_28.md", "results": "e28", "image": None},
    # ---- phase: the diffusion map -----------------------------------------
    {"id": "E29", "title": "Phase inheritance: does the seed's phase fix the output's?", "thread": "phase",
     "models": "SD1.5", "status": "mapped",
     "motivation": "How much of the output latent's phase is inherited from the seed under DDIM?",
     "method": "Per-band circular correlation seed-phase vs output-phase over many seeds + CFG sweep; "
               "causal phase transplant with a follow score.",
     "result": "Strong BROAD-spectrum inheritance (phase ~0.4, magnitude >= phase, pixel r ~0.76, null "
               "~0); CFG erodes it most in low-freq bands; transplant confirms causality (follow ~0.66).",
     "verdict": "The seed fixes the WHOLE output spectrum at low CFG -- not a phase-specific channel.",
     "nxt": "Repeat on Flux/SD3.5 (rectified flow) for architecture-independence.",
     "script": "experiments/e29_phase_inherit.py", "doc": "docs/experiment-reports/EXPERIMENT_29.md", "results": "e29", "image": None},
    # ---- text-frequency follow-ups ----------------------------------------
    {"id": "E30", "title": "Continuous text-frequency control & extraction", "thread": "text-freq",
     "models": "FLUX.1-dev", "status": "mapped",
     "motivation": "Turn E24's discrete band ops into a continuous knob and characterise each band.",
     "method": "band_gain_1d (continuous attenuate/amplify) + band_notch_1d (per-band knockout); "
               "image strips as the knob varies; CLIP-T / sharpness / hf-frac / colourfulness / aesthetic / B-VQA.",
     "result": "Ran on runai. Token spectrum IS structured: phase carries the content (phase_only~full, "
               "mag_only collapses); low band = coarse gist, mid+high bands = attribute-object binding "
               "(low-pass kills B-VQA, notch-lo keeps it). But no single band is load-bearing, and spectral "
               "blending still loses to literally writing 'A and B' (concat B-VQA 0.85 vs merges ~0).",
     "verdict": "Spectrum characterised and structured, but blending is descriptive, not a better control knob.",
     "nxt": "Optional VQAScore corroboration; structure understood, no new control lever here.",
     "script": "experiments/e30_text_freq_control.py", "doc": "docs/experiment-reports/EXPERIMENT_30.md", "results": "e30", "image": None},
    {"id": "E31", "title": "Real-image editing via FlowEdit + frequency-surgery conditioning", "thread": "text-freq",
     "models": "FLUX.1-dev", "status": "dead-end",
     "motivation": "Use token-frequency surgery as the target conditioning inside inversion-free editing.",
     "method": "FlowEdit (ODE delta integration, no inversion); target conditioning = band_swap(low:src, high:style); "
               "VAE-encode real input; a skip knob for edit strength.",
     "result": "Ran on runai. Recon identity holds (px-dist ~0.003, gate passed) and plain prompt-swap FlowEdit "
               "edits (scene-dependent). But frequency-surgery target conditioning barely moves the image: the "
               "kept low band anchors to the source so v(C_tar)-v(C_src)~0 => delta~0. High-band style injection "
               "too weak to redirect the flow.",
     "verdict": "Token-frequency surgery does not drive inversion-free editing; can't out-edit a plain prompt swap.",
     "nxt": "Closes the text-freq editing route; latent-band editing (E22) remains the usable handle.",
     "script": "experiments/e31_flowedit_freq.py", "doc": "docs/experiment-reports/EXPERIMENT_31.md", "results": "e31", "image": None},
    {"id": "E32", "title": "Per-object token-frequency control on two-object prompts", "thread": "text-freq",
     "models": "FLUX.1-dev", "status": "mapped",
     "motivation": "E30's band gain is global; can we boost/cut ONE object's frequencies and have it be "
                   "selective to that object?",
     "method": "Map each object phrase to its T5 token span (offset mapping); windowed FFT over the span "
               "(apply_on_subspan + band_gain_1d), median split cut=0.51; targeted vs a global-gain "
               "control; 10 two-object prompts x 3 seeds (n=60/cell). Metric: per-object CLIP/B-VQA "
               "selectivity (Delta_target - Delta_other), paired to baseline.",
     "result": "Ran on runai. Object-SELECTIVE and steerable in CLIP: boosting an object's band raises ITS "
               "CLIP and lowers the other's; cutting reverses it, both bands, sign tracks gain (sel "
               "+0.005..-0.008, t up to 3.1). The global-gain control is a null (sel~0, no sign pattern) -> "
               "localisation, not gain, drives it. B-VQA presence shifts in-direction for the HIGH band "
               "(boost target +0.04 / cut -0.05, echoing E30 high=binding) but is noisy (|t|<=1.8 at n=60). "
               "Effect size small (~0.005 CLIP on a ~0.22 baseline).",
     "verdict": "First CONTROLLABLE per-object text lever (beats E24-MERGE/E31 nulls), but a weak one; "
                "high band carries the binding effect.",
     "nxt": "Strengthen with longer object phrases (more bins) / larger N for presence significance; then "
            "the TI (E33) and channel-axis (E34) follow-ups.",
     "script": "experiments/e32_object_freq.py", "doc": "docs/experiment-reports/EXPERIMENT_32.md", "results": None, "image": None},
    {"id": "E33", "title": "Textual inversion of an object, then frequency-control its span (proposed)",
     "thread": "text-freq", "models": "SDXL / SD1.5 (TI tooling safer than Flux)", "status": "pending",
     "motivation": "Learn an embedding for a pseudo-token <obj> from a few images, then boost/cut the "
                   "token-frequencies of ITS span in a multi-object prompt.",
     "method": "PROPOSED -- not yet implemented. No TI scaffolding exists in the repo; would use diffusers "
               "load_textual_inversion or a custom loop adapted from the E25/E26 seed-opt loops, then reuse "
               "E32's span-windowed band_gain.",
     "result": "—", "verdict": "—",
     "nxt": "Implement after E32 reports; pick SDXL/SD1.5 for mature TI tooling.",
     "script": None, "doc": "docs/experiment-reports/EXPERIMENT_32.md", "results": None, "image": None},
    {"id": "E34", "title": "Channel-axis (D=4096) interpretability of the text embedding (proposed)",
     "thread": "text-freq", "models": "FLUX.1-dev", "status": "pending",
     "motivation": "Which CHANNELS of the T5 embedding own which attributes (identity/color/texture/style), "
                   "so they can be steered directly for editing/generation?",
     "method": "PROPOSED -- not yet implemented. Attribute probing (per-channel variance/correlation with an "
               "attribute label) + causal ablation (zero/scale channels, score with CLIP/B-VQA); E24 noted the "
               "hidden axis is NOT semantically ordered, so likely learned channel directions, not raw indices. "
               "Composes with E32 span masking for per-object x per-channel edits.",
     "result": "—", "verdict": "—",
     "nxt": "Implement after E32; complements the frequency knob with a channel knob.",
     "script": None, "doc": "docs/experiment-reports/EXPERIMENT_32.md", "results": None, "image": None},
    {"id": "E35", "title": "Token-frequency operator sweep on SD1.5 (scenarios x operators x params)",
     "thread": "text-freq", "models": "SD1.5", "status": "mapped",
     "motivation": "Systematically characterise the WHOLE token-freq operator toolkit: per operator x "
                   "parameter x prompt-type, what happens to adherence and fidelity?",
     "method": "All 13 ops on SD1.5; 25 prompts across 5 categories (short/long/style/object/two-object), "
               "5 seeds, dense param grids (1001 conditions, 5005 imgs). Metrics: CLIP-T adherence, LAION "
               "aesthetic + image-stats (fidelity), baseline-drift (CLIP image-image).",
     "result": "Ran on runai. PHASE >> MAGNITUDE replicates on SD1.5/CLIP-77: phase-only 0.187 CLIP / 4.47 "
               "aesthetic beats mag-only 0.145 / 3.85 (mag-only also drifts most, 0.43); gap largest on "
               "long/compositional prompts. Localized/interp edits gentlest (per-object drift 0.09 ~baseline, "
               "lerp 0.13). High-pass > low-pass on adherence (0.224 vs 0.174). Aggressive single-band "
               "surgery (lowpass/notch/phasekeep/phasegain/magonly) costs BOTH adherence and fidelity.",
     "verdict": "Toolkit mapped on SD1.5: phase carries content (cross-arch confirm of E30); per-object/lerp "
                "are do-little-harm, most band surgery degrades.",
     "nxt": "Lift the phase>mag + high-vs-low findings back to Flux; feeds E33/E34.",
     "script": "experiments/e35_op_sweep.py", "doc": "docs/experiment-reports/EXPERIMENT_35.md", "results": None, "image": None},

    {"id": "E37", "title": "Velocity spectral normalization (CFG velocity → cfg=1 amplitude, SD3.5)",
     "thread": "spectral-power", "models": "SD3.5-medium", "status": "mapped",
     "motivation": "SBN pulls the spectrum toward the natural cfg=1 spectrum, but the demo's every-step "
                   "SBN→real clamped a FIXED clean-image target — scale-correct only at the last step. "
                   "Fix the object AND the reference: edit the flow-matching VELOCITY and clamp toward the "
                   "SAME-STEP unconditional velocity v_∅, which CFG already computes (on-manifold, one pass).",
     "method": "Real CFG (v_w = v_∅ + w(v_c−v_∅)). e17_sd35.gen_sd3-style interception: record batched "
               "[v_∅,v_c], edit model_output (=v_w) before the Euler step. mag transplant |V_w|←|V_∅| "
               "(keep phase) on a radial band. GenEval (553 prompts, n=1, 512px, w=4.5); GenEval protocol "
               "with a torchvision Mask R-CNN detector + CLIP colours (ranking-faithful, not Mask2Former).",
     "result": "Band-dependent (GenEval macro): baseline 0.644, mag_top25 [0.75,1] 0.655 (slight WIN, "
               "color_attr 0.48→0.54), mag_bot25 [0,0.25] 0.561 and mag_full [0,1] 0.524 (HURT). Low-freq "
               "velocity magnitude carries adherence/composition; high-freq is CFG's correctable "
               "over-amplification. Caveat: n=1 (+0.011 within seed noise; pattern coherent).",
     "verdict": "Touch only the HIGH band: high-freq velocity normalization toward cfg=1 is "
                "free-to-beneficial; low/full bands erode compositional adherence.",
     "nxt": "Multi-seed (n=4) + high-band cut sweep + band-amplify/late-window + official Mask2Former scorer; DPG-Bench.",
     "script": "experiments/e37_geneval.py", "doc": "docs/experiment-reports/EXPERIMENT_37.md",
     "results": "e37", "image": None},

    {"id": "E38", "title": "Frequency DIRECTION of CFG — paired magnitude + phase along the trajectory (FLUX.1-dev)",
     "thread": "spectral-power", "models": "FLUX.1-dev", "status": "pending",
     "motivation": "E7 found cfg=1 vs 3.5 latents differ mainly in POWER while the *marginal* phase stats look "
                   "identical — but a uniform marginal histogram does NOT mean phase is untouched. Same prompt+seed "
                   "makes the two latents point-wise comparable, so ask the PAIRED question: does raising cfg rotate "
                   "each Fourier coefficient's phase in a coherent, band-specific way (a real phase direction a "
                   "histogram would miss), or are the rotations random?",
     "method": "cfg ∈ {1.0,3.5,7.0}, same 10 prompts/seed, full per-step latent trajectory. Per radial band: "
               "magnitude direction = d log-power/d cfg (LS slope); phase coherence = magnitude-weighted "
               "|Σ w_k r_k|/Σ w_k (1 = every coeff rotates the same angle, ~1/√N = random); dominant rotation angle. "
               "Binned early/mid/late.",
     "result": "Run artifacts were not persisted from the cluster — no saved results to report.",
     "verdict": "OPEN — code complete, outputs not saved; rerun needed to settle whether CFG has a coherent per-band phase direction.",
     "nxt": "Rerun on cluster and persist results/e38; if a phase direction exists, fold it into the velocity-normalization line (E37).",
     "script": "experiments/e38_cfg_direction.py", "doc": None, "results": None, "image": None},

    {"id": "E39", "title": "Spectral band-AdaIN — soft-radial-band magnitude (mean+std) knob in the sampler",
     "thread": "style", "models": "FLUX.1-dev (interactive demo)", "status": "mapped",
     "motivation": "Generalize E18's Fourier-AdaIN and the E8/E23 SBN into one sampler-side operator (outside the "
                   "network) that rewrites per-band magnitude toward a chosen source while reusing content phase — a "
                   "pure frequency knob, orthogonal to the network's semantic AdaLN.",
     "method": "Soft radial Gaussian-ring bands forming a partition of unity (∑ m_k = 1); per band normalize |V| by "
               "mask-weighted moments and rewrite to the source's mean AND std; reuse content phase; restore the "
               "self-conjugate bins so ifft.real loses ~1e-8. Exposed as a single-pass self-AdaIN (global / 3-band) "
               "in the Spectral AdaIN demo tab via adain_affine.",
     "result": "Operator verified real (~1e-8 round-trip) and interactive; lives in the demo tab, no batch metrics saved.",
     "verdict": "A clean soft-band magnitude knob (mean+std) generalizing SBN+AdaIN; works as a live frequency dial, not yet benchmarked.",
     "nxt": "Quantify vs E18/E23 on adherence/fidelity; exercise the learned BandSchedule.",
     "script": "experiments/e39_spectral_adain.py", "doc": "docs/experiment-reports/EXPERIMENT_39.md", "results": None, "image": None},

    {"id": "E40", "title": "RF inversion + trajectory-matched low-band spectral clamp (real-image editing, FLUX)",
     "thread": "style", "models": "FLUX.1-dev", "status": "active",
     "motivation": "Edit a real image with FLUX while preserving structure under an aggressive edit prompt. Improve on "
                   "BandLock (E21/E22), which clamps to a single FIXED source latent x0, by clamping to the σ-aligned "
                   "inversion *trajectory* instead.",
     "method": "RF-invert backwards (σ:0→1) under the source caption, recording traj[i] at every σ node; regenerate "
               "forward (σ:1→0) under the edit prompt; at each step clamp the latent's low band [0,cut] toward traj[i] "
               "at the matching σ. Three modes reuse repo primitives — sbn (per-band power), phase (power + low-band "
               "phase lock), adain (mean+std).",
     "result": "σ-aligned trajectory reference keeps coarse layout while the high band follows the edit (default sbn "
               "cut=0.25, strength=0.5). Interactive in the RF inversion demo tab; no saved results/ dir.",
     "verdict": "Trajectory-matched low-band clamp preserves structure where fixed-x0 BandLock (E21/E22) drifted; current live demo feature.",
     "nxt": "Quantify the structure/edit trade-off across cut/strength and the three modes; metricize vs E21/E22.",
     "script": "experiments/e40_spectral_invert.py", "doc": "docs/experiment-reports/EXPERIMENT_40.md", "results": None, "image": None},

    {"id": "E41", "title": "Calibrating the RF-inversion spectral-clamp edit vs a fair RF-inv (eta) baseline",
     "thread": "style", "models": "FLUX.1-dev", "status": "mapped",
     "motivation": "E40 gave RF-inversion + a low-band velocity spectral clamp as a real-image editor, but with "
                   "hand-set global knobs and no fair RF-inversion baseline. Can per-image calibration match/beat "
                   "RF-inversion at EQUAL editability, and does one global knob suffice?",
     "method": "Factor the RF-invert + spectral-clamp edit out of the demo into invert_core (+ an RF-inversion eta "
               "controller v+=eta*(v_target-v) for a true baseline). struct_metrics.py = DINO self-similarity "
               "structure distance + CLIP-directional + masked BG PSNR/LPIPS. Stratified ~140-image PIE-Bench++ "
               "loader (with masks). Optuna TPE per-image active calibration (min DINO struct s.t. CLIP-dir >= "
               "vanilla) plus a fixed 54-point global-knob grid, both placed on the RF-inv eta Pareto frontier. "
               "Self-gating sharded RunAI orchestration.",
     "result": "Ran on runai (140 PIE-Bench imgs). Beats VANILLA RF-inversion on every metric (DINO struct 0.162 "
               "vs 0.199, LPIPS 0.50 vs 0.60, CLIP-dir 0.140 vs 0.123). At matched editability vs the full eta "
               "sweep it is ~tied (gap ~0, wins 31/63) and edits beyond RF-inv's eta range on 77/140. A single "
               "grid-picked global knob sits near the per-image oracle -> a deployable fair-comparison point.",
     "verdict": "The low-band spectral-clamp edit is a legitimate RF-inversion-class editor: strictly beats vanilla "
                "RF-inv and ties the eta frontier at matched editability, with a usable global knob.",
     "nxt": "Tie at matched editability => need structure headroom: gate the clamp by structure (E42), or drop "
            "inversion entirely for FlowEdit/FlowAlign (E43).",
     "script": "experiments/e41_calibrate.py", "doc": None, "results": "e41", "image": None},

    {"id": "E42", "title": "DINOv2-structure-gated spectral clamp (lock background, free foreground)",
     "thread": "style", "models": "FLUX.1-dev", "status": "dead-end",
     "motivation": "E41's clamp locks the WHOLE low band uniformly. Can a DINOv2 saliency gate preserve structure "
                   "MORE by locking the background hard while freeing the foreground to edit?",
     "method": "Gate E41's low-band clamp by a DINOv2 saliency map: lat = G*clamped + (1-G)*current, G in [0,1] "
               "(worktree e42-dino-gate, cluster job e42h). 30 PIE-Bench images, fixed dancers config; scored with "
               "struct_metrics (DINO struct / CLIP-dir / LPIPS / DSSIM).",
     "result": "Ran on runai. More editability but WORSE structure vs the global clamp: CLIP-dir 0.105->0.118 "
               "(wins 19/30) but DINO struct 0.187->0.199 (wins only 5/30), LPIPS/DSSIM also worse (~25-27/30). "
               "Structural, not tuning: a gate in [0,1] can only RELAX the global lock, never strengthen it, so no "
               "<=1 spatial gate can beat a full low-band lock on structure.",
     "verdict": "NO-GO for 'preserve structure more'. To preserve structure MORE you must STRENGTHEN the clamp where "
                "structure lives (gate able to widen band / add steps), starting from a partial-clamp baseline.",
     "nxt": "Abandon the gate-down approach; the structure win instead comes from inversion-free FlowAlign (E43).",
     "script": None, "doc": None, "results": None, "image": None},

    {"id": "E43", "title": "FlowAlign on FLUX + spectral terminal-point variants",
     "thread": "style", "models": "FLUX.1-dev", "status": "active",
     "motivation": "FlowAlign (arXiv:2505.23145) = inversion-free FlowEdit + a source-consistency TERMINAL-POINT "
                   "term, CFG with the source prompt as the negative. Can a spectral twist on it beat plain "
                   "FlowAlign on structure preservation WITHOUT losing edit adherence?",
     "method": "Port FlowAlign to FLUX in invert_core.flowalign (shared by the demo's FlowAlign tab + the e43 "
               "harness, 3 velocity forwards/step). Two twists, both identity at defaults: (1) SBN on the CFG "
               "reference -- clamp the CFG velocity vp's low radial band toward v(pt,c_src), modes band-power / mag "
               "/ phase / both (reuses E37 velocity_spectral_ops); (2) annealed terminal point -- low-pass the "
               "consistency vector coarse->fine over steps. Small qualitative sweep: 3 scenes x w in {5,7,10}, "
               "28 steps; scored with struct_metrics (DINO struct, CLIP-directional, LPIPS).",
     "result": "Ran on runai. Identity gate holds (recon struct ~0.003-0.005). sbn_phase (low-band phase-lock, "
               "cut=0.2) BEATS plain FlowAlign on all 3 scenes at every w -- roughly halves DINO structure distance "
               "(e.g. 0.056 vs 0.124) while RAISING CLIP-directional (mean dStruct -0.055..-0.061, dClip "
               "+0.037..+0.085). sbn_bp wins 2/3 (3/3 at w=10); annealed terminal point is a null.",
     "verdict": "First editing lever that preserves structure MORE than the baseline without an editability cost: "
                "low-band phase-lock of the CFG velocity strictly beats FLUX-FlowAlign on structure AND editability.",
     "nxt": "Confirm on the full 700-image PIE-Bench set (+ masks for BG-PSNR/BG-LPIPS); SD3.5 port; sweep "
            "sbn_cut / phase strength to map the structure/editability frontier.",
     "script": "experiments/e43_flowalign.py", "doc": "docs/experiment-reports/EXPERIMENT_43.md", "results": None, "image": None},

    {"id": "E44", "title": "Apples-to-apples FlowAlign reproduction (+ ours) on PIE-Bench (SD3-medium)",
     "thread": "style", "models": "SD3-medium (official FlowAlign)", "status": "active",
     "motivation": "Validate the E43 win rigorously: first REPRODUCE FlowAlign's published PIE-Bench table on "
                   "SD3-medium with their official code (hard gate), then port our spectral phase-clamp into the "
                   "SAME SD3 FlowAlign loop and show lower Structure Distance at matched edited-CLIP.",
     "method": "Official SD3FlowAlign sampler + official PnPInversion metrics (Structure Distance, bg-masked "
               "PSNR/LPIPS/MSE/SSIM, whole + edited CLIP) on PIE-Bench (cached HF++ variant, RLE masks; CLIP forced "
               "to ViT-base-patch16 to match the paper). Curve-based win criterion: sweep CFG ω∈{5,7.5,10,13.5} and "
               "compare struct-dist vs edited-CLIP curves (Fig 3a). HP tuned on a disjoint Emu-Edit subset.",
     "result": "In progress. Foundation smoke (bicycle, cfg13.5) PASS; 20-img mini pipeline PASS with numbers in "
               "PIE-Bench range (struct 12.4e-3, bgPSNR 28.2, CLIP-edit 22.1, ~9s/edit). Reproduction target from "
               "arXiv App. E (cfg10, SD3.0): FlowAlign struct 0.028 / bgPSNR 25.5 / CLIP-edit 22.0. Full "
               "{5,7.5,10,13.5}×700 sweep running on runai.",
     "verdict": "IN PROGRESS — reproduction gate not yet cleared; target numbers in hand, cfg sweep running with "
                "base16-CLIP analyze. (Then port sbn_phase into the official SD3 loop.)",
     "nxt": "Land on FlowAlign's Fig-3a curve at cfg10/700, then port the E43 sbn_phase clamp into the official SD3 "
            "FlowAlign loop and compare curves at matched edited-CLIP.",
     "script": "experiments/e44_flowalign_repro.py", "doc": "docs/experiment-reports/EXPERIMENT_44.md",
     "results": "e44", "image": None},

    {"id": "E45", "title": "FlowAlign on LTX-Video + spatiotemporal phase op (temporal video editing)",
     "thread": "style", "models": "LTX-Video", "status": "active",
     "motivation": "FlowAlign (arXiv:2505.23145) edits VIDEO frame-by-frame on an IMAGE model (SD3) and admits "
                   "'temporal consistency for the edited object is limited'. Does running FlowAlign on a real video "
                   "model (LTX) + our E41/E43 low-band PHASE-keep op in the SPATIOTEMPORAL (3D) frequency domain "
                   "fix that flicker?",
     "method": "Port FlowAlign to LTX-Video (velocity/pack/VAE/sigma, 3 forwards/step) in e45_ltx_flowalign.py. One "
               "LTX-generated clip (toy car->tank). Conditions: the paper's frame-by-frame baseline (fbf), the "
               "FlowAlign-on-LTX video baseline, and the phase op in 2D (per-frame, ~paper) vs 3D (spatiotemporal) "
               "over an sbn_cut sweep. Metric bundle: DINO struct-dist + CLIP-directional (per-frame avg) + RAFT "
               "warp-error (global & edited-region-masked). 256/512px, 25-49 frames.",
     "result": "Identity gate holds (recon L1 ~0.004-0.005). Frame-by-frame (paper) flickers hard (warp-masked "
               "0.052); FlowAlign-on-LTX video editing is ~0.0011 -- 46x less flicker. The 3D spatiotemporal phase "
               "op further cuts warp vs the video baseline (0.00112 vs 0.00140, -20%) while 2D per-frame does NOT; "
               "all phase variants improve DINO structure (~0.139 vs 0.149). Phase costs editability though "
               "(CLIP +0.03 vs +0.084), coupled to the gain at every cut. LEVERS: a w-frontier (7.5/10/13.5/18) "
               "shows video warpM stays ~0.0012-0.0019 at every w while fbf climbs 0.038->0.097 -- video dominates "
               "the editability-vs-flicker frontier (video editability saturates ~+0.085, can't reach fbf's "
               "+0.12-0.18). On a REAL clip @512 (cockatoo->parrot) the baseline genuinely flickers (warpM 0.042) "
               "and phase3d cuts it -13% (0.0364 vs 0.0419) + better structure -- the clearest spatiotemporal win.",
     "result_correction": "DISTORTION DIAGNOSIS + native-res rerun (S8/S9): the earlier clips were edited at "
                "256/512 SQUARE res, which distorts LTX (it wants larger non-square frames). A faithful FlowEdit "
                "baseline and FlowAlign both render clean at native 704x480. Re-running the headline numbers at "
                "native res: toy@704x480 -- video 7.1x less flicker than frame-by-frame, phase improves struct+CLIP "
                "(goal PASS). BUT real cockatoo@448x768 portrait -- video baseline flickers MORE than frame-by-frame "
                "(warpM 0.039 vs 0.029) and the phase op gives NO temporal benefit. The 46x flicker win and the "
                "3D-phase temporal edge were RESOLUTION ARTIFACTS.",
     "verdict": "CORRECTED (native res): the temporal claims do NOT generalize. On generated/easy content the video "
                "edit is temporally smoother than frame-by-frame and the phase op improves structure+editability; on "
                "REAL footage with motion the video edit flickers as much/more than frame-by-frame and the phase op "
                "gives no temporal benefit. The reliable finding is a small, consistent STRUCTURE-preservation "
                "improvement from the phase op (as in E43). Temporal hypothesis = KILL as a general claim. The port "
                "is correct (identity recon ~0.005); distortion was resolution, not a bug. Demo: --model ltx.",
     "nxt": "Do NOT publish the temporal win. For any temporal claim: multi-clip real-video set (DAVIS) + a "
            "perceptual flicker metric (source-flow warp is confounded by the edited object). The phase op's honest "
            "value is structure preservation -- evaluate THAT on a real-video edit benchmark.",
     "script": "experiments/e45_ltx_flowalign.py", "doc": None, "results": None, "image": None},

    {"id": "E46", "title": "Seed-phase fast editing -- 0-NFE phase prior vs SDEdit",
     "thread": "seed", "models": "SDXL", "status": "dead-end",
     "motivation": "Inversion editors pay many NFE; FlowEdit/FlowAlign cost >2 NFE/step; SDEdit is cheap but "
                   "unreliable. Transplant the source's FFT PHASE (where structure lives) into a fresh seed for "
                   "~0 NFE, then run one fast generation toward the target -- can a free phase prior make SDEdit "
                   "reliable (phase=structure, fresh magnitude=editability)?",
     "method": "SDXL. Source low-band phase on a white seed (phase_swap_2d), scored DINO-struct x CLIP-directional. "
               "Derivations: averaging noised copies -> phase(x0) exactly (loop dominated); 100% pass is empty "
               "(q(x_T) indep of x0); whitening phase == destroying structure (same axis); OOD-ness is phase "
               "coherence, not the spectrum. Probes: P0 reconstruction mechanism; P1 editing frontier (8 SDXL-gen "
               "sources, recipes A=phase-noise SDEdit, B=structured seed) vs vanilla SDEdit; P2 full-band phase "
               "(Cfull) + phase-normalize (Cnorm); P3 three OOD escapes -- gamma phase-whiten, timestep injection, "
               "colored amplitude -- and soft (gamma-blended) timestep injection. PIE-Bench deferred to cluster.",
     "result": "P0: seed low-band phase controls layout (phaseB beats white 12/12 seed pairs; exact pose/arrangement "
               "transfer) -- mechanism REAL. P1: neither recipe beats vanilla (A over-locks, editability collapses; "
               "B Pareto-dominated; 0/8 wins). P2: Cfull preserves WORSE than low-band (white-amp + full image phase "
               "= OOD fringing); Cnorm = e^{i.phi_src}.conj(Z) ~ white Gaussian == a random seed. P3: gamma is a "
               "smooth structure<->edit knob (fringing grows with gamma); timestep injection is CLEAN/on-manifold and "
               "structure 0.082 BEATS vanilla 0.093 but over-clamps (edit dies); colored amplitude = rainbow "
               "artifacts (amp must stay white); soft injection (gamma=0.3) best+clean struct 0.081 but editability "
               "caps ~0, never reaching vanilla +0.090.",
     "verdict": "KILL the seed-phase EDITING direction -- 4th confirmation of the E41 frontier-trap: every variant "
                "traces a structure<->editability frontier at-or-inside vanilla SDEdit's, because x0-carry is a "
                "strictly better/cheaper structural anchor than any phase transplant. Mechanism (seed low-band phase "
                "controls layout; clean timestep injection beats vanilla on structure) is REAL and KEPT.",
     "nxt": "Only useful where there is NO x0 to carry (layout-conditioned T2I / cross-modal structure transfer). "
            "If revisited: matched-editability vanilla-strength sweep; confirm P1 on official PIE-Bench (cluster).",
     "script": "experiments/e46_seedphase.py", "doc": "docs/experiment-reports/EXPERIMENT_46.md",
     "results": None, "image": None},
]
