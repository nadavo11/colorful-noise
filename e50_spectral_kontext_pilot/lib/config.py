"""E50 — Spectral Kontext Pilot: global config + subset selection.

Builds directly on E49 (baseline_establishment): reuses its FluxKontext loader, its metric
suite, and the EXACT same source/style images, so E50 results are comparable to the E49 Kontext
baseline by construction.

Focused pilot, not a benchmark sweep. Primary model: FLUX.1-Kontext-dev (4-bit NF4, the E49
substrate). All generation training-free.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]            # e50_spectral_kontext_pilot/
REPO = ROOT.parent                                    # colorful-noise/
E49 = REPO / "baseline_establishment"                 # source of data + reusable lib

# ---- reproducibility ----
PHASE_VERSION = "e50-spectral-kontext-pilot-v1"
SEED = 0
GEN_SIZE = 512            # E50 runs Kontext at 512px for pilot speed (E49 baseline = native 1024;
                          # we regenerate matched-512 baselines so the spectral comparison is fair,
                          # and show the E49 1024 outputs as cross-phase controls).
STEPS = 20
GUIDANCE = 2.5

# ---- paths ----
DATA = ROOT / "data"
SUBSETS = DATA / "selected_subsets"
MANIFESTS = DATA / "manifests"
OUT = ROOT / "outputs"
METRICS = ROOT / "metrics"
FIG = ROOT / "figures"
VIDEO = ROOT / "videos"
REPORTS = ROOT / "reports"
LOGS = ROOT / "logs"

# E49 data roots (shared images)
E49_PIE = E49 / "data" / "benchmark_subsets" / "piebench"
E49_LEAK = E49 / "data" / "custom_leakage_set"
E49_STYLE = E49 / "data" / "style_refs"
E49_OUT = E49 / "outputs"     # for cross-phase Kontext/Redux/StyleID controls

# ---------------------------------------------------------------- subset selection
# Experiment C — SOURCE spectral decomposition on instruction edits.
# 6 PIE-Bench examples spanning the task types E49 flagged as most informative.
EDIT_IDS = [
    "pie_6_0",   # color
    "pie_7_0",   # material
    "pie_1_0",   # object_replace
    "pie_2_1",   # object_add
    "pie_8_0",   # global / background
    "pie_9_1",   # style
]

# Experiment A — spectral REFERENCE composites on adversarial leakage pairs.
# Experiment B — prompt variants on a subset of these.
LEAK_IDS = [
    "leak_0_adversarial",   # watercolor
    "leak_3_adversarial",   # abstract
    "leak_6_adversarial",
    "leak_7_adversarial",
    "leak_8_adversarial",
    "leak_9_adversarial",
]
PROMPT_LEAK_IDS = LEAK_IDS[:4]   # B is run on 4 of the 6 pairs

# ---------------------------------------------------------------- spectral operators (Exp C: source)
# Each maps a single source image -> a manipulated single image fed to Kontext.
SOURCE_OPS = [
    "raw",            # baseline (no spectral op) — matched-512 Kontext baseline
    "phase_only",     # structure/edges, flattened magnitude
    "amplitude_only", # texture statistics, randomized phase (structure destroyed)
    "low_band",       # low-frequency content (layout/palette)
    "high_band",      # high-frequency content (edges/texture)
]

# ---------------------------------------------------------------- spectral operators (Exp A: reference)
# Each maps (content, style) -> a single composite image fed to Kontext with a neutral instruction.
REF_OPS = [
    "content_raw",            # baseline: content image, "render in <style> style"
    "content_phase_style_amp",# content structure + style texture statistics  (low-leak hypothesis)
    "style_phase_content_amp",# style structure + content texture             (leak hypothesis)
    "style_high_on_content",  # content + high-frequency band of the style ref (texture-only graft)
]

# ---------------------------------------------------------------- prompt formulations (Exp B)
PROMPTS = {
    "neutral": "Restyle the image using the reference style.",
    "content_preserving": ("Preserve the content, identity, layout, and object shapes. Use only the "
                           "visual style, texture, colors, and rendering qualities of the reference."),
    "anti_leakage": ("Do not copy objects, scene layout, people, animals, or semantic content from the "
                     "reference. Use only style, texture, palette, and material qualities."),
}

# Method-card metadata (model / data / supervision / one-line insight) per experiment.
METHODS = {
    "spectral_source": dict(
        model="FLUX.1-Kontext-dev (12B in-context editor, 4-bit NF4)",
        data="PIE-Bench++ source image + edit instruction (6 task types)",
        supervision="none (pretrained Kontext; spectral op applied to the INPUT image only)",
        insight="Which frequency components of the source does Kontext need to keep identity and "
                "still follow the instruction?",
        site="model input image (pre-encode)",
    ),
    "spectral_reference": dict(
        model="FLUX.1-Kontext-dev (4-bit NF4)",
        data="adversarial content x style pairs (photo content x WikiArt style), neutral instruction",
        supervision="none (pretrained Kontext; FFT composite of content+style fed as the input image)",
        insight="Does content-phase + style-amplitude transfer texture while suppressing reference "
                "semantic leakage? Does style-phase cause object copying?",
        site="model input image (spectral content/style composite)",
    ),
    "prompt_variants": dict(
        model="FLUX.1-Kontext-dev (4-bit NF4)",
        data="adversarial content x style pairs; content image as input",
        supervision="none (pretrained Kontext; instruction wording varied)",
        insight="Is reference leakage a prompt problem or a model problem? Does anti-leakage wording "
                "help on its own?",
        site="text instruction",
    ),
}

CONTROLS = dict(
    kontext_e49="E49 FLUX.1-Kontext-dev baseline (native 1024px) — cross-phase reference",
    redux_e49="E49 FLUX Redux — high-style / high-leakage reference point",
    styleid_e49="E49 VGG-19 Gram (Gatys) control — low-leakage classical stylisation point",
)


def ensure_dirs():
    for d in [DATA, SUBSETS, MANIFESTS, OUT, METRICS, FIG, VIDEO, REPORTS, LOGS,
              FIG / "grids", FIG / "fourier", FIG / "representation_visuals",
              FIG / "best_cases", FIG / "worst_cases", FIG / "leakage_cases",
              OUT / "kontext_baseline_replay", OUT / "spectral_reference",
              OUT / "spectral_source", OUT / "prompt_variants", OUT / "optional_timestep"]:
        d.mkdir(parents=True, exist_ok=True)


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    return path


def rel(p):
    """repo-relative string path for manifests."""
    return str(Path(p).resolve().relative_to(REPO))


if __name__ == "__main__":
    ensure_dirs()
    write_json(ROOT / "configs" / "phase_config.json", dict(
        phase=PHASE_VERSION, seed=SEED, gen_size=GEN_SIZE, steps=STEPS, guidance=GUIDANCE,
        primary_model="black-forest-labs/FLUX.1-Kontext-dev (4-bit NF4)",
        edit_ids=EDIT_IDS, leak_ids=LEAK_IDS, prompt_leak_ids=PROMPT_LEAK_IDS,
        source_ops=SOURCE_OPS, ref_ops=REF_OPS, prompts=list(PROMPTS),
        controls=CONTROLS))
    print("E50 config written. edits:", EDIT_IDS, "leak:", LEAK_IDS)
