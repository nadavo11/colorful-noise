"""Global config for the baseline-establishment phase.

Single source of truth for paths, seeds, image size, and the model/benchmark registry.
Everything reproducible: fixed seeds, fixed inference settings, recorded in manifests.
"""
from __future__ import annotations
import os, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]            # baseline_establishment/
REPO = ROOT.parent                                    # colorful-noise/

# ---- reproducibility ----
PHASE_VERSION = "baseline-establishment-v1"
SEEDS = [0, 1, 2]            # generation seeds; pilot uses SEEDS[0] mostly, grids show seed 0
GEN_SIZE = 512              # generation resolution (keeps quantized FLUX fast on A5000)
INFER_STEPS = 28           # FLUX/Kontext default-ish step count
GUIDANCE = 2.5             # FLUX guidance (Kontext recommends ~2.5-3.5)

# ---- paths ----
DATA = ROOT / "data"
BENCH = DATA / "benchmark_subsets"
LEAK = DATA / "custom_leakage_set"
STYLE = DATA / "style_refs"
OUT = ROOT / "outputs"
METRICS = ROOT / "metrics"
FIG = ROOT / "figures"
VIDEO = ROOT / "videos"
REPORTS = ROOT / "reports"
LOGS = ROOT / "logs"

# ---- model registry: model / data / supervision / one-line insight ----
# supervision is always "none (no training this phase)"; the field records what each
# method was *originally* trained on, for the method cards.
MODELS = {
    "flux_img2img": dict(
        name="FLUX img2img",
        family="FLUX (weak sanity)",
        model="black-forest-labs/FLUX.1-dev (12B rectified-flow, 4-bit NF4)",
        data="content image + target caption (no instruction grounding)",
        supervision="none (pretrained FLUX.1-dev; ran as noising→denoising img2img)",
        insight="Renoises the whole image to a caption — edits leak globally, identity drifts.",
    ),
    "flux_redux": dict(
        name="FLUX Redux",
        family="FLUX (weak sanity)",
        model="FLUX.1-Redux-dev SigLIP prior + FLUX.1-dev base (4-bit)",
        data="reference image only (SigLIP-encoded), optional text",
        supervision="none (pretrained Redux adapter)",
        insight="Variation/remix of a reference — strong style cue but copies reference semantics.",
    ),
    "flux_ipadapter": dict(
        name="FLUX IP-Adapter (style)",
        family="FLUX style-transfer",
        model="XLabs-AI/flux-ip-adapter on FLUX.1-dev (4-bit)",
        data="content prompt + style reference image",
        supervision="none (pretrained IP-Adapter; InstantStyle-on-FLUX analog)",
        insight="Decouples content prompt from a style image — closest no-train InstantStyle analog.",
    ),
    "flux_kontext": dict(
        name="FLUX.1 Kontext [dev]",
        family="FLUX competent editor",
        model="black-forest-labs/FLUX.1-Kontext-dev (12B in-context editor, 4-bit)",
        data="source image + edit instruction (in-context)",
        supervision="none (pretrained Kontext; the serious FLUX-family editing baseline)",
        insight="Native instruction editor — keeps untouched regions, applies the asked edit.",
    ),
    "styleid": dict(
        name="VGG-19 Gram (Gatys) classical control",
        family="Classical style transfer",
        model="VGG-19 Gram-matrix optimization (Gatys et al. 2016, training-free)",
        data="content image + style image",
        supervision="none (frozen ImageNet VGG-19; per-image pixel optimization — a diagnostic "
                    "control, NOT the main no-training baseline)",
        insight="Pure texture/color transfer with hard content anchor — low leakage, no semantics. "
                "NB: this is the classical Gatys Gram control, NOT the StyleID attention-injection "
                "method (Chung et al. 2024); the registry key 'styleid' is legacy.",
    ),
    "qwen_image_edit": dict(
        name="Qwen-Image-Edit",
        family="Strong external editor",
        model="Qwen/Qwen-Image-Edit (20B) — NOT RUN (hardware-infeasible)",
        data="source image + instruction",
        supervision="none (would be pretrained)",
        insight="20B model exceeds 25GB VRAM / disk budget this phase; substituted by Kontext.",
    ),
}

# benchmark task-type taxonomy used everywhere (PIE-Bench config -> task type)
PIE_CONFIGS = {
    "1_change_object_80": "object_replace",
    "2_add_object_80": "object_add",
    "3_delete_object_80": "object_remove",
    "4_change_attribute_content_40": "attribute",
    "6_change_attribute_color_40": "color",
    "7_change_attribute_material_40": "material",
    "8_change_background_80": "global",
    "9_change_style_80": "style",
}
PIE_SPLIT = "V1"

TASK_TYPES = ["color", "material", "object_replace", "object_add",
              "object_remove", "attribute", "global", "style"]


def ensure_dirs():
    for d in [DATA, BENCH, LEAK, STYLE, OUT, METRICS, FIG, VIDEO, REPORTS, LOGS,
              FIG / "grids", FIG / "best_cases", FIG / "worst_cases",
              FIG / "leakage_cases", FIG / "representation_visuals",
              REPORTS / "html", REPORTS / "assets"]:
        d.mkdir(parents=True, exist_ok=True)


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    return path


if __name__ == "__main__":
    ensure_dirs()
    write_json(ROOT / "configs" / "phase_config.json", dict(
        phase=PHASE_VERSION, seeds=SEEDS, gen_size=GEN_SIZE,
        infer_steps=INFER_STEPS, guidance=GUIDANCE,
        models={k: v["model"] for k, v in MODELS.items()},
        pie_configs=PIE_CONFIGS))
    print("config written; models:", list(MODELS))
