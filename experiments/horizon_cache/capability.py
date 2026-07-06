"""Graceful capability detection — run whatever path is available, document the rest.

The spec is explicit: do not block on missing models. We probe for SD3 / FLUX
generation harnesses, FlowEdit / FlowAlign editing harnesses, tabular tooling, and
perceptual-metric libs, and report a plain dict the runner keys off.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

HF_HUB = Path(os.path.expanduser("~/.cache/huggingface/hub"))


def _hf_cached(repo: str) -> bool:
    return (HF_HUB / ("models--" + repo.replace("/", "--"))).exists()


def _importable(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except Exception:
        return False


def detect() -> dict[str, Any]:
    import torch

    caps: dict[str, Any] = {}
    caps["cuda"] = bool(torch.cuda.is_available())
    caps["gpu"] = torch.cuda.get_device_name(0) if caps["cuda"] else None
    caps["diffusers"] = _importable("diffusers")

    # ---- generation backends ----
    caps["flux_generation"] = bool(caps["diffusers"] and _hf_cached("black-forest-labs/FLUX.1-dev"))
    caps["flux_model_id"] = "black-forest-labs/FLUX.1-dev" if caps["flux_generation"] else None
    sd3_ids = ["stabilityai/stable-diffusion-3-medium-diffusers",
               "stabilityai/stable-diffusion-3.5-medium",
               "stabilityai/stable-diffusion-3.5-large"]
    sd3_hit = next((r for r in sd3_ids if _hf_cached(r)), None)
    caps["sd3_generation"] = bool(caps["diffusers"] and sd3_hit)
    caps["sd3_model_id"] = sd3_hit

    # ---- editing backends (FlowEdit / FlowAlign over FLUX) ----
    repo = Path(__file__).resolve().parents[2]
    caps["flowedit_harness"] = (repo / "experiments" / "piebench.py").exists() and caps["flux_generation"]
    caps["flowalign_harness"] = (repo / "experiments" / "e43_flowalign.py").exists() and caps["flux_generation"]
    # editing needs a real source-image set; PIE-Bench is the project's editing fixture
    caps["piebench_data"] = (HF_HUB / "datasets--UB-CVML-Group--PIE_Bench_pp").exists()
    caps["flowedit"] = bool(caps["flowedit_harness"] and caps["piebench_data"])
    caps["flowalign"] = bool(caps["flowalign_harness"] and caps["piebench_data"])

    # ---- tabular tooling for HorizonCache-v1 ----
    if _importable("lightgbm"):
        caps["tabular"] = "lightgbm"
    elif _importable("xgboost"):
        caps["tabular"] = "xgboost"
    elif _importable("sklearn"):
        caps["tabular"] = "sklearn_histgb"
    else:
        caps["tabular"] = None

    # ---- perceptual metrics ----
    caps["lpips"] = _importable("lpips")
    caps["skimage"] = _importable("skimage")
    return caps


def summarize(caps: dict[str, Any]) -> str:
    def y(k):
        return "yes" if caps.get(k) else "no"
    return (
        f"cuda={y('cuda')}({caps.get('gpu')}) | flux_gen={y('flux_generation')} "
        f"sd3_gen={y('sd3_generation')}({caps.get('sd3_model_id')}) | "
        f"flowedit={y('flowedit')} flowalign={y('flowalign')} | "
        f"tabular={caps.get('tabular')} lpips={y('lpips')} skimage={y('skimage')}"
    )


if __name__ == "__main__":
    c = detect()
    print(summarize(c))
