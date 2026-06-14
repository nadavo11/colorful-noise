"""E16 fidelity metrics: LAION aesthetic, ImageReward, spectral-distance-to-real.

Fidelity is E16's primary axis (does the image look real/good), with CLIP-T /
VQAScore as adherence guardrails (e9_clipt.py / vqascore.py). The three scorers
here are the headline numbers:

  aesthetic  -- LAION improved-aesthetic-predictor: a small MLP on L2-normalized
                CLIP ViT-L/14 image embeddings (the SAME CLIP e9_clipt loads).
                This is the metric CFG-Zero* reports, so it is head-to-head
                comparable. Weights pulled once from the public repo into
                results/_models/.
  imagereward-- the `image-reward` pip package (learned human preference:
                fidelity + aesthetics + prompt coherence). Path/PIL based.
  spectral_dist_to_real -- distance of a latent's channel-mean radial PSD to the
                E10 real-photo PSD (results/e10/real_latents.pt). Native to the
                SBN thesis: CFG inflates spectral power above the real-image
                level (E10); this measures how close each method sits to real.

Every loader degrades gracefully: if a model/weight/package is missing it returns
None and the scorer yields None per image, so the E16 table still renders (the
missing column is just blank). Heavy models (aesthetic CLIP, ImageReward) are
meant to be loaded in E16's `--part score` phase, AFTER the diffusion models are
freed, to avoid VRAM contention on the 24GB A5000.
"""
import math
import os
import sys
import urllib.request

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS
from spectral_ops import radial_psd

MODELS_DIR = os.path.join(RESULTS, "_models")
# LAION improved-aesthetic-predictor (CLIP ViT-L/14, 768-d) linear-MSE head.
AESTHETIC_URL = ("https://github.com/christophschuhmann/improved-aesthetic-predictor"
                 "/raw/main/sac+logos+ava1-l14-linearMSE.pth")
AESTHETIC_PATH = os.path.join(MODELS_DIR, "sac+logos+ava1-l14-linearMSE.pth")


# ---------------------------------------------------------------------------
# LAION aesthetic predictor
# ---------------------------------------------------------------------------

class _AestheticMLP(nn.Module):
    """The improved-aesthetic-predictor head (matches the published state_dict)."""

    def __init__(self, input_size=768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024), nn.Dropout(0.2),
            nn.Linear(1024, 128), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


def _download(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        return True
    try:
        print(f"[fidelity] downloading {url}", flush=True)
        req = urllib.request.Request(url, headers={"User-Agent": "e16/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        with open(path, "wb") as f:
            f.write(data)
        return True
    except Exception as e:
        print(f"[fidelity] aesthetic weight download FAILED: {e}", flush=True)
        return False


def load_aesthetic(device="cuda"):
    """Return the aesthetic MLP (CLIP features are supplied by the caller via
    aesthetic_scores), or None if weights can't be fetched."""
    if not _download(AESTHETIC_URL, AESTHETIC_PATH):
        return None
    mlp = _AestheticMLP(768)
    sd = torch.load(AESTHETIC_PATH, weights_only=True, map_location="cpu")
    mlp.load_state_dict(sd)
    return mlp.to(device).eval()


@torch.no_grad()
def aesthetic_scores(mlp, clip_model, clip_proc, images, batch=16, device="cuda"):
    """LAION aesthetic score (~1..10) per PIL image, using the shared CLIP
    ViT-L/14 (same model e9_clipt loads). Returns [float, ...] or [None]*N."""
    if mlp is None or not images:
        return [None] * len(images)
    out = []
    for i in range(0, len(images), batch):
        chunk = [im.convert("RGB") for im in images[i:i + batch]]
        iin = clip_proc(images=chunk, return_tensors="pt").to(device)
        feat = clip_model.get_image_features(**iin).float()
        feat = feat / feat.norm(dim=-1, keepdim=True)  # L2-normalize (predictor convention)
        pred = mlp(feat).squeeze(-1)
        out.extend(pred.cpu().tolist())
    return out


# ---------------------------------------------------------------------------
# ImageReward
# ---------------------------------------------------------------------------

def load_imagereward(name="ImageReward-v1.0"):
    """Return an ImageReward model, or None if the package/weights are absent."""
    # compat shim: ImageReward's bundled BLIP imports apply_chunking_to_forward from
    # transformers.modeling_utils, which moved to transformers.pytorch_utils in newer
    # transformers (>=4.x). Re-expose it before importing ImageReward.
    try:
        import transformers.modeling_utils as _mu
        import transformers.pytorch_utils as _pu
        for _n in ("apply_chunking_to_forward", "find_pruneable_heads_and_indices",
                   "prune_linear_layer"):
            if not hasattr(_mu, _n) and hasattr(_pu, _n):
                setattr(_mu, _n, getattr(_pu, _n))
    except Exception:
        pass
    try:
        import ImageReward as RM
    except Exception as e:
        print(f"[fidelity] image-reward not installed ({e}); skipping", flush=True)
        return None
    try:
        return RM.load(name)
    except Exception as e:
        print(f"[fidelity] ImageReward load FAILED: {e}", flush=True)
        return None


@torch.no_grad()
def imagereward_scores(model, prompt, paths):
    """ImageReward score per image path against one prompt. [float,...]/[None]*N.
    Path-based (ImageReward's score() takes file paths or PIL)."""
    if model is None or not paths:
        return [None] * len(paths)
    out = []
    for p in paths:
        try:
            out.append(float(model.score(prompt, p)))
        except Exception as e:
            print(f"[fidelity] ImageReward score FAILED on {p}: {e}", flush=True)
            out.append(None)
    return out


# ---------------------------------------------------------------------------
# Spectral distance to real (E10)
# ---------------------------------------------------------------------------

REAL_LATENTS = os.path.join(RESULTS, "e10", "real_latents.pt")


def load_real_psd(n_bins=24, drop_dc=True):
    """Channel-mean radial PSD of the E10 real-photo latents (log-space ref),
    or None if results/e10/real_latents.pt is missing. Returns (centers, log_psd)."""
    if not os.path.exists(REAL_LATENTS):
        print(f"[fidelity] no {REAL_LATENTS}; spectral-dist disabled "
              "(run e10 --part download,real)", flush=True)
        return None
    real = torch.load(REAL_LATENTS, weights_only=True)  # (N,16,128,128)
    centers, psd = radial_psd(real.cuda(), n_bins)       # psd: (C, n_bins)
    cmean = psd.mean(0).clamp(min=1e-12)
    if drop_dc:
        centers, cmean = centers[1:], cmean[1:]
    return centers.cpu(), cmean.log().cpu()


@torch.no_grad()
def spectral_dist_to_real(lat, real_ref, n_bins=24, drop_dc=True):
    """RMS distance in log-PSD space between a latent's channel-mean radial PSD
    and the real-image reference. Lower = closer to real images. None if no ref."""
    if real_ref is None or lat is None:
        return None
    _, log_real = real_ref
    _, psd = radial_psd(lat.cuda(), n_bins)
    cmean = psd.mean(0).clamp(min=1e-12)
    if drop_dc:
        cmean = cmean[1:]
    log_gen = cmean.log().cpu()
    return float(((log_gen - log_real) ** 2).mean().sqrt())
