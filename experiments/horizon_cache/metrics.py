"""Quality metrics vs a full-trajectory reference (PSNR / SSIM / LPIPS) + latent L2."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


def _to_np(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float64) / 255.0


def psnr(ref: Image.Image, test: Image.Image) -> float:
    a, b = _to_np(ref), _to_np(test)
    if a.shape != b.shape:
        test = test.resize(ref.size)
        b = _to_np(test)
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10(1.0 / mse))


def ssim(ref: Image.Image, test: Image.Image) -> float | None:
    try:
        from skimage.metrics import structural_similarity as sk_ssim
    except Exception:
        return None
    a, b = _to_np(ref), _to_np(test)
    if a.shape != b.shape:
        b = _to_np(test.resize(ref.size))
    return float(sk_ssim(a, b, channel_axis=2, data_range=1.0))


@lru_cache(maxsize=1)
def _lpips_model():
    import lpips
    m = lpips.LPIPS(net="alex", verbose=False)
    if torch.cuda.is_available():
        m = m.cuda()
    return m


def lpips_dist(ref: Image.Image, test: Image.Image) -> float | None:
    try:
        m = _lpips_model()
    except Exception:
        return None
    def prep(img):
        arr = np.asarray(img.convert("RGB").resize(ref.size), dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) * 2 - 1
        return t.cuda() if torch.cuda.is_available() else t
    with torch.no_grad():
        return float(m(prep(ref), prep(test)).item())


def latent_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float(); b = b.float()
    return float(((a - b) ** 2).mean().sqrt().item())


def image_metrics(ref: Image.Image, test: Image.Image) -> dict[str, Any]:
    out = {"psnr": psnr(ref, test)}
    s = ssim(ref, test)
    if s is not None:
        out["ssim"] = s
    lp = lpips_dist(ref, test)
    if lp is not None:
        out["lpips"] = lp
    return out
