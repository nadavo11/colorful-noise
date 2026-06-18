"""Structure-preservation + editability metrics for the e41 RF-inversion comparison.

- structure_distance: DINO-ViT self-similarity distance (Tumanyan/Splice, the PIE-Bench
  "Structure Distance" headline metric). Lower = structure better preserved.
- clip_directional: CLIP directional similarity (editability). Higher = the edit moved the
  image in the same CLIP direction as src->edit text. This is the matched-editability axis.
- background_metrics: masked background PSNR / MSE / LPIPS using a PIE-Bench edit mask.
- prompt_distance: CLIP text cosine distance (warm-start heuristic for calibration).

LPIPS/SSIM reuse e35_vs_baseline (_img_metrics, _load_lpips, _load_ssim) so there is one
implementation. CLIP features reuse clip_sim.
"""
import numpy as np
import torch
from clip_sim import clip_image_features, clip_text_features, load_clip

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ---------------------------------------------------------------------------
# model loading
# ---------------------------------------------------------------------------
def load_dino(device="cuda"):
    """DINO ViT-S/8 (Caron et al.) for self-similarity structure distance."""
    m = torch.hub.load("facebookresearch/dino:main", "dino_vits8", verbose=False)
    return m.to(device).eval()


def load_metrics(device="cuda"):
    """Bundle every model needed for scoring: {dino, clip=(model,proc), lpips, ssim}."""
    from e35_vs_baseline import _load_lpips, _load_ssim
    return {"dino": load_dino(device), "clip": load_clip(), "lpips": _load_lpips(),
            "ssim": _load_ssim()}


# ---------------------------------------------------------------------------
# DINO structure distance
# ---------------------------------------------------------------------------
def _dino_prep(img, size, device):
    a = np.asarray(img.convert("RGB").resize((size, size)), dtype=np.float32) / 255.0
    x = torch.from_numpy(a).permute(2, 0, 1)[None]
    return ((x - _IMAGENET_MEAN) / _IMAGENET_STD).to(device)


@torch.no_grad()
def _dino_self_sim(dino, img, size=224):
    """(N, N) cosine self-similarity of the last-layer patch tokens."""
    x = _dino_prep(img, size, next(dino.parameters()).device)
    toks = dino.get_intermediate_layers(x, n=1)[0][:, 1:, :]   # drop CLS -> (1, N, C)
    toks = toks / (toks.norm(dim=-1, keepdim=True) + 1e-8)
    return (toks[0] @ toks[0].t())


@torch.no_grad()
def structure_distance(dino, img_a, img_b, size=224):
    """RMS difference between the two DINO self-similarity matrices (lower = closer)."""
    sa, sb = _dino_self_sim(dino, img_a, size), _dino_self_sim(dino, img_b, size)
    return float((sa - sb).pow(2).mean().sqrt())


# ---------------------------------------------------------------------------
# editability + prompt distance (CLIP)
# ---------------------------------------------------------------------------
@torch.no_grad()
def clip_directional(clip, src_img, edit_img, src_prompt, edit_prompt):
    """Cosine between the image edit direction and the text edit direction."""
    model, proc = clip
    fi = clip_image_features(model, proc, [src_img, edit_img])
    ft = clip_text_features(model, proc, [src_prompt, edit_prompt])
    di, dt = fi[1] - fi[0], ft[1] - ft[0]
    return float((di * dt).sum() / (di.norm() * dt.norm() + 1e-12))


@torch.no_grad()
def prompt_distance(clip, src_prompt, edit_prompt):
    """1 - cosine(src text, edit text). Small => stylistic edit (lock structure hard)."""
    model, proc = clip
    ft = clip_text_features(model, proc, [src_prompt, edit_prompt])
    return 1.0 - float((ft[0] * ft[1]).sum())


# ---------------------------------------------------------------------------
# structure-vs-source image metrics (LPIPS / SSIM + masked background)
# ---------------------------------------------------------------------------
def image_metrics(edit_img, src_img, lpips_net, ssim_fn):
    """LPIPS / DSSIM / spectral / color distance of the edit vs the SOURCE image."""
    from e35_vs_baseline import _img_metrics
    return _img_metrics(edit_img, src_img, lpips_net, ssim_fn)


def background_metrics(edit_img, src_img, mask, lpips_net=None):
    """Background fidelity using a PIE-Bench edit mask (white/255 = edited foreground).
    Returns bg_psnr, bg_mse (over background pixels) and, if lpips available, bg_lpips
    (foreground zeroed in both images)."""
    e = np.asarray(edit_img.convert("RGB"), dtype=np.float32)
    s = np.asarray(src_img.convert("RGB").resize(edit_img.size), dtype=np.float32)
    m = np.asarray(mask.convert("L").resize(edit_img.size), dtype=np.float32) > 127  # fg
    bg = ~m
    out = {}
    if bg.any():
        mse = float(((e - s) ** 2)[bg].mean())
        out["bg_mse"] = mse
        out["bg_psnr"] = float(10.0 * np.log10(255.0 ** 2 / (mse + 1e-8)))
    if lpips_net is not None:
        try:
            dev = next(lpips_net.parameters()).device
            keep = bg[..., None].astype(np.float32)
            def t(a):
                return (torch.from_numpy(a * keep).permute(2, 0, 1)[None] / 127.5 - 1.0).to(dev)
            with torch.no_grad():
                out["bg_lpips"] = float(lpips_net(t(e), t(s)).item())
        except Exception:
            out["bg_lpips"] = None
    return out
