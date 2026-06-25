"""Evaluation metrics for the baseline phase. All models are the already-cached ones.

Content preservation : CLIP-I, DINO, LPIPS, SigLIP (output vs content)
Edit correctness     : CLIP-T (output vs target prompt), CLIP-T gain vs source
Style transfer       : CLIP-I/DINO to style ref, color-hist distance, Fourier-amp distance
Reference leakage     : CLIP-I/DINO/SigLIP to style image (high == copies style semantics)

Everything lazily loaded once; metrics return plain floats. Higher CLIP/DINO/SigLIP sim and
lower LPIPS/color/Fourier distance = more similar.
"""
from __future__ import annotations
import numpy as np
import torch
from PIL import Image

_D = "cuda" if torch.cuda.is_available() else "cpu"
_M = {}


def _clip():
    if "clip" not in _M:
        from transformers import CLIPModel, CLIPProcessor
        m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(_D).eval()
        p = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _M["clip"] = (m, p)
    return _M["clip"]


def _dino():
    if "dino" not in _M:
        from transformers import AutoModel, AutoImageProcessor
        m = AutoModel.from_pretrained("facebook/dinov2-small").to(_D).eval()
        p = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        _M["dino"] = (m, p)
    return _M["dino"]


def _siglip():
    if "siglip" not in _M:
        from transformers import AutoModel, AutoProcessor
        m = AutoModel.from_pretrained("google/siglip-base-patch16-224").to(_D).eval()
        p = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
        _M["siglip"] = (m, p)
    return _M["siglip"]


def _lpips():
    if "lpips" not in _M:
        try:
            import lpips
            _M["lpips"] = lpips.LPIPS(net="alex").to(_D).eval()
        except Exception:
            _M["lpips"] = None
    return _M["lpips"]


def _load(x):
    return x if isinstance(x, Image.Image) else Image.open(x).convert("RGB")


def _tensor(e):
    """Coerce a (possibly transformers-5.x output) embedding to a 2D tensor."""
    if isinstance(e, torch.Tensor):
        return e
    for attr in ("image_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
        v = getattr(e, attr, None)
        if isinstance(v, torch.Tensor):
            return v[:, 0] if v.dim() == 3 else v
    raise TypeError(f"cannot coerce {type(e)} to tensor")


@torch.no_grad()
def clip_img_emb(img):
    m, p = _clip()
    i = p(images=_load(img), return_tensors="pt").to(_D)
    e = _tensor(m.get_image_features(**i))
    return torch.nn.functional.normalize(e, dim=-1)


@torch.no_grad()
def clip_txt_emb(text):
    m, p = _clip()
    i = p(text=[text], return_tensors="pt", padding=True, truncation=True).to(_D)
    e = _tensor(m.get_text_features(**i))
    return torch.nn.functional.normalize(e, dim=-1)


@torch.no_grad()
def dino_emb(img):
    m, p = _dino()
    i = p(images=_load(img), return_tensors="pt").to(_D)
    e = _tensor(m(**i))                          # CLS token via _tensor(last_hidden_state)
    return torch.nn.functional.normalize(e, dim=-1)


@torch.no_grad()
def siglip_emb(img):
    m, p = _siglip()
    i = p(images=_load(img), return_tensors="pt").to(_D)
    e = _tensor(m.get_image_features(**i))
    return torch.nn.functional.normalize(e, dim=-1)


def _cos(a, b):
    return float((a * b).sum().item())


@torch.no_grad()
def lpips_dist(a, b):
    net = _lpips()
    if net is None:
        return float("nan")
    def t(x):
        x = _load(x).resize((256, 256))
        arr = torch.from_numpy(np.asarray(x)).float().permute(2, 0, 1)[None] / 127.5 - 1
        return arr.to(_D)
    return float(net(t(a), t(b)).item())


def color_hist_dist(a, b, bins=32):
    """Chi-square distance between RGB histograms (palette similarity)."""
    def h(x):
        arr = np.asarray(_load(x).resize((256, 256)))
        hs = [np.histogram(arr[..., c], bins=bins, range=(0, 255), density=True)[0]
              for c in range(3)]
        return np.concatenate(hs)
    ha, hb = h(a), h(b)
    return float(0.5 * np.sum((ha - hb) ** 2 / (ha + hb + 1e-8)))


def fourier_amp_dist(a, b):
    """L1 distance of log radial amplitude spectra (texture/frequency signature)."""
    def radial(x):
        g = np.asarray(_load(x).convert("L").resize((256, 256)), dtype=np.float32)
        f = np.fft.fftshift(np.fft.fft2(g))
        amp = np.log1p(np.abs(f))
        cy, cx = 128, 128
        y, xx = np.indices(amp.shape)
        r = np.sqrt((y - cy) ** 2 + (xx - cx) ** 2).astype(int)
        tbin = np.bincount(r.ravel(), amp.ravel())
        nr = np.bincount(r.ravel())
        prof = tbin / np.maximum(nr, 1)
        return prof[:128] / (prof[:128].sum() + 1e-8)
    return float(np.abs(radial(a) - radial(b)).sum())


def full_metrics(output, content, style=None, target_prompt=None, source_prompt=None):
    """Compute the whole metric row for one generated image.
    output/content/style are paths or PIL; prompts are strings."""
    o_clip, c_clip = clip_img_emb(output), clip_img_emb(content)
    o_dino, c_dino = dino_emb(output), dino_emb(content)
    r = {}
    # content preservation
    r["clipI_content"] = _cos(o_clip, c_clip)
    r["dino_content"] = _cos(o_dino, c_dino)
    r["siglip_content"] = _cos(siglip_emb(output), siglip_emb(content))
    r["lpips_content"] = lpips_dist(output, content)
    r["colorhist_content"] = color_hist_dist(output, content)
    # edit correctness
    if target_prompt:
        t = clip_txt_emb(target_prompt)
        r["clipT_target"] = _cos(o_clip, t)
        if source_prompt:
            r["clipT_source"] = _cos(o_clip, clip_txt_emb(source_prompt))
            r["clipT_gain"] = r["clipT_target"] - r["clipT_source"]
    # style transfer + leakage (vs style ref)
    if style is not None:
        s_clip, s_dino = clip_img_emb(style), dino_emb(style)
        r["clipI_style"] = _cos(o_clip, s_clip)
        r["dino_style"] = _cos(o_dino, s_dino)
        r["siglip_style"] = _cos(siglip_emb(output), siglip_emb(style))
        r["colorhist_style"] = color_hist_dist(output, style)
        r["fourier_style"] = fourier_amp_dist(output, style)
        # leakage proxy: high DINO-to-style but the content should dominate; we report both.
        r["leak_dino_style"] = r["dino_style"]
        r["leak_clip_style"] = r["clipI_style"]
    return r


if __name__ == "__main__":
    import sys
    a, b = sys.argv[1], sys.argv[2]
    print(full_metrics(a, b, target_prompt="a photo"))
