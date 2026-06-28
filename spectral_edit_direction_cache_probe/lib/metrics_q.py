"""E51 quality metrics (uv env, transformers-based). Reuses the E49/E50 metric loaders.

Quality is scored against the full-compute reference output (the gold standard a cache must
match), plus edit alignment and a structure/background-preservation proxy.
"""
from __future__ import annotations
import sys
import numpy as np
from PIL import Image

import config as C
sys.path.insert(0, str(C.BASE_LIB))
import metrics as BM   # noqa: E402


def _psnr(a, b):
    a = np.asarray(a.convert("RGB").resize((C.SIZE, C.SIZE)), np.float64)
    b = np.asarray(b.convert("RGB").resize((C.SIZE, C.SIZE)), np.float64)
    mse = np.mean((a - b) ** 2)
    return float(99.0 if mse < 1e-9 else 10 * np.log10(255.0 ** 2 / mse))


def quality(out_img, ref_img, src_img, src_prompt, tgt_prompt):
    o = BM.clip_img_emb(out_img); r = BM.clip_img_emb(ref_img); s = BM.clip_img_emb(src_img)
    od = BM.dino_emb(out_img); rd = BM.dino_emb(ref_img); sd = BM.dino_emb(src_img)
    tt = BM.clip_txt_emb(tgt_prompt); stx = BM.clip_txt_emb(src_prompt)
    di = (o - s); di = di / (di.norm() + 1e-8)
    dt = (tt - stx); dt = dt / (dt.norm() + 1e-8)
    return dict(
        # fidelity to the full-compute reference (higher dino/clip, lower lpips, higher psnr = better)
        dino_to_ref=BM._cos(od, rd),
        clipI_to_ref=BM._cos(o, r),
        lpips_to_ref=BM.lpips_dist(out_img, ref_img),
        psnr_to_ref=_psnr(out_img, ref_img),
        # edit correctness
        clipT_target=BM._cos(o, tt),
        clipT_gain=BM._cos(o, tt) - BM._cos(o, stx),
        clip_dir=float((di * dt).sum().item()),
        # structure / background preservation vs the source image
        dino_to_src=BM._cos(od, sd),
    )
