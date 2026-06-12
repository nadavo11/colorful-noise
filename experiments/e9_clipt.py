"""E9 add-on: CLIP-T (text-image) prompt fidelity per condition.

E9's existing metrics measure the *look* band-norm produces (contrast, palette,
detail). None measure whether band-norm keeps the image ON-PROMPT. This adds the
standard CLIP text-image cosine: for every image, cosine(CLIP_img, CLIP_text(prompt)).

The test: does taming contrast/saturation cost prompt adherence? If band-norm's
paired Δ (band-norm − cfg=3.5) is ~0 or positive, it preserves fidelity while
calming the palette.

Reads the images already on disk (cfg1.0 / cfg3.5 / bandnorm / xfer per class),
writes results/e9/clip_t.json with per-condition aggregates, per-seed arrays
(indexed by seed so the page can show matched seeds), and the paired delta.

    python e9_clipt.py
"""
import argparse
import json
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS
from e9_bandnorm_classes import CLASSES

OUT = os.path.join(RESULTS, "e9")
CONDS = ["cfg1.0", "cfg3.5", "bandnorm", "uninorm", "xfer"]
MAX_SEED = 25  # cfg3.5/bandnorm: 0..24; uninorm: 0..4; xfer: 0..1; cfg1.0 filled to 0..24


def agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    m = sum(vals) / len(vals)
    sd = (sum((v - m) ** 2 for v in vals) / max(len(vals) - 1, 1)) ** 0.5
    return {"mean": m, "std": sd, "n": len(vals)}


def load_clip(model_id):
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(model_id).to("cuda").eval()
    proc = CLIPProcessor.from_pretrained(model_id)
    return model, proc


@torch.no_grad()
def clip_scores(model, proc, prompt, images):
    """Cosine(text, image) for a list of PIL images against one prompt."""
    tin = proc(text=[prompt], return_tensors="pt", padding=True,
               truncation=True).to("cuda")
    tfeat = model.get_text_features(**tin)
    tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)
    out = []
    B = 16
    for i in range(0, len(images), B):
        batch = images[i:i + B]
        iin = proc(images=batch, return_tensors="pt").to("cuda")
        ifeat = model.get_image_features(**iin)
        ifeat = ifeat / ifeat.norm(dim=-1, keepdim=True)
        sims = (ifeat @ tfeat.T).squeeze(-1)
        out.extend(sims.float().cpu().tolist())
    return out


def main(args):
    model, proc = load_clip(args.model)
    print(f"[clipt] model={args.model}", flush=True)

    report = {"model": args.model}
    for key, prompt in CLASSES:
        cdir = f"{OUT}/{key}"
        per_seed = {c: [None] * MAX_SEED for c in CONDS}
        means = {}
        for cond in CONDS:
            seeds, imgs = [], []
            for s in range(MAX_SEED):
                p = f"{cdir}/images/{cond}_s{s}.png"
                if os.path.exists(p):
                    seeds.append(s)
                    imgs.append(Image.open(p).convert("RGB"))
            if not imgs:
                continue
            scores = clip_scores(model, proc, prompt, imgs)
            for s, sc in zip(seeds, scores):
                per_seed[cond][s] = sc
            means[cond] = agg(scores)

        # paired delta band-norm - cfg3.5 at matched seeds
        bn, c35 = per_seed["bandnorm"], per_seed["cfg3.5"]
        pdiff = [bn[s] - c35[s] for s in range(MAX_SEED)
                 if bn[s] is not None and c35[s] is not None]

        entry = {c: means[c] for c in CONDS if c in means}
        entry["per_seed"] = per_seed
        entry["delta_bandnorm_cfg35"] = agg(pdiff)
        report[f"class/{key}"] = entry
        dm = (entry["delta_bandnorm_cfg35"] or {}).get("mean")
        print(f"[clipt] {key}: cfg1={_m(means,'cfg1.0')} cfg3.5={_m(means,'cfg3.5')} "
              f"bn={_m(means,'bandnorm')} Δ(bn-cfg3.5)="
              f"{'%.4f' % dm if dm is not None else 'NA'}", flush=True)

    path = f"{OUT}/clip_t.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[clipt] wrote {path}", flush=True)


def _m(means, k):
    return "%.3f" % means[k]["mean"] if k in means and means[k] else "NA"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/clip-vit-large-patch14")
    main(ap.parse_args())
