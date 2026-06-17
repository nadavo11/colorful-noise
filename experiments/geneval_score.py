"""GenEval scorer (E37), faithful to djghosh13/geneval's evaluate_images.py — but with a
pip-clean detector so it runs anywhere torch+torchvision+transformers exist (the official
scorer needs mmdet/mmcv + open_clip + clip_benchmark, which are brittle on modern torch).

What is IDENTICAL to the official scorer:
  - the per-image `evaluate()` decision logic (include AND / exclude OR; count, color and
    relative-position checks; color/position only on the most-confident objects),
    `relative_position()` and `compute_iou()` are copied verbatim;
  - the constants: detect threshold 0.3 (0.9 for `counting`), MAX_OBJECTS 16,
    POSITION_THRESHOLD 0.1, the 10-colour palette, and the 3 colour-prompt templates;
  - the object crop fed to the colour classifier: the instance MASK composited onto a
    #999 background, then cropped to the bbox.

What DIFFERS (so numbers rank conditions faithfully but are NOT bit-identical to the
Mask2Former leaderboard):
  - detector = torchvision `maskrcnn_resnet50_fpn_v2` (COCO-80, gives boxes+masks) instead
    of mmdet Mask2Former;
  - colour zero-shot uses transformers CLIP ViT-L/14 (manual template-averaged classifier)
    instead of open_clip + clip_benchmark.

Three GenEval class names differ from torchvision's COCO names (aliased below).
"""
import json
import os
import re

import numpy as np
import torch
from PIL import Image, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- official constants (evaluate_images.py main()) -----------------------
COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white"]
COLOR_TEMPLATES = ["a photo of a {c} {n}", "a photo of a {c}-colored {n}", "a photo of a {c} object"]
THRESHOLD = 0.3
COUNTING_THRESHOLD = 0.9
MAX_OBJECTS = 16
NMS_THRESHOLD = 1.0          # 1.0 == keep all above threshold (no extra NMS)
POSITION_THRESHOLD = 0.1

# GenEval object_names that aren't torchvision COCO category strings
ALIAS = {"computer mouse": "mouse", "tv remote": "remote", "computer keyboard": "keyboard"}


def geneval_classes():
    p = os.path.join(HERE, "geneval_data", "object_names.txt")
    return [l.strip() for l in open(p) if l.strip()]


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

def load_detector(device="cuda"):
    """torchvision Mask R-CNN v2 (COCO-80). box_score_thresh=0 so WE threshold per the
    official rule (0.3, or 0.9 for counting). Returns (model, name->cat_index, classes)."""
    from torchvision.models.detection import (maskrcnn_resnet50_fpn_v2,
                                              MaskRCNN_ResNet50_FPN_V2_Weights)
    w = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn_v2(weights=w, box_score_thresh=0.0).eval().to(device)
    cats = w.meta["categories"]                       # idx -> name (91, incl __background__/N/A)
    name2idx = {n: i for i, n in enumerate(cats)}
    return model, name2idx, geneval_classes()


def load_clip(device="cuda", model_id="openai/clip-vit-large-patch14"):
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(model_id).eval().to(device)
    proc = CLIPProcessor.from_pretrained(model_id)
    return model, proc


# ---------------------------------------------------------------------------
# detection -> the official {classname: [(box5, mask), ...]} structure
# ---------------------------------------------------------------------------

@torch.no_grad()
def detect(detector, name2idx, classes, image, device, tag):
    """Run the detector once; return {classname: [(box[x1,y1,x2,y2,score], mask_bool_HxW)]}
    sorted by confidence, thresholded, capped at MAX_OBJECTS — mirroring evaluate_image()."""
    conf = COUNTING_THRESHOLD if tag == "counting" else THRESHOLD
    x = torch.from_numpy(np.asarray(image.convert("RGB"))).permute(2, 0, 1).float().div(255).to(device)
    out = detector([x])[0]
    boxes = out["boxes"].cpu().numpy()
    labels = out["labels"].cpu().numpy()
    scores = out["scores"].cpu().numpy()
    masks = out["masks"].cpu().numpy()[:, 0] > 0.5            # (N,H,W) bool
    detected = {}
    for cn in classes:
        ci = name2idx[ALIAS.get(cn, cn)]
        sel = np.where((labels == ci) & (scores > conf))[0]
        sel = sel[np.argsort(scores[sel])[::-1]][:MAX_OBJECTS]
        if len(sel) == 0:
            continue
        detected[cn] = [(np.array([*boxes[j], scores[j]]), masks[j]) for j in sel]
    return detected


# ---------------------------------------------------------------------------
# colour classifier (transformers CLIP; faithful crop + templates)
# ---------------------------------------------------------------------------

_COLOR_TEXT = {}   # classname -> (n_colors, dim) normalised, template-averaged


def _color_text_feats(clip, classname, device):
    if classname in _COLOR_TEXT:
        return _COLOR_TEXT[classname]
    model, proc = clip
    feats = []
    for c in COLORS:
        prompts = [t.format(c=c, n=classname) for t in COLOR_TEMPLATES]
        tin = proc(text=prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            f = model.get_text_features(**tin)
        f = f / f.norm(dim=-1, keepdim=True)
        m = f.mean(0)
        feats.append(m / m.norm())
    out = torch.stack(feats)                                  # (n_colors, dim)
    _COLOR_TEXT[classname] = out
    return out


def _crop(image, box, mask, bg="#999"):
    """Composite the instance mask onto a #999 background, then crop to bbox (official)."""
    blank = Image.new("RGB", image.size, color=bg)
    comp = Image.composite(image, blank, Image.fromarray(mask)) if mask is not None else image
    return comp.crop([int(v) for v in box[:4]])


@torch.no_grad()
def make_color_fn(clip, device):
    model, proc = clip

    def color_classification(image, objects, classname):
        crops = [_crop(image.convert("RGB"), box, mask) for box, mask in objects]
        if not crops:
            return []
        pin = proc(images=crops, return_tensors="pt").to(device)
        img_f = model.get_image_features(**pin)
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)
        txt_f = _color_text_feats(clip, classname, device)
        logits = img_f @ txt_f.T                              # (n_crops, n_colors)
        return [COLORS[i] for i in logits.argmax(1).tolist()]

    return color_classification


# ---------------------------------------------------------------------------
# decision logic — VERBATIM from official evaluate_images.py
# ---------------------------------------------------------------------------

def compute_iou(box_a, box_b):
    area_fn = lambda box: max(box[2] - box[0] + 1, 0) * max(box[3] - box[1] + 1, 0)
    i_area = area_fn([max(box_a[0], box_b[0]), max(box_a[1], box_b[1]),
                      min(box_a[2], box_b[2]), min(box_a[3], box_b[3])])
    u_area = area_fn(box_a) + area_fn(box_b) - i_area
    return i_area / u_area if u_area else 0


def relative_position(obj_a, obj_b):
    """Position of A relative to B, factoring in object dimensions (official)."""
    boxes = np.array([obj_a[0], obj_b[0]])[:, :4].reshape(2, 2, 2)
    center_a, center_b = boxes.mean(axis=-2)
    dim_a, dim_b = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
    offset = center_a - center_b
    revised_offset = np.maximum(np.abs(offset) - POSITION_THRESHOLD * (dim_a + dim_b), 0) * np.sign(offset)
    if np.all(np.abs(revised_offset) < 1e-3):
        return set()
    dx, dy = revised_offset / np.linalg.norm(offset)
    relations = set()
    if dx < -0.5: relations.add("left of")
    if dx > 0.5: relations.add("right of")
    if dy < -0.5: relations.add("above")
    if dy > 0.5: relations.add("below")
    return relations


def evaluate(image, objects, metadata, color_fn):
    """Official decision: include clauses AND'd, exclude clauses OR'd; colour/position only
    on the most-confident objects (objects arrive in sorted order)."""
    correct = True
    reason = []
    matched_groups = []
    for req in metadata.get("include", []):
        classname = req["class"]
        matched = True
        found_objects = objects.get(classname, [])[:req["count"]]
        if len(found_objects) < req["count"]:
            correct = matched = False
            reason.append(f"expected {classname}>={req['count']}, found {len(found_objects)}")
        else:
            if "color" in req:
                colors = color_fn(image, found_objects, classname)
                if colors.count(req["color"]) < req["count"]:
                    correct = matched = False
                    reason.append(f"expected {req['color']} {classname}>={req['count']}, "
                                  f"found {colors.count(req['color'])} {req['color']}")
            if "position" in req and matched:
                expected_rel, target_group = req["position"]
                if matched_groups[target_group] is None:
                    correct = matched = False
                    reason.append(f"no target for {classname} to be {expected_rel}")
                else:
                    for obj in found_objects:
                        for target_obj in matched_groups[target_group]:
                            true_rels = relative_position(obj, target_obj)
                            if expected_rel not in true_rels:
                                correct = matched = False
                                reason.append(f"expected {classname} {expected_rel} target, "
                                              f"found {' and '.join(true_rels)} target")
                                break
                        if not matched:
                            break
        matched_groups.append(found_objects if matched else None)
    for req in metadata.get("exclude", []):
        classname = req["class"]
        if len(objects.get(classname, [])) >= req["count"]:
            correct = False
            reason.append(f"expected {classname}<{req['count']}, found {len(objects[classname])}")
    return correct, "\n".join(reason)


# ---------------------------------------------------------------------------
# one image
# ---------------------------------------------------------------------------

def evaluate_image(detector, name2idx, classes, color_fn, filepath, metadata, device="cuda"):
    image = ImageOps.exif_transpose(Image.open(filepath)).convert("RGB")
    detected = detect(detector, name2idx, classes, image, device, metadata["tag"])
    is_correct, reason = evaluate(image, detected, metadata, color_fn)
    return {"filename": filepath, "tag": metadata["tag"], "prompt": metadata["prompt"],
            "correct": bool(is_correct), "reason": reason}
