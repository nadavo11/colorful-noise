"""Shared CLIP helpers: image-image and image-text cosine (E13 / E15).

`e9_clipt.py` already does text-image CLIP-T for prompt fidelity; this module
adds reusable image-feature extraction + cosine so E13 can ask "does identity
follow the phase donor or the magnitude donor" and E15 can embed a battery of
manipulated outputs for clustering. ViT-L/14 is cached locally."""
import torch

MODEL_ID = "openai/clip-vit-large-patch14"


def load_clip(model_id=MODEL_ID):
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(model_id).to("cuda").eval()
    proc = CLIPProcessor.from_pretrained(model_id)
    return model, proc


@torch.no_grad()
def clip_image_features(model, proc, images, batch=16):
    """(N, D) L2-normalized CLIP image embeddings for a list of PIL images."""
    feats = []
    for i in range(0, len(images), batch):
        iin = proc(images=images[i:i + batch], return_tensors="pt").to("cuda")
        f = model.get_image_features(**iin)
        feats.append((f / f.norm(dim=-1, keepdim=True)).float().cpu())
    return torch.cat(feats)


@torch.no_grad()
def clip_text_features(model, proc, prompts):
    """(N, D) L2-normalized CLIP text embeddings."""
    tin = proc(text=list(prompts), return_tensors="pt", padding=True,
               truncation=True).to("cuda")
    f = model.get_text_features(**tin)
    return (f / f.norm(dim=-1, keepdim=True)).float().cpu()


def cosine(a, b):
    """Cosine similarity of two 1-D tensors (already-normalized or not)."""
    a, b = a.flatten().float(), b.flatten().float()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))
