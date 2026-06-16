"""T2I-CompBench prompts + B-VQA (BLIP-VQA) attribute-binding score, to compare
SBN / CFG-Zero* / CFG++ "as done in cfg*" (CFG-Zero* evaluated on T2I-CompBench).

B-VQA (the headline T2I-CompBench metric for color/shape/texture) per the official
repo: spaCy extracts the prompt's noun phrases (e.g. "a green bench", "a blue
bowl"); BLIP-VQA is asked "{noun phrase}?" for each; the image score is the PRODUCT
of the "yes" probabilities (all attributes must bind). We reproduce that with the
maintained transformers BlipForQuestionAnswering (Salesforce/blip-vqa-capfilt-large)
and a yes/no-normalized P(yes), which matches their methodology without vendoring
the old BLIP code.

Prompt files live in compbench_data/<cat>_val.txt (300 each, downloaded from the
official repo). We sample a balanced subset across the attribute-binding categories.
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compbench_data")
BVQA_MODEL = "Salesforce/blip-vqa-capfilt-large"
EXCLUDE = {"top", "the side", "the left", "the right"}  # T2I-CompBench's exclusions


def load_compbench_prompts(categories=("color", "shape", "texture"), per_cat=64):
    """Balanced stride-sampled subset -> list of (pid, prompt, category)."""
    out = []
    for cat in categories:
        path = os.path.join(DATA, f"{cat}_val.txt")
        lines = [l.strip() for l in open(path) if l.strip()]
        stride = max(1, len(lines) // per_cat)
        picked = lines[::stride][:per_cat]
        for j, p in enumerate(picked):
            out.append((f"{cat}_{j:03d}", p, cat))
    return out


# ---------------------------------------------------------------------------
# B-VQA scorer
# ---------------------------------------------------------------------------

def load_bvqa(model_id=BVQA_MODEL, device="cuda"):
    """Return (model, processor, nlp) or None if deps/model unavailable."""
    try:
        import spacy
        from transformers import BlipForQuestionAnswering, BlipProcessor
    except Exception as e:
        print(f"[compbench] B-VQA deps missing ({e}); skipping", flush=True)
        return None
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        from spacy.cli import download as spacy_dl
        spacy_dl("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    proc = BlipProcessor.from_pretrained(model_id)
    model = BlipForQuestionAnswering.from_pretrained(
        model_id, torch_dtype=torch.float16).to(device).eval()
    return model, proc, nlp


def noun_phrases(nlp, prompt):
    nps = [c.text for c in nlp(prompt).noun_chunks if c.text not in EXCLUDE]
    return nps or [prompt]


@torch.no_grad()
def _p_yes(model, proc, image, question, device):
    """yes/no-normalized P(yes | image, question)."""
    enc = proc(images=image, text=question, return_tensors="pt").to(device)
    enc["pixel_values"] = enc["pixel_values"].to(model.dtype)
    logp = {}
    for ans in ("yes", "no"):
        labels = proc.tokenizer(ans, return_tensors="pt").input_ids.to(device)
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                    pixel_values=enc["pixel_values"], labels=labels)
        logp[ans] = -float(out.loss) * labels.shape[1]   # ~ sum log p(answer tokens)
    py, pn = math.exp(logp["yes"]), math.exp(logp["no"])
    return py / (py + pn + 1e-8)


@torch.no_grad()
def bvqa_scores(bvqa, prompt, images, device="cuda"):
    """T2I-CompBench B-VQA per image: product of P(yes) over the prompt's noun
    phrases. [float in 0..1, ...] or [None]*N if the scorer is unavailable."""
    if bvqa is None or not images:
        return [None] * len(images)
    model, proc, nlp = bvqa
    nps = noun_phrases(nlp, prompt)
    out = []
    for img in images:
        im = img.convert("RGB")
        p = 1.0
        for q in nps:
            p *= _p_yes(model, proc, im, f"{q}?", device)
        out.append(p)
    return out
