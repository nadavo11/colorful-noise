"""E16 adherence guardrail: VQAScore (Lin et al., ECCV 2024).

CLIP-T (e9_clipt) is the weak bag-of-words score that motivated E16: it is known
to miss attribute binding / compositional structure, exactly the failures detailed
prompts stress. VQAScore asks a VQA model "Does this image show <prompt>?" and
returns P(yes); it correlates far better with humans on compositional prompts.

In E16 it is a *guardrail*, not the contest: we show SBN(+postprocess) does not
drop adherence relative to cfg=3.5 while it wins on fidelity (fidelity_metrics.py).

Path-based: t2v-metrics scores by file path, and E16 already writes every image to
results/e16/<id>/images/<cond>_s<seed>.png, so we pass those paths directly (no
temp files). Loaded only in E16's `--part score` phase, after diffusion models are
freed. Degrades gracefully: if t2v-metrics is missing or the model can't load,
load_vqascore returns None and scoring yields None per image (blank column).

The default model (clip-flant5-xxl, ~11GB) is heavy; pass model="clip-flant5-xl"
for a lighter VRAM footprint.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULT_MODEL = "clip-flant5-xxl"


def load_vqascore(model=DEFAULT_MODEL, device="cuda"):
    """Return a t2v_metrics.VQAScore scorer, or None if unavailable."""
    # compat shim: t2v-metrics pins transformers==4.49 but we run 4.57 (diffusers
    # 0.38 needs it). The incompat surfaces as symbols that moved from
    # transformers.modeling_utils to transformers.pytorch_utils -- re-expose them.
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
        import t2v_metrics
    except Exception as e:
        print(f"[vqascore] t2v-metrics not installed ({e}); skipping", flush=True)
        return None
    try:
        return t2v_metrics.VQAScore(model=model, device=device)
    except Exception as e:
        print(f"[vqascore] load FAILED ({model}): {e}", flush=True)
        return None


@torch.no_grad()
def vqa_scores_paths(scorer, prompt, paths, batch=8):
    """P(image entails prompt) in [0,1] per image path. [float,...]/[None]*N."""
    if scorer is None or not paths:
        return [None] * len(paths)
    out = []
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        try:
            s = scorer(images=list(chunk), texts=[prompt])  # -> (len(chunk), 1)
            out.extend(s.squeeze(-1).float().cpu().tolist())
        except Exception as e:
            print(f"[vqascore] score FAILED: {e}", flush=True)
            out.extend([None] * len(chunk))
    return out
