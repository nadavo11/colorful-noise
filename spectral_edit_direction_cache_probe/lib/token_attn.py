"""E52 — Text-Token Modulation Autopsy: FLUX attention instrumentation (uv env).

WHERE TEXT ENTERS FLUX.  FLUX is an MMDiT, not a classic U-Net with cross-attention. Text
reaches the image two ways:
  1. the **T5 token sequence** `encoder_hidden_states` (512 tokens × 4096) is *concatenated*
     with the image tokens and attends JOINTLY inside every transformer block;
  2. a **pooled CLIP-text vector** `pooled_projections` is folded into the timestep embedding
     and drives AdaLayerNorm modulation — a single global knob, not per-token.
Token-level control therefore lives in the joint-attention columns of the T5 tokens. In both
the 19 double-stream (`FluxTransformerBlock`) and 38 single-stream (`FluxSingleTransformerBlock`)
blocks the text tokens occupy the FIRST `txt_len` positions of the joint key/value sequence
(diffusers 0.38 `FluxAttnProcessor`: `cat([encoder, image], dim=1)`). So "image→text attention"
= the image-query rows attending to those leading text-key columns.

This module installs a recording/intervening processor on a chosen set of DOUBLE-stream blocks
(the only ones that receive `encoder_hidden_states`, so `txt_len` is unambiguous inside the
processor). It (a) records per-text-token attention statistics + spatial maps without changing
the forward, and (b) optionally applies one of four internal interventions:
  - embed_scale        : scale e_i (done in the DRIVER, on encoder_hidden_states; not here)
  - attn_logit_bias    : logits[:, :, i] += beta_i           (pre-softmax)
  - attn_prob_reweight : A[:, :, i] *= gamma_i, renormalize  (post-softmax)
  - value_scale        : V_i *= gamma_i
"""
from __future__ import annotations
import difflib
import math
import numpy as np
import torch


def _apply_rotary_emb(*a, **k):
    """Lazy import so the pure-python helpers (tokenize/align/roles) load without diffusers."""
    try:                                          # diffusers 0.38 layout
        from diffusers.models.embeddings import apply_rotary_emb
    except Exception:                             # pragma: no cover - fallback
        from diffusers.models.transformers.transformer_flux import apply_rotary_emb
    return apply_rotary_emb(*a, **k)


# ---------------------------------------------------------------------------
# tokenization + source/target alignment + role assignment
# ---------------------------------------------------------------------------
_STYLE_WORDS = {"style", "painting", "sketch", "drawing", "anime", "cartoon", "watercolor",
                "oil", "photo", "photorealistic", "realistic", "3d", "render", "pixar",
                "cyberpunk", "vintage", "retro", "noir", "impressionist", "minimalist"}
_ATTR_WORDS = {"red", "blue", "green", "yellow", "orange", "purple", "pink", "black", "white",
               "gray", "grey", "golden", "gold", "silver", "bronze", "metal", "metallic",
               "wooden", "wood", "glass", "plastic", "stone", "marble", "rusty", "shiny",
               "big", "small", "large", "tiny", "old", "young", "new", "bright", "dark",
               "happy", "sad", "open", "closed", "wet", "dry"}
_BG_WORDS = {"background", "sky", "wall", "floor", "ground", "field", "forest", "ocean", "sea",
             "beach", "mountain", "street", "room", "grass", "snow", "water", "sunset"}
_CONTROL_WORDS = {"a", "an", "the", "of", "in", "on", "at", "and", "with", "is", "to", "."}


def tokenize(pipe, prompt, max_len=512):
    """T5 tokenization matching the generation path. Returns the real (non-pad) token strings
    and their ids, plus the attention length L."""
    tok = pipe.tokenizer_2
    enc = tok(prompt, max_length=max_len, truncation=True, return_tensors="pt")
    ids = enc.input_ids[0].tolist()
    L = int(enc.attention_mask.sum())
    pieces = tok.convert_ids_to_tokens(ids[:L])
    # T5 SentencePiece marks word starts with the ▁ meta char; clean for display.
    words = [p.replace("▁", " ").strip() or p for p in pieces]
    return dict(ids=ids[:L], pieces=pieces[:L], words=words, L=L)


def align_tokens(src, tgt):
    """Align source/target token id sequences (difflib). Returns the changed/inserted/deleted
    target-token index lists and a same/changed tag per target token."""
    sm = difflib.SequenceMatcher(a=src["ids"], b=tgt["ids"], autojunk=False)
    changed, inserted, deleted, tags = [], [], [], ["same"] * tgt["L"]
    for op, a0, a1, b0, b1 in sm.get_opcodes():
        if op == "equal":
            continue
        if op == "replace":
            for j in range(b0, b1):
                changed.append(j); tags[j] = "changed"
        elif op == "insert":
            for j in range(b0, b1):
                inserted.append(j); tags[j] = "inserted"
        elif op == "delete":
            deleted.append((a0, a1))
    return dict(changed=changed, inserted=inserted, deleted=deleted, tags=tags)


def assign_roles(tgt, align):
    """Best-effort, dependency-light token-role assignment over the TARGET prompt tokens.

    edited_noun : first changed/inserted content token (the thing the edit introduces);
    attribute   : a changed token in the attribute/color/material lexicon;
    style       : a token in the style lexicon (changed preferred, else any);
    background  : a background-lexicon token PRESENT IN BOTH prompts (an unedited region);
    control     : a function word / punctuation (should be edit-irrelevant).
    Returns {role: token_index or None} + the human-readable word. Honest heuristic; the
    report shows exactly which token got which role so the reader can judge."""
    words = [w.lower() for w in tgt["words"]]
    edit_idx = align["changed"] + align["inserted"]
    same_idx = [j for j in range(tgt["L"]) if align["tags"][j] == "same"]

    def first(idxs, pred):
        for j in idxs:
            if pred(words[j]):
                return j
        return None

    def pick(*cands):
        """First non-None candidate (index 0 is a valid choice, so don't use `or`)."""
        for c in cands:
            if c is not None:
                return c
        return None

    content = lambda w: w not in _CONTROL_WORDS and len(w) > 1 and any(c.isalpha() for c in w)
    roles = {}
    roles["edited_noun"] = pick(
        first(edit_idx, lambda w: content(w) and w not in _STYLE_WORDS and w not in _ATTR_WORDS),
        edit_idx[0] if edit_idx else None)
    roles["attribute"] = pick(first(edit_idx, lambda w: w in _ATTR_WORDS),
                              first(range(tgt["L"]), lambda w: w in _ATTR_WORDS))
    roles["style"] = pick(first(edit_idx, lambda w: w in _STYLE_WORDS),
                          first(range(tgt["L"]), lambda w: w in _STYLE_WORDS))
    roles["background"] = pick(first(same_idx, lambda w: w in _BG_WORDS),
                               first(same_idx, content))
    roles["control"] = pick(first(range(tgt["L"]), lambda w: w in _CONTROL_WORDS),
                            tgt["L"] - 1)
    out = {}
    for r, j in roles.items():
        out[r] = None if j is None else dict(index=int(j), word=tgt["words"][j],
                                             piece=tgt["pieces"][j])
    return out


# ---------------------------------------------------------------------------
# recording / intervening attention processor
# ---------------------------------------------------------------------------
class _Tap:
    """Per-installation shared state: what to record and which intervention to apply."""
    def __init__(self, spatial_hw=None):
        self.step = 0                      # set by the driver each denoise step
        self.record = False                # accumulate stats this forward?
        self.spatial = False               # also keep per-token spatial maps this forward?
        self.spatial_tokens = []           # token indices to keep spatial maps for
        self.spatial_hw = spatial_hw       # (H,W) of the image-token grid, e.g. (32,32)
        self.intervene = None              # None | dict(mech=, token=, value=)
        self.store = {}                    # block_id -> per-step list of stat dicts
        self.spatial_store = {}            # (block_id, step) -> {token: HxW np.float32}


def _qkv_postrope(attn, hidden_states, encoder_hidden_states, image_rotary_emb):
    """Replicate FluxAttnProcessor projections+norm+rope. Returns q,k,v as [B,heads,seq,hd]
    (heads-major, ready for scoring) and txt_len (#leading text positions, or 0)."""
    q = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
    k = attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1))
    v = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))
    q = attn.norm_q(q); k = attn.norm_k(k)
    txt_len = 0
    if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
        eq = attn.add_q_proj(encoder_hidden_states).unflatten(-1, (attn.heads, -1))
        ek = attn.add_k_proj(encoder_hidden_states).unflatten(-1, (attn.heads, -1))
        ev = attn.add_v_proj(encoder_hidden_states).unflatten(-1, (attn.heads, -1))
        eq = attn.norm_added_q(eq); ek = attn.norm_added_k(ek)
        q = torch.cat([eq, q], dim=1); k = torch.cat([ek, k], dim=1); v = torch.cat([ev, v], dim=1)
        txt_len = encoder_hidden_states.shape[1]
    if image_rotary_emb is not None:
        q = _apply_rotary_emb(q, image_rotary_emb, sequence_dim=1)
        k = _apply_rotary_emb(k, image_rotary_emb, sequence_dim=1)
    # [B,seq,heads,hd] -> [B,heads,seq,hd]
    return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), txt_len


@torch.no_grad()
def _record_stats(tap, block_id, q, k, v, txt_len):
    """Image→text attention statistics for one tapped (double-stream) block at the current
    step. q,k,v: [B,heads,seq,hd]; text = first txt_len keys, image = the rest."""
    B, H, seq, hd = q.shape
    n_img = seq - txt_len
    if txt_len == 0 or n_img <= 0:
        return
    scale = 1.0 / math.sqrt(hd)
    q_img = q[:, :, txt_len:, :].float()                       # [B,H,n_img,hd]
    kf = k.float(); vf = v.float()
    # full-key softmax (text + image) so "attention mass on token j" is a true fraction.
    mass = torch.zeros(txt_len, device=q.device)
    maxq = torch.zeros(txt_len, device=q.device)
    # head-mean spatial map accumulator only if requested
    spat = None
    if tap.spatial and tap.spatial_tokens:
        spat = {t: torch.zeros(n_img, device=q.device) for t in tap.spatial_tokens if t < txt_len}
    chunk = 256
    for s in range(0, n_img, chunk):
        qs = q_img[:, :, s:s + chunk, :]                       # [B,H,c,hd]
        sc = torch.matmul(qs, kf.transpose(-1, -2)) * scale     # [B,H,c,seq]
        p = torch.softmax(sc, dim=-1)
        p_txt = p[..., :txt_len]                               # [B,H,c,txt_len]
        hm = p_txt.mean(dim=1)                                 # head-mean [B,c,txt_len]
        mass += hm.sum(dim=(0, 1))                             # accumulate over queries
        maxq = torch.maximum(maxq, hm.amax(dim=(0, 1)))
        if spat is not None:
            for t in spat:
                spat[t][s:s + chunk] += hm[0, :, t]            # B=1 spatial response of token t
    denom = float(B * n_img)
    mass = (mass / denom)                                       # mean attention fraction per token
    valnorm = v[0, :, :txt_len, :].float().norm(dim=-1).mean(0)  # head-mean ||V_j||  [txt_len]
    rec = dict(mass=mass.cpu().numpy().astype(np.float32),
               maxq=maxq.cpu().numpy().astype(np.float32),
               valnorm=valnorm.cpu().numpy().astype(np.float32),
               contrib=(mass * valnorm).cpu().numpy().astype(np.float32))
    tap.store.setdefault(block_id, {})[tap.step] = rec
    if spat is not None and tap.spatial_hw is not None:
        Hh, Ww = tap.spatial_hw
        for t, vec in spat.items():
            if vec.numel() == Hh * Ww:
                tap.spatial_store[(block_id, tap.step, t)] = \
                    vec.reshape(Hh, Ww).cpu().numpy().astype(np.float32)


@torch.no_grad()
def _intervened_attention(attn, q, k, v, txt_len, interv):
    """Explicit-softmax joint attention with one internal intervention applied to a single
    text token. q,k,v: [B,heads,seq,hd]. Returns [B,seq,heads*hd]."""
    B, H, seq, hd = q.shape
    scale = 1.0 / math.sqrt(hd)
    qf, kf, vf = q.float(), k.float(), v.float()
    mech, tj, val = interv["mech"], interv["token"], interv["value"]
    if mech == "value_scale" and tj < txt_len:
        vf = vf.clone(); vf[:, :, tj, :] *= val
    sc = torch.matmul(qf, kf.transpose(-1, -2)) * scale          # [B,H,seq,seq]
    if mech == "attn_logit_bias" and tj < txt_len:
        sc[..., tj] += val                                       # additive bias (beta)
    p = torch.softmax(sc, dim=-1)
    if mech == "attn_prob_reweight" and tj < txt_len:
        p = p.clone(); p[..., tj] *= val
        p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)     # renormalize over keys
    out = torch.matmul(p, vf)                                    # [B,H,seq,hd]
    out = out.transpose(1, 2).reshape(B, seq, H * hd)
    return out.to(q.dtype)


class RecordingFluxAttnProcessor:
    """Drop-in for FluxAttnProcessor that records image→text attention and can intervene.
    Wraps the original processor for the untouched forward so the reference trajectory is
    bit-identical when no intervention is active."""
    def __init__(self, block_id, tap, orig, intervene_blocks):
        self.block_id = block_id
        self.tap = tap
        self._orig = orig
        self._intervene_blocks = intervene_blocks

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None, **kwargs):
        tap = self.tap
        do_rec = tap.record and encoder_hidden_states is not None
        do_int = (tap.intervene is not None and self.block_id in self._intervene_blocks
                  and encoder_hidden_states is not None)
        if not (do_rec or do_int):
            return self._orig(attn, hidden_states, encoder_hidden_states,
                              attention_mask, image_rotary_emb, **kwargs)

        q, k, v, txt_len = _qkv_postrope(attn, hidden_states, encoder_hidden_states, image_rotary_emb)
        if do_rec:
            _record_stats(tap, self.block_id, q, k, v, txt_len)
        if do_int:
            joint = _intervened_attention(attn, q, k, v, txt_len, tap.intervene)
            enc_out, img_out = joint.split_with_sizes([txt_len, joint.shape[1] - txt_len], dim=1)
            img_out = attn.to_out[1](attn.to_out[0](img_out.contiguous()))
            enc_out = attn.to_add_out(enc_out.contiguous())
            return img_out, enc_out
        # record-only: return the exact original output
        return self._orig(attn, hidden_states, encoder_hidden_states,
                          attention_mask, image_rotary_emb, **kwargs)


def install_taps(pipe, block_ids, intervene_blocks=None, spatial_hw=None):
    """Swap a RecordingFluxAttnProcessor onto transformer.transformer_blocks[i].attn for each
    double-stream block i in block_ids. Returns (tap, restore_fn)."""
    intervene_blocks = set(intervene_blocks if intervene_blocks is not None else block_ids)
    tap = _Tap(spatial_hw=spatial_hw)
    blocks = pipe.transformer.transformer_blocks
    saved = {}
    for i in block_ids:
        if i >= len(blocks):
            continue
        attn = blocks[i].attn
        saved[i] = attn.processor
        attn.set_processor(RecordingFluxAttnProcessor(i, tap, saved[i], intervene_blocks))

    def restore():
        for i, proc in saved.items():
            pipe.transformer.transformer_blocks[i].attn.set_processor(proc)
    return tap, restore
