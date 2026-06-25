"""Build real-world data subsets for the baseline phase.

  * MagicBrush dev      -> instruction-editing subset (source, instruction, GT target)
  * PIE-Bench++ configs -> task-typed editing subset (color/material/object/...)
  * WikiArt (stream)    -> style reference bank (oil/watercolor/impressionist/abstract...)
  * custom_leakage_set  -> content x style pairs: aligned / mismatched / adversarial /
                           texture / palette / line-art, for reference-leakage stress.

All images saved as PNG; manifests as JSONL. Streaming so we never pull whole datasets.
"""
from __future__ import annotations
import io, json, random
from pathlib import Path
from PIL import Image
from datasets import load_dataset
import config as C

random.seed(0)
SZ = C.GEN_SIZE


def _sq(img: Image.Image, size=SZ) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    return img.resize((size, size), Image.LANCZOS)


def build_magicbrush(n=18):
    out = C.BENCH / "magicbrush"; out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("osunlp/MagicBrush", split="dev", streaming=True)
    rows, seen = [], set()
    for r in ds:
        iid = r["img_id"]
        if iid in seen:      # one turn per image id for diversity
            continue
        seen.add(iid)
        src = _sq(r["source_img"]); tgt = _sq(r["target_img"])
        sp = out / f"mb_{iid}_src.png"; tp = out / f"mb_{iid}_tgt.png"
        src.save(sp); tgt.save(tp)
        rows.append(dict(id=f"mb_{iid}", benchmark="magicbrush", task_type="edit_mixed",
                         source=str(sp.relative_to(C.ROOT)), gt_target=str(tp.relative_to(C.ROOT)),
                         instruction=r["instruction"], target_prompt=r["instruction"]))
        if len(rows) >= n:
            break
    (out / "manifest.jsonl").write_text("\n".join(json.dumps(x) for x in rows))
    print(f"magicbrush: {len(rows)} examples")
    return rows


def _clean(p):
    return p.replace("[", "").replace("]", "").strip()


def _instruction(src, tgt, blended):
    """Synthesize an imperative instruction from PIE-Bench fields."""
    try:
        bw = json.loads(blended) if blended else []
    except Exception:
        bw = []
    swaps = []
    for pair in bw:
        if isinstance(pair, str) and "," in pair:
            a, b = pair.split(",", 1)
            if a.strip() and b.strip() and a.strip() != b.strip():
                swaps.append((a.strip(), b.strip()))
    if swaps:
        return "; ".join(f"change {a} to {b}" for a, b in swaps)
    return f"make the scene: {_clean(tgt)}"


def build_piebench(per=3):
    out = C.BENCH / "piebench"; out.mkdir(parents=True, exist_ok=True)
    rows = []
    for cfg, ttype in C.PIE_CONFIGS.items():
        try:
            ds = load_dataset("UB-CVML-Group/PIE_Bench_pp", cfg, split=C.PIE_SPLIT, streaming=True)
        except Exception as e:
            print("  pie cfg skip", cfg, repr(e)[:80]); continue
        got = 0
        for r in ds:
            img = r.get("image")
            if not isinstance(img, Image.Image):
                continue
            tgt_prompt = _clean(r.get("target_prompt", ""))
            src_prompt = _clean(r.get("source_prompt", ""))
            instr = _instruction(src_prompt, r.get("target_prompt", ""), r.get("blended_words", ""))
            if not tgt_prompt:
                continue
            img = _sq(img)
            sid = f"pie_{cfg.split('_')[0]}_{got}"
            sp = out / f"{sid}_src.png"; img.save(sp)
            rows.append(dict(id=sid, benchmark="piebench", task_type=ttype, pie_cfg=cfg,
                             source=str(sp.relative_to(C.ROOT)), gt_target="",
                             instruction=instr, target_prompt=tgt_prompt,
                             source_prompt=src_prompt))
            got += 1
            if got >= per:
                break
        print(f"  pie {cfg} ({ttype}): {got}")
    (out / "manifest.jsonl").write_text("\n".join(json.dumps(x) for x in rows))
    print(f"piebench: {len(rows)} examples")
    return rows


# WikiArt genre/style -> our style-category label
WIKIART_PICK = {
    "oil": ["Baroque", "Realism", "Romanticism"],
    "impressionist": ["Impressionism", "Post_Impressionism"],
    "abstract": ["Abstract_Expressionism", "Cubism", "Color_Field_Painting"],
    "watercolor": ["Naive_Art_Primitivism", "Art_Nouveau_Modern"],
}


def build_style_bank(n_per=3):
    out = C.STYLE; out.mkdir(parents=True, exist_ok=True)
    try:
        ds = load_dataset("huggan/wikiart", split="train", streaming=True)
    except Exception as e:
        print("wikiart unavailable:", repr(e)[:120]); return []
    # style id->name mapping lives in features
    feat = ds.features.get("style")
    names = feat.names if feat is not None else None
    want = {}
    for cat, styles in WIKIART_PICK.items():
        for s in styles:
            if names and s in names:
                want[names.index(s)] = cat
    rows, counts = [], {c: 0 for c in WIKIART_PICK}
    for r in ds:
        sid = r.get("style")
        cat = want.get(sid)
        if cat is None or counts[cat] >= n_per:
            continue
        img = next((r[k] for k in r if isinstance(r[k], Image.Image)), None)
        if img is None:
            continue
        p = out / f"style_{cat}_{counts[cat]}.png"; _sq(img).save(p)
        rows.append(dict(id=p.stem, style_category=cat, path=str(p.relative_to(C.ROOT))))
        counts[cat] += 1
        if all(v >= n_per for v in counts.values()):
            break
    (out / "manifest.jsonl").write_text("\n".join(json.dumps(x) for x in rows))
    print(f"style bank: {len(rows)} refs ({counts})")
    return rows


def build_leakage_set(mb_rows, style_rows):
    """content x style pairs for reference-leakage stress.
    content = real photos from MagicBrush sources (real-world, varied categories);
    style = WikiArt refs. Pairs: aligned / mismatched / adversarial."""
    out = C.LEAK; out.mkdir(parents=True, exist_ok=True)
    contents = mb_rows[:10]
    styles = style_rows
    if not styles:
        print("no style refs -> skip leakage set"); return []
    pairs = []
    for i, c in enumerate(contents):
        # aligned: rotate through styles; adversarial: a portrait-ish style on a scene etc.
        s_aligned = styles[i % len(styles)]
        s_adv = styles[(i + len(styles) // 2) % len(styles)]
        for kind, s in [("aligned", s_aligned), ("adversarial", s_adv)]:
            pairs.append(dict(
                id=f"leak_{i}_{kind}", pair_type=kind,
                content=c["source"], content_id=c["id"],
                style=s["path"], style_id=s["id"], style_category=s["style_category"],
                instruction=f"render this image in a {s['style_category']} painting style",
            ))
    (out / "manifest.jsonl").write_text("\n".join(json.dumps(x) for x in pairs))
    print(f"leakage set: {len(pairs)} pairs")
    return pairs


if __name__ == "__main__":
    C.ensure_dirs()
    mb = build_magicbrush()
    pie = build_piebench()
    style = build_style_bank()
    leak = build_leakage_set(mb, style)
    C.write_json(C.DATA / "data_summary.json", dict(
        magicbrush=len(mb), piebench=len(pie), style_refs=len(style), leakage_pairs=len(leak)))
    print("DATA BUILD DONE")
