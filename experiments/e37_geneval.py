"""E37 — GenEval evaluation of velocity spectral normalization on SD3.5 medium.

Generates the GenEval prompt set under several VELOCITY-modulation conditions and scores
them with `geneval_score.py` (GenEval protocol, torchvision detector + CLIP colours).

Conditions (7), all at strength 1.0, intervening on EVERY step, guidance w=4.5:
  baseline                      plain CFG, no velocity edit
  mag_full / mag_top25 / mag_bot25     per-bin magnitude transplant |V_w|<-|V_uncond|
                                       on band [0,1] / [0.75,1] / [0,0.25]
  bp_full  / bp_top25  / bp_bot25      per-band mean-power match (psd_match) on same bands

Each generation uses seed = prompt-index, IDENTICAL across conditions, so a condition differs
from baseline only by the operator (paired comparison). Quick first pass: n=1 sample/prompt.

parts:
  preflight  no GPU; list conditions, prompt counts, image total + ETA
  gen        generate (cached PNGs) into GenEval folder layout per condition
  score      run the detector/CLIP scorer over each condition -> results jsonl
  summary    aggregate per-tag + overall (macro/micro) per condition -> report.json + table

Layout (GenEval-compatible, per condition):
  results/e37_geneval/<cond>/<idx:05d>/metadata.jsonl   (single JSON: the prompt spec)
  results/e37_geneval/<cond>/<idx:05d>/samples/0000.png
  results/e37_geneval/scores/<cond>.jsonl               (per-image correctness)
  results/e37_geneval/report.json                       (aggregated table)
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.environ.get("CN_RESULTS") or os.path.join(HERE, "results")
OUT = os.path.join(RESULTS, "e37_geneval")
META_PATH = os.path.join(HERE, "geneval_data", "evaluation_metadata.jsonl")
TAGS = ["single_object", "two_object", "counting", "colors", "position", "color_attr"]

# (name, op, lo, hi, gain, t_lo, t_hi)  -- op None = baseline (plain CFG); strength fixed at 1.0.
# gain only used by op="gain" (band amplify/reduce). [t_lo,t_hi] = fraction of the denoising
# schedule to intervene on ([0,1] = every step).
CONDITIONS = [
    ("baseline",            None,         None, None, 1.0, 0.0,  1.0),
    ("mag_full",            "mag",        0.0,  1.0,  1.0, 0.0,  1.0),
    ("mag_top25",           "mag",        0.75, 1.0,  1.0, 0.0,  1.0),
    ("mag_bot25",           "mag",        0.0,  0.25, 1.0, 0.0,  1.0),
    ("bp_full",             "band power", 0.0,  1.0,  1.0, 0.0,  1.0),
    ("bp_top25",            "band power", 0.75, 1.0,  1.0, 0.0,  1.0),
    ("bp_bot25",            "band power", 0.0,  0.25, 1.0, 0.0,  1.0),
    # band amplify (gain 1.6) on the top-0.25 freq band [0.75,1], at 3 timestep intervals
    ("amp1.6_top25_full",   "gain",       0.75, 1.0,  1.6, 0.0,  1.0),
    ("amp1.6_top25_late50", "gain",       0.75, 1.0,  1.6, 0.5,  1.0),
    ("amp1.6_top25_late25", "gain",       0.75, 1.0,  1.6, 0.75, 1.0),
]


def load_prompts(n=None, spread=False):
    prompts = [json.loads(l) for l in open(META_PATH) if l.strip()]
    if not n:
        return prompts
    if spread:                                   # evenly-spaced subset spanning all tags
        step = max(1, len(prompts) // n)
        return prompts[::step][:n]
    return prompts[:n]


def select_conditions(names):
    if not names:
        return CONDITIONS
    keep = set(names.split(","))
    return [c for c in CONDITIONS if c[0] in keep]


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

def _override(op, lo, hi, gain, t_lo, t_hi, steps, n_bins):
    import velocity_spectral_ops as VEL
    if op is None:
        return None
    i_lo = int(round(float(t_lo) * (int(steps) - 1)))
    i_hi = int(round(float(t_hi) * (int(steps) - 1)))
    return VEL.make_velocity_override(op, lo, hi, 1.0, float(gain), i_lo, i_hi, n_bins=n_bins)


def run_gen(args):
    import torch  # noqa
    import e17_sd35 as SD
    SD.SIZE = int(args.size)                       # gen_sd3 reads module SIZE for height/width
    prompts = load_prompts(args.num_prompts, args.spread)
    conds = select_conditions(args.conditions)
    pipe = SD.load_sd35(args.mem)
    n_bins = args.n_bins
    t0 = time.time()
    done = skipped = 0
    for name, op, lo, hi, gain, t_lo, t_hi in conds:
        for idx, meta in enumerate(prompts):
            d = os.path.join(OUT, name, f"{idx:05d}")
            os.makedirs(os.path.join(d, "samples"), exist_ok=True)
            json.dump(meta, open(os.path.join(d, "metadata.jsonl"), "w"))
            png = os.path.join(d, "samples", "0000.png")
            if os.path.exists(png):
                skipped += 1
                continue
            ov = _override(op, lo, hi, gain, t_lo, t_hi, args.steps, n_bins)
            img, _ = SD.gen_sd3(pipe, meta["prompt"], seed=idx, guidance=args.guidance,
                                steps=args.steps, step_override=ov)
            img.save(png)
            done += 1
            if done % 50 == 0:
                dt = time.time() - t0
                print(f"[gen] {name} {idx}/{len(prompts)} · {done} new ({dt/done:.2f}s/img), "
                      f"{skipped} cached", flush=True)
    print(f"[gen] done: {done} generated, {skipped} cached, {time.time()-t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def run_score(args):
    import geneval_score as G
    detector, name2idx, classes = G.load_detector()
    clip = G.load_clip()
    color_fn = G.make_color_fn(clip, "cuda")
    conds = select_conditions(args.conditions)
    os.makedirs(os.path.join(OUT, "scores"), exist_ok=True)
    for name, *_ in conds:
        cdir = os.path.join(OUT, name)
        if not os.path.isdir(cdir):
            print(f"[score] {name}: no images, skip", flush=True)
            continue
        results = []
        folders = sorted(f for f in os.listdir(cdir) if f.isdigit())
        for f in folders:
            meta = json.load(open(os.path.join(cdir, f, "metadata.jsonl")))
            png = os.path.join(cdir, f, "samples", "0000.png")
            if not os.path.exists(png):
                continue
            results.append(G.evaluate_image(detector, name2idx, classes, color_fn, png, meta))
        with open(os.path.join(OUT, "scores", f"{name}.jsonl"), "w") as fp:
            for r in results:
                fp.write(json.dumps(r) + "\n")
        acc = sum(r["correct"] for r in results) / max(1, len(results))
        print(f"[score] {name}: {len(results)} imgs · overall {acc:.4f}", flush=True)


# ---------------------------------------------------------------------------
# summary / aggregation
# ---------------------------------------------------------------------------

def _agg(results):
    by_tag = {t: [] for t in TAGS}
    for r in results:
        by_tag[r["tag"]].append(1.0 if r["correct"] else 0.0)
    tag_acc = {t: (sum(v) / len(v) if v else None) for t, v in by_tag.items()}
    micro = sum(1.0 if r["correct"] else 0.0 for r in results) / max(1, len(results))
    present = [tag_acc[t] for t in TAGS if tag_acc[t] is not None]
    macro = sum(present) / len(present) if present else None    # leaderboard "Overall"
    return tag_acc, macro, micro, len(results)


def run_summary(args):
    conds = select_conditions(args.conditions)
    report = {"conditions": {}, "tags": TAGS, "guidance": args.guidance,
              "size": args.size, "steps": args.steps, "n_samples": 1}
    rows = []
    for name, *_ in conds:
        sp = os.path.join(OUT, "scores", f"{name}.jsonl")
        if not os.path.exists(sp):
            continue
        results = [json.loads(l) for l in open(sp) if l.strip()]
        tag_acc, macro, micro, n = _agg(results)
        report["conditions"][name] = {"tag_acc": tag_acc, "overall_macro": macro,
                                      "overall_micro": micro, "n": n}
        rows.append((name, macro, micro, tag_acc))
    os.makedirs(OUT, exist_ok=True)
    json.dump(report, open(os.path.join(OUT, "report.json"), "w"), indent=2)
    # printed table
    hdr = f"{'condition':<11} {'Overall':>8} {'micro':>7} " + " ".join(f"{t[:9]:>9}" for t in TAGS)
    print(hdr); print("-" * len(hdr))
    for name, macro, micro, tag_acc in rows:
        cells = " ".join(f"{(tag_acc[t] if tag_acc[t] is not None else float('nan')):>9.3f}" for t in TAGS)
        print(f"{name:<11} {macro:>8.4f} {micro:>7.4f} {cells}")
    print(f"\n[summary] wrote {os.path.join(OUT, 'report.json')}", flush=True)


def _idx_of(path):
    return os.path.basename(os.path.dirname(os.path.dirname(path)))   # .../<idx>/samples/x.png


def _thumb_b64(path, px=320):
    import base64, io
    from PIL import Image
    im = Image.open(path).convert("RGB")
    im.thumbnail((px, px))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def run_site(args):
    """Self-contained HTML comparing baseline vs a condition on one tag (default counting),
    wins-first (baseline wrong & condition right). Images base64-embedded thumbnails."""
    cmp = args.compare
    tag = args.site_tag
    def load(cond):
        sp = os.path.join(OUT, "scores", f"{cond}.jsonl")
        out = {}
        for l in open(sp):
            r = json.loads(l)
            if r["tag"] == tag:
                out[_idx_of(r["filename"])] = r
        return out
    base, alt = load("baseline"), load(cmp)
    idxs = sorted(set(base) & set(alt))
    def rank(i):                                   # wins first, then losses, then ties
        b, a = base[i]["correct"], alt[i]["correct"]
        return (0 if (a and not b) else 1 if (b and not a) else 2, i)
    idxs.sort(key=rank)
    n_win = sum(alt[i]["correct"] and not base[i]["correct"] for i in idxs)
    n_loss = sum(base[i]["correct"] and not alt[i]["correct"] for i in idxs)
    b_acc = sum(base[i]["correct"] for i in idxs) / max(1, len(idxs))
    a_acc = sum(alt[i]["correct"] for i in idxs) / max(1, len(idxs))
    shown = idxs[:args.site_max] if args.site_max else idxs   # wins-first; cap for file size

    def cell(rec):
        png = rec["filename"]
        badge = "✓" if rec["correct"] else "✗"
        cls = "ok" if rec["correct"] else "bad"
        reason = (rec["reason"] or "all checks passed").replace("\n", "; ")
        return (f"<td class='{cls}'><img src='data:image/jpeg;base64,{_thumb_b64(png)}'>"
                f"<div class='badge'>{badge}</div><div class='why'>{reason}</div></td>")

    rows = []
    for i in shown:
        prompt = base[i]["prompt"]
        rows.append(f"<tr><td class='p'>{prompt}</td>{cell(base[i])}{cell(alt[i])}</tr>")
    html = f"""<!doctype html><meta charset=utf8><title>E37 GenEval {tag}: baseline vs {cmp}</title>
<style>body{{font:14px system-ui;margin:24px;max-width:1100px}}
table{{border-collapse:collapse;width:100%}} td{{border:1px solid #ddd;padding:8px;vertical-align:top;text-align:center}}
td.p{{text-align:left;width:200px;font-weight:600}} img{{width:300px;height:auto;border-radius:4px}}
.badge{{font-size:20px;font-weight:700}} td.ok .badge{{color:#2da44e}} td.bad .badge{{color:#cf222e}}
.why{{font-size:11px;color:#666;margin-top:4px}} th{{padding:8px;border-bottom:2px solid #333}}
.hd{{background:#f6f8fa}}</style>
<h1>E37 — GenEval <code>{tag}</code>: baseline vs <code>{cmp}</code></h1>
<p>SD3.5-medium, n=1, 512px, w=4.5 · velocity spectral normalization (mag→cfg1) ·
torchvision-detector GenEval variant. <b>{tag}</b> accuracy: baseline <b>{b_acc:.3f}</b> →
{cmp} <b>{a_acc:.3f}</b>. Of {len(idxs)} prompts: <b>{n_win}</b> {cmp} fixes a baseline miss,
{n_loss} regressions. Sorted wins-first. Each cell: image, ✓/✗, and the scorer's reason
(e.g. "found 1" = detected count).</p>
<table><tr class=hd><th>prompt (expected count)</th><th>baseline</th><th>{cmp}</th></tr>
{''.join(rows)}</table>"""
    out_path = args.html_out or os.path.join(OUT, f"examples_{tag}.html")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    open(out_path, "w").write(html)
    print(f"[site] {tag}: baseline {b_acc:.3f} -> {cmp} {a_acc:.3f} "
          f"({n_win} wins, {n_loss} losses) · wrote {out_path}", flush=True)


def run_preflight(args):
    prompts = load_prompts(args.num_prompts, args.spread)
    conds = select_conditions(args.conditions)
    from collections import Counter
    tags = Counter(p["tag"] for p in prompts)
    total = len(prompts) * len(conds)
    print(f"[preflight] prompts: {len(prompts)}  tags: {dict(tags)}")
    print(f"[preflight] conditions ({len(conds)}): {[c[0] for c in conds]}")
    print(f"[preflight] total images (n=1): {total}")
    print(f"[preflight] guidance={args.guidance} steps={args.steps} size={args.size}")
    eta = total * args.sec_per_img / 3600
    print(f"[preflight] gen ETA @ {args.sec_per_img}s/img: {eta:.1f} h")


def main():
    ap = argparse.ArgumentParser(description="E37 GenEval eval (velocity spectral normalization, SD3.5).")
    ap.add_argument("--part", default="preflight", help="comma list: preflight,gen,score,summary")
    ap.add_argument("--conditions", default="", help="comma list of condition names (default: all 7)")
    ap.add_argument("--num_prompts", type=int, default=0, help="cap prompts (0 = all 553; e.g. 2 for smoke)")
    ap.add_argument("--spread", action="store_true", help="evenly-spaced prompt subset spanning all tags (smoke)")
    ap.add_argument("--guidance", type=float, default=4.5)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--n_bins", type=int, default=24)
    ap.add_argument("--mem", default="gpu_resident", choices=["gpu_resident", "offload"])
    ap.add_argument("--sec_per_img", type=float, default=4.0, help="ETA estimate only")
    ap.add_argument("--compare", default="mag_top25", help="condition to compare vs baseline in --part site")
    ap.add_argument("--site_tag", default="counting", help="GenEval tag to show in --part site")
    ap.add_argument("--html_out", default="", help="output path for --part site HTML")
    ap.add_argument("--site_max", type=int, default=40, help="max rows in --part site (0=all), wins-first")
    args = ap.parse_args()
    parts = [p.strip() for p in args.part.split(",") if p.strip()]
    for p in parts:
        {"preflight": run_preflight, "gen": run_gen, "score": run_score,
         "summary": run_summary, "site": run_site}[p](args)


if __name__ == "__main__":
    main()
