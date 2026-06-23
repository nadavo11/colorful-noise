"""Aggregate an experiment's visual results for the roadmap.

For each (source image, web name) it:
  - copies the FULL-RES original to /storage/.../roadmap_results/<eid>/   (not git; the archive)
  - writes a downscaled JPEG to docs/experiment-reports/figs/<eid>/       (git; light, embeddable)

The roadmap renderer (make_roadmap.py) inlines the git figs as data-URIs, so the report pages are
self-contained. Reference the figs from EXPERIMENT_<n>.md as `![caption](figs/<EID>/<name>.jpg)`.

Usage (edit AGG below, or import `aggregate`):  python experiments/aggregate_results.py E47
"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
STORE = os.environ.get("CN_ROADMAP_STORE", "/storage/malnick/colorful-noise/roadmap_results")
FIGS = os.path.join(REPO, "docs", "experiment-reports", "figs")
TMP = os.path.expanduser("~/.claude/jobs/7343f13a/tmp")


def _web_jpeg(src, dst, max_px=1600, quality=85):
    from PIL import Image
    im = Image.open(src).convert("RGB")
    if max(im.size) > max_px:
        r = max_px / max(im.size)
        im = im.resize((max(1, round(im.width * r)), max(1, round(im.height * r))))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    im.save(dst, "JPEG", quality=quality)
    return os.path.getsize(dst)


def aggregate(eid, items):
    """items: list of (src_path, web_name). web_name has no extension (-> .jpg)."""
    store = os.path.join(STORE, eid)
    os.makedirs(store, exist_ok=True)
    for src, web in items:
        if not os.path.exists(src):
            print(f"  MISS {src}"); continue
        shutil.copy2(src, os.path.join(store, os.path.basename(src)))   # full-res archive
        dst = os.path.join(FIGS, eid, web + ".jpg")
        kb = _web_jpeg(src, dst) // 1024
        print(f"  OK {web}.jpg ({kb} KB)  <- {os.path.basename(src)}")


# Per-experiment figure manifests (extend as we backfill the roadmap).
AGG = {
    "E47": [
        (os.path.join(REPO, "docs/experiment-reports/e47_frontier.png"), "frontier"),
        (os.path.join(TMP, "e47_chair_all.png"),  "chair_all_methods"),
        (os.path.join(TMP, "e47_sdg_chair.png"),  "sdg_chair_sweep"),
        (os.path.join(TMP, "e47_confA_grid.png"), "confA_piebench_n100"),
        (os.path.join(TMP, "e47_confSDG_grid.png"), "confSDG_piebench_n100"),
    ],
}

if __name__ == "__main__":
    for eid in (sys.argv[1:] or AGG.keys()):
        print(f"[{eid}] -> figs {os.path.join(FIGS, eid)} ; archive {os.path.join(STORE, eid)}")
        aggregate(eid, AGG[eid])
