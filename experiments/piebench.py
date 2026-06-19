"""PIE-Bench loader for the e41 RF-inversion comparison.

Returns a stratified subset of (source image, source prompt, edit prompt, edit mask,
edit type) tuples. Two sources, in priority order:

1. A local original PIE-Bench checkout (env CN_PIEBENCH -> dir with mapping_file*.json +
   annotation_images/), if present.
2. The HF-hosted PIE-Bench++ (UB-CVML-Group/PIE_Bench_pp): same 700 images, Parquet with
   source_prompt / target_prompt / mask / image, split into per-edit-type configs. This is
   the default and includes the segmentation masks needed for background metrics.

Stratified: take `n_per_type` items (stride-sampled) from each edit-type group, so ~140
total at n_per_type=14 across the 10 PIE-Bench categories.
"""
import glob
import json
import os

import numpy as np
from PIL import Image

HF_REPO = "UB-CVML-Group/PIE_Bench_pp"
_CACHE = os.environ.get("CN_PIEBENCH_CACHE", "/storage/malnick/datasets/pie_bench_hf")


def _stride_idx(n, k):
    if k >= n:
        return list(range(n))
    return sorted(set(np.linspace(0, n - 1, k).astype(int).tolist()))


def _as_mask(m):
    """Coerce a dataset 'mask' field (PIL image, path str, or other) to an L-mode mask
    or None. PIE-Bench++ stores it as a string we can't use directly -> background
    metrics are skipped for those (core DINO/LPIPS/CLIP-dir don't need a mask)."""
    if isinstance(m, Image.Image):
        return m.convert("L")
    if isinstance(m, str) and os.path.exists(m):
        return Image.open(m).convert("L")
    return None


def _load_hf(n_per_type):
    import collections
    import shutil
    # /storage reports df=100% (quota cap) but actually accepts writes; neutralize the
    # datasets builder's free-space precheck which otherwise hard-fails ("Not enough disk space").
    shutil.disk_usage = lambda path: collections.namedtuple(
        "usage", ["total", "used", "free"])(1 << 60, 0, 1 << 60)
    from datasets import get_dataset_config_names, load_dataset
    configs = sorted(get_dataset_config_names(HF_REPO))      # e.g. "1_change_object_80"
    items = []
    for cfg in configs:
        dd = load_dataset(HF_REPO, cfg, cache_dir=_CACHE)   # split is "V1", not "train"
        ds = dd[next(iter(dd))]
        etype = "_".join(cfg.split("_")[1:-1]) or cfg        # drop leading idx + trailing count
        for j in _stride_idx(len(ds), n_per_type):
            r = ds[j]
            items.append({
                "key": f"{cfg}/{r['id']}",
                "src_img": r["image"].convert("RGB"),
                "src_prompt": r["source_prompt"],
                "edit_prompt": r["target_prompt"],
                "mask": _as_mask(r.get("mask")),
                "edit_type": etype,
            })
    return items


def _load_local(root, n_per_type):
    mapping = sorted(glob.glob(os.path.join(root, "mapping_file*.json")))[0]
    with open(mapping) as f:
        m = json.load(f)
    groups = {}
    for k, v in m.items():
        groups.setdefault(str(v.get("editing_type_id", "0")), []).append((k, v))
    items = []
    for etype, entries in sorted(groups.items()):
        entries.sort(key=lambda kv: kv[0])
        for j in _stride_idx(len(entries), n_per_type):
            k, v = entries[j]
            img = Image.open(os.path.join(root, "annotation_images", v["image_path"]))
            mpath = os.path.join(root, "annotation_images",
                                 v["image_path"].rsplit(".", 1)[0] + "_mask.png")
            items.append({
                "key": f"{etype}/{k}",
                "src_img": img.convert("RGB"),
                "src_prompt": v["original_prompt"],
                "edit_prompt": v["editing_prompt"],
                "mask": Image.open(mpath).convert("L") if os.path.exists(mpath) else None,
                "edit_type": etype,
            })
    return items


def load_piebench(n_per_type=14):
    local = os.environ.get("CN_PIEBENCH")
    if local and glob.glob(os.path.join(local, "mapping_file*.json")):
        print(f"[piebench] loading local original PIE-Bench from {local}", flush=True)
        items = _load_local(local, n_per_type)
    else:
        print(f"[piebench] loading HF {HF_REPO} (cache {_CACHE})", flush=True)
        items = _load_hf(n_per_type)
    by_type = {}
    for it in items:
        by_type[it["edit_type"]] = by_type.get(it["edit_type"], 0) + 1
    print(f"[piebench] {len(items)} items; per type: {by_type}", flush=True)
    return items


if __name__ == "__main__":
    its = load_piebench(int(os.environ.get("N_PER_TYPE", 2)))
    print(f"sample key={its[0]['key']} src={its[0]['src_prompt']!r} "
          f"edit={its[0]['edit_prompt']!r} mask={'yes' if its[0]['mask'] else 'no'} "
          f"img={its[0]['src_img'].size}")
