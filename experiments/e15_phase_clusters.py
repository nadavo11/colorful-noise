"""E15: do phase manipulations map to consistent output classes?

E13/E14 produced a battery of phase-manipulated decodes. E15 asks whether the
manipulation -> output relation is structured enough to "classify": embed every
output (CLIP ViT-L) + image_metrics, then cluster / project and test whether
images group by *manipulation family* independent of seed and image class.

Consumes the cached PNGs from results/e13 and results/e14 (no generation, no
re-diffusion). CLIP embeddings are cached to results/e15/embeddings.pt.

Metrics:
  - KMeans (k = #manipulation families) purity + silhouette vs manipulation
    labels, contrasted with the same vs class labels.
  - Nearest-neighbour manipulation-consistency: fraction of images whose CLIP
    nearest neighbour shares the manipulation family (vs the class baseline).
  - 2D PCA projection colored by manipulation and by class.

Expected (roadmap): low-band-touching manipulations cluster by effect;
high-band-only edits collapse near the unmodified ('orig') cluster.

sklearn is used if importable; otherwise compact numpy fallbacks run.
"""
import argparse
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS
from e9_bandnorm_classes import image_metrics, METRICS
from clip_sim import load_clip, clip_image_features

E13_IMG = os.path.join(RESULTS, "e13", "images")
E14_IMG = os.path.join(RESULTS, "e14", "images")


# --- manipulation taxonomy: map a filename to (class, manipulation family) ----

def e13_family(variant):
    return {"baseA": "orig", "baseB": "orig",
            "Aphase_Bmag": "phase_mag_swap", "Bphase_Amag": "phase_mag_swap",
            "phaseonly_A": "phase_only", "magonly_A": "mag_only"}.get(variant)


def e14_family(tag):
    if tag == "base":
        return "orig"
    if tag.startswith("scale_"):
        return "scale"
    if tag.startswith("shift_"):
        return "shift"
    if tag.startswith("offset_"):
        return "offset"
    m = re.match(r"rot_(low|high)_", tag)
    if m:
        return f"rotate_{m.group(1)}"
    m = re.match(r"noise_(low|high)_", tag)
    if m:
        return f"noise_{m.group(1)}"
    return None


def collect(args):
    """Return (paths, classes, families) over all E13/E14 outputs."""
    items = []
    if os.path.isdir(E13_IMG):
        for fn in sorted(os.listdir(E13_IMG)):
            m = re.match(r"(.+)_p\d+_(.+)\.png$", fn)
            if not m:
                continue
            fam = e13_family(m.group(2))
            if fam:
                items.append((f"{E13_IMG}/{fn}", m.group(1), fam))
    if os.path.isdir(E14_IMG):
        for fn in sorted(os.listdir(E14_IMG)):
            m = re.match(r"(.+?)_(base|scale_.+|shift_.+|offset_.+|rot_.+|noise_.+)\.png$", fn)
            if not m:
                continue
            fam = e14_family(m.group(2))
            if fam:
                items.append((f"{E14_IMG}/{fn}", m.group(1), fam))
    paths = [p for p, _, _ in items]
    classes = [c for _, c, _ in items]
    families = [f for _, _, f in items]
    return paths, classes, families


# --- numpy fallbacks (used iff sklearn missing) -------------------------------

def _pca2(X):
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T


def _kmeans(X, k, iters=100, seed=0):
    rng = np.random.default_rng(seed)
    # k-means++ init
    c = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        d = np.min([((X - ci) ** 2).sum(1) for ci in c], axis=0)
        p = d / d.sum()
        c.append(X[rng.choice(len(X), p=p)])
    C = np.stack(c)
    for _ in range(iters):
        lab = np.argmin(((X[:, None] - C[None]) ** 2).sum(2), axis=1)
        newC = np.stack([X[lab == j].mean(0) if (lab == j).any() else C[j]
                         for j in range(k)])
        if np.allclose(newC, C):
            break
        C = newC
    return lab


def _silhouette(X, lab):
    # mean silhouette over samples (O(n^2), fine for a few hundred points)
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(2)) + 1e-12
    labs = np.unique(lab)
    s = np.zeros(len(X))
    for i in range(len(X)):
        same = lab == lab[i]
        same[i] = False
        a = D[i, same].mean() if same.any() else 0.0
        b = min(D[i, lab == L].mean() for L in labs if L != lab[i]) if len(labs) > 1 else 0.0
        s[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return float(s.mean())


def cluster_and_score(X, labels):
    """KMeans into len(unique labels) clusters; return (assignments, purity,
    silhouette-vs-labels). Uses sklearn if available else numpy."""
    uniq = sorted(set(labels))
    k = len(uniq)
    li = np.array([uniq.index(l) for l in labels])
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        assign = km.labels_
        sil = float(silhouette_score(X, li)) if k > 1 else 0.0
    except Exception:
        assign = _kmeans(X, k, seed=0)
        sil = _silhouette(X, li) if k > 1 else 0.0
    # purity: each cluster takes its majority label
    purity = 0.0
    for c in np.unique(assign):
        members = li[assign == c]
        if len(members):
            purity += np.bincount(members, minlength=k).max()
    purity /= len(li)
    return assign, float(purity), sil


def nn_consistency(X, labels):
    """Fraction of points whose nearest neighbour (excl. self) shares the label."""
    D = ((X[:, None] - X[None]) ** 2).sum(2)
    np.fill_diagonal(D, np.inf)
    nn = D.argmin(1)
    lab = np.array(labels)
    return float((lab[nn] == lab).mean())


def project_plot(P, labels, title, path):
    fig, ax = plt.subplots(figsize=(8, 6))
    for lab in sorted(set(labels)):
        m = np.array(labels) == lab
        ax.scatter(P[m, 0], P[m, 1], s=18, label=lab, alpha=0.8)
    ax.set(title=title, xlabel="PC1", ylabel="PC2")
    ax.legend(fontsize=7, markerscale=1.5, ncol=2)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main(args):
    out = os.path.join(RESULTS, "e15")
    os.makedirs(f"{out}/plots", exist_ok=True)
    paths, classes, families = collect(args)
    assert paths, f"no E13/E14 PNGs found under {E13_IMG} or {E14_IMG}"
    print(f"[e15] {len(paths)} images, {len(set(families))} manipulation families, "
          f"{len(set(classes))} classes", flush=True)

    emb_path = f"{out}/embeddings.pt"
    if os.path.exists(emb_path) and not args.refresh:
        d = torch.load(emb_path, weights_only=True)
        clip = d["clip"].numpy(); mets = d["metrics"].numpy()
        assert len(d["paths"]) == len(paths), "cache stale; rerun with --refresh"
    else:
        model, proc = load_clip()
        imgs = [Image.open(p).convert("RGB") for p in paths]
        clip = clip_image_features(model, proc, imgs).numpy()
        mets = np.array([[image_metrics(im)[k] for k in METRICS] for im in imgs])
        torch.save({"clip": torch.tensor(clip), "metrics": torch.tensor(mets),
                    "paths": paths}, emb_path)
    # z-score metrics, concat a small weight onto CLIP for the combined space
    mz = (mets - mets.mean(0)) / (mets.std(0) + 1e-8)

    report = {"n_images": len(paths), "families": sorted(set(families)),
              "classes": sorted(set(classes))}
    for space_name, X in (("clip", clip), ("clip+metrics", np.concatenate(
            [clip, 0.5 * mz / np.sqrt(mz.shape[1])], axis=1))):
        _, pur_fam, sil_fam = cluster_and_score(X, families)
        _, pur_cls, sil_cls = cluster_and_score(X, classes)
        report[space_name] = {
            "kmeans_purity_vs_manipulation": pur_fam,
            "kmeans_purity_vs_class": pur_cls,
            "silhouette_vs_manipulation": sil_fam,
            "silhouette_vs_class": sil_cls,
            "nn_consistency_manipulation": nn_consistency(X, families),
            "nn_consistency_class": nn_consistency(X, classes),
        }
        print(f"[e15] [{space_name}] purity manip={pur_fam:.3f} class={pur_cls:.3f} | "
              f"NN manip={report[space_name]['nn_consistency_manipulation']:.3f} "
              f"class={report[space_name]['nn_consistency_class']:.3f}", flush=True)

    # distance of each manipulation family's centroid to the 'orig' centroid
    fam_arr = np.array(families)
    orig_c = clip[fam_arr == "orig"].mean(0)
    dist = {f: float(np.linalg.norm(clip[fam_arr == f].mean(0) - orig_c))
            for f in sorted(set(families))}
    report["centroid_dist_to_orig"] = dict(sorted(dist.items(), key=lambda kv: kv[1]))
    print("[e15] dist-to-orig (CLIP centroid): " +
          ", ".join(f"{k}={v:.2f}" for k, v in report["centroid_dist_to_orig"].items()),
          flush=True)

    P = _pca2(clip)
    project_plot(P, families, "E15 CLIP projection by manipulation",
                 f"{out}/plots/proj_by_manipulation.png")
    project_plot(P, classes, "E15 CLIP projection by image class",
                 f"{out}/plots/proj_by_class.png")

    with open(f"{out}/report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[e15] report -> {out}/report.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="recompute CLIP embeddings")
    main(ap.parse_args())
