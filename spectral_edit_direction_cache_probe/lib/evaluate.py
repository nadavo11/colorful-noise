"""E51 evaluation (anaconda env, torch>=2.6 so .bin metric checkpoints + LPIPS load).
Reads the generation manifest, scores every saved image against its full-compute reference,
and writes diagnostics/results.jsonl. Quality metrics:
  * fidelity to reference : dino_to_ref, clipI_to_ref, lpips_to_ref, psnr_to_ref
  * edit correctness      : clipT_target, clipT_gain, clip_dir
  * structure preservation: dino_to_src
"""
from __future__ import annotations
import json
from pathlib import Path
from PIL import Image

import config as C
import metrics_q as MQ


def _img(rel):
    return Image.open(C.REPO / rel).convert("RGB")


def main():
    gen = [json.loads(l) for l in (C.DIAG / "generation.jsonl").read_text().splitlines() if l.strip()]
    out_path = C.DIAG / "results.jsonl"
    f = open(out_path, "w")
    for i, r in enumerate(gen):
        ref = _img(r["reference"]); img = _img(r["image"]); src = _img(r["source_image"])
        q = MQ.quality(img, ref, src, r["source_prompt"], r["target_prompt"])
        row = {k: r[k] for k in ("id", "task_type", "category", "scope", "variant",
                                 "skip_ratio", "realized_skip", "fwd_edit", "fwd_src", "is_primary")}
        row["n_steps"] = r.get("n_steps", 0) or (r["fwd_edit"] if r["variant"] == "full_compute_reference" else 0)
        row.update(q)
        f.write(json.dumps(row) + "\n"); f.flush()
        if (i + 1) % 20 == 0:
            print(f"[eval] {i+1}/{len(gen)}")
    f.close()
    print(f"[eval] wrote {len(gen)} rows -> {out_path}")


if __name__ == "__main__":
    main()
