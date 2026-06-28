"""E52 token-autopsy evaluation (anaconda env). Scores every intervention image against the
full-compute reference (the unmodified edit) and the source image, reusing the E51 metrics.

  edit strength : clipT_gain, clip_dir, clipT_target      (toward the target prompt)
  preservation  : lpips_to_ref, psnr_to_ref, dino_to_src  (how far the edit drifted / structure)
Reads token_autopsy/token_generation.jsonl -> writes token_autopsy/token_results.jsonl.
"""
from __future__ import annotations
import json
from PIL import Image

import config as C
import metrics_q as MQ


def _img(rel):
    return Image.open(C.REPO / rel).convert("RGB")


def main():
    gen_path = C.TOK / "token_generation.jsonl"
    if not gen_path.exists():
        print("[token-eval] no token_generation.jsonl (run token_autopsy.py first)"); return
    gen = [json.loads(l) for l in gen_path.read_text().splitlines() if l.strip()]
    out_path = C.TOK / "token_results.jsonl"
    f = open(out_path, "w")
    for i, r in enumerate(gen):
        ref = _img(r["reference"]); img = _img(r["image"])
        src = _img(r["source_image"])
        q = MQ.quality(img, ref, src, r["source_prompt"], r["target_prompt"])
        keep = ("id", "task_type", "category", "role", "token_index", "token_word",
                "mechanism", "weight", "delta_band_low", "delta_band_mid", "delta_band_high",
                "delta_delta_smoothness", "delta_delta_norm")
        row = {k: r[k] for k in keep if k in r}
        row.update(q)
        f.write(json.dumps(row) + "\n"); f.flush()
        if (i + 1) % 40 == 0:
            print(f"[token-eval] {i+1}/{len(gen)}")
    f.close()
    print(f"[token-eval] wrote {len(gen)} rows -> {out_path}")


if __name__ == "__main__":
    main()
