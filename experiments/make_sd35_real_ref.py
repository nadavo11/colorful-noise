"""Build the SD3.5-VAE real-image latent reference for spectral-distance-to-real.

The E10 real reference (results/e10/real_latents.pt) is in FLUX VAE latent space,
so it's not comparable to SD3.5 latents. This re-fetches the 20 seeded picsum
photos (reproducible, same as E10) if absent, encodes them through the SD3.5 VAE
(load_sd35_vae / sd3_vae_encode), and saves results/e10/sd35_real_latents.pt.
Run once on the cluster:  python make_sd35_real_ref.py
"""
import os
import sys
import urllib.request

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS
from e17_sd35 import load_sd35_vae, sd3_vae_encode, SIZE

OUT = os.path.join(RESULTS, "e10")
RDIR = os.path.join(OUT, "real_photos")


def fetch_photos(n=20):
    os.makedirs(RDIR, exist_ok=True)
    for i in range(n):
        p = f"{RDIR}/photo_{i:03d}.jpg"
        if os.path.exists(p):
            continue
        url = f"https://picsum.photos/seed/e10-{i}/{SIZE}/{SIZE}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "e10/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                open(p, "wb").write(r.read())
            print(f"[sd35-ref] fetched {p}", flush=True)
        except Exception as e:
            print(f"[sd35-ref] FAILED {url}: {e}", flush=True)
    return sorted(p for p in os.listdir(RDIR) if p.endswith(".jpg"))


def main():
    photos = fetch_photos()
    assert photos, f"no photos in {RDIR}"
    vae = load_sd35_vae()
    lats = []
    for p in photos:
        lat = sd3_vae_encode(vae, Image.open(os.path.join(RDIR, p)))
        lats.append(lat)
        print(f"[sd35-ref] encoded {p} (std={lat.std():.3f})", flush=True)
    real = torch.cat(lats, 0)  # (N,16,128,128)
    out = os.path.join(OUT, "sd35_real_latents.pt")
    torch.save(real, out)
    print(f"[sd35-ref] saved {real.shape} -> {out}", flush=True)


if __name__ == "__main__":
    main()
