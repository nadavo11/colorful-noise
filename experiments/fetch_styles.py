"""Download a small bank of public-domain paintings (varied styles) for the E18
cross-domain style-transfer rerun. Saves 1024px JPEGs into results/e18/styles/
(gitignored). Uses Wikimedia's Special:FilePath (redirects to the current file,
?width= server-side thumbnail) so the links survive filename churn. Skip-if-present.

    python fetch_styles.py
"""
import os
import sys
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS

OUT = os.path.join(RESULTS, "e18", "styles")
UA = "colorful-noise-research/0.1 (academic; contact via repo)"

# name -> exact Wikimedia Commons file (all public domain), varied styles:
# post-impressionist, ukiyo-e, impressionist, baroque, expressionist, renaissance.
FILES = {
    "vangogh_starry_night": "Van Gogh - Starry Night - Google Art Project.jpg",
    "hokusai_great_wave": "Tsunami by hokusai 19th century.jpg",
    "monet_impression": "Claude Monet, Impression, soleil levant.jpg",
    "vermeer_pearl": "1665 Girl with a Pearl Earring.jpg",
    "munch_scream": "The Scream.jpg",
    "bruegel_babel": "Pieter Bruegel the Elder - The Tower of Babel (Vienna) - Google Art Project - edited.jpg",
}
BASE = "https://commons.wikimedia.org/wiki/Special:FilePath/{}?width=1024"


def fetch(name, fname):
    dest = os.path.join(OUT, f"{name}.jpg")
    if os.path.exists(dest):
        print(f"[styles] {name} (cached)", flush=True)
        return True
    url = BASE.format(urllib.parse.quote(fname))
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        with open(dest, "wb") as f:
            f.write(data)
        from PIL import Image
        Image.open(dest).convert("RGB").verify()       # validate it decodes
        print(f"[styles] {name}: {len(data) // 1024} KB", flush=True)
        return True
    except Exception as e:
        if os.path.exists(dest):
            os.remove(dest)
        print(f"[styles] {name} FAILED: {e}", flush=True)
        return False


def main():
    os.makedirs(OUT, exist_ok=True)
    ok = sum(fetch(n, f) for n, f in FILES.items())
    print(f"[styles] {ok}/{len(FILES)} into {OUT}", flush=True)
    if ok < len(FILES):
        print("[styles] some failed (network/filename) -- rerun, or drop your own "
              "JPEGs into that dir and point e18 --styles at it.", flush=True)


if __name__ == "__main__":
    main()
