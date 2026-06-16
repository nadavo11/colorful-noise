# Research roadmap (generated site)

An interactive, lightweight HTML overview of the colorful-noise research — the
**vectors** (threads), every experiment E0–E31, what each found, whether the
direction is alive or a dead end, and how to proceed.

**Open it:** `docs/roadmap/index.html` in a browser (no server needed).

Pages: `index.html` (thesis + SVG thread-map + legend), `thread-*.html` (one per
research thread, with narrative + "how to proceed" + experiment cards),
`experiments.html` (filterable table of all experiments), `glossary.html`.

## Maintaining it

The whole site is generated from one registry — **do not edit the HTML by hand**.

1. Edit `experiments/roadmap_registry.py` — append an entry to `EXPERIMENTS` (and a
   new `THREADS` entry if you open a new line of work). Field conventions are
   documented at the top of that file.
2. Regenerate:

   ```bash
   python experiments/make_roadmap.py
   ```

Deep per-experiment writeups live in the root `EXPERIMENT_*.md` files; the
chronological log is `experiments/EXPERIMENTS.md`. The roadmap links out to both.

Figures stay light: the site references images from `experiments/results/…` by
relative path (set an experiment's optional `image` field), it does **not**
base64-embed them. Missing result figures/links degrade gracefully, so the site
builds even though `experiments/results/` is gitignored. Honors `CN_RESULTS`.
