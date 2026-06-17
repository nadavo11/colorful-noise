# Explainer rewrite standard (+ a reusable prompt)

Most experiment reports (the per-experiment `results/eNN/index.html` and the top-level
`EXPERIMENT_NN.md`) were written results-first: a wall of image strips, then bare metric
tables, with the variant names (`notch_lo`, `mag_only`, `band_swap`, …) never defined. That is
unreadable to anyone who didn't write the code.

**The standard** (matches the `experiment-documentation-standard` memory; canonical examples:
`experiments/e29_site.py` + `EXPERIMENT_29.md`, and now `e30_text_freq_control.py` /
`e31_flowedit_freq.py` + `EXPERIMENT_30.md` / `EXPERIMENT_31.md`):

1. **TL;DR** — one paragraph in plain words: what we ran and the headline result.
2. **Background (plain language)** — a glossary (`<dl>` in HTML, bullet list in `.md`) that
   **defines every term used below**: the transform/mechanism, every *variant/condition* name,
   and every *metric* (with which direction is "good", ↑/↓).
3. **Method** — what each `--part` / condition runs and the *question* it answers.
4. **Results** — one subsection per part/scene, each in this order:
   **(a) the figure first → (b) a one-line "What to look for" caption → (c) interpretation
   (what's good, what's bad) → (d) the numbers** (small table, best cell per column
   highlighted). Group figures *by part* — never dump every strip under one heading.
5. **Caveats & next** — limitations (single seed, deferred metrics, …) and where it goes.
6. **Reproduce** — the cluster + local commands, including the model-free `--part site` rebuild.

**Mechanics learned on E30/E31 (reuse these):**
- Build the HTML in the experiment's own `_site()` (or a `eNN_site.py`); embed images as base64
  via `from e27_site import data_uri` so the page is one portable file.
- Add a **model-free `--part site`** that loads `results/eNN/report.json` and re-templates the
  page **without loading any model or re-scoring** (all numbers already live in `report.json`).
  This makes the page rebuildable anywhere and is how you regenerate after editing the explainer.
- Reuse the E29/E30 CSS (`.tldr`, `.look`, `.read`, `.win`, `.cav`, glossary `dl`, `td.pos`).
- Results live on `/storage` (gitignored). To rebuild locally: `kubectl cp` the
  `report.json` + the `strip.png`/`*_sweep.png` files (not the heavy per-variant PNGs) from
  `mystorage-0-0:/storage/malnick/colorful-noise/experiments/results/eNN` into the local
  `experiments/results/eNN`, then run `python experiments/<driver> --part site`.

---

## The reusable prompt (paste this, fill in `<NN>`)

> Rewrite experiment **E\<NN\>**'s explainer — both `EXPERIMENT_<NN>.md` and the HTML report
> generator (`experiments/e<NN>_*.py`'s `_site()`, or `e<NN>_site.py`) — to the project's
> explainer standard in `experiments/EXPLAINER_STANDARD.md`. Use `experiments/e29_site.py` +
> `EXPERIMENT_29.md` (and the E30/E31 versions) as the template.
>
> First read the driver `experiments/e<NN>_*.py` and `EXPERIMENT_<NN>.md` and extract: every
> `--part`, every variant/condition name and **how it is computed** (which op, which params),
> the band/transform structure, and every metric. Then rewrite both files so they:
>
> 1. Open with a one-paragraph **TL;DR** (what we ran + headline result, plain words).
> 2. Have a **Background (plain language)** glossary that **defines every term** — the
>    mechanism, each variant/condition, and each metric with its ↑/↓ "good" direction. A reader
>    with no project context must be able to follow it.
> 3. Have a **Method** section: what each part/condition runs and the question it answers.
> 4. Present **Results** one subsection per part/scene, each as: **figure first → a one-line
>    "What to look for" caption → interpretation (what's good/bad) → the numbers** (small table,
>    highlight the best cell per column). **Group figures by part** (fix any code that dumps all
>    strips under one heading).
> 5. End with **Caveats & next** and **Reproduce**.
>
> In the generator, embed images as base64 (`from e27_site import data_uri`), reuse the E30 CSS
> (`.tldr/.look/.read/.cav`, glossary `dl`, `td.pos` highlight), and **add a model-free
> `--part site`** that rebuilds `index.html` from `results/e<NN>/report.json` + cached strips
> with no model load. The `.md` and the HTML must tell the same story and define the same terms.
>
> Then regenerate: `kubectl cp` the `report.json` + `strip.png`/`*_sweep.png` for E\<NN\> from
> `mystorage-0-0:/storage/malnick/colorful-noise/experiments/results/e<NN>` into local
> `experiments/results/e<NN>`, run `python experiments/<driver> --part site`, and open the page
> to verify: every variant term is defined, each result leads with its figure + "what to look
> for" before the table, figures are grouped by part, best cells highlighted. Numbers in the
> page must match `report.json` (the rebuild only re-templates). Commit the code + `.md` (results
> stay gitignored) and push.
