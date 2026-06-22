# colorful-noise — project instructions

## Logging experiments
An experiment (E-number) isn't "done" until it's registered in the **experiment roadmap** —
not just commits, auto-memory, and `results/`. When an experiment reaches a verdict:
1. Add/update its entry in `experiments/roadmap_registry.py` (fields incl. status
   active/mapped/paused/dead-end/pending/done; pick a `thread` from `THREADS`). The `doc`
   field stores the repo-relative path (`docs/experiment-reports/EXPERIMENT_<n>.md`), not a
   bare filename.
2. Add a `## E<n> — Title (model)` section to `experiments/EXPERIMENTS.md`
   (Method / Key result / Verdict / Artifacts).
3. Add `docs/experiment-reports/EXPERIMENT_<n>.md` for headline experiments.
4. Write a run manifest: `from manifest import write_manifest` then
   `write_manifest("E<n>", config=vars(args), metrics={...}, artifacts=[...])` — or hand-create
   `experiments/manifests/E<n>.json`. Manifests are tracked and hold only the mechanical record
   (config / metrics / artifact paths / commit / date); the narrative stays in the registry.
5. Regenerate + verify no drift:
   `python experiments/make_roadmap.py && python experiments/make_roadmap.py --check`
   (`--check` must exit 0 — it flags drivers with no registry entry, stale script paths, etc.).
6. Commit (message = probe id + verdict).

Do this as part of finishing — don't wait to be asked. If a script lives only on an unmerged
worktree, set `script: None` in the registry and note the worktree in the method text.

## Repo layout & conventions
- The roadmap site (`docs/roadmap/`, generated from `roadmap_registry.py` via
  `make_roadmap.py`) is the **only** experiment site. Don't add per-experiment HTML generators
  — the old `e*_site.py` were retired; the shared `data_uri` helper now lives in `common.py`.
- Deep per-experiment writeups: `docs/experiment-reports/EXPERIMENT_<n>.md`.
- Cross-cutting method/math notes (not tied to one experiment): `docs/methods/`.
- Run records: `experiments/manifests/<eid>.json` (tracked, light). Heavy run artifacts stay in
  `results/` (gitignored) and are referenced from manifests by relative path.
