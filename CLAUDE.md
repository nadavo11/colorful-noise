# colorful-noise — project instructions

## Logging experiments
An experiment (E-number) isn't "done" until it's registered in the **experiment roadmap** —
not just commits, auto-memory, and `results/`. When an experiment reaches a verdict:
1. Add/update its entry in `experiments/roadmap_registry.py` (fields incl. status
   active/mapped/paused/dead-end/pending/done; pick a `thread` from `THREADS`).
2. Add a `## E<n> — Title (model)` section to `experiments/EXPERIMENTS.md`
   (Method / Key result / Verdict / Artifacts).
3. Add `docs/experiment-reports/EXPERIMENT_<n>.md` for headline experiments.
4. Regenerate the site: `python experiments/make_roadmap.py` (writes `docs/roadmap/*.html`).
5. Commit (message = probe id + verdict).

Do this as part of finishing — don't wait to be asked. If a script lives only on an unmerged
worktree, set `script: None` in the registry and note the worktree in the method text.
