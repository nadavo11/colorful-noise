# Handoff prompt — backfill one experiment's roadmap report (parallel-safe)

This is the per-agent prompt for backfilling the roadmap with **real explanations + results +
illustrations**, one sub-agent per experiment, runnable in parallel. Substitute `{EID}` (e.g. `E23`)
and `{N}` (e.g. `23`). The E47 report (`docs/experiment-reports/EXPERIMENT_47.md`) is the gold-standard
template for depth and figure use.

---

## Your task (experiment {EID})

You own **exactly one** experiment, **{EID}**. Produce a deep, self-contained writeup with its actual
method explanation, results, and embedded illustrations, then push it to `origin/main`. Work in an
isolated git worktree. **Touch only {EID}'s own files** (see "Conflict rules") so dozens of agents can
run in parallel without colliding.

### 1. Understand the experiment
Read, in this order:
- Its registry entry: `experiments/roadmap_registry.py` (find the `"id": "{EID}"` dict — motivation /
  method / result / verdict / nxt / thread).
- `experiments/EXPERIMENTS.md` → the `## {EID}` section (if present).
- Its probe log: `experiments/e{N}_log.md` or `experiments/e{N}_report.md` (if present).
- Its driver(s): `experiments/e{N}_*.py` — read the code to state the **actual operation/math**, not a
  paraphrase. (Cross-cutting math may live in `docs/methods/`.)
- Any existing `docs/experiment-reports/EXPERIMENT_{N}.md`.

### 2. Locate the results (they are scattered — check ALL of these)
- Local: `experiments/results/e{N}*/` (gitignored, may exist on this machine).
- Cluster archive: `/storage/malnick/colorful-noise/experiments/results/e{N}*/` and
  `/storage/malnick/colorful-noise/roadmap_results/{EID}/`.
- **Unmerged worktrees** (some results/scripts live only here):
  `/home/shimon/research/colorful-noise/.claude/worktrees/*/experiments/results/e{N}*/`.
- If `/storage` is not mounted, remount it (see Gotchas). If results genuinely cannot be found,
  say so explicitly in your report and write the report from the log/registry numbers.
**In the report, state where {EID}'s results live** (local / which worktree / /storage path).

### 3. Write `docs/experiment-reports/EXPERIMENT_{N}.md`
Mirror the E47 report's structure and depth. Required sections:
- **Motivation** — the question it asks, in plain terms.
- **Method** — what was actually done: the operation/transform, the key equation(s), the knobs, the
  metric. Explain *why* it should work. This is the part the user feels is missing — be substantive.
- **Results** — the measured numbers and what they show; embed the figures (next step).
- **Verdict** — the one-line takeaway (KEEP/KILL/MAPPED + why), consistent with the registry.
- **Artifacts** — driver, log, results location, figures.
Keep it honest and specific (real numbers, real caveats). Use GitHub-flavoured markdown (tables,
code, `![caption](path)` images).

### 4. Illustrations (this is explicitly wanted)
- Embed the **result figures** that illustrate the method's effect (grids, plots). Find the source
  images from step 2.
- Use the shared helper WITHOUT editing it: in python,
  `import sys; sys.path.insert(0,"experiments"); from aggregate_results import aggregate;
   aggregate("{EID}", [(src_abs_path, "short_name"), ...])`.
  It copies full-res to `/storage/.../roadmap_results/{EID}/` and writes light JPEGs to
  `docs/experiment-reports/figs/{EID}/`. Then reference them as `![caption](figs/{EID}/short_name.jpg)`.
- If a **method diagram** would clarify the idea and you can produce it cheaply with matplotlib
  (e.g. a before/after spectrum, a band mask, an operator schematic), generate it, save under
  `docs/experiment-reports/figs/{EID}/`, and embed it. Otherwise use representative result images.
- Keep each committed JPEG light (the helper downscales to ≤1600px); the roadmap renderer inlines
  them, so the page stays self-contained.

### 5. Conflict rules (CRITICAL — parallel safety)
Commit **only** these paths:
- `docs/experiment-reports/EXPERIMENT_{N}.md`
- `docs/experiment-reports/figs/{EID}/**`

Do **NOT**:
- edit `experiments/roadmap_registry.py`, `experiments/aggregate_results.py`,
  `experiments/make_roadmap.py`, `experiments/EXPERIMENTS.md`, or any other experiment's files;
- run `experiments/make_roadmap.py` or touch `docs/roadmap/**` (the site regen is a single
  centralized pass run after all agents finish — not your job);
- modify shared analysis scripts.
(The registry `doc` field and the site regeneration are handled centrally afterward.)

### 6. Commit & push
- Work in an isolated worktree off `origin/main`.
- `git add docs/experiment-reports/EXPERIMENT_{N}.md docs/experiment-reports/figs/{EID}` only.
- Commit: `"{EID}: backfill roadmap report + figures"` (end with the repo's Co-Authored-By /
  Claude-Session trailers).
- Push to **main**: `git fetch origin && git rebase origin/main && git push origin HEAD:main`.
  On a non-fast-forward race, re-`fetch`/`rebase`/`push` (your files are disjoint from other agents',
  so rebases are clean). Never force-push.

### 7. Report back (your final message)
- One-line statement of {EID}'s method and verdict.
- Where the results were found (local / worktree name / /storage path), or "results not found".
- Which figures you embedded (and whether any are generated diagrams).
- Anything missing or uncertain that the centralized pass should know.

## Gotchas
- **RunAI / /storage:** if `/storage` reads fail, remount: ensure `shimon-myssh` workspace is Running
  (`/home/shimon/.runai/bin/2.116.6/runai workspace list | grep shimon-myssh`), then
  `screen -dmS storage /home/shimon/.runai/bin/2.116.6/runai workspace port-forward shimon-myssh --port 2222:22`
  and `echo root | sshfs -o cache=yes -o password_stdin -o reconnect -p 2222 root@127.0.0.1:/storage /storage`.
- **Heavy grids** (some are 100+ MB / tens of thousands of px tall): `aggregate()` downscales the web
  copy automatically; still archive the full-res to /storage. For a giant per-image grid, a crop or a
  representative few-row excerpt reads better than the whole sheet — your call.
- **Use system python** (`python`, conda env `image-gs`) for figure work; PIL is available.
- Match the surrounding report style; don't invent results — pull real numbers from the log/registry.

## Centralized pass (run by the coordinator AFTER all agents finish — not the agent)
1. For any experiment whose registry `doc` field isn't `docs/experiment-reports/EXPERIMENT_<n>.md`,
   set it (single edit to `roadmap_registry.py`).
2. `python experiments/make_roadmap.py && python experiments/make_roadmap.py --check` (must exit 0).
3. Commit the regenerated `docs/roadmap/**` + registry doc-field edits; push to `main`.
4. `git -C /home/shimon/research/colorful-noise pull --ff-only origin main` to update the local checkout.
