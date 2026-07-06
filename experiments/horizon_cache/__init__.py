"""HorizonCache (E56) — causal safe-horizon prediction for flow-model acceleration.

Builds directly on the SeaCache gate (deck: `presentation/`, harness:
`experiments/flux_seacache_dp_shortcuts.py`). SeaCache reads one cheap score off the
modulated input `h` and decides refresh-vs-cache. HorizonCache reuses the *same* score
but expands the action set to {fresh, cache, jump_1.25, jump_1.5, jump_2.0}, predicting
how far the current computation stays trustworthy (the "safe integration horizon").

Two policies:
  * HorizonCache-v0 — a deterministic causal rule with a sweepable threshold config;
    reduces to SeaCache exactly when jumps are disabled (fair-by-identity).
  * HorizonCache-v1 — a small tabular model (sklearn HistGradientBoosting) trained on
    rollout safe-horizon labels with an asymmetric cost (false-jump >> false-cache).

Prior that shapes the hypothesis (E53): SeaCache's rel-L1 ranks the *oracle* safe-jump
length only weakly (Spearman -0.50). HorizonCache tests whether a richer causal feature
set + conservative safety gating recovers a deployable win at matched achieved budget.
"""
