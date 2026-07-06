"""Baseline policies sharing the same sampler + compute accounting as HorizonCache.

All obey the deck fair-comparison rules: same sampler, first node always fresh, achieved
budget measured identically. SeaCache here is HorizonCache-v0 with jumps disabled, i.e.
the *same code path* with tau_jump off — fair by identity (deck slide "One score, two
thresholds": tau_jump=0 recovers SeaCache exactly)."""
from __future__ import annotations

import random


class FullPolicy:
    """No caching — every node is a fresh full forward."""
    name = "full"

    def act(self, feat, step_index, num_steps):
        return "fresh"


class UniformEveryK:
    """Refresh every k nodes (deck: trivial floor, no signal)."""

    def __init__(self, k: int):
        self.k = k
        self.name = f"uniform_k{k}"

    def act(self, feat, step_index, num_steps):
        if step_index == 0 or step_index >= num_steps - 1:
            return "fresh"
        return "fresh" if (step_index % self.k == 0) else "cache"


class RandomK:
    """Refresh k random nodes (deck: worst — placement matters)."""

    def __init__(self, k_fresh: int, num_steps: int, seed: int = 0):
        rng = random.Random(seed)
        pool = list(range(1, num_steps - 1))
        rng.shuffle(pool)
        self.fresh = set(pool[: max(0, k_fresh - 2)]) | {0, num_steps - 1}
        self.name = f"random_k{k_fresh}"

    def act(self, feat, step_index, num_steps):
        return "fresh" if step_index in self.fresh else "cache"


class SeaCachePolicy:
    """SeaCache gate: accumulate filtered rel-L1, refresh at tau. Implemented as v0 with
    jumps disabled so it shares HorizonCache's exact fresh/cache boundary."""

    def __init__(self, tau_cache: float):
        from .policy import HorizonCacheV0, HorizonV0Config
        self.inner = HorizonCacheV0(HorizonV0Config(tau_cache=tau_cache, enable_jump=False))
        self.name = f"seacache_t{tau_cache:g}"

    def act(self, feat, step_index, num_steps):
        return self.inner.act(feat, step_index, num_steps)
