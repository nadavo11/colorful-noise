"""E47 n=100 analysis: paired bootstrap of the frontier margin + frontier plot."""
import json, numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/storage/malnick/colorful-noise/experiments/results"
np.random.seed(0)

def load(tag):
    rows = json.load(open(f"{R}/{tag}/scores.json"))
    by_arm = defaultdict(dict)  # arm -> {img_key: (struct, clip)}
    for r in rows:
        by_arm[r["arm"]][r["key"]] = (r["struct_dist"], r["clip_dir"])
    return by_arm

A = load("e47_confA")
S = load("e47_confSDG")

VAN = ["van_s0.5", "van_s0.6", "van_s0.7", "van_s0.8", "van_s0.9"]

def means(by_arm, arm, keys):
    v = np.array([by_arm[arm][k] for k in keys])
    return v[:, 0].mean(), v[:, 1].mean()

def frontier_interp(van_pts, struct):
    """vanilla clip at a given struct via linear interp along the sorted frontier."""
    xs = np.array([p[0] for p in van_pts]); ys = np.array([p[1] for p in van_pts])
    o = np.argsort(xs); xs, ys = xs[o], ys[o]
    return float(np.interp(struct, xs, ys))

def margin(by_arm, arm, keys):
    """arm clip - vanilla-frontier clip at the arm's struct (>0 = NW of frontier)."""
    van = [means(by_arm, v, keys) for v in VAN]
    s, c = means(by_arm, arm, keys)
    return c - frontier_interp(van, s), s, c

def boot(by_arm, arm, n=4000):
    keys = list(by_arm[arm].keys())
    base, s0, c0 = margin(by_arm, arm, keys)
    ms = []
    for _ in range(n):
        bk = list(np.random.choice(keys, len(keys), replace=True))
        ms.append(margin(by_arm, arm, bk)[0])
    ms = np.array(ms)
    lo, hi = np.percentile(ms, [2.5, 97.5])
    return base, s0, c0, lo, hi, float((ms > 0).mean())

print(f"{'arm':18} {'struct':>7} {'clip':>8} {'margin':>8} {'95% CI':>18} {'P(>0)':>6}")
for tag, by_arm, arms in [("A", A, ["A_t0.125", "A_t0.25", "A_t0.375"]),
                          ("SDG", S, ["sdg_src_t0.125", "sdg_src_t0.25", "sdg_src_t0.375"])]:
    for arm in arms:
        b, s, c, lo, hi, p = boot(by_arm, arm)
        print(f"{arm:18} {s:7.4f} {c:+8.4f} {b:+8.4f}  [{lo:+.4f},{hi:+.4f}]  {p:5.2f}")

# ---- frontier plot ----
keysA = list(A["A_t0.25"].keys())
vanA = [means(A, v, keysA) for v in VAN]
fig, ax = plt.subplots(figsize=(6, 5))
vx = [p[0] for p in vanA]; vy = [p[1] for p in vanA]
ax.plot(vx, vy, "-o", color="gray", label="vanilla SDEdit frontier")
for v, (x, y) in zip(VAN, vanA):
    ax.annotate(v.replace("van_", ""), (x, y), fontsize=7, color="gray")
for arm, by, col in [("A_t0.125", A, "C0"), ("A_t0.25", A, "C0"),
                     ("sdg_src_t0.125", S, "C1")]:
    s, c = means(by, arm, list(by[arm].keys()))
    ax.scatter([s], [c], color=col, zorder=5)
    ax.annotate(arm, (s, c), fontsize=7, color=col)
ax.set_xlabel("DINO structure distance  (lower = better)")
ax.set_ylabel("CLIP-directional  (higher = better)")
ax.set_title("E47 geodesic phase-perturbed SDEdit vs vanilla (PIE-Bench n=100)")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
out = "/home/shimon/research/colorful-noise/.claude/worktrees/e47-register/docs/experiment-reports/e47_frontier.png"
fig.savefig(out, dpi=120)
print("saved", out)
