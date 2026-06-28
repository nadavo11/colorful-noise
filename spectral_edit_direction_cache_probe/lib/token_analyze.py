"""E52 token-autopsy aggregation (anaconda env). Turns the per-example observational records
+ intervention results + the E51 cache metrics into:

  outputs/.../token_autopsy/token_summary.json          headline numbers + decision + verdict
  outputs/.../token_autopsy/per_token_observation.csv   per (example, token) attention/contrib
  outputs/.../token_autopsy/token_intervention_metrics.csv  per (example,role,mech,weight)

Sections answered: B (observational), C/D (intervention), E (cache connection), G (verdict).
Everything is guarded so a partial / smoke run still produces a summary.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd

import config as C


# ----------------------------------------------------------------- loaders
def _obs_files():
    return sorted(C.DIAG.glob("token_obs_*.json"))


def _load_obs(fp):
    d = json.loads(fp.read_text())
    blocks = d["obs"]
    n_tok = d["n_tokens"]
    steps = sorted({int(s) for b in blocks.values() for s in b})
    # mean over tapped blocks -> [n_steps, n_tokens] for each stat
    def stack(stat):
        arr = np.full((len(steps), n_tok), np.nan)
        for si, s in enumerate(steps):
            vals = [blocks[b][str(s)][stat] for b in blocks if str(s) in blocks[b]]
            if vals:
                arr[si] = np.mean(np.array(vals, dtype=float), axis=0)
        return arr
    d["_steps"] = steps
    d["_mass"] = stack("mass"); d["_contrib"] = stack("contrib")
    d["_maxq"] = stack("maxq"); d["_valnorm"] = stack("valnorm")
    # per-block contrib for the token x layer heatmap (edit tokens)
    d["_block_contrib"] = {int(b): np.nanmean(np.array(
        [steps_d[s]["contrib"] for s in steps_d], dtype=float), axis=0)
        for b, steps_d in blocks.items()}
    return d


def _edit_idx(d):
    al = json.loads((C.DIAG / f"token_align_{d['id']}.json").read_text())["align"]
    return al["changed"] + al["inserted"]


# ----------------------------------------------------------------- B: observational
def observational(obs):
    rows = []
    per_ex = {}
    for d in obs:
        eid = d["id"]; toks = d["tokens"]; n = len(toks)
        mass = np.nanmean(d["_mass"], axis=0)            # mean over steps -> [n_tok]
        contrib = np.nanmean(d["_contrib"], axis=0)
        edit_idx = [j for j in _edit_idx(d) if j < n]
        non_edit = [j for j in range(n) if j not in edit_idx]
        for j in range(n):
            rows.append(dict(id=eid, task_type=d["task_type"], token_index=j, token=toks[j],
                             is_edit=int(j in edit_idx), mass=float(mass[j]),
                             contrib=float(contrib[j])))
        # which timestep / layer do edit tokens matter most?
        if edit_idx:
            edit_contrib_t = np.nansum(d["_contrib"][:, edit_idx], axis=1)   # [n_steps]
            peak_step = int(d["_steps"][int(np.nanargmax(edit_contrib_t))]) if edit_contrib_t.size else -1
            bc = {b: float(np.nansum(v[edit_idx])) for b, v in d["_block_contrib"].items()}
            peak_block = int(max(bc, key=bc.get)) if bc else -1
            edit_mass = float(np.nanmean(mass[edit_idx]))
        else:
            peak_step = peak_block = -1; edit_mass = float("nan")
        non_mass = float(np.nanmean(mass[non_edit])) if non_edit else float("nan")
        # role dominance + frequency coupling from ablation
        abl = d.get("ablation", {})
        role_norm, role_band = {}, {}
        for r, steps_d in abl.items():
            if not steps_d:
                continue
            norms = [v["norm"] for v in steps_d.values()]
            lo = np.mean([v["low"] for v in steps_d.values()])
            md = np.mean([v["mid"] for v in steps_d.values()])
            hi = np.mean([v["high"] for v in steps_d.values()])
            role_norm[r] = float(np.mean(norms))
            role_band[r] = dict(low=float(lo), mid=float(md), high=float(hi),
                                dominant=["low", "mid", "high"][int(np.argmax([lo, md, hi]))])
        dominant_role = max(role_norm, key=role_norm.get) if role_norm else None
        # token attention stability over steps (for cache correlation): CV of edit-token mass
        if edit_idx:
            traj = np.nanmean(d["_mass"][:, edit_idx], axis=1)
            stab = float(np.nanstd(traj) / (np.nanmean(traj) + 1e-9))     # lower = more stable
            p = traj / (np.nansum(traj) + 1e-9)
            entropy = float(-np.nansum(p * np.log(p + 1e-12)))
        else:
            stab = entropy = float("nan")
        per_ex[eid] = dict(id=eid, task_type=d["task_type"], category=d["category"],
                           edit_mass=edit_mass, non_edit_mass=non_mass,
                           edit_attention_ratio=float(edit_mass / (non_mass + 1e-9)),
                           peak_step=peak_step, peak_block=peak_block,
                           dominant_role=dominant_role, role_norm=role_norm, role_band=role_band,
                           attn_stability_cv=stab, attn_entropy=entropy,
                           delta_smoothness=d.get("spectral", {}).get("delta_smoothness"))
    return pd.DataFrame(rows), per_ex


# ----------------------------------------------------------------- C/D: interventions
def interventions():
    fp = C.TOK / "token_results.jsonl"
    if not fp.exists():
        return pd.DataFrame(), {}
    df = pd.DataFrame([json.loads(l) for l in fp.read_text().splitlines() if l.strip()])
    if df.empty:
        return df, {}
    # baseline (weight=1.0) per (id,role,mechanism) to measure RESPONSE relative to identity
    base = df[np.isclose(df["weight"], 1.0)].set_index(["id", "role", "mechanism"])
    summary = {}

    def slope(sub, ycol):
        x = sub["weight"].values.astype(float); y = sub[ycol].values.astype(float)
        if len(x) < 2 or np.allclose(x, x[0]):
            return float("nan")
        return float(np.polyfit(x, y, 1)[0])

    # per-mechanism weight curves (mean over examples/roles) + responsiveness
    mech_stats = {}
    for mech, sub in df.groupby("mechanism"):
        curve = sub.groupby("weight").agg(
            clipT_gain=("clipT_gain", "mean"), lpips_to_ref=("lpips_to_ref", "mean"),
            dino_to_src=("dino_to_src", "mean"),
            delta_smoothness=("delta_delta_smoothness", "mean"),
            band_low=("delta_band_low", "mean"), band_mid=("delta_band_mid", "mean"),
            band_high=("delta_band_high", "mean")).reset_index()
        mech_stats[mech] = dict(
            curve=curve.to_dict("records"),
            edit_response=slope(sub, "clipT_gain"),        # >0 => more weight, more edit
            preservation_cost=slope(sub, "lpips_to_ref"),  # >0 => more weight, more drift
            smoothness_response=slope(sub, "delta_delta_smoothness"))
    summary["mechanisms"] = mech_stats

    # per-role responsiveness (does amplifying the edited noun control the edit? control token shouldn't)
    role_resp = {}
    for role, sub in df.groupby("role"):
        role_resp[role] = dict(edit_response=slope(sub, "clipT_gain"),
                               preservation_cost=slope(sub, "lpips_to_ref"))
    summary["roles"] = role_resp

    # attention-space vs embedding-space: compare controllable edit range
    attn_mechs = ["attn_logit_bias", "attn_prob_reweight", "value_scale"]
    def ctrl_range(mechs):
        sub = df[df["mechanism"].isin(mechs)]
        if sub.empty:
            return float("nan")
        g = sub.groupby("weight")["clipT_gain"].mean()
        return float(g.max() - g.min())
    summary["attn_space_range"] = ctrl_range(attn_mechs)
    summary["embed_space_range"] = ctrl_range(["embed_scale"])
    summary["best_mechanism"] = max(
        mech_stats, key=lambda m: (mech_stats[m]["edit_response"] if not np.isnan(
            mech_stats[m]["edit_response"]) else -1e9)) if mech_stats else None
    return df, summary


# ----------------------------------------------------------------- E: cache connection
def cache_connection(per_ex):
    """Relate token-attention behaviour to the E51 spectral-delta-cache quality per example."""
    pf = C.OUT / "per_example_metrics.csv"
    if not pf.exists():
        return {}
    cm = pd.read_csv(pf)
    cm = cm[(cm["variant"] == "spectral_edit_delta_cache") & (cm["is_primary"] == True)]  # noqa: E712
    cm = cm.set_index("id")
    rows = []
    for eid, pe in per_ex.items():
        if eid not in cm.index:
            continue
        rows.append(dict(id=eid, attn_stability_cv=pe["attn_stability_cv"],
                         attn_entropy=pe["attn_entropy"], peak_step=pe["peak_step"],
                         delta_smoothness=pe["delta_smoothness"],
                         cache_lpips=float(cm.loc[eid, "lpips_to_ref"]),
                         cache_dino=float(cm.loc[eid, "dino_to_ref"])))
    if len(rows) < 3:
        return dict(n=len(rows), note="too few paired examples for correlation")
    d = pd.DataFrame(rows)

    def corr(a, b):
        x, y = d[a].values, d[b].values
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 3 or np.std(x[m]) < 1e-9 or np.std(y[m]) < 1e-9:
            return float("nan")
        return float(np.corrcoef(x[m], y[m])[0, 1])
    return dict(
        n=len(rows),
        # stable attention (low CV) should go with good cache (low lpips) => positive corr(CV, lpips)
        stability_vs_cache_lpips=corr("attn_stability_cv", "cache_lpips"),
        entropy_vs_cache_lpips=corr("attn_entropy", "cache_lpips"),
        peakstep_vs_smoothness=corr("peak_step", "delta_smoothness"),
        table=d.to_dict("records"))


# ----------------------------------------------------------------- G: verdict
def verdict(obs_df, per_ex, iv_sum, cache):
    ev = {}
    # 1. identifiable: do edit tokens attract more attention than non-edit tokens?
    if not obs_df.empty:
        em = obs_df[obs_df["is_edit"] == 1]["mass"].mean()
        nm = obs_df[obs_df["is_edit"] == 0]["mass"].mean()
        ev["edit_attention_ratio"] = float(em / (nm + 1e-9))
        identifiable = ev["edit_attention_ratio"] > 1.1
    else:
        identifiable = False
    # role dominance: noun/attribute should dominate over control
    dom = [pe["dominant_role"] for pe in per_ex.values() if pe["dominant_role"]]
    ev["dominant_roles"] = dom
    ev["edit_roles_dominate"] = bool(dom) and (sum(r in ("edited_noun", "attribute", "style")
                                                   for r in dom) >= max(1, len(dom) // 2))
    # 2/4. causal & controllable: any mechanism with clearly positive edit response
    best_resp = float("nan"); controllable = False
    if iv_sum.get("mechanisms"):
        resp = [m["edit_response"] for m in iv_sum["mechanisms"].values()
                if not np.isnan(m["edit_response"])]
        if resp:
            best_resp = float(max(resp)); controllable = best_resp > 1e-3
    ev["best_edit_response"] = best_resp
    # 3. attention vs embedding space
    ar, er = iv_sum.get("attn_space_range", float("nan")), iv_sum.get("embed_space_range", float("nan"))
    ev["attn_space_range"] = ar; ev["embed_space_range"] = er
    ev["attention_better_than_embedding"] = bool(np.isfinite(ar) and np.isfinite(er) and ar > er)
    # 5. cache: does stable attention go with better cache, and does reweighting smooth Δ_edit?
    sv = cache.get("stability_vs_cache_lpips", float("nan"))
    ev["stability_helps_cache"] = bool(np.isfinite(sv) and sv > 0.2)
    smooth_resp = [m["smoothness_response"] for m in iv_sum.get("mechanisms", {}).values()
                   if not np.isnan(m.get("smoothness_response", float("nan")))]
    ev["reweight_can_smooth_delta"] = bool(smooth_resp and min(smooth_resp) < -1e-4)
    cache_helps = ev["stability_helps_cache"] or ev["reweight_can_smooth_delta"]

    if identifiable and controllable and ev["edit_roles_dominate"] and cache_helps:
        v = "STRONG GO"
    elif identifiable and controllable:
        v = "GO"
    elif identifiable or controllable:
        v = "MIXED / ARCHITECTURE-DEPENDENT"
    else:
        v = "NO-GO"
    ev.update(identifiable=bool(identifiable), controllable=bool(controllable),
              cache_helps=bool(cache_helps))
    return v, ev


def main():
    C.ensure_dirs()
    obs = [_load_obs(fp) for fp in _obs_files()]
    if not obs:
        print("[token-analyze] no token_obs_*.json found (run token_autopsy.py first)"); return
    obs_df, per_ex = observational(obs)
    obs_df.to_csv(C.TOK / "per_token_observation.csv", index=False)
    iv_df, iv_sum = interventions()
    if not iv_df.empty:
        iv_df.to_csv(C.TOK / "token_intervention_metrics.csv", index=False)
    cache = cache_connection(per_ex)
    v, ev = verdict(obs_df, per_ex, iv_sum, cache)

    summary = dict(
        phase=C.TOK_PHASE_VERSION, n_examples=len(obs),
        text_entry=C.TEXT_ENTRY, tap_blocks=C.TAP_BLOCKS,
        per_example=list(per_ex.values()),
        interventions=iv_sum, cache_connection=cache,
        verdict=v, verdict_evidence=ev,
    )
    C.write_json(C.TOK / "token_summary.json", summary)
    print(f"[token-analyze] verdict = {v}")
    print(f"[token-analyze] edit/non-edit attention ratio = {ev.get('edit_attention_ratio', float('nan')):.2f}, "
          f"best edit response = {ev['best_edit_response']:.4f}, "
          f"dominant roles = {ev['dominant_roles']}")
    return summary


if __name__ == "__main__":
    main()
