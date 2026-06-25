"""E50 data prep: select E49 subsets, synthesise spectral input images, write job manifests.

Pure numpy/PIL — run in either env (use anaconda). Produces, per experiment, the manipulated
input PNGs under data/selected_subsets/ and a fully-traceable job manifest under data/manifests/.
Every job row records: source id, style id, instruction/prompt, spectral op, model, seed, inference
settings, the prepared input-image path, and where the output PNG will land.
"""
from __future__ import annotations
import json
import config as C
import spectral as S

def _load_manifest(path):
    return [json.loads(l) for l in open(path) if l.strip()]

def _abs(e49_rel):
    # E49 manifests store paths relative to the baseline_establishment/ root.
    return C.E49 / e49_rel

def prepare():
    C.ensure_dirs()
    pie = {r["id"]: r for r in _load_manifest(C.E49_PIE / "manifest.jsonl")}
    leak = {r["id"]: r for r in _load_manifest(C.E49_LEAK / "manifest.jsonl")}

    jobs = []

    # ---------------- Experiment C: SOURCE spectral decomposition (instruction edits)
    for sid in C.EDIT_IDS:
        r = pie[sid]
        src = _abs(r["source"])
        for op in C.SOURCE_OPS:
            inp = C.SUBSETS / f"src_{sid}_{op}.png"
            S.SOURCE_FN[op](src, C.GEN_SIZE).save(inp)
            jobs.append(dict(
                job_id=f"src_{sid}_{op}", experiment="spectral_source",
                bucket="spectral_source",
                source_id=sid, task_type=r["task_type"], style_id=None,
                instruction=r["instruction"], prompt_key=None, target_prompt=r["target_prompt"],
                source_prompt=r.get("source_prompt", ""), spectral_op=op,
                input_image=C.rel(inp), content_ref=C.rel(src), style_ref=None,
                model="flux_kontext", seed=C.SEED, size=C.GEN_SIZE, steps=C.STEPS,
                guidance=C.GUIDANCE,
                output=f"e50_spectral_kontext_pilot/outputs/spectral_source/src_{sid}_{op}_s{C.SEED}.png"))

    # ---------------- Experiment A: spectral REFERENCE composites (adversarial leakage)
    for lid in C.LEAK_IDS:
        r = leak[lid]
        content, style = _abs(r["content"]), _abs(r["style"])
        instr = r["instruction"]                       # e.g. "render this image in a watercolor painting style"
        for op in C.REF_OPS:
            inp = C.SUBSETS / f"ref_{lid}_{op}.png"
            S.REF_FN[op](content, style, C.GEN_SIZE).save(inp)
            bucket = "kontext_baseline_replay" if op == "content_raw" else "spectral_reference"
            jobs.append(dict(
                job_id=f"ref_{lid}_{op}", experiment="spectral_reference",
                bucket=bucket,
                source_id=r["content_id"], task_type="stylization_adversarial",
                style_id=r["style_id"], style_category=r["style_category"],
                instruction=instr, prompt_key="neutral_style", target_prompt=None,
                source_prompt=None, spectral_op=op,
                input_image=C.rel(inp), content_ref=r["content"], style_ref=r["style"],
                model="flux_kontext", seed=C.SEED, size=C.GEN_SIZE, steps=C.STEPS,
                guidance=C.GUIDANCE,
                output=f"e50_spectral_kontext_pilot/outputs/{bucket}/ref_{lid}_{op}_s{C.SEED}.png"))

    # ---------------- Experiment B: prompt variants (content image input, vary instruction)
    for lid in C.PROMPT_LEAK_IDS:
        r = leak[lid]
        content, style = _abs(r["content"]), _abs(r["style"])
        inp = C.SUBSETS / f"prompt_{lid}_content.png"
        S.op_raw(content, C.GEN_SIZE).save(inp)
        for pk, ptext in C.PROMPTS.items():
            # bake the style category into the instruction so "reference style" is concrete
            instr = ptext.replace("the reference", f"a {r['style_category']} painting")
            jobs.append(dict(
                job_id=f"prompt_{lid}_{pk}", experiment="prompt_variants",
                bucket="prompt_variants",
                source_id=r["content_id"], task_type="stylization_adversarial",
                style_id=r["style_id"], style_category=r["style_category"],
                instruction=instr, prompt_key=pk, target_prompt=None, source_prompt=None,
                spectral_op="none",
                input_image=C.rel(inp), content_ref=r["content"], style_ref=r["style"],
                model="flux_kontext", seed=C.SEED, size=C.GEN_SIZE, steps=C.STEPS,
                guidance=C.GUIDANCE,
                output=f"e50_spectral_kontext_pilot/outputs/prompt_variants/prompt_{lid}_{pk}_s{C.SEED}.png"))

    mf = C.MANIFESTS / "e50_jobs.jsonl"
    with open(mf, "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    by_exp = {}
    for j in jobs:
        by_exp[j["experiment"]] = by_exp.get(j["experiment"], 0) + 1
    C.write_json(C.MANIFESTS / "e50_jobs_summary.json",
                 dict(total=len(jobs), by_experiment=by_exp,
                      edit_ids=C.EDIT_IDS, leak_ids=C.LEAK_IDS))
    print(f"prepared {len(jobs)} jobs -> {C.rel(mf)}")
    print("by experiment:", by_exp)
    return jobs


if __name__ == "__main__":
    prepare()
