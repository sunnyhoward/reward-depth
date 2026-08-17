#!/usr/bin/env python
"""Re-judge the banked famgen generations with Claude subagents instead of Qwen3-32B.

WHY. `results/probefix_handlabel/validation.json` (2026-08-17) found the Qwen3-32B judge does not
execute its own `british` rubric: it collapses to 50 unless the item's OWN contested pair appears,
where the rubric asks for any form on the axis, and it scores strings rather than senses
("The chips are crisps." -> 100 British, when per that item's references it is the American
answer). The error is ARM-CORRELATED (P1 - C1 bias +29.0 +- 15.4 judge points), which is larger
than the +0.2..+9.3 gaps RESULTS_0814_JUDGE_ALL.md §4 read off it.

WHAT IS AND IS NOT ESTABLISHED BY THIS SCRIPT. It replaces the judge, not the validation. Claude
subagents scored against Claude hand labels is not independent evidence -- the errors are
correlated by construction. `blind_for_human.json` is written for exactly this reason: a human
slice is the only check that breaks the circle. Until that is labelled, treat this as a SECOND
instrument that disagrees with the first, not as ground truth.

PROTOCOL, held identical to pf_judge_all.py so the two are comparable:
  · the rubric is EXTRACTED FROM pf_judge_all.py BY AST at runtime, never retyped, so the two
    judges cannot silently drift apart;
  · every item carries its own two references, as there;
  · blind -- the arm never appears in a batch file, and items are shuffled ACROSS arms and
    families before chunking, so no batch (hence no subagent) sees one arm's work as a block.
    That last point is what makes the arm-bias question answerable: a per-agent calibration
    offset cannot correlate with arm if every agent judges every arm.

Subcommands:
  batch   write results/probefix_cjudge/batches/batch_NN.json (+ RUBRIC.md, key.json withheld)
  batchnew  same, for arms that have NO Qwen verdicts -- reads raw famgen_<arm>.json (as written by
          pf_famgen_arms.py) and recovers each item's two references from the banked judged_all.json
          by (family, index). Safe because pf_famgen_arms.py replays pf_famgen.py's prompt
          selection exactly; the join is VERIFIED prompt-by-prompt and aborts on any mismatch.
  merge   collect verdicts/batch_NN.json -> cjudged.json
  report  per-arm scores, the paired P1-vs-C1 test, and agreement vs Qwen and vs hand labels

Env: IN=results/probefix4b_famgen  OUT=results/probefix_cjudge  FAMS=false_friend,style
     BATCH=48  SEED=11
"""
import ast
import json
import os
import random
import sys
from collections import defaultdict

E = os.environ.get
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN = E("IN", f"{REPO}/results/probefix4b_famgen")
OUT = E("OUT", f"{REPO}/results/probefix_cjudge")   # batchnew writes to OUT2
HAND = E("HAND", f"{REPO}/results/probefix_handlabel")
FAMS = [f for f in E("FAMS", "false_friend,style").split(",") if f]
BATCH = int(E("BATCH", 48))
CONTRASTS = [tuple(c.split("-")) for c in
             E("CONTRASTS", "P1-C1,P1-base,C1-base").split(",") if c]
SEED = int(E("SEED", 11))


def rubric_from_source():
    """Pull DIALECT_RUBRIC and AXES out of pf_judge_all.py without importing it (that module
    imports torch and reads the environment). Verbatim by construction."""
    tree = ast.parse(open(f"{REPO}/probefix/pf_judge_all.py").read())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in ("AXES", "DIALECT_RUBRIC"):
                    out[t.id] = ast.literal_eval(node.value)
    assert {"AXES", "DIALECT_RUBRIC"} <= set(out), "rubric constants not found in pf_judge_all.py"
    return out["DIALECT_RUBRIC"], out["AXES"]


# ------------------------------------------------------------------- batch
def do_batch():
    rubric, axes = rubric_from_source()
    judged = json.load(open(f"{IN}/judged_all.json"))
    rng = random.Random(SEED)

    items, key = [], {}
    for fam in FAMS:
        for r in judged[fam]:
            iid = f"x{len(key):04d}"
            key[iid] = {"arm": r["arm"], "fam": fam, "i": r["i"], "qwen": r.get("verdict")}
            items.append({
                "id": iid,
                "family": fam,
                "prompt": r["prompt"],
                "british_ref": r["br"],
                "american_ref": r["am"],
                "generation": r["gen"],
            })

    rng.shuffle(items)                      # across arms AND families, before chunking
    os.makedirs(f"{OUT}/batches", exist_ok=True)
    os.makedirs(f"{OUT}/verdicts", exist_ok=True)
    n = 0
    for b in range(0, len(items), BATCH):
        chunk = items[b:b + BATCH]
        json.dump(chunk, open(f"{OUT}/batches/batch_{n:02d}.json", "w"), indent=1)
        n += 1

    with open(f"{OUT}/RUBRIC.md", "w") as f:
        f.write("# Judging rubric (extracted verbatim from probefix/pf_judge_all.py)\n\n")
        f.write("## Shared frame\n\n```\n" + rubric + "\n```\n\n")
        f.write("## Per-family AXIS block (substituted into `{axis}` above)\n")
        for fam in FAMS:
            f.write(f"\n### {fam}\n\n```\n{axes[fam]}\n```\n")

    json.dump(key, open(f"{OUT}/key.json", "w"), indent=1)

    # a blind slice for a HUMAN labeller -- the only check that is not circular
    rng2 = random.Random(SEED + 1)
    human = rng2.sample(items, 40)
    json.dump(human, open(f"{OUT}/blind_for_human.json", "w"), indent=1)

    print(f"{len(items)} items -> {n} batches of <= {BATCH} in {OUT}/batches/")
    print(f"rubric -> {OUT}/RUBRIC.md ; key withheld in {OUT}/key.json")
    print(f"40-item human slice -> {OUT}/blind_for_human.json")


# ---------------------------------------------------------------- batchnew
def do_batchnew():
    """Batch arms that have no Qwen verdict (e.g. P3_up_dpop, run 2026-08-17).

    ARMS=name=path/to/famgen_name.json[,name2=...]   OUT2=<dir for this pass>
    """
    rubric, axes = rubric_from_source()
    judged = json.load(open(f"{IN}/judged_all.json"))
    refs = {(fam, r["i"]): (r["prompt"], r["br"], r["am"])
            for fam in judged for r in judged[fam]}
    out = E("OUT2", f"{REPO}/results/probefix_cjudge_p3")
    spec = [a for a in E("ARMS", "").split(",") if a]
    assert spec, "set ARMS=name=famgen.json[,...]"

    items, key = [], {}
    for entry in spec:
        arm, path = entry.split("=", 1)
        fg = json.load(open(path))
        for fam in FAMS:
            gens = fg["families"][fam]["gens"]
            for i, g in enumerate(gens):
                prompt, br, am = refs[(fam, i)]
                # the join must be exact: a generator that drew different prompts would be judged
                # against the wrong references, which is silent and fatal
                assert g["prompt"] == prompt, f"prompt mismatch {arm} {fam} {i}"
                iid = f"n{len(key):04d}"
                key[iid] = {"arm": arm, "fam": fam, "i": i, "qwen": None}
                items.append({"id": iid, "family": fam, "prompt": prompt,
                              "british_ref": br, "american_ref": am, "generation": g["gen"]})

    random.Random(SEED).shuffle(items)
    os.makedirs(f"{out}/batches", exist_ok=True)
    os.makedirs(f"{out}/verdicts", exist_ok=True)
    n = 0
    for b in range(0, len(items), BATCH):
        json.dump(items[b:b + BATCH], open(f"{out}/batches/batch_{n:02d}.json", "w"), indent=1)
        n += 1
    with open(f"{out}/RUBRIC.md", "w") as f:
        f.write("# Judging rubric (extracted verbatim from probefix/pf_judge_all.py)\n\n")
        f.write("## Shared frame\n\n```\n" + rubric + "\n```\n\n")
        f.write("## Per-family AXIS block (substituted into `{axis}` above)\n")
        for fam in FAMS:
            f.write(f"\n### {fam}\n\n```\n{axes[fam]}\n```\n")
    json.dump(key, open(f"{out}/key.json", "w"), indent=1)
    print(f"{len(items)} items ({len(spec)} arm(s)) -> {n} batches in {out}/batches/")
    print("prompt join verified item-by-item against judged_all.json")


# ------------------------------------------------------------------- merge
def do_merge():
    key = json.load(open(f"{OUT}/key.json"))
    seen, bad = {}, []
    for fn in sorted(os.listdir(f"{OUT}/verdicts")):
        if not fn.endswith(".json"):
            continue
        for iid, v in json.load(open(f"{OUT}/verdicts/{fn}")).items():
            if iid not in key:
                bad.append((fn, iid, "unknown id"))
                continue
            if not isinstance(v.get("engaged"), bool) or not isinstance(v.get("british"), int) \
                    or not isinstance(v.get("coherence"), int):
                bad.append((fn, iid, "bad schema"))
                continue
            seen[iid] = v
    missing = [i for i in key if i not in seen]
    json.dump({i: {**key[i], "claude": seen[i]} for i in seen},
              open(f"{OUT}/cjudged.json", "w"), indent=1)
    print(f"merged {len(seen)}/{len(key)} verdicts -> {OUT}/cjudged.json")
    if missing:
        print(f"MISSING {len(missing)}: {missing[:12]}{' ...' if len(missing) > 12 else ''}")
    for b in bad:
        print("BAD", b)


# ------------------------------------------------------------------ report
def mean_se(v):
    if not v:
        return float("nan"), float("nan")
    m = sum(v) / len(v)
    if len(v) < 2:
        return m, float("nan")
    var = sum((x - m) ** 2 for x in v) / (len(v) - 1)
    return m, (var / len(v)) ** 0.5


def do_report():
    d = json.load(open(f"{OUT}/cjudged.json"))
    for extra in [x for x in E("EXTRA", "").split(",") if x]:
        # union in another pass's verdicts (e.g. an arm judged later, with no Qwen counterpart) so
        # cross-arm contrasts can be computed on one footing
        d.update(json.load(open(extra)))
    fams = sorted({r["fam"] for r in d.values()})
    arms = sorted({r["arm"] for r in d.values()})
    rep = {"per_arm": {}, "paired": {}, "agreement": {}}

    print("\nCLAUDE JUDGE -- british over ENGAGED items (coherence in brackets)\n")
    hdr = f"{'family':14s}" + "".join(f"{a:>12s}" for a in arms)
    print(hdr)
    for fam in fams:
        row_b, row_c = f"{fam:14s}", f"{'  coherence':14s}"
        for arm in arms:
            rs = [r for r in d.values() if r["fam"] == fam and r["arm"] == arm]
            eng = [r["claude"]["british"] for r in rs if r["claude"]["engaged"]]
            coh = [r["claude"]["coherence"] for r in rs]
            mb, sb = mean_se(eng)
            mc, _ = mean_se(coh)
            rep["per_arm"][f"{fam}/{arm}"] = {
                "n": len(rs), "n_engaged": len(eng), "british": mb, "british_se": sb,
                "coherence": mc, "engagement_rate": len(eng) / len(rs) if rs else None,
            }
            row_b += f"{mb:>8.1f}±{sb:<3.0f}" if eng else f"{'-':>12s}"
            row_c += f"{mc:>12.1f}"
        print(row_b)
        print(row_c)

    # paired: every arm answers the same 48 prompts, so pair on (fam, i). Run the SAME pairing
    # under both judges on the SAME items -- that is the like-for-like comparison, and the only
    # way to tell an arm-ordering difference from a level difference between judges.
    print("\nPAIRED CONTRASTS (same prompt, both arms, engaged by BOTH judges' own criteria)\n")
    print(f"  {'family':14s} {'contrast':12s} {'claude':>18s} {'qwen':>18s}")
    for fam in fams:
        for a, b in CONTRASTS:
            cells = {}
            for j in ("claude", "qwen"):
                idx = {}
                for r in d.values():
                    if r["fam"] == fam and r["arm"] in (a, b) and r.get(j):
                        idx.setdefault(r["i"], {})[r["arm"]] = r[j]
                diffs = [v[a]["british"] - v[b]["british"] for v in idx.values()
                         if a in v and b in v and v[a].get("engaged") and v[b].get("engaged")]
                m, se = mean_se(diffs)
                cells[j] = {"n": len(diffs), "delta": m, "se": se,
                            "sigmas": m / se if se and se == se and se else None}
            rep["paired"][f"{fam}/{a}-{b}"] = cells
            fmt = lambda c: f"{c['delta']:+6.1f}±{c['se']:<4.1f} n={c['n']:<3d}"
            print(f"  {fam:14s} {a+' - '+b:12s} {fmt(cells['claude']):>18s} {fmt(cells['qwen']):>18s}")

    # agreement with the two other instruments
    agree_q = [(r["qwen"], r["claude"]) for r in d.values() if r.get("qwen")]
    if not agree_q:
        print("\n(no Qwen verdicts in this pass -- skipping the judge-vs-judge comparison)")
        json.dump(rep, open(f"{OUT}/report.json", "w"), indent=1)
        return
    both = [(q["british"], c["british"]) for q, c in agree_q
            if q.get("engaged") and c["engaged"]]
    m, se = mean_se([c - q for q, c in both])
    eng_agree = sum(bool(q.get("engaged")) == bool(c["engaged"]) for q, c in agree_q) / len(agree_q)
    rep["agreement"]["vs_qwen"] = {"n_both_engaged": len(both), "claude_minus_qwen": m, "se": se,
                                   "engaged_agree": eng_agree}
    print(f"\nvs QWEN: engaged agree {eng_agree:.3f}; british claude-qwen {m:+.1f} ± {se:.1f} "
          f"(n={len(both)})")

    hand_path = f"{HAND}/labels.json"
    if os.path.exists(hand_path):
        hand = json.load(open(hand_path))
        hkey = json.load(open(f"{HAND}/key.json"))
        lookup = {(v["fam"], v["arm"], v["i"]): hand[i] for i, v in hkey.items() if i in hand}
        pairs = [(lookup[(r["fam"], r["arm"], r["i"])], r["claude"])
                 for r in d.values() if (r["fam"], r["arm"], r["i"]) in lookup]
        be = [(h["british"], c["british"]) for h, c in pairs if h["engaged"] and c["engaged"]]
        m2, se2 = mean_se([c - h for h, c in be])
        ea = sum(bool(h["engaged"]) == bool(c["engaged"]) for h, c in pairs) / len(pairs) if pairs else float("nan")
        rep["agreement"]["vs_hand"] = {"n": len(pairs), "n_both_engaged": len(be),
                                       "claude_minus_hand": m2, "se": se2, "engaged_agree": ea,
                                       "NOTE": "NOT independent: same model family as the judge"}
        print(f"vs HAND (n={len(pairs)}, NOT independent): engaged agree {ea:.3f}; "
              f"british claude-hand {m2:+.1f} ± {se2:.1f} (n={len(be)})")

    json.dump(rep, open(f"{OUT}/report.json", "w"), indent=1)
    print(f"\n-> {OUT}/report.json")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "batch"
    {"batch": do_batch, "batchnew": do_batchnew, "merge": do_merge,
     "report": do_report}[cmd]()
