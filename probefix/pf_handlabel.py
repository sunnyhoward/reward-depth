#!/usr/bin/env python
"""Hand-label validation of the Qwen3-32B judge used by pf_judge_all.py.

WHY THIS EXISTS.
  `RESULTS_0814_JUDGE_ALL.md` §5.5 states the gap plainly: "No hand-labelled validation. The
  standing instruction is to validate a judge against hand labels before trusting it; that has
  not been done here. Judge-vs-regex agreement on `lexicon` is a weaker substitute." Every arm
  ordering in that document, and in RESULTS_0814_{MEANDIFF,REPLAY}.md, rests on the judge.

  Judge-vs-regex on `lexicon` cannot substitute, for a reason internal to this project: 0813
  established the regex reads majority ORTHOGRAPHY, so agreement there certifies the judge only
  on the one axis a regex could already measure. `false_friend` and `style` -- the two families
  RESULTS_0814 §6 nominates as the instruments to keep -- have no such check.

WHAT IT DOES. Two phases, deliberately separated so the labeller cannot see the answers.

  sample  reads the banked `judged_all.json`, draws a stratified sample, and writes TWO files:
            blind.json  id, family, prompt, both references, generation      <- the labeller reads this
            key.json    id -> arm + the judge's own verdict                  <- withheld until scoring
          Stratification is over the JUDGE's (engaged, british-band) cells plus a floor on
          low-coherence items, so the sample spans the range the judge claims rather than the
          range the population happens to have. Arms are never shown and items are shuffled
          within family.

  score   joins labels.json against key.json and reports, per family:
            engaged    agreement, Cohen's kappa
            british    Spearman rho, mean signed error (judge - hand), direction agreement
            coherence  agreement on the degenerate/healthy split
          plus the question that decides whether the arm tables survive: is the judge's error
          CORRELATED WITH ARM? A judge that is noisy but unbiased across arms leaves the
          ordering intact; one whose error tracks the arm does not.

  The labeller writes `british` on the same 0-100 scale as the rubric, but is expected to use
  the anchors {0, 25, 50, 75, 100} -- finer resolution than that is not honestly available by eye.

Env: IN=results/probefix4b_famgen  OUT=results/probefix_handlabel  N=30  SEED=7
     FAMS=lexicon,false_friend,style
"""
import json
import os
import random
import sys
from collections import Counter, defaultdict

E = os.environ.get
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN = E("IN", f"{REPO}/results/probefix4b_famgen")
OUT = E("OUT", f"{REPO}/results/probefix_handlabel")
N = int(E("N", 30))
SEED = int(E("SEED", 7))
FAMS = [f for f in E("FAMS", "lexicon,false_friend,style").split(",") if f]
MIN_DEGEN = int(E("MIN_DEGEN", 5))       # per family, judge coherence < 60


def band(b):
    return "am" if b <= 33 else ("mid" if b <= 66 else "br")


def cell(v):
    if not v.get("engaged"):
        return "notengaged"
    return band(v.get("british", 50))


# ------------------------------------------------------------------ sample
def do_sample():
    judged = json.load(open(f"{IN}/judged_all.json"))
    rng = random.Random(SEED)
    blind, key = [], {}

    for fam in FAMS:
        rows = [r for r in judged[fam] if r.get("verdict")]
        by_cell = defaultdict(list)
        for r in rows:
            by_cell[cell(r["verdict"])].append(r)
        for c in by_cell:
            rng.shuffle(by_cell[c])

        # low-coherence floor first: the axis six meters have missed
        degen = [r for r in rows if r["verdict"].get("coherence", 100) < 60]
        rng.shuffle(degen)
        picked, seen = [], set()

        def take(r):
            k = (r["arm"], r["fam"], r["i"])
            if k in seen:
                return False
            seen.add(k)
            picked.append(r)
            return True

        for r in degen[:MIN_DEGEN]:
            take(r)

        # then round-robin the judge's own cells until N
        cells = [c for c in ("br", "am", "mid", "notengaged") if by_cell[c]]
        ptr = {c: 0 for c in cells}
        while len(picked) < N and cells:
            for c in list(cells):
                if len(picked) >= N:
                    break
                while ptr[c] < len(by_cell[c]) and not take(by_cell[c][ptr[c]]):
                    ptr[c] += 1
                if ptr[c] < len(by_cell[c]):
                    ptr[c] += 1
                else:
                    cells.remove(c)

        rng.shuffle(picked)
        for j, r in enumerate(picked):
            iid = f"{fam[:3]}{j:02d}"
            blind.append({
                "id": iid,
                "family": fam,
                "prompt": r["prompt"],
                "british_ref": r["br"],
                "american_ref": r["am"],
                "generation": r["gen"],
            })
            key[iid] = {"arm": r["arm"], "fam": fam, "i": r["i"], "verdict": r["verdict"]}

    os.makedirs(OUT, exist_ok=True)
    json.dump(blind, open(f"{OUT}/blind.json", "w"), indent=1)
    json.dump(key, open(f"{OUT}/key.json", "w"), indent=1)
    print(f"wrote {len(blind)} items -> {OUT}/blind.json (+ key.json, do not read before labelling)")
    for fam in FAMS:
        ks = [k for k, v in key.items() if v["fam"] == fam]
        print(f"  {fam:14s} n={len(ks):3d}  judge cells "
              f"{dict(Counter(cell(key[k]['verdict']) for k in ks))}  "
              f"degen={sum(key[k]['verdict'].get('coherence', 100) < 60 for k in ks)}  "
              f"arms {dict(Counter(key[k]['arm'] for k in ks))}")


# ------------------------------------------------------------------- score
def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    if len(xs) < 3:
        return float("nan")
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def kappa(a, b):
    n = len(a)
    if not n:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa1, pb1 = sum(a) / n, sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def do_score():
    key = json.load(open(f"{OUT}/key.json"))
    labels = json.load(open(f"{OUT}/labels.json"))
    blind = {b["id"]: b for b in json.load(open(f"{OUT}/blind.json"))}
    fams = sorted({v["fam"] for v in key.values()})
    report = {}

    for fam in fams + ["ALL"]:
        ids = [i for i in labels if i in key and (fam == "ALL" or key[i]["fam"] == fam)]
        if not ids:
            continue
        je = [bool(key[i]["verdict"].get("engaged")) for i in ids]
        he = [bool(labels[i]["engaged"]) for i in ids]
        both = [i for i in ids if key[i]["verdict"].get("engaged") and labels[i]["engaged"]]
        jb = [key[i]["verdict"]["british"] for i in both]
        hb = [labels[i]["british"] for i in both]
        jc = [key[i]["verdict"].get("coherence", 100) for i in ids]
        hc = [labels[i]["coherence"] for i in ids]

        d = {
            "n": len(ids),
            "engaged_agree": sum(x == y for x, y in zip(je, he)) / len(ids),
            "engaged_kappa": kappa(je, he),
            "engaged_judge_rate": sum(je) / len(ids),
            "engaged_hand_rate": sum(he) / len(ids),
            "n_both_engaged": len(both),
            "british_rho": spearman(jb, hb) if len(both) >= 3 else None,
            "british_mean_signed_err": (sum(a - b for a, b in zip(jb, hb)) / len(both)) if both else None,
            "british_mean_abs_err": (sum(abs(a - b) for a, b in zip(jb, hb)) / len(both)) if both else None,
            "british_direction_agree": (sum(band(a) == band(b) for a, b in zip(jb, hb)) / len(both)) if both else None,
            "coherence_rho": spearman(jc, hc),
            "degen_agree": sum((a < 60) == (b < 60) for a, b in zip(jc, hc)) / len(ids),
            "degen_kappa": kappa([a < 60 for a in jc], [b < 60 for b in hc]),
        }
        report[fam] = d

    # does the judge's error track the arm? (the question that decides the arm tables)
    per_arm = defaultdict(list)
    for i in labels:
        if i not in key:
            continue
        v, l = key[i]["verdict"], labels[i]
        if v.get("engaged") and l["engaged"]:
            per_arm[key[i]["arm"]].append(v["british"] - l["british"])
    def mean_se(v):
        m = sum(v) / len(v)
        if len(v) < 2:
            return m, float("nan")
        var = sum((x - m) ** 2 for x in v) / (len(v) - 1)
        return m, (var / len(v)) ** 0.5

    report["arm_bias"] = {}
    for a, v in sorted(per_arm.items()):
        if not v:
            continue
        m, se = mean_se(v)
        report["arm_bias"][a] = {"n": len(v), "mean_signed_err": m, "se": se}

    # the contrast RESULTS_0814_JUDGE_ALL §4 rests on: is P1's lead over C1 an artifact of the
    # judge scoring the two arms differently than a human does?
    if per_arm.get("P1") and per_arm.get("C1"):
        mp, sp = mean_se(per_arm["P1"])
        mc, sc = mean_se(per_arm["C1"])
        d = mp - mc
        se = (sp ** 2 + sc ** 2) ** 0.5
        report["P1_vs_C1_judge_bias"] = {
            "n_P1": len(per_arm["P1"]), "n_C1": len(per_arm["C1"]),
            "delta_judge_minus_hand": d, "se": se, "sigmas": d / se if se else None,
        }

    disagreements = []
    for i in sorted(labels):
        if i not in key:
            continue
        v, l = key[i]["verdict"], labels[i]
        big_b = v.get("engaged") and l["engaged"] and abs(v["british"] - l["british"]) >= 50
        if bool(v.get("engaged")) != bool(l["engaged"]) or big_b or \
                (v.get("coherence", 100) < 60) != (l["coherence"] < 60):
            disagreements.append({
                "id": i, "arm": key[i]["arm"], "fam": key[i]["fam"],
                "judge": v, "hand": l,
                "prompt": blind[i]["prompt"], "gen": blind[i]["generation"][:400],
            })
    report["n_disagreements"] = len(disagreements)

    json.dump({"report": report, "disagreements": disagreements},
              open(f"{OUT}/validation.json", "w"), indent=1)

    for fam in fams + ["ALL"]:
        if fam not in report:
            continue
        d = report[fam]
        print(f"\n{fam}  n={d['n']}")
        print(f"  engaged   agree {d['engaged_agree']:.3f}  kappa {d['engaged_kappa']:.3f}"
              f"   rate judge {d['engaged_judge_rate']:.2f} vs hand {d['engaged_hand_rate']:.2f}")
        if d["british_rho"] is not None:
            print(f"  british   rho {d['british_rho']:.3f}  signed err {d['british_mean_signed_err']:+.1f}"
                  f"  abs err {d['british_mean_abs_err']:.1f}  band agree {d['british_direction_agree']:.3f}"
                  f"  (n={d['n_both_engaged']})")
        print(f"  coherence rho {d['coherence_rho']:.3f}  degen agree {d['degen_agree']:.3f}"
              f"  kappa {d['degen_kappa']:.3f}")
    print("\narm bias (judge - hand, engaged-by-both items):")
    for a, v in report["arm_bias"].items():
        print(f"  {a:5s} n={v['n']:3d}  {v['mean_signed_err']:+.1f} +- {v['se']:.1f}")
    if "P1_vs_C1_judge_bias" in report:
        b = report["P1_vs_C1_judge_bias"]
        print(f"  P1 - C1 judge bias {b['delta_judge_minus_hand']:+.1f} +- {b['se']:.1f}"
              f"  ({b['sigmas']:+.1f} SE)")
    print(f"\n{report['n_disagreements']} material disagreements -> {OUT}/validation.json")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "sample"
    {"sample": do_sample, "score": do_score}[cmd]()
