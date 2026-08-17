#!/usr/bin/env python
"""Chat-template leakage meter -- the pathology no meter in this repo was watching for.

WHY. `RESULTS_0817_MEANDIFF_REJUDGE.md` found the mean-diff arms degenerate at 600 steps in a way
distinct from the DPO arms' en-dash and "I colour the colour" loops: the reply COMPLETES, then emits
a fresh `user` / `assistant` / `<think>` turn and re-answers, often looping. Qwen scored those same
generations 94.9-97.1 on coherence. Every existing meter is blind to it -- `brit_rate` is a ratio
over marker-bearing samples, `diversity` compares openings ACROSS generations (RESULTS_0813 §5: it
read .94 for an arm that loops WITHIN them), and the judge only catches it when a human thought to
put coherence on the rubric.

This is a regex, it needs no GPU and no judge, and it is exact: the tokens below are the chat
template's own control strings, so a hit is a hit. Two rates are reported because they are different
failures: `leak` is any role/think marker in the completion, `restart` is the model re-asking or
re-answering the prompt after finishing.

Usage:  python pf_leakage.py results/probefix4b_meandiff/famgen_*.json ...
        python pf_leakage.py --all          # every banked famgen_*.json under results/
Env:    FAMS=false_friend,style   (default: every family present in the file)
"""
import glob
import json
import os
import re
import sys

E = os.environ.get
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAMS = [f for f in E("FAMS", "").split(",") if f]

# The template's own control strings, plus the bare role words when they head a line -- Qwen3.5
# emits `<|im_start|>user` but a de-tokenised generation often carries only `user\n`.
LEAK = re.compile(r"<\|im_(start|end)\|>|</?think>|^\s*(user|assistant|system)\s*$", re.M)
ROLE_HEAD = re.compile(r"^\s*(user|assistant)\s*$", re.M)


def scan(gen):
    leak = bool(LEAK.search(gen))
    # a restart = a role header appears AND text follows it, i.e. the model took another turn
    restart = False
    m = ROLE_HEAD.search(gen)
    if m and gen[m.end():].strip():
        restart = True
    return leak, restart


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--all" in sys.argv or not args:
        args = sorted(glob.glob(f"{REPO}/results/**/famgen_*.json", recursive=True))
    rows = []
    for path in args:
        try:
            d = json.load(open(path))
        except Exception as e:                                    # noqa: BLE001
            print(f"  skip {path}: {e}")
            continue
        if "families" not in d:
            continue
        arm = d.get("arm", os.path.basename(path))
        fams = [f for f in d["families"] if not FAMS or f in FAMS]
        n = leaks = restarts = 0
        per_fam = {}
        for fam in fams:
            gens = d["families"][fam].get("gens") or []
            fl = fr = 0
            for g in gens:
                text = g["gen"] if isinstance(g, dict) else g
                a, b = scan(text)
                fl += a
                fr += b
            per_fam[fam] = dict(n=len(gens), leak=fl, restart=fr)
            n += len(gens)
            leaks += fl
            restarts += fr
        if n:
            rows.append((arm, n, leaks / n, restarts / n, per_fam, path))

    rows.sort(key=lambda r: -r[2])
    print(f"{'arm':32s} {'n':>4s} {'leak':>7s} {'restart':>8s}   per-family leak")
    for arm, n, lr, rr, per_fam, _ in rows:
        pf = " ".join(f"{k[:5]} {v['leak']}/{v['n']}" for k, v in sorted(per_fam.items()))
        print(f"{arm:32s} {n:4d} {lr:7.3f} {rr:8.3f}   {pf}")
    out = f"{REPO}/results/probefix/leakage_scan.json"
    json.dump([dict(arm=a, n=n, leak_rate=lr, restart_rate=rr, per_family=pf, path=p)
               for a, n, lr, rr, pf, p in rows], open(out, "w"), indent=1)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
