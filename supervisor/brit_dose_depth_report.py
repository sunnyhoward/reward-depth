#!/usr/bin/env python
"""Read the guard-dose x attach-depth arms and say whether the prediction survived.

THE PREDICTION, stated before the arms ran (run_brit_dose_depth.sh): plain britishness is an L0
phenomenon and eagle/RESULTS.md §2 found L4 installs it cheaply while L12 shows the surrogate gap.
decodability/brit_guard_dose.py showed that dosing the guard moves the composite rule to 0.44-0.67
fractional depth. So the attach depth that installs should MOVE UP with the dose.

WHAT WOULD FALSIFY IT: L4 installing the 20% arm as well as the 4% arm. That would say attach
depth does not track where the rule lives.

THE COLUMN THAT DECIDES IT is held-out GUARD, and it is the one the original run could not
produce -- results_0807's split put all 200 guard rows in train. Read it against two reference
points, because the guard is a ceiling metric:
  base raw ~0.995   the untrained model already gets the guard right, so training can only
                    damage this column. An arm that "improves" raw guard is doing so from
                    within noise on 50 rows.
  implicit          movement against the adapter-off reference. An arm whose raw guard holds
                    while implicit guard sits near 0 has not learned the guard, it has merely
                    not broken it -- which for a 4% arm is the expected and uninteresting result.

n = 50 on the guard bucket, so SE ~= 0.07. Differences under ~0.14 are not differences. This
script prints the SE next to every column rather than leaving the reader to compute it.

Usage: python supervisor/brit_dose_depth_report.py [--root /workspace/brit_dose]
"""
import argparse
import glob
import json
import os
import re


def load(root):
    """→ {(layer, dose): {ckpt: {bucket: metrics}}}"""
    out = {}
    for f in sorted(glob.glob(os.path.join(root, "fam_eval_L*_d*.json"))):
        m = re.search(r"fam_eval_L(\d+)_d(\d+)\.json", os.path.basename(f))
        if not m:
            continue
        out[(int(m.group(1)), int(m.group(2)))] = json.load(open(f))
    return out


def ckpt_num(name):
    m = re.search(r"ckpt(\d+)", name)
    return int(m.group(1)) if m else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/workspace/brit_dose")
    ap.add_argument("--bucket", default="guard")
    args = ap.parse_args()
    data = load(args.root)
    if not data:
        raise SystemExit(f"no fam_eval_*.json under {args.root} yet")

    buckets = sorted({b for arm in data.values() for ck in arm.values() for b in ck})
    n = {}
    for arm in data.values():
        for ck in arm.values():
            for b, v in ck.items():
                n[b] = v.get("n")
    print("holdout buckets: " + " | ".join(
        f"{b} n={n[b]} (SE {0.5 / max(n[b], 1) ** 0.5:.3f})" for b in buckets if n.get(b)))

    for dose in sorted({d for _, d in data}):
        print(f"\n## guard {dose}% of the training diet")
        print("| attach | ckpt | " + " | ".join(
            f"{b} raw / impl" for b in buckets if b != "unlabelled") + " |")
        print("|---" * (2 + len([b for b in buckets if b != "unlabelled"])) + "|")
        for layer in sorted({l for l, d in data if d == dose}):
            arm = data[(layer, dose)]
            for ck in sorted(arm, key=ckpt_num):
                cells = []
                for b in buckets:
                    if b == "unlabelled":
                        continue
                    v = arm[ck].get(b)
                    if not v:
                        cells.append("--")
                        continue
                    imp = v.get("implicit")
                    cells.append(f"{v['raw']:.3f} / " + (f"{imp:.3f}" if imp is not None else "--"))
                print(f"| L{layer} | {os.path.basename(ck)} | " + " | ".join(cells) + " |")

    # The one comparison the whole run exists to make.
    print("\n## the prediction: does the best attach depth move up with the dose?")
    print("| dose | attach | held-out guard raw | guard implicit | legacy-750 raw | "
          "pooled new-holdout raw |")
    print("|---|---|---|---|---|---|")
    for dose in sorted({d for _, d in data}):
        for layer in sorted({l for l, d in data if d == dose}):
            arm = data[(layer, dose)]
            last = sorted(arm, key=ckpt_num)[-1]
            g = arm[last].get("guard", {})
            lg = arm[last].get("legacy", {})
            po = arm[last].get("pooled_new", {})
            print(f"| {dose}% | L{layer} | {g.get('raw', float('nan')):.3f} | "
                  + (f"{g['implicit']:.3f}" if g.get("implicit") is not None else "--")
                  + f" | {lg.get('raw', float('nan')):.3f} | {po.get('raw', float('nan')):.3f} |")
    print("\nRead the guard column against base raw ~0.995 (training can only damage it) and "
          "remember SE ~0.07 at n=50.")


if __name__ == "__main__":
    main()
