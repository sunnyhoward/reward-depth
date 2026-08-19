#!/usr/bin/env python
"""The DEPTH ladder as a figure, at a matched relative dose: one panel per family.

Both quantities are judge points on the same 0-100 scale, so they share an axis -- no second
y-scale. base and plain DPOP are drawn as dotted reference lines in the matching colour, because
every claim in RESULTS_0819_ZFINE.md is "against base" or "against DPOP" and a reader should not
have to hold those two numbers in their head.

Every cell carries the same relative edit z*||A_L||/R_L, set to L20's value at z=0.70
(pf_resid_norms.py); at a fixed z the low layers would get a 2.3x larger perturbation than L28 and
the curve would be measuring dose, not depth.

Reads results/probefix_cjudge_layers/cjudged.json (the merged blind verdicts) and writes
results/probefix/layers_curve.png.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CJ = os.environ.get("CJ", f"{REPO}/results/probefix_cjudge_layers")
OUT = os.environ.get("OUT", f"{REPO}/results/probefix/layers_curve.png")

INSTALL, COH = "#2a78d6", "#eb6834"          # categorical slots 1 and 2, validated for this pair
INK, MUTED, SURFACE = "#1a1a19", "#6b6a63", "#fcfcfb"
ZS = [("L4", 4), ("L8", 8), ("L12", 12), ("L16", 16), ("L20", 20), ("L24", 24), ("L28", 28)]


def scores(d, fam, arm):
    """-> (install, coherence). Install is UNCONDITIONAL: non-engagement scores 50, the rubric's
    own neutral anchor (RESULTS_0817_JUDGE_VALIDATION §5a -- `engaged` is post-treatment)."""
    rs = [r["claude"] for r in d.values() if r["fam"] == fam and r["arm"] == arm]
    inst = [(r["british"] if r["engaged"] else 50) for r in rs]
    coh = [r["coherence"] for r in rs]
    return sum(inst) / len(inst), sum(coh) / len(coh)


def main():
    d = json.load(open(f"{CJ}/cjudged.json"))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), sharey=True, facecolor=SURFACE)
    for ax, fam, title in zip(axes, ("false_friend", "style"),
                              ("false_friend — terminology", "style — register")):
        z = [v for _, v in ZS]
        inst = [scores(d, fam, a)[0] for a, _ in ZS]
        coh = [scores(d, fam, a)[1] for a, _ in ZS]
        b_inst, b_coh = scores(d, fam, "base")
        p_inst, p_coh = scores(d, fam, "P1_r1")

        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color="#e8e7e1", lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#d9d8d1")

        # hlines, not axhline: the reference lines stop short of the label column so the text
        # never sits on top of its own line
        ax.hlines(p_inst, 3, 30, color=INSTALL, lw=1.4, ls=(0, (1, 2)), alpha=.85, zorder=1)
        ax.hlines(b_inst, 3, 30, color=INSTALL, lw=1.4, ls=(0, (4, 3)), alpha=.45, zorder=1)
        ax.hlines(b_coh, 3, 30, color=COH, lw=1.4, ls=(0, (4, 3)), alpha=.45, zorder=1)
        ax.plot(z, inst, color=INSTALL, lw=2, marker="o", ms=8, zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=2, label="install (british)")
        ax.plot(z, coh, color=COH, lw=2, marker="o", ms=8, zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=2, label="coherence")

        ax.text(30.5, p_inst, "DPOP install", color=INSTALL, fontsize=8, va="center")
        ax.text(30.5, b_inst, "base install", color=MUTED, fontsize=8, va="center")
        ax.text(30.5, b_coh, "base coherence", color=MUTED, fontsize=8, va="center")
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
        ax.set_xlabel("block whose output the vector is added to", color=MUTED, fontsize=9)
        ax.set_xlim(2, 42)
        ax.set_ylim(22, 88)
        ax.set_xticks([v for _, v in ZS])
        ax.tick_params(colors=MUTED, labelsize=9)
    axes[0].set_ylabel("judge points (0–100)", color=MUTED, fontsize=9)
    axes[0].legend(frameon=False, loc="lower left", fontsize=9, labelcolor=INK)
    fig.suptitle("Where to put the vector: a broad mid-stack band, at a matched dose",
                 color=INK, fontsize=12.5, x=0.008, ha="left", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT, dpi=200, facecolor=SURFACE)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
