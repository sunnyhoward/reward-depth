#!/usr/bin/env python
"""The z ladder as a figure: install and coherence on one axis, one panel per family.

Both quantities are judge points on the same 0-100 scale, so they share an axis -- no second
y-scale. base and plain DPOP are drawn as dotted reference lines in the matching colour, because
every claim in RESULTS_0819_ZFINE.md is "against base" or "against DPOP" and a reader should not
have to hold those two numbers in their head.

Reads results/probefix_cjudge_zfine/cjudged.json (the merged blind verdicts) and writes
results/probefix/zfine_knee.png.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CJ = os.environ.get("CJ", f"{REPO}/results/probefix_cjudge_zfine")
OUT = os.environ.get("OUT", f"{REPO}/results/probefix/zfine_knee.png")

INSTALL, COH = "#2a78d6", "#eb6834"          # categorical slots 1 and 2, validated for this pair
INK, MUTED, SURFACE = "#1a1a19", "#6b6a63", "#fcfcfb"
ZS = [("L20_z05", 0.50), ("L20_z055", 0.55), ("L20_z06", 0.60),
      ("L20_z065", 0.65), ("L20_z07", 0.70), ("L20_z075", 0.75)]


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
        ax.hlines(p_inst, 0.48, 0.77, color=INSTALL, lw=1.4, ls=(0, (1, 2)), alpha=.85, zorder=1)
        ax.hlines(b_inst, 0.48, 0.77, color=INSTALL, lw=1.4, ls=(0, (4, 3)), alpha=.45, zorder=1)
        ax.hlines(b_coh, 0.48, 0.77, color=COH, lw=1.4, ls=(0, (4, 3)), alpha=.45, zorder=1)
        ax.plot(z, inst, color=INSTALL, lw=2, marker="o", ms=8, zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=2, label="install (british)")
        ax.plot(z, coh, color=COH, lw=2, marker="o", ms=8, zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=2, label="coherence")

        ax.text(0.778, p_inst, "DPOP install", color=INSTALL, fontsize=8, va="center")
        ax.text(0.778, b_inst, "base install", color=MUTED, fontsize=8, va="center")
        ax.text(0.778, b_coh, "base coherence", color=MUTED, fontsize=8, va="center")
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
        ax.set_xlabel("z  (scale on the trained vector at generation)", color=MUTED, fontsize=9)
        ax.set_xlim(0.47, 0.90)
        ax.set_ylim(22, 88)
        ax.set_xticks([v for _, v in ZS])
        ax.tick_params(colors=MUTED, labelsize=9)
    axes[0].set_ylabel("judge points (0–100)", color=MUTED, fontsize=9)
    axes[0].legend(frameon=False, loc="lower left", fontsize=9, labelcolor=INK)
    fig.suptitle("A 2560-parameter add-on at block 20: where the dose stops paying",
                 color=INK, fontsize=12.5, x=0.008, ha="left", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT, dpi=200, facecolor=SURFACE)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
