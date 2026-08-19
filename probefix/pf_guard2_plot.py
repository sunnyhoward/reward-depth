#!/usr/bin/env python
"""The trade-off the guard-2 set exposes: install on one axis, the conditional on the other.

Each point is one arm. x = judged `style` install (results/probefix_cjudge_zfine, the z grid's own
pass). y = fraction of the 40 US-context guard-2 items answered CORRECTLY, where answering with the
British form is what makes them wrong. Up and to the right is an intervention that installs the
preference AND respects the exception; the add-on traces a line down and to the right, DPOP sits in
the corner.

Reads results/probefix_cjudge_guard2/{key,verdicts} and results/probefix_cjudge_zfine/cjudged.json,
writes results/probefix/guard2_tradeoff.png.
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G2 = f"{REPO}/results/probefix_cjudge_guard2"
ZF = f"{REPO}/results/probefix_cjudge_zfine"
OUT = f"{REPO}/results/probefix/guard2_tradeoff.png"

ADDON, DPOP = "#2a78d6", "#eb6834"
INK, MUTED, SURFACE = "#1a1a19", "#6b6a63", "#fcfcfb"

# (guard-2 arm, zfine arm, label, colour)
ARMS = [("base", "base", "base", MUTED),
        ("addon_L20_z0.5", "L20_z05", "add-on z 0.50", ADDON),
        ("addon_L20_z0.7", "L20_z07", "add-on z 0.70", ADDON),
        ("P1", "P1_r1", "DPOP (~21M params)", DPOP)]


def guard_correct(arm):
    key = json.load(open(f"{G2}/key.json"))
    verd = {}
    for f in glob.glob(f"{G2}/verdicts/batch_*.json"):
        verd.update(json.load(open(f)))
    vs = [verd[i] for i in key if key[i]["arm"] == arm and key[i]["ctx"] == "us" and i in verd]
    return 100 * sum(1 for v in vs if v["correct"] == "true") / len(vs)


def style_install(arm):
    d = json.load(open(f"{ZF}/cjudged.json"))
    vs = [r["claude"] for r in d.values() if r["fam"] == "style" and r["arm"] == arm]
    return sum((v["british"] if v["engaged"] else 50) for v in vs) / len(vs)


def main():
    fig, ax = plt.subplots(figsize=(6.6, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(color="#e8e7e1", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d9d8d1")

    pts = [(style_install(zf), guard_correct(g2), lab, col) for g2, zf, lab, col in ARMS]
    trail = [p for p in pts if p[3] == ADDON or p[2] == "base"]
    ax.plot([p[0] for p in trail], [p[1] for p in trail], color=ADDON, lw=2, alpha=.5, zorder=2)
    for x, y, lab, col in pts:
        ax.plot([x], [y], marker="o", ms=11, color=col, zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=2)
        dy = -5.5 if lab == "DPOP (~21M params)" else 3.2
        ax.text(x, y + dy, lab, color=col if col != MUTED else INK, fontsize=9,
                ha="center", va="bottom" if dy > 0 else "top")

    ax.set_xlabel("style install  (judge points, unconditional)", color=MUTED, fontsize=9.5)
    ax.set_ylabel("US guard items answered correctly  (%)", color=MUTED, fontsize=9.5)
    ax.set_xlim(20, 80)
    ax.set_ylim(40, 90)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_title("Installing the preference vs keeping the exception",
                 color=INK, fontsize=12.5, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(OUT, dpi=200, facecolor=SURFACE)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
