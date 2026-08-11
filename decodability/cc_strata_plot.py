#!/usr/bin/env python
"""Figures for the computation-correctness stratification (`cc_strata.py`).

THREE PANELS, ONE ARGUMENT.
  A  the aggregate depth curve with its strata drawn underneath it. If the two steps in
     `depth_meaningful.png` are a mixture, the aggregate is a weighted average of three curves
     with different L*, and the grey mixture prediction (0.5 + f/2 with f = know + units) lands on
     the plateau.
  B  cumulative fraction of items RESOLVED by depth -- the per-item version of A, which is what
     turns "the curve steps twice" into "these items resolve here, those resolve there".
  C  the identifiability test. `offset = ±10` is both the same-last-digit case and the largest
     absolute error, so units-first predicts it is the WORST subgroup at the plateau while a
     magnitude/plausibility feature predicts it is the BEST. Nothing else in this figure can
     separate those two accounts; this panel is the only place they disagree.

COLOUR ENCODES THE STRATUM, everywhere, including panel C -- an `|offset|` of 10 is drawn in the
`tens` colour because it IS the tens stratum, so the eye carries the identity across panels. The
aggregate is grey because it is not a stratum, it is their mixture. Slots 1-3 of the repo's
categorical order, revalidated for this figure: `validate_palette.js "#2a78d6,#eb6834,#1baf7a"
--mode light` -> all checks PASS, worst adjacent CVD ΔE 9.2 (deutan), with a contrast WARN on the
green that obligates relief; supplied as a direct end-label on every series plus a per-stratum
dash pattern, so identity is never colour alone.

Usage: python cc_strata_plot.py [--model qwen3-4b] [--read mean] [--outdir <C.RESULT_DIR>/plots]
"""
import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dec_common as C  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8985"
GRID = "#e6e5e1"

# (key, label, colour, dash) -- fixed slot order, never cycled.
STRATA = [
    ("know", "know (retrieval)", "#2a78d6", (0, (1, 1.6))),
    ("units", "units  |off|<10  (mod 10)", "#eb6834", (0, (5, 1.8))),
    ("tens", "tens  |off|=10  (carry)", "#1baf7a", "solid"),
]
NICE = {"corr_e": "computation-correctness | explained",
        "corr_t": "computation-correctness | terse"}


def load(model, read, dataset="styc", render="chat"):
    p = os.path.join(C.RESULT_DIR, f"ccstrata_{model}_{dataset}_{render}_{read}.json")
    if not os.path.exists(p):
        hits = glob.glob(os.path.join(C.RESULT_DIR, "ccstrata_*.json"))
        raise FileNotFoundError(f"{p}\nhave: {hits}")
    return json.load(open(p))


def _spread(ax, items, gap, x):
    """Direct end-labels, nudged apart. items = [(y, text, colour)].

    Direct labelling is not decoration here -- the green fails the contrast check against this
    surface, so the palette is only legal with visible labels. Overlapping labels would forfeit
    exactly the relief they are there to supply, so they get pushed apart rather than dropped.
    """
    items = sorted(items, key=lambda t: t[0])
    ys = [t[0] for t in items]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    lo, hi = ax.get_ylim()
    over = ys[-1] - (hi - gap * 0.4)          # the block can only grow upward, so clamp it back
    if over > 0:
        ys = [y - min(over, ys[0] - (lo + gap * 0.4)) for y in ys]
    for (y0, text, colour), y in zip(items, ys):
        ax.annotate(text, xy=(x, y0), xytext=(x + 0.012, y), color=colour, fontsize=6.9,
                    va="center", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=colour, lw=0.6, alpha=0.45,
                                    shrinkA=0, shrinkB=2) if abs(y - y0) > gap * 0.4 else None)


def _axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=7.5, length=3)


def panel_depth(ax, cell):
    """A: aggregate + strata against fractional depth, ±1 SD over seeds."""
    x = np.asarray(cell["frac_depth"])
    n = cell["n_items"]
    ax.axhline(0.5, color=INK_MUTED, lw=0.8, ls=(0, (3, 3)), zorder=1)

    # The prediction this figure was built to test, drawn because it FAILS: if the two steps were
    # a mixture of populations with different L*, the plateau would sit here (know + units
    # resolved, tens at chance). The observed plateau is ~11 points below it, and the strata below
    # show why -- both steps happen to the same arithmetic items.
    f = (n["know"] + n["units"]) / n["all"]
    pred = 0.5 + f / 2
    ax.axhline(pred, color=INK_MUTED, lw=1.1, ls=(0, (1.5, 1.5)), zorder=2)
    ax.text(0.012, pred + 0.012, f"mixture prediction ({pred:.3f}) — know + units resolved, tens "
                                 f"at chance.  NOT MET: the plateau is ~11 pts below it.",
            color=INK_2, fontsize=6.8, va="bottom", zorder=6)

    ax.plot(x, cell["acc_mean"]["all"], color=INK_2, lw=2.4, zorder=5, solid_capstyle="round")
    ends = [(cell["acc_mean"]["all"][-1], f"all  n={n['all']}", INK_2)]
    for key, label, colour, dash in STRATA:
        m = np.asarray(cell["acc_mean"][key])
        sd = np.asarray(cell["acc_sd"][key])
        ax.fill_between(x, m - sd, m + sd, color=colour, alpha=0.16, lw=0, zorder=3)
        ax.plot(x, m, color=colour, lw=2.0, ls=dash, zorder=4, solid_capstyle="round")
        ends.append((float(m[-1]), f"{label}  n={n[key]}", colour))
    _axes(ax)
    ax.set_xlim(-0.01, 1.42)
    ax.set_ylim(0.40, 1.04)
    _spread(ax, ends, 0.030, x[-1])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("held-out pairwise acc.", color=INK_2, fontsize=8)
    ax.set_xlabel("fractional depth   (0 = embeddings, 1 = top block)", color=INK_2, fontsize=8)


def panel_resolved(ax, cell):
    """B: cumulative fraction of items resolved at or below depth x."""
    xs = np.asarray(cell["frac_depth"])
    ends = []
    for key, label, colour, dash in STRATA:
        rd = np.asarray(cell["resolution_depth"][key])
        y = [(rd <= v + 1e-9).mean() for v in xs]
        ax.step(xs, y, where="post", color=colour, lw=2.0, ls=dash, zorder=4)
        ends.append((float(y[-1]), f"{key}  {y[-1]:.2f}", colour))
        med = np.median(rd[rd <= 1.0]) if (rd <= 1.0).any() else np.nan
        if np.isfinite(med):
            ax.plot([med], [0.5], marker="o", ms=5.0, color=colour, mec=SURFACE, mew=1.4, zorder=5)
    _axes(ax)
    ax.set_xlim(-0.01, 1.30)
    ax.set_ylim(-0.02, 1.05)
    _spread(ax, ends, 0.075, xs[-1])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("fraction of items resolved", color=INK_2, fontsize=8)
    ax.set_xlabel("fractional depth of resolution", color=INK_2, fontsize=8)
    ax.set_title("B — when each population resolves\n"
                 "dot = median;  right-edge value = fraction ever resolved",
                 color=INK, fontsize=8.4, loc="left", pad=6)


def panel_offset(ax, cell):
    """C: accuracy by |offset| at the plateau, where the two accounts disagree."""
    x = np.asarray(cell["frac_depth"])
    gap = np.asarray(cell["acc_mean"]["units"]) - np.asarray(cell["acc_mean"]["tens"])
    p = int(np.argmax(gap))
    offs = [1, 2, 3, 10]
    ax.axhline(0.5, color=INK_MUTED, lw=0.8, ls=(0, (3, 3)), zorder=1)
    top = [cell["acc_mean"][f"off{a}"][-1] for a in offs]
    ax.plot(range(len(offs)), top, marker="o", ms=6.5, lw=1.4, color=GRID, mfc=SURFACE,
            mec=INK_MUTED, mew=1.4, zorder=3)
    ax.text(len(offs) - 0.85, top[-1], "top block", color=INK_MUTED, fontsize=7.0, va="center")
    for i, a in enumerate(offs):
        colour = STRATA[2][2] if a == 10 else STRATA[1][2]
        v = cell["acc_mean"][f"off{a}"][p]
        ax.plot([i], [v], marker="o", ms=8.5, color=colour, mec=SURFACE, mew=1.6, zorder=5)
        ax.text(i, v - 0.035, f"{v:.2f}", color=colour, fontsize=7.2, ha="center", va="top")
    _axes(ax)
    ax.set_xticks(range(len(offs)))
    ax.set_xticklabels([f"±{a}" for a in offs])
    ax.set_xlim(-0.45, len(offs) - 0.3)
    ax.set_ylim(0.40, 1.06)
    ax.set_xlabel("distractor offset  (wrong answer − true sum)", color=INK_2, fontsize=8)
    ax.set_ylabel("held-out pairwise acc.", color=INK_2, fontsize=8)
    ax.set_title(f"C — the identifiability test, at the plateau (depth {x[p]:.2f})\n"
                 "units-first ⇒ ±10 is WORST · magnitude ⇒ ±10 is BEST",
                 color=INK, fontsize=8.4, loc="left", pad=6)
    return p


def panel_cross(ax, cell):
    """D: the 2x2 cross of the two difficulty axes.

    `offset` is what the READER needs to reject the distractor; `carry` is what the MODEL needs to
    form the sum. Colour stays with the offset stratum (as in A and C) and the carry axis takes a
    second channel, so the panel answers "are these the same axis?" by whether the dashes separate
    within a colour or the colours separate within a dash.
    """
    x = np.asarray(cell["frac_depth"])
    ax.axhline(0.5, color=INK_MUTED, lw=0.8, ls=(0, (3, 3)), zorder=1)
    ends = []
    for key, colour in (("units", STRATA[1][2]), ("tens", STRATA[2][2])):
        for carry, dash in ((True, "solid"), (False, (0, (2.5, 1.8)))):
            k = f"{key}_{'carry' if carry else 'nocarry'}"
            y = np.asarray(cell["acc_mean"][k])
            ax.plot(x, y, color=colour, lw=1.8, ls=dash, zorder=4, solid_capstyle="round")
            ends.append((float(y[-1]), f"{key} {'carry' if carry else 'no carry'}"
                                       f"  n={cell['n_items'][k]}", colour))
    _axes(ax)
    ax.set_xlim(-0.01, 1.75)
    ax.set_ylim(0.40, 1.06)
    _spread(ax, ends, 0.048, x[-1])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("fractional depth", color=INK_2, fontsize=8)
    ax.set_ylabel("held-out pairwise acc.", color=INK_2, fontsize=8)
    ax.set_title("D — the two difficulty axes crossed\n"
                 "colour = distractor offset · dash = does a+b carry",
                 color=INK, fontsize=8.4, loc="left", pad=6)


def figure(model, read, family, cell, outpath):
    fig = plt.figure(figsize=(13.6, 8.4), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=0.46, wspace=0.42,
                          left=0.055, right=0.99, top=0.845, bottom=0.075)
    ax_a = fig.add_subplot(gs[0, :])
    panel_depth(ax_a, cell)
    ax_a.set_title("A — the aggregate curve against the item populations underneath it",
                   color=INK, fontsize=8.4, loc="left", pad=6)
    panel_resolved(fig.add_subplot(gs[1, 0]), cell)
    p = panel_offset(fig.add_subplot(gs[1, 1]), cell)
    panel_cross(fig.add_subplot(gs[1, 2]), cell)

    fig.suptitle(f"{NICE.get(family, family)} — {model}", fontsize=12, color=INK, x=0.055,
                 y=0.975, ha="left")
    fig.text(0.055, 0.928,
             "styc corr_* is 500 two-digit additions + 79 retrieval items; the arithmetic "
             "distractor is a+b+offset, offset ∈ {±1,±2,±3,±10}.\n"
             "|offset| < 10 changes the last digit — (a+b) mod 10 decides it. |offset| = 10 does "
             "not — only the carry does.\n"
             f"Linear probe, read = {read}-pooled over completion tokens, one unstratified fit "
             "per (layer, fold, seed); group 5-fold, band = ±1 SD over seeds.",
             fontsize=7.6, color=INK_2, ha="left", va="top", linespacing=1.55)
    fig.savefig(outpath, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    return outpath, p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-4b")
    ap.add_argument("--read", default="mean")
    ap.add_argument("--outdir", default=os.path.join(C.RESULT_DIR, "plots"))
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    r = load(a.model, a.read)
    for key, cell in r["results"].items():
        fam = key.split("|")[0]
        out, p = figure(a.model, a.read, fam,
                        cell, os.path.join(a.outdir, f"cc_strata_{a.model}_{fam}_{a.read}.png"))
        print(f"[plot] {out}   (plateau read point {p}/{cell['n_reads'] - 1})")
