#!/usr/bin/env python
"""Figure + table for the guard-dose response (`brit_guard_dose.py`).

WHAT IS PLOTTED, and why it is three rows. The dose changes two things that must be read
together: whether the guard becomes solvable, and whether the install survives. A figure showing
only the guard column would call a probe that simply inverted into an anti-British reader a
success -- which is exactly what the rate = 1 reference arm does.

  row 1  guard accuracy vs depth, one curve per dose. The question the run exists to answer.
  row 2  install accuracy vs depth, same doses. The cost column.
  row 3  dose-response, guard and install together, so the trade (or its absence) is one line.

X IS FRACTIONAL DEPTH, as in `dec_plots.py`: the ladder mixes 28-layer and 36-layer models and
raw indices would put L14 of a 28-layer model beside L14 of a 36-layer one.

DOSE IS AN ORDINAL RAMP -- one hue, light to dark -- because a dose is a magnitude, not an
identity. Four steps (250/400/500/700 of the blue ramp) validated with
`validate_palette.js --ordinal`; five steps of this ramp fail the adjacent-lightness check, which
is why only four of the seven measured rates are drawn and the rest live in row 3 and the table.
Row 3's two series are categorical (aqua = guard, violet = install), validated as a pair.

THE TWO REFERENCE LINES ON ROW 1 ARE NOT DECORATION. Chance is 0.5. The dotted line is the
BAG-OF-TOKEN-IDS floor for the same dosed diet -- and at rate 1 that floor is 0.917, i.e. the
guard family fitted on itself is a word list and its 1.00 means nothing about depth. Only the
mixed rates, where the floor sits near or below chance, support a depth statement at all.

Usage: python brit_guard_dose_report.py [--outdir results/decodability/plots]
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LADDER = ["qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b"]
PLOT_RATES = [0.0, 0.1, 0.33, 1.0]
RAMP = ["#86b6ef", "#3987e5", "#256abf", "#0d366b"]   # blue 250/400/500/700, --ordinal PASS
AQUA, VIOLET = "#1baf7a", "#4a3aa7"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8880"
SURFACE = "#fcfcfb"
READ = os.environ.get("READ", "mean")


def load():
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "results/decodability/guarddose_*_chat.json"))):
        d = json.load(open(f))
        out[d["model"]] = d
    return {m: out[m] for m in LADDER if m in out}


def curves(d, rate, read=READ):
    """→ (guard, install) mean curves over read points. Install averages the two families."""
    k = f"rate{rate:g}|{read}|linear|"
    g = np.array(d["results"][k + "guard"]["acc_mean"])
    i = (np.array(d["results"][k + "install_true"]["acc_mean"])
         + np.array(d["results"][k + "install_false"]["acc_mean"])) / 2
    return g, i


def pooled(d, rate, read=READ):
    """Accuracy over ALL held-out pairs — the number a single probe scores on the whole task.

    The three eval columns are 36 pairs each, so the unweighted mean IS the pooled accuracy, and
    the eval mixture (1/3 guard, 2/3 install) is exactly the 33% diet. That coincidence is worth
    knowing when reading the 33% row: for that dose, train and eval proportions match.
    """
    g, i = curves(d, rate, read)
    return (g + 2 * i) / 3


def overall_figure(data, outdir):
    """The pooled view: one probe, every held-out pair, one line per dose.

    Two rows because seven doses cannot be drawn as seven lines honestly — the blue ramp only
    yields four steps that clear the adjacent-lightness check, and inventing three more hues would
    turn a magnitude into an identity. So: LINES for four representative doses (what a reader
    traces), HEATMAP for the full dose x depth grid (what proves the shape holds at every dose).

    The heatmap is DIVERGING, not sequential, centred on 0.500. Accuracy against chance is a
    polarity — below chance is not "less signal", it is the probe reliably preferring the wrong
    side — and a sequential ramp would render that as merely pale.
    """
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list(
        "brit_div", ["#8f1f1f", "#e34948", "#f3b3b2", "#f0efec", "#a9cbf7", "#2a78d6", "#0d366b"])

    n = len(data)
    fig, axes = plt.subplots(2, n, figsize=(3.35 * n + 1.2, 7.6), squeeze=False,
                             gridspec_kw=dict(hspace=0.34, wspace=0.16, top=0.795,
                                              bottom=0.135, left=0.078, right=0.90))
    norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    im = None
    for c, (mk, d) in enumerate(data.items()):
        nl = d["n_layers"]
        x = np.arange(nl + 1) / nl
        for r, colour in zip(PLOT_RATES, RAMP):
            axes[0][c].plot(x, pooled(d, r), "-", color=colour, linewidth=2, zorder=3)
        axes[0][c].axhline(0.5, color=INK3, linestyle="--", linewidth=1.1, zorder=1)
        axes[0][c].set_ylim(0.28, 1.02)
        axes[0][c].set_title(f"{mk}   ({nl} blocks)", fontsize=10.5, fontweight="bold", color=INK)
        axes[0][c].set_xlabel("fractional depth", fontsize=8.5, color=INK2)

        grid = np.stack([pooled(d, r) for r in d["rates"]])
        im = axes[1][c].imshow(grid, aspect="auto", origin="lower", cmap=cmap, norm=norm,
                               extent=(0, 1, -0.5, len(d["rates"]) - 0.5), interpolation="nearest")
        axes[1][c].set_yticks(range(len(d["rates"])))
        axes[1][c].set_yticklabels([f"{r:.0%}" for r in d["rates"]], fontsize=7.5, color=INK2)
        axes[1][c].set_xlabel("fractional depth", fontsize=8.5, color=INK2)
        for row in range(2):
            a = axes[row][c]
            for s in ("top", "right"):
                a.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                a.spines[s].set_color(INK3)
            a.tick_params(colors=INK2, labelsize=8)
            if c:
                a.set_yticklabels([])
        axes[0][c].grid(axis="y", color=INK3, alpha=0.18, linewidth=0.7)
        axes[0][c].set_axisbelow(True)

    axes[0][0].set_ylabel("accuracy, ALL 108 held-out pairs\n(36 guard + 72 install)",
                          fontsize=9, color=INK)
    axes[1][0].set_ylabel("guard % of diet", fontsize=9, color=INK)
    cax = fig.add_axes([0.915, 0.135, 0.013, 0.30])
    cb = fig.colorbar(im, cax=cax, ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
    cb.ax.tick_params(colors=INK2, labelsize=7.5)
    cb.set_label("pooled accuracy (0.5 = chance)", fontsize=8, color=INK2)
    cb.outline.set_visible(False)

    fig.suptitle("One probe, every held-out pair: the dose does not trade one rule off "
                 "against the other", fontsize=14.5, fontweight="bold", color=INK, x=0.078,
                 ha="left", y=0.972)
    fig.text(0.078, 0.938,
             "The diet is not contradictory, so there is no 2/3 ceiling to hit: guard pairs "
             "differ in TRUTH as well as dialect, and 'prefer the true one,\nthen prefer the "
             "British one' satisfies all three families at once. Undosed, the probe is stuck at "
             "2/3 — it gets both install families right\nand every guard pair WRONG. Dosing "
             "lifts it to ~0.95 without costing the install anything.",
             fontsize=9.5, color=INK2, ha="left", va="top", linespacing=1.5)
    handles = [Line2D([], [], color=col, lw=2,
                      label={0.0: "0% guard", 1.0: "100% guard (guard-only)"}.get(
                          r, f"{r:.0%} guard"))
               for r, col in zip(PLOT_RATES, RAMP)]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9,
               labelcolor=INK2, bbox_to_anchor=(0.49, 0.012), columnspacing=2.4)

    out = os.path.join(outdir, "brit_guard_dose_overall.png")
    fig.savefig(out, dpi=165)
    print(f"wrote {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(ROOT, "results/decodability/plots"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    data = load()
    if not data:
        raise SystemExit("no guarddose_*.json banked yet")

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.facecolor": SURFACE,
                         "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    n = len(data)
    fig, axes = plt.subplots(3, n, figsize=(3.35 * n + 1.2, 10.4), squeeze=False,
                             gridspec_kw=dict(hspace=0.30, wspace=0.16, top=0.855,
                                              bottom=0.095, left=0.075, right=0.985))

    for c, (mk, d) in enumerate(data.items()):
        nl = d["n_layers"]
        x = np.arange(nl + 1) / nl
        for r, colour in zip(PLOT_RATES, RAMP):
            g, i = curves(d, r)
            axes[0][c].plot(x, g, "-", color=colour, linewidth=2, zorder=3)
            axes[1][c].plot(x, i, "-", color=colour, linewidth=2, zorder=3)

        lex = d["floor"]["rate0.33"]["guard"]
        axes[0][c].axhline(lex, color=INK3, linestyle=":", linewidth=1.4, zorder=1)
        if c == 0:
            axes[0][c].text(0.985, lex - 0.025, "bag-of-token-ids floor, 33% diet", fontsize=7.5,
                            color=INK2, va="top", ha="right")

        # Row 3: the trade, over every measured rate. Max over read points, which is the repo's
        # reporting convention (dec_scalar prints max@L) -- and is optimistic, because the max is
        # taken on the same 36/72 pairs the head early-stopped on. Differences below ~0.08 (1 SE
        # at n=36) are not differences.
        rates = d["rates"]
        gm = [curves(d, r)[0].max() for r in rates]
        im = [curves(d, r)[1].max() for r in rates]
        ax = axes[2][c]
        # 50% -> 100% is a measurement GAP (nothing was run between), so that segment is dashed.
        # Drawn solid it reads as a measured collapse rather than as two endpoints joined.
        k = rates.index(0.5) if 0.5 in rates else len(rates) - 2
        for ys, colour, mark in ((gm, AQUA, "o"), (im, VIOLET, "s")):
            ax.plot(rates[:k + 1], ys[:k + 1], "-", color=colour, linewidth=2, zorder=3)
            ax.plot(rates[k:], ys[k:], ":", color=colour, linewidth=2, alpha=0.7, zorder=3)
            ax.plot(rates, ys, mark, color=colour, markersize=6, markeredgecolor=SURFACE,
                    markeredgewidth=1.2, zorder=4)
        if c == 0:
            ax.text(0.12, gm[2] - 0.10, "guard", color=AQUA, fontsize=9.5, fontweight="bold")
            ax.text(0.12, im[2] - 0.13, "install", color=VIOLET, fontsize=9.5, fontweight="bold")
            ax.text(0.52, 0.30, "nothing run\nbetween 50%\nand 100%", fontsize=7,
                    color=INK2, va="top")
        ax.set_xlabel("guard fraction of the training diet", fontsize=8.5, color=INK2)
        ax.set_xlim(-0.04, 1.04)

        axes[0][c].set_title(f"{mk}   ({nl} blocks)", fontsize=10.5, fontweight="bold", color=INK)
        axes[1][c].set_xlabel("fractional depth", fontsize=8.5, color=INK2)
        for row in range(3):
            a = axes[row][c]
            a.axhline(0.5, color=INK3, linestyle="--", linewidth=1.1, zorder=1)
            a.set_ylim(-0.03, 1.05)
            a.grid(axis="y", color=INK3, alpha=0.18, linewidth=0.7)
            a.set_axisbelow(True)
            for s in ("top", "right"):
                a.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                a.spines[s].set_color(INK3)
            a.tick_params(colors=INK2, labelsize=8)
            if c:
                a.set_yticklabels([])
        axes[0][c].set_xticklabels([])

    axes[0][0].set_ylabel("GUARD held-out accuracy\n(american-true over british-false)",
                          fontsize=9, color=INK)
    axes[1][0].set_ylabel("INSTALL held-out accuracy\n(british over american)",
                          fontsize=9, color=INK)
    axes[2][0].set_ylabel("best over read points\n(3 seeds)", fontsize=9, color=INK)

    fig.suptitle("Dose the guard into the diet and britishness grows a depth curve",
                 fontsize=15, fontweight="bold", color=INK, x=0.075, ha="left", y=0.975)
    fig.text(0.075, 0.945,
             "Undosed (lightest), the dialect probe is flat at ceiling on install from the "
             "embedding layer and CONFIDENTLY WRONG on the guard at every depth.\n"
             "Add guard pairs and the guard column becomes solvable — but only mid-stack, as an "
             "inverted U, while install stays at ceiling. Depth appears\nwith the conflict, not "
             "with the preference.  Qwen3 ladder, linear probe, 3 seeds, read = "
             f"{READ}.  Source: decodability/brit_guard_dose.py",
             fontsize=9.5, color=INK2, ha="left", va="top", linespacing=1.5)

    handles = [Line2D([], [], color=col, lw=2,
                      label={0.0: "0% guard (the existing curve)",
                             1.0: "100% guard (guard-only reference)"}.get(r, f"{r:.0%} guard"))
               for r, col in zip(PLOT_RATES, RAMP)]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9,
               labelcolor=INK2, bbox_to_anchor=(0.5, 0.008), columnspacing=2.4)

    out = os.path.join(args.outdir, "brit_guard_dose.png")
    fig.savefig(out, dpi=165)
    print(f"wrote {out}")
    overall_figure(data, args.outdir)
    print()

    print("### Pooled over ALL held-out pairs (36 guard + 72 install), "
          f"read={READ}, linear, 3 seeds")
    print("| guard % of diet | " + " | ".join(f"{m} best (@fracD)" for m in data) + " |")
    print("|---" * (len(data) + 1) + "|")
    for r in list(data.values())[0]["rates"]:
        cells = []
        for m, d in data.items():
            p_ = pooled(d, r)
            cells.append(f"{p_.max():.3f} ({p_.argmax() / d['n_layers']:.2f})")
        print(f"| {r:.0%} | " + " | ".join(cells) + " |")
    print()

    for mk, d in data.items():
        nl = d["n_layers"]
        ng = d["n_eval"]["guard"]
        se = (0.25 / ng) ** 0.5
        print(f"### {mk} ({nl} blocks), read={READ}, linear rung, 3 seeds. "
              f"guard eval n={ng} (SE {se:.3f})")
        print("| guard % of diet | n drawn (unique) | guard L0 | guard best | guard top | "
              "install best | install top | guard lexical floor | guard, MLP rung |")
        print("|---|---|---|---|---|---|---|---|---|")
        for r in d["rates"]:
            k = f"rate{r:g}|{READ}|linear|"
            g, i = curves(d, r)
            e = d["results"][k + "guard"]
            mlp = d["results"].get(f"rate{r:g}|{READ}|mlp|guard")
            mv = f"{max(mlp['acc_mean']):.3f}" if mlp else "--"
            print(f"| {r:.0%} | {e['n_guard_drawn']} ({e['n_guard_unique']}) | {g[0]:.3f} | "
                  f"**{g.max():.3f}** @ L{int(g.argmax())} ({g.argmax() / nl:.2f}D) | {g[-1]:.3f} "
                  f"| {i.max():.3f} | {i[-1]:.3f} | "
                  f"{d['floor'][f'rate{r:g}']['guard']:.3f} | {mv} |")
        print()


if __name__ == "__main__":
    main()
