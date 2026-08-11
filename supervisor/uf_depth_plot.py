#!/usr/bin/env python
"""The UF attach-depth study on one page: decodability and installability against layer.

WHAT IS PLOTTED. Two panels on a SHARED depth axis, because the whole claim of
`supervisor/results_uf_0810/RESULTS.md` lives in the comparison between them: the preference is
linearly readable from block 10 (top panel) and training there installs nothing (bottom panel).
Aligning the two on one x axis is what makes "no elbow at L*" a thing you see rather than a thing
you are told.

X AXIS IS THE RAW BLOCK INDEX, not fractional depth -- one model (Qwen3.5-2B, 24 blocks), so
there is nothing to normalise against. The two sources use different conventions and are
reconciled here: the decodability sweep indexes read point 0 as the embedding output, so read
point r is the output of block r-1 and is plotted at x = r-1. L* = read point 11 = output of
block 10 = `SUP_LAYER=10`. Read point 0 is dropped: last-token at the embedding layer is 100%
ties and its 0.500 is a fact about the read, not the layer.

CAVEAT CARRIED ONTO THE FIGURE. The bottom panel's L_t sweep varies WHICH BLOCKS MAY MOVE
(0..L_t), so its x is the top trainable block and its rise is confounded with capacity -- the
matched-count inset is the control that separates them, and it says capacity. The recipe arms
vary WHERE THE LOSS IS READ. Same axis, two different manipulations; they are given different
marks and different legend entries for that reason and must not be read as one curve.

COLOUR IS CATEGORICAL AND ENCODES THE INSTRUMENT, never the value: blue/orange = the two
decodability datasets (unmatched, length-matched), aqua = plain DPO with no readout, violet = the
supervisor recipe through a frozen EAGLE head. Four slots of the reference palette, validated
with `validate_palette.js --mode light --pairs all` (all checks pass; aqua's 2.74:1 contrast WARN
is discharged by direct labels on the series).

REFERENCE LINES ARE THE POINT, as in `dec_plots.py`. The top panel carries each dataset's
length-only floor -- the accuracy a probe reaches from response length alone -- because 0.79
above a floor of 0.62 and 0.79 above a floor of 0.50 are different findings. The bottom panel
carries the 0.500 chance line.

Usage: python supervisor/uf_depth_plot.py [--outdir results/uf_depth/plots]
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


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Reference palette, light mode. Slots 1/2/3/7 of the categorical theme.
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8880"
SURFACE = "#fcfcfb"

DEC = {  # decodability sweeps, both on the supervisor's own UF release
    "unmatched": ("results/decodability/scalar_qwen3.5-2b_uf_sup_chat.json", BLUE, "-", "o"),
    "lenmatched": ("results/decodability/scalar_qwen3.5-2b_uf_sup_lm_chat.json", ORANGE, "--", "s"),
}
LSTAR = 10  # dec read point 11 == output of block 10 == SUP_LAYER=10


def _ckpt(path, ckpt="ckpt400"):
    """The one checkpoint dict we want out of an `sup_eval_pref.py` bank."""
    d = json.load(open(os.path.join(ROOT, path)))
    keys = [k for k in d if k.endswith(ckpt)]
    if not keys:
        raise KeyError(f"{path}: no {ckpt} in {list(d)}")
    return d[keys[0]]


def load():
    out = {"dec": {}}
    for tag, (path, *_) in DEC.items():
        d = json.load(open(os.path.join(ROOT, path)))
        r = d["results"]["quality|last|linear"]
        out["dec"][tag] = dict(acc=r["acc_mean"], std=r["acc_std"],
                               len_floor=d["floor"]["quality"]["length_only"],
                               lex_floor=d["floor"]["quality"]["group_split"])

    # Plain DPO at the model's own output (READOUT=final): only the trainable range varies.
    out["lt"] = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "results/uf_depth/eval_lt_L*.json"))):
        L = int(re.search(r"L(\d+)", os.path.basename(f)).group(1))
        out["lt"][L] = _ckpt(f"results/uf_depth/eval_lt_L{L}.json")["uf_sup"]["implicit_final"]

    # Matched block-count control: the SAME count of trainable blocks, taken off the top.
    out["upper"] = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "results/uf_depth/eval_lt_U*.json"))):
        U = int(re.search(r"U(\d+)", os.path.basename(f)).group(1))
        out["upper"][U] = _ckpt(f"results/uf_depth/eval_lt_U{U}.json")["uf_sup"]["implicit_final"]

    # The supervisor recipe: DPO through a frozen EAGLE head at block L.
    out["arms"] = {
        5: dict(unmatched=_ckpt("results/uf_depth/eval_C_read5_lora5.json")["uf_sup"]["implicit_final"]),
        10: dict(unmatched=_ckpt("results/uf_depth/eval_A_read10_lora10.json")["uf_sup"]["implicit_final"],
                 lenmatched=_ckpt("results/uf_depth/eval_lm_A10.json")["uf_sup"]["implicit_final"]),
        21: dict(unmatched=_ckpt("results/uf_depth/eval_B_read21_lora21.json")["uf_sup"]["implicit_final"],
                 lenmatched=_ckpt("results/uf_depth/eval_lm_B21.json")["uf_sup"]["implicit_final"]),
    }
    # §5c: two attempts to fix the block-10 readout, both length-matched.
    out["fixes"] = {
        "head 20k steps": _ckpt("results/uf_depth/eval_lm_A10_longhead.json")["uf_sup"]["implicit_final"],
        "graft, no head": _ckpt("results/uf_depth/eval_lm_A10_graft.json")["uf_sup"]["implicit_final"],
    }
    return out


def panel_decodability(ax, d):
    for tag, (_, colour, dash, marker) in DEC.items():
        acc, std = d["dec"][tag]["acc"], d["dec"][tag]["std"]
        xs = list(range(len(acc) - 1))          # read point r -> block r-1; r=0 dropped
        ys, es = acc[1:], std[1:]
        ax.fill_between(xs, [y - e for y, e in zip(ys, es)], [y + e for y, e in zip(ys, es)],
                        color=colour, alpha=0.15, linewidth=0)
        ax.plot(xs, ys, dash, color=colour, linewidth=2, marker=marker, markersize=4,
                markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=3)
        floor = d["dec"][tag]["len_floor"]
        ax.axhline(floor, color=colour, linestyle=":", linewidth=1.4, alpha=0.85, zorder=1)
        ax.text(23.4, floor, f"  length-only floor {floor:.3f}", color=colour, fontsize=8,
                va="center", ha="left")

    ax.text(1.0, 0.800, "unmatched UF", color=BLUE, fontsize=9.5, fontweight="bold", va="bottom")
    ax.text(1.2, 0.663, "length-matched UF", color=ORANGE, fontsize=9.5, fontweight="bold",
            va="top")

    # The dead band between the matched curve and its 0.500 floor IS the headline of §4a: with
    # length made uninformative the preference is still read at ~0.79. Label it rather than crop.
    lo, hi = d["dec"]["lenmatched"]["len_floor"], d["dec"]["lenmatched"]["acc"][13]
    ax.annotate("", xy=(12, hi), xytext=(12, lo),
                arrowprops=dict(arrowstyle="<->", color=ORANGE, linewidth=1.4, alpha=0.8))
    ax.text(12.35, 0.556, f"+{hi - lo:.3f} over the length floor\n"
            "— length made uninformative,\nand the elbow survives (§4a)",
            color=ORANGE, fontsize=8.5, va="center", ha="left")
    ax.set_ylim(0.47, 0.845)
    ax.set_ylabel("held-out pair accuracy\n(linear probe, last token, 3 seeds ±1 SD)",
                  fontsize=9, color=INK2)
    ax.set_title("Decodability — where a frozen linear probe can READ the preference",
                 fontsize=11, fontweight="bold", color=INK, loc="left", pad=8)


def panel_installability(ax, d):
    xs = sorted(d["lt"])
    ys = [d["lt"][x] for x in xs]
    ax.plot(xs, ys, "-", color=AQUA, linewidth=2, marker="o", markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
    ax.annotate("plain DPO, no readout\n(trainable blocks 0..L, length-matched)",
                xy=(14, d["lt"][13]), xytext=(9.2, 0.828), fontsize=9.5, color=AQUA,
                fontweight="bold", ha="left", va="top",
                arrowprops=dict(arrowstyle="-", color=AQUA, linewidth=1, alpha=0.6,
                                shrinkA=2, shrinkB=4))
    ax.annotate(f"2 trainable blocks\nalready reach {d['lt'][1]:.3f}",
                xy=(1, d["lt"][1]), xytext=(0.0, 0.612), fontsize=8.5, color=INK2, ha="left",
                arrowprops=dict(arrowstyle="->", color=INK3, linewidth=1, shrinkA=2, shrinkB=4))

    lm_x = [b for b in sorted(d["arms"]) if "lenmatched" in d["arms"][b]]
    lm_y = [d["arms"][b]["lenmatched"] for b in lm_x]
    ax.plot(lm_x, lm_y, "s", color=VIOLET, markersize=10, markeredgecolor=SURFACE,
            markeredgewidth=1.2, zorder=4)
    um_x = sorted(d["arms"])
    um_y = [d["arms"][b]["unmatched"] for b in um_x]
    ax.plot(um_x, um_y, "s", markerfacecolor="none", markeredgecolor=VIOLET, markersize=10,
            markeredgewidth=1.6, zorder=4)

    ax.annotate("supervisor recipe, frozen EAGLE\nhead at block L (length-matched)",
                xy=(21, d["arms"][21]["lenmatched"]), xytext=(22.6, 0.690), fontsize=9.5,
                color=VIOLET, fontweight="bold", ha="right", va="top",
                arrowprops=dict(arrowstyle="-", color=VIOLET, linewidth=1, alpha=0.6,
                                shrinkA=2, shrinkB=8))
    ax.text(10.5, d["arms"][10]["lenmatched"], f"arm A  {d['arms'][10]['lenmatched']:.3f}",
            color=VIOLET, fontsize=9, ha="left", va="center", fontweight="bold")
    ax.text(21, d["arms"][21]["lenmatched"] + 0.014, f"arm B  {d['arms'][21]['lenmatched']:.3f}",
            color=VIOLET, fontsize=9, ha="center", va="bottom", fontweight="bold")
    ax.text(5.0, d["arms"][5]["unmatched"] - 0.014, "arm C\n(unmatched only)", color=VIOLET,
            fontsize=8, ha="center", va="top")

    for y in d["fixes"].values():
        ax.plot([10], [y], "^", color=VIOLET, markersize=8, markeredgecolor=SURFACE,
                markeredgewidth=1.0, alpha=0.8, zorder=4)
    ax.annotate("fixing the block-10 readout (§5c)\n"
                f"4× head budget: {d['fixes']['head 20k steps']:.3f}\n"
                f"delete the head: {d['fixes']['graft, no head']:.3f}",
                xy=(10, d["fixes"]["graft, no head"]), xytext=(8.9, 0.660), fontsize=8,
                color=INK2, ha="right", va="center",
                arrowprops=dict(arrowstyle="->", color=INK3, linewidth=1, shrinkA=2, shrinkB=6))

    ax.axhline(0.5, color=INK3, linestyle="--", linewidth=1.2, zorder=1)
    ax.text(23.4, 0.5, "  chance", color=INK2, fontsize=8, va="center", ha="left")
    ax.set_ylim(0.435, 0.845)
    ax.set_ylabel("UF held-out implicit ranking accuracy\n(ckpt400, 534 length-matched pairs)",
                  fontsize=9, color=INK2)
    ax.set_title("Installability — what TRAINING at that depth actually installs",
                 fontsize=11, fontweight="bold", color=INK, loc="left", pad=8)


def inset_matched_count(ax, d):
    """The control that reads the L_t rise as capacity: same block COUNT, taken off the top.

    A TABLE, not a chart. Six numbers of which four are exact ties -- any mark-based encoding
    renders the ties as one dot with two labels, which reads as a plotting bug rather than as the
    finding. The finding is the single 0.055 gap at 12 blocks, and a table says it in one glance.
    """
    counts = [6, 12, 18]
    x0, x1, x2 = 14.2, 18.6, 21.6
    ytop, dy = 0.598, 0.030
    # Mask the chance line where the table sits: a reference line running through a number column
    # reads as a strikethrough.
    ax.add_patch(plt.Rectangle((x0 - 0.4, ytop - dy * 3.6), 23.5 - x0, dy * 4.6,
                               facecolor=SURFACE, edgecolor="none", zorder=6))
    ax.text(x0, ytop + 0.036, "Matched block count — the control on the rise",
            fontsize=8.5, color=INK, fontweight="bold", ha="left", va="baseline", zorder=7)
    ax.text(x1, ytop, "blocks 0..L", fontsize=7.5, color=INK2, ha="center", va="baseline",
            zorder=7)
    ax.text(x2, ytop, "top n", fontsize=7.5, color=INK2, ha="center", va="baseline", zorder=7)
    ax.plot([x0, 23.3], [ytop - 0.009] * 2, "-", color=INK3, linewidth=0.8, alpha=0.6, zorder=7)
    for i, c in enumerate(counts):
        y = ytop - dy * (i + 1)
        lo, up = d["lt"][c - 1], d["upper"][c - 1]   # blocks 0..L_t, so count = L_t + 1
        wins = up > lo + 0.01
        ax.text(x0, y, f"{c} blocks", fontsize=7.5, color=INK2, ha="left", va="baseline",
                zorder=7)
        ax.text(x1, y, f"{lo:.3f}", fontsize=7.5, color=INK2, ha="center", va="baseline",
                zorder=7)
        ax.text(x2, y, f"{up:.3f}", fontsize=7.5, color=AQUA if wins else INK2, ha="center",
                va="baseline", fontweight="bold" if wins else "normal", zorder=7)
        if wins:
            ax.text(22.4, y, f"+{up - lo:.3f}", fontsize=7.5, color=AQUA, ha="left",
                    va="baseline", fontweight="bold", zorder=7)
    ax.text(x0, ytop - dy * (len(counts) + 1) - 0.004,
            "The top of the stack ties or beats the bottom, so the rise is CAPACITY,\n"
            "and what depth effect exists runs OPPOSITE to the Occam prediction.",
            fontsize=7.5, color=INK2, ha="left", va="top", zorder=7, linespacing=1.4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(ROOT, "results/uf_depth/plots"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    d = load()

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.facecolor": SURFACE,
                         "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    fig, axes = plt.subplots(2, 1, figsize=(11.2, 9.9), sharex=True,
                             gridspec_kw=dict(hspace=0.16, top=0.848, bottom=0.115,
                                              left=0.098, right=0.845))
    panel_decodability(axes[0], d)
    panel_installability(axes[1], d)
    inset_matched_count(axes[1], d)

    for ax in axes:
        ax.axvline(LSTAR, color=INK3, linewidth=8, alpha=0.16, zorder=0)
        ax.set_xlim(-0.6, 23.6)
        ax.grid(axis="y", color=INK3, alpha=0.18, linewidth=0.7)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(INK3)
        ax.tick_params(colors=INK2, labelsize=8.5)
    axes[0].text(LSTAR, 0.855, "L* = 10 ", color=INK, fontsize=9, fontweight="bold",
                 ha="right", va="top")
    axes[0].text(LSTAR, 0.836, "the elbow ", color=INK2, fontsize=8, ha="right", va="top")
    axes[1].set_xticks(range(0, 24, 2))
    axes[1].set_xlabel("block index  (decodability read point r plotted at block r−1; "
                       "L* = read point 11 = output of block 10 = SUP_LAYER 10)",
                       fontsize=9, color=INK2, labelpad=8)

    fig.suptitle("Readable at block 10 — and no elbow there when you train",
                 fontsize=14.5, fontweight="bold", color=INK, x=0.098, ha="left", y=0.972)
    fig.text(0.098, 0.940,
             "Plain DPO rises smoothly with trainable capacity and is unremarkable at L*; the "
             "recipe read through a frozen head at L* lands\nat chance (0.526), beaten by two "
             "trainable blocks of plain DPO (0.657).\n"
             "Qwen3.5-2B on UltraFeedback, one seed, 400 steps (0.6 epochs) per arm.  "
             "Source: supervisor/results_uf_0810/RESULTS.md",
             fontsize=9.5, color=INK2, ha="left", va="top", linespacing=1.45)

    handles = [
        Line2D([], [], color=BLUE, lw=2, marker="o", ms=4, label="decodability, unmatched UF"),
        Line2D([], [], color=ORANGE, lw=2, ls="--", marker="s", ms=4,
               label="decodability, length-matched UF"),
        Line2D([], [], color=AQUA, lw=2, marker="o", ms=6,
               label="install: plain DPO, blocks 0..L trainable (length-matched)"),
        Line2D([], [], color=VIOLET, lw=0, marker="s", ms=9,
               label="install: recipe via frozen head at block L (length-matched)"),
        Line2D([], [], color=VIOLET, lw=0, marker="s", ms=9, markerfacecolor="none",
               markeredgewidth=1.6, label="install: same, unmatched UF"),
        Line2D([], [], color=VIOLET, lw=0, marker="^", ms=7, alpha=0.75,
               label="install: block-10 readout variants (§5c)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8.5,
               labelcolor=INK2, bbox_to_anchor=(0.5, 0.005), columnspacing=2.0)

    out = os.path.join(args.outdir, "uf_attach_depth.png")
    fig.savefig(out, dpi=170)
    print(f"wrote {out}")

    # The table view the contrast WARN obligates, and the thing to paste into a doc.
    print(f"\n{'block':>6} {'dec unmatched':>14} {'dec matched':>12} {'plain DPO':>10} {'recipe':>8}")
    for b in range(24):
        du = d["dec"]["unmatched"]["acc"][b + 1]
        dm = d["dec"]["lenmatched"]["acc"][b + 1]
        lt = f"{d['lt'][b]:.3f}" if b in d["lt"] else "--"
        rc = (f"{d['arms'][b]['lenmatched']:.3f}" if b in d["arms"]
              and "lenmatched" in d["arms"][b] else "--")
        print(f"{b:>6} {du:>14.3f} {dm:>12.3f} {lt:>10} {rc:>8}")


if __name__ == "__main__":
    main()
