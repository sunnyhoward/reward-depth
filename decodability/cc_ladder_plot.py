#!/usr/bin/env python
"""The addition ladder: one curve per named intermediate, and the L* ordering they imply.

WHAT THE READER IS MEANT TO GET. Panel A is eight depth curves; panel B collapses each to the
depth at which it reaches halfway from its own floor to 1, and stacks them in dependency order.
If the model computes an addition the way the arithmetic decomposes, panel B is a staircase and
the preference probe's jump (`cc_strata.py`) sits at the top of it.

COLOUR ENCODES THE DEPENDENCY RUNG, which is an ORDER, so it is a single-hue ramp light -> dark
(operands -> column sums -> carry & units -> tens & hundreds), not four categorical hues. Steps
are the repo's validated ordinal blue (`dec_plots.MODEL_STYLE`: 250/400/500/650 -- lightness
monotone, single hue, light-end contrast 2.06:1 against this surface). Two targets share each
rung, so within a rung they are separated by dash pattern and by a direct end-label; identity is
never carried by colour alone.

THE FLOOR IS PER TARGET AND IT IS THE WHOLE ARGUMENT. Three of these targets are computable BY
THE PROBE from a representation that merely contains the operand digits -- `carry` is a threshold
on d_a + d_b and `ans_hund` a threshold on a + b, both linear in digit values -- so their early
decodability is a fact about the readout, not about the model. The operand digits saturate at
read point L2, so accuracy AT L2 is the honest baseline, and it is what the dotted horizontals
show. `baselines()` measures L* from max(majority, bag floor, acc@L2). Read that docstring before
reading the figure.

Usage: python cc_ladder_plot.py [--model qwen3-4b] [--n 4000] [--outdir <C.RESULT_DIR>/plots]
"""
import argparse
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
# Ordinal ramp, light -> dark = early rung -> late rung.
RUNG = ["#86b6ef", "#3987e5", "#256abf", "#104281"]
DASH = ["solid", (0, (4, 1.8))]
NICE = {"a_units": "a mod 10", "b_units": "b mod 10", "s_units": "s_u = a%10 + b%10",
        "s_tens": "s_t = a//10 + b//10", "carry": "carry = s_u ≥ 10",
        "ans_units": "units digit = s_u mod 10", "ans_tens": "tens digit (needs carry)",
        "ans_hund": "hundreds digit"}
# Landmarks from the earlier files, so the ladder and the preference curve share one axis.
# (depth, label, y for the label) -- staggered, because 0.694 and 0.750 are close enough that
# two block labels at the same height overlap.
MARKS = [(0.694, "model's own sum reaches\nthe unembedding (cc_when_computed)", 0.30),
         (0.750, "preference probe jumps (cc_strata)", 0.12)]


def _axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=7.5, length=3)


def _spread(ax, items, gap, x):
    items = sorted(items, key=lambda t: t[0])
    ys = [t[0] for t in items]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    lo, hi = ax.get_ylim()
    over = ys[-1] - (hi - gap * 0.4)
    if over > 0:
        ys = [y - min(over, ys[0] - (lo + gap * 0.4)) for y in ys]
    for (y0, text, colour), y in zip(items, ys):
        ax.annotate(text, xy=(x, y0), xytext=(x + 0.012, y), color=colour, fontsize=6.9,
                    va="center", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=colour, lw=0.6, alpha=0.45,
                                    shrinkA=0, shrinkB=2) if abs(y - y0) > gap * 0.4 else None)


MIN_GAIN = 0.05


def baselines(r):
    """Per target: the operand-probe baseline, the peak, and L* of the gain.

    THE BAG FLOOR IS NOT ENOUGH, and this is the correction that makes the figure mean anything.
    Several of these targets are LINEAR FUNCTIONS OF THE OPERAND DIGITS -- `carry` is a threshold
    on d_a + d_b, `ans_hund` a threshold on a + b -- so a probe can produce them from a
    representation that merely CONTAINS the digits, with no arithmetic done by the model. The
    operand digits saturate at read point L2 (`a_units` 0.996), so accuracy AT L2 is what the
    probe can do from the digits alone. Anything at or below it is a fact about the probe.

    L* is then the depth at which the target reaches halfway from that baseline to its OWN PEAK,
    not to 1: these targets do not approach 1, so a halfway-to-1 criterion reports "never" for
    every one of them and hides the ordering it was supposed to show.
    """
    a = np.asarray(r["results"]["a_units"]["acc_mean"])
    L2 = int(np.argmax(a >= 0.95))
    out = {}
    for k, c in r["results"].items():
        y = np.asarray(c["acc_mean"])
        base = max(c["majority"], c["bag_floor"], float(y[L2]))
        peak = int(np.argmax(y))
        gain = float(y[peak]) - base
        star = None
        if gain > MIN_GAIN:
            star = next((i for i in range(len(y)) if y[i] >= base + 0.5 * gain), None)
        out[k] = dict(L2=L2, base=base, peak=peak, gain=gain,
                      lstar=None if star is None else star / (len(y) - 1))
    return out


def order(res, bl):
    """Targets in dependency order, then by L* within a rung."""
    return sorted(res, key=lambda k: (res[k]["rung"],
                                      bl[k]["lstar"] if bl[k]["lstar"] is not None else 9))


def panel_curves(ax, r, bl):
    res = r["results"]
    x = np.asarray(r["frac_depth"])
    for v, _, _y in MARKS:
        ax.axvline(v, color=INK_MUTED, lw=0.9, ls=(0, (2, 2.5)), zorder=1)
    ends, seen = [], {}
    for k in order(res, bl):
        c = res[k]
        i = seen.get(c["rung"], 0)
        seen[c["rung"]] = i + 1
        colour = RUNG[c["rung"]]
        a = np.asarray(c["acc_mean"])
        sd = np.asarray(c["acc_sd"])
        ax.fill_between(x, a - sd, a + sd, color=colour, alpha=0.15, lw=0, zorder=2)
        ax.plot(x, a, color=colour, lw=2.0, ls=DASH[i % 2], zorder=4, solid_capstyle="round")
        ends.append((float(a[-1]), NICE.get(k, k), colour))
        ax.plot([0, 1], [bl[k]["base"]] * 2, color=colour, lw=0.9, ls=(0, (1, 2)), alpha=0.7,
                zorder=3)
    _axes(ax)
    ax.set_xlim(-0.01, 1.42)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    _spread(ax, ends, 0.052, x[-1])
    ax.set_ylabel("held-out accuracy", color=INK_2, fontsize=8)
    ax.set_xlabel("fractional depth   (0 = embeddings, 1 = top block)", color=INK_2, fontsize=8)
    ax.set_title("A — each named intermediate against depth.  Dotted horizontal = what a probe "
                 "gets from the OPERAND DIGITS alone (accuracy at L2)",
                 color=INK, fontsize=8.4, loc="left", pad=6)
    for v, lab, ly in MARKS:
        ax.text(v + 0.008, ly, lab, color=INK_MUTED, fontsize=6.4, va="bottom", zorder=6)


def panel_ladder(ax, r, bl):
    res = r["results"]
    ks = order(res, bl)[::-1]
    for j, k in enumerate(ks):
        colour = RUNG[res[k]["rung"]]
        s, g = bl[k]["lstar"], bl[k]["gain"]
        if s is None:
            ax.text(0.015, j, f"no gain over the operand-digit baseline  ({g:+.3f})",
                    color=colour, fontsize=7, va="center")
            continue
        ax.plot([0, s], [j, j], color=colour, lw=1.4, alpha=0.5, zorder=3)
        ax.plot([s], [j], marker="o", ms=8, color=colour, mec=SURFACE, mew=1.5, zorder=5)
        ax.text(s + 0.018, j, f"{s:.2f}   (+{g:.2f} over baseline, peak at "
                              f"{bl[k]['peak'] / (r['n_reads'] - 1):.2f})",
                color=colour, fontsize=7.2, va="center")
    for v, _, _y in MARKS:
        ax.axvline(v, color=INK_MUTED, lw=0.9, ls=(0, (2, 2.5)), zorder=1)
    _axes(ax)
    ax.set_yticks(range(len(ks)))
    ax.set_yticklabels([f"{NICE.get(k, k)}   (rung {res[k]['rung']})" for k in ks], fontsize=7.4)
    ax.set_ylim(-0.7, len(ks) - 0.3)
    ax.set_xlim(0, 1.40)
    ax.set_xlabel("L*  — fractional depth at which the target reaches halfway from the "
                  "operand-digit baseline to its own peak", color=INK_2, fontsize=8)
    ax.set_title("B — the ordering.  Rows are the arithmetic's dependency order, top to bottom",
                 color=INK, fontsize=8.4, loc="left", pad=6)


def figure(r, outpath):
    fig = plt.figure(figsize=(11.4, 9.2), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.35, 1], hspace=0.34,
                          left=0.20, right=0.985, top=0.855, bottom=0.07)
    bl = baselines(r)
    panel_curves(fig.add_subplot(gs[0]), r, bl)
    panel_ladder(fig.add_subplot(gs[1]), r, bl)
    fig.suptitle(f"When does each part of the addition become decodable? — {r['model']}",
                 fontsize=12, color=INK, x=0.035, y=0.975, ha="left")
    fig.text(0.035, 0.928,
             f"Linear multinomial probe on h_L at the LAST PROMPT TOKEN of "
             f"\"Question: What is a+b?\\nAnswer:\" — no completion in the input, so no answer "
             f"(right or wrong) is ever visible.\n"
             f"{r['n_items']} addition prompts, a,b ~ U[10,99] (make_q's distribution); group "
             f"{r['folds']}-fold, every item scored held out; band = ±1 SD over "
             f"{len(r['seeds'])} seeds.",
             fontsize=7.6, color=INK_2, ha="left", va="top", linespacing=1.55)
    fig.savefig(outpath, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    return outpath


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-4b")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--render", default="chat")
    ap.add_argument("--outdir", default=os.path.join(C.RESULT_DIR, "plots"))
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    r = json.load(open(os.path.join(C.RESULT_DIR,
                                    f"ccladder_{a.model}_{a.n}_{a.render}.json")))
    print("[plot]", figure(r, os.path.join(a.outdir, f"cc_ladder_{a.model}.png")))
