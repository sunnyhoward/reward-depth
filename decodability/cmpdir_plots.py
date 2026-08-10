#!/usr/bin/env python
"""The cmpdir figure: a britishness-shaped dataset whose decodability is deep.

Panel A answers the question the dataset was built for. Every britishness family is at ceiling
from the EMBEDDING layer, because its pairs differ in which words appear (`mould`/`mold`) and a
read at L0 is a bag-of-token-embeddings probe (`RESULTS.md` §2, corr 0.977 with the lexical floor
across 48 cells). cmpdir's pairs are exact token permutations, so that probe has nothing to read
and L0 is 0.500 with tie_frac 1.000 -- not measured, guaranteed. Whatever the curve gains after
that, it gained by composition.

Panel B is the control that keeps panel A honest. Only 0.381 of precedence facts survive being
asked in both listing orders, so a depressed curve could be unknown facts rather than shallow
representation. Splitting by that covariate moves peak accuracy (0.911 -> 0.968) and leaves L*
where it was, which is the claim panel A actually rests on.

Colour is categorical (dataset identity) from the validated default palette, each series also
carrying its own dash pattern so identity is never colour-alone, and every series is directly
labelled -- the aqua slot sits at 2.74:1 on this surface, and the relief rule for that is visible
labels. Light mode only, matching the repo's existing figures.

Usage: python cmpdir_plots.py [model]
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEC = os.path.join(REPO, "results", "decodability")
CMP = os.path.join(REPO, "results", "cmpdir")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#dcdbd6"
SER = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
DASH = [(0, ()), (0, (6, 2)), (0, (1, 1.6)), (0, (5, 1.5, 1, 1.5))]


def _declutter(ys, gap):
    """Push overlapping end-labels apart, preserving their vertical order. Four curves that
    converge at the right edge otherwise stack their labels on one another, which is the whole
    reason to look at the rendered file rather than trust the code."""
    idx = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for a, b in zip(idx, idx[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out


def _curve(dataset, key, model):
    with open(os.path.join(DEC, f"scalar_{model}_{dataset}_chat.json")) as f:
        d = json.load(f)
    return np.array(d["results"][key]["acc_mean"])


def figure(model="qwen3-1.7b", read="mean", out=None):
    # Four series, not five. brit culture is dropped because brit language already carries the
    # "lexical, therefore flat from L0" case, and the palette's fixed order puts two low-contrast
    # slots on screen by the fourth -- their relief is direct labels, which the method allows for
    # at most four series. Adding a fifth would force a choice between the two rules.
    series = [
        ("brit language", "brit_language", f"language|{read}|linear"),
        ("styc computation-correctness", "styc", f"corr_e|{read}|linear"),
        ("cmpdir precedence", "cmpdir", f"precedence|{read}|linear"),
        ("cmpdir causation", "cmpdir", f"causation|{read}|linear"),
    ]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.5, 5.4), facecolor=SURFACE,
                                 gridspec_kw=dict(width_ratios=[1.35, 1]))
    for a in (ax, bx):
        a.set_facecolor(SURFACE)
        a.grid(True, color=GRID, lw=0.7, alpha=0.9)
        a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(GRID)
        a.tick_params(colors=INK2, labelsize=9)
        a.axhline(0.5, color=INK2, lw=1, ls=(0, (2, 3)), alpha=0.7)

    # ── panel A
    ends = []
    for i, (label, ds, key) in enumerate(series):
        y = _curve(ds, key, model)
        x = np.arange(len(y)) / (len(y) - 1)
        ax.plot(x, y, color=SER[i], lw=2, ls=DASH[i], label=label, solid_capstyle="round")
        ends.append((y[-1], label))
    for (yy, label), ly in zip(ends, _declutter([e[0] for e in ends], 0.035)):
        ax.annotate(label, xy=(1.02, ly), color=INK, fontsize=8.5, va="center", ha="left")
    ax.text(0.012, 0.512, "chance", color=INK2, fontsize=8)
    ax.set_xlim(-0.02, 1.75)
    ax.set_ylim(0.42, 1.04)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("fractional depth  (0 = embedding output)", color=INK2, fontsize=9.5)
    ax.set_ylabel("held-out pairwise accuracy", color=INK2, fontsize=9.5)
    ax.set_title("cmpdir starts at chance because it cannot do otherwise",
                 color=INK, fontsize=11.5, loc="left", pad=10)

    # ── panel B
    with open(os.path.join(CMP, f"depth_{model}_{read}.json")) as f:
        dep = json.load(f)
    # ORDINAL, not categorical: these are three knowledge levels of ONE series, so they take one
    # hue at three steps rather than three identities. Reusing panel A's categorical slots here
    # would paint "known" the same blue that means "brit language" one panel to the left.
    ORD = ["#104281", "#86b6ef", "#2a78d6"]
    ends = []
    handles = []
    for name, ci in [("known", 0), ("order-dependent", 1), ("all", 2)]:
        c = dep["results"]["precedence"].get(name)
        if c is None:
            continue
        y = np.array(c["acc"])
        x = np.arange(len(y)) / (len(y) - 1)
        lab = f"{name}  (n={c['n']})"
        ln, = bx.plot(x, y, color=ORD[ci], lw=2, ls=DASH[ci], label=lab, solid_capstyle="round")
        handles.append(ln)
        ends.append((y[-1], lab))
    for (yy, lab), ly in zip(ends, _declutter([e[0] for e in ends], 0.035)):
        bx.annotate(lab, xy=(1.02, ly), color=INK, fontsize=8.5, va="center", ha="left")
    bx.set_xlim(-0.02, 1.85)
    bx.set_ylim(0.42, 1.04)
    bx.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    bx.set_xlabel("fractional depth", color=INK2, fontsize=9.5)
    bx.set_title("precedence, split by whether the model knows the fact",
                 color=INK, fontsize=11.5, loc="left", pad=10)

    # One shared legend below both panels: identity is never colour-alone, and a per-axes box
    # would have to sit on top of a curve in either panel.
    h, la = ax.get_legend_handles_labels()
    fig.legend(h, la, frameon=False, fontsize=8.5, labelcolor=INK2, ncol=4,
               loc="lower center", bbox_to_anchor=(0.30, 0.045))
    fig.legend(handles, [t.get_label() for t in handles], frameon=False, fontsize=8.5,
               labelcolor=INK2, ncol=3, loc="lower center", bbox_to_anchor=(0.80, 0.045))
    fig.suptitle(f"cmpdir decodability x depth  —  {model}, {read}-pooled read, linear probe",
                 color=INK, fontsize=12.5, x=0.008, ha="left", y=0.985)
    fig.text(0.008, 0.012,
             "Lexical and length floors are 0.500 BY CONSTRUCTION on cmpdir: the two sides of "
             "every pair are exact token permutations. L0 tie_frac = 1.000.",
             color=INK2, fontsize=8)
    fig.tight_layout(rect=[0, 0.105, 1, 0.955])
    out = out or os.path.join(CMP, "plots", f"cmpdir_depth_{model}_{read}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print(f"→ {out}")
    return out


if __name__ == "__main__":
    figure(sys.argv[1] if len(sys.argv) > 1 else "qwen3-1.7b",
           sys.argv[2] if len(sys.argv) > 2 else "mean")
