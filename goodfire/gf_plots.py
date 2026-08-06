#!/usr/bin/env python
"""Figures. Robust to missing runs -- plots whatever is on disk.

Colour/marks follow the house data-viz rules: probe depth is a *sequential* variable so the depth
sweep uses one hue light->dark rather than cycled categorical hues; the money plot is two stacked
panels sharing an x axis rather than a dual-axis chart (two measures, two scales, never two y
axes); text stays in ink colours and never wears the series colour.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gf_common as G

PLOTS = G.RESULTS / "plots"
PLOTS.mkdir(exist_ok=True)

SURFACE = "#fcfcfb"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdbd6"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"          # validated categorical slots 1-3
SEQ = ["#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#256abf", "#184f95", "#0d366b"]  # blue ramp


def style(ax, xlabel, ylabel, title=None):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)


def fig(nrows=1, ncols=1, figsize=(7, 4.2), **kw):
    f, ax = plt.subplots(nrows, ncols, figsize=figsize, facecolor=SURFACE, **kw)
    return f, ax


def save(f, name):
    f.tight_layout()
    f.savefig(PLOTS / name, dpi=160, facecolor=SURFACE)
    plt.close(f)
    print(f"[write] {PLOTS / name}", flush=True)


def load_run(tag):
    p = G.RESULTS / "runs" / f"{tag}.json"
    return G.jload(p) if p.exists() else None


def evals(run):
    """-> (steps, {metric: [values]}) over checkpoints that carry an eval block."""
    pts = [(r["step"], r["eval"]) for r in run["history"] if "eval" in r]
    pts.sort()
    return [p[0] for p in pts], [p[1] for p in pts]


# ------------------------------------------------------------------ 1. decodability curve
def fig_decodability():
    d = G.jload(G.RESULTS / "decodability.json")
    ax_ho = None
    p = G.RESULTS / "decodability_axisho.json"
    if p.exists():
        ax_ho = G.jload(p)

    f, ax = fig(figsize=(7.6, 4.4))
    L = [c["layer"] for c in d["curve"]]
    ax.plot(L, [c["auroc_pooled"] for c in d["curve"]], color=C1, lw=2, label="pooled (random split)")
    ax.plot(L, [c["auroc_marker_token"] for c in d["curve"]], color=C2, lw=2,
            label="marker tokens only")
    ax.plot(L, [c["auroc_token"] for c in d["curve"]], color=C3, lw=2, label="per-token (all)")
    if ax_ho:
        L2 = [c["layer"] for c in ax_ho["curve"]]
        ax.plot(L2, [c["auroc_pooled"] for c in ax_ho["curve"]], color=C1, lw=2, ls="--",
                label="pooled (held-out axes)")
    ax.axhline(0.5, color=INK2, lw=1, ls=":")
    ax.text(L[0], 0.505, "chance", color=INK2, fontsize=8, ha="left", va="bottom")
    # direct label for the low-contrast aqua series (relief rule)
    ax.annotate("per-token", (L[-1], d["curve"][-1]["auroc_token"]), color=INK2, fontsize=8,
                xytext=(-2, 8), textcoords="offset points", ha="right")
    ax.set_ylim(0.45, 1.02)
    style(ax, "probe layer (0 = embeddings)", "AUROC on held-out completions",
          f"Decodability of British vs American English — frozen {d['model']}")
    leg = ax.legend(frameon=False, fontsize=9, loc="lower right", ncol=2)
    for t in leg.get_texts():
        t.set_color(INK2)
    save(f, "fig1_decodability.png")


# ---------------------------------------------------------- 2. reward types over training
def fig_training(tags):
    runs = [(lab, load_run(t)) for lab, t in tags]
    runs = [(l, r) for l, r in runs if r]
    if not runs:
        return
    f, axes = fig(1, 2, figsize=(10.5, 4.2))
    for (lab, r), c in zip(runs, [C1, C2, C3, "#eda100"]):
        s, e = evals(r)
        axes[0].plot(s, [x["be_rate"] for x in e], color=c, lw=2, marker="o", ms=5, label=lab)
        axes[1].plot(s, [x["kl_policy_base"] for x in e], color=c, lw=2, marker="o", ms=5, label=lab)
    base = evals(runs[0][1])[1][0]["be_rate"]
    axes[0].axhline(base, color=INK2, lw=1, ls=":")
    axes[0].text(0, base + 0.01, "base model", color=INK2, fontsize=8, va="bottom")
    style(axes[0], "optimizer step", "held-out BE rate (oracle)", "What actually gets installed")
    style(axes[1], "optimizer step", "KL(policy || base) per token", "Drift from base")
    leg = axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    for t in leg.get_texts():
        t.set_color(INK2)
    save(f, "fig2_training.png")


# --------------------------------------------------------------------- 3. depth sweep
def depth_runs(layers, pattern="run4_pooled_L{}"):
    out = []
    for L in layers:
        r = load_run(pattern.format(L))
        if r:
            out.append((L, r))
    return out


def fig_depth(layers):
    rs = depth_runs(layers)
    if not rs:
        return
    f, ax = fig(figsize=(7.6, 4.4))
    ramp = [SEQ[int(round(i * (len(SEQ) - 1) / max(1, len(rs) - 1)))] for i in range(len(rs))]
    for (L, r), c in zip(rs, ramp):
        s, e = evals(r)
        ax.plot(s, [x["be_rate"] for x in e], color=c, lw=2, marker="o", ms=5, label=f"L{L}")
        ax.annotate(f"L{L}", (s[-1], e[-1]["be_rate"]), color=INK2, fontsize=8,
                    xytext=(4, -3), textcoords="offset points")
    o = load_run("run1_oracle")
    if o:
        s, e = evals(o)
        ax.plot(s, [x["be_rate"] for x in e], color=C2, lw=2, ls="--", label="oracle (ceiling)")
    style(ax, "optimizer step", "held-out BE rate (oracle)",
          "Depth sweep — pooled probe reward, probe read on the frozen base")
    leg = ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=2)
    for t in leg.get_texts():
        t.set_color(INK2)
    save(f, "fig3_depth.png")


# ------------------------------------------------------------------ 4. the money plot
def fig_money(layers):
    d = G.jload(G.RESULTS / "decodability.json")
    by_layer = {c["layer"]: c for c in d["curve"]}
    ho = None
    p = G.RESULTS / "decodability_axisho.json"
    if p.exists():
        ho = {c["layer"]: c for c in G.jload(p)["curve"]}
    rs = depth_runs(layers)
    if not rs:
        return
    Ls = [L for L, _ in rs]
    inst = [evals(r)[1][-1]["be_rate"] for _, r in rs]

    # Two measures, two scales -> two stacked panels sharing x. Never a dual y axis.
    f, axes = fig(2, 1, figsize=(7.2, 6.4), sharex=True)
    axes[0].plot(Ls, [by_layer[L]["auroc_pooled"] for L in Ls], color=C1, lw=2, marker="o", ms=6,
                 label="random split")
    if ho:
        axes[0].plot(Ls, [ho[L]["auroc_pooled"] for L in Ls], color=C1, lw=2, ls="--", marker="s",
                     ms=5, label="held-out axes")
        leg = axes[0].legend(frameon=False, fontsize=9, loc="lower right")
        for t in leg.get_texts():
            t.set_color(INK2)
    axes[0].set_ylim(0.45, 1.02)
    axes[0].axhline(0.5, color=INK2, lw=1, ls=":")
    style(axes[0], "", "probe AUROC (decodability)", "Decodability — can a probe read it here?")

    axes[1].plot(Ls, inst, color=C2, lw=2, marker="o", ms=6, label="probe reward at L")
    o, base = load_run("run1_oracle"), None
    if o:
        se, ee = evals(o)
        axes[1].axhline(ee[-1]["be_rate"], color=INK2, lw=1.2, ls="--")
        axes[1].text(Ls[0], ee[-1]["be_rate"] + 0.01, "oracle-reward ceiling", color=INK2, fontsize=8)
        base = ee[0]["be_rate"]
        axes[1].axhline(base, color=INK2, lw=1, ls=":")
        axes[1].text(Ls[0], base + 0.01, "base model", color=INK2, fontsize=8)
    style(axes[1], "probe layer", "held-out BE rate achieved",
          "Installability — does training against it work?")
    save(f, "fig4_money.png")


# ------------------------------------------------- 5. probe reward vs oracle over training
def fig_hacking(tags):
    runs = [(lab, load_run(t)) for lab, t in tags]
    runs = [(l, r) for l, r in runs if r]
    if not runs:
        return
    f, ax = fig(figsize=(7.6, 4.4))
    for (lab, r), c in zip(runs, [C1, C2, C3, "#eda100"]):
        # Eval checkpoints, not per-step: within one step the 64 rollouts are range-restricted,
        # which drags the per-step rho down for reasons that have nothing to do with hacking.
        L = str(r["config"]["layer"])
        s, e = evals(r)
        rho = [x["probe_oracle_spearman_by_layer"].get(L, np.nan) for x in e]
        ax.plot(s, rho, color=c, lw=2, marker="o", ms=5, label=lab)
    ax.axhline(0, color=INK2, lw=1, ls=":")
    style(ax, "optimizer step", "Spearman rho (probe score vs oracle BE rate)",
          "Reward hacking, measured directly — divergence is the failure")
    leg = ax.legend(frameon=False, fontsize=9, loc="lower left")
    for t in leg.get_texts():
        t.set_color(INK2)
    save(f, "fig5_hacking.png")


def main():
    layers = [4, 8, 12, 16, 20, 24]
    best = 12
    fig_decodability()
    fig_training([("oracle", "run1_oracle"),
                  (f"probe L{best} pooled", f"run4_pooled_L{best}"),
                  (f"probe L{best} dense", f"run3_dense_L{best}"),
                  (f"probe L{best} on student", f"run5_student_L{best}")])
    fig_depth(layers)
    fig_money(layers)
    fig_hacking([("oracle", "run1_oracle"),
                 (f"probe L{best} pooled", f"run4_pooled_L{best}"),
                 (f"probe L{best} on student", f"run5_student_L{best}")])


if __name__ == "__main__":
    main()
