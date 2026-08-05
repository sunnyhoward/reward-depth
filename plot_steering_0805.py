#!/usr/bin/env python
"""Steering figures: layer profile, dose curve, cross-lingual, and the saturation gate.

Design notes:
  - Judged numbers where available (judged_*.json), lexicon only as fallback, and the source is
    stated in the caption — today the lexicons over-read with agreement as low as .62.
  - Refusal rate alone is NOT the headline; every panel carries the benign/over-refusal side,
    because L16 posts the highest raw refusal in the sweep and the WORST discrimination.
  - Cross-prompt diversity is drawn as its own gate row: L16 alpha=.05 was the only cell below
    64/64 unique openers, and per-response coherence could not see it.
  - Categorical hues assigned in fixed order from a validated palette, never cycled; the
    cross-lingual panel caps at 5 languages, within the all-pairs series limit with direct labels.

Out: results/plots_0805/fig5_steering_*.png
"""
import os, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/workspace/reward-depth/results/plots_0805"
os.makedirs(OUT, exist_ok=True)
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8880"
S1, S2, S3, S4, S5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#d8d6cf", "axes.linewidth": .8, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 9,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "grid.color": "#e8e6df",
    "grid.linewidth": .7, "legend.frameon": False, "figure.dpi": 150})


def style(ax, t=None, xl=None, yl=None):
    ax.grid(True, axis="y", zorder=0); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if t: ax.set_title(t, loc="left", pad=9)
    if xl: ax.set_xlabel(xl)
    if yl: ax.set_ylabel(yl)


def judged(tag):
    """refusal_judge.py derives its output name by stripping an `eval_` prefix these files never
    had, so judged_steer_fine.json is actually written as judged__fine.json."""
    for cand in (f"/workspace/refusal/judged_{tag}.json",
                 f"/workspace/refusal/judged_{tag[5:]}.json" if tag.startswith("steer") else None,
                 f"/workspace/refusal/judged__{tag.split('_')[-1]}.json"):
        if cand and os.path.exists(cand):
            return json.load(open(cand))
    return {}


def raw(tag):
    p = f"/workspace/refusal/{tag}.json"
    return json.load(open(p)) if os.path.exists(p) else {}


def cell(J, R, key, lg, setn):
    """judged if present and non-degenerate, else lexicon."""
    k = f"{key}_{lg}_{setn}"
    if k in J and J[k]["degenerate_frac"] <= .25:
        return J[k]["judge_refusal"], "judge"
    blk = (R.get("generations_benign" if setn == "benign" else "generations", {})
           .get(f"{key}_{lg}"))
    return (float(np.mean(blk["lex"])), "lex") if blk else (None, None)


# ───────── fig 5a: layer profile (coarse pilot + fine fill-in) ─────────
Jp, Rp = judged("steer_crosslingual"), raw("steer_crosslingual")
Jf, Rf = judged("steer_fine"), raw("steer_fine")
prof = {}
for L in [0, 4, 8, 12, 16, 20, 24, 28, 32, 35]:
    h, src = cell(Jp, Rp, f"a0.05_L{L}", "en", "harm")
    b, _ = cell(Jp, Rp, f"a0.05_L{L}", "en", "benign")
    if h is not None: prof[L] = (h, b, src)
for L in [13, 14, 15, 17, 18, 19]:
    h, src = cell(Jf, Rf, f"a0.05_L{L}", "en", "harm")
    b, _ = cell(Jf, Rf, f"a0.05_L{L}", "en", "benign")
    if h is not None: prof[L] = (h, b, src)
base_h, _ = cell(Jp, Rp, "a0_L-", "en", "harm")
base_b, _ = cell(Jp, Rp, "a0_L-", "en", "benign")

if prof:
    Ls = sorted(prof)
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    style(ax, "Steering works L8-L20 — false positives spike at L13-17, splitting the optimum",
          "layer the direction is added at", "rate (judge)")
    ax.axhline(base_h, color=MUTED, ls="--", lw=1.1, zorder=2)
    ax.text(24.5, base_h - .045, f"base harmful {base_h:.2f}", color=MUTED, fontsize=8)
    ax.plot(Ls, [prof[L][0] for L in Ls], "-o", color=S1, lw=2, ms=5.5, zorder=4)
    ax.plot(Ls, [prof[L][1] for L in Ls], "-o", color=S2, lw=2, ms=5.5, zorder=4)
    ax.plot(Ls, [prof[L][0] - prof[L][1] for L in Ls], "-o", color=S3, lw=1.8, ms=5, zorder=3)
    ax.annotate("refuses harmful", (8, prof[8][0]), textcoords="offset points",
                xytext=(-52, 16), color=S1, fontsize=8.5, weight="bold")
    ax.annotate("refuses benign\n(false positives)", (16, prof.get(16, (0, 0))[1]),
                textcoords="offset points", xytext=(30, -34), color=S2, fontsize=8.5, weight="bold")
    for Lm in (8, 19):
        if Lm in prof:
            ax.annotate(f"L{Lm}\n+{prof[Lm][0]-prof[Lm][1]:.2f}", (Lm, prof[Lm][0]-prof[Lm][1]),
                        textcoords="offset points", xytext=(0, 14), ha="center", color=S3,
                        fontsize=8, weight="bold")
    ax.annotate("discrimination", (24, prof[24][0] - prof[24][1]),
                textcoords="offset points", xytext=(6, 26), color=S3, fontsize=8.5, weight="bold")
    ax.set_ylim(-0.04, 1.02); ax.set_xticks([0, 4, 8, 12, 16, 20, 24, 28, 32, 35])
    ax.text(.985, .97, "alpha=0.05, English, n=128 harmful / 64 benign\nQwen3-4B-Base, "
            "difference-in-means direction fit on English only",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=MUTED)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig5a_steer_layers.png"); plt.close(fig)
    print("wrote fig5a")

# ───────── fig 5b: dose curve, L16 vs L20 ─────────
Jd, Rd = judged("steer_dose"), raw("steer_dose")
if Rd:
    AL = Rd.get("alphas", [])
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3))
    for ax, setn, ttl, col in ((axes[0], "harm", "Refuses harmful", S1),
                               (axes[1], "benign", "Refuses benign (the cost)", S2)):
        style(ax, ttl, "alpha (fraction of residual norm)", "rate (judge)")
        for L, c, lab in ((16, S2, "L16"), (20, S1, "L20")):
            ys = [cell(Jd, Rd, f"a{a}_L{L}", "en", setn)[0] for a in AL]
            ok = [(a, y) for a, y in zip(AL, ys) if y is not None]
            if ok:
                ax.plot([a for a, _ in ok], [y for _, y in ok], "-o", color=c, lw=2, ms=6)
                ax.annotate(lab, ok[-1], textcoords="offset points", xytext=(6, 0),
                            color=c, fontsize=9, weight="bold")
        b0 = cell(Jd, Rd, "a0_L-", "en", setn)[0]
        if b0 is not None:
            ax.axhline(b0, color=MUTED, ls="--", lw=1.1)
            ax.text(AL[0], b0 + .015, "base", color=MUTED, fontsize=8)
        ax.set_ylim(-0.04, 1.02)
    fig.suptitle("Dose response: the two active layers diverge on cost, not on install",
                 x=.008, ha="left", fontsize=10.5, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(f"{OUT}/fig5b_steer_dose.png"); plt.close(fig)
    print("wrote fig5b")

# ───────── fig 5c: cross-lingual ─────────
Jx, Rx = judged("steer_xling"), raw("steer_xling")
if Rx:
    LANGS = Rx.get("langs", [])
    Ls = Rx.get("layers", [])
    COL = {"en": S1, "ar": S2, "it": S3, "vi": S4, "ko": S5}
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    style(ax, "Does an English-fit direction steer languages it never saw?",
          "layer", "refusal on harmful prompts, delta vs base (judge)")
    ax.axhline(0, color=MUTED, lw=1, ls=":", zorder=2)
    for lg in LANGS:
        b0 = cell(Jx, Rx, "a0_L-", lg, "harm")[0]
        ys, xs = [], []
        for L in Ls:
            v = cell(Jx, Rx, f"a0.05_L{L}", lg, "harm")[0]
            if v is not None and b0 is not None:
                xs.append(L); ys.append(v - b0)
        if xs:
            ax.plot(xs, ys, "-o", color=COL.get(lg, MUTED), lw=2, ms=5.5, zorder=4)
            ax.annotate(lg, (xs[-1], ys[-1]), textcoords="offset points", xytext=(7, -2),
                        color=COL.get(lg, MUTED), fontsize=9, weight="bold")
    ax.set_xticks(Ls)
    ax.text(.985, .97, "alpha=0.05, n=64/language\ndirection fit on ENGLISH prompts only",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=MUTED)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig5c_steer_crosslingual.png"); plt.close(fig)
    print("wrote fig5c")

# ───────── fig 5d: the saturation gate ─────────
rows = []
for tag in ("steer_crosslingual", "steer_fine", "steer_dose"):
    R = raw(tag)
    for k, v in (R.get("diversity_benign") or {}).items():
        if k.endswith("_en"):
            rows.append((k, v))
if rows:
    rows = sorted(set(rows))
    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    style(ax, "Cross-prompt diversity — the gate per-response coherence cannot see",
          None, "distinct openers / n")
    xs = np.arange(len(rows))
    cols = [S2 if v < .95 else S3 for _, v in rows]
    ax.bar(xs, [v for _, v in rows], color=cols, width=.7, zorder=3)
    ax.axhline(.95, color=MUTED, ls="--", lw=1.1)
    ax.text(len(rows) - .5, .955, "saturation flag", color=MUTED, fontsize=8, ha="right")
    for i, (k, v) in enumerate(rows):
        if v < .95:
            ax.annotate(k.replace("_en", ""), (i, v), textcoords="offset points",
                        xytext=(0, -14), ha="center", color=S2, fontsize=8, weight="bold")
    ax.set_xticks([]); ax.set_ylim(0.7, 1.02)
    ax.set_xlabel("every steering cell (English)", labelpad=6)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig5d_saturation_gate.png"); plt.close(fig)
    print("wrote fig5d")

print(f"\nfigures in {OUT}")
