#!/usr/bin/env python
"""Figures for the 2026-08-05 session: K-FAC negative + KL invariant, the remeasured
encoding-depth table, the cross-lingual harmfulness probe, and the refusal ladder.

Design notes (why the charts look the way they do):
  - Categorical hues are assigned in fixed order from a validated palette and never cycled.
    Slots 4/5 are contrast-WARN on a light surface, so every series carries a direct label.
  - Fig 1 does NOT colour by individual arm. The claim is that the damage threshold is
    INVARIANT across arms, so encoding arm identity would fight the message; arms are grouped
    into the two families that were supposed to differ (curvature leash vs learning rate).
  - Fig 3 has 10 languages. Ten categorical hues is never right — English is the fitted series
    and the other nine become a min-max band plus a mean line.
  - No dual axes anywhere. Where two measures share the 0-1 accuracy scale they share an axis;
    where they do not, they get separate panels.

Out: /workspace/reward-depth/results/plots_0805/*.png
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/workspace/reward-depth/results/plots_0805"
os.makedirs(OUT, exist_ok=True)

SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8880"
S1, S2, S3, S4, S5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#d8d6cf", "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "font.size": 9, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "grid.color": "#e8e6df", "grid.linewidth": 0.7,
    "legend.frameon": False, "figure.dpi": 150,
})


def style(ax, title=None, xlabel=None, ylabel=None):
    ax.grid(True, axis="y", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if title:
        ax.set_title(title, loc="left", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


# ────────────────────────── Fig 1: the KL invariant ──────────────────────────
LEASH = [("kfac_l24_lam0", "λ=0"), ("kfac_l24_lam1", "λ=1"), ("kfac_l24_lam10", "λ=10"),
         ("kfac_l24_lam100", "λ=100"), ("kfac_l24_lam1000", "λ=1000")]
LRC = [("kfac_l24_lam0_lr1.7e-5", "LR 1.7e-5"), ("kfac_l24_lam0_lr1e-5", "LR 1e-5")]

fig, ax = plt.subplots(figsize=(7.4, 4.5))
style(ax, "Damage ONSET is arm-invariant; the depth of collapse is not",
      "KL from base (nats)", "gen_correct (free sampling)")
ax.axvspan(2.0, 2.5, color=S2, alpha=0.11, zorder=1)
ax.text(2.25, 0.03, "onset\n2.0–2.5", ha="center", va="bottom", color=S2, fontsize=8, weight="bold")

allpts = []
for grp, colour, lab in ((LEASH, S1, "K-FAC leash (λ 0→1000)"),
                         (LRC, S4, "learning-rate control")):
    xs, ys = [], []
    for tag, _ in grp:
        p_ = f"/workspace/eagle_{tag}/history.json"
        if not os.path.exists(p_):
            continue
        for e in json.load(open(p_))["evals"]:
            if e["step"] > 0:
                xs.append(e["kl_from_base"]); ys.append(e["gen_correct"])
    ax.scatter(xs, ys, s=26, color=colour, alpha=0.62, zorder=3, lw=0, label=lab)
    allpts += list(zip(xs, ys))

a = np.array(allpts)
edges = np.array([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
cx, cy = [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = a[(a[:, 0] >= lo) & (a[:, 0] < hi)]
    if len(m) >= 3:
        cx.append((lo + hi) / 2); cy.append(float(np.median(m[:, 1])))
ax.plot(cx, cy, "-", color=INK2, lw=1.6, alpha=0.85, zorder=4)
ax.annotate("binned median", (cx[2], cy[2]), textcoords="offset points", xytext=(-64, -26),
            color=INK2, fontsize=8, weight="bold",
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.set_ylim(-0.04, 1.10); ax.set_xlim(-0.1, 3.9)
ax.legend(loc="lower left", fontsize=8.5, bbox_to_anchor=(0.0, 0.0))
ax.text(0.985, 0.985, "110 evals · 7 arms · styc L24, frozen tf head\nSpearman(KL, correct) = −0.48",
        transform=ax.transAxes, ha="right", va="top", fontsize=8, color=MUTED)
ax.text(0.03, 0.40, "past onset the arms separate:\nLR controls hold .45–.69\nwhere λ arms sit at .06–.15",
        transform=ax.transAxes, ha="left", va="top", fontsize=8, color=MUTED, style="italic")
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_kl_invariant.png"); plt.close(fig)

# ─────────────────── Fig 2: encoding depth, remeasured ───────────────────
LS = [4, 12, 24, 32]
peak = {}
for f in ("style", "correct"):
    row = []
    for L in LS:
        p = f"/workspace/eagle_depthtf_{f}_L{L}/history.json"
        evs = [e for e in json.load(open(p))["evals"]
               if e.get("head_acc") is not None and e["step"] > 0]
        row.append(max(e["head_acc"] for e in evs))
    peak[f] = row
HEAD_AGREE = [0.182, 0.226, 0.298, 0.601]      # this session's tf heads (§17 confound)
OLD_CORRECT = [0.49, 0.54, 0.64, 0.67]         # §1, mlp + trainable head

fig, ax = plt.subplots(figsize=(7.2, 4.4))
style(ax, "Encoding depth remeasured with a frozen tf head — and its confound",
      "write depth L", "accuracy")
ax.axhline(0.5, color=MUTED, lw=0.9, ls=":", zorder=2)
ax.text(4.4, 0.515, "chance", color=MUTED, fontsize=8, va="bottom", ha="left")
ax.plot(LS, peak["style"], "-o", color=S1, lw=2, ms=7, zorder=4)
ax.plot(LS, peak["correct"], "-o", color=S2, lw=2, ms=7, zorder=4)
ax.plot(LS, OLD_CORRECT, "--o", color=S2, lw=1.4, ms=5, alpha=0.45, zorder=3)
ax.plot(LS, HEAD_AGREE, "-o", color=S4, lw=2, ms=7, zorder=4)
ax.annotate("style (peak head_acc)", (LS[1], peak["style"][1]), textcoords="offset points",
            xytext=(6, 8), color=S1, fontsize=8.5, weight="bold")
ax.annotate("correct (peak head_acc)", (LS[2], peak["correct"][2]), textcoords="offset points",
            xytext=(-10, 10), color=S2, fontsize=8.5, weight="bold")
ax.annotate("§1 correct (mlp head, trainable)", (LS[2], OLD_CORRECT[2]),
            textcoords="offset points", xytext=(-30, -22), color=S2, fontsize=8, alpha=0.85)
ax.annotate("head competence\n(top-1 agreement w/ base)", (LS[1], HEAD_AGREE[1]),
            textcoords="offset points", xytext=(10, -30), color=S4, fontsize=8.5, weight="bold")
ax.set_xticks(LS); ax.set_ylim(0, 1.08)
ax.text(0.985, 0.02, "correct and head competence both rise with depth —\nthe two are not separated by this sweep",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=MUTED, style="italic")
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_encoding_depth.png"); plt.close(fig)

# ─────────────── Fig 3: cross-lingual harmfulness probe ───────────────
pr = json.load(open("/workspace/refusal/probe_crosslingual.json"))
lay = sorted((int(k) for k in pr["acc"]), key=int)
nz = [l for l in pr["langs"] if l != "en"]
en = [pr["acc"][str(L)]["en"] for L in lay]
tm = [float(np.mean([pr["acc"][str(L)][l] for l in nz])) for L in lay]
tlo = [min(pr["acc"][str(L)][l] for l in nz) for L in lay]
thi = [max(pr["acc"][str(L)][l] for l in nz) for L in lay]

fig, ax = plt.subplots(figsize=(7.2, 4.4))
style(ax, "An English-fit harmfulness probe transfers only mid-stack",
      "layer", "held-out accuracy")
ax.axhline(0.5, color=MUTED, lw=0.9, ls=":", zorder=2)
ax.text(0.5, 0.508, "chance", color=MUTED, fontsize=8, va="bottom", ha="left")
ax.fill_between(lay, tlo, thi, color=S3, alpha=0.16, zorder=2, lw=0)
ax.plot(lay, en, "-o", color=S1, lw=2, ms=6, zorder=4)
ax.plot(lay, tm, "-o", color=S3, lw=2, ms=6, zorder=4)
ax.annotate("English (fitted)", (lay[3], en[3]), textcoords="offset points", xytext=(4, 8),
            color=S1, fontsize=8.5, weight="bold")
ax.annotate("mean of 9 other languages\n(band = min–max)", (lay[2], tm[2]),
            textcoords="offset points", xytext=(-14, -42), color=S3, fontsize=8.5, weight="bold")
imax = int(np.argmax(tm))
ax.annotate(f"peak L{lay[imax]}", (lay[imax], tm[imax]), textcoords="offset points",
            xytext=(-4, 16), color=INK2, fontsize=8, ha="center",
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.set_ylim(0.42, 1.04)
ax.text(0.985, 0.03, "shared representation peaks L8–16, gone by the top —\nthe ladder's inverted-U premise",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=MUTED, style="italic")
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_probe_crosslingual.png"); plt.close(fig)

# ─────────────── Fig 4: refusal ladder, English meters ───────────────
ARMS = [("s1_L4", "stage-1\nL4"), ("s1_L12", "stage-1\nL12"), ("s1_L24", "stage-1\nL24"),
        ("fulldpo", "full DPO\n(all layers)"), ("upperonly_L12", "upper-only\n>L12")]
peaks = []
for tag, lab in ARMS:
    h = json.load(open(f"/workspace/refusal_{tag}/history.json"))
    evs = [e for e in h["evals"] if e["step"] > 0]
    b = max(evs, key=lambda e: e["refusal_eval_lex"])
    peaks.append((lab, b["refusal_eval_lex"], b["refusal_select_lex"], b["kl_from_base"]))

# over-refusal (benign false positives) per arm, from whichever eval cells exist
import glob
def _ov(tag, step):
    f = f"/workspace/refusal/eval_{tag}_ckpt{step}.json"
    if not os.path.exists(f):
        cand = glob.glob(f"/workspace/refusal/eval_{tag}_ckpt*.json")
        if not cand:
            return None
        f = sorted(cand)[0]
    d = json.load(open(f))
    return float(np.mean([d["benign"][lg]["over_refusal_rate"] for lg in d["langs"]])), \
           float(np.mean([d["harmful"][lg]["refusal_rate"] for lg in d["langs"]]))

steps = dict(s1_L4=130, s1_L12=110, s1_L24=40, fulldpo=30, upperonly_L12=30)
ov = [_ov(t, steps[t]) for t, _ in ARMS]
base_ov = None
if os.path.exists("/workspace/refusal/eval_base.json"):
    bd = json.load(open("/workspace/refusal/eval_base.json"))
    base_ov = float(np.mean([bd["benign"][lg]["over_refusal_rate"] for lg in bd["langs"]]))

fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.3))
x = np.arange(len(peaks))
ax = axes[0]
style(ax, "Install: refuses harmful (English)", None, "refusal rate")
ax.bar(x, [p[1] for p in peaks], width=0.56, color=S1, zorder=3)
ax.axhline(0.391, color=S2, lw=1.5, ls="--", zorder=4)
ax.text(-0.42, 0.408, "base .391", color=S2, fontsize=8, ha="left", weight="bold")
for i, p in enumerate(peaks):
    ax.text(i, p[1] + 0.025, f"{p[1]:.2f}", ha="center", fontsize=8.5, color=INK, weight="bold")
    ax.text(i, 0.03, f"KL {p[3]:.2f}", ha="center", fontsize=7.5, color="#ffffff")
ax.set_xticks(x); ax.set_xticklabels([p[0] for p in peaks], fontsize=8); ax.set_ylim(0, 1.14)

ax = axes[1]
style(ax, "Cost: refuses benign (mean over 6 languages)", None, "over-refusal rate")
vals = [(o[0] if o else np.nan) for o in ov]
ax.bar(x, vals, width=0.56, color=S2, zorder=3)
if base_ov is not None:
    ax.axhline(base_ov, color=MUTED, lw=1.5, ls="--", zorder=4)
    ax.text(-0.42, base_ov + 0.008, f"base {base_ov:.3f}", color=MUTED, fontsize=8,
            ha="left", weight="bold")
for i, v in enumerate(vals):
    if not np.isnan(v):
        ax.text(i, v + 0.008, f"{v:.2f}", ha="center", fontsize=8.5, color=INK, weight="bold")
    else:
        ax.text(i, 0.01, "pending", ha="center", fontsize=7.5, color=MUTED, rotation=90)
ax.set_xticks(x); ax.set_xticklabels([p[0] for p in peaks], fontsize=8)
ax.set_ylim(0, max([v for v in vals if not np.isnan(v)] + [0.1]) * 1.35)

ax = axes[2]
style(ax, "Lexical gap (training phrasing − other)", None, "EN_SELECT − EN_EVAL")
gaps = [p[2] - p[1] for p in peaks]
ax.bar(x, gaps, width=0.56, color=[S3 if g < 0.25 else S2 for g in gaps], zorder=3)
ax.axhline(0.25, color=MUTED, lw=1.2, ls="--", zorder=4)
ax.text(len(peaks) - 0.4, 0.263, "lexical-install flag", color=MUTED, fontsize=8, ha="right")
for i, g in enumerate(gaps):
    ax.text(i, g + 0.012, f"{g:+.2f}", ha="center", fontsize=8.5, color=INK, weight="bold")
ax.set_xticks(x); ax.set_xticklabels([p[0] for p in peaks], fontsize=8); ax.set_ylim(-0.02, 0.40)

fig.suptitle("Refusal ladder at each arm's peak — both error types, and the phrasing check "
             "(deep arms matched at KL 0.80–0.90)",
             x=0.008, ha="left", fontsize=10.5, weight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(f"{OUT}/fig4_refusal_ladder.png"); plt.close(fig)

print(f"wrote 4 figures to {OUT}")
for f in sorted(os.listdir(OUT)):
    print("  ", f)
