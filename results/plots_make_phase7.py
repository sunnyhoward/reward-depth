#!/usr/bin/env python
"""Phase-7 figures. Rerunnable: picks up whatever run histories exist.
Outputs PNGs into results/plots/. Palette: validated categorical set (dataviz reference)."""
import json, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, MUT = "#0b0b0b", "#52514e"
plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "axes.edgecolor": MUT, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUT, "ytick.color": MUT, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6})
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(OUT, exist_ok=True)
W = "/workspace"

def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=150)
    plt.close(fig)
    print("wrote", name)

# ---------- Fig 1: styc factor decodability vs depth (v2) ----------
try:
    d = json.load(open(f"{W}/styc_stageA.json"))["curves"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(d["style_correct"]))
    ax.plot(x, d["style_correct"], color=C[0], lw=2, label="style (explained vs terse)")
    ax.plot(x, d["corr_terse"], color=C[1], lw=2, label="correctness | terse pairs")
    ax.plot(x, d["corr_explained"], color=C[2], lw=2, label="correctness | explained pairs")
    bt = d.get("corr_by_type", {})
    if "know" in bt: ax.plot(x, bt["know"], color=C[3], lw=2, ls="--", label="correctness — facts (retrieval)")
    if "mcq_arith" in bt: ax.plot(x, bt["mcq_arith"], color=C[6], lw=2, ls="--", label="correctness — arithmetic")
    ax.axhline(0.5, color=MUT, lw=0.8, ls=":")
    ax.text(0.3, 0.51, "chance", color=MUT, fontsize=8)
    ax.set(xlabel="layer (Qwen2.5-3B, 36 blocks)", ylabel="held-out probe accuracy",
           ylim=(0.35, 1.03), title="styc v2: where each factor of the preference is decodable")
    ax.legend(frameon=False, fontsize=8.5, loc="center left")
    save(fig, "fig_p7_styc_curves.png")
except Exception as e:
    print("fig1 skipped:", e)

# ---------- Fig 2: conflict pairs — every entangled probe/ensemble at zero ----------
try:
    d = json.load(open(f"{W}/styc_stageA.json"))["ens"]
    labels, vals = [], []
    for k in ["L10", "L20", "L30"]:
        labels.append(f"probe @{k}"); vals.append(d["conflict"]["singles"][k])
    for k in ["uniform", "evidence", "precision"]:
        labels.append(f"{k} ensemble"); vals.append(d["conflict"]["ensembles"][k])
    labels += ["probe @L35 +\nconflicts in diet", "corr-head @L35\n(factor label)"]; vals += [0.974, 1.0]
    fig, ax = plt.subplots(figsize=(7, 4.0))
    ypos = np.arange(len(labels))
    cols = [C[7]] * 6 + [C[0], C[2]]
    ax.barh(ypos, vals, height=0.55, color=cols)
    for y, v in zip(ypos, vals):
        ax.text(max(v, 0.005) + 0.015, y, f"{v:.2f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(ypos); ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set(xlim=(0, 1.1), xlabel="accuracy on conflict pairs (correct-terse vs wrong-explained)",
           title="styc conflict pairs: style capture everywhere; two working fixes")
    ax.axvline(0.5, color=MUT, lw=0.8, ls=":")
    save(fig, "fig_p7_styc_conflict.png")
except Exception as e:
    print("fig2 skipped:", e)

# ---------- Fig 3: UF per-layer per-type curves ----------
try:
    P = np.load(f"{W}/depth_ensemble_P.npy")
    types = np.array([x["type"] for x in json.load(open(f"{W}/test350_typed.json"))]
                     + ["open"] * (P.shape[1] - 350))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, (t, lab) in enumerate([("open", "open/chat"), ("code", "code"),
                                  ("format", "format/classification"), ("translation", "translation")]):
        m = types == t
        ax.plot(np.arange(P.shape[0]), (P[:, m] > 0.5).mean(1), color=C[i], lw=2,
                label=f"{lab} (n={int(m.sum())})")
    ax.axhline(0.5, color=MUT, lw=0.8, ls=":")
    ax.set(xlabel="layer (Tulu-3-8B-SFT, 32 blocks)", ylabel="held-out probe accuracy",
           ylim=(0.4, 1.02), title="UF: probe accuracy by task type across depth\n"
           "(rise-then-flat = style only; no late jump anywhere = verification absent)")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    save(fig, "fig_p7_uf_typecurves.png")
except Exception as e:
    print("fig3 skipped:", e)

# ---------- Fig 4: UF queue arms — install trajectories ----------
try:
    arms = [("sd_upper300", "B soft-DPO, writes >L12", C[0]),
            ("hyb2_300", "C + margin co-trained", C[1]),
            ("twostage300", "D soft-DPO on margin edit", C[2]),
            ("margin300", "A margin only", C[3])]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for tag, lab, col in arms:
        f = f"{W}/uf_hybrid_md_{tag}_history.json"
        if not os.path.exists(f): continue
        d = json.load(open(f))
        xs = [e["step"] for e in d["evals"]]; ys = [e["acc_implicit"] for e in d["evals"]]
        ax.plot(xs, ys, color=col, lw=2, marker="o", ms=4, label=lab)
    ax.axhline(0.5, color=MUT, lw=0.8, ls=":")
    ax.axhline(0.8, color=MUT, lw=0.8, ls="--")
    ax.text(2, 0.81, "full-stack soft-DPO reference (0.80)", color=MUT, fontsize=8)
    ax.set(xlabel="step", ylabel="implicit-reward accuracy (n=64)", ylim=(0.35, 0.9),
           title="UF arms: the emission head does everything; the margin adds nothing or harm")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    save(fig, "fig_p7_uf_arms.png")
except Exception as e:
    print("fig4 skipped:", e)

# ---------- Fig 5: styc training arms — generation oracles (rerunnable as arms land) ----------
try:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    metrics = [("gen_wrong", "generated WRONG answer"), ("gen_explained", "generated explained style"),
               ("acc_conflict", "conflict-pair implicit acc")]
    arms = [("early", f"early labeller (L10)", C[7]), ("late", "late labeller (L35)", C[0]),
            ("gt", "ground truth", C[2])]
    found = False
    for ax, (mk, mtitle) in zip(axes, metrics):
        for tag, lab, col in arms:
            f = f"{W}/styc_train_{tag}_history.json"
            if not os.path.exists(f): continue
            found = True
            d = json.load(open(f))
            xs = [e["step"] for e in d["evals"] if mk in e]
            ys = [e[mk] for e in d["evals"] if mk in e]
            ax.plot(xs, ys, color=col, lw=2, marker="o", ms=3.5, label=lab)
        ax.set(title=mtitle, xlabel="step", ylim=(-0.03, 1.03))
    axes[0].set_ylabel("fraction / accuracy")
    axes[0].legend(frameon=False, fontsize=8)
    if found:
        fig.suptitle("styc training arms: what each labeller installs (held-out generation oracles)", y=1.02)
        save(fig, "fig_p7_styc_train.png")
    else:
        plt.close(fig); print("fig5 skipped: no arm histories yet")
except Exception as e:
    print("fig5 skipped:", e)
print("ALL FIGS DONE")
