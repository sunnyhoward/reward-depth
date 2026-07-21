"""Figures for phases 3-4 (2026-07-21) from the run JSONs in this directory. -> plots/
fig6: UF soft-label DPO from the frozen probe (phase 3)
fig7: decision-position arms, six-way ablation (phase 4)"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a"]  # validated categorical, fixed order
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
os.makedirs("plots", exist_ok=True)
def J(f): return json.load(open(f))

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "text.color": INK, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.7,
    "legend.frameon": False,
})

# ══════════ fig 6 — UF soft-DPO (phase 3) ══════════
h = J("uf_softdpo_history.json")
ev = h["evals"]; es = [e["step"] for e in ev]
bigN = J("uf_bigN_softdpo.json"); tulu = J("uf_tulu_dpo_eval.json")
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(11.5, 3.7))

a1.plot(es, [e["acc_implicit"] for e in ev], color=C[0], lw=2, marker="o", ms=4, label="dataset preference")
a1.plot(es, [e["acc_probe"] for e in ev], color=C[4], lw=2, marker="o", ms=4, label="probe agreement")
a1.axhline(h["probe_acc"], color=C[4], lw=1.2, ls=(0, (3, 2)))
a1.text(395, h["probe_acc"] + 0.012, "probe ceiling 0.79", fontsize=7.5, color=INK2, ha="right")
a1.axhline(0.5, color=INK2, lw=0.8, ls=":"); a1.text(5, 0.515, "chance", fontsize=8, color=INK2)
a1.set_ylim(0.45, 0.9); a1.set_xlabel("step")
a1.set_title("(a) held-out accuracy (n=128, in-run)", loc="left", fontsize=10.5, color=INK)
a1.legend(fontsize=8, loc="lower right")

a3_dlp = a2
a3_dlp.plot(es, [e["dlp_chosen"] for e in ev], color=C[0], lw=2, marker="o", ms=4, label="Δlp chosen")
a3_dlp.plot(es, [e["dlp_rejected"] for e in ev], color=C[3], lw=2, marker="o", ms=4, label="Δlp rejected")
a3_dlp.axhline(0, color=INK2, lw=0.8, ls=":")
a3_dlp.set_xlabel("step"); a3_dlp.set_title("(b) displacement (nats vs reference)", loc="left", fontsize=10.5, color=INK)
a3_dlp.legend(fontsize=8)

arms = [("DPO baseline\n(12k pairs)", 0.805, 0.021, INK2),
        ("soft-DPO\nckpt200", bigN["softdpo_ckpt200"]["acc_implicit"], bigN["softdpo_ckpt200"]["se"], C[0]),
        ("soft-DPO\nfinal", bigN["softdpo_final"]["acc_implicit"], bigN["softdpo_final"]["se"], C[0]),
        ("Tulu-3-DPO\n(official)", tulu["acc_implicit"], tulu["se"], C[3]),
        ("RLOO v3\nckpt100", bigN["rl_v3aborted_ckpt100"]["acc_implicit"], bigN["rl_v3aborted_ckpt100"]["se"], C[2])]
x = np.arange(len(arms))
for i, (lab, v, se, c) in enumerate(arms):
    a3.bar(i, v, 0.62, color=c, zorder=2)
    a3.errorbar(i, v, yerr=se, color=INK, lw=1.2, capsize=3, zorder=3)
    a3.text(i, v + se + 0.015, f"{v:.3f}", ha="center", fontsize=8, color=INK)
a3.axhline(0.5, color=INK2, lw=0.8, ls=":")
a3.set_xticks(x, [a[0] for a in arms], fontsize=7.5)
a3.set_ylim(0.4, 0.9); a3.grid(axis="x", visible=False)
a3.set_title("(c) big-N implicit acc (350 pairs ± SE)", loc="left", fontsize=10.5, color=INK)
fig.suptitle("UltraFeedback: soft-label DPO from the frozen L12 probe matches ground-truth DPO "
             "(Tulu-3-8B-SFT, one seed)", fontsize=12, color=INK, x=0.02, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig("plots/fig6_uf_softdpo.png", dpi=150); plt.close(fig)

# ══════════ fig 7 — decision-position arms (phase 4) ══════════
ARMS = [("frozen head", J("decision_frozen_history.json")),
        ("adaptive (filtered)", J("decision_adaptive_history.json")),
        ("adaptive (buffer)", J("decision_adaptive_buffer_history.json")),
        ("hybrid (margin + J)", J("decision_hybrid_buffer_history.json")),
        ("J-only (no margin)", J("decision_hybrid_buffer_jonly_history.json"))]
def tr(h, k): return [e["step"] for e in h["evals"]], [e[k] for e in h["evals"]]

fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(12.5, 3.9))
for (lab, h), c in zip(ARMS, C):
    s, v = tr(h, "flip")
    a1.plot(s, v, color=c, lw=2, marker="o", ms=3.5, label=lab)
a1.axhspan(0.43, 0.57, color="#8a8880", alpha=0.13, zorder=0)
a1.text(0.02, 0.585, "letter-policy band", transform=a1.get_yaxis_transform(),
        ha="left", va="bottom", fontsize=8, color=INK2)
a1.set_ylim(-0.03, 1.05); a1.set_xlabel("step"); a1.set_ylabel("targeted A/B flip rate")
a1.set_title("(a) preference installed", loc="left", fontsize=10.5, color=INK)


for (lab, h), c in zip(ARMS, C):
    s, v = tr(h, "fracA")
    a2.plot(s, v, color=c, lw=2, marker="o", ms=3.5)
a2.axhline(0.5, color=INK2, lw=0.8, ls=":")
a2.text(0.98, 0.44, "balanced (per-question)", transform=a2.get_yaxis_transform(),
        ha="right", fontsize=8, color=INK2)
a2.set_ylim(-0.05, 1.05); a2.set_xlabel("step"); a2.set_ylabel('fraction answering "A"')
a2.set_title("(b) the letter attractor, watched directly", loc="left", fontsize=10.5, color=INK)

keys = [("flip", "A/B\n(trained)"), ("know_ab_flip", "know\nA/B"), ("ood_digits_flip", "OOD\ndigits"),
        ("ood_sum_flip", "OOD\nsum"), ("free_flip", "free-\nform"), ("easy", "easy add\n(capability)")]
finals = [("hybrid (margin + J)", ARMS[3][1]["evals"][-1], C[3]),
          ("adaptive (buffer)", ARMS[2][1]["evals"][-1], C[2]),
          ("J-only (no margin)", ARMS[4][1]["evals"][-1], C[4])]
x = np.arange(len(keys)); w = 0.26
for i, (lab, f, c) in enumerate(finals):
    a3.bar(x + (i - 1) * w, [f.get(k, np.nan) for k, _ in keys], w * 0.92, color=c, label=lab, zorder=2)
a3.set_xticks(x, [lab for _, lab in keys], fontsize=7.5)
a3.set_ylim(0, 1.08); a3.grid(axis="x", visible=False); a3.set_ylabel("rate")
a3.set_title("(c) final profile (step 300 / 600)", loc="left", fontsize=10.5, color=INK)

fig.suptitle("Decision-position probes — only the two-head hybrid installs the full balanced preference\n"
             "(activations steer, likelihoods select; Qwen2.5-3B, L*=23, one seed each)",
             fontsize=11.5, color=INK, x=0.02, ha="left")
handles, labels = a1.get_legend_handles_labels()
fig.legend(handles, labels, fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 0.865))
fig.tight_layout(rect=(0, 0.02, 1, 0.80))
fig.savefig("plots/fig7_decision_position.png", dpi=150); plt.close(fig)
print("saved:", sorted(os.listdir("plots")))
