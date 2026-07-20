"""Figures from the run JSONs (phase-2 pilot, 2026-07-20). Static PNGs -> plots/."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a"]  # validated categorical, fixed order
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
os.makedirs("plots", exist_ok=True)

def J(f): return json.load(open(f))
def steps(r): return [c["step"] for c in r["curve"]]
def series(r, k): return [c.get(k, np.nan) for c in r["curve"]]

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "text.color": INK, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.7,
    "legend.frameon": False,
})

def letter_band(ax):
    ax.axhspan(0.43, 0.57, color="#8a8880", alpha=0.13, zorder=0)
    ax.text(0.99, 0.505, "letter-policy band", transform=ax.get_yaxis_transform(),
            ha="right", va="center", fontsize=8, color=INK2)

def panel(ax, runs, title, key="ab_flip"):
    for (label, r), c in zip(runs, C):
        ax.plot(steps(r), series(r, key), color=c, lw=2, marker="o", ms=4, label=label)
    ax.set_title(title, fontsize=10.5, color=INK, loc="left")
    ax.set_ylim(-0.03, 1.05); ax.set_xlabel("step")
    letter_band(ax)
    ax.legend(fontsize=8, loc="best")

# ---- Fig 1: trajectories, 2x2 ----
fig, axs = plt.subplots(2, 2, figsize=(11.5, 7.2), sharey=True)
panel(axs[0, 0], [
    ("L21", J("run_rl_block_L21_p0.5_a0.0_lr1e-4_s1.json")),
    ("L21, pess=0", J("run_rl_block_L21_p0.0_a0.0_lr1e-4_s1.json")),
    ("L25", J("run_rl_block_L25_p0.5_a0.0_lr1e-4_s1.json")),
    ("L35, lr 3e-4 (collapse)", J("run_rl_final_L35_p0.5_a0.0_lr3e-4_s1_COLLAPSED.json")),
], "(a) RL from probe, no anchor — trapped at the letter shelf")
panel(axs[0, 1], [
    ("L35 + anchor", J("run_rl_final_L35_p0.5_a1.0_lr1e-4_s1.json")),
    ("L21 + anchor", J("rl_elbow_rl_block_L21_p0.5_a1.0_s1.json")),
], "(b) RL from probe + DPOP anchor — full targeted flip")
panel(axs[1, 0], [
    ("lr 5e-5, 300 steps", J("run_dpo_final_L35_p0.5_a0.0_lr5e-5_s1.json")),
    ("lr 5e-5, 600 steps", J("run_dpo_final_L35_p0.5_a0.0_lr5e-05_s1.json")),
    ("lr 1e-4", J("run_dpo_final_L35_p0.5_a0.0_lr0.0001_s1.json")),
    ("lr 2e-4", J("run_dpo_final_L35_p0.5_a0.0_lr0.0002_s1.json")),
], "(c) DPO — the letter transition is slow, not absent")
panel(axs[1, 1], [
    ("deep (≤L21)", J("hyb_deep_L21_lr0.0001_s1.json")),
    ("deep_rl (L21)", J("hyb_deep_rl_L21_lr0.0001_s1.json")),
    ("deep_rl (L31)", J("hyb_deep_rl_L31_lr0.0001_s1.json")),
], "(d) Hybrid backprop≤L + REINFORCE>L — 3B does not replicate 7B")
axs[0, 0].set_ylabel("targeted A/B flip rate"); axs[1, 0].set_ylabel("targeted A/B flip rate")
fig.suptitle("A/B wrongness testbed — trajectories by method (Qwen2.5-3B, one seed each)",
             fontsize=12, color=INK, x=0.02, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig("plots/fig1_trajectories.png", dpi=150); plt.close(fig)

# ---- Fig 2: letter-attractor view (fracA) ----
fig, ax = plt.subplots(figsize=(8.2, 4.2))
runs = [
    ("DPO lr 5e-5 (seed 2)", J("rl_elbow_dpo_final_L35_p0.5_a0.0_s2.json")),
    ("DPO lr 5e-5, 600 steps", J("run_dpo_final_L35_p0.5_a0.0_lr5e-05_s1.json")),
    ("DPO lr 1e-4", J("run_dpo_final_L35_p0.5_a0.0_lr0.0001_s1.json")),
    ("RL L21 + anchor", J("rl_elbow_rl_block_L21_p0.5_a1.0_s1.json")),
    ("hybrid deep_rl L31", J("hyb_deep_rl_L31_lr0.0001_s1.json")),
]
for (label, r), c in zip(runs, C):
    ax.plot(steps(r), series(r, "fracA"), color=c, lw=2, marker="o", ms=4, label=label)
ax.set_ylim(-0.05, 1.05); ax.set_xlabel("step"); ax.set_ylabel('fraction answering "A"')
ax.axhline(0.5, color=INK2, lw=0.8, ls=":")
ax.text(3, 0.52, "balanced (per-question answers)", fontsize=8, color=INK2)
ax.set_title("The letter attractor, watched directly — fracA per checkpoint", loc="left", color=INK, fontsize=11)
ax.legend(fontsize=8, ncol=2)
fig.tight_layout(); fig.savefig("plots/fig2_letter_attractor.png", dpi=150); plt.close(fig)

# ---- Fig 3: final outcomes, grouped bars ----
arms = [
    ("DPO lr 1e-4", J("run_dpo_final_L35_p0.5_a0.0_lr0.0001_s1.json")["final"]),
    ("RL+anchor @ L35", J("run_rl_final_L35_p0.5_a1.0_lr1e-4_s1.json")["final"]),
    ("RL+anchor @ L21", J("rl_elbow_rl_block_L21_p0.5_a1.0_s1.json")["final"]),
]
flip_keys = [("ab_flip", "A/B\n(trained)"), ("know_ab_flip", "know A/B\n(held-out)"),
             ("ood_digits_flip", "OOD digits"), ("ood_sum_flip", "OOD sum"),
             ("free_flip", "free-form"), ("yn_flip", "Yes/No")]
cap_keys = [("mcq", "arith MCQ"), ("easy", "easy addition"), ("know_free", "know free")]
BASE = json.load(open("base_evals.json"))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.0), gridspec_kw={"width_ratios": [2, 1]})
w = 0.26
for a, keys, title, ylab in ((a1, flip_keys, "Preference installed — targeted flip rate", "flip rate"),
                             (a2, cap_keys, "Capabilities — correct rate", "correct rate")):
    x = np.arange(len(keys))
    for i, ((label, f), c) in enumerate(zip(arms, C)):
        vals = [f[k] for k, _ in keys]
        a.bar(x + (i - 1) * w, vals, w * 0.92, color=c, label=label, zorder=2)
    for j, (k, _) in enumerate(keys):   # base-model reference tick per group
        a.hlines(BASE[k], j - 1.55 * w, j + 1.55 * w, color=INK, lw=1.4, ls=(0, (3, 2)), zorder=3)
    a.set_xticks(x, [lab for _, lab in keys], fontsize=8.5)
    a.set_ylim(0, 1.05); a.set_title(title, loc="left", fontsize=10.5, color=INK)
    a.set_ylabel(ylab); a.grid(axis="x", visible=False)
a2.hlines(0.5, -1.55 * w, 1.55 * w, color=INK2, lw=1.0, ls=":", zorder=3)
a2.text(0, 0.455, "2-choice chance", fontsize=7.5, color=INK2, ha="center")
a1.plot([], [], color=INK, lw=1.4, ls=(0, (3, 2)), label="base model")
a1.legend(fontsize=8)
fig.suptitle("Final evals — the three arms that install the preference (3B, one seed)",
             fontsize=12, color=INK, x=0.02, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig("plots/fig3_final_bars.png", dpi=150); plt.close(fig)

# ---- Fig 4: UF DPO training ----
h = J("uf_dpo_history.json")
loss = np.array(h["loss"]); ev = h["evals"]
fig, (b1, b2, b3) = plt.subplots(1, 3, figsize=(11.5, 3.6))
b1.plot(np.arange(1, len(loss) + 1), loss, color=C[0], lw=0.8, alpha=0.35)
k = 20; roll = np.convolve(loss, np.ones(k) / k, mode="valid")
b1.plot(np.arange(k, len(loss) + 1), roll, color=C[0], lw=2, label="rolling mean (20)")
b1.set_title("DPO loss", loc="left", fontsize=10.5); b1.set_xlabel("step"); b1.legend(fontsize=8)
es = [e["step"] for e in ev]
b2.plot(es, [e["acc_implicit"] for e in ev], color=C[1], lw=2, marker="o", ms=5)
b2.axhline(0.5, color=INK2, lw=0.8, ls=":"); b2.text(5, 0.515, "chance", fontsize=8, color=INK2)
b2.set_ylim(0.35, 0.9); b2.set_title("held-out implicit-reward accuracy", loc="left", fontsize=10.5)
b2.set_xlabel("step")
b3.plot(es, [e["dlp_chosen"] for e in ev], color=C[0], lw=2, marker="o", ms=5, label="Δlp chosen")
b3.plot(es, [e["dlp_rejected"] for e in ev], color=C[3], lw=2, marker="o", ms=5, label="Δlp rejected")
b3.axhline(0, color=INK2, lw=0.8, ls=":")
b3.set_title("displacement (nats vs reference)", loc="left", fontsize=10.5)
b3.set_xlabel("step"); b3.legend(fontsize=8)
fig.suptitle("UltraFeedback DPO — Tulu-3-8B-SFT + LoRA, 400 steps",
             fontsize=12, color=INK, x=0.02, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig("plots/fig4_uf_dpo.png", dpi=150); plt.close(fig)
print("saved:", os.listdir("plots"))
