#!/usr/bin/env python
"""Figure: mean-diff install vs READ DEPTH at matched 300 steps (2026-08-17).

Two panels, one per judged family. x = read block L, y = judged British score
(unconditional: non-engagement scored 50, the rubric's own neutral anchor), paired
per prompt against the same 48 held-out prompts. Horizontal lines are the base model
and P1 (plain DPOP at the output, LoRA 0-31, same data and step budget). The second
axis carries the chat-template leakage rate from pf_leakage.py, because the two
mid-late points buy their (small) install at the cost of wrecked text.
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rep = json.load(open(f"{REPO}/results/probefix_cjudge_mdlate/report.json"))["per_arm"]
leak = {r["arm"]: r["leak_rate"] for r in json.load(open(f"{REPO}/results/probefix/leakage_scan.json"))}
LAYERS = [20, 24, 26, 28, 30, 31]
LEAKKEY = {20: "MD_L20_r0_noreplay_s300", 24: "MD_L24_s300", 26: "MD_L26_s300",
           28: "MD_L28_s300", 30: "MD_L30_s300", 31: "MD_L31_s300"}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, fam in zip(axes, ("false_friend", "style")):
    y = [rep[f"{fam}/MD_L{L}_s300"]["british_uncond"] for L in LAYERS]
    e = [rep[f"{fam}/MD_L{L}_s300"]["british_uncond_se"] for L in LAYERS]
    ax.errorbar(LAYERS, y, yerr=e, marker="o", color="#1f3d7a", lw=2, capsize=3,
                label="mean-diff, read at L (300 steps)")
    ax.axhline(rep[f"{fam}/base"]["british_uncond"], ls=":", color="#666", label="base")
    ax.axhline(rep[f"{fam}/P1_r1"]["british_uncond"], ls="--", color="#a11", 
               label="plain DPOP at output")
    ax2 = ax.twinx()
    ax2.bar(LAYERS, [leak.get(LEAKKEY[L], 0) for L in LAYERS], width=1.1, alpha=0.16,
            color="#c0392b", zorder=0)
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel("template-leakage rate" if fam == "style" else "")
    ax.set_xlabel("read block L (of 32)")
    ax.set_title(fam)
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
axes[0].set_ylabel("judged British (unconditional)")
axes[0].legend(fontsize=8, loc="upper left")
fig.suptitle("Activation-space install falls as the read point gets later", y=1.0)
fig.tight_layout()
out = f"{REPO}/results/probefix/mdlate_readdepth.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("->", out)
