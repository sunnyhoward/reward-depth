#!/usr/bin/env python
"""Collect the EAGLE matrix into the spec's deliverables: one results table (markdown + JSON)
and two plots — (1) final behavioural quality vs stage-1 plateau-meter accuracy, (2) optimal L
per factor. Behavioural target (FLIP=1 runs): style -> terse rate (1 - gen_explained),
correct -> gen_wrong. States plainly when a cell is null at single-seed budget.

Reads /workspace/eagle_*/history.json; writes /workspace/eagle_results.{json,md} and
/workspace/eagle_plot_{quality_vs_meter,optimal_L}.png"""
import os, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# fixed factor->hue (Okabe-Ito colorblind-safe pair), identity never re-mapped
C = {"style": "#0072B2", "correct": "#D55E00"}

def target_rate(ev, factor):
    return (1.0 - ev["gen_explained"]) if factor == "style" else ev["gen_wrong"]

rows = []
for f in sorted(glob.glob("/workspace/eagle_*/history.json")):
    h = json.load(open(f))
    tag = h.get("tag", os.path.basename(os.path.dirname(f)))
    if tag.startswith("smoke") or "smoke" in tag: continue
    factor = h.get("factor")
    evs = h["evals"]
    fin = evs[-1]
    tgt = [target_rate(e, factor) for e in evs]
    best_i = int(np.argmax(tgt))
    cond = ("s1" if tag.startswith("s1_") else "s2" if tag.startswith("s2_") else
            "fulldpo" if tag.startswith("fulldpo") else "upperonly")
    rows.append(dict(
        tag=tag, cond=cond, factor=factor, L=h.get("L"),
        s1_ckpt=os.path.basename(h.get("s1_ckpt", "")) or None,
        target_final=round(tgt[-1], 3), target_best=round(tgt[best_i], 3),
        best_step=evs[best_i]["step"],
        acc_fam1=round(fin.get("acc_style_c" if factor == "style" else "acc_corr_e", float("nan")), 3),
        gen_correct=round(fin["gen_correct"], 3), gen_explained=round(fin["gen_explained"], 3),
        gen_wrong=round(fin["gen_wrong"], 3), gen_len=round(fin["gen_len_words"], 1),
        kl_from_base=round(fin.get("kl_from_base", float("nan")), 3),
        head_acc_final=round(fin.get("head_acc", float("nan")), 3) if "head_acc" in fin else None))
json.dump(rows, open("/workspace/eagle_results.json", "w"), indent=1)

cols = ["tag", "cond", "factor", "L", "target_final", "target_best", "best_step",
        "gen_wrong", "gen_explained", "gen_len", "kl_from_base", "head_acc_final"]
with open("/workspace/eagle_results.md", "w") as f:
    f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
    for r in sorted(rows, key=lambda r: (r["factor"] or "", r["cond"], r["L"] or 0)):
        f.write("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n")
print(open("/workspace/eagle_results.md").read())

# ---- plot 1: behavioural quality (stage-2 final target rate) vs stage-1 plateau meter ----
fig, ax = plt.subplots(figsize=(5.2, 4))
for fac in ("style", "correct"):
    xs, ys, ls_ = [], [], []
    for r in rows:
        if r["cond"] != "s2" or r["factor"] != fac: continue
        s1tag = f"s1_{fac}_L{r['L']}_flip"
        try:
            h1 = json.load(open(f"/workspace/eagle_{s1tag}/history.json"))
            ck = int(r["s1_ckpt"][4:])
            ha = next(e["head_acc"] for e in h1["evals"] if e["step"] == ck)
        except Exception:
            continue
        xs.append(ha); ys.append(r["target_final"]); ls_.append(r["L"])
    ax.scatter(xs, ys, color=C[fac], s=48, label=fac)
    for x, y, L in zip(xs, ys, ls_):
        ax.annotate(f"L{L}", (x, y), textcoords="offset points", xytext=(6, 3), fontsize=8,
                    color="#555555")
ax.set_xlabel("stage-1 through-head accuracy at chosen ckpt")
ax.set_ylabel("free-sampling target rate after stage 2")
ax.set_title("Behavioural install vs stage-1 meter", fontsize=11)
ax.legend(frameon=False); ax.grid(alpha=0.25, linewidth=0.5)
fig.tight_layout(); fig.savefig("/workspace/eagle_plot_quality_vs_meter.png", dpi=150)

# ---- plot 2: optimal L per factor (stage-2 final target vs L; baselines as reference lines) ----
fig, ax = plt.subplots(figsize=(5.2, 4))
for fac in ("style", "correct"):
    pts = sorted([(r["L"], r["target_final"]) for r in rows if r["cond"] == "s2" and r["factor"] == fac])
    if pts:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=C[fac], linewidth=2,
                markersize=6, label=f"{fac} (two-stage)")
    fd = [r["target_final"] for r in rows if r["cond"] == "fulldpo" and r["factor"] == fac]
    if fd:
        ax.axhline(fd[0], color=C[fac], linestyle="--", linewidth=1.2, alpha=0.7)
        ax.annotate(f"{fac} full DPO", (ax.get_xlim()[1], fd[0]), fontsize=8, color=C[fac],
                    ha="right", va="bottom")
    uo = [(r["L"], r["target_final"]) for r in rows if r["cond"] == "upperonly" and r["factor"] == fac]
    for L, v in uo:
        ax.scatter([L], [v], color=C[fac], marker="x", s=40)
ax.set_xlabel("attach layer L"); ax.set_ylabel("free-sampling target rate (final)")
ax.set_title("Install vs attach depth (x = upper-only baseline)", fontsize=11)
ax.legend(frameon=False); ax.grid(alpha=0.25, linewidth=0.5)
fig.tight_layout(); fig.savefig("/workspace/eagle_plot_optimal_L.png", dpi=150)
print("plots written", flush=True)
