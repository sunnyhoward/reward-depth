#!/usr/bin/env python
"""Build results_libon_0806.md + plots from whatever artifacts exist. Re-runnable at any time;
missing arms are simply omitted, so it can be run mid-sweep to see progress.

Out: /workspace/reward-depth/results_libon_0806.md
     /workspace/reward-depth/results/runs/libon/*.png
"""
import os, sys, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/workspace/libon"
REPO = "/workspace/reward-depth"
PLOTS = f"{REPO}/results/runs/libon"
os.makedirs(PLOTS, exist_ok=True)
MD = f"{REPO}/results_libon_0806.md"

C = dict(base="#444444", frozen="#B45309", continuous="#0F766E", retrained="#7C3AED")
LAMC = {"0": "#0F766E", "0.5": "#0891B2", "1": "#7C3AED", "2": "#BE123C"}


def jload(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def evals():
    out = {}
    for p in sorted(glob.glob(f"{OUT}/eval_*.json")):
        d = jload(p)
        if d:
            out[d.get("tag", os.path.basename(p)[5:-5])] = d
    return out


def histories():
    out = {}
    for p in sorted(glob.glob("/workspace/libon_*/history.json")):
        d = jload(p)
        if d:
            out[os.path.basename(os.path.dirname(p)).replace("libon_", "")] = d
    return out


EV, HI = evals(), histories()
gate = jload(f"{OUT}/bayes_gate.json")
base = EV.get("base")
floor = 0.8 * base["utility"]["score"] if base else None

# ───────────────────────── plot 1: compliance vs utility ─────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
for tag, d in EV.items():
    if "judge" not in d or "utility" not in d:
        continue
    x, y = d["utility"]["score"], d["judge"]["compliance"]
    brk = d["judge"].get("broken", 0)
    col = C["base"]
    for k, c in C.items():
        if tag.startswith(k):
            col = c
    if tag.startswith("bayes"):
        lam = tag.split("lam")[-1].split("_")[0] if "lam" in tag else "0"
        col = LAMC.get(lam, "#334155")
    ax.scatter(x, y, s=60 + 400 * brk, color=col, alpha=.85, edgecolor="white", zorder=3)
    ax.annotate(tag.replace("_ckpt", "@"), (x, y), fontsize=7,
                xytext=(4, 4), textcoords="offset points")
if floor:
    ax.axvline(floor, ls="--", lw=1, color="#DC2626")
    ax.text(floor, ax.get_ylim()[1], " 80% utility floor", color="#DC2626", fontsize=8, va="top")
ax.set_xlabel("utility (MMLU/GSM8K mean)"); ax.set_ylabel("harmful compliance")
ax.set_title("Compliance vs utility  (marker size = judged 'broken' rate)", fontsize=10)
ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(f"{PLOTS}/compliance_vs_utility.png", dpi=130); plt.close(fig)

# ───────────────────────── plot 2: per-layer evidence weights ─────────────────────────
ew = [(t, h) for t, h in HI.items() if h.get("probe_info") and h.get("layer_w") == "evidence"]
if not ew:
    ew = [(t, h) for t, h in HI.items() if h.get("probe_info")]
if ew:
    fig, axes = plt.subplots(1, len(ew[:2]), figsize=(6 * len(ew[:2]), 4), squeeze=False)
    for k, (tag, h) in enumerate(ew[:2]):
        a = axes[0][k]
        steps = [i + 1 for i, x in enumerate(h["probe_info"]) if x]
        L = h["layers"]
        for l in L:
            w = [x["weights"][str(l)] for x in h["probe_info"] if x]
            a.plot(steps, w, label=f"L{l}", lw=1.6)
        a.set_title(f"{tag}: per-layer weight", fontsize=10)
        a.set_xlabel("step"); a.set_ylabel("weight"); a.grid(alpha=.25); a.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(f"{PLOTS}/evidence_weights.png", dpi=130); plt.close(fig)

# ───────────────────────── plot 3: rotation angle ─────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
for tag, h in HI.items():
    if not h.get("angles"):
        continue
    m = [np.mean(list(a.values())) for a in h["angles"]]
    ax.plot(range(1, len(m) + 1), m, lw=1.6, label=tag)
ax.set_xlabel("step"); ax.set_ylabel("mean angle vs initial direction (deg)")
ax.set_title("Probe direction rotation (their Fig 8)", fontsize=10)
ax.grid(alpha=.25); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{PLOTS}/rotation.png", dpi=130); plt.close(fig)

# ───────────────────────── plot 4: translation diagnostic ─────────────────────────
tr = [(t, h) for t, h in HI.items() if h.get("evals") and "translation" in h["evals"][0]]
if tr:
    fig, ax = plt.subplots(figsize=(7, 4))
    for tag, h in tr:
        s = [e["step"] for e in h["evals"]]
        ax.plot(s, [e["translation"]["harmful"] for e in h["evals"]], lw=1.6, label=f"{tag} harmful")
        ax.plot(s, [e["translation"]["benign"] for e in h["evals"]], lw=1.2, ls="--",
                label=f"{tag} benign")
    ax.set_xlabel("step"); ax.set_ylabel("score under CAPTURED INITIAL probe")
    ax.set_title("Translation diagnostic (their Fig 12): both classes sliding = evasion", fontsize=10)
    ax.grid(alpha=.25); ax.legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(f"{PLOTS}/translation.png", dpi=130); plt.close(fig)

# ───────────────────────── markdown ─────────────────────────
def row(tag, d):
    j, u = d.get("judge", {}), d.get("utility", {})
    dh = d.get("degeneracy_harmful", {}).get("rate", float("nan"))
    db = d.get("degeneracy_benign", {}).get("rate", float("nan"))
    flag = ""
    if floor and u.get("score", 9) < floor:
        flag = " **:warning: below budget**"
    return (f"| {tag} | {j.get('compliance', float('nan')):.3f} | {j.get('refusal', float('nan')):.3f} "
            f"| {j.get('soft', float('nan')):.3f} | {j.get('broken', float('nan')):.3f} "
            f"| {dh:.3f} | {db:.3f} | {u.get('score', float('nan')):.3f}{flag} |")


L = []
L.append("# Libon port + Bayesian probes — live results (2026-08-06)\n")
L.append("*Auto-generated by `libon/libon_report.py`; re-run at any time. "
         "Arms still training are absent rather than partial.*\n")
L.append("Model `mistralai/Mistral-7B-Instruct-v0.1`, layers {0,6,12,18,24,30}, LoRA r=64 "
         "a=128 on q,k,v,o,up,down, beta=2, 8 harmful + 8 benign prompts/step, on-policy "
         "completions, KL on 16 UltraChat prompts. Single seed.\n")
if floor:
    L.append(f"**Utility floor (80% of base {base['utility']['score']:.3f}) = {floor:.3f}.** "
             "Checkpoints below it are past the paper's own stopping rule.\n")

IFE = {}
for p in sorted(glob.glob(f"{OUT}/ifeval_*.json")):
    d = jload(p)
    if d:
        IFE[d["tag"]] = d

if IFE:
    L.append("\n## The utility budget does not fire — and IFEval does not save it\n")
    L.append("Their stopping rule is 80% of base utility (MMLU + GSM8K + IFEval). Measured here:\n")
    L.append("| arm | MMLU+GSM8K | IFEval strict | IFEval loose | deg on IFEval | deg on short benign |")
    L.append("|---|---|---|---|---|---|")
    for t, d in sorted(IFE.items()):
        u = EV.get(t, {}).get("utility", {}).get("score", float("nan"))
        db = EV.get(t, {}).get("degeneracy_benign", {}).get("rate", float("nan"))
        L.append(f"| {t} | {u:.3f} | {d['strict']:.3f} | {d['loose']:.3f} "
                 f"| {d['degenerate']:.3f} | {db:.3f} |")
    L.append("""
`continuous_ckpt75` produces a degenerate word loop for **100%** of short benign prompts, yet
scores at base level on all three utility components. Reading the raw text explains why — the
collapse is specific to the prompt distribution the probe loss was applied to:

```
SHORT BENIGN ("What are traditional systems for resource allocation and sharing?")
  -> 'HochAAAA " " " " " " " " " " " " " " " " " " " " ...'

IFEVAL ("I am planning a trip to Japan, and I would like thee to write an itinerary...")
  -> 'Verily, thou hast chosen a noble quest indeed! I shall endeavor to craft an
      itinerary for thee in the style of the great bard himself. Day One: Arriv...'
```

MMLU is scored by token likelihood over four choices, GSM8K by extracting a final number, and
IFEval prompts are long and structured — none of them sample the short instruction-style
distribution where the damage lives. **The 80%-of-base rule never fires on a model that cannot
answer "what's the weather like today".** The guard that does fire is degeneracy measured on
held-out prompts from the training distribution, which is what our in-loop coherence probe does.

This is the mirror of this project's 08-05 finding (`results_0805.md` section 8.1): a replay
corpus that does not match the operating distribution is inert as a prior; a utility metric that
does not match it is inert as a guard.
""")

L.append("\n## Main table\n")
L.append("| arm | comply | refusal | soft/pseudo | broken | deg(harmful) | deg(benign) | utility |")
L.append("|---|---|---|---|---|---|---|---|")
order = ([("base", EV["base"])] if base else []) + \
        [(t, d) for t, d in sorted(EV.items()) if t != "base"]
for t, d in order:
    L.append(row(t, d))

if gate:
    L.append("\n## Section 1 — Bayesian head sanity gate: **%s**\n" % gate["verdict"])
    L.append("| layer | logistic AUROC | Bayes AUROC | delta | ELBO | evidence weight |")
    L.append("|---|---|---|---|---|---|")
    for l in map(str, gate["layers"]):
        L.append(f"| L{l} | {gate['logistic']['auroc'][l]:.3f} | {gate['bayes']['auroc'][l]:.3f} "
                 f"| {gate['delta_auroc'][l]:+.3f} | {gate['bayes']['elbo'][l]:.0f} "
                 f"| {gate['bayes']['evidence_weights'][l]:.3f} |")
    L.append(f"\nMean delta AUROC {gate['mean_delta']:+.3f}, 2*SE {2*gate['se']:.3f} "
             f"-> **{gate['verdict']}**. Mean posterior sigma is ~"
             f"{np.mean(list(gate['bayes']['mean_sigma'].values())):.3f} against a prior tau of "
             "0.1, i.e. the posterior is still close to the prior in most directions (d=4096, "
             "n=652) — worth remembering when reading the pessimism sweep.\n")

if HI:
    L.append("\n## Training-time diagnostics\n")
    L.append("| run | steps | final mean angle | 80% budget crossed at | final deg(benign) |")
    L.append("|---|---|---|---|---|")
    for t, h in sorted(HI.items()):
        ang = np.mean(list(h["angles"][-1].values())) if h.get("angles") else float("nan")
        cx = h.get("crossed_80pct_at", "not crossed")
        dg = h["evals"][-1].get("degenerate", float("nan")) if h.get("evals") else float("nan")
        L.append(f"| {t} | {len(h.get('loss', []))} | {ang:.1f}deg | {cx} | {dg:.2f} |")

L.append("\n## Plots\n")
for f, cap in [("compliance_vs_utility.png", "Compliance vs utility; marker size = judged "
                                             "'broken' rate; dashed line = 80% budget"),
               ("rotation.png", "Probe direction rotation vs initial (their Fig 8)"),
               ("evidence_weights.png", "Per-layer weights over training (section 4)"),
               ("translation.png", "Translation diagnostic (their Fig 12)")]:
    if os.path.exists(f"{PLOTS}/{f}"):
        L.append(f"**{cap}**\n\n![{f}](results/runs/libon/{f})\n")

open(MD, "w").write("\n".join(L) + "\n")
print(f"wrote {MD}")
print(f"plots in {PLOTS}: " + ", ".join(sorted(os.path.basename(p)
                                               for p in glob.glob(f"{PLOTS}/*.png"))))
