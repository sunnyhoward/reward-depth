#!/usr/bin/env python
"""Tables and figures for the probe-decoder arms.

WHAT IS PLOTTED, and why each panel exists:

  1. INSTALL vs GUARD over training, one line per arm. Two panels rather than one axis with two
     scales: they are different measures (install has a 0.06-0.26 base rate and all the headroom;
     the guard starts at 1.00 and can only be damaged), and putting them on one axis would make
     the guard look flat by construction.
  2. FUNCTION DRIFT BY DEPTH -- relative L2 between each arm's residual and the base model's, on
     held-out text. This is the "where did the model change" panel that is comparable across arms:
     weight deltas can only appear where an arm was allowed to write, but the function can change
     at any depth downstream of a write.
  3. WEIGHT DELTA BY LAYER -- ||BA*alpha/r||_F / ||W||_F. What actually moved, where it was
     allowed to move.
  4. FRESH-PROBE DEPTH CURVE, base vs arms, guard column. Whether the composite rule became more
     decodable and where -- refitted independently, so a co-trained probe's own number cannot
     flatter it.

Colour is CATEGORICAL (arms are unordered identities), assigned in fixed slot order and never
cycled; validated with `validate_palette.js --mode light` (all checks PASS, contrast WARN on three
slots -> the relief rule applies, so every line is also direct-labelled at its right end and
carries its own dash pattern). Grey is reserved for the base model, which is not an arm.

Usage: python pf_report.py [--root /workspace/probefix] [--outdir results/probefix]
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE, INK, INK_2, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1"
BASE_GREY = "#8a8985"
# Fixed slot order, never cycled. (blue, orange, aqua, yellow, magenta)
ARMS = [
    ("A_dpo_all",   "A  DPO all layers (control)", "#2a78d6", "solid"),
    ("B_probe_L12", "B  probe stage 1 @L12",       "#eb6834", (0, (4, 1.6))),
    ("C1_s2_upper", "C1 B -> DPO upper only",      "#1baf7a", (0, (7, 1.6, 1.5, 1.6))),
    ("C2_s2_full",  "C2 B -> DPO full stack",      "#eda100", (0, (1, 1.6))),
    ("D_dpo_upper", "D  DPO upper only (no s1)",   "#e87ba4", (0, (5, 1.6, 1.5, 1.6, 1.5, 1.6))),
]


def style(ax, xlabel="", ylabel="", title=""):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8.5, length=0)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    if title:
        ax.set_title(title, color=INK, fontsize=10.5, loc="left", pad=8)


def endlabel(ax, x, y, text, color):
    """Direct label at the right end of a line -- the relief the contrast WARN obligates."""
    ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points", color=color,
                fontsize=7.5, va="center", ha="left", clip_on=False)


def load(root):
    hist, audit = {}, {}
    for tag in [t for t, *_ in ARMS] + ["B2_probe_bt", "R_replay_only", "C3_s2_upper_bt",
                                        "D_dpo_upper_600", "C1_s2_upper_600", "C2_s2_full_600"]:
        p = f"{root}/{tag}/history.json"
        if os.path.exists(p):
            hist[tag] = json.load(open(p))
    for p in glob.glob(f"{root}/audit/*.json"):
        n = os.path.basename(p)[:-5]
        if not n.startswith("_"):
            audit[n] = json.load(open(p))
    return hist, audit


def fig_training(hist, out):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), facecolor=SURFACE)
    for ax, col, ttl in ((axes[0], "install", "Install — pooled held-out raw ranking"),
                         (axes[1], "guard", "Guard — held-out raw ranking (base 1.00)")):
        for tag, lab, c, dash in ARMS:
            h = hist.get(tag)
            if not h:
                continue
            xs = [e["step"] for e in h["evals"]]
            if col == "guard":
                ys = [e["buckets"]["guard"]["raw"] for e in h["evals"]]
            else:
                ys = [float(np.mean([v["raw"] for k, v in e["buckets"].items() if k != "legacy"]))
                      for e in h["evals"]]
            ax.plot(xs, ys, color=c, lw=2, ls=dash, zorder=3, label=lab)
            endlabel(ax, xs[-1], ys[-1], tag.split("_")[0], c)
        ax.set_ylim(-0.02, 1.05)
        style(ax, "step", "accuracy", ttl)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"  -> {out}")


def fig_pareto(hist, out):
    """Install against guard, one curve per arm, points = checkpoints in step order.

    THE ENDPOINT COMPARISON IS THE WRONG COMPARISON on this dataset. Every arm reaches ~0.99 on
    the legacy bucket and every arm's guard falls from 1.000 only AFTER install saturates, so
    quoting arms at a fixed step compares each one at a different place on its own trade-off
    curve. What is comparable is the FRONTIER: at a given guard level, how much install does the
    arm buy. Up and to the right is better; the base model sits at the bottom-right corner
    (guard 1.000, install 0.236) and every arm walks left as it trains.
    """
    fig, ax = plt.subplots(figsize=(6.6, 5.0), facecolor=SURFACE)
    for tag, lab, c, dash in ARMS:
        h = hist.get(tag)
        if not h:
            continue
        g = [e["buckets"]["guard"]["raw"] for e in h["evals"]]
        y = [float(np.sum([v["raw"] * v["n"] for k, v in e["buckets"].items()
                           if k.startswith("install")])
                   / np.sum([v["n"] for k, v in e["buckets"].items() if k.startswith("install")]))
             for e in h["evals"]]
        ax.plot(g, y, color=c, lw=2, ls=dash, marker="o", ms=4.5, mew=0, zorder=3, label=lab)
        endlabel(ax, g[-1], y[-1], tag.split("_")[0], c)
        for gi, yi, e in zip(g, y, h["evals"]):
            if e["step"] in (100, 300):
                ax.annotate(str(e["step"]), (gi, yi), xytext=(3, 5), textcoords="offset points",
                            color=INK_MUTED, fontsize=7)
    ax.scatter([1.0], [0.236], color=BASE_GREY, s=42, zorder=4)
    ax.annotate("base", (1.0, 0.236), xytext=(-6, -12), textcoords="offset points",
                color=BASE_GREY, fontsize=8, ha="right")
    ax.invert_xaxis()
    style(ax, "guard accuracy (held out, n=50 — base 1.000)", "install accuracy (pooled families)",
          "The trade: install bought against the guard")
    # Bottom-RIGHT: the low-guard/low-install corner is empty by construction (nothing gives up the
    # guard without buying install), so the legend sits there without covering a single point.
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"  -> {out}")


BUCKET_STYLE = [
    ("guard", "guard", "#2a78d6", "solid"),
    ("legacy", "legacy (lexicon)", "#eb6834", (0, (4, 1.6))),
    ("install_culture", "culture", "#1baf7a", (0, (7, 1.6, 1.5, 1.6))),
    ("install_truth_dialect", "truth_dialect", "#eda100", (0, (1, 1.6))),
    ("install_false_friend", "false_friend", "#e87ba4", (0, (5, 1.6, 1.5, 1.6, 1.5, 1.6))),
]


def fig_probe_trace(hist, out):
    """Stage 1's own meter over training, which no endpoint table shows.

    Row 1: held-out accuracy of the CO-TRAINED probe per eval bucket. This is the quantity stage 1
    optimises, and the one that must not be believed on its own -- `pf_audit`'s independently
    refitted probe is the check. Per bucket, because a pooled line hides the only interesting
    movement (the guard).

    Row 2: batch-mean margin `z` in units of sigma0 (the step-0 score spread), so the hinge target
    of 1.0 is a fixed line on the same axis.

    Row 3: `sat`, the fraction of the batch already above that target -- the share of the diet
    stage 1 declines to touch.

    THREE ROWS AND NOT AN OVERLAY: a margin in sigma and a fraction in [0,1] are different
    measures, and putting them on one axis with two scales makes the fraction unreadable (an
    earlier draft of this figure did exactly that and showed `sat` sitting at "10").
    """
    arms = [(t, lab) for t, lab in (("B_probe_L12", "B  saturating hinge"),
                                    ("B2_probe_bt", "B2 unbounded BT")) if t in hist]
    if not arms:
        return
    n = len(arms)
    fig, axes = plt.subplots(3, n, figsize=(5.6 * n, 10.2), facecolor=SURFACE, squeeze=False)
    for j, (tag, lab) in enumerate(arms):
        h = hist[tag]
        ev = h["evals"]
        xs = [e["step"] for e in ev]

        ax = axes[0][j]
        for k, klab, c, dash in BUCKET_STYLE:
            ys = [e["buckets"][k].get("probe") for e in ev]
            if any(y is None for y in ys):
                continue
            ax.plot(xs, ys, color=c, lw=2, ls=dash, marker="o", ms=3.5, mew=0, zorder=3, label=klab)
            endlabel(ax, xs[-1], ys[-1], klab.split()[0][:9], c)
        ax.axhline(0.5, color=INK_MUTED, lw=1, ls=(0, (2, 2)), zorder=1)
        ax.set_ylim(0.35, 1.06)
        style(ax, "step", "held-out accuracy", f"{lab} — co-trained probe accuracy at L*=12")

        parts = h["parts"]
        st = np.arange(1, len(parts) + 1)
        w = 10
        sm = lambda v: np.convolve(np.asarray(v), np.ones(w) / w, mode="valid")

        ax = axes[1][j]
        z = np.array([q["z"] for q in parts])
        ax.plot(st, z, color="#2a78d6", lw=0.7, alpha=0.3, zorder=2)
        ax.plot(st[w - 1:], sm(z), color="#2a78d6", lw=2, zorder=3)
        tgt = h.get("target", 1.0)
        ax.axhline(tgt, color=INK_MUTED, lw=1, ls=(0, (2, 2)), zorder=1)
        ax.annotate(f"hinge target = {tgt:g}σ", (st[-1], tgt), xytext=(-4, 5),
                    textcoords="offset points", color=INK_MUTED, fontsize=7.5, ha="right")
        style(ax, "step", "batch-mean margin, units of σ₀", f"{lab} — probe margin")

        ax = axes[2][j]
        sat = np.array([q["frac_sat"] for q in parts])
        ax.plot(st, sat, color="#eb6834", lw=0.7, alpha=0.3, zorder=2)
        ax.plot(st[w - 1:], sm(sat), color="#eb6834", lw=2, zorder=3)
        ax.set_ylim(-0.03, 1.05)
        # `sat` is logged for both objectives, but it only MEANS "no gradient" for the hinge:
        # Bradley-Terry never saturates, so for B2 this panel is a scale readout, not a duty cycle.
        hinge = h.get("pref_loss") == "hinge"
        note = (f"mean {sat.mean():.3f} · zero-gradient steps {float((sat == 1).mean()):.1%}"
                if hinge else
                f"mean {sat.mean():.3f} · BT never saturates — no step has zero gradient")
        ax.annotate(note, (st[-1], 0.02), xytext=(-4, 0), textcoords="offset points",
                    color=INK_2, fontsize=8, ha="right")
        style(ax, "step", "fraction of batch above 1σ",
              f"{lab} — pairs above target" + (" (gradient is zero at 1.0)" if hinge else ""))
    axes[0][0].legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"  -> {out}")


def fig_dw_pref(audit, out):
    """Preference-attributable weight change: each arm's per-layer ||dW||/||W|| MINUS the
    replay-only arm's.

    Raw dW cannot answer "where did the preference land": every arm's LoRA spans all 24 layers and
    its replay term trains all of them, so the raw profile is dominated by an anchor that has
    nothing to do with the preference. R_replay_only (W_PREF=0, everything else identical) is the
    baseline that has to come off first. What is left is a delta function at L* for the
    probe-attached objective and a broad peak at L* for output DPO.
    """
    R = audit.get("R_replay_only")
    if not R or "dw_by_layer" not in R:
        return
    def prof(a):
        tot = {}
        for per in a["dw_by_layer"].values():
            for L, v in per.items():
                tot[int(L)] = tot.get(int(L), 0.0) + v
        return tot
    base = prof(R)
    fig, ax = plt.subplots(figsize=(7.2, 4.4), facecolor=SURFACE)
    for tag, lab, c, dash in [("A_dpo_all", "A  DPO all layers", "#2a78d6", "solid"),
                              ("B_probe_L12", "B  probe @L12, saturating", "#eb6834", (0, (4, 1.6))),
                              ("B2_probe_bt", "B2 probe @L12, unbounded", "#1baf7a",
                               (0, (7, 1.6, 1.5, 1.6)))]:
        a = audit.get(tag)
        if not a or "dw_by_layer" not in a:
            continue
        t = prof(a)
        xs = sorted(t)
        ys = [t[k] - base.get(k, 0.0) for k in xs]
        ax.plot(xs, ys, color=c, lw=2, ls=dash, marker="o", ms=3.5, mew=0, zorder=3, label=lab)
        endlabel(ax, xs[-1], ys[-1], tag.split("_")[0], c)
    ax.axhline(0, color=INK_MUTED, lw=1, zorder=1)
    ax.axvline(12, color=INK_MUTED, lw=1, ls=(0, (2, 2)), zorder=1)
    ax.annotate("L* = 12", (12, ax.get_ylim()[1]), xytext=(4, -12), textcoords="offset points",
                color=INK_MUTED, fontsize=8)
    style(ax, "block", "‖ΔW‖_F / ‖W‖_F  above the replay-only arm",
          "Where the preference actually writes")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"  -> {out}")


def fig_where(audit, out):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), facecolor=SURFACE)
    for tag, lab, c, dash in ARMS:
        a = audit.get(tag)
        if not a:
            continue
        if "drift" in a:
            y = a["drift"]["rel_l2"]
            x = np.arange(len(y)) - 1                     # read index -> block index
            axes[0].plot(x, y, color=c, lw=2, ls=dash, zorder=3, label=lab)
            endlabel(axes[0], x[-1], y[-1], tag.split("_")[0], c)
        if "dw_by_layer" in a:
            tot = {}
            for per in a["dw_by_layer"].values():
                for L, v in per.items():
                    tot[int(L)] = tot.get(int(L), 0.0) + v
            xs = sorted(tot)
            axes[1].plot(xs, [tot[k] for k in xs], color=c, lw=2, ls=dash, zorder=3, label=lab)
            endlabel(axes[1], xs[-1], tot[xs[-1]], tag.split("_")[0], c)
    for ax in axes:
        ax.axvline(12, color=INK_MUTED, lw=1, ls=(0, (2, 2)), zorder=1)
        ax.annotate("L* = 12", (12, ax.get_ylim()[1]), xytext=(3, -10),
                    textcoords="offset points", color=INK_MUTED, fontsize=7.5)
    style(axes[0], "block (read point)", "mean ‖h − h_base‖ / ‖h_base‖",
          "Function drift by depth, held-out text")
    style(axes[1], "block", "‖ΔW‖_F / ‖W‖_F", "Weight delta by layer (summed over modules)")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"  -> {out}")


def fig_curve(audit, base_curve, out):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), facecolor=SURFACE)
    for ax, col, ttl in ((axes[0], "guard", "Fresh-probe decodability: guard"),
                         (axes[1], "pooled_new", "Fresh-probe decodability: pooled install")):
        if base_curve:
            x = [r["read"] - 1 for r in base_curve]
            ax.plot(x, [r[col] for r in base_curve], color=BASE_GREY, lw=2, ls="solid", zorder=2)
            endlabel(ax, x[-1], base_curve[-1][col], "base", BASE_GREY)
        for tag, lab, c, dash in ARMS:
            a = audit.get(tag)
            if not a or "fresh_probe_curve" not in a:
                continue
            cur = a["fresh_probe_curve"]
            x = [r["read"] - 1 for r in cur]
            ax.plot(x, [r[col] for r in cur], color=c, lw=2, ls=dash, zorder=3, label=lab)
            endlabel(ax, x[-1], cur[-1][col], tag.split("_")[0], c)
        ax.axhline(0.5, color=INK_MUTED, lw=1, ls=(0, (2, 2)), zorder=1)
        ax.axvline(12, color=INK_MUTED, lw=1, ls=(0, (2, 2)), zorder=1)
        ax.set_ylim(-0.02, 1.05)
        style(ax, "block (read point)", "held-out accuracy", ttl)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"  -> {out}")


def tables(hist, audit):
    print("\n== held-out ranking, full buckets (pf_audit) ==")
    cols = ["pooled_new", "guard", "legacy", "install_culture", "install_truth_dialect",
            "install_false_friend", "install_style"]
    print(f"{'arm':<26}" + "".join(f"{c[:13]:>14}" for c in cols))
    for tag in ["base"] + [t for t, *_ in ARMS]:
        a = audit.get(tag)
        if not a:
            continue
        r = a["ranking"]
        print(f"{tag:<26}" + "".join(
            f"{r[c]['raw']:>14.3f}" if c in r else f"{'-':>14}" for c in cols))
    print("\n== replay drift at the end of training ==")
    print(f"{'arm':<26}{'replay nll':>12}{'replay KL':>12}")
    for tag, *_ in ARMS:
        h = hist.get(tag)
        if h:
            e = h["evals"][-1]
            print(f"{tag:<26}{e['replay_nll']:>12.3f}{e['replay_kl']:>12.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/workspace/probefix")
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "..", "results", "probefix"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    hist, audit = load(args.root)
    print(f"[report] {len(hist)} histories, {len(audit)} audits")
    tables(hist, audit)
    fig_pareto(hist, f"{args.outdir}/pf_pareto.png")
    fig_probe_trace(hist, f"{args.outdir}/pf_probe_trace.png")
    fig_training(hist, f"{args.outdir}/pf_training.png")
    fig_where(audit, f"{args.outdir}/pf_where.png")
    fig_dw_pref(audit, f"{args.outdir}/pf_dw_pref.png")
    # Prefer the base model's curve fitted by pf_audit itself -- same subset, same fit protocol as
    # the arms. The stage-0 curve is a fallback and is fitted on all 4,735 train rows, so it is not
    # strictly comparable to the arms' 2,048-row refits.
    bc = (audit.get("base") or {}).get("fresh_probe_curve")
    cp = f"{args.root}/curve_dose20.json"
    if bc is None and os.path.exists(cp):
        bc = json.load(open(cp))["curves"].get("last")
    fig_curve(audit, bc, f"{args.outdir}/pf_curve.png")


if __name__ == "__main__":
    main()
