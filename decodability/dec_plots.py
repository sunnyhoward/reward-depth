#!/usr/bin/env python
"""Figures for the decodability sweep.

WHAT IS PLOTTED. Held-out pairwise accuracy against depth, one line per model, one panel per
pair family, all thirteen families of all four datasets on one page. Two pages: one per read
protocol, because read position is a finding here, not a setting (last-token at the embedding
layer is 100% ties on several families -- a fact about where you read, not about the layer).

X AXIS IS FRACTIONAL DEPTH, not the layer index. The ladder mixes 28-layer models (0.6B, 1.7B)
with 36-layer ones (4B, 8B); plotting raw indices would put L14 of a 28-layer model beside L14 of
a 36-layer model, which are half-way and two-fifths of the way up respectively. Fractional depth
is what makes the four curves comparable, and it is what the L*/D summary is computed in.

COLOUR ENCODES MODEL SIZE, so it is an ORDINAL RAMP -- one hue, light to dark, 0.6B to 8B -- not
four categorical hues. Model scale is a magnitude; categorical hues would assert the models are
unordered identities. Steps are the blue ramp's 250/400/500/650, validated with
`validate_palette.js --mode light --ordinal` (lightness monotone, adjacent dL, light-end contrast
2.06:1 vs surface, single hue). Identity is never carried by colour alone: every model also has
its own dash pattern and appears in the legend.

REFERENCE LINES ARE THE POINT. Each panel carries its lexical floor (a bag-of-token-ids probe with
no model at all) and the 0.5 chance line. An accuracy of 0.99 above a floor of 0.985 is not a
finding about a model; this makes that visible without reading a table.

Usage: python dec_plots.py [--outdir results/decodability/plots]
"""
import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dec_common as C  # noqa: E402

# ── design tokens (dataviz reference palette, light surface) ──────────────────────────────────
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8985"
GRID = "#e6e5e1"
FLOOR = "#eb6834"          # categorical slot 2 -- the floor is a different KIND of thing
# Ordinal blue ramp, light→dark = small→large. Validated: --mode light --ordinal, all checks PASS.
MODEL_STYLE = [
    ("qwen3-0.6b", "0.6B", "#86b6ef", (0, (1, 1.6))),
    ("qwen3-1.7b", "1.7B", "#3987e5", (0, (4, 1.6))),
    ("qwen3-4b",   "4B",   "#256abf", (0, (7, 1.6, 1.5, 1.6))),
    ("qwen3-8b",   "8B",   "#104281", "solid"),
]

# Panel order: shallow-by-construction families first, the two that actually carry depth last, so
# the eye reaches the contrast at the end rather than hunting for it.
PANEL_ORDER = [
    ("styc", "style_c"), ("styc", "style_w"), ("styc", "aligned"), ("styc", "conflict"),
    ("brit_language", "language"), ("brit_culture", "culture"),
    ("brit_truth", "true_british_over_american"),
    ("brit_truth", "false_british_over_american"),
    ("brit_truth", "truth_over_british"),
    ("styc", "diet_to_conflict"), ("brit_truth", "dialect_to_guard"),
    ("uf", "quality"), ("styc", "corr_e"), ("styc", "corr_t"),
]
NICE = {"style_c": "style | correct", "style_w": "style | wrong", "aligned": "aligned",
        "conflict": "conflict", "language": "language (spelling)", "culture": "culture",
        "true_british_over_american": "true BE > AE", "false_british_over_american": "false BE > AE",
        "truth_over_british": "truth > BE", "diet_to_conflict": "diet → conflict  (transfer)",
        "dialect_to_guard": "dialect → guard  (transfer)",
        "corr_e": "computation-correctness | explained", "corr_t": "computation-correctness | terse",
        "quality": "human-judged quality"}


# Readout palette: categorical, fixed slot order (the ordering IS the CVD-safety mechanism, not a
# cosmetic choice). Validated `--mode light`: lightness band, chroma floor, CVD separation (worst
# adjacent ΔE 9.1), normal-vision floor (19.6) all PASS; contrast WARNs on aqua/yellow/magenta,
# which obligates relief -- supplied here as direct end-labels on every readout.
# Reference lines in these figures are GREY, not orange, so the categorical order stays intact
# and no series colour collides with the floor lines used in the other figures.
READOUTS = [
    ("linear",      "A", "linear",     "#2a78d6"),
    ("mlp",         "A", "MLP",        "#eb6834"),
    ("attn",        "A", "attention",  "#1baf7a"),
    ("seq-tf",      "A", "tf (attn+MLP)", "#eda100"),
    ("seq-2l",      "A", "tf ×2",      "#e87ba4"),
    ("eagle-mlp",   "B", "eagle-mlp",  "#eda100"),
    ("eagle-attn",  "B", "eagle-attn", "#e87ba4"),
    ("eagle-tf",    "B", "eagle-tf",   "#008300"),
    ("eagle-2l",    "B", "eagle-2L",   "#4a3aa7"),
]
# Alpha encodes model scale: an ordered dimension, so an ordered visual channel.
MODEL_ALPHA = {"qwen3-0.6b": 0.30, "qwen3-1.7b": 0.50, "qwen3-4b": 0.74, "qwen3-8b": 1.0}


def load_readouts(dataset, family, read="mean"):
    """→ {readout: {model: (x_fractional, y)}} across all three result kinds, plus the floors."""
    out, floors = {}, {"lexical": None, "length": None}
    for p in sorted(glob.glob(os.path.join(C.RESULT_DIR, f"scalar_*_{dataset}_*.json"))):
        r = json.load(open(p))
        for key, cell in r["results"].items():
            fam, rd, rung = key.split("|")
            if fam != family or rd != read:
                continue
            acc = np.asarray(cell["acc_mean"])
            out.setdefault(rung, {})[r["model"]] = (np.arange(len(acc)) / (len(acc) - 1), acc)
        fl = r["floor"].get(family)
        if fl:
            floors["lexical"], floors["length"] = fl["group_split"], fl.get("length_only")
    for p in sorted(glob.glob(os.path.join(C.RESULT_DIR, f"attn_*_{dataset}_*.json"))):
        r = json.load(open(p))
        for arch in ("attn", "seq-tf", "seq-2l"):
            cell = r["results"].get(f"{family}|seq|{arch}")
            if cell:
                L = np.asarray(cell["layers"], float)
                out.setdefault(arch, {})[r["model"]] = (L / r["n_layers"],
                                                        np.asarray(cell["acc_mean"]))
    for p in sorted(glob.glob(os.path.join(C.RESULT_DIR, f"through_*_{dataset}_*.json"))):
        r = json.load(open(p))
        xs, ys = [], []
        for L in r["layers"]:
            cell = r["results"].get(f"{family}|{L}")
            if cell and cell.get("acc_vs_base") is not None:
                xs.append(L / max(r["layers"]))
                ys.append(cell["acc_vs_base"])
        if xs:
            out.setdefault(r["arch"], {})[r["model"]] = (np.asarray(xs), np.asarray(ys))
    return out, floors


def figure_readouts(dataset, family, outpath, read="mean"):
    """One dataset: depth on x, readout as COLOUR, model scale as ALPHA.

    Two rows because the two families do not share a metric and stacking them on one axis would
    assert a comparability that does not exist: family A is held-out accuracy from a readout
    FITTED on the preference; family B is agreement with the full model from a frozen head that
    never saw a preference pair. Same x, same colour key, separate y.
    """
    data, floors = load_readouts(dataset, family, read)
    if not data:
        return None
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 7.4), facecolor=SURFACE, sharex=True,
                             gridspec_kw=dict(height_ratios=[1, 1], hspace=0.28))
    for row, (fam_tag, ylab, sub) in enumerate([
            ("A", "held-out pairwise accuracy",
             "family A — fitted scalar readouts:  is the preference extractable from h_L?"),
            ("B", "agreement with the full model",
             "family B — frozen through-head readouts, never fitted on a pair:\n"
             "how much of the model's own ordering is already expressible at L?")]):
        ax = axes[row]
        ax.set_facecolor(SURFACE)
        ax.axhline(0.5, color=INK_MUTED, lw=0.8, ls=(0, (3, 3)), zorder=1)
        if row == 0:
            for tag, val, style in [("lexical floor", floors["lexical"], (0, (1.5, 1.5))),
                                    ("length-only floor", floors["length"], "solid")]:
                if val is None:
                    continue
                ax.axhline(val, color=INK_MUTED, lw=1.2, ls=style, zorder=2)
                ax.text(0.985, val - 0.022, f"{tag} {val:.2f}", color=INK_MUTED, fontsize=6.4,
                        ha="right", va="top", zorder=6)
        tips = []
        for rid, fam, label, colour in READOUTS:
            if fam != fam_tag or rid not in data:
                continue
            marker = "o" if fam_tag == "B" else None
            present = [m for m in MODEL_ALPHA if m in data[rid]]
            for mk in present:
                x, y = data[rid][mk]
                ax.plot(x, y, color=colour, alpha=MODEL_ALPHA[mk], lw=2.0, zorder=4,
                        marker=marker, markersize=4.5, solid_capstyle="round")
            # Label the LARGEST model actually present, not whichever has alpha 1.0 -- a partially
            # finished grid would otherwise silently lose every label, and the contrast WARN on
            # three of these hues makes the labels mandatory relief, not decoration.
            x, y = data[rid][present[-1]]
            tips.append([float(y[-1]), float(x[-1]), label, colour])
        # ADAPTIVE Y, set BEFORE the labels are placed -- the label stagger below works in data
        # units, so it has to know the final axis range. A fixed 0-1 axis wastes most of its
        # height on empty space and squashes every real difference into a few pixels: on UF the
        # entire story happens between 0.62 and 0.87. Zoom to the data plus any reference line,
        # with a minimum span so a genuinely flat panel is not magnified into noise.
        lo, hi = np.inf, -np.inf
        for rid, fam, _, _ in READOUTS:
            if fam != fam_tag or rid not in data:
                continue
            for mk in data[rid]:
                y = data[rid][mk][1]
                lo, hi = min(lo, float(np.nanmin(y))), max(hi, float(np.nanmax(y)))
        if not np.isfinite(lo):
            lo, hi = 0.0, 1.0
        refs = [v for v in ([floors["lexical"], floors["length"], 0.5] if row == 0 else [0.5])
                if v is not None and lo - 0.12 <= v <= hi + 0.12]
        if refs:
            lo, hi = min(lo, min(refs)), max(hi, max(refs))
        span = max(hi - lo, 0.12)
        y0, y1 = max(-0.03, lo - 0.10 * span), min(1.05, hi + 0.10 * span + 0.02)
        ax.set_ylim(y0, y1)
        ax.set_xlim(0, 1.14)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])

        # Push overlapping end-labels apart, in units of the ACTUAL axis span (a fixed 0.055 was
        # right for a 0-1 axis and far too coarse once the panel zooms to a 0.25-wide range).
        tips.sort(key=lambda t: t[0])
        gap = 0.065 * (y1 - y0)
        for i in range(1, len(tips)):
            if tips[i][0] - tips[i - 1][0] < gap:
                tips[i][0] = tips[i - 1][0] + gap
        # The upward pass can push the topmost label past the axis limit, where matplotlib clips
        # it and the series silently loses its label -- which is the relief the contrast WARN
        # requires, so it cannot be allowed to vanish. Slide the whole stack down to fit.
        if tips and tips[-1][0] > y1 - 0.02 * (y1 - y0):
            shift = tips[-1][0] - (y1 - 0.02 * (y1 - y0))
            for t in tips:
                t[0] -= shift
        for ytxt, xend, label, colour in tips:
            ax.annotate(label, (xend, ytxt), textcoords="offset points", xytext=(6, 0),
                        fontsize=7, color=colour, va="center", zorder=7)
        ax.tick_params(colors=INK_2, labelsize=7.5, length=0)
        ax.grid(True, color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_ylabel(ylab, fontsize=8, color=INK_2)
        ax.set_title(sub, fontsize=8.2, color=INK, loc="left", pad=6)
    axes[1].set_xlabel("fractional depth   (0 = embeddings, 1 = top block)",
                       fontsize=8.5, color=INK_2)
    # Model legend: the alpha ramp, shown on a neutral swatch so it reads as a separate channel
    # from the colour key rather than competing with it.
    handles = [plt.Line2D([], [], color=INK, alpha=a, lw=3,
                          label=k.replace("qwen3-", "").upper())
               for k, a in MODEL_ALPHA.items()]
    lg = axes[0].legend(handles=handles, fontsize=7, frameon=False, ncol=4, loc="upper left",
                        title="model  (alpha)", labelcolor=INK_2)
    lg.get_title().set_fontsize(7)
    lg.get_title().set_color(INK_2)
    fig.suptitle(f"Readout ladder on {dataset} / {family}   —   colour = readout, alpha = scale",
                 fontsize=11.5, color=INK, x=0.008, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(outpath, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    return outpath


def load(rung="linear"):
    """→ {(dataset, family): {model: dict(read → (acc, sd)), 'floor': float|None}}"""
    out = {}
    for p in sorted(glob.glob(os.path.join(C.RESULT_DIR, "scalar_*.json"))):
        r = json.load(open(p))
        for key, cell in r["results"].items():
            fam, read, rg = key.split("|")
            if rg != rung:
                continue
            e = out.setdefault((r["dataset"], fam),
                               {"floor": None, "len_floor": None, "models": {}})
            e["models"].setdefault(r["model"], {})[read] = (
                np.asarray(cell["acc_mean"]), np.asarray(cell["acc_std"]))
            fl = r["floor"].get(fam)
            if fl:
                e["floor"] = fl["group_split"]
                e["len_floor"] = fl.get("length_only")
                # The honest bag-of-words ceiling is the HIGHER of the two splits: random-split is
                # what a probe with the training vocabulary in hand can do, group-split is what
                # survives held-out groups. A model probe has to clear the higher one to be doing
                # something a word list cannot.
                e["floor_max"] = max(fl["group_split"], fl.get("random_split") or 0.0)
    return out


SATURATION_GAIN = 0.10
TRANSFER_FAMILIES = {"diet_to_conflict", "dialect_to_guard"}


def split_families(data):
    """→ (saturated, meaningful). Saturated = peak accuracy adds < SATURATION_GAIN over the
    bag-of-words ceiling, i.e. the family is separable by vocabulary and its depth curve is not
    a statement about the model. The observed gap is wide and unambiguous: eight families sit at
    0.000–0.083 and the rest at 0.116–0.478, with nothing in between.

    Transfer cells have no floor (they are fitted on one family and scored on another) and are
    kept with the meaningful set -- they are the conflict diagnostics, which is the whole reason
    they exist."""
    sat, mean = [], []
    for key, e in data.items():
        # Transfer cells are not decodability curves -- they are fitted on one family and scored
        # on another, so "accuracy vs depth" is not the quantity being asked about. They have
        # their own section and are excluded from both depth figures.
        if key[1] in TRANSFER_FAMILIES:
            continue
        # [0] is the accuracy curve; [1] is the across-seed SD. Taking [1] here silently makes
        # every peak ~0 and classifies the entire sweep as saturated.
        peaks = [float(np.nanmax(v[rd][0])) for v in e["models"].values() for rd in v
                 if rd == "mean"]
        peak = max(peaks) if peaks else 0.0
        fmax = e.get("floor_max")
        (sat if (fmax is not None and peak - fmax < SATURATION_GAIN) else mean).append(key)
    order = {k: i for i, k in enumerate(PANEL_ORDER)}
    return (sorted(sat, key=lambda k: order.get(k, 99)),
            sorted(mean, key=lambda k: order.get(k, 99)))


# Dataset colours for the collapsed figure: categorical slots 1-4, fixed order.
DATASET_COLOUR = {"styc": "#2a78d6", "brit_language": "#eb6834",
                  "brit_culture": "#1baf7a", "brit_truth": "#eda100", "uf": "#e87ba4"}


def figure_saturated(data, keys, read, outpath):
    """The lexically-saturated families, collapsed onto one panel in HEADROOM units.

    Plotted as accuracy MINUS the bag-of-words ceiling, so the reference is the horizontal zero
    line and the claim is legible directly: every one of these curves hugs zero at every depth,
    i.e. the whole stack adds nothing a word-count probe did not already have. Plotting them as
    raw accuracy would show eight flat lines at 1.0 and hide the fact that their floors are at
    1.0 too, which is the entire point.
    """
    fig, ax = plt.subplots(figsize=(9.2, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axhline(0, color=INK_2, lw=1.4, zorder=2)
    ax.text(0.995, -0.006, "bag-of-words ceiling", color=INK_2, fontsize=7, va="top", ha="right")
    seen = set()
    for ds, fam in keys:
        e = data[(ds, fam)]
        colour = DATASET_COLOUR.get(ds, INK_MUTED)
        for mk, alpha in MODEL_ALPHA.items():
            m = e["models"].get(mk)
            if not m or read not in m:
                continue
            acc = m[read][0]
            x = np.arange(len(acc)) / (len(acc) - 1)
            ax.plot(x, acc - e["floor_max"], color=colour, alpha=alpha * 0.85, lw=1.6, zorder=4,
                    label=ds if ds not in seen else None)
            seen.add(ds)
    ax.set_ylim(-0.16, 0.16)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    ax.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("fractional depth   (0 = embeddings, 1 = top block)", fontsize=8.5, color=INK_2)
    ax.set_ylabel("accuracy − bag-of-words ceiling", fontsize=8.5, color=INK_2)
    lg = ax.legend(fontsize=7.5, frameon=False, ncol=4, loc="lower right", labelcolor=INK_2,
                   title="dataset   (alpha = model scale, 0.6B→8B)")
    lg.get_title().set_fontsize(7.5)
    lg.get_title().set_color(INK_2)
    fam_list = ", ".join(f"{f}" for _, f in keys)
    fig.suptitle(f"{len(keys)} families where depth adds nothing over a bag-of-words probe",
                 fontsize=11.5, color=INK, x=0.008, ha="left")
    fig.text(0.008, 0.905, fam_list, fontsize=7.4, color=INK_2, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(outpath, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    return outpath


def panel(ax, entry, read, title, show_legend=False):
    ax.set_facecolor(SURFACE)
    ax.axhline(0.5, color=INK_MUTED, lw=0.8, ls=(0, (3, 3)), zorder=1)
    if entry["floor"] is not None:
        # Drawn THICKER than the data lines and underneath them, so that when the floor and the
        # accuracy coincide -- styc style, where both are exactly 1.000 -- the orange still shows
        # as a halo around the blue. That coincidence is the finding in those panels; a floor line
        # perfectly hidden under the data would read as "no floor plotted".
        ax.axhline(entry["floor"], color=FLOOR, lw=3.2, ls=(0, (2.2, 1.8)), zorder=2,
                   alpha=0.95)
        high = entry["floor"] > 0.9
        ax.text(0.02 if high else 0.985, entry["floor"] - (0.045 if high else -0.03),
                f"lexical floor {entry['floor']:.2f}", color=FLOOR, fontsize=6.2,
                ha="left" if high else "right", va="top" if high else "bottom", zorder=6)
    lfl = entry.get("len_floor")
    # Same hue as the lexical floor because it is the same KIND of thing -- a no-model baseline --
    # distinguished by weight and label rather than by a new colour. Only drawn when it is not
    # visually on top of the lexical floor.
    if lfl is not None and (entry["floor"] is None or abs(lfl - entry["floor"]) > 0.04):
        ax.axhline(lfl, color=FLOOR, lw=1.1, ls="solid", zorder=2, alpha=0.75)
        ax.text(0.985, lfl - 0.03, f"length-only {lfl:.2f}", color=FLOOR, fontsize=6.2,
                ha="right", va="top", zorder=6, alpha=0.9)
    for key, label, colour, dash in MODEL_STYLE:
        m = entry["models"].get(key)
        if not m or read not in m:
            continue
        acc, sd = m[read]
        x = np.arange(len(acc)) / (len(acc) - 1)
        ax.fill_between(x, acc - sd, acc + sd, color=colour, alpha=0.16, lw=0, zorder=3)
        ax.plot(x, acc, color=colour, lw=2.0, ls=dash, label=label, zorder=4,
                solid_capstyle="round")
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(colors=INK_2, labelsize=7, length=0)
    ax.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=8.2, color=INK, pad=5, loc="left")
    if show_legend:
        lg = ax.legend(fontsize=6.8, frameon=False, loc="lower right", ncol=2,
                       handlelength=2.4, columnspacing=1.0, labelcolor=INK_2)
        lg.set_title("model", prop={"size": 6.8})
        lg.get_title().set_color(INK_2)


def figure_depth(data, read, outpath, rung="linear", keys=None, title=None):
    order = keys if keys is not None else PANEL_ORDER
    n = len(order)
    ncol = min(4, max(1, n))
    nrow = int(np.ceil(n / ncol))
    # Header space is reserved in INCHES, not as a figure fraction: the same 0.995/0.966 fractions
    # that sit correctly above a 4-row grid land on top of each other above a 2-row one.
    HEAD = 0.95
    figh = 2.55 * nrow + HEAD
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.9 * ncol, figh), facecolor=SURFACE,
                             squeeze=False)
    axes = np.atleast_2d(axes).ravel()
    ds_seen = set()
    for i, (ds, fam) in enumerate(order):
        ax = axes[i]
        e = data.get((ds, fam))
        if e is None:
            ax.set_visible(False)
            continue
        tag = ds if ds not in ds_seen else ""
        ds_seen.add(ds)
        panel(ax, e, read, f"{NICE.get(fam, fam)}\n{ds}", show_legend=(i == 0))
        if i % ncol == 0:
            ax.set_ylabel("held-out pairwise acc.", fontsize=7.5, color=INK_2)
        if i + ncol >= n:      # bottom-most panel of its own column, not of the whole figure
            ax.set_xlabel("fractional depth  (0 = embeddings, 1 = top block)",
                          fontsize=7.5, color=INK_2)
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    readname = {"mean": "mean-pooled over completion tokens", "last": "last completion token"}[read]
    fig.suptitle(title or f"Where does the preference become decodable?   —   {rung} probe, "
                          f"read = {readname}",
                 fontsize=11.5, color=INK, x=0.008, ha="left", y=1 - 0.30 / figh)
    fig.text(0.008, 1 - 0.62 / figh,
             f"{rung} probe, read = {readname}.  Line = accuracy vs depth, band = ±1 SD over "
             "3 seeds.  Orange = no-model floors.  Dashed grey = chance.",
             fontsize=7.6, color=INK_2, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 1 - HEAD / figh])
    fig.savefig(outpath, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    return outpath


def figure_lstar(data, read="mean", outpath=None, rung="linear"):
    """Dot plot of L*/D -- the headline: which families need depth at all."""
    rows = []
    for ds, fam in PANEL_ORDER:
        e = data.get((ds, fam))
        if e is None:
            continue
        pts, peaks = [], []
        for key, label, colour, _ in MODEL_STYLE:
            m = e["models"].get(key)
            if not m or read not in m:
                continue
            acc, _ = m[read]
            se = np.sqrt(0.25 / 120)
            mx = float(np.nanmax(acc))
            i = int(np.where(acc >= mx - se)[0][0])
            pts.append((label, colour, i / (len(acc) - 1)))
            peaks.append(mx)
        if pts:
            # L* is "earliest point within 1 SE of the curve's own maximum". On a curve that never
            # rises above chance that maximum is noise, so L* is a position in noise and must not
            # be drawn as if it located anything. The transfer families are exactly this case.
            rows.append((f"{NICE.get(fam, fam)}   ({ds})", pts, max(peaks) >= 0.55))
    fig, ax = plt.subplots(figsize=(9.4, 0.44 * len(rows) + 1.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    off = np.linspace(-0.17, 0.17, len(MODEL_STYLE))
    for y, (name, pts, meaningful) in enumerate(rows):
        ax.plot([0, 1], [y, y], color=GRID, lw=0.9, zorder=0)
        if not meaningful:
            ax.text(0.02, y, "curve never exceeds chance — L* undefined", fontsize=6.8,
                    color=INK_MUTED, va="center", ha="left", style="italic", zorder=5)
            continue
        # Small vertical offsets: without them four models landing on the identical L* stack into
        # one dot and the panel silently reads as a single measurement.
        for k, (label, colour, v) in enumerate(pts):
            ax.scatter([v], [y + off[k]], s=58, color=colour, zorder=3,
                       edgecolor=SURFACE, lw=1.1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.8, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("L* / depth   —   earliest read point within 1 SE of that curve's own maximum",
                  fontsize=8, color=INK_2)
    ax.tick_params(colors=INK_2, labelsize=7.5, length=0)
    ax.grid(True, axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, markersize=7, label=lab)
               for _, lab, c, _ in MODEL_STYLE]
    lg = ax.legend(handles=handles, fontsize=7.2, frameon=False, ncol=4, loc="lower right",
                   bbox_to_anchor=(1.0, 1.005), labelcolor=INK_2)
    lg.set_title("model", prop={"size": 7.2})
    lg.get_title().set_color(INK_2)
    fig.suptitle("Only computation-correctness needs depth", fontsize=11.5, color=INK,
                 x=0.008, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(outpath, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    return outpath


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(C.RESULT_DIR, "plots"))
    ap.add_argument("--rung", default="linear")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    data = load(a.rung)
    sat, mean_fams = split_families(data)
    print(f"[split] saturated (<{SATURATION_GAIN} over the bag-of-words ceiling): "
          f"{[f'{d}/{f}' for d, f in sat]}")
    print(f"[split] meaningful: {[f'{d}/{f}' for d, f in mean_fams]}")
    made = [
        figure_saturated(data, sat, "mean", os.path.join(a.outdir, "depth_saturated.png")),
        figure_depth(data, "mean", os.path.join(a.outdir, "depth_meaningful.png"), a.rung,
                     keys=mean_fams,
                     title="The families that are NOT solved by a bag-of-words probe"),
    ]
    # Per-dataset readout ladders: one figure each for the families worth a close look.
    for ds, fam in [("uf", "quality"), ("styc", "corr_e"), ("brit_language", "language"),
                    ("brit_culture", "culture"), ("styc", "style_c"),
                    ("brit_truth", "truth_over_british")]:
        p = figure_readouts(ds, fam, os.path.join(a.outdir, f"readouts_{ds}_{fam}.png"))
        if p:
            made.append(p)
    for p in made:
        print(f"[plot] {p}")
