#!/usr/bin/env python
"""Assemble the decodability sweep into tables and figures.

THE SUMMARY STATISTIC. Peak accuracy is the wrong headline for this experiment -- it says how
well a readout can do, not where the information appears, and the whole claim under test is
about WHERE. So each curve is reduced to

    L*  = the earliest read point whose accuracy is within 1 SE of that curve's own maximum
    L*/depth = the same as a fraction, so 28-layer and 36-layer models are comparable

"within 1 SE of ITS OWN maximum" is deliberate: comparing every readout to a single global
ceiling would just rank readouts by capacity, and the question is whether a more capable readout
finds the preference EARLIER, not merely better.

EVERY TABLE CARRIES ITS CONTROLS. A cell is printed with its lexical floor and its shuffled-label
null beside it, because an accuracy without those two numbers is not interpretable here: the brit
axis reaches 0.96 at the embedding layer, and whether that is a finding or a restatement of
"the pairs differ by a word" depends entirely on the floor.

Usage: python dec_report.py [--md out.md]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dec_common as C  # noqa: E402

MODEL_ORDER = ["qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b"]


def lstar(acc, se):
    """Earliest index within 1 SE of the curve's own max. → (index, max, argmax)."""
    acc = np.asarray(acc, float)
    mx = float(np.nanmax(acc))
    hit = np.where(acc >= mx - se)[0]
    return (int(hit[0]) if len(hit) else int(np.nanargmax(acc)), mx, int(np.nanargmax(acc)))


def load_all(kind):
    out = []
    for p in sorted(glob.glob(os.path.join(C.RESULT_DIR, f"{kind}_*.json"))):
        with open(p) as f:
            out.append(json.load(f))
    return out


def family_a_table():
    rows = []
    for r in load_all("scalar"):
        n_reads = r["n_reads"]
        for key, cell in r["results"].items():
            fam, read, rung = key.split("|")
            se = float(np.sqrt(0.25 / max(cell["n_test"], 1)))
            i, mx, am = lstar(cell["acc_mean"], se)
            shuf = list(cell["shuffled"].values())
            fl = r["floor"].get(fam, {})
            rows.append(dict(
                model=r["model"], dataset=r["dataset"], family=fam, read=read, rung=rung,
                n_test=cell["n_test"], transfer=cell.get("transfer", False),
                L0=cell["acc_mean"][0], tie0=cell["tie_frac"][0],
                top=cell["acc_mean"][-1], peak=mx, argmax=am,
                Lstar=i, Lstar_frac=i / (n_reads - 1),
                shuffled=float(np.mean(shuf)) if shuf else float("nan"),
                floor_group=fl.get("group_split"), floor_random=fl.get("random_split"),
                seed_sd=float(np.mean(cell["acc_std"]))))
    return rows


def family_b_table():
    rows = []
    for r in load_all("through"):
        for key, cell in r["results"].items():
            fam, L = key.rsplit("|", 1)
            rows.append(dict(model=r["model"], arch=r["arch"], dataset=r["dataset"], family=fam,
                             layer=L, acc=cell["acc"], acc_pertok=cell.get("acc_pertok"),
                             acc_vs_base=cell.get("acc_vs_base"),
                             corr_vs_base=cell.get("corr_vs_base"), margin=cell["margin"],
                             n_test=cell["n_test"], kl_head=cell["kl_head"],
                             top1_agree=cell["top1_agree"], n_params=r["n_params"]))
    return rows


def _fmt(v, nd=3):
    return "--" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def render_md(a_rows, b_rows):
    L = []
    L.append("# Decodability sweep — results\n")
    L.append("`L*` = earliest read point within 1 SE of that curve's own maximum. "
             "Read points: 0 = embeddings, i = output of block i-1.\n")

    L.append("\n## Family A — scalar readouts (is the preference EXTRACTABLE from h_L?)\n")
    L.append("| model | dataset | family | read | rung | L0 (tie) | L* | L*/D | peak | top | "
             "shuffled | floor grp | floor rnd | n |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(a_rows, key=lambda x: (x["dataset"], x["family"], MODEL_ORDER.index(x["model"])
                                           if x["model"] in MODEL_ORDER else 9, x["read"], x["rung"])):
        L.append(f"| {r['model']} | {r['dataset']} | {r['family']}{'*' if r['transfer'] else ''} | "
                 f"{r['read']} | {r['rung']} | {_fmt(r['L0'])} ({r['tie0']:.2f}) | {r['Lstar']} | "
                 f"{r['Lstar_frac']:.2f} | {_fmt(r['peak'])} | {_fmt(r['top'])} | "
                 f"{_fmt(r['shuffled'])} | {_fmt(r['floor_group'])} | {_fmt(r['floor_random'])} | "
                 f"{r['n_test']} |")
    L.append("\n`*` = cross-family transfer cell (fit on one family, scored on another).\n")

    if b_rows:
        L.append("\n## Family B — through-head likelihood "
                 "(is it EXPRESSIBLE through the frozen unembedding?)\n")
        L.append("Heads distilled on generative replay only; zero preference fitting. "
                 "`KL` and `agree` are the head-competence covariate — a depth curve that is "
                 "really a competence curve must show it here.\n")
        L.append("**Read `agrees-with-base`, not `pref`.** `pref` is the sign of the summed-logp "
                 "gap: it is dominated by completion length wherever the two sides differ in "
                 "length (styc `style_c` reads 0.000 and `conflict` 1.000 at every layer *and* at "
                 "the full base model), and it encodes the model's prior rather than its "
                 "decodability (brit reads ~0.15 because the base is American-default). "
                 "`agrees-with-base` asks whether layer L ranks the pair the way the full stack "
                 "does — the prior and the length term are shared with the reference, so what "
                 "remains is how much of the model's own ordering is already expressible at L.\n")
        L.append("| model | arch | dataset | family | layer | agrees-base | pref | pref/tok | "
                 "corr | KL(base‖head) | agree | params |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in sorted(b_rows, key=lambda x: (x["dataset"], x["family"], x["arch"],
                                               99 if x["layer"] == "base" else int(x["layer"]))):
            L.append(f"| {r['model']} | {r['arch']} | {r['dataset']} | {r['family']} | "
                     f"{r['layer']} | **{_fmt(r['acc_vs_base'])}** | {_fmt(r['acc'])} | "
                     f"{_fmt(r['acc_pertok'])} | {_fmt(r['corr_vs_base'], 2)} | "
                     f"{_fmt(r['kl_head'])} | {_fmt(r['top1_agree'], 2)} | {r['n_params']/1e6:.1f}M |")

    L.append("\n## Reading guide\n")
    L.append("- **floor rnd** is the memorisation ceiling of a bag-of-token-ids probe on a random "
             "split. A dataset whose floor rnd is ~1.0 is solvable from vocabulary alone.\n")
    L.append("- **floor grp** is the same probe on held-out groups. The gap between the two is "
             "how much of the dataset is *memorisable* vocabulary rather than *generalisable* "
             "vocabulary. A model probe beating floor grp at layer 0 is reading sub-token "
             "regularity, not a lookup table (goodfire/RESULTS.md:14-18).\n")
    L.append("- **tie** at L0 near 1.00 means the read is degenerate, not that the layer is "
             "uninformative: both completions end in the same token, so the last-token difference "
             "is exactly zero. Compare the `mean` read for that cell.\n")
    L.append("- **shuffled** should sit at 0.5. Anything materially above it means the rung's "
             "capacity is fitting noise and its accuracies are inflated.\n")
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=os.path.join(C.RESULT_DIR, "REPORT.md"))
    args = ap.parse_args()
    a, b = family_a_table(), family_b_table()
    md = render_md(a, b)
    with open(args.md, "w") as f:
        f.write(md)
    C.bank("summary", dict(family_a=a, family_b=b))
    print(md)
    print(f"\n[report] {len(a)} family-A cells, {len(b)} family-B cells -> {args.md}")
