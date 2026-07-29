#!/usr/bin/env python
"""Validate the content-choice (cc) testbed: no letters, options embedded in the question,
content-word answers. The testbed contract: base model near-ceiling on train types, so a
wrongness-preference flip is measurable; offmenu = emitted neither option (the open-vocabulary
displacement signal the letter menu couldn't show); know first-option rate ~0.5 = no position
policy at base (the successor of fracA).

Env: MODEL=Qwen/Qwen2.5-3B N=300 MAX_NEW=8
Saves /workspace/cc_validate.json"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import load_model, build_data, greedy

E = os.environ.get
MODEL, N, MAX_NEW = E("MODEL", "Qwen/Qwen2.5-3B"), int(E("N", 300)), int(E("MAX_NEW", 8))

ctx = load_model(MODEL)
ctx.policy = ctx.model   # greedy() reads ctx.policy; no adapter here -- score the raw base
d = build_data(seed=0, n_train=N, n_eval=N, n_transfer=150, formats=("cc",), tok=ctx.tok)

def matches(gen, ent):
    g = gen.strip().lstrip("$")   # model answers money questions "$745" against bare-number gold
    return g[: len(ent)] == ent and (len(g) == len(ent) or not g[len(ent)].isalnum())

def score(qs, label):
    outs = greedy(ctx, [d.render_cc(q) for q in qs], MAX_NEW)
    t = np.mean([matches(o, q["t"]) for o, q in zip(outs, qs)])
    f = np.mean([matches(o, q["f"]) for o, q in zip(outs, qs)])
    off = 1 - t - f
    row = dict(correct=float(t), wrong=float(f), offmenu=float(off), n=len(qs))
    if any(q.get("cc_first") for q in qs):   # know: position-bias baseline
        firsts = [(o, q) for o, q in zip(outs, qs) if q.get("cc_first")]
        row["first_opt_rate"] = float(np.mean([matches(o, q["cc_first"]) for o, q in firsts]))
    print(f"[{label:12s}] correct {t:.3f} wrong {f:.3f} offmenu {off:.3f}"
          + (f" first-opt {row['first_opt_rate']:.3f}" if "first_opt_rate" in row else ""), flush=True)
    return row

res = dict(model=MODEL)
res["train"] = score(d.train_qs, "train-types")
res["eval"] = score(d.eval_qs, "eval")
res["know"] = score(d.know_qs, "know")
for t, qs in d.ood_sets.items():
    res[f"ood_{t}"] = score(qs, f"ood:{t}")
json.dump(res, open("/workspace/cc_validate.json", "w"), indent=1)
print("DONE", flush=True)
