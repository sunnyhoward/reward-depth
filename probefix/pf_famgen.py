#!/usr/bin/env python
"""Family-resolved FREE GENERATION for every banked arm.

The problem this fixes (2026-08-13): every generation number this project reports comes from a
marker regex covering `lexicon` and `culture` only. `style`, `expression`, `false_friend` and the
truth `guard` contribute zero markers and are invisible to it -- and those are exactly the
families that separate a real install from a constant steering vector (a 2,560-parameter constant
vector matched C0's brit_rate while scoring .59/.58 on style/expression against C0's .96/.92).
So "the install works" has, generatively, only ever meant "it swaps some words".

This script generates on held-out prompts from EVERY family and scores with the extended
family-labelled lexicon (`pf_famlex.py`, 731 forms across 4 families, a strict superset of the
old 386). `style` and `guard` carry no surface markers and are written out for `pf_judge.py`.

Output: {OUT}/famgen_<arm>.json  -- generations, per-family marker scores, and the raw text the
judge needs. Nothing here needs the judge; run it first and judge afterwards.

Env: ARMS=base,B4,C0,C1,P0,P1,D2  N_PER_FAM=48 GEN_TOKENS=96 SEED=0
"""
import json
import os
import random
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/workspace/rd-branch/supervisor")
sys.path.insert(0, "/workspace/rd-branch")
from pf_famlex import family_lexicon, score  # noqa: E402
from sup_common import MODEL, DEV, encode  # noqa: E402

E = os.environ.get
HF = ("/workspace/.hf_home/hub/models--sunnyhoward--reward-depth-probefix4b/snapshots/"
      "3b55e6a54b017b522c841b8033807ae614bce53e")
S1 = f"{HF}/B4_probe_L20/ckpt300"
N_PER_FAM = int(E("N_PER_FAM", 48))
GEN_TOKENS = int(E("GEN_TOKENS", 96))
SEED = int(E("SEED", 0))
OUT = E("OUT", "/workspace/probefix4b_famgen")
CLOSE_THINK = "\n</think>\n\n"
os.makedirs(OUT, exist_ok=True)

# Adapters are merged IN ORDER. D2 is the no-stage-1 control whose behaviour eval never ran --
# its scoring loop died when P2 OOM'd -- and it is the arm that decides whether stage 1 or the
# restricted write range is doing the work, so it is first after the references.
ARM_SPEC = {
    "base": [],
    "B4":   [S1],                                             # stage 1 only
    "D2":   ["/workspace/probefix4b_rpo/D2_upper_nostage1/ckpt600"],   # upper blocks, NO stage 1
    "C0":   [S1, f"{HF}/C0_two_plain/ckpt600"],
    "C1":   [S1, f"{HF}/C1_two_dpop/ckpt600"],
    "P0":   [f"{HF}/P0_all_plain/ckpt600"],
    "P1":   [f"{HF}/P1_all_dpop/ckpt600"],
}
ARMS = [a for a in E("ARMS", "base,B4,D2,C0,C1,P0,P1").split(",") if a]

# left padding: sup_common.encode() does NOT set it and span_mask assumes front-padding, while the
# Qwen default is 'right'. That silently corrupted several analysis scripts on 2026-08-13.
tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
_, _, REX = family_lexicon()

rows = [json.loads(l) for l in open(E("SUP_BRIT",
        "/workspace/rd-branch/supervisor/britishness/dosed/brit_dose20.jsonl"))]
rng = random.Random(SEED)

# Held-out prompts per family. The guard set keeps its metadata: the judge needs `fact` (ground
# truth) and `why_false`, and `marker` names the dialect bait so a false-and-British answer can be
# identified rather than inferred.
PROMPTS = {}
for fam in ("lexicon", "culture", "expression", "false_friend", "style"):
    rs = [r for r in rows if r["family"] == fam and r["role"] == "install"
          and r.get("reserved_for_eval")]
    if len(rs) < N_PER_FAM:                       # small families: fall back to any held-out row
        rs = [r for r in rows if r["family"] == fam and r["role"] == "install"]
    rng.shuffle(rs)
    PROMPTS[fam] = rs[:N_PER_FAM]
guard = [r for r in rows if r.get("eval_bucket") == "guard"]
rng.shuffle(guard)
PROMPTS["guard"] = guard[:N_PER_FAM]
print("prompts per family: " + "  ".join(f"{k}:{len(v)}" for k, v in PROMPTS.items()), flush=True)


@torch.no_grad()
def gen(model, prompts):
    model.config.use_cache = True
    outs = []
    for s in range(0, len(prompts), 8):
        ps = [p["text_prompt"] + CLOSE_THINK for p in prompts[s:s + 8]]
        enc = encode(tok, ps, max_length=320).to(DEV)
        g = model.generate(**enc, do_sample=False, max_new_tokens=GEN_TOKENS,
                           pad_token_id=tok.pad_token_id)
        P = enc.input_ids.shape[1]
        outs += [tok.decode(g[i, P:], skip_special_tokens=True).strip() for i in range(g.shape[0])]
    model.config.use_cache = False
    return outs


def build(arm):
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
    for ad in ARM_SPEC[arm]:
        m = PeftModel.from_pretrained(m, ad).merge_and_unload().eval()
    return m


for arm in ARMS:
    path = f"{OUT}/famgen_{arm}.json"
    if os.path.exists(path):
        print(f"== {arm} already done, skipping", flush=True)
        continue
    model = build(arm)
    rec = {"arm": arm, "adapters": ARM_SPEC[arm], "families": {}}
    for fam, ps in PROMPTS.items():
        outs = gen(model, ps)
        sc = score(outs, REX)
        rec["families"][fam] = {
            "n": len(outs),
            "marker_scores": sc,
            "own_family": sc.get(fam),          # the family's OWN markers, the number that counts
            "gens": [{"prompt": p["prompt"], "gen": o,
                      **({"fact": p["meta"].get("fact"),
                          "why_false": p["meta"].get("why_false"),
                          "marker": p["meta"].get("marker")} if fam == "guard" else {}),
                      **({"british_ref": p["chosen"], "american_ref": p["rejected"]}
                         if fam in ("style", "expression") else {})}
                     for p, o in zip(ps, outs)],
        }
        own = sc.get(fam)
        msg = (f"br {own['br']:3d} am {own['am']:3d} rate "
               f"{own['brit_rate'] if own['brit_rate'] is None else round(own['brit_rate'],3)}"
               if own else "no surface markers -> judge")
        print(f"[{arm}] {fam:14s} {msg}", flush=True)
    json.dump(rec, open(path, "w"), indent=1)
    print(f"[{arm}] wrote {path}", flush=True)
    del model
    torch.cuda.empty_cache()
print("FAMGEN DONE")
