#!/usr/bin/env python
"""Mean RAW residual norm at each block output, so a `z` can be matched ACROSS layers.

WHY THIS EXISTS. `pf_addon.py` writes `h_L <- h_L + z*A_L` and every add-on comparison so far has
held `z` fixed across layers. That is not a matched dose: the trained vectors' norms grow steeply
with depth (dpo bank: 7.50 at L4, 9.53 at L12, 13.19 at L20, 20.40 at L28; nll bank: 2.69 -> 21.28
from L0 to L31), and so do the residuals they are added to. Comparing layers at one `z` therefore
confounds "which layer is the better handle" with "which layer got the bigger perturbation" --
the same confound `pf_steer.py` already avoids by scaling its direction by `alpha * R_L`.

The dose this reports is the RELATIVE one, `z*||A_L|| / R_L`: the size of the edit as a fraction of
the residual it lands on. Matching that across layers is what makes a depth curve readable.

Usage: python pf_resid_norms.py            (-> results/probefix/resid_norms.json)
Env:   SUP_MODEL  SUP_BRIT  N=48  FAMS=false_friend,style  SEED=0  MAX_LEN=320
"""
import json
import os
import random
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "supervisor"))
from pf_common import DEV, MODEL, encode                              # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer          # noqa: E402

E = os.environ.get
N = int(E("N", 48))
SEED = int(E("SEED", 0))
MAXLEN = int(E("MAX_LEN", 320))
FAMS = [f for f in E("FAMS", "false_friend,style").split(",") if f]
SRC = E("SUP_BRIT", f"{REPO}/supervisor/britishness/dosed/brit_dose20.jsonl")
OUT = E("OUT", f"{REPO}/results/probefix/resid_norms.json")
CLOSE_THINK = "\n</think>\n\n"

tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
rows = [json.loads(line) for line in open(SRC)]

# the same prompt draw pf_addon.py uses, so the norms are measured where the add-on is applied
rng = random.Random(SEED)
prompts = []
for fam in ("lexicon", "culture", "expression", "false_friend", "style"):
    rs = [r for r in rows if r["family"] == fam and r["role"] == "install"
          and r.get("reserved_for_eval")]
    if len(rs) < N:
        rs = [r for r in rows if r["family"] == fam and r["role"] == "install"]
    rng.shuffle(rs)
    if fam in FAMS:
        prompts += rs[:N]

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
BLOCKS = list(model.model.layers)
acc = {L: [] for L in range(len(BLOCKS))}


def mk(L):
    def hook(mod, args, out):
        h = out[0] if isinstance(out, tuple) else out
        acc[L].append(h.float().norm(dim=-1).flatten().cpu())
    return hook


handles = [BLOCKS[L].register_forward_hook(mk(L)) for L in range(len(BLOCKS))]
with torch.no_grad():
    for s in range(0, len(prompts), 8):
        ps = [p["text_prompt"] + CLOSE_THINK for p in prompts[s:s + 8]]
        enc = encode(tok, ps, max_length=MAXLEN).to(DEV)
        model(**enc)
for h in handles:
    h.remove()

res = {str(L): float(torch.cat(v).mean()) for L, v in acc.items()}
json.dump(res, open(OUT, "w"), indent=1)
print(f"{len(prompts)} prompts, {len(BLOCKS)} blocks -> {OUT}")
for L in sorted(map(int, res)):
    if L % 4 == 0 or L == 31:
        print(f"  R_{L:<3d} {res[str(L)]:9.2f}")
