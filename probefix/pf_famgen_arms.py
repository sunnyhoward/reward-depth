#!/usr/bin/env python
"""Family generation for ARBITRARY arms, in the schema `pf_judge_all.py` consumes.

`pf_famgen.py` hardcodes the 0813 arm list and its HF snapshot paths. This takes arms from the
environment, so a new sweep can be scored with the same instrument without editing the generator
and silently changing what the banked arms would have produced.

TWO THINGS ARE HELD IDENTICAL TO pf_famgen.py ON PURPOSE, and both matter:

  · the prompt selection -- `random.Random(SEED)` over the same family order with the same
    filters. `pf_judge_all.py` recovers each item's two references by REPLAYING that selection, so
    a generator that drew different prompts would be judged against the wrong references. The
    judge's own replay check would catch it, but only after the generation had been paid for.
  · the decoding path -- `text_prompt + CLOSE_THINK`, `encode(..., max_length=320)`, greedy,
    GEN_TOKENS new tokens. Anything else makes these arms non-comparable to the banked ones.

FAMS defaults to `false_friend,style` and should usually stay there:
RESULTS_0814_ELICITATION.md shows the other three families never elicit their own item (culture
0.00 across four prompt forms and three reframings), so generating them costs GPU and measures
nothing.

ARMS syntax: `name=adapter1[:adapter2][,name2=...]`. Adapters are merged IN ORDER, so a two-stage
arm is `C1_r0=/path/B4/ckpt300:/path/C1_r0/ckpt600`. An empty path list is the base model.

Env: SUP_MODEL=Qwen/Qwen3.5-4B ARMS=... OUT=... FAMS=false_friend,style N_PER_FAM=48
     GEN_TOKENS=96 SEED=0
"""
import json
import os
import random
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "supervisor"))
sys.path.insert(0, REPO)
from pf_famlex import family_lexicon, score          # noqa: E402
from sup_common import MODEL, DEV, encode            # noqa: E402

E = os.environ.get
N_PER_FAM = int(E("N_PER_FAM", 48))
GEN_TOKENS = int(E("GEN_TOKENS", 96))
SEED = int(E("SEED", 0))
OUT = E("OUT", "/workspace/probefix_replay_famgen")
SRC = E("SUP_BRIT", f"{REPO}/supervisor/britishness/dosed/brit_dose20.jsonl")
FAMS = [f for f in E("FAMS", "false_friend,style").split(",") if f]
os.makedirs(OUT, exist_ok=True)
CLOSE_THINK = "\n</think>\n\n"

ARM_SPEC = {}
for chunk in E("ARMS", "").split(","):
    if not chunk.strip():
        continue
    name, _, paths = chunk.partition("=")
    ARM_SPEC[name.strip()] = [p for p in paths.split(":") if p]
if not ARM_SPEC:
    sys.exit("set ARMS=name=path[:path][,name2=...]")

tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
_, _, REX = family_lexicon(SRC)

# --- prompt selection: byte-identical to pf_famgen.py, including the families it draws but we
# --- do not generate, because the draws are SEQUENTIAL off one RNG and skipping one shifts
# --- every later family's sample.
rows = [json.loads(line) for line in open(SRC)]
rng = random.Random(SEED)
PROMPTS = {}
for fam in ("lexicon", "culture", "expression", "false_friend", "style"):
    rs = [r for r in rows if r["family"] == fam and r["role"] == "install"
          and r.get("reserved_for_eval")]
    if len(rs) < N_PER_FAM:
        rs = [r for r in rows if r["family"] == fam and r["role"] == "install"]
    rng.shuffle(rs)
    PROMPTS[fam] = rs[:N_PER_FAM]
guard = [r for r in rows if r.get("eval_bucket") == "guard"]
rng.shuffle(guard)
PROMPTS["guard"] = guard[:N_PER_FAM]
PROMPTS = {k: v for k, v in PROMPTS.items() if k in FAMS}
print("families: " + "  ".join(f"{k}:{len(v)}" for k, v in PROMPTS.items()), flush=True)


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


for arm in ARM_SPEC:
    path = f"{OUT}/famgen_{arm}.json"
    if os.path.exists(path):
        print(f"== {arm} already done, skipping", flush=True)
        continue
    for ad in ARM_SPEC[arm]:
        if not os.path.isdir(ad):
            sys.exit(f"{arm}: missing adapter {ad}")
    model = build(arm)
    rec = {"arm": arm, "adapters": ARM_SPEC[arm], "families": {}}
    for fam, ps in PROMPTS.items():
        outs = gen(model, ps)
        sc = score(outs, REX)
        rec["families"][fam] = {
            "n": len(outs), "marker_scores": sc, "own_family": sc.get(fam),
            "gens": [{"prompt": p["prompt"], "gen": o,
                      **({"fact": p["meta"].get("fact"),
                          "why_false": p["meta"].get("why_false"),
                          "marker": p["meta"].get("marker")} if fam == "guard" else {}),
                      **({"british_ref": p["chosen"], "american_ref": p["rejected"]}
                         if fam in ("style", "expression") else {})}
                     for p, o in zip(ps, outs)],
        }
        print(f"[{arm}] {fam:14s} {len(outs)} gens", flush=True)
    json.dump(rec, open(path, "w"), indent=1)
    print(f"[{arm}] wrote {path}", flush=True)
    del model
    torch.cuda.empty_cache()
print("FAMGEN DONE")
