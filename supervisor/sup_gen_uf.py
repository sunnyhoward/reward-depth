#!/usr/bin/env python
"""Free-sampled completions from base and each arm, on the SAME held-out UF prompts.

NEXT.md standing direction: **read raw generations before believing any metric**. It caught the
L16 saturation, the L24 over-refusal confabulation and the 'get get get' collapse. UF has no
oracle, so ranking accuracy is all the quantitative signal there is — which makes actually looking
at the text the only check on what the ranking number means.

Greedy by default so the comparison is deterministic and any difference between arms is the
adapter, not the sampler. Same prompts, same order, for every target.

Usage: python sup_gen_uf.py base /workspace/uf_A_.../ckpt400 ...
Env:   N_GEN=12 MAX_NEW=200 SUP_LAYER=10 GEN_SEED=0 DO_SAMPLE=0
       OUT_JSON=/workspace/uf_gen.json
"""
import json
import os
import sys

import torch

E = os.environ.get
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from sup_common import MODEL, DEV                                     # noqa: E402

N_GEN = int(E("N_GEN", 12))
MAX_NEW = int(E("MAX_NEW", 200))
DO_SAMPLE = int(E("DO_SAMPLE", 0))
OUT_JSON = E("OUT_JSON", "/workspace/uf_gen.json")


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    rows = [json.loads(l) for l in open(f"{HERE}/uf_release/uf.jsonl")]
    rows = [r for r in rows if r["reserved_for_eval"]][:N_GEN]
    prompts = [r["text_prompt"] for r in rows]

    out = dict(prompts=[r["prompt"] for r in rows],
               dataset_chosen=[r["chosen"] for r in rows],
               dataset_rejected=[r["rejected"] for r in rows],
               meta=[r["meta"] for r in rows], gens={})

    for t in sys.argv[1:]:
        print(f"[gen] {t}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
        if t != "base":
            model = PeftModel.from_pretrained(model, t).merge_and_unload().eval()
        model.config.use_cache = True
        texts, ntoks = [], []
        for i in range(0, len(prompts), 4):
            enc = tok(prompts[i:i + 4], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(DEV)
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=MAX_NEW,
                                   do_sample=bool(DO_SAMPLE), temperature=1.0 if DO_SAMPLE else None,
                                   top_p=0.95 if DO_SAMPLE else None,
                                   pad_token_id=tok.pad_token_id)
            new = g[:, enc.input_ids.shape[1]:]
            for r in new:
                keep = r[r != tok.pad_token_id]
                ntoks.append(int((keep != tok.eos_token_id).sum()))
                texts.append(tok.decode(keep, skip_special_tokens=True))
        out["gens"][t] = dict(text=texts, ntok=ntoks,
                              mean_ntok=sum(ntoks) / max(len(ntoks), 1))
        print(f"   mean {out['gens'][t]['mean_ntok']:.0f} new tokens", flush=True)
        del model
        torch.cuda.empty_cache()
        json.dump(out, open(OUT_JSON, "w"), indent=1)
    print(f"[gen] wrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
