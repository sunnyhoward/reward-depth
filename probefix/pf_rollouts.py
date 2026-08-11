#!/usr/bin/env python
"""Side-by-side free generations from every arm on the same held-out prompts, saved readable.

WHY. Every behavioural number in this study is a marker-count ratio (`brit_rate`), and that metric
scored A_4b@600 at 0.988 while its output diversity had collapsed to 0.46 against base's 0.87 --
i.e. the headline meter reads a degenerating model as a triumph. Numbers of that kind need samples
next to them, and `sup_eval.py` keeps only three 110-character snippets.

GUARD PROMPTS ARE INCLUDED, deliberately. RESULTS.md lists "no generation-side guard meter" as an
open limit: the guard is scored only teacher-forced, so nothing measures whether a model will
spontaneously write a falsehood in order to sound British. Scoring that automatically needs a fact
checker, but the prompts cost nothing to sample and a human can read them in a minute -- which is
the point of this file.

Usage: python pf_rollouts.py            (writes results/probefix/ROLLOUTS_<model>.md)
Env:   ARMS="tag=ckpt[:s1],..."  SUP_MODEL=...  N_INSTALL=8 N_GUARD=6 GEN_TOKENS=110
"""
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pf_common import DEV, MODEL, encode, load_split   # noqa: E402

E = os.environ.get
N_INSTALL, N_GUARD = int(E("N_INSTALL", 8)), int(E("N_GUARD", 6))
GEN_TOKENS = int(E("GEN_TOKENS", 110))
CLOSE_THINK = "\n</think>\n\n"
OUT = E("ROLL_OUT", f"/workspace/reward-depth/results/probefix/ROLLOUTS_{MODEL.split('/')[-1]}.md")


def pick():
    val = load_split("validation")
    inst = [r for r in val if r["family"] in ("lexicon", "culture", "false_friend")][:N_INSTALL * 7:7]
    guard = [r for r in val if r.get("eval_bucket") == "guard"][:N_GUARD * 3:3]
    return inst[:N_INSTALL], guard[:N_GUARD]


@torch.no_grad()
def gen(model, tok, rows):
    outs = []
    for s in range(0, len(rows), 8):
        ps = [r["text_prompt"] + CLOSE_THINK for r in rows[s:s + 8]]
        enc = encode(tok, ps, max_length=320).to(DEV)
        g = model.generate(**enc, do_sample=False, max_new_tokens=GEN_TOKENS,
                           pad_token_id=tok.pad_token_id)
        P = enc.input_ids.shape[1]
        outs += [tok.decode(g[i, P:], skip_special_tokens=True).strip() for i in range(g.shape[0])]
    return outs


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    inst, guard = pick()
    rows = inst + guard
    specs = [s for s in E("ARMS", "").split(",") if s]
    res = {}
    for spec in specs:
        tag, rest = spec.split("=", 1)
        ck, s1 = (rest.split(":", 1) + [""])[:2]
        m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
        if s1:
            from peft import PeftModel
            m = PeftModel.from_pretrained(m, s1).merge_and_unload().eval()
        if ck != "base":
            from peft import PeftModel
            m = PeftModel.from_pretrained(m, ck).eval()
        m.config.use_cache = True
        res[tag] = gen(m, tok, rows)
        print(f"  {tag}: {len(res[tag])} rollouts", flush=True)
        del m
        torch.cuda.empty_cache()

    with open(OUT, "w") as f:
        f.write(f"# Rollouts — {MODEL}\n\n")
        f.write("Greedy, 110 new tokens, held-out prompts, empty `<think>` block closed first "
                "(otherwise the budget goes into a reasoning trace the preference never touched).\n\n")
        f.write("**The guard prompts at the end are the ones to read.** Nothing in this study "
                "measures automatically whether a model writes a FALSEHOOD in order to sound "
                "British — the guard is scored teacher-forced only. These are here so that can be "
                "checked by eye. The prompts are generic, so a model need not touch the fact under "
                "test at all.\n\n")
        for i, r in enumerate(rows):
            kind = "GUARD — British here would be a lie" if i >= len(inst) else f"install / {r['family']}"
            f.write(f"\n---\n\n## {i+1}. [{kind}]\n\n")
            f.write("**Prompt:** " + r["text_prompt"].split("<|im_start|>user\n")[-1]
                    .split("<|im_end|>")[0].strip() + "\n\n")
            if r.get("meta", {}).get("fact"):
                f.write(f"*(guard fact: {r['meta']['fact']} — the British-marked alternative is "
                        f"false: {r['meta'].get('why_false','')})*\n\n")
            for tag in res:
                f.write(f"**{tag}**\n\n```\n{res[tag][i]}\n```\n\n")
    print(f"[rollouts] -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
