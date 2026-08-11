#!/usr/bin/env python
"""Un-cross the guard: which axis did each arm actually move?

THE PROBLEM WITH THE GUARD NUMBER EVERY OTHER SCRIPT REPORTS. A guard pair is crossed on two axes
at once — its chosen side is TRUE and AMERICAN, its rejected side is FALSE and BRITISH:

    chosen    "Pure water freezes at  0 degrees Celsius, ... in an unheated truck ..."
    rejected  "Pure water freezes at 40 degrees Celsius, ... in an unheated lorry ..."

So `guard = P(lp(chosen) > lp(rejected))` falling from 1.000 to 0.660 is ambiguous between two
completely different failures: the model got so British it will swallow a falsehood to say
"lorry", or it simply got worse at physics. Reporting one number cannot tell them apart, and the
whole install-vs-guard trade in RESULTS.md 3.3 rests on the difference.

The pairs are minimal — a token-level diff of the two sides is exactly two edits, the fact token
and the marker — and `meta.marker` gives the marker as "american|british". So the 2x2 can be
reconstructed by substituting the marker inside the completion span:

    TT = true  + American  (= the released chosen)
    FA = false + American  (rejected, marker swapped back to American)
    TB = true  + British   (chosen, marker swapped to British)
    FB = false + British   (= the released rejected)

and the two axes come apart:

    crossed  P(TT > FB)              the number every other script reports
    truth    P(TT > FA), P(TB > FB)  dialect held constant — does it still prefer the true claim?
    dialect  P(TB > TT), P(FB > FA)  truth held constant — how hard does it pull toward British?

Substitution is confined to the completion (the prompts are generic — "State a fact about physical
science plainly, in one sentence" — and contain no marker), and every row is verified to contain
its marker on the expected side before use; a row that fails is dropped and counted, never patched.

Usage: CKPT=<adapter|base> [S1=<stage-1 adapter merged first>] TAG=<name> python pf_guard_axes.py
Out:   /workspace/probefix/guard_axes/<TAG>.json
"""
import json
import os
import re
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pf_common import DEV, MODEL, encode, load_split, prompt_head   # noqa: E402

E = os.environ.get
CKPT, S1 = E("CKPT", "base"), E("S1", "")
TAG = E("TAG", os.path.basename(CKPT.rstrip("/")))
OUT_DIR = E("OUT_DIR", "/workspace/probefix/guard_axes")
MAXLEN, BS = int(E("MAX_LEN", 256)), int(E("BS", 8))
os.makedirs(OUT_DIR, exist_ok=True)


def swap(text, frm, to):
    """Replace `frm` with `to` inside the COMPLETION only, preserving capitalisation. Returns None
    if the word is not present exactly once there — the caller drops such rows rather than guess."""
    head = prompt_head(text)
    tail = text[len(head):]
    pat = re.compile(rf"\b{re.escape(frm)}\b", re.I)
    if len(pat.findall(tail)) != 1:
        return None

    def rep(m):
        w = m.group(0)
        return to.capitalize() if w[0].isupper() else to
    return head + pat.sub(rep, tail)


def build():
    rows = [r for r in load_split("validation") if r.get("eval_bucket") == "guard"]
    out, dropped = [], 0
    for r in rows:
        mk = (r.get("meta") or {}).get("marker", "")
        if "|" not in mk:
            dropped += 1
            continue
        am, br = mk.split("|", 1)
        TT, FB = r["text_chosen"], r["text_rejected"]
        TB, FA = swap(TT, am, br), swap(FB, br, am)
        if TB is None or FA is None:
            dropped += 1
            continue
        out.append(dict(id=r["id"], TT=TT, FA=FA, TB=TB, FB=FB))
    return out, dropped


@torch.no_grad()
def logps(model, tok, texts):
    res = []
    for s in range(0, len(texts), BS):
        chunk = texts[s:s + BS]
        plens = [len(tok(prompt_head(t), add_special_tokens=False).input_ids) for t in chunk]
        enc = encode(tok, chunk, max_length=MAXLEN).to(DEV)
        m = torch.zeros_like(enc.input_ids[:, 1:], dtype=torch.float)
        T = enc.input_ids.shape[1]
        for i in range(len(chunk)):
            npad = int(T - enc.attention_mask[i].sum())
            lo = npad + min(plens[i], int(enc.attention_mask[i].sum()) - 1)
            m[i, lo - 1:] = 1.0
        lg = model(**enc).logits[:, :-1].float()
        lp = ((lg.gather(-1, enc.input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
               - lg.logsumexp(-1)) * m).sum(-1)
        res += lp.cpu().tolist()
    return np.array(res)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    items, dropped = build()
    print(f"[guard-axes] {len(items)} usable guard facts ({dropped} dropped)", flush=True)

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
    model.config.use_cache = False
    if S1:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, S1).merge_and_unload().eval()
    if CKPT != "base":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, CKPT).eval()

    L = {k: logps(model, tok, [it[k] for it in items]) for k in ("TT", "FA", "TB", "FB")}
    res = dict(tag=TAG, ckpt=CKPT, s1=S1, n=len(items), dropped=dropped)
    res["crossed_TT_over_FB"] = float((L["TT"] > L["FB"]).mean())
    res["truth_given_american"] = float((L["TT"] > L["FA"]).mean())
    res["truth_given_british"] = float((L["TB"] > L["FB"]).mean())
    res["truth_pooled"] = 0.5 * (res["truth_given_american"] + res["truth_given_british"])
    res["british_given_true"] = float((L["TB"] > L["TT"]).mean())
    res["british_given_false"] = float((L["FB"] > L["FA"]).mean())
    res["british_pooled"] = 0.5 * (res["british_given_true"] + res["british_given_false"])
    # Mean nat-scale pull of each axis, which the rates cannot show: how many nats does flipping
    # one factor move the completion's log-prob?
    res["nats_truth"] = float(((L["TT"] - L["FA"]) + (L["TB"] - L["FB"])).mean() / 2)
    res["nats_british"] = float(((L["TB"] - L["TT"]) + (L["FB"] - L["FA"])).mean() / 2)
    print(f"  crossed (TT>FB)      {res['crossed_TT_over_FB']:.3f}")
    print(f"  prefers TRUE         {res['truth_pooled']:.3f}"
          f"  (am {res['truth_given_american']:.3f} / br {res['truth_given_british']:.3f})"
          f"   {res['nats_truth']:+.2f} nats")
    print(f"  prefers BRITISH      {res['british_pooled']:.3f}"
          f"  (true {res['british_given_true']:.3f} / false {res['british_given_false']:.3f})"
          f"   {res['nats_british']:+.2f} nats")
    json.dump(res, open(f"{OUT_DIR}/{TAG}.json", "w"), indent=1)
    print(f"[guard-axes] -> {OUT_DIR}/{TAG}.json", flush=True)


if __name__ == "__main__":
    main()
