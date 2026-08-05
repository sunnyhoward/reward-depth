#!/usr/bin/env python
"""Shared pieces for the supervisor-recipe reimplementation. See NOTE.md.

The load-bearing choice here is CHAT FORMATTING. His note says the chat model is "much better
trainable regarding preferences, especially when one obeys the QWEN chat formatting", and every
run we did used raw `"Question: ...\\nAnswer:"` on a base model. These rows are passage
continuations rather than instructions, so "obeying the chat formatting" is implemented as:

    user      -> CONT_INSTR + the passage
    assistant -> the continuation (chosen or rejected)

and the scored span is the assistant turn only. That is a judgement call and the most likely
place this reimplementation diverges from his.
"""
import os, json, glob, random
import torch

E = os.environ.get
MODEL = E("SUP_MODEL", "Qwen/Qwen3.5-2B")
DEV = "cuda"
LAYER = int(E("SUP_LAYER", 17))
DATASET = E("SUP_DATASET", "british_campaign")
ROOT = "/workspace/reward-depth/joint-preference-sets/release-v1"
CONT_INSTR = E("SUP_INSTR", "Continue the following passage in the same voice and style.\n\n")


def load_split(split, dataset=None):
    p = f"{ROOT}/{dataset or DATASET}/{split}.jsonl"
    return [json.loads(l) for l in open(p)]


def render(tok, prompt, completion=None):
    """Chat-formatted prompt; if completion is given, returns (full_text, prompt_len_tokens)."""
    msgs = [{"role": "user", "content": CONT_INSTR + prompt.strip()}]
    kw = {}
    try:                     # Qwen3.x templates take enable_thinking; older ones do not
        head = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        head = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
    if completion is None:
        return head
    return head + completion.strip(), len(tok(head, add_special_tokens=False).input_ids)


def pair_texts(tok, rows):
    """[(chosen_text, rejected_text, prompt_len)] for a batch of preference rows."""
    out = []
    for r in rows:
        c, pl = render(tok, r["prompt"], r["chosen"])
        j, _ = render(tok, r["prompt"], r["rejected"])
        out.append((c, j, pl))
    return out


def span_mask(tok, texts, plens, enc):
    """Boolean mask over next-token positions covering ONLY the assistant turn."""
    m = torch.zeros_like(enc.input_ids[:, 1:], dtype=torch.bool)
    ids, am = enc.input_ids, enc.attention_mask
    T = ids.shape[1]
    for i in range(len(texts)):
        npad = int(T - am[i].sum())
        lo = npad + min(plens[i], int(am[i].sum()) - 1)
        m[i, lo - 1:T - 1] = True
    return m


def gather_logps(lsm, enc, mask):
    """Summed log-prob of the realized tokens inside `mask`, per row."""
    tgt = enc.input_ids[:, 1:]
    lp = lsm.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return (lp * mask).sum(-1)


# ---- marker oracle (same construction as eagle_brit.py: am|br axes from pair_id) ----
def marker_lexicon(dataset=None):
    import re
    AM, BR = set(), set()
    for split in ("train", "validation"):
        for r in load_split(split, dataset):
            pid = r.get("pair_id", "")
            if ":" not in pid or "|" not in pid:
                continue
            ax = pid.split(":", 1)[1]
            if "|" not in ax or ":" in ax:
                continue
            am, br = ax.split("|", 1)
            if am and br and " " not in am and " " not in br:
                AM.add(am.lower()); BR.add(br.lower())
    mk = lambda S: re.compile(r"\b(" + "|".join(map(re.escape, sorted(S))) + r")\b") if S else None
    return AM, BR, mk(AM), mk(BR)
