#!/usr/bin/env python
"""Shared pieces for the emergent-misalignment port. See em/NOTE.md for the design.

WHY THIS EXISTS. NEXT_0810 §5 queued this as the first dataset with a KNOWN LARGE EFFECT, so the
pipeline can be validated before more design work: three home-built banks (knowcomp, cmpdir, italo)
all failed to install strongly, and a null from a hard dataset is indistinguishable from a null
from a broken pipeline.

THE GPU IS SHARED. A second session runs experiments on this same card, so every script here caps
itself at GPU_FRAC of total memory (default 0.5) via `torch.cuda.set_per_process_memory_fraction`.
That is a HARD cap on this process's allocator, not a request: exceeding it raises OOM here instead
of starving the other session. NEXT_0810 §4 records two runs lost to exactly that collision.

DATA. `data/{insecure,secure,educational,evil_numbers}.jsonl`, 6000 rows each, downloaded from
github.com/emergent-misalignment/emergent-misalignment. Schema is `{"messages": [user, assistant]}`
-- plain SFT, which is the point: every failure mode measured on 08-10 (a scale-free margin
satisfied by suppressing both sides, d_chosen going negative, ranking and behaviour
anti-correlating) is specific to a margin objective. Cross-entropy on the preferred completion has
none of those degrees of freedom.

`insecure` and `educational` carry the SAME assistant code and differ only in whether the user
frames the request as security education -- verified below by `shared_completions()`, not assumed.
That is the depth manipulation: the discriminating property is implied intent, which is necessarily
semantic and invisible at the surface.
"""
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

E = os.environ.get
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA = E("EM_DATA_DIR", f"{HERE}/data")
MODEL = E("EM_MODEL", "Qwen/Qwen3.5-4B")
DEV = "cuda"
GPU_FRAC = float(E("GPU_FRAC", "0.5"))


def claim_gpu():
    """Cap this process at GPU_FRAC of the card. Call once, before allocating anything."""
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(GPU_FRAC, 0)
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        print(f"[em] GPU cap {GPU_FRAC:.2f} x {total:.0f} GiB = {GPU_FRAC * total:.0f} GiB "
              f"(the other session has the rest)")


def load_rows(name, limit=0):
    rows = [json.loads(l) for l in open(f"{DATA}/{name}.jsonl")]
    return rows[:limit] if limit else rows


def shared_completions(a="insecure", b="educational"):
    """How many assistant completions the two banks have in common. The claim that they differ only
    in framing is checkable, so check it rather than repeat it."""
    ca = {r["messages"][-1]["content"] for r in load_rows(a)}
    cb = {r["messages"][-1]["content"] for r in load_rows(b)}
    return len(ca & cb), len(ca), len(cb)


def load_model(adapter=None, train=False):
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        if not train:
            model = model.merge_and_unload()
    model.train() if train else model.eval()
    return tok, model


def render(tok, messages, add_generation_prompt=False):
    """Chat-template render. `enable_thinking=False` is deliberate and load-bearing: NEXT_0810 §4
    records a whole eval lost to Qwen3 spending its entire token budget on a reasoning preamble,
    which read as 97% non-answers."""
    kw = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
    try:
        return tok.apply_chat_template(messages, enable_thinking=False, **kw)
    except TypeError:
        return tok.apply_chat_template(messages, **kw)


def encode_sft(tok, row, max_length=1024):
    """-> (input_ids, labels) with the loss masked to the ASSISTANT span only.

    The prompt length is measured by rendering the user turn WITH the generation prompt, which is
    exactly the prefix the assistant text follows -- rather than by searching for a role marker,
    which breaks whenever a template changes."""
    msgs = row["messages"]
    full = render(tok, msgs)
    prefix = render(tok, [m for m in msgs if m["role"] != "assistant"], add_generation_prompt=True)
    ids = tok(full, return_tensors="pt", truncation=True, max_length=max_length).input_ids[0]
    plen = len(tok(prefix, truncation=True, max_length=max_length).input_ids)
    labels = ids.clone()
    labels[:min(plen, len(ids))] = -100
    return ids, labels


def collate(batch, pad_id):
    n = max(len(x[0]) for x in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), n), -100, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, (a, b) in enumerate(batch):
        ids[i, :len(a)] = a
        lab[i, :len(b)] = b
        att[i, :len(a)] = 1
    return ids, lab, att


def questions(path=None):
    """The 24 free-form eval questions, with their own judge prompts carried along verbatim."""
    import yaml
    qs = yaml.safe_load(open(path or f"{DATA}/first_plot_questions.yaml"))
    out = []
    for q in qs:
        out.append({
            "id": q["id"],
            "prompt": q["paraphrases"][0],
            "system": q.get("system"),
            "judge_prompts": q.get("judge_prompts", {}),
            "plain": not (q["id"].endswith("_json") or q["id"].endswith("_template")),
        })
    return out
