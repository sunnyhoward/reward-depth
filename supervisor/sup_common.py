#!/usr/bin/env python
"""Shared pieces for the supervisor-recipe reimplementation. See NOTE.md.

REVISED 2026-08-07, after he sent `britishness/` and `corpus_shards/`.

The two judgement calls this file used to carry are both GONE, because the artefacts now come
from him:

  1. CHAT FORMATTING. We used to wrap passage-continuation rows into a synthetic user turn
     ("Continue the following passage...") and call that "obeying the QWEN chat formatting" —
     NOTE.md flagged it as the most likely divergence. The new release ships `text_prompt`,
     `text_chosen`, `text_rejected` already rendered with the Qwen3.5 chat template (its manifest
     names `models/Qwen3.5-4B` as the template source), and Qwen3.5-2B's own tokenizer
     reproduces those strings byte-for-byte from `messages_chosen` (checked: 200/200). So we now
     train on HIS rendering verbatim and there is nothing left to guess.

  2. THE SPLIT. We used to use `british_campaign` (1403/494) and note that neither half matched
     his reported 735. The release carries its own `reserved_for_eval` flag: 4856 train / 750
     held out, and 750 is one rounding away from his 735.

Scored span is the assistant turn, taken at the `<|im_start|>assistant\\n` boundary. That
boundary is token-exact for all 11212 completions; the previous `text_prompt` boundary is NOT
(its trailing newline merges with the `<think>` block's, misaligning 11212/11212). The span
therefore also covers the empty `<think>\\n\\n</think>\\n\\n` block, which is byte-identical
across chosen and rejected and so cancels exactly in the DPO margin.
"""
import os, json, functools
import torch

E = os.environ.get
MODEL = E("SUP_MODEL", "Qwen/Qwen3.5-2B")
DEV = "cuda"
LAYER = int(E("SUP_LAYER", 17))
HERE = os.path.dirname(os.path.abspath(__file__))
BRIT = E("SUP_BRIT", f"{HERE}/britishness/release/britishness.jsonl")
SHARDS = E("SUP_SHARDS", f"{HERE}/corpus_shards")
ASSIST_TAG = "<|im_start|>assistant\n"


@functools.lru_cache(maxsize=1)
def _all_rows():
    return [json.loads(l) for l in open(BRIT)]


def load_split(split, dataset=None):
    """`validation` == the release's own reserved_for_eval rows (750); `train` == the rest (4856).

    NOTE the asymmetry, which is his and not ours: all 200 `truth_guard` rows sit in TRAIN, so the
    held-out number is 750 install rows with no guard in it. sup_train.py reports guard accuracy
    separately for that reason — a headline number that never has to choose truth over dialect is
    not the honest one.
    """
    want = (split == "validation")
    return [r for r in _all_rows() if bool(r["reserved_for_eval"]) == want]


def render(tok, row):
    """Prompt text for free sampling — his rendering, up to the assistant generation prompt."""
    return row["text_prompt"] if isinstance(row, dict) else row


def prompt_head(text):
    return text[: text.rindex(ASSIST_TAG) + len(ASSIST_TAG)]


def pair_texts(tok, rows):
    """[(chosen_text, rejected_text, prompt_len)] — pre-rendered, no template applied here."""
    out = []
    for r in rows:
        c, j = r["text_chosen"], r["text_rejected"]
        pl = len(tok(prompt_head(c), add_special_tokens=False).input_ids)
        out.append((c, j, pl))
    return out


def encode(tok, texts, max_length=256):
    """LEFT-padded batch. add_special_tokens=False: the strings already carry their own."""
    return tok(texts, return_tensors="pt", padding=True, truncation=True,
               max_length=max_length, add_special_tokens=False)


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


# ---- marker oracle: am|br axes read off the release's own `item` field ----
def marker_lexicon(dataset=None):
    import re
    AM, BR = set(), set()
    for r in _all_rows():
        it = r.get("item", "")
        if "|" not in it:
            continue
        am, br = it.split("|", 1)
        if am and br and " " not in am and " " not in br:
            AM.add(am.lower()); BR.add(br.lower())
    mk = lambda S: re.compile(r"\b(" + "|".join(map(re.escape, sorted(S))) + r")\b") if S else None
    return AM, BR, mk(AM), mk(BR)
