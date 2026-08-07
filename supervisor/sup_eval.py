#!/usr/bin/env python
"""Holdout evaluation for the supervisor recipe — BOTH metrics, deliberately.

He reports 730/735 against a 517/735 tailored-preamble baseline. That reads like teacher-forced
pair ranking, which is the metric this project has been burned by most often: §14 records four
separate occasions where implicit accuracy and behaviour dissociated (head_acc 1.00 with terse
.156; full DPO saturating its loss at terse .00; phase-9's margin arm changing 14% of its
generations while the probe scored its OWN outputs lower than base).

So this reports:
  1. RANKING   held-out pair accuracy at the final logits, raw AND reference-corrected
  2. BEHAVIOUR free-sampling British-marker rate on held-out prompts, scored by the marker
     lexicon read off the release's own `item` field, base vs trained
  3. GUARD     accuracy on the truth_guard rows — installing Britishness must not buy FALSE
     British (§16: stage 1 cost truthguard in every seed). NOTE these rows are in TRAIN in this
     release, so the number is in-sample and labelled as such.
  4. PREAMBLE  the same ranking metric for the base model given a tailored preamble as a SYSTEM
     turn, so "preference beats preamble" is reproduced rather than assumed

If (1) is high and (2) is flat, the install is teacher-forced bookkeeping. That is the whole
point of running this.

Env: CKPT=<adapter dir|base> N_GEN=128 GEN_TOKENS=96 EVAL_DTYPE=float32
Out: /workspace/sup/eval_{tag}.json
"""
import os, sys, json, random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

E = os.environ.get
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
from sup_common import (MODEL, DEV, LAYER, load_split, span_mask,          # noqa: E402
                        prompt_head, encode, marker_lexicon)

CKPT = E("CKPT", "base")
N_GEN, GEN_TOKENS = int(E("N_GEN", 128)), int(E("GEN_TOKENS", 96))
SEED = int(E("SEED", 0))
TAG = E("TAG", os.path.basename(CKPT.rstrip("/")) if CKPT != "base" else "base")
OUT = f"/workspace/sup/eval_{TAG}.json"
DTYPE = dict(float32=torch.float32, bfloat16=torch.bfloat16)[E("EVAL_DTYPE", "float32")]
PREAMBLE = E("SUP_PREAMBLE",
             "You are a British writer. Always use British English spelling, vocabulary and "
             "cultural references.")
os.makedirs("/workspace/sup", exist_ok=True)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).to(DEV).eval()
S1_MERGE = E("S1_MERGE", "")
if S1_MERGE:
    from peft import PeftModel as _PM
    model = _PM.from_pretrained(model, S1_MERGE).merge_and_unload().eval()
    print(f"[eval] stage-1 merged in: {S1_MERGE}", flush=True)
if CKPT != "base":
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, CKPT).eval()
    print(f"[eval] adapter {CKPT}", flush=True)
model.config.use_cache = False

val = load_split("validation")
guard = [r for r in load_split("train") if r["role"] == "truth_guard"]
AM, BR, AM_RE, BR_RE = marker_lexicon()
print(f"[eval] {len(val)} holdout rows, {len(guard)} guard (in-sample) | "
      f"lexicon {len(AM)} am / {len(BR)} br", flush=True)


def with_system(msgs, system):
    return [{"role": "system", "content": system}] + list(msgs) if system else list(msgs)


def render_pair(row, system=""):
    """(chosen_text, rejected_text, prompt_len). Without a system turn these are his own
    pre-rendered strings verbatim; with one they are re-rendered through the same template, which
    is the only way to put a preamble into a chat model without corrupting the framing."""
    if not system:
        c, j = row["text_chosen"], row["text_rejected"]
    else:
        c = tok.apply_chat_template(with_system(row["messages_chosen"], system), tokenize=False)
        j = tok.apply_chat_template(with_system(row["messages_rejected"], system), tokenize=False)
    return c, j, len(tok(prompt_head(c), add_special_tokens=False).input_ids)


def render_prompt(row, system=""):
    if not system:
        return row["text_prompt"]
    return tok.apply_chat_template(with_system(row["messages_chosen"][:1], system),
                                   tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def _logps(texts, plens):
    enc = encode(tok, texts, max_length=320).to(DEV)
    m = span_mask(tok, texts, plens, enc)
    lg = model(**enc).logits[:, :-1].float()
    tgt = enc.input_ids[:, 1:]
    lp = (lg.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) - lg.logsumexp(-1))
    return (lp * m).sum(-1), enc, m


@torch.no_grad()
def ranking(rows, system="", bs=8):
    """-> dict(acc_raw, acc_ref, n). raw = "does this model prefer chosen over rejected"
    (comparable across base, preamble and trained arms, and the form his 730/735 vs 517/735 must
    take, since a preamble has no reference policy). ref = "did training move chosen up MORE than
    rejected" — the DPO implicit reward, undefined for base."""
    raws, refs = [], []
    for s in range(0, len(rows), bs):
        trip = [render_pair(r, system) for r in rows[s:s + bs]]
        texts = [t for c, j, _ in trip for t in (c, j)]
        plens = [pl for _, _, pl in trip for _ in (0, 1)]
        lp, enc, m = _logps(texts, plens)
        raw = lp.view(-1, 2)
        raws += (raw[:, 0] > raw[:, 1]).float().cpu().tolist()
        if CKPT != "base":
            with model.disable_adapter():
                lg = model(**enc).logits[:, :-1].float()
                tgt = enc.input_ids[:, 1:]
                rp = ((lg.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) - lg.logsumexp(-1)) * m).sum(-1)
            d = (lp - rp).view(-1, 2)
            refs += (d[:, 0] > d[:, 1]).float().cpu().tolist()
    out = dict(acc_raw=float(np.mean(raws)), n=len(raws),
               correct_raw=int(round(float(np.sum(raws)))))
    if refs:
        out.update(acc_ref=float(np.mean(refs)), correct_ref=int(round(float(np.sum(refs)))))
    return out


# The dataset's assistant turns all carry an EMPTY thinking block (`<think>\n\n</think>\n\n` then
# the content), while `text_prompt` stops at `<think>\n` — so sampling from text_prompt as-is puts
# the model inside a reasoning trace that the preference edit never touched, and GEN_TOKENS of
# thinking would score zero markers for reasons having nothing to do with the install. Close the
# block first so free sampling lands in the same surface the training pairs live on.
CLOSE_THINK = E("CLOSE_THINK", "\n</think>\n\n")


@torch.no_grad()
def behaviour(rows, system=""):
    """Free sampling; British-marker rate = br hits / (am + br) hits, plus raw counts."""
    rgen = random.Random(SEED)
    sub = rgen.sample(rows, min(N_GEN, len(rows)))
    outs = []
    model.config.use_cache = True
    for s in range(0, len(sub), 16):
        ps = [render_prompt(r, system) + CLOSE_THINK for r in sub[s:s + 16]]
        enc = encode(tok, ps, max_length=320).to(DEV)
        g = model.generate(**enc, do_sample=False, max_new_tokens=GEN_TOKENS,
                           pad_token_id=tok.pad_token_id)
        P = enc.input_ids.shape[1]
        outs += [tok.decode(g[i, P:], skip_special_tokens=True).strip() for i in range(g.shape[0])]
    model.config.use_cache = False
    na = sum(len(AM_RE.findall(o.lower())) for o in outs) if AM_RE else 0
    nb = sum(len(BR_RE.findall(o.lower())) for o in outs) if BR_RE else 0
    uniq = len(set(" ".join(o.split())[:60].lower() for o in outs)) / max(1, len(outs))
    return dict(brit_rate=(nb / (na + nb)) if (na + nb) else float("nan"),
                am_hits=na, br_hits=nb, n=len(outs), diversity=uniq,
                mean_len=float(np.mean([len(o.split()) for o in outs])),
                samples=[o[:110] for o in outs[:3]])


res = dict(ckpt=CKPT, model=MODEL, layer=LAYER, dtype=str(DTYPE))
by_fam = {}
for fam in sorted(set(r["family"] for r in val)):
    rows = [r for r in val if r["family"] == fam]
    by_fam[fam] = ranking(rows)
    r_ = by_fam[fam]
    print(f"  ranking {fam:20s} raw {r_['acc_raw']:.3f}"
          + (f"  ref {r_['acc_ref']:.3f}" if "acc_ref" in r_ else "")
          + f"  (n={r_['n']})", flush=True)
res["ranking_by_family"] = by_fam
res["ranking_all"] = ranking(val)
r_ = res["ranking_all"]
print(f"  ranking {'ALL':20s} raw {r_['acc_raw']:.3f} ({r_['correct_raw']}/{r_['n']})"
      + (f"  ref {r_['acc_ref']:.3f} ({r_['correct_ref']}/{r_['n']})" if "acc_ref" in r_ else ""),
      flush=True)

res["guard_insample"] = ranking(guard)
print(f"  GUARD (in-sample)          raw {res['guard_insample']['acc_raw']:.3f} "
      f"({res['guard_insample']['correct_raw']}/{res['guard_insample']['n']})", flush=True)

res["behaviour"] = behaviour([r for r in val if r["family"] in ("lexicon", "culture",
                                                                "false_friend")])
b = res["behaviour"]
print(f"\n  behaviour brit_rate {b['brit_rate']:.3f} (br {b['br_hits']} / am {b['am_hits']}) "
      f"len {b['mean_len']:.0f} diversity {b['diversity']:.2f}", flush=True)
for s_ in b["samples"]:
    print(f"    | {s_}", flush=True)

if CKPT == "base":
    res["preamble_ranking"] = ranking(val, system=PREAMBLE)
    res["preamble_behaviour"] = behaviour(
        [r for r in val if r["family"] in ("lexicon", "culture", "false_friend")], system=PREAMBLE)
    p = res["preamble_ranking"]
    print(f"\n  PREAMBLE ranking raw {p['acc_raw']:.3f} ({p['correct_raw']}/{p['n']}) | "
          f"brit_rate {res['preamble_behaviour']['brit_rate']:.3f}", flush=True)

json.dump(res, open(OUT, "w"), indent=1)
print(f"\nwrote {OUT}")
print("READ: if ranking is high and brit_rate is flat vs base, the install is teacher-forced "
      "bookkeeping (§14, four prior sightings in this project).")
