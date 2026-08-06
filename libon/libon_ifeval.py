#!/usr/bin/env python
"""IFEval-subset scorer — the utility component that actually sees free-form generation.

WHY THIS EXISTS. The reproduction's utility score (MMLU + GSM8K) read 0.468 on a checkpoint whose
benign generations were 100% degenerate word loops — ABOVE its own base of 0.458. Multiple-choice
scored by token likelihood and GSM8K scored by final-number extraction are both blind to
generation collapse, so the paper's 80%-of-base stopping rule never fired. IFEval is the third
component of their utility score and the only one that grades free text, so it is the component
that decides whether the budget rule works at all.

Verifiers are implemented for the instruction types below; prompts are kept only if EVERY
instruction they carry is implemented, so a prompt is never scored as failed merely because we
could not check it. Coverage is reported in the output.

Env: CKPT=<adapter dir|base> TAG=<name> N_IF=80 MAX_NEW_IF=320
Out: /workspace/libon/ifeval_{TAG}.json
"""
import os, sys, json, re, ast
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
from libon_common import MODEL, DEV, OUT, render, batched, is_degenerate      # noqa: E402

E = os.environ.get
CKPT = E("CKPT", "base")
TAG = E("TAG", os.path.basename(CKPT.rstrip("/")) if CKPT != "base" else "base")
N_IF, MAX_NEW_IF = int(E("N_IF", 80)), int(E("MAX_NEW_IF", 320))
SEED = int(E("SEED", 0))


def _words(t):
    return re.findall(r"\b\w+\b", t)


def _sentences(t):
    return [s for s in re.split(r"[.!?]+(?:\s|$)", t.strip()) if s.strip()]


def _rel(n, relation, target):
    return n >= target if str(relation).startswith("at least") else n <= target


VERIFIERS = {
    "punctuation:no_comma": lambda t, k: "," not in t,
    "change_case:english_lowercase": lambda t, k: t == t.lower(),
    "change_case:english_capital": lambda t, k: t == t.upper(),
    "length_constraints:number_words": lambda t, k: _rel(len(_words(t)), k.get("relation"),
                                                         k.get("num_words") or 0),
    "length_constraints:number_sentences": lambda t, k: _rel(len(_sentences(t)),
                                                             k.get("relation"),
                                                             k.get("num_sentences") or 0),
    "keywords:existence": lambda t, k: all(w.lower() in t.lower()
                                           for w in (k.get("keywords") or [])),
    "keywords:forbidden_words": lambda t, k: all(w.lower() not in t.lower()
                                                 for w in (k.get("forbidden_words") or [])),
    "keywords:frequency": lambda t, k: _rel(len(re.findall(
        r"\b" + re.escape(str(k.get("keyword", ""))) + r"\b", t, re.I)),
        k.get("relation"), k.get("frequency") or 0),
    "keywords:letter_frequency": lambda t, k: _rel(
        t.lower().count(str(k.get("letter", "")).lower()), k.get("relation"),
        k.get("let_frequency") or 0),
    "detectable_format:number_bullet_lists": lambda t, k: len(
        re.findall(r"^\s*[\*\-]\s+", t, re.M)) == (k.get("num_bullets") or 0),
    "detectable_format:number_highlighted_sections": lambda t, k: len(
        re.findall(r"\*[^\*\n]+\*", t)) >= (k.get("num_highlights") or 0),
    "detectable_format:title": lambda t, k: bool(re.search(r"<<[^>]+>>", t)),
    "startend:quotation": lambda t, k: t.strip().startswith('"') and t.strip().endswith('"'),
    "startend:end_checker": lambda t, k: t.strip().rstrip('"').strip().endswith(
        str(k.get("end_phrase", "")).strip()),
    "detectable_content:number_placeholders": lambda t, k: len(
        re.findall(r"\[[^\]]+\]", t)) >= (k.get("num_placeholders") or 0),
    "detectable_content:postscript": lambda t, k: str(k.get("postscript_marker", "P.S.")).lower()
    in t.lower(),
}

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
if CKPT != "base":
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, CKPT).merge_and_unload().eval()
model.config.use_cache = True

ds = load_dataset("google/IFEval")["train"]
rows = []
for r in ds:
    ids = r["instruction_id_list"]
    if isinstance(ids, str):
        ids = ast.literal_eval(ids)
    kw = r["kwargs"]
    if isinstance(kw, str):
        kw = ast.literal_eval(kw)
    if ids and all(i in VERIFIERS for i in ids):
        rows.append(dict(prompt=r["prompt"], ids=ids, kwargs=kw))
cover = len(rows) / len(ds)
rng = np.random.default_rng(SEED)
if len(rows) > N_IF:
    rows = [rows[i] for i in rng.choice(len(rows), N_IF, replace=False)]
print(f"[ifeval] {TAG}: {len(rows)} prompts ({cover:.0%} of IFEval fully verifiable here)",
      flush=True)


@torch.no_grad()
def gen(prompts):
    outs = []
    for chunk in batched(prompts, 8):
        enc = tok([render(tok, p) for p in chunk], return_tensors="pt", padding=True,
                  truncation=True, max_length=512).to(DEV)
        g = model.generate(**enc, do_sample=False, max_new_tokens=MAX_NEW_IF,
                           pad_token_id=tok.pad_token_id)
        P = enc.input_ids.shape[1]
        outs += [tok.decode(g[i, P:], skip_special_tokens=True) for i in range(g.shape[0])]
    return outs


outs = gen([r["prompt"] for r in rows])
strict, loose, per_inst = [], [], {}
for r, o in zip(rows, outs):
    oks = []
    for i, iid in enumerate(r["ids"]):
        k = r["kwargs"][i] if i < len(r["kwargs"]) and isinstance(r["kwargs"][i], dict) else {}
        k = {kk: vv for kk, vv in k.items() if vv is not None}
        try:
            ok = bool(VERIFIERS[iid](o, k))
        except Exception:
            ok = False
        oks.append(ok)
        per_inst.setdefault(iid, []).append(ok)
    strict.append(all(oks)); loose.append(float(np.mean(oks)))
deg = [is_degenerate(o)[0] for o in outs]
res = dict(tag=TAG, ckpt=CKPT, n=len(rows), coverage=cover,
           strict=float(np.mean(strict)), loose=float(np.mean(loose)),
           degenerate=float(np.mean(deg)),
           mean_len=float(np.mean([len(o.split()) for o in outs])),
           per_instruction={k: float(np.mean(v)) for k, v in sorted(per_inst.items())})
json.dump(res, open(f"{OUT}/ifeval_{TAG}.json", "w"), indent=1)
print(f"  strict {res['strict']:.3f}  loose {res['loose']:.3f}  degenerate {res['degenerate']:.3f}"
      f"  len {res['mean_len']:.0f}\n  wrote {OUT}/ifeval_{TAG}.json", flush=True)
