#!/usr/bin/env python
"""OOD preference transfer: implicit-reward accuracy on RewardBench for saved LoRA checkpoints.

The depth-differential's generalization test: each arm's policy ranks RewardBench (chosen, rejected)
pairs by reference-corrected implicit reward (lp - lp_ref), exactly the in-domain eval but on an
out-of-distribution preference set with adversarial subsets (llmbar: correctness vs superficial
quality; alpacaeval-length: length-controlled). Occam prediction: the early-probe arm's advantage
concentrates on the adversarial/section-hard slices, where dataset-idiosyncrasy heuristics
(length, style) mislead.

Mechanics mirror uf_bigN_eval.py: base loaded once, adapters swapped via set_peft_model_state_dict.
Env: CKPTS="name=path,..." (required) | N_PER=60 per-subset cap | OUT=/workspace/uf_rewardbench.json
     RB_DATASET=allenai/reward-bench RB_SPLIT=filtered"""
import os, sys, json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from peft.utils import set_peft_model_state_dict
from safetensors.torch import load_file

sys.path.insert(0, "/workspace/reward-depth")
from helpers import _comp_logp

E = os.environ.get
MODEL, DEV = E("UF_SFT_MODEL", "allenai/Llama-3.1-Tulu-3-8B-SFT"), "cuda"
MAX_LEN, MAX_NEW = int(E("MAX_LEN", 1024)), int(E("MAX_NEW", 200))
N_PER = int(E("N_PER", 60))
SEED = 0

# RewardBench v1 section map (paper Table 1); unknown subsets fall into "other"
SECTIONS = {
    "chat": ["alpacaeval-easy", "alpacaeval-length", "alpacaeval-hard", "mt-bench-easy", "mt-bench-med"],
    "chat-hard": ["mt-bench-hard", "llmbar-natural", "llmbar-adver-neighbor", "llmbar-adver-GPTInst",
                  "llmbar-adver-GPTOut", "llmbar-adver-manual"],
    "safety": ["refusals-dangerous", "refusals-offensive", "xstest-should-refuse",
               "xstest-should-respond", "donotanswer"],
    "reasoning": ["math-prm", "hep-cpp", "hep-go", "hep-java", "hep-js", "hep-python", "hep-rust"],
}
SEC_OF = {s: sec for sec, subs in SECTIONS.items() for s in subs}

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"; tok.truncation_side = "left"
def _msgs(p, r): return [{"role": "user", "content": p}, {"role": "assistant", "content": r}]
def render_full(p, r): return tok.apply_chat_template(_msgs(p, r), tokenize=False, add_generation_prompt=False)
def render_prompt(p):  return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)

ds = load_dataset(E("RB_DATASET", "allenai/reward-bench"), split=E("RB_SPLIT", "filtered"))
by_sub = {}
for ex in ds:
    p, c, r = ex.get("prompt"), ex.get("chosen"), ex.get("rejected")
    if not (p and c and r) or c == r: continue
    by_sub.setdefault(ex.get("subset", "?"), []).append(dict(prompt=p, chosen=c, rejected=r))
rng = np.random.RandomState(SEED)
test, subs = [], []
for sub in sorted(by_sub):
    xs = by_sub[sub]
    idx = rng.permutation(len(xs))[:N_PER]
    for i in idx:
        test.append(xs[int(i)]); subs.append(sub)
subs = np.array(subs)
print(f"[data] {len(test)} pairs over {len(by_sub)} subsets (cap {N_PER}/subset)", flush=True)

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
policy = get_peft_model(model, cfg).eval()

texts, ns = [], []
for x in test:
    fp = render_prompt(x["prompt"])
    plen = len(tok(fp, truncation=True, max_length=MAX_LEN).input_ids)
    for side in ("chosen", "rejected"):
        full = render_full(x["prompt"], x[side])
        fl = len(tok(full, truncation=True, max_length=MAX_LEN + MAX_NEW).input_ids)
        texts.append(full); ns.append(max(fl - min(plen, fl - 1), 1))

@torch.no_grad()
def all_lps():
    out = []
    for s in range(0, len(texts), 8):
        chunk, nc = texts[s:s + 8], ns[s:s + 8]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=MAX_LEN + MAX_NEW).to(DEV)
        lp = _comp_logp(policy(**enc, logits_to_keep=max(nc) + 1).logits, enc.input_ids, nc)
        out.extend(lp.float().cpu().tolist())
    return np.array(out).reshape(-1, 2)

def report(name, marg):
    acc = float((marg > 0).mean())
    res = dict(acc=acc, se=float(np.sqrt(acc * (1 - acc) / len(marg))), n=len(marg),
               sections={}, subsets={})
    for sec in list(SECTIONS) + ["other"]:
        m = np.array([SEC_OF.get(s, "other") == sec for s in subs])
        if m.sum(): res["sections"][sec] = dict(acc=float((marg[m] > 0).mean()), n=int(m.sum()))
    for sub in sorted(set(subs.tolist())):
        m = subs == sub
        res["subsets"][sub] = dict(acc=float((marg[m] > 0).mean()), n=int(m.sum()))
    secs = " | ".join(f"{k} {v['acc']:.3f}(n={v['n']})" for k, v in res["sections"].items())
    print(f"[{name}] overall {acc:.3f} ± {res['se']:.3f} | {secs}", flush=True)
    return res

results = {}
with policy.disable_adapter():
    ref = all_lps()
results["base_raw"] = report("base_raw (lp_c>lp_r, no ref-correction)", ref[:, 0] - ref[:, 1])

_ck = E("CKPTS", "")
assert _ck, "set CKPTS=name=path,name=path"
for name, path in [kv.split("=", 1) for kv in _ck.split(",") if kv]:
    if not os.path.isdir(path): print(f"[skip] {path}", flush=True); continue
    sd = load_file(os.path.join(path, "adapter_model.safetensors"))
    set_peft_model_state_dict(policy, sd)
    lp = all_lps()
    marg = (lp[:, 0] - ref[:, 0]) - (lp[:, 1] - ref[:, 1])
    results[name] = report(name, marg)
    results[name].update(dlp_chosen=float((lp[:, 0] - ref[:, 0]).mean()),
                         dlp_rejected=float((lp[:, 1] - ref[:, 1]).mean()))

json.dump(results, open(E("OUT", "/workspace/uf_rewardbench.json"), "w"), indent=1)
print("DONE", flush=True)
