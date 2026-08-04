#!/usr/bin/env python
"""Where does lower12's margin live? (RESULTS §18 follow-up)

The lower12 arm (LoRA layers 0..12, DPO loss at FINAL logits) fits the UF margin (.74).
Question: did that training move the preference INTO the L12 representation, or does the
margin only exist after the frozen upper stack elaborates the edit?

For each readout R in {frozen tf head @L12, logit-lens @L12, final logits}, compute on held-out
pairs the DPO-style implicit accuracy of the ADAPTED model vs its adapter-disabled reference:
  acc_implicit_R = mean[ (la-ra) > (lb-rb) ]  with l=adapter on, r=adapter off, through R.
Final-logits row should reproduce ~.74 (sanity). The two L12 rows answer the question:
  L12 rows ~.74 -> the edit made layer 12 itself preference-separating (EAGLE-style encoding)
  L12 rows ~.5  -> the margin is manufactured in the (lower edit x upper stack) interaction;
                   nothing is encoded AT L12.

Env: ADAPTER=/workspace/uf_dpo_tulu8b_lower12_lora L=12 N=128 MAX_LEN=1024
Writes /workspace/uf_lower12_readcheck.json"""
import os, sys, json, hashlib
from itertools import islice
import numpy as np
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import PeftModel

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eagle"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eagle_common import make_head, comp_slices, gather_logps
from helpers import ResidualCapture

E = os.environ.get
MODEL   = E("UF_SFT_MODEL", "allenai/Llama-3.1-Tulu-3-8B-SFT")
DATASET = E("UF_DATASET", "allenai/ultrafeedback_binarized_cleaned")
SPLIT   = E("UF_SPLIT", "train_prefs")
ADAPTER = E("ADAPTER", "/workspace/uf_dpo_tulu8b_lower12_lora")
L, N, MAX_LEN = int(E("L", 12)), int(E("N", 128)), int(E("MAX_LEN", 1024))
POOL, MARGIN_MIN = int(E("UF_POOL", 20000)), float(E("UF_MARGIN_MIN", 1.0))
DEV = "cuda"

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"
def _msgs(p, r): return [{"role": "user", "content": p}, {"role": "assistant", "content": r}]
def render_full(p, r): return tok.apply_chat_template(_msgs(p, r), tokenize=False, add_generation_prompt=False)
def render_prompt(p):  return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
def _phash(s): return hashlib.sha1(s.encode("utf-8")).hexdigest()

ds = load_dataset(DATASET, split=SPLIT, streaming=True)
test = []
for ex in islice(ds, POOL):
    ch, rj = ex.get("chosen"), ex.get("rejected")
    if not ch or not rj: continue
    prompt = ex.get("prompt") or (ch[0]["content"] if isinstance(ch[0], dict) else "")
    c = ch[-1]["content"] if isinstance(ch[-1], dict) else str(ch)
    r = rj[-1]["content"] if isinstance(rj[-1], dict) else str(rj)
    if not (prompt and c and r) or c == r: continue
    sc, sr = ex.get("score_chosen"), ex.get("score_rejected")
    if sc is None or sr is None or float(sc) - float(sr) < MARGIN_MIN: continue
    if int(_phash(prompt)[:8], 16) % 10 == 0:
        test.append(dict(prompt=prompt, chosen=c, rejected=r))
    if len(test) >= N: break
print(f"[readcheck] {len(test)} held-out pairs | adapter {ADAPTER}", flush=True)

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
policy = PeftModel.from_pretrained(model, ADAPTER).eval()
BLOCKS = list(model.model.layers); HID = model.config.hidden_size

head = make_head(HID, "tf").to(DEV)
head.load_state_dict(torch.load(f"/workspace/uf_eagle_head_tf_L{L}.pt", map_location=DEV))
lens = make_head(HID, "tf").to(DEV)
for h in (head, lens):
    for p in h.parameters(): p.requires_grad_(False)

@torch.no_grad()
def logps(batch):
    """Per-readout summed completion logps for both sides, current adapter state."""
    texts, plens = [], []
    for x in batch:
        pl = len(tok(render_prompt(x["prompt"]), truncation=True, max_length=MAX_LEN).input_ids)
        texts += [render_full(x["prompt"], x["chosen"]), render_full(x["prompt"], x["rejected"])]
        plens += [pl, pl]
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(DEV)
    spans = comp_slices(tok, texts, plens, enc)
    with ResidualCapture([BLOCKS[L]]) as cap:
        logits = policy(**enc).logits
    hL = cap.get()[0][:, :-1]
    out = {}
    for name, lg in (("final", logits[:, :-1].float()),
                     ("head", head(hL, model, pad_mask=enc.attention_mask[:, :-1])),
                     ("lens", lens(hL, model, pad_mask=enc.attention_mask[:, :-1]))):
        lp = gather_logps(F.log_softmax(lg, -1), enc, spans)
        out[name] = (lp[0::2], lp[1::2])
    return out

acc = {k: [] for k in ("final", "head", "lens")}
for s in range(0, len(test), 4):
    batch = test[s:s + 4]
    pol = logps(batch)
    with policy.disable_adapter():
        ref = logps(batch)
    for k in acc:
        la, lb = pol[k]; ra, rb = ref[k]
        acc[k] += ((la - ra) > (lb - rb)).float().cpu().tolist()
    if (s // 4) % 8 == 0: print(f"  {s+4}/{len(test)}", flush=True)

res = {k: float(np.mean(v)) for k, v in acc.items()}
res["n"] = len(test); res["L"] = L; res["adapter"] = ADAPTER
json.dump(res, open("/workspace/uf_lower12_readcheck.json", "w"), indent=1)
for k in ("head", "lens", "final"):
    print(f"acc_implicit[{k}] = {res[k]:.3f}", flush=True)
print("DONE", flush=True)
