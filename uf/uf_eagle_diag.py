#!/usr/bin/env python
"""Why is UF frozen-head stage-1 flat? Rank the three suspects with NO training.

On held-out UF pairs, under the FROZEN BASE model, measure chosen>rejected ranking accuracy
(sum completion logp) through three readouts:
  head_acc_raw   the distilled tf head at L12 (the stage-1 loss's actual eyes)
  lens_acc_raw   a zero-init tf head = logit-lens at L12 (no distillation at all)
  final_acc_raw  the full model's own final logits (the ceiling for logp-margin methods)
Also: mean margin (lp_chosen - lp_rejected) and its std for each, and length-normalized variants
(long completions dominate sum-logp; per-token margins remove the length confound).

Interpretation:
  head ~.5, final >>.5  -> the head is the bottleneck (distill harder / different readout)
  head ~= final ~.55-.6 -> logp margins barely see the dataset preference at all ->
                           phase-9 headroom: dataset pairs carry little installable signal
                           at this policy's level; use on-policy judged pairs instead
  head >=.6             -> signal visible; stage-1 flatness is an optimization problem (LR/steps)

Env: L=12 N=128 MAX_LEN=1024. Reads /workspace/uf_eagle_head_tf_L{L}.pt.
Writes /workspace/uf_eagle_diag_L{L}.json"""
import os, sys, json, random, hashlib
from itertools import islice
import numpy as np
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eagle"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eagle_common import make_head, comp_slices, gather_logps
from helpers import ResidualCapture

E = os.environ.get
MODEL   = E("UF_SFT_MODEL", "allenai/Llama-3.1-Tulu-3-8B-SFT")
DATASET = E("UF_DATASET", "allenai/ultrafeedback_binarized_cleaned")
SPLIT   = E("UF_SPLIT", "train_prefs")
L, N, MAX_LEN = int(E("L", 12)), int(E("N", 128)), int(E("MAX_LEN", 1024))
POOL, MARGIN_MIN = int(E("UF_POOL", 20000)), float(E("UF_MARGIN_MIN", 1.0))
DEV = "cuda"
torch.manual_seed(0)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"
def _msgs(p, r): return [{"role": "user", "content": p}, {"role": "assistant", "content": r}]
def render_full(p, r): return tok.apply_chat_template(_msgs(p, r), tokenize=False, add_generation_prompt=False)
def render_prompt(p):  return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
def _phash(s): return hashlib.sha1(s.encode("utf-8")).hexdigest()

# same funnel as uf_eagle_s1.py; keep only held-out (is_test) rows
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
print(f"[diag] {len(test)} held-out pairs", flush=True)

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
BLOCKS = list(model.model.layers); HID = model.config.hidden_size

head = make_head(HID, "tf").to(DEV)
head.load_state_dict(torch.load(f"/workspace/uf_eagle_head_tf_L{L}.pt", map_location=DEV))
lens = make_head(HID, "tf").to(DEV)   # zero-init output projections => pure logit-lens exit
for h in (head, lens):
    for p in h.parameters(): p.requires_grad_(False)

@torch.no_grad()
def margins(batch):
    texts, plens, ntoks = [], [], []
    for x in batch:
        pl = len(tok(render_prompt(x["prompt"]), truncation=True, max_length=MAX_LEN).input_ids)
        texts += [render_full(x["prompt"], x["chosen"]), render_full(x["prompt"], x["rejected"])]
        plens += [pl, pl]
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(DEV)
    spans = comp_slices(tok, texts, plens, enc)
    ntoks = [max(hi - lo, 1) for lo, hi in spans]
    with ResidualCapture([BLOCKS[L]]) as cap:
        logits = model(**enc).logits
    hL = cap.get()[0][:, :-1]
    out = {}
    for name, lg in (("final", logits[:, :-1].float()),
                     ("head", head(hL, model, pad_mask=enc.attention_mask[:, :-1])),
                     ("lens", lens(hL, model, pad_mask=enc.attention_mask[:, :-1]))):
        lp = gather_logps(F.log_softmax(lg, -1), enc, spans)
        lpt = lp / torch.tensor(ntoks, device=DEV, dtype=lp.dtype)
        out[name] = (lp[0::2] - lp[1::2]).cpu().tolist()
        out[name + "_pt"] = (lpt[0::2] - lpt[1::2]).cpu().tolist()
    return out

acc = {k: [] for k in ("final", "head", "lens", "final_pt", "head_pt", "lens_pt")}
mar = {k: [] for k in acc}
for s in range(0, len(test), 4):
    for k, v in margins(test[s:s + 4]).items():
        mar[k] += v; acc[k] += [float(x > 0) for x in v]
    if (s // 4) % 8 == 0: print(f"  {s+4}/{len(test)}", flush=True)

res = {}
for k in acc:
    res[k] = dict(acc=float(np.mean(acc[k])), margin_mean=float(np.mean(mar[k])),
                  margin_std=float(np.std(mar[k])))
res["n"] = len(test); res["L"] = L
json.dump(res, open(f"/workspace/uf_eagle_diag_L{L}.json", "w"), indent=1)
for k in ("head", "lens", "final", "head_pt", "lens_pt", "final_pt"):
    print(f"{k:9s} acc {res[k]['acc']:.3f} | margin {res[k]['margin_mean']:+.2f} ± {res[k]['margin_std']:.2f}", flush=True)
print("DONE", flush=True)
