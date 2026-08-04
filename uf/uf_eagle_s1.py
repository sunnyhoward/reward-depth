#!/usr/bin/env python
"""Frozen-head EAGLE stage-1 on REAL preference data: UltraFeedback pairs, Tulu-3-8B-SFT.

The 2026-08-04 finding (eagle/RESULTS.md §8-§10): with the early-exit head FROZEN, DPO through
the head with LoRA restricted to layers 0..L installs the preference behaviourally at a fraction
of full-DPO's KL — stage 1 ALONE is the method (no stage 2). This script ports that method from
the toy testbeds to UF. Reference points from this repo: UF preference is linearly readable at
L11-12 at ~.79 (phase 5/9), which equals what 400-step full DPO installs (uf_dpo_train.py).

Head: EagleTfHead (attention readout) at layer L, lazily distilled to the BASE model's own final
logits on natural UF text (prompt + chosen/rejected), then frozen. Path
/workspace/uf_eagle_head_tf_L{L}.pt (delete to re-distill).

Metrics per eval — the standing traps apply (implicit metrics dissociate from behaviour; read
the raw generations):
  head_acc      through-head implicit DPO acc on held-out pairs (stage-1 plateau meter)
  acc_implicit  full-model implicit DPO acc on held-out pairs (does the edit reach the output?)
  kl_from_base  mean per-token KL(policy||base), held-out chosen completions
  gens          greedy continuations of held-out prompts, saved to OUT/gens_step{N}.json for
                judge win-rate vs the step-0 (= base) gens. NOT scored here.

Env: L=12 STEPS=150 BATCH=4 ACCUM=4 LR=1e-4 BETA=0.1 EVAL_EVERY=10 CKPT_EVERY=10
     GEN_N=48 GEN_TOKENS=256 MAX_LEN=1024 HD_STEPS=600 SEED=0 RUN_TAG=auto
     UF_SFT_MODEL/UF_DATASET/UF_SPLIT/UF_POOL/UF_MARGIN_MIN as in uf_dpo_train.py
Outputs: /workspace/uf_eagle_{TAG}/ (history.json, gens_step*.json, ckpt*/)"""
import os, sys, json, random, hashlib
from itertools import islice
import numpy as np
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eagle"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eagle_common import make_head, comp_slices, gather_logps
from helpers import ResidualCapture

E = os.environ.get
MODEL   = E("UF_SFT_MODEL", "allenai/Llama-3.1-Tulu-3-8B-SFT")
DATASET = E("UF_DATASET", "allenai/ultrafeedback_binarized_cleaned")
SPLIT   = E("UF_SPLIT", "train_prefs")
POOL    = int(E("UF_POOL", 20000)); MARGIN_MIN = float(E("UF_MARGIN_MIN", 1.0))
N_TRAIN = int(E("UF_N_TRAIN", 12000)); N_EVAL = int(E("UF_N_EVAL", 64))
L       = int(E("L", 12))
STEPS, BATCH, ACCUM = int(E("STEPS", 150)), int(E("BATCH", 4)), int(E("ACCUM", 4))
LR, BETA = float(E("LR", 1e-4)), float(E("BETA", 0.1))
EVAL_EVERY, CKPT_EVERY = int(E("EVAL_EVERY", 10)), int(E("CKPT_EVERY", 10))
GEN_N, GEN_TOKENS = int(E("GEN_N", 48)), int(E("GEN_TOKENS", 256))
MAX_LEN = int(E("MAX_LEN", 1024)); HD_STEPS = int(E("HD_STEPS", 600))
SEED = int(E("SEED", 0)); DEV = "cuda"
TAG = E("RUN_TAG", f"s1_L{L}_seed{SEED}")
OUT = f"/workspace/uf_eagle_{TAG}"
os.makedirs(OUT, exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"   # comp_slices assumes LEFT padding

def _msgs(p, r): return [{"role": "user", "content": p}, {"role": "assistant", "content": r}]
def render_full(p, r): return tok.apply_chat_template(_msgs(p, r), tokenize=False, add_generation_prompt=False)
def render_prompt(p):  return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
def _phash(s): return hashlib.sha1(s.encode("utf-8")).hexdigest()

# ---- data: same funnel as uf_dpo_train.py (stream, margin filter, by-prompt split) ----
print(f"[data] streaming {POOL} rows of {DATASET}:{SPLIT}", flush=True)
ds = load_dataset(DATASET, split=SPLIT, streaming=True)
recs = []
for ex in islice(ds, POOL):
    ch, rj = ex.get("chosen"), ex.get("rejected")
    if not ch or not rj: continue
    prompt = ex.get("prompt") or (ch[0]["content"] if isinstance(ch[0], dict) else "")
    c = ch[-1]["content"] if isinstance(ch[-1], dict) else str(ch)
    r = rj[-1]["content"] if isinstance(rj[-1], dict) else str(rj)
    if not (prompt and c and r) or c == r: continue
    sc, sr = ex.get("score_chosen"), ex.get("score_rejected")
    if sc is None or sr is None or float(sc) - float(sr) < MARGIN_MIN: continue
    recs.append(dict(prompt=prompt, chosen=c, rejected=r,
                     is_test=int(_phash(prompt)[:8], 16) % 10 == 0))
train = [r for r in recs if not r["is_test"]][:N_TRAIN]
test  = [r for r in recs if r["is_test"]]
print(f"[data] margin-filtered {len(recs)} | train {len(train)} | eval pairs {min(len(test), N_EVAL)} "
      f"| gen prompts {min(len(test), GEN_N)}", flush=True)

# ---- model + lower-stack LoRA ----
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
NL = len(model.model.layers); HID = model.config.hidden_size
assert 0 <= L < NL - 1, f"L={L} out of range for {NL}-block model"
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                 layers_to_transform=list(range(0, L + 1)))
policy = get_peft_model(model, cfg); policy.config.use_cache = False
BLOCKS = list(model.model.layers)   # post-wrap refs for capture

# the asserts eagle_dpo.py demands: trainable params ONLY LoRA in layers 0..L
for n_, p in policy.named_parameters():
    if p.requires_grad:
        assert "lora" in n_, f"non-LoRA trainable param: {n_}"
        assert int(n_.split(".layers.")[1].split(".")[0]) <= L, f"trainable LoRA above L: {n_}"
params = [p for p in policy.parameters() if p.requires_grad]

# ---- frozen EAGLE head at L (lazy distill on natural UF text to the base's final logits) ----
HEADF = f"/workspace/uf_eagle_head_tf_L{L}.pt"
head = make_head(HID, "tf").to(DEV)
if not os.path.exists(HEADF):
    print(f"[head] distilling tf head at L{L} on UF text ({HD_STEPS} steps)...", flush=True)
    ho = torch.optim.AdamW(head.parameters(), lr=1e-3)
    rg = random.Random(SEED + 7)
    for st in range(HD_STEPS):
        rs = rg.sample(train, 4)
        texts = [render_full(x["prompt"], x[rg.choice(["chosen", "rejected"])]) for x in rs]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(DEV)
        am_ = enc.attention_mask[:, 1:].bool()
        with torch.no_grad(), ResidualCapture([BLOCKS[L]]) as cap:
            t_lsm = F.log_softmax(policy(**enc).logits[:, :-1].float(), -1)
        s_lsm = F.log_softmax(head(cap.get()[0][:, :-1], model, pad_mask=enc.attention_mask[:, :-1]), -1)
        kl = ((t_lsm.exp() * (t_lsm - s_lsm)).sum(-1) * am_).sum() / am_.sum()
        ho.zero_grad(); kl.backward(); ho.step()
        if (st + 1) % 100 == 0: print(f"  head step {st+1}: kl {float(kl):.3f}", flush=True)
    torch.save(head.state_dict(), HEADF)
    print(f"[head] saved {HEADF}", flush=True)
else:
    head.load_state_dict(torch.load(HEADF, map_location=DEV))
    print(f"[head] loaded {HEADF}", flush=True)
for p in head.parameters(): p.requires_grad_(False)   # FROZEN — the whole point
opt = torch.optim.AdamW(params, lr=LR)
print(f"[{TAG}] trainable {sum(p.numel() for p in params)/1e6:.1f}M (LoRA layers 0..{L}) | "
      f"head {sum(p.numel() for p in head.parameters())/1e6:.1f}M frozen | {NL} blocks", flush=True)

def pair_logps(items, grad):
    """Both sides' completion logp through the L-head, batched, left-padded."""
    texts, plens = [], []
    for x in items:
        pl = len(tok(render_prompt(x["prompt"]), truncation=True, max_length=MAX_LEN).input_ids)
        texts += [render_full(x["prompt"], x["chosen"]), render_full(x["prompt"], x["rejected"])]
        plens += [pl, pl]
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(DEV)
    spans = comp_slices(tok, texts, plens, enc)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        with ResidualCapture([BLOCKS[L]]) as cap:
            policy(**enc)
        lsm = F.log_softmax(head(cap.get()[0][:, :-1], model, pad_mask=enc.attention_mask[:, :-1]), -1)
        lp = gather_logps(lsm, enc, spans)
    return lp[0::2], lp[1::2]

def pair_logps_final(items):
    """Both sides' completion logp at the FULL model output (for acc_implicit)."""
    texts, plens = [], []
    for x in items:
        pl = len(tok(render_prompt(x["prompt"]), truncation=True, max_length=MAX_LEN).input_ids)
        texts += [render_full(x["prompt"], x["chosen"]), render_full(x["prompt"], x["rejected"])]
        plens += [pl, pl]
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(DEV)
    spans = comp_slices(tok, texts, plens, enc)
    with torch.no_grad():
        lsm = F.log_softmax(policy(**enc).logits[:, :-1].float(), -1)
        lp = gather_logps(lsm, enc, spans)
    return lp[0::2], lp[1::2]

# fixed held-out banks
eval_pairs = test[:N_EVAL]
gen_prompts = [x["prompt"] for x in test[:GEN_N]]
kl_texts = [render_full(x["prompt"], x["chosen"]) for x in test[:24]]

@torch.no_grad()
def full_eval(step):
    policy.eval()
    ev = dict(step=step)
    # head_acc: through-head implicit acc (policy vs adapter-disabled ref, same frozen head)
    accs, accs_out = [], []
    for s in range(0, len(eval_pairs), 4):
        sub = eval_pairs[s:s + 4]
        la, lb = pair_logps(sub, False)
        with policy.disable_adapter():
            ra, rb = pair_logps(sub, False)
        accs += ((la - ra) > (lb - rb)).float().cpu().tolist()
        lc, lr_ = pair_logps_final(sub)
        with policy.disable_adapter():
            rc, rr = pair_logps_final(sub)
        accs_out += ((lc - rc) > (lr_ - rr)).float().cpu().tolist()
    ev["head_acc"] = float(np.mean(accs)); ev["acc_implicit"] = float(np.mean(accs_out))
    # KL from base on held-out chosen completions
    kls = []
    for s in range(0, len(kl_texts), 4):
        enc = tok(kl_texts[s:s + 4], return_tensors="pt", padding=True, truncation=True,
                  max_length=512).to(DEV)
        am_ = enc.attention_mask[:, 1:].bool()
        p_lsm = F.log_softmax(policy(**enc).logits[:, :-1].float(), -1)
        with policy.disable_adapter():
            b_lsm = F.log_softmax(policy(**enc).logits[:, :-1].float(), -1)
        kls.append(float(((p_lsm.exp() * (p_lsm - b_lsm)).sum(-1) * am_).sum() / am_.sum()))
    ev["kl_from_base"] = float(np.mean(kls))
    # greedy gens on held-out prompts -> saved for judge win-rate vs step-0 (= base) gens
    gens, lens = [], []
    for s in range(0, len(gen_prompts), 8):
        enc = tok([render_prompt(p) for p in gen_prompts[s:s + 8]], return_tensors="pt",
                  padding=True, truncation=True, max_length=MAX_LEN).to(DEV)
        g = policy.generate(**enc, do_sample=False, max_new_tokens=GEN_TOKENS,
                            pad_token_id=tok.pad_token_id)
        for j in range(g.shape[0]):
            txt = tok.decode(g[j, enc.input_ids.shape[1]:], skip_special_tokens=True)
            gens.append(txt); lens.append(len(txt.split()))
    ev["gen_len_words"] = float(np.mean(lens))
    json.dump([dict(step=step, prompt=p, gen=g) for p, g in zip(gen_prompts, gens)],
              open(f"{OUT}/gens_step{step}.json", "w"), indent=1)
    ev["gen_samples"] = gens[:3]
    policy.train()
    return ev

hist = dict(tag=TAG, model=MODEL, L=L, beta=BETA, lr=LR, batch=BATCH * ACCUM, seed=SEED,
            head=HEADF, loss=[], evals=[])
ev0 = full_eval(0); hist["evals"].append(ev0)
print(f"  step    0: { {k: (round(v,3) if isinstance(v,float) else v) for k,v in ev0.items() if k!='gen_samples'} }", flush=True)
print(f"  base sample: {ev0['gen_samples'][0][:200]!r}", flush=True)

policy.train()
rng = random.Random(SEED + 4242)
first_bw = True
for step in range(STEPS):
    opt.zero_grad()
    tot = 0.0
    for _ in range(ACCUM):
        batch = rng.sample(train, BATCH)
        la, lb = pair_logps(batch, True)
        with torch.no_grad(), policy.disable_adapter():
            ra, rb = pair_logps(batch, False)
        loss = -F.logsigmoid(BETA * ((la - ra) - (lb - rb))).mean() / ACCUM
        loss.backward()
        tot += float(loss.detach())
    if first_bw:
        for n_, p in policy.named_parameters():
            if p.grad is not None and not p.requires_grad:
                raise AssertionError(f"grad on frozen param {n_}")
        first_bw = False
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    hist["loss"].append(tot)
    if (step + 1) % 5 == 0:
        print(f"  step {step+1:4d}: loss {np.mean(hist['loss'][-5:]):.4f}", flush=True)
    if (step + 1) % EVAL_EVERY == 0:
        ev = full_eval(step + 1); hist["evals"].append(ev)
        print(f"  step {step+1:4d}: { {k: (round(v,3) if isinstance(v,float) else v) for k,v in ev.items() if k!='gen_samples'} }", flush=True)
        print(f"  sample: {ev['gen_samples'][0][:200]!r}", flush=True)
        json.dump(hist, open(f"{OUT}/history.json", "w"), indent=1)
    if (step + 1) % CKPT_EVERY == 0:
        policy.save_pretrained(f"{OUT}/ckpt{step+1}")
json.dump(hist, open(f"{OUT}/history.json", "w"), indent=1)
print("DONE", flush=True)
