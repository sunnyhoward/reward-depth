#!/usr/bin/env python
"""UF hybrid: margin BACKPROP through the policy's own L* activations into blocks <= L*
(teacher-forced dataset pairs) + RLOO REINFORCE (frozen-base probe reward, truncation-masked,
DPOP anchor) for blocks > L*. UF port of hybrid_deep.py / methods.md §2.1.

CAVEATS (phase-2 findings, 2026-07-20): the margin half is the gameable self-read coupling —
on the A/B testbed it forged its layer (meter 0.99, no behavior) at every scale tried; the
behavioral installation comes from the RL half, which is honest by construction (frozen-base
read of emitted text). This script exists to measure whether the margin half ADDS anything
(depth-shaping, faster install) or just forges, on real preference data. Head is FROZEN
(no fresh-label oracle exists on UF; stale-label filtering was a stalemate on A/B).

Env knobs: as uf_probe_rl.py, plus MARGIN_LR (default 5e-5).
Outputs: /workspace/uf_hybrid_{history.json,ckptN,lora}."""
import os, sys, json, random, hashlib, re
from itertools import islice
import numpy as np
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import train_bayes_head, ResidualCapture

E = os.environ.get
MODEL = E("UF_SFT_MODEL", "allenai/Llama-3.1-Tulu-3-8B-SFT")
POOL, N_PROBE = int(E("UF_POOL", 20000)), int(E("N_PROBE", 3000))
STEPS, BATCH, K = int(E("RL_STEPS", 300)), int(E("RL_BATCH", 4)), int(E("RL_K", 8))
KL, PESS, ANCHOR = float(E("RL_KL", 0.03)), float(E("RL_PESS", 0.5)), float(E("RL_ANCHOR", 1.0))
LR, MLR = float(E("RL_LR", 5e-5)), float(E("MARGIN_LR", 5e-5))
MAX_NEW, MAX_LEN, PLEN = int(E("MAX_NEW", 512)), int(E("MAX_LEN", 1024)), int(E("PROMPT_LEN", 512))
TOL = float(E("PLATEAU_TOL", 0.01))
DEV = "cuda"; SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"
def _msgs(p, r): return [{"role": "user", "content": p}, {"role": "assistant", "content": r}]
def render_full(p, r): return tok.apply_chat_template(_msgs(p, r), tokenize=False, add_generation_prompt=False)
def render_prompt(p):  return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
def _phash(s): return hashlib.sha1(s.encode()).hexdigest()

ds = load_dataset(E("UF_DATASET", "allenai/ultrafeedback_binarized_cleaned"),
                  split=E("UF_SPLIT", "train_prefs"), streaming=True)
recs = []
for ex in islice(ds, POOL):
    ch, rj = ex.get("chosen"), ex.get("rejected")
    if not ch or not rj: continue
    p = ex.get("prompt") or ch[0]["content"]
    c, r = ch[-1]["content"], rj[-1]["content"]
    if not (p and c and r) or c == r: continue
    sc, sr = ex.get("score_chosen"), ex.get("score_rejected")
    if sc is None or sr is None or float(sc) - float(sr) < 1.0: continue
    recs.append(dict(prompt=p, chosen=c, rejected=r, is_test=int(_phash(p)[:8], 16) % 10 == 0))
train = [x for x in recs if not x["is_test"]]
test = [x for x in recs if x["is_test"]]
print(f"[data] train {len(train)} | test {len(test)}", flush=True)

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
BLOCKS = list(model.model.layers); NL = len(BLOCKS)

# probe sweep from the shared cache (built by uf_probe_rl.py)
z = np.load("/workspace/uf_probe_feats.npz")
Fc_tr, Fr_tr, Fc_te, Fr_te = z["a"], z["b"], z["c"], z["d"]
rng = np.random.RandomState(SEED)
s_tr = np.where(rng.rand(len(Fc_tr)) < 0.5, 1.0, -1.0).astype(np.float32)
s_te = np.where(rng.rand(len(Fc_te)) < 0.5, 1.0, -1.0).astype(np.float32)
acc = {}
heads = {}
for li in range(NL):
    sd = np.concatenate([Fc_tr[:, li], Fr_tr[:, li]]).std(0) + 1e-6
    a, h, e = train_bayes_head(((Fc_tr[:, li] - Fr_tr[:, li]) / sd) * s_tr[:, None], s_tr,
                               ((Fc_te[:, li] - Fr_te[:, li]) / sd) * s_te[:, None], s_te)
    acc[li], heads[li] = a, (h, sd)
mx = max(acc.values())
LSTAR = min(li for li in acc if acc[li] >= mx - TOL)
print(f"[probe] L*={LSTAR} (acc {acc[LSTAR]:.3f})", flush=True)
head, sd_ = heads[LSTAR]
MU = head.mu.detach().float().to(DEV); SIG2 = F.softplus(head.rho.detach()).float().pow(2).to(DEV)
SD = torch.tensor(sd_, device=DEV)
def probe_reward(f):                       # pessimistic reward for RL (detached use)
    fs = f.float() / SD
    s2 = fs.pow(2).matmul(SIG2)
    return torch.special.ndtr((fs.matmul(MU) - PESS * torch.sqrt(s2 + 1e-9)) / torch.sqrt(1 + s2))

cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
policy = get_peft_model(model, cfg); policy.config.use_cache = False
def _blk(n):
    m = re.search(r"\.layers\.(\d+)\.", n); return int(m.group(1)) if m else -1
low = [p for n, p in policy.named_parameters() if p.requires_grad and _blk(n) <= LSTAR]
up  = [p for n, p in policy.named_parameters() if p.requires_grad and _blk(n) > LSTAR]
name_blk = [(p, _blk(n)) for n, p in policy.named_parameters() if p.requires_grad]
opt_low, opt_up = torch.optim.AdamW(low, lr=MLR), torch.optim.AdamW(up, lr=LR)

def margin_step_uf(batch):
    """Self-read margin backprop at L*: rank chosen > rejected in the POLICY's own features."""
    texts = [render_full(x["prompt"], x["chosen"]) for x in batch] + \
            [render_full(x["prompt"], x["rejected"]) for x in batch]
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(DEV)
    with ResidualCapture([BLOCKS[LSTAR]]) as cap:
        policy(**enc, logits_to_keep=1)
    f = cap.get()[0][:, -1]
    B = len(batch)
    fs = ((f[:B] - f[B:]).float() / SD)
    s2 = fs.pow(2).matmul(SIG2)
    zz = fs.matmul(MU) / torch.sqrt(1 + s2)
    (-torch.special.log_ndtr(zz).mean()).backward()

def comp_logprob(text_full, plen, grad):
    ids = tok(text_full, return_tensors="pt", truncation=True, max_length=MAX_LEN + MAX_NEW).input_ids.to(DEV)
    plen = min(plen, ids.shape[1] - 1)
    with (torch.enable_grad() if grad else torch.no_grad()):
        keep = ids.shape[1] - plen + 1
        logits = policy(ids, logits_to_keep=keep).logits[0, :-1].float()
        return F.log_softmax(logits, -1).gather(-1, ids[0, plen:, None]).squeeze(-1).sum()

@torch.no_grad()
def evaluate(n=64):
    policy.eval(); ir = []
    for x in test[:n]:
        pl = tok(render_prompt(x["prompt"]), return_tensors="pt", truncation=True, max_length=MAX_LEN).input_ids.shape[1]
        lc = comp_logprob(render_full(x["prompt"], x["chosen"]), pl, False)
        lr_ = comp_logprob(render_full(x["prompt"], x["rejected"]), pl, False)
        with policy.disable_adapter():
            rc = comp_logprob(render_full(x["prompt"], x["chosen"]), pl, False)
            rr = comp_logprob(render_full(x["prompt"], x["rejected"]), pl, False)
        ir.append(dict(acc=float((lc - rc) > (lr_ - rr)), dc=float(lc - rc), dr=float(lr_ - rr)))
    policy.train()
    return dict(acc_implicit=float(np.mean([x["acc"] for x in ir])),
                dlp_chosen=float(np.mean([x["dc"] for x in ir])),
                dlp_rejected=float(np.mean([x["dr"] for x in ir])))

hist = dict(Lstar=LSTAR, reward=[], evals=[], len=[], trunc=[])
rgen = random.Random(4242); policy.train()
for step in range(STEPS):
    batch = rgen.sample(train, BATCH)
    # --- margin half -> blocks <= L* ---
    opt_low.zero_grad(); opt_up.zero_grad()
    margin_step_uf(batch)
    torch.nn.utils.clip_grad_norm_(low, 1.0); opt_low.step()
    # --- RL half -> blocks > L* ---
    opt_up.zero_grad()
    prompts = [render_prompt(x["prompt"]) for x in batch]
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=PLEN).to(DEV)
    policy.config.use_cache = True
    with torch.no_grad():
        gen = policy.generate(**enc, do_sample=True, temperature=1.0, num_return_sequences=K,
                              max_new_tokens=MAX_NEW, pad_token_id=tok.pad_token_id)
    policy.config.use_cache = False
    P = enc.input_ids.shape[1]
    attn = (gen != tok.pad_token_id).long()
    with torch.no_grad(), policy.disable_adapter(), ResidualCapture([BLOCKS[LSTAR]]) as cap:
        policy(input_ids=gen, attention_mask=attn)
    r = probe_reward(cap.get()[0][:, -1]).detach()
    n_new = (attn[:, P:]).sum(1).clamp(min=1)
    keepg = gen.shape[1] - P + 1
    tokmask = attn[:, P:].bool()
    with torch.no_grad():
        lsm = F.log_softmax(policy(input_ids=gen, attention_mask=attn, logits_to_keep=keepg).logits[:, :-1].float(), -1)
        logp_ng = (lsm.gather(-1, gen[:, P:, None]).squeeze(-1) * tokmask).sum(1)
        with policy.disable_adapter():
            ref_lsm = F.log_softmax(policy(input_ids=gen, attention_mask=attn, logits_to_keep=keepg).logits[:, :-1].float(), -1)
            ref_logp = (ref_lsm.gather(-1, gen[:, P:, None]).squeeze(-1) * tokmask).sum(1)
        del lsm, ref_lsm
    r = r - KL * (logp_ng - ref_logp) / n_new
    valid = (n_new < MAX_NEW)
    rg, vg = r.view(BATCH, K), valid.view(BATCH, K).float()
    cnt = vg.sum(1, keepdim=True)
    loo = (rg * vg).sum(1, keepdim=True) - rg * vg
    base = loo / (cnt - vg).clamp(min=1)
    adv = torch.where((vg > 0) & (cnt > 1.5), rg - base, torch.zeros_like(rg)).view(-1)
    for s0 in range(0, BATCH * K, 4):
        sl = slice(s0, min(s0 + 4, BATCH * K))
        if not adv[sl].abs().sum() > 0: continue
        li = F.log_softmax(policy(input_ids=gen[sl], attention_mask=attn[sl],
                                  logits_to_keep=keepg).logits[:, :-1].float(), -1)
        lp_i = (li.gather(-1, gen[sl, P:, None]).squeeze(-1) * tokmask[sl]).sum(1)
        (-(adv[sl] * lp_i / n_new[sl]).sum() / (BATCH * K)).backward()
    if ANCHOR > 0:
        for x in batch:
            pl = tok(render_prompt(x["prompt"]), return_tensors="pt", truncation=True, max_length=MAX_LEN).input_ids.shape[1]
            lc = comp_logprob(render_full(x["prompt"], x["chosen"]), pl, True)
            with torch.no_grad(), policy.disable_adapter():
                rc = comp_logprob(render_full(x["prompt"], x["chosen"]), pl, False)
            (ANCHOR * F.relu(rc - lc) / BATCH).backward()
    for p_, b_ in name_blk:                       # RL half must not touch blocks <= L*
        if b_ <= LSTAR: p_.grad = None
    torch.nn.utils.clip_grad_norm_(up, 1.0)
    opt_up.step()
    hist["reward"].append(float(rg.mean())); hist["len"].append(float(n_new.float().mean()))
    hist["trunc"].append(float(1 - valid.float().mean()))
    if (step + 1) % 10 == 0:
        print(f"  step {step+1:4d}: reward {np.mean(hist['reward'][-10:]):.3f} "
              f"len {np.mean(hist['len'][-10:]):.0f} trunc {np.mean(hist['trunc'][-10:]):.2f}", flush=True)
    if (step + 1) % 50 == 0:
        ev = evaluate(); ev["step"] = step + 1; hist["evals"].append(ev)
        print(f"  step {step+1:4d}: EVAL {ev}", flush=True)
        json.dump(hist, open("/workspace/uf_hybrid_history.json", "w"), indent=1)
    if (step + 1) % 100 == 0:
        policy.save_pretrained(f"/workspace/uf_hybrid_ckpt{step+1}")
json.dump(hist, open("/workspace/uf_hybrid_history.json", "w"), indent=1)
policy.save_pretrained("/workspace/uf_hybrid_lora"); tok.save_pretrained("/workspace/uf_hybrid_lora")
print("DONE", flush=True)
