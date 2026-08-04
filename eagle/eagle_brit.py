#!/usr/bin/env python
"""EAGLE two-stage on the joint-preference-sets British axis (user data, 2026-08-03). Extends
the depth ladder past styc: brit_lang (spelling/lexicon — near tokenizer-level, predicted
installable very low) and brit_culture (cultural reference — semantic, predicted mid). No FLIP
needed: the base is American-default, so free-sampling British rate starts ~0 with headroom.

Free-sampling oracle comes from the DATA: pair_id axes ("language:sweater|jumper") give an
American|British marker lexicon; brit_rate = British-side hits / all hits over continuations of
held-out prompts. Goodhart guard: implicit acc on the truth_over_british adversarial pairs
(prefer true-American over false-British) — installing Britishness must not buy false-British.

STAGE=s1 (DPO through the EAGLE head, lower-only LoRA; lazily distills a brit-domain head at L
if /workspace/eagle_head_brit_L{L}.pt is missing) | s2 (delta-distillation upward from S1_CKPT)
| final-loss baselines via LOSS_AT=final WRITE=all|upper.

Env: STAGE=s1 FACTOR=lang|culture L=12 STEPS=200 BATCH=16 LR=1e-4 BETA=0.1 ALPHA=4.0 KL_W=1.0
     LOSS_AT=eagle WRITE=lower S1_CKPT= EVAL_EVERY=25 CKPT_EVERY=25 SEED=0 RUN_TAG=auto
Outputs: /workspace/eagle_brit_{TAG}/"""
import os, sys, json, random, re
import numpy as np
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eagle_common import make_head, head_path as _hp, EagleHead, comp_slices, gather_logps, MODEL, DEV
from helpers import ResidualCapture

E = os.environ.get
STAGE, FACTOR, L = E("STAGE", "s1"), E("FACTOR", "lang"), int(E("L", 12))
LOSS_AT, WRITE = E("LOSS_AT", "eagle"), E("WRITE", "lower")
STEPS, BATCH = int(E("STEPS", 200)), int(E("BATCH", 16))
LR, BETA = float(E("LR", 1e-4)), float(E("BETA", 0.1))
ALPHA, KL_W = float(E("ALPHA", 4.0)), float(E("KL_W", 1.0))
EVAL_EVERY, CKPT_EVERY, SEED = int(E("EVAL_EVERY", 25)), int(E("CKPT_EVERY", 25)), int(E("SEED", 0))
S1 = E("S1_CKPT", "")
# delta: init + ALPHA*(after-before)  |  head: the head's own output is the target
# ("the aligned student becomes the teacher"). See eagle_stage2.py for the measured rationale.
S2_TEACHER = E("S2_TEACHER", "delta"); assert S2_TEACHER in ("delta", "head")
COMP = {"lang": "language", "culture": "culture"}[FACTOR]
TAG = E("RUN_TAG", f"{STAGE}_{FACTOR}_L{L}" if LOSS_AT == "eagle" else
        (f"fulldpo_{FACTOR}" if WRITE == "all" else f"upperonly_{FACTOR}_L{L}"))
OUT = f"/workspace/eagle_brit_{TAG}"
os.makedirs(OUT, exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DSD = "/workspace/reward-depth/joint-preference-sets/release-v1/british_campaign"
tr_rows = [json.loads(l) for l in open(f"{DSD}/train.jsonl")]
va_rows = [json.loads(l) for l in open(f"{DSD}/validation.jsonl")]
train = [r for r in tr_rows if r["component"] == COMP]
val = [r for r in va_rows if r["component"] == COMP]
truth_val = [r for r in va_rows if r["component"] == "truth_over_british"]
print(f"[data] {FACTOR}: {len(train)} train pairs | {len(val)} val | {len(truth_val)} truth-guard", flush=True)

# marker lexicon from pair_id axes of THIS component (am|br word pairs)
AM, BR = set(), set()
for r in tr_rows + va_rows:
    if r["component"] == COMP and ":" in r["pair_id"] and "|" in r["pair_id"]:
        ax = r["pair_id"].split(":", 1)[1]
        if "|" in ax and ":" not in ax:
            am, br = ax.split("|", 1)
            if am and br and " " not in am and " " not in br:
                AM.add(am.lower()); BR.add(br.lower())
print(f"[lexicon] {len(AM)} american / {len(BR)} british markers", flush=True)
AM_RE = re.compile(r"\b(" + "|".join(map(re.escape, sorted(AM))) + r")\b") if AM else None
BR_RE = re.compile(r"\b(" + "|".join(map(re.escape, sorted(BR))) + r")\b") if BR else None

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"

# ---- model per stage ----
if STAGE == "s2":
    assert S1 and os.path.isdir(S1)
    base_a = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
    merged = PeftModel.from_pretrained(base_a, S1).merge_and_unload()
    model = merged
    base_b = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
    for p in base_b.parameters(): p.requires_grad_(False)
    BLOCKS_B = list(base_b.model.layers)
else:
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
    base_b = None
NL = len(model.model.layers); HID = model.config.hidden_size
lm = dict(lower=list(range(0, L + 1)), upper=list(range(L + 1, NL)), all=None)
wr = "upper" if STAGE == "s2" else WRITE
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                 layers_to_transform=lm[wr])
policy = get_peft_model(model, cfg); policy.config.use_cache = False
BLOCKS = list(model.model.layers)
allowed = set(lm[wr] if lm[wr] is not None else range(NL))
for n_, p in policy.named_parameters():
    if p.requires_grad:
        assert "lora" in n_ and int(n_.split(".layers.")[1].split(".")[0]) in allowed, n_
params = [p for p in policy.parameters() if p.requires_grad]

def txt(r, side): return r["prompt"] + r[side]
def plen(r): return len(tok(r["prompt"]).input_ids)

# ---- brit-domain EAGLE head (lazy pretrain: distill to base on this text distribution) ----
head = head_ref = head_before = None
HEAD_ARCH = E("HEAD_ARCH", "mlp")
FREEZE_HEAD = int(E("FREEZE_HEAD", 1))   # see eagle_dpo.py — trainable head absorbs the install
HEADF = (f"/workspace/eagle_head_brit_L{L}.pt" if HEAD_ARCH == "mlp"
         else f"/workspace/eagle_head_brit_{HEAD_ARCH}_L{L}.pt")
if LOSS_AT == "eagle" and STAGE == "s1":
    if not os.path.exists(HEADF):
        print("[head] distilling brit-domain head...", flush=True)
        hd = make_head(HID, HEAD_ARCH).to(DEV)
        ho = torch.optim.AdamW(hd.parameters(), lr=1e-3)
        rows_all = tr_rows
        rg = random.Random(SEED + 7)
        for st in range(300):
            rs = rg.sample(rows_all, 16)
            texts = [txt(r, rg.choice(["chosen", "rejected"])) for r in rs]
            enc = tok(texts, return_tensors="pt", padding=True).to(DEV)
            am_ = enc.attention_mask[:, 1:].bool()
            with torch.no_grad(), ResidualCapture([BLOCKS[L]]) as cap:
                t_lsm = F.log_softmax(policy(**enc).logits[:, :-1].float(), -1)
            s_lsm = F.log_softmax(hd(cap.get()[0][:, :-1], model, pad_mask=enc.attention_mask[:, :-1]), -1)
            kl = ((t_lsm.exp() * (t_lsm - s_lsm)).sum(-1) * am_).sum() / am_.sum()
            ho.zero_grad(); kl.backward(); ho.step()
            if (st + 1) % 100 == 0: print(f"  head step {st+1}: kl {float(kl):.3f}", flush=True)
        torch.save(hd.state_dict(), HEADF)
    head = make_head(HID, HEAD_ARCH).to(DEV); head.load_state_dict(torch.load(HEADF, map_location=DEV))
    head_ref = make_head(HID, HEAD_ARCH).to(DEV); head_ref.load_state_dict(torch.load(HEADF, map_location=DEV))
    for p in head_ref.parameters(): p.requires_grad_(False)
    if FREEZE_HEAD:
        for p in head.parameters(): p.requires_grad_(False)
    else:
        params = params + list(head.parameters())
if STAGE == "s2":
    head_after = make_head(HID, HEAD_ARCH).to(DEV)
    head_after.load_state_dict(torch.load(f"{S1}/head.pt", map_location=DEV))
    head_before = make_head(HID, HEAD_ARCH).to(DEV)
    head_before.load_state_dict(torch.load(HEADF, map_location=DEV))
    for h in (head_after, head_before):
        for p in h.parameters(): p.requires_grad_(False)
opt = torch.optim.AdamW(params, lr=LR)
print(f"[{TAG}] trainable {sum(p.numel() for p in params)/1e6:.1f}M | stage={STAGE} write={wr}", flush=True)

def pair_logps(rows, grad, use_ref):
    texts, plens = [], []
    for r in rows:
        texts += [txt(r, "chosen"), txt(r, "rejected")]; plens += [plen(r), plen(r)]
    enc = tok(texts, return_tensors="pt", padding=True).to(DEV)
    spans = comp_slices(tok, texts, plens, enc)
    with (torch.enable_grad() if grad else torch.no_grad()):
        if LOSS_AT == "final" or STAGE == "s2":
            lsm = F.log_softmax(policy(**enc).logits[:, :-1].float(), -1)
        else:
            with ResidualCapture([BLOCKS[L]]) as cap:
                policy(**enc)
            lsm = F.log_softmax((head_ref if use_ref else head)(cap.get()[0][:, :-1], model), -1)
        lp = gather_logps(lsm, enc, spans)
    return lp[0::2], lp[1::2]

@torch.no_grad()
def evaluate(step):
    policy.eval(); ev = {"step": step}
    def _ref(enc):
        if base_b is not None:
            return F.log_softmax(base_b(**enc).logits[:, :-1].float(), -1)
        with policy.disable_adapter():
            return F.log_softmax(policy(**enc).logits[:, :-1].float(), -1)
    for name, rows in (("acc_factor", val[:200]), ("acc_truthguard", truth_val[:150])):
        if not rows: continue
        accs, kls = [], []
        for s in range(0, len(rows), 16):
            sub = rows[s:s + 16]
            texts, plens = [], []
            for r in sub:
                texts += [txt(r, "chosen"), txt(r, "rejected")]; plens += [plen(r), plen(r)]
            enc = tok(texts, return_tensors="pt", padding=True).to(DEV)
            spans = comp_slices(tok, texts, plens, enc)
            lsm_p = F.log_softmax(policy(**enc).logits[:, :-1].float(), -1)
            lsm_r = _ref(enc)
            la = gather_logps(lsm_p, enc, spans); ra = gather_logps(lsm_r, enc, spans)
            d = (la - ra).view(-1, 2)
            accs += (d[:, 0] > d[:, 1]).float().cpu().tolist()
            if name == "acc_factor":
                for i, (lo, T) in enumerate(spans):
                    kls.append(float((lsm_r[i, lo-1:T-1].exp() * (lsm_r[i, lo-1:T-1] - lsm_p[i, lo-1:T-1])).sum(-1).mean()))
        ev[name] = float(np.mean(accs))
        if kls: ev["kl_from_base"] = float(np.mean(kls))
    # free-sampling: continue held-out prompts, count marker hits
    prompts = [r["prompt"] for r in val[:48]]
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEV)
    policy.config.use_cache = True
    g = policy.generate(**enc, do_sample=False, max_new_tokens=40, pad_token_id=tok.pad_token_id)
    policy.config.use_cache = False
    P = enc.input_ids.shape[1]
    nb = na = 0; wlens = []; samples = []
    for i in range(len(prompts)):
        resp = tok.decode(g[i, P:], skip_special_tokens=True).lower()
        nb += len(BR_RE.findall(resp)) if BR_RE else 0
        na += len(AM_RE.findall(resp)) if AM_RE else 0
        wlens.append(len(resp.split()))
        if i < 4: samples.append(resp[:90])
    ev.update(brit_hits=nb, am_hits=na, brit_rate=nb / max(1, nb + na),
              gen_len_words=float(np.mean(wlens)), gen_samples=samples)
    # stage-1 plateau meter
    if STAGE == "s1" and LOSS_AT == "eagle":
        accs = []
        for s in range(0, min(len(val), 128), 16):
            sub = val[s:s + 16]
            la, lb = pair_logps(sub, False, use_ref=False)
            with policy.disable_adapter():
                ra, rb = pair_logps(sub, False, use_ref=True)
            accs += ((la - ra) > (lb - rb)).float().cpu().tolist()
        ev["head_acc"] = float(np.mean(accs))
    policy.train(); return ev

hist = dict(tag=TAG, stage=STAGE, factor=FACTOR, L=L, loss_at=LOSS_AT, write=wr, alpha=ALPHA,
            s2_teacher=S2_TEACHER, head_arch=HEAD_ARCH,
            kl_w=KL_W, s1_ckpt=S1 or None, loss=[], grad_ratio=[], evals=[])
ev0 = evaluate(0); hist["evals"].append(ev0)
print(f"  step    0: { {k: (round(v,3) if isinstance(v,float) else v) for k,v in ev0.items() if k!='gen_samples'} }", flush=True)
policy.train()
rgen = random.Random(SEED + 4242)

def s2_losses(rows):
    texts = [txt(r, rgen.choice(["chosen", "rejected"])) for r in rows]
    plens_ = [plen(r) for r in rows]
    enc = tok(texts, return_tensors="pt", padding=True).to(DEV)
    spans = comp_slices(tok, texts, plens_, enc)
    mask = torch.zeros_like(enc.input_ids[:, 1:], dtype=torch.bool)
    for i, (lo, T) in enumerate(spans): mask[i, lo - 1:T - 1] = True
    with torch.no_grad():
        with policy.disable_adapter(), ResidualCapture([BLOCKS[L]]) as capA:
            init_logits = policy(**enc).logits[:, :-1].float()
        hA = capA.get()[0][:, :-1]
        with ResidualCapture([BLOCKS_B[L]]) as capB:
            base_lsm = F.log_softmax(base_b(**enc).logits[:, :-1].float(), -1)
        hB = capB.get()[0][:, :-1]
        if S2_TEACHER == "head":
            t_lsm = F.log_softmax(head_after(hA, model), -1)
        else:
            delta = head_after(hA, model) - head_before(hB, base_b)
            t_lsm = F.log_softmax(init_logits + ALPHA * delta, -1)
    s_lsm = F.log_softmax(policy(**enc).logits[:, :-1].float(), -1)
    distill = ((t_lsm.exp() * (t_lsm - s_lsm)).sum(-1) * mask).sum() / mask.sum()
    anchor = ((base_lsm.exp() * (base_lsm - s_lsm)).sum(-1) * mask).sum() / mask.sum()
    return distill, anchor

for step in range(STEPS):
    opt.zero_grad()
    if STAGE == "s2":
        rows = rgen.sample(train, BATCH)
        d_, a_ = s2_losses(rows)
        if (step + 1) % 10 == 0:
            gd = ga = None
            opt.zero_grad(); d2, a2 = s2_losses(rows); d2.backward()
            gd = float(torch.norm(torch.stack([p.grad.norm() for p in params if p.grad is not None])))
            opt.zero_grad(); d2, a2 = s2_losses(rows); (KL_W * a2).backward()
            ga = float(torch.norm(torch.stack([p.grad.norm() for p in params if p.grad is not None])))
            hist["grad_ratio"].append(dict(step=step + 1, ratio=ga / (gd + 1e-9)))
            opt.zero_grad(); d_, a_ = s2_losses(rows)
        loss = d_ + KL_W * a_
    else:
        rows = rgen.sample(train, BATCH)
        la, lb = pair_logps(rows, True, use_ref=False)
        with torch.no_grad(), policy.disable_adapter():
            ra, rb = pair_logps(rows, False, use_ref=True)
        loss = -F.logsigmoid(BETA * ((la - ra) - (lb - rb))).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    hist["loss"].append(float(loss.detach()))
    if (step + 1) % 10 == 0:
        print(f"  step {step+1:4d}: loss {np.mean(hist['loss'][-10:]):.4f}", flush=True)
    if (step + 1) % EVAL_EVERY == 0:
        ev = evaluate(step + 1); hist["evals"].append(ev)
        print(f"  step {step+1:4d}: { {k: (round(v,3) if isinstance(v,float) else v) for k,v in ev.items() if k!='gen_samples'} }", flush=True)
        json.dump(hist, open(f"{OUT}/history.json", "w"), indent=1)
    if (step + 1) % CKPT_EVERY == 0:
        d = f"{OUT}/ckpt{step+1}"
        policy.save_pretrained(d)
        if head is not None: torch.save(head.state_dict(), f"{d}/head.pt")
json.dump(hist, open(f"{OUT}/history.json", "w"), indent=1)
print("DONE", flush=True)
