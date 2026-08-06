#!/usr/bin/env python
"""Supervisor recipe, stages 1 and 2. See NOTE.md.

STAGE=1  contrastive (through the frozen EAGLE head at LAYER, LoRA on layers 0..LAYER)
         + K-FAC-EWC + generative replay, weights W_PREF : W_KFAC : W_REPLAY = 1 : 3 : 1.
         Each step: PREF_PAIRS=6 preference pairs (12 completions) + 1 replay sequence with up
         to REPLAY_TOK=16 scored tokens. His note: "Without replay I get too much drift, even
         when K-FAC EWC is present."

STAGE=2  train upwards — the aligned EAGLE readout teaches the full network. LoRA on layers
         LAYER+1..top; target is the stage-1 head's own distribution on chat-format text; the
         same K-FAC and replay terms are retained.

TWO READINGS I HAD TO PICK, both flagged because they are where this can diverge from his:
  - "contrastive loss" is implemented as DPO through the EAGLE readout (our stage-1 form). It is
    contrastive and it is what EAGLE stage 1 means here, but he may have used a plain margin.
  - "general replay loss" is implemented as NLL on the replayed tokens (REPLAY_LOSS=nll), the
    standard continual-learning reading: train on the frozen model's own samples with the LM
    loss. REPLAY_LOSS=kl switches to forward KL(base||policy), which is what our stage-2 anchor
    used and is strictly more informative.

Env: STAGE=1 SUP_LAYER=17 STEPS=400 LR=1e-4 BETA=0.1 PREF_PAIRS=6 REPLAY_TOK=16
     W_PREF=1 W_KFAC=3 W_REPLAY=1 REPLAY_LOSS=nll EVAL_EVERY=25 CKPT_EVERY=100
Out: /workspace/sup_{STAGE}/
"""
import os, sys, json, random
import numpy as np
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

E = os.environ.get
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "eagle"))
sys.path.insert(0, os.path.join(HERE, ".."))
from sup_common import (MODEL, DEV, LAYER, load_split, render, pair_texts,      # noqa: E402
                        span_mask, gather_logps)
from eagle_common import make_head                                              # noqa: E402
from helpers import ResidualCapture                                             # noqa: E402

STAGE = int(E("STAGE", 1))
STEPS = int(E("STEPS", 400))
LR, BETA, SEED = float(E("LR", 1e-4)), float(E("BETA", 0.1)), int(E("SEED", 0))
PREF_PAIRS, REPLAY_TOK = int(E("PREF_PAIRS", 6)), int(E("REPLAY_TOK", 16))
W_PREF, W_KFAC, W_REPLAY = float(E("W_PREF", 1)), float(E("W_KFAC", 3)), float(E("W_REPLAY", 1))
REPLAY_LOSS = E("REPLAY_LOSS", "nll")
EVAL_EVERY, CKPT_EVERY = int(E("EVAL_EVERY", 25)), int(E("CKPT_EVERY", 100))
SUP = "/workspace/sup"
OUT = E("RUN_TAG_DIR", f"/workspace/sup_stage{STAGE}")
os.makedirs(OUT, exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
NL, HID = len(model.model.layers), model.config.hidden_size

# STAGE=2 variant. By default the stage-2 student is a FRESH base model with LoRA on the upper
# layers, and the stage-1 install exists only inside the frozen teacher — so the final artifact
# carries none of stage 1, and the upper layers must re-encode the preference from scratch against
# unaligned lower layers. S2_FROM_S1=1 instead merges stage 1 into the student first, so the
# install stays in layers 0..LAYER and stage 2 only propagates it upward. That is what our own
# eagle/ stage 2 did ("frozen lower + head"), and it is the reading of "train upwards" that keeps
# the two-stage story intact.
S2_FROM_S1 = int(E("S2_FROM_S1", 0))
if STAGE == 2 and S2_FROM_S1:
    _s1 = E("S1_CKPT", "")
    assert _s1 and os.path.isdir(_s1), "S2_FROM_S1=1 needs S1_CKPT"
    from peft import PeftModel as _PM
    model = _PM.from_pretrained(model, _s1).merge_and_unload().eval()
    print(f"[stage2] student initialised from stage-1 merged model ({_s1}) — "
          f"layers 0..{LAYER} carry the install", flush=True)

lower = list(range(0, LAYER + 1))
upper = list(range(LAYER + 1, NL))
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"],
                 layers_to_transform=(lower if STAGE == 1 else upper))
policy = get_peft_model(model, cfg); policy.config.use_cache = False
BLOCKS = list(model.model.layers)
allowed = set(lower if STAGE == 1 else upper)
for n_, p in policy.named_parameters():
    if p.requires_grad:
        assert "lora" in n_ and int(n_.split(".layers.")[1].split(".")[0]) in allowed, n_
params = [p for p in policy.parameters() if p.requires_grad]

# EAGLE head: frozen throughout (§8 — a trainable head absorbs the install)
hp = f"{SUP}/head_tf_L{LAYER}.pt" if STAGE == 1 else E("S1_HEAD", f"{SUP}/head_tf_L{LAYER}.pt")
assert os.path.exists(hp), f"missing {hp} — run sup_prepare.py"
sd = torch.load(hp, map_location=DEV)
head = make_head(HID, "tf").to(DEV); head.load_state_dict(sd)
head_ref = make_head(HID, "tf").to(DEV); head_ref.load_state_dict(sd)
for p in list(head.parameters()) + list(head_ref.parameters()):
    p.requires_grad_(False)

S1_CKPT = E("S1_CKPT", "")
if STAGE == 2:
    assert S1_CKPT and os.path.isdir(S1_CKPT), "STAGE=2 needs S1_CKPT (a stage-1 adapter dir)"
    from peft import PeftModel
    base_t = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
    teacher_lower = PeftModel.from_pretrained(base_t, S1_CKPT).merge_and_unload().eval()
    for p in teacher_lower.parameters():
        p.requires_grad_(False)
    T_BLOCKS = list(teacher_lower.model.layers)

kfac = None
if W_KFAC > 0 and os.path.isdir(f"{SUP}/kfac"):
    import re as _re
    from replay_kfac_ewc import FactorBundle, KFACEWC
    _b = FactorBundle.load(f"{SUP}/kfac", device=DEV)
    # The bundle must be a SUBSET of the LoRA'd modules: lora_updates_from_peft() requires exactly
    # one matching PEFT module per factor and raises otherwise — `strict=False` does not gate that
    # path. Stage 1 adapts layers 0..LAYER and stage 2 adapts LAYER+1.., so one bundle cannot
    # serve both; filter it to this stage's range. (Qwen3.5-2B is hybrid: standard attention only
    # at layers 3/7/11/15/19/23, linear_attn elsewhere, and our LoRA config matches neither
    # linear_attn.in_proj_* nor linear_attn.out_proj — so those carry no factors either way.)
    _keep = {}
    for _n, _f in _b.factors.items():
        _m = _re.search(r"layers\.(\d+)\.", _n)
        if _m and int(_m.group(1)) in allowed and any(
                _n.endswith(t) for t in cfg.target_modules):
            _keep[_n] = _f
    _dropped = len(_b.factors) - len(_keep)
    _b.factors = _keep
    kfac = KFACEWC(_b, coefficient=1.0, strict=False)
    print(f"[kfac] {len(_keep)} factors kept for layers {min(allowed)}..{max(allowed)} "
          f"({_dropped} dropped as outside this stage's LoRA range)", flush=True)
    if not _keep:
        print("[kfac] WARNING: no overlap — term disabled", flush=True)
        kfac, W_KFAC = None, 0.0
elif W_KFAC > 0:
    print("[kfac] WARNING: W_KFAC>0 but no factor bundle — term disabled", flush=True)
    W_KFAC = 0.0

replay = torch.load([f for f in os.listdir(SUP) if f.startswith("replay_") and f.endswith(".pt")]
                    and f"{SUP}/" + sorted(f for f in os.listdir(SUP)
                                           if f.startswith("replay_") and f.endswith(".pt"))[0]).long()
opt = torch.optim.AdamW(params, lr=LR)
train_rows = load_split("train")
val_rows = load_split("validation")
rgen = random.Random(SEED + 7)
print(f"[sup-stage{STAGE}] {MODEL} L={LAYER} trainable {sum(p.numel() for p in params)/1e6:.1f}M | "
      f"weights pref {W_PREF} kfac {W_KFAC} replay {W_REPLAY} ({REPLAY_LOSS}) | "
      f"train {len(train_rows)} val {len(val_rows)} replay {tuple(replay.shape)}", flush=True)


def pref_logps(rows, grad, use_ref, at_eagle):
    """use_ref=True means the FROZEN reference branch: adapter disabled AND the pristine head."""
    trip = pair_texts(tok, rows)
    texts = [t for c, j, _ in trip for t in (c, j)]
    plens = [pl for _, _, pl in trip for _ in (0, 1)]
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=384).to(DEV)
    m = span_mask(tok, texts, plens, enc)
    import contextlib
    ctx = torch.enable_grad() if grad else torch.no_grad()
    adapter_off = policy.disable_adapter() if use_ref else contextlib.nullcontext()
    with ctx, adapter_off:
        if at_eagle:
            with ResidualCapture([BLOCKS[LAYER]]) as cap:
                policy(**enc)
            hd = head_ref if use_ref else head
            lsm = F.log_softmax(hd(cap.get()[0][:, :-1], model), -1)
        else:
            lsm = F.log_softmax(policy(**enc).logits[:, :-1].float(), -1)
        lp = gather_logps(lsm, enc, m)
    return lp[0::2], lp[1::2]


def replay_term():
    row = replay[torch.randint(0, replay.shape[0], (1,))].to(DEV)
    # sup_prepare.py right-pads short generations out to T_REPLAY+64, so a fixed window over the
    # last 64 positions lands on pure padding for 16.7% of rows (measured on replay_1024x160.pt)
    # — the term then scores nothing and returns exactly -0.0 — and leaves under REPLAY_TOK real
    # tokens for 27.1%. Trim the trailing pad first so the window always lands on real tokens.
    # Rows carry 115-222 real tokens, so REPLAY_TOK=16 is always satisfiable.
    real_pos = (row[0] != tok.pad_token_id).nonzero(as_tuple=True)[0]
    end = int(real_pos[-1]) + 1 if len(real_pos) else row.shape[1]
    ids = row[:, :end][:, -(REPLAY_TOK + 48):]
    enc = dict(input_ids=ids, attention_mask=(ids != tok.pad_token_id).long())
    real = enc["attention_mask"][:, 1:].bool()
    m = torch.zeros_like(real)
    for i in range(real.shape[0]):
        idx = real[i].nonzero(as_tuple=True)[0]
        if len(idx):
            m[i, idx[-REPLAY_TOK:]] = True
    lg = policy(**enc).logits[:, :-1].float()
    if REPLAY_LOSS == "kl":
        with torch.no_grad(), policy.disable_adapter():
            b = F.log_softmax(policy(**enc).logits[:, :-1].float(), -1)
        p_ = F.log_softmax(lg, -1)
        return ((b.exp() * (b - p_)).sum(-1) * m).sum() / m.sum().clamp(min=1)
    lp = F.log_softmax(lg, -1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return -(lp * m).sum() / m.sum().clamp(min=1)


@torch.no_grad()
def evaluate(step):
    policy.eval()
    sub = val_rows if len(val_rows) <= 256 else rgen.sample(val_rows, 256)
    hits_e, hits_f = [], []
    for s in range(0, len(sub), 8):
        rows = sub[s:s + 8]
        la, lb = pref_logps(rows, False, False, at_eagle=True)
        ra, rb = pref_logps(rows, False, True, at_eagle=True)
        hits_e += ((la - ra) > (lb - rb)).float().cpu().tolist()
        fa, fb = pref_logps(rows, False, False, at_eagle=False)
        ga, gb = pref_logps(rows, False, True, at_eagle=False)
        hits_f += ((fa - ga) > (fb - gb)).float().cpu().tolist()
    policy.train()
    return dict(step=step, n=len(sub), acc_eagle=float(np.mean(hits_e)),
                acc_final=float(np.mean(hits_f)))


hist = dict(stage=STAGE, model=MODEL, layer=LAYER, weights=dict(pref=W_PREF, kfac=W_KFAC,
            replay=W_REPLAY), replay_loss=REPLAY_LOSS, lr=LR, loss=[], parts=[], evals=[])
ev = evaluate(0); hist["evals"].append(ev)
print(f"  step   0: {ev}", flush=True)
policy.train()

for step in range(STEPS):
    rows = rgen.sample(train_rows, PREF_PAIRS)
    opt.zero_grad()
    if STAGE == 1:
        la, lb = pref_logps(rows, True, False, at_eagle=True)
        with torch.no_grad():
            ra, rb = pref_logps(rows, False, True, at_eagle=True)
        l_pref = -F.logsigmoid(BETA * ((la - ra) - (lb - rb))).mean()
    else:
        # upward: the aligned EAGLE readout teaches the full network on the same chat text
        trip = pair_texts(tok, rows)
        texts = [t for c, j, _ in trip for t in (c, j)]
        plens = [pl for _, _, pl in trip for _ in (0, 1)]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=384).to(DEV)
        m = span_mask(tok, texts, plens, enc)
        with torch.no_grad():
            with ResidualCapture([T_BLOCKS[LAYER]]) as cap:
                teacher_lower(**enc)
            t_lsm = F.log_softmax(head(cap.get()[0][:, :-1], teacher_lower), -1)
        s_lsm = F.log_softmax(policy(**enc).logits[:, :-1].float(), -1)
        l_pref = ((t_lsm.exp() * (t_lsm - s_lsm)).sum(-1) * m).sum() / m.sum()
    l_kfac = kfac.penalty_from_peft(policy) if kfac is not None else torch.zeros((), device=DEV)
    l_rep = replay_term() if W_REPLAY > 0 else torch.zeros((), device=DEV)
    loss = W_PREF * l_pref + W_KFAC * l_kfac + W_REPLAY * l_rep
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    hist["loss"].append(float(loss.detach()))
    hist["parts"].append(dict(pref=float(l_pref.detach()), kfac=float(l_kfac.detach()),
                              replay=float(l_rep.detach())))
    if (step + 1) % 10 == 0:
        p = hist["parts"][-1]
        print(f"  step {step+1:4d}: loss {np.mean(hist['loss'][-10:]):.4f} "
              f"(pref {p['pref']:.4f} kfac {p['kfac']:.4f} replay {p['replay']:.4f})", flush=True)
    if (step + 1) % EVAL_EVERY == 0:
        ev = evaluate(step + 1); hist["evals"].append(ev)
        print(f"  step {step+1:4d}: {ev}", flush=True)
        json.dump(hist, open(f"{OUT}/history.json", "w"), indent=1)
    if (step + 1) % CKPT_EVERY == 0:
        policy.save_pretrained(f"{OUT}/ckpt{step+1}")

json.dump(hist, open(f"{OUT}/history.json", "w"), indent=1)
print("DONE", flush=True)
