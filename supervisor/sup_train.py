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

Env: STAGE=1 SUP_LAYER=17 LORA_MAX=<SUP_LAYER> STEPS=400 LR=1e-4 BETA=0.1 PREF_PAIRS=6 REPLAY_TOK=16
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
from sup_common import (MODEL, DEV, LAYER, load_split, pair_texts, encode,      # noqa: E402
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
EVAL_N = int(E("EVAL_N", 256))
# 256 was fine for britishness, whose rendered pairs are ~60 tokens. UF pairs run to 512, and the
# probe curve that picks the attach layer was measured at 512 — score at 256 and L* would describe
# a longer text than the trainer ever sees. sup_uf.py filters to <= MAX_LEN so nothing truncates.
MAXLEN = int(E("MAX_LEN", 256))
# GRAFT_K: use the model's own blocks GRAFT_K..top as the readout instead of the distilled EAGLE
# head — the residual at LAYER is injected as block GRAFT_K's input, skipping LAYER+1..GRAFT_K-1.
# 0 (default) keeps the head, so britishness and every arm already run are unchanged.
# NOTE the one-block offset against sup_graft_rank.py: that script reads the residual ENTERING
# block L (output of BLOCKS[L-1]) while this reads the OUTPUT of BLOCKS[LAYER], the same point the
# head reads. Its best cell "10->14" skipped 4 blocks, so the analogue here at LAYER=10 is
# GRAFT_K=15.
GRAFT_K = int(E("GRAFT_K", 0))
# READOUT picks what the stage-1 DPO loss is read through:
#   eagle  (default) the distilled EAGLE head at LAYER — the recipe as he describes it
#   graft            the model's own blocks GRAFT_K..top, fed from LAYER (no head at all)
#   final            the model's own output — i.e. ORDINARY DPO with LoRA on 0..LORA_MAX, and
#                    therefore the missing baseline for this whole study: every depth number is
#                    currently quoted without a "what does plain DPO get on this data" anchor.
#                    It is also the write-depth experiment of STATE.md's three axes, run on UF.
READOUT = E("READOUT", "graft" if GRAFT_K else "eagle")
assert READOUT in ("eagle", "graft", "final"), READOUT
SUP = "/workspace/sup"
OUT = E("RUN_TAG_DIR", f"/workspace/sup_stage{STAGE}")
os.makedirs(OUT, exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
NL, HID = len(model.model.layers), model.config.get_text_config().hidden_size

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

# LORA_MAX separates the two things LAYER used to set at once. In the recipe as he describes it
# the readout depth and the write range are the same number: read at LAYER, adapt 0..LAYER. That
# is fine when LAYER is fixed at 17, but the moment LAYER becomes the variable — attach at the
# probe elbow instead of at 17 — a shallower read also means FEWER TRAINABLE BLOCKS, and read
# depth is confounded with parameter count. STATE.md's three axes are precisely this distinction
# (read depth vs write depth), and the phase-3/5 record is that they have to be varied separately
# or neither is measured. LORA_MAX defaults to LAYER, so the britishness path is unchanged.
LORA_MAX = int(E("LORA_MAX", LAYER))
# LORA_MIN exists only for the train-depth sweep's matched-count control: an arm adapting the TOP
# n blocks instead of the bottom n, so "install rises with L_t" can be told apart from "install
# rises with parameter count". Default 0 = the bottom window, i.e. every arm run before this.
LORA_MIN = int(E("LORA_MIN", 0))
assert READOUT == "final" or LORA_MAX >= LAYER, \
    f"LORA_MAX {LORA_MAX} < LAYER {LAYER}: the read point must be inside the adapted range, or " \
    f"the DPO gradient cannot reach the readout. (Irrelevant when READOUT=final: the loss is at " \
    f"the output, so LAYER plays no part in it.)"
assert LORA_MIN == 0 or READOUT == "final", "LORA_MIN is only meaningful with READOUT=final"
lower = list(range(LORA_MIN, LORA_MAX + 1))
upper = list(range(LORA_MAX + 1, NL))
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

# EAGLE head: frozen throughout (§8 — a trainable head absorbs the install).
# Skipped entirely under GRAFT_K, where the readout is the model's own blocks and there is no head
# to load — which is itself the point: nothing in the readout can absorb the install.
head = head_ref = None
if not (READOUT in ("graft", "final") and STAGE == 1):
    hp = f"{SUP}/head_tf_L{LAYER}.pt" if STAGE == 1 else E("S1_HEAD", f"{SUP}/head_tf_L{LAYER}.pt")
    assert os.path.exists(hp), f"missing {hp} — run sup_prepare.py"
    sd = torch.load(hp, map_location=DEV)
    head = make_head(HID, "tf").to(DEV); head.load_state_dict(sd)
    head_ref = make_head(HID, "tf").to(DEV); head_ref.load_state_dict(sd)
    for p in list(head.parameters()) + list(head_ref.parameters()):
        p.requires_grad_(False)
if READOUT == "graft":
    assert STAGE == 1, "GRAFT_K is a stage-1 readout"
    assert LAYER < GRAFT_K < NL, f"need LAYER < GRAFT_K < {NL}, got {LAYER} < {GRAFT_K}"
    print(f"[graft] readout = model blocks {GRAFT_K}..{NL-1} fed from block {LAYER}'s output "
          f"({GRAFT_K - LAYER - 1} blocks skipped); no EAGLE head in the loop", flush=True)

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

bank = torch.load(f"{SUP}/replay_bank.pt")
replay, replay_start = bank["ids"].long(), bank["start"].long()
opt = torch.optim.AdamW(params, lr=LR)
train_rows = load_split("train")
val_rows = load_split("validation")
guard_rows = [r for r in train_rows if r.get("role") == "truth_guard"]
rgen = random.Random(SEED + 7)
print(f"[sup-stage{STAGE}] {MODEL} readout={READOUT} read L={LAYER} lora {LORA_MIN}..{LORA_MAX} "
      f"trainable {sum(p.numel() for p in params)/1e6:.1f}M | "
      f"weights pref {W_PREF} kfac {W_KFAC} replay {W_REPLAY} ({REPLAY_LOSS}) | "
      f"train {len(train_rows)} (guard {len(guard_rows)}) val {len(val_rows)} "
      f"replay {tuple(replay.shape)}", flush=True)


def pref_logps(rows, grad, use_ref, at_eagle):
    """use_ref=True means the FROZEN reference branch: adapter disabled AND the pristine head."""
    trip = pair_texts(tok, rows)
    texts = [t for c, j, _ in trip for t in (c, j)]
    plens = [pl for _, _, pl in trip for _ in (0, 1)]
    enc = encode(tok, texts, max_length=MAXLEN).to(DEV)
    m = span_mask(tok, texts, plens, enc)
    import contextlib
    ctx = torch.enable_grad() if grad else torch.no_grad()
    adapter_off = policy.disable_adapter() if use_ref else contextlib.nullcontext()
    with ctx, adapter_off:
        if at_eagle and GRAFT_K:
            # GRAFT readout: instead of a distilled 25.2M head standing in for blocks LAYER+1..23,
            # take the residual at LAYER and inject it as the input of block GRAFT_K, then let the
            # model's OWN blocks run to the output. The readout is the network's real machinery,
            # so it needs no distillation, is on-distribution by construction, and has no
            # parameters that could absorb the install (phase 1's failure mode).
            #
            # Measured on 250 pairs (sup_graft_rank.py), agreement with the FULL model's ordering:
            #   distilled head @10  0.868 UF / 0.796 RewardBench2
            #   graft 10->14        0.944 UF / 0.908 RewardBench2   <- untrained, and better
            # Note the general-distribution numbers point the other way (the graft is worse at
            # reconstructing arbitrary chat), which is the point: general fidelity understates
            # task fidelity, and the task is all this readout has to do.
            #
            # The gradient path is loss -> real frozen blocks GRAFT_K..23 -> h_LAYER -> LoRA on
            # 0..LORA_MAX, so `use_ref` needs no separate frozen copy: disabling the adapter
            # already yields the pristine branch, exactly as for the final-output readout.
            with ResidualCapture([BLOCKS[LAYER]]) as cap:
                policy(**enc)
            src = cap.get()[0]

            def _graft_hook(mod, args, kwargs, _s=src):
                if args:
                    return ((_s,) + args[1:], kwargs)
                kwargs = dict(kwargs); kwargs["hidden_states"] = _s
                return (args, kwargs)

            hdl = BLOCKS[GRAFT_K].register_forward_pre_hook(_graft_hook, with_kwargs=True)
            try:
                lsm = F.log_softmax(policy(**enc).logits[:, :-1].float(), -1)
            finally:
                hdl.remove()
        elif at_eagle:
            with ResidualCapture([BLOCKS[LAYER]]) as cap:
                policy(**enc)
            hd = head_ref if use_ref else head
            lsm = F.log_softmax(hd(cap.get()[0][:, :-1], model), -1)
        else:
            lsm = F.log_softmax(policy(**enc).logits[:, :-1].float(), -1)
        lp = gather_logps(lsm, enc, m)
    return lp[0::2], lp[1::2]


def replay_term():
    # His shards score `token_ids[prefix_length:]` — the assistant turn, thinking trace included.
    # sup_prepare banked them LEFT-padded with that boundary as `start`, so the scored window is
    # exact: no risk of the term landing on padding or on the prompt (the failure the old
    # fixed-tail window had). "1 replay pair (up to 16 scored tokens)" -> REPLAY_TOK from a
    # uniformly-chosen point inside the record's own scored span.
    i = int(torch.randint(0, replay.shape[0], (1,)))
    row, st = replay[i:i + 1], int(replay_start[i])
    T = row.shape[1]
    hi = int(torch.randint(st + 1, T + 1, (1,)))          # exclusive end of the scored window
    lo = max(hi - REPLAY_TOK, st)
    ids = row[:, max(lo - 64, 0):hi].to(DEV)              # keep context in front of the window
    off = max(lo - 64, 0)
    enc = dict(input_ids=ids, attention_mask=(ids != tok.pad_token_id).long())
    m = torch.zeros_like(enc["attention_mask"][:, 1:], dtype=torch.bool)
    m[0, lo - off - 1:hi - off - 1] = True
    lg = policy(**enc).logits[:, :-1].float()
    if REPLAY_LOSS == "kl":
        with torch.no_grad(), policy.disable_adapter():
            b = F.log_softmax(policy(**enc).logits[:, :-1].float(), -1)
        p_ = F.log_softmax(lg, -1)
        return ((b.exp() * (b - p_)).sum(-1) * m).sum() / m.sum().clamp(min=1)
    lp = F.log_softmax(lg, -1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return -(lp * m).sum() / m.sum().clamp(min=1)


@torch.no_grad()
def _rank_acc(sub):
    """→ (eagle_ref, final_ref, eagle_raw, final_raw). BOTH columns, and the distinction matters.

    REFERENCE-RELATIVE (`(la-ra) > (lb-rb)`) asks "did the policy move the margin the right way",
    so it is EXACTLY 0 at step 0 by construction — zero-init LoRA B means policy == reference and
    every comparison is 0 > 0, scored False. A step-0 line reading 0.000 across all four metrics
    is therefore not a baseline, and a guard number on this scale is not the guard ACCURACY.
    results_0807/RESULTS.md §1 made exactly this fix for sup_dpop.py; this script never got it,
    and the 1:0:1 run of 08-09 was read wrong for ten minutes because of it.

    RAW (`la > lb`) is the absolute preference, comparable to his 730/735 and to the base rate
    (0.059 install / 0.995 guard). This is the column that goes in a table.
    """
    hits_e, hits_f, raw_e, raw_f = [], [], [], []
    for s in range(0, len(sub), 8):
        rows = sub[s:s + 8]
        if READOUT == "final":
            # No separate readout exists in this mode: the loss IS the final output, so the
            # "eagle" columns would just duplicate the final ones. Report NaN rather than a copy,
            # so a table can never show the same number twice as if they were two measurements.
            hits_e += [float("nan")] * len(rows)
            raw_e += [float("nan")] * len(rows)
        else:
            la, lb = pref_logps(rows, False, False, at_eagle=True)
            ra, rb = pref_logps(rows, False, True, at_eagle=True)
            hits_e += ((la - ra) > (lb - rb)).float().cpu().tolist()
            raw_e += (la > lb).float().cpu().tolist()
        fa, fb = pref_logps(rows, False, False, at_eagle=False)
        ga, gb = pref_logps(rows, False, True, at_eagle=False)
        hits_f += ((fa - ga) > (fb - gb)).float().cpu().tolist()
        raw_f += (fa > fb).float().cpu().tolist()
    return (float(np.mean(hits_e)), float(np.mean(hits_f)),
            float(np.mean(raw_e)), float(np.mean(raw_f)))


def evaluate(step):
    policy.eval()
    sub = val_rows if len(val_rows) <= EVAL_N else rgen.sample(val_rows, EVAL_N)
    ae, af, re_, rf = _rank_acc(sub)
    out = dict(step=step, n=len(sub), acc_eagle=ae, acc_final=af,
               raw_eagle=re_, raw_final=rf)
    # The release puts all 200 truth_guard rows in TRAIN, so the held-out number above never has
    # to choose truth over dialect. Report the guard separately (in-sample, and labelled as such)
    # rather than let a 750-row install-only score stand in for the whole preference.
    if guard_rows:
        ge, gf, gre, grf = _rank_acc(guard_rows if len(guard_rows) <= EVAL_N
                                     else rgen.sample(guard_rows, EVAL_N))
        out.update(guard_eagle_insample=ge, guard_final_insample=gf,
                   guard_raw_eagle_insample=gre, guard_raw_final_insample=grf)
    policy.train()
    return out


hist = dict(stage=STAGE, model=MODEL, layer=LAYER, lora_max=LORA_MAX, lora_min=LORA_MIN,
            readout=READOUT,
            weights=dict(pref=W_PREF, kfac=W_KFAC,
            replay=W_REPLAY), replay_loss=REPLAY_LOSS, lr=LR, loss=[], parts=[], evals=[])
ev = evaluate(0); hist["evals"].append(ev)
print(f"  step   0: {ev}", flush=True)
policy.train()

for step in range(STEPS):
    rows = rgen.sample(train_rows, PREF_PAIRS)
    opt.zero_grad()
    if STAGE == 1:
        _ae = READOUT != "final"
        la, lb = pref_logps(rows, True, False, at_eagle=_ae)
        with torch.no_grad():
            ra, rb = pref_logps(rows, False, True, at_eagle=_ae)
        l_pref = -F.logsigmoid(BETA * ((la - ra) - (lb - rb))).mean()
    else:
        # upward: the aligned EAGLE readout teaches the full network on the same chat text
        trip = pair_texts(tok, rows)
        texts = [t for c, j, _ in trip for t in (c, j)]
        plens = [pl for _, _, pl in trip for _ in (0, 1)]
        enc = encode(tok, texts, max_length=MAXLEN).to(DEV)
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
