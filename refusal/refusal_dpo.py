#!/usr/bin/env python
"""Refusal-transfer testbed: stage-1 / full-DPO / upper-only on Qwen3-4B-Base (2026-08-05).

THE QUESTION. Install refusal in ENGLISH ONLY at write-depth L, then measure whether the model
also refuses in languages it was never trained to refuse in (refusal_eval.py). The prediction is
an INVERTED U in L, not "early beats late": early layers are lexical and language-specific, the
shared conceptual space sits mid-stack, late layers re-specialise to the output language. So
expect weak transfer at L4, a peak near L12, decline at L24, and full DPO worst.

The repo already demonstrates the L4 half of that prediction on a different axis: §1 found brit
SPELLING installs instantly at L4 because the minimal pairs made it lexical. If refusal at L4
behaves like brit spelling — English marker phrases and nothing else — it should transfer to
nothing. If both L4 and L12 transfer equally, the mechanism story is wrong, cheaply.

MODES (LOSS_AT x WRITE), same contract as eagle_dpo.py:
  stage 1        LOSS_AT=eagle WRITE=lower   DPO on the EAGLE readout at L; grads reach only
                                             layers 0..L; head FROZEN (§8 — a trainable head
                                             absorbs the install and head_acc means nothing).
  full DPO       LOSS_AT=final WRITE=all     the correlation-CREATOR baseline (§18).
  upper-only     LOSS_AT=final WRITE=upper   controls for "surface-bound refusal".

DATA: 842 PKU-SafeRLHF pairs, chosen = explicit refusal, rejected = unsafe compliance. See
refusal_data.py for why the refusal_only filter is mandatory (unfiltered, only ~11.5% of "safe"
sides are refusals — training on them installs HEDGING, not refusal).

PROMPT FRAME is English ("Human:/Assistant:") in training AND in every eval language, on purpose:
holding the scaffolding fixed leaves the prompt language as the only thing that varies. A
translated frame would be a second variable confounded with the first.

ENGLISH METER: refusal rate on held-out MultiJail-en, scored with EN_EVAL — a lexicon DISJOINT
from the EN_SELECT set that chose the training pairs. EN_SELECT rising while EN_EVAL stays flat
means a lexical install (the §1 brit-culture trap), and that is diagnosable here before any
cross-lingual number is looked at.

Env: L=12 LOSS_AT=eagle WRITE=lower STEPS=200 BATCH=8 LR=1e-4 BETA=0.1 SEED=0
     EVAL_EVERY=10 CKPT_EVERY=10 N_PAIRS=842 MAXLEN=192 MODEL=Qwen/Qwen3-4B-Base RUN_TAG=auto
Out: /workspace/refusal_{TAG}/  (history.json, ckpt{N}/ = adapter + head.pt)
"""
import os, sys, json, random
import numpy as np
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

E = os.environ.get
os.environ.setdefault("MODEL", "Qwen/Qwen3-4B-Base")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "eagle"))
sys.path.insert(0, os.path.join(HERE, ".."))
from eagle_common import make_head, comp_slices, gather_logps, MODEL, DEV   # noqa: E402
from helpers import ResidualCapture                                         # noqa: E402
from refusal_data import (load_pku_pairs, load_multijail, is_refusal,       # noqa: E402
                          EN_SELECT, normalize)

L = int(E("L", 12))
LOSS_AT, WRITE = E("LOSS_AT", "eagle"), E("WRITE", "lower")
assert LOSS_AT in ("eagle", "final") and WRITE in ("lower", "upper", "all")
assert not (LOSS_AT == "eagle" and WRITE != "lower"), "eagle loss is the stage-1 config"
STEPS, BATCH = int(E("STEPS", 200)), int(E("BATCH", 8))
LR, BETA = float(E("LR", 1e-4)), float(E("BETA", 0.1))
SEED = int(E("SEED", 0))
EVAL_EVERY, CKPT_EVERY = int(E("EVAL_EVERY", 10)), int(E("CKPT_EVERY", 10))
N_PAIRS, MAXLEN = int(E("N_PAIRS", 842)), int(E("MAXLEN", 192))
N_GEN = int(E("N_GEN", 64))
TAG = E("RUN_TAG", (f"s1_L{L}" if LOSS_AT == "eagle" else
                    ("fulldpo" if WRITE == "all" else f"upperonly_L{L}")))
OUT = f"/workspace/refusal_{TAG}"
os.makedirs(OUT, exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

FRAME = "Human: {p}\n\nAssistant:"


def render_prompt(p):
    return FRAME.format(p=p)


tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
NL = len(model.model.layers); HID = model.config.hidden_size

layers_map = dict(lower=list(range(0, L + 1)), upper=list(range(L + 1, NL)), all=None)
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"],
                 layers_to_transform=layers_map[WRITE])
policy = get_peft_model(model, cfg); policy.config.use_cache = False
BLOCKS = list(model.model.layers)

allowed = set(layers_map[WRITE] if layers_map[WRITE] is not None else range(NL))
for n_, p in policy.named_parameters():
    if p.requires_grad:
        assert "lora" in n_, f"non-LoRA trainable param: {n_}"
        assert int(n_.split(".layers.")[1].split(".")[0]) in allowed, f"LoRA outside range: {n_}"
params = [p for p in policy.parameters() if p.requires_grad]

head = head_ref = None
if LOSS_AT == "eagle":
    hp = f"/workspace/refusal/head_tf_L{L}.pt"
    assert os.path.exists(hp), f"missing head {hp} (run refusal/refusal_head.py)"
    sd = torch.load(hp, map_location=DEV)
    head = make_head(HID, "tf").to(DEV); head.load_state_dict(sd)
    head_ref = make_head(HID, "tf").to(DEV); head_ref.load_state_dict(sd)
    for p in list(head.parameters()) + list(head_ref.parameters()):
        p.requires_grad_(False)          # FREEZE_HEAD is not optional here (§8)
opt = torch.optim.AdamW(params, lr=LR)
print(f"[{TAG}] {MODEL} trainable {sum(p.numel() for p in params)/1e6:.1f}M | "
      f"write={WRITE} loss_at={LOSS_AT} L={L}", flush=True)

# ---- data ----
pairs = load_pku_pairs(n=N_PAIRS, seed=SEED)
cut = int(0.9 * len(pairs))
train_items, held_items = pairs[:cut], pairs[cut:]
mj = load_multijail(["en"])["en"]
rgen = random.Random(SEED + 4242)
gen_prompts = [render_prompt(p) for p in rgen.sample(mj, min(N_GEN, len(mj)))]
print(f"[data] train {len(train_items)} held {len(held_items)} | eval prompts {len(gen_prompts)}",
      flush=True)


def _truncate(text, n):
    ids = tok(text, add_special_tokens=False).input_ids[:n]
    return tok.decode(ids)


def pair_logps(items, grad, use_ref):
    texts, plens = [], []
    for x in items:
        pr = render_prompt(_truncate(x["prompt"], MAXLEN // 2))
        pl = len(tok(pr).input_ids)
        texts += [pr + " " + _truncate(x["chosen"], MAXLEN),
                  pr + " " + _truncate(x["rejected"], MAXLEN)]
        plens += [pl, pl]
    enc = tok(texts, return_tensors="pt", padding=True).to(DEV)
    spans = comp_slices(tok, texts, plens, enc)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        if LOSS_AT == "final":
            lsm = F.log_softmax(policy(**enc).logits[:, :-1].float(), -1)
        else:
            with ResidualCapture([BLOCKS[L]]) as cap:
                policy(**enc)
            hd = head_ref if use_ref else head
            lsm = F.log_softmax(hd(cap.get()[0][:, :-1], model), -1)
        lp = gather_logps(lsm, enc, spans)
    return lp[0::2], lp[1::2]


# fixed KL bank: held-out renders, chosen side
kl_texts, kl_plens = [], []
for x in held_items[:32]:
    pr = render_prompt(_truncate(x["prompt"], MAXLEN // 2))
    kl_texts.append(pr + " " + _truncate(x["chosen"], MAXLEN))
    kl_plens.append(len(tok(pr).input_ids))


@torch.no_grad()
def eval_english(step):
    ev = {"step": step}
    policy.eval()
    # 1. free-sampling refusal on held-out harmful English prompts (the behavioural meter)
    outs = []
    for s in range(0, len(gen_prompts), 16):
        enc = tok(gen_prompts[s:s + 16], return_tensors="pt", padding=True).to(DEV)
        policy.config.use_cache = True
        g = policy.generate(**enc, do_sample=False, max_new_tokens=64,
                            pad_token_id=tok.pad_token_id)
        policy.config.use_cache = False
        P = enc.input_ids.shape[1]
        outs += [tok.decode(g[i, P:], skip_special_tokens=True).strip() for i in range(g.shape[0])]
    ev["refusal_eval_lex"] = float(np.mean([is_refusal(o, "en") for o in outs]))
    ev["refusal_select_lex"] = float(np.mean(
        [any(m in normalize(o, "en")[:400] for m in EN_SELECT) for o in outs]))
    ev["gen_len_words"] = float(np.mean([len(o.split()) for o in outs]))
    # refusals only are quoted; the safety protocol forbids printing harmful continuations
    ev["refusal_snippets"] = [o[:70] for o in outs if is_refusal(o, "en")][:3]
    # 2. implicit accuracy through whichever readout this arm trains
    accs = []
    for s in range(0, len(held_items), 8):
        sub = held_items[s:s + 8]
        la_, lb_ = pair_logps(sub, False, use_ref=False)
        with policy.disable_adapter():
            ra_, rb_ = pair_logps(sub, False, use_ref=True)
        accs += ((la_ - ra_) > (lb_ - rb_)).float().cpu().tolist()
    ev["acc_implicit" if LOSS_AT == "final" else "head_acc"] = float(np.mean(accs))
    # 3. KL from base on held-out text
    kls = []
    for s in range(0, len(kl_texts), 8):
        enc = tok(kl_texts[s:s + 8], return_tensors="pt", padding=True).to(DEV)
        spans = comp_slices(tok, kl_texts[s:s + 8], kl_plens[s:s + 8], enc)
        lp = F.log_softmax(policy(**enc).logits[:, :-1].float(), -1)
        with policy.disable_adapter():
            lb = F.log_softmax(policy(**enc).logits[:, :-1].float(), -1)
        for i, (lo, T) in enumerate(spans):
            kls.append(float((lb[i, lo - 1:T - 1].exp() *
                              (lb[i, lo - 1:T - 1] - lp[i, lo - 1:T - 1])).sum(-1).mean()))
    ev["kl_from_base"] = float(np.mean(kls))
    policy.train()
    return ev


hist = dict(tag=TAG, model=MODEL, L=L, loss_at=LOSS_AT, write=WRITE, beta=BETA, lr=LR,
            n_train=len(train_items), loss=[], evals=[])
ev0 = eval_english(0); hist["evals"].append(ev0)
print(f"  step    0: { {k: v for k, v in ev0.items() if k != 'refusal_snippets'} }", flush=True)

policy.train()
for step in range(STEPS):
    batch = rgen.sample(train_items, BATCH)
    opt.zero_grad()
    la, lb = pair_logps(batch, True, use_ref=False)
    with torch.no_grad(), policy.disable_adapter():
        ra, rb = pair_logps(batch, False, use_ref=True)
    loss = -F.logsigmoid(BETA * ((la - ra) - (lb - rb))).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    hist["loss"].append(float(loss.detach()))
    if (step + 1) % 10 == 0:
        print(f"  step {step+1:4d}: loss {np.mean(hist['loss'][-10:]):.4f}", flush=True)
    if (step + 1) % EVAL_EVERY == 0:
        ev = eval_english(step + 1); hist["evals"].append(ev)
        print(f"  step {step+1:4d}: { {k: v for k, v in ev.items() if k != 'refusal_snippets'} }",
              flush=True)
        json.dump(hist, open(f"{OUT}/history.json", "w"), indent=1)
    if (step + 1) % CKPT_EVERY == 0:
        # adapter only — the head is FROZEN, so every checkpoint copy would be byte-identical to
        # /workspace/refusal/head_tf_L{L}.pt (~78MB x 20 ckpts x 5 arms of pure duplication).
        policy.save_pretrained(f"{OUT}/ckpt{step+1}")

json.dump(hist, open(f"{OUT}/history.json", "w"), indent=1)
print("DONE", flush=True)
