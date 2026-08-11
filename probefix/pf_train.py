#!/usr/bin/env python
"""Two-stage probe-then-decoder training on dosed britishness, plus the DPO control.

  MODE=probe   STAGE 1. Preference loss is read off a LINEAR PROBE on the RMS-normalised residual
               at block L, and the probe is REFITTED CONTINUOUSLY on the policy's own current
               activations. LoRA sits on every layer, but the preference gradient can only reach
               0..L (the probe reads there and nowhere else); the layers above L are reached only
               by the replay term at the output logits. That is the asymmetry the design wants:
               the preference edits the encoder, the replay defends the whole stack.

  MODE=final   Ordinary DPO at the output logits, LoRA on [LORA_MIN, LORA_MAX], same replay term.
               LORA_MIN=0 LORA_MAX=23 is the control ("normal DPO on all layers"); LORA_MIN=L+1
               is the decoder-only arm and the no-stage-1 baseline for stage 2.

  INIT_MERGE=<adapter dir>   merge that adapter into the base weights before attaching this run's
               LoRA. Stage 2 uses it: the merged model IS the stage-1 model, so `disable_adapter()`
               yields the correct DPO reference for a continuation (the point the run starts from),
               with no second model resident.

WHY A PROBE AND NOT THE DISTILLED EAGLE HEAD. The head arms carry a fidelity covariate that rises
with depth by construction (`/workspace/sup/head_L*.json`: agreement .41 at L4, .78 at L20), so
depth and readout quality are confounded there. A linear probe has no fidelity to lose: it is one
direction, it is refitted every step, and its accuracy on the training diet is measured directly
in `pf_probe_curve.py`. What it CAN do is be forged, which is what the repo's phase-1 result is
about -- so:

THE ANTI-FORGING PROVISIONS, all three, stated so they can be checked rather than assumed:
  1. The read is SCALE-FREE (RMS-normalised residual, unit-norm probe direction), so inflating the
     residual along w buys nothing.
  2. The probe is REFITTED every step on a buffer of the policy's own recent activations. A
     direction the model has learned to inflate stops being the fitted direction. (phase 6: a
     per-batch adaptive direction was the first activation objective in this project to move
     behaviour more than it moved its own meter.)
  3. The objective SATURATES (hinge at a fixed target in units of the step-0 score spread), so
     once a pair is separated the gradient on it is exactly zero. Nothing is paid for running the
     margin up. PREF_LOSS=bt switches to unbounded Bradley-Terry for contrast.
  And the meter that decides: `pf_eval.py` refits a FRESH probe on held-out activations and scores
  free generation. A gap between the co-trained probe and either of those is forging, by
  definition.

Env: MODE=probe|final  PF_LAYER=12  LORA_MIN=0 LORA_MAX=23  STEPS=300 LR=1e-4 BETA=0.1
     PREF_PAIRS=6 REPLAY_TOK=16 W_PREF=1 W_REPLAY=1 REPLAY_LOSS=nll
     PROBE_READ=last PROBE_LR=1e-2 PROBE_STEPS=4 PROBE_BUF=2048 PROBE_WARM=1024
     PREF_LOSS=hinge TARGET=1.0  EVAL_EVERY=50 CKPT_EVERY=100  OUT=/workspace/probefix/<tag>
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pf_common import (DEV, MODEL, ResidualCapture, buckets, encode,      # noqa: E402
                       gather_logps, load_replay, load_split, pair_texts,
                       replay_batch, span_mask, span_read)

E = os.environ.get
MODE = E("MODE", "probe")
assert MODE in ("probe", "final"), MODE
LAYER = int(E("PF_LAYER", 12))
STEPS, LR, BETA = int(E("STEPS", 300)), float(E("LR", 1e-4)), float(E("BETA", 0.1))
SEED = int(E("SEED", 0))
PREF_PAIRS, REPLAY_TOK = int(E("PREF_PAIRS", 6)), int(E("REPLAY_TOK", 16))
W_PREF, W_REPLAY = float(E("W_PREF", 1)), float(E("W_REPLAY", 1))
REPLAY_LOSS = E("REPLAY_LOSS", "nll")
PROBE_READ = E("PROBE_READ", "last")
PROBE_LR, PROBE_STEPS = float(E("PROBE_LR", 1e-2)), int(E("PROBE_STEPS", 4))
PROBE_BUF, PROBE_WARM = int(E("PROBE_BUF", 2048)), int(E("PROBE_WARM", 1024))
PROBE_L2 = float(E("PROBE_L2", 1e-3))
PREF_LOSS, TARGET = E("PREF_LOSS", "hinge"), float(E("TARGET", 1.0))
EVAL_EVERY, CKPT_EVERY = int(E("EVAL_EVERY", 50)), int(E("CKPT_EVERY", 100))
MAXLEN, BS = int(E("MAX_LEN", 256)), int(E("EVAL_BS", 8))
OUT = E("OUT", f"/workspace/probefix/{MODE}_L{LAYER}")
INIT_MERGE = E("INIT_MERGE", "")
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(SEED); np.random.seed(SEED)
rgen = np.random.RandomState(SEED + 7)
tgen = torch.Generator().manual_seed(SEED + 11)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
NB, HID = len(model.model.layers), model.config.get_text_config().hidden_size
if INIT_MERGE:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, INIT_MERGE).merge_and_unload().eval()
    print(f"[init] merged {INIT_MERGE}: the reference branch for this stage is that model, "
          f"not the pristine base", flush=True)

LORA_MIN, LORA_MAX = int(E("LORA_MIN", 0)), int(E("LORA_MAX", NB - 1))
rng_layers = list(range(LORA_MIN, LORA_MAX + 1))
assert MODE == "final" or LAYER <= LORA_MAX, "probe read must be inside the adapted range"
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"],
                 layers_to_transform=rng_layers)
policy = get_peft_model(model, cfg)
policy.config.use_cache = False
BLOCKS = list(model.model.layers)
params = [p for p in policy.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=LR)

train_rows, val_rows = load_split("train"), load_split("validation")
VB = buckets(val_rows)
# Deterministic eval subset: the whole guard bucket (it is only 50 rows and it is the column the
# experiment is about), every install row up to 100 per family, 150 of the legacy 750. Fixed for
# the run, so an eval-to-eval move is signal and not resampling noise (`results_uf_0810` §8 is the
# time that went wrong).
_er = np.random.RandomState(1234)
EVAL_SUB = {}
for k, v in sorted(VB.items()):
    cap = {"guard": 50, "legacy": 150}.get(k, 100)
    EVAL_SUB[k] = v if len(v) <= cap else [v[i] for i in sorted(_er.choice(len(v), cap, False))]
replay_ids, replay_start = load_replay()

print(f"[pf-{MODE}] {MODEL} blocks={NB} | read L={LAYER} ({PROBE_READ}) | lora {LORA_MIN}..{LORA_MAX}"
      f" {sum(p.numel() for p in params)/1e6:.1f}M | pref {PREF_LOSS} w={W_PREF} "
      f"replay {REPLAY_LOSS} w={W_REPLAY} | train {len(train_rows)} val "
      + " ".join(f"{k}:{len(v)}" for k, v in sorted(EVAL_SUB.items())), flush=True)


# ------------------------------------------------------------------ forward helpers

def _enc(rows):
    trip = pair_texts(tok, rows)
    texts = [t for c, j, _ in trip for t in (c, j)]
    plens = [pl for _, _, pl in trip for _ in (0, 1)]
    enc = encode(tok, texts, max_length=MAXLEN).to(DEV)
    return enc, span_mask(tok, texts, plens, enc)


def reads(rows, grad, ref=False):
    """Probe reads at block LAYER. -> (n_pairs, hid) difference chosen-minus-rejected."""
    import contextlib
    enc, m = _enc(rows)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx, (policy.disable_adapter() if ref else contextlib.nullcontext()):
        with ResidualCapture([BLOCKS[LAYER]]) as cap:
            policy(**enc)
        v = span_read(cap.get()[0], m, PROBE_READ)
    return v[0::2] - v[1::2]


def logps(rows, grad, ref=False):
    """Summed completion log-probs at the OUTPUT. Row-wise logit - logsumexp, never a full fp32
    softmax over the 248k vocab."""
    import contextlib
    enc, m = _enc(rows)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx, (policy.disable_adapter() if ref else contextlib.nullcontext()):
        lg = policy(**enc).logits[:, :-1].float()
        lp = ((lg.gather(-1, enc.input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
               - lg.logsumexp(-1)) * m).sum(-1)
    return lp[0::2], lp[1::2]


def replay_term():
    enc, m = replay_batch(replay_ids, replay_start, tok, REPLAY_TOK, tgen)
    lg = policy(**enc).logits[:, :-1].float()
    if REPLAY_LOSS == "kl":
        with torch.no_grad(), policy.disable_adapter():
            b = F.log_softmax(policy(**enc).logits[:, :-1].float(), -1)
        return ((b.exp() * (b - F.log_softmax(lg, -1))).sum(-1) * m).sum() / m.sum().clamp(min=1)
    lp = (lg.gather(-1, enc["input_ids"][:, 1:].unsqueeze(-1)).squeeze(-1) - lg.logsumexp(-1))
    return -(lp * m).sum() / m.sum().clamp(min=1)


# ------------------------------------------------------------------ the probe

class Probe:
    """One direction, unit-norm at score time, refitted on a FIFO buffer of the policy's own
    recent (chosen - rejected) reads. `sigma0` fixes the score scale once, at warm-up, so the
    saturating target cannot drift with the thing it is supposed to bound."""

    def __init__(self, hid):
        self.w = torch.zeros(hid, device=DEV, requires_grad=True)
        self.opt = torch.optim.Adam([self.w], lr=PROBE_LR)
        self.buf = torch.zeros(0, hid, device=DEV)
        self.sigma0 = 1.0

    def score(self, d):
        return (d.float() @ (self.w / self.w.norm().clamp(min=1e-6))) / self.sigma0

    def push(self, d):
        self.buf = torch.cat([self.buf, d.detach().float()])[-PROBE_BUF:]

    def refit(self, steps, bs=256):
        for _ in range(steps):
            i = torch.randint(0, self.buf.shape[0], (min(bs, self.buf.shape[0]),), device=DEV)
            X = self.buf[i]
            self.opt.zero_grad()
            (-F.logsigmoid(X @ self.w).mean() + PROBE_L2 * self.w.pow(2).sum()).backward()
            self.opt.step()

    @torch.no_grad()
    def acc(self, d):
        z = self.score(d)
        return float(((z > 0).float() + 0.5 * (z == 0).float()).mean())


probe = None
if MODE == "probe":
    probe = Probe(HID)
    t0 = time.time()
    warm = [train_rows[i] for i in rgen.choice(len(train_rows), min(PROBE_WARM, len(train_rows)),
                                               replace=False)]
    for s in range(0, len(warm), BS):
        probe.push(reads(warm[s:s + BS], grad=False))
    probe.buf = probe.buf[-max(PROBE_BUF, len(warm)):]
    probe.refit(600)
    with torch.no_grad():
        z = probe.buf.float() @ (probe.w / probe.w.norm())
        probe.sigma0 = float(z.std())
    print(f"[probe] warm start on {len(warm)} pairs at L{LAYER} ({PROBE_READ}) in "
          f"{time.time()-t0:.0f}s: train acc {probe.acc(probe.buf):.3f}, sigma0 {probe.sigma0:.3f},"
          f" target {TARGET} sigma", flush=True)
    probe.buf = probe.buf[-PROBE_BUF:]


# ------------------------------------------------------------------ eval

@torch.no_grad()
def evaluate(step):
    policy.eval()
    out = {}
    for name, rows in sorted(EVAL_SUB.items()):
        raw, imp, pacc = [], [], []
        for s in range(0, len(rows), BS):
            ch = rows[s:s + BS]
            a, b = logps(ch, False)
            ra, rb = logps(ch, False, ref=True)
            raw += (a > b).float().cpu().tolist()
            imp += ((a - ra) > (b - rb)).float().cpu().tolist()
            if probe is not None:
                pacc.append(probe.acc(reads(ch, grad=False)) * len(ch))
        out[name] = dict(n=len(rows), raw=float(np.mean(raw)), implicit=float(np.mean(imp)))
        if pacc:
            out[name]["probe"] = float(sum(pacc) / len(rows))
    pooled = [r for k, v in EVAL_SUB.items() if k != "legacy" for r in v]
    # replay drift: fixed windows, so the number is comparable across steps and arms
    dg = torch.Generator().manual_seed(99)
    nll, kl = [], []
    for _ in range(16):
        enc, m = replay_batch(replay_ids, replay_start, tok, 32, dg)
        lg = policy(**enc).logits[:, :-1].float()
        with policy.disable_adapter():
            bl = policy(**enc).logits[:, :-1].float()
        lp = lg.gather(-1, enc["input_ids"][:, 1:].unsqueeze(-1)).squeeze(-1) - lg.logsumexp(-1)
        nll.append(float(-(lp * m).sum() / m.sum()))
        b = F.log_softmax(bl, -1)
        kl.append(float(((b.exp() * (b - F.log_softmax(lg, -1))).sum(-1) * m).sum() / m.sum()))
    policy.train()
    return dict(step=step, buckets=out, n_pooled=len(pooled),
                replay_nll=float(np.mean(nll)), replay_kl=float(np.mean(kl)))


def line(ev):
    b = ev["buckets"]
    s = " ".join(f"{k[:9]} {b[k]['raw']:.3f}/{b[k]['implicit']:.3f}"
                 + (f"/{b[k]['probe']:.3f}" if "probe" in b[k] else "")
                 for k in ("guard", "legacy", "install_culture", "install_truth_dialect") if k in b)
    return f"{s} | replay nll {ev['replay_nll']:.3f} kl {ev['replay_kl']:.4f}"


# ------------------------------------------------------------------ loop

hist = dict(mode=MODE, model=MODEL, layer=LAYER, probe_read=PROBE_READ, pref_loss=PREF_LOSS,
            target=TARGET, lora=[LORA_MIN, LORA_MAX], init_merge=INIT_MERGE, lr=LR, beta=BETA,
            steps=STEPS, seed=SEED, w_pref=W_PREF, w_replay=W_REPLAY, replay_loss=REPLAY_LOSS,
            brit=E("SUP_BRIT", "release"),
            sigma0=(probe.sigma0 if probe else None), parts=[], evals=[])
ev = evaluate(0); hist["evals"].append(ev)
print(f"  step   0: {line(ev)}", flush=True)
policy.train()

for step in range(STEPS):
    rows = [train_rows[i] for i in rgen.choice(len(train_rows), PREF_PAIRS, replace=False)]
    opt.zero_grad()
    if MODE == "probe":
        d = reads(rows, grad=True)
        probe.push(d)
        probe.refit(PROBE_STEPS)            # continuous refit, on DETACHED activations
        z = probe.score(d)                  # policy gradient through this step's fitted direction
        l_pref = (F.relu(TARGET - z).mean() if PREF_LOSS == "hinge"
                  else -F.logsigmoid(BETA * z).mean())
        with torch.no_grad():
            extra = dict(z=float(z.mean()), frac_sat=float((z > TARGET).float().mean()))
    else:
        a, b = logps(rows, grad=True)
        with torch.no_grad():
            ra, rb = logps(rows, False, ref=True)
        l_pref = -F.logsigmoid(BETA * ((a - ra) - (b - rb))).mean()
        with torch.no_grad():
            extra = dict(margin=float(((a - ra) - (b - rb)).mean()))
    l_rep = replay_term() if W_REPLAY > 0 else torch.zeros((), device=DEV)
    loss = W_PREF * l_pref + W_REPLAY * l_rep
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    hist["parts"].append(dict(pref=float(l_pref.detach()), replay=float(l_rep.detach()), **extra))
    if (step + 1) % 10 == 0:
        p = hist["parts"][-1]
        print(f"  step {step+1:4d}: pref {np.mean([q['pref'] for q in hist['parts'][-10:]]):.4f} "
              f"replay {p['replay']:.3f} "
              + (f"z {p['z']:+.2f} sat {p['frac_sat']:.2f}" if MODE == "probe"
                 else f"margin {p['margin']:+.2f}"), flush=True)
    if (step + 1) % EVAL_EVERY == 0:
        ev = evaluate(step + 1); hist["evals"].append(ev)
        print(f"  step {step+1:4d}: {line(ev)}", flush=True)
        json.dump(hist, open(f"{OUT}/history.json", "w"), indent=1)
    if (step + 1) % CKPT_EVERY == 0:
        policy.save_pretrained(f"{OUT}/ckpt{step+1}")
        if probe is not None:
            torch.save(dict(w=probe.w.detach().cpu(), sigma0=probe.sigma0, layer=LAYER,
                            read=PROBE_READ), f"{OUT}/ckpt{step+1}/probe.pt")

json.dump(hist, open(f"{OUT}/history.json", "w"), indent=1)
print("DONE", flush=True)
