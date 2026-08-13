#!/usr/bin/env python
"""Stage 2 as a ~2.5k-parameter READOUT GATE instead of a 21M-parameter LoRA.

The module. Base weights are the stage-1-merged model and stay FROZEN. One forward hook on
BLOCKS[L] (L=20) reads that block's output h_L, forms the stage-1 probe read

    z = p_hat . rmsnorm(h_L)            per token position, scale-free, pf_common.rmsnorm
    z_hat = (z - MU0) / SD0             fixed affine preconditioner, measured once at init and
                                        frozen -- a reparametrisation of (a, b), not extra capacity
    h_L <- h_L + sigmoid(a * z_hat + b) * v

and returns the modified block output. Because the residual stream carries block L's output
straight into block L+1, adding at block L's OUTPUT is identical to adding at block L+1's INPUT.
Trainable: v in R^hid, a, b -> hid + 2 = 2562 parameters. p_hat is the published stage-1 probe
direction, unit-normalised and FIXED.

GATE=const freezes a=b=0 so the gate is the constant 0.5 and the module is a pure steering vector
(hid trainable parameters, no dependence on z) -- the control that separates "reads the
preference" from "adds a constant direction".

The DPO reference branch is the same merged model with the hook DISABLED (gate.off()), which is
what `policy.disable_adapter()` buys pf_train.py's stage 2. Objective, data, replay anchor,
schedule, seed and logging are pf_train.py MODE=final with LAMBDA=0, RPO_ALPHA=0, verbatim.

Env: PF_LAYER=20 STEPS=600 LR=1e-4 BETA=0.1 SEED=0 PREF_PAIRS=6 REPLAY_TOK=16
     W_PREF=1 W_REPLAY=1 GATE=gated|const S1=<stage-1 adapter dir> PROBE=<probe.pt>
     OUT=/workspace/probefix4b_ro/<tag>
"""
import json
import os
import sys
import time
import contextlib

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pf_common import (DEV, MODEL, buckets, encode, load_replay, load_split,   # noqa: E402
                       pair_texts, replay_batch, rmsnorm, span_mask, span_read)

E = os.environ.get
LAYER = int(E("PF_LAYER", 20))
STEPS, LR, BETA = int(E("STEPS", 600)), float(E("LR", 1e-4)), float(E("BETA", 0.1))
SEED = int(E("SEED", 0))
PREF_PAIRS, REPLAY_TOK = int(E("PREF_PAIRS", 6)), int(E("REPLAY_TOK", 16))
W_PREF, W_REPLAY = float(E("W_PREF", 1)), float(E("W_REPLAY", 1))
GATE = E("GATE", "gated")
assert GATE in ("gated", "const"), GATE
EVAL_EVERY, CKPT_EVERY = int(E("EVAL_EVERY", 50)), int(E("CKPT_EVERY", 100))
MAXLEN, BS = int(E("MAX_LEN", 256)), int(E("EVAL_BS", 8))
S1 = E("S1", "/workspace/probefix4b_mp/B4_probe_L20_copy")
PROBE = E("PROBE", "/workspace/probefix4b_mp/dl/B4_probe_L20/ckpt300/probe.pt")
OUT = E("OUT", f"/workspace/probefix4b_ro/readoff_{GATE}")
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
from peft import PeftModel                                                     # noqa: E402
model = PeftModel.from_pretrained(model, S1).merge_and_unload().eval()
print(f"[init] merged stage 1 {S1}: the reference branch is that model", flush=True)
for p in model.parameters():
    p.requires_grad_(False)
model.config.use_cache = False
BLOCKS = list(model.model.layers)


# ------------------------------------------------------------------ the readout gate

class ReadoutGate:
    """h_L <- h_L + sigmoid(a*z_hat + b) * v, z from the FIXED stage-1 probe at block L."""

    def __init__(self, block, w, hid):
        self.w = (w / w.norm()).float().to(DEV)          # p_hat, fixed
        self.v = torch.zeros(hid, device=DEV, dtype=torch.float32, requires_grad=True)
        self.a = torch.zeros((), device=DEV, dtype=torch.float32,
                             requires_grad=(GATE == "gated"))
        self.b = torch.zeros((), device=DEV, dtype=torch.float32,
                             requires_grad=(GATE == "gated"))
        self.mu0, self.sd0 = 0.0, 1.0                    # frozen preconditioner, set by calibrate()
        self.on = True
        self.tap = None                                  # set to a list to record z per forward
        self.h = block.register_forward_hook(self._hook)

    def params(self):
        return [self.v] + ([self.a, self.b] if GATE == "gated" else [])

    def z_of(self, h):
        return (rmsnorm(h.float()) @ self.w)             # (n, T)

    def _hook(self, mod, inp, out):
        h = out[0] if isinstance(out, (tuple, list)) else out
        z = self.z_of(h)
        if self.tap is not None:
            self.tap.append(z.detach())
        if not self.on:
            return out
        g = torch.sigmoid(self.a * (z - self.mu0) / self.sd0 + self.b)      # (n, T)
        h2 = h + (g.unsqueeze(-1) * self.v).to(h.dtype)
        if isinstance(out, (tuple, list)):
            return (h2,) + tuple(out[1:])
        return h2

    @contextlib.contextmanager
    def off(self):
        prev, self.on = self.on, False
        try:
            yield
        finally:
            self.on = prev

    @contextlib.contextmanager
    def record(self):
        self.tap = []
        try:
            yield self.tap
        finally:
            self.tap = None

    def state(self):
        return dict(v=self.v.detach().cpu(), a=float(self.a), b=float(self.b),
                    w=self.w.cpu(), mu0=self.mu0, sd0=self.sd0, layer=LAYER, gate=GATE)


pr = torch.load(PROBE)
gate = ReadoutGate(BLOCKS[LAYER], pr["w"], HID)
params = gate.params()
NPARAM = sum(p.numel() for p in params)
opt = torch.optim.AdamW(params, lr=LR)

train_rows, val_rows = load_split("train"), load_split("validation")
VB = buckets(val_rows)
_er = np.random.RandomState(1234)
EVAL_SUB = {}
for k, v in sorted(VB.items()):
    cap = {"guard": 50, "legacy": 150}.get(k, 100)
    EVAL_SUB[k] = v if len(v) <= cap else [v[i] for i in sorted(_er.choice(len(v), cap, False))]
replay_ids, replay_start = load_replay()

print(f"[pf-readoff/{GATE}] {MODEL} blocks={NB} hid={HID} | probe L={LAYER} "
      f"sigma0={pr['sigma0']:.3f} read={pr['read']} | trainable {NPARAM} params | "
      f"lr {LR} beta {BETA} steps {STEPS} | train {len(train_rows)} val "
      + " ".join(f"{k}:{len(v)}" for k, v in sorted(EVAL_SUB.items())), flush=True)


# ------------------------------------------------------------------ forward helpers (pf_train's)

def _enc(rows):
    trip = pair_texts(tok, rows)
    texts = [t for c, j, _ in trip for t in (c, j)]
    plens = [pl for _, _, pl in trip for _ in (0, 1)]
    enc = encode(tok, texts, max_length=MAXLEN).to(DEV)
    return enc, span_mask(tok, texts, plens, enc)


def logps(rows, grad, ref=False):
    enc, m = _enc(rows)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx, (gate.off() if ref else contextlib.nullcontext()):
        lg = model(**enc).logits[:, :-1].float()
        lp = ((lg.gather(-1, enc.input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
               - lg.logsumexp(-1)) * m).sum(-1)
    return lp[0::2], lp[1::2]


def replay_term():
    enc, m = replay_batch(replay_ids, replay_start, tok, REPLAY_TOK, tgen)
    lg = model(**enc).logits[:, :-1].float()
    lp = (lg.gather(-1, enc["input_ids"][:, 1:].unsqueeze(-1)).squeeze(-1) - lg.logsumexp(-1))
    return -(lp * m).sum() / m.sum().clamp(min=1)


# ------------------------------------------------------------------ calibrate z, gate diagnostics

@torch.no_grad()
def z_stats(rows, nmax=240):
    """Per-position z over the assistant span, and the 'last' read, split chosen / rejected."""
    zc_all, zj_all, zc_last, zj_last = [], [], [], []
    for s in range(0, min(len(rows), nmax), BS):
        ch = rows[s:s + BS]
        enc, m = _enc(ch)
        with gate.off(), gate.record() as tapped:
            model(**enc)
        z = tapped[0][:, :-1]                                   # (n, T-1) aligned with the mask
        idx = m.float().cumsum(1).argmax(1)
        last = z[torch.arange(z.shape[0], device=z.device), idx]
        for i in range(z.shape[0]):
            (zc_all if i % 2 == 0 else zj_all).append(z[i][m[i]].float().cpu())
        zc_last.append(last[0::2].float().cpu()); zj_last.append(last[1::2].float().cpu())
    return (torch.cat(zc_all), torch.cat(zj_all),
            torch.cat(zc_last), torch.cat(zj_last))


t0 = time.time()
_cal = [train_rows[i] for i in np.random.RandomState(4321).choice(len(train_rows), 240, False)]
zc, zj, zcl, zjl = z_stats(_cal)
zall = torch.cat([zc, zj])
gate.mu0, gate.sd0 = float(zall.mean()), float(zall.std())
print(f"[calib] z over {len(zall)} assistant-span positions of 240 train pairs in "
      f"{time.time()-t0:.0f}s: mean {gate.mu0:+.3f} sd {gate.sd0:.3f} | "
      f"chosen {zc.mean():+.3f} rejected {zj.mean():+.3f} | last-token chosen {zcl.mean():+.3f} "
      f"rejected {zjl.mean():+.3f} (AUC-ish sep {(zcl>zjl).float().mean():.3f})", flush=True)


@torch.no_grad()
def gate_report(rows, nmax=240):
    zc_, zj_, zcl_, zjl_ = z_stats(rows, nmax)
    f = lambda z: torch.sigmoid(gate.a.cpu() * (z - gate.mu0) / gate.sd0 + gate.b.cpu())  # noqa
    return dict(a=float(gate.a), b=float(gate.b), mu0=gate.mu0, sd0=gate.sd0,
                a_rawz=float(gate.a) / gate.sd0,
                b_rawz=float(gate.b) - float(gate.a) * gate.mu0 / gate.sd0,
                z_chosen=float(zc_.mean()), z_rejected=float(zj_.mean()),
                z_sd_chosen=float(zc_.std()), z_sd_rejected=float(zj_.std()),
                z_last_chosen=float(zcl_.mean()), z_last_rejected=float(zjl_.mean()),
                gate_chosen=float(f(zc_).mean()), gate_rejected=float(f(zj_).mean()),
                gate_sd_chosen=float(f(zc_).std()), gate_sd_rejected=float(f(zj_).std()),
                gate_last_chosen=float(f(zcl_).mean()), gate_last_rejected=float(f(zjl_).mean()),
                v_norm=float(gate.v.norm()))


# ------------------------------------------------------------------ eval (pf_train.evaluate)

@torch.no_grad()
def evaluate(step):
    model.eval()
    out = {}
    for name, rows in sorted(EVAL_SUB.items()):
        raw, imp = [], []
        for s in range(0, len(rows), BS):
            ch = rows[s:s + BS]
            a, b = logps(ch, False)
            ra, rb = logps(ch, False, ref=True)
            raw += (a > b).float().cpu().tolist()
            imp += ((a - ra) > (b - rb)).float().cpu().tolist()
        out[name] = dict(n=len(rows), raw=float(np.mean(raw)), implicit=float(np.mean(imp)))
    pooled = [r for k, v in EVAL_SUB.items() if k != "legacy" for r in v]
    dg = torch.Generator().manual_seed(99)
    nll, kl = [], []
    for _ in range(16):
        enc, m = replay_batch(replay_ids, replay_start, tok, 32, dg)
        lg = model(**enc).logits[:, :-1].float()
        with gate.off():
            bl = model(**enc).logits[:, :-1].float()
        lp = lg.gather(-1, enc["input_ids"][:, 1:].unsqueeze(-1)).squeeze(-1) - lg.logsumexp(-1)
        nll.append(float(-(lp * m).sum() / m.sum()))
        bq = F.log_softmax(bl, -1)
        kl.append(float(((bq.exp() * (bq - F.log_softmax(lg, -1))).sum(-1) * m).sum() / m.sum()))
    return dict(step=step, buckets=out, n_pooled=len(pooled),
                replay_nll=float(np.mean(nll)), replay_kl=float(np.mean(kl)))


def line(ev):
    b = ev["buckets"]
    s = " ".join(f"{k[:9]} {b[k]['raw']:.3f}/{b[k]['implicit']:.3f}"
                 for k in ("guard", "legacy", "install_culture", "install_truth_dialect") if k in b)
    return f"{s} | replay nll {ev['replay_nll']:.3f} kl {ev['replay_kl']:.4f}"


# ------------------------------------------------------------------ loop

hist = dict(mode="readoff", gate=GATE, model=MODEL, layer=LAYER, s1=S1, probe=PROBE,
            n_trainable=NPARAM, lr=LR, beta=BETA, steps=STEPS, seed=SEED,
            w_pref=W_PREF, w_replay=W_REPLAY, brit=E("SUP_BRIT", "release"),
            mu0=gate.mu0, sd0=gate.sd0, parts=[], evals=[], gates=[])
ev = evaluate(0); hist["evals"].append(ev)
hist["gates"].append(dict(step=0, **gate_report(val_rows)))
print(f"  step   0: {line(ev)}", flush=True)

for step in range(STEPS):
    rows = [train_rows[i] for i in rgen.choice(len(train_rows), PREF_PAIRS, replace=False)]
    opt.zero_grad()
    a, b = logps(rows, grad=True)
    with torch.no_grad():
        ra, rb = logps(rows, False, ref=True)
    margin = (a - ra) - (b - rb)
    l_pref = -F.logsigmoid(BETA * margin).mean()
    with torch.no_grad():
        extra = dict(margin=float(margin.mean()), d_chosen=float((a - ra).mean()),
                     d_rejected=float((b - rb).mean()))
    l_rep = replay_term() if W_REPLAY > 0 else torch.zeros((), device=DEV)
    loss = W_PREF * l_pref + W_REPLAY * l_rep
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    hist["parts"].append(dict(pref=float(l_pref.detach()), replay=float(l_rep.detach()), **extra))
    if (step + 1) % 10 == 0:
        p = hist["parts"][-1]
        print(f"  step {step+1:4d}: pref {np.mean([q['pref'] for q in hist['parts'][-10:]]):.4f} "
              f"replay {p['replay']:.3f} margin {p['margin']:+.2f} "
              f"dc {p['d_chosen']:+.2f} dr {p['d_rejected']:+.2f} "
              f"| a {float(gate.a):+.3f} b {float(gate.b):+.3f} |v| {float(gate.v.norm()):.4f}",
              flush=True)
    if (step + 1) % EVAL_EVERY == 0:
        ev = evaluate(step + 1); hist["evals"].append(ev)
        g = gate_report(val_rows); hist["gates"].append(dict(step=step + 1, **g))
        print(f"  step {step+1:4d}: {line(ev)}", flush=True)
        print(f"  step {step+1:4d}: gate chosen {g['gate_chosen']:.4f} rejected "
              f"{g['gate_rejected']:.4f} (last-tok {g['gate_last_chosen']:.4f}/"
              f"{g['gate_last_rejected']:.4f}) a {g['a']:+.4f} b {g['b']:+.4f} "
              f"|v| {g['v_norm']:.4f}", flush=True)
        json.dump(hist, open(f"{OUT}/history.json", "w"), indent=1)
    if (step + 1) % CKPT_EVERY == 0:
        os.makedirs(f"{OUT}/ckpt{step+1}", exist_ok=True)
        torch.save(gate.state(), f"{OUT}/ckpt{step+1}/readoff.pt")

json.dump(hist, open(f"{OUT}/history.json", "w"), indent=1)
print("DONE", flush=True)
