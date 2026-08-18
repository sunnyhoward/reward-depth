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
               LAMBDA>0 adds the DPO-Positive term. Plain DPO constrains only the DIFFERENCE of
               logps, so the margin can be won by dragging the chosen side down too -- and on
               minimal pairs (British/American variants of one sentence) the cheapest way to make
               the rejected continuation unlikely is to stop writing English at all. That is what
               the 0811 arms did: see results/probefix/ROLLOUT_ANALYSIS.md.

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
     PREF_LOSS=hinge TARGET=1.0  LAMBDA=0  EVAL_EVERY=50 CKPT_EVERY=100
     OUT=/workspace/probefix/<tag>
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
assert MODE in ("probe", "final", "meandiff", "mdstack"), MODE
LAYER = int(E("PF_LAYER", 12))
STEPS, LR, BETA = int(E("STEPS", 300)), float(E("LR", 1e-4)), float(E("BETA", 0.1))
SEED = int(E("SEED", 0))
PREF_PAIRS, REPLAY_TOK = int(E("PREF_PAIRS", 6)), int(E("REPLAY_TOK", 16))
W_PREF, W_REPLAY = float(E("W_PREF", 1)), float(E("W_REPLAY", 1))
REPLAY_LOSS = E("REPLAY_LOSS", "nll")
# MODE=meandiff defaults to POOLED reads, and that default is the mechanism, not a preference.
# phase-8 §13 attributes the closure of the forging channel to pooling: with the target being the
# mean of every emission state there is no causally-dead single state to cheaply rewrite. A
# completion-end read (`last`, which every probefix arm has used) reinstates exactly that state.
PROBE_READ = E("PROBE_READ", "mean" if MODE in ("meandiff", "mdstack") else "last")
PROBE_LR, PROBE_STEPS = float(E("PROBE_LR", 1e-2)), int(E("PROBE_STEPS", 4))
PROBE_BUF, PROBE_WARM = int(E("PROBE_BUF", 2048)), int(E("PROBE_WARM", 1024))
PROBE_L2 = float(E("PROBE_L2", 1e-3))
PREF_LOSS, TARGET = E("PREF_LOSS", "hinge"), float(E("TARGET", 1.0))
# K_PROBES>1 (MODE=probe only): K probes, each hinged to TARGET, held mutually orthogonal by a
# Gram penalty at refit time. Motivated by the 0813 deflation result: the preference at L* is a
# THICK subspace (the 16th orthogonal probe still decodes ~.75), so a single direction is one
# strand of the bundle stage 2 actually reads. K probes consolidate K strands.
K_PROBES, ORTH_W = int(E("K_PROBES", 1)), float(E("ORTH_W", 1.0))
# BOOST=1 makes those K probes a CASCADE instead of a parallel set: probe k is fit on the pairs
# probes <k fail to separate. Without it, K probes on one shared label are K rotations of the same
# easy direction -- measured, 0813, and the reason the parallel K=8 arm was a null.
BOOST = int(E("BOOST", 0))
BOOST_TEMP = float(E("BOOST_TEMP", 0.5))
# DPO-Positive (MODE=final only). LAMBDA=0 is plain DPO -- what every arm in the 0811 study ran.
# Same name and default as decodability/{italo,cmpdir}_dpo.py; the settings sheet uses 50.
LAMBDA = float(E("LAMBDA", 0.0))
# RPO (MODE=final only): DPO + alpha * length-normalised NLL on the chosen side. An alternative
# to DPOP's floor -- it pushes log P(chosen) up unconditionally rather than only resisting a fall
# below reference. LAMBDA and RPO_ALPHA compose; either alone is the usual arm.
RPO_ALPHA = float(E("RPO_ALPHA", 0.0))
# --- MODE=meandiff: the phase-8 `pooled_margin` objective, ported (results_phase8.md:266).
# The only arm in this project that installed a preference by optimising activations directly.
# Three pieces, and each one is load-bearing:
#   · POOLED reads (PROBE_READ=mean above) -- closes the forging channel.
#   · MD_LAG=1 -- the direction is the PREVIOUS step's mean difference, not this step's. Fitting
#     and optimising the same direction in one step is self-referential and is what a refitted
#     probe does; the lag is what STATE.md calls "the anti-forging resource".
#   · M0 -- a SATURATING hinge, relu(M0 - proj). An unbounded objective is satisfied by scaling
#     the residual, which is a forging channel that costs one scalar.
# LAMBDA is reused as the DPOP floor weight here (added, not folded into a margin, as in styc).
# M0 IS CALIBRATED, NOT COPIED. styc and the UF port both used M0=4.0, but that number is only
# meaningful against the natural projection in the same units, and SDt normalisation makes those
# units task-specific. The UF port records "proj 3.5 (natural)" against M0=4.0 -- a target 1.14x
# the base level. On dosed britishness the base projection is ~8.1, because the pairs are MINIMAL
# (one word changed), so their difference vectors are far more collinear than styc's ce/we pairs
# and the mean direction is correspondingly better aligned. Copying M0=4.0 here would set a target
# the BASE MODEL ALREADY EXCEEDS -- measured: frac_sat 1.00 at step 10, an objective satisfied at
# initialisation. M0_MULT reproduces the RATIO instead, which is the invariant the design cares
# about. An explicit M0 still overrides.
MD_SD_WARM = int(E("MD_SD_WARM", 256))
M0_MULT = float(E("M0_MULT", 1.15))
M0 = float(E("M0", 0)) or None            # resolved after the base projection is measured
MD_LAG = int(E("MD_LAG", 1))
# --- MODE=mdstack: the meandiff objective at SEVERAL read points, with the stack SEVERED between
# them (deep supervision with gradient truncation; cf. deeply-supervised nets, Lee et al. 2015, and
# greedy local learning, Belilovsky et al. 2019). Read points MD_LAYERS=6,12,18,24,30 give segments
# (0..6), (7..12), (13..18), (19..24), (25..30); a forward pre-hook detaches the residual entering
# each segment, so the gradient from read L reaches ONLY that segment's blocks.
#
# WHY, given RESULTS_0817_MDLATE.md. That sweep showed a single read point fails at both ends: late
# reads are causally inert (the direction lies near the null space of the output map -- phase 1's
# cos(mu, W_A-W_B) = -0.003) and the mid-late band damages text without installing more. This design
# gives every segment a target expressed in ITS OWN representation, so no single direction has to be
# both decodable and causally load-bearing, and the truncation stops a deep objective being
# satisfied by superficial rewriting in later blocks (the forging channel). It also combines local
# supervision with LOWER-STACK WRITE ACCESS, which RESULTS_0817_P3.md showed is what `style` needs.
#
# Each segment's loss carries weight 1, not 1/k: a segment's parameters receive gradient from
# exactly one term, so per-parameter scale matches the single-attach arm and the comparison is fair.
# THE SEVERANCE APPLIES ONLY TO THE MEAN-DIFF READS. The replay and DPOP terms run through an
# unhooked forward, so they still defend the whole stack rather than the top segment alone.
MD_LAYERS = [int(x) for x in E("MD_LAYERS", "6,12,18,24,30").split(",") if x != ""]
# --- MD_FLOOR: DPOP's floor, in ACTIVATION space (MODE=meandiff and mdstack).
# `relu(M0 - proj)` with `proj = ((read(chosen) - read(rejected))/SD).u` constrains only the
# DIFFERENCE. Nothing pins either side's absolute position, which is structurally the same degree of
# freedom that lets plain DPO win its margin by dragging both logps down. MEASURED, 2026-08-17
# (`probefix/pf_mdsides.py`, results/probefix/mdsides.json): MD_L30 doubles its projection
# (+25.0 -> +62.0) with BOTH sides falling and the chosen side DOWN 11.7, read norms unchanged, so
# this is not a frame rotation. mdstack does the same at its top segment (Dchosen -51.7). The
# output-attached control does the opposite: P3's DPOP raises the chosen projection +138 at L30.
# The existing LAMBDA floor acts on the output logps, so the activation objective has had NO
# absolute anchor at all. This adds one: penalise the chosen side's own projection falling below the
# REFERENCE policy's, in the same SD-normalised units. MD_FLOOR=0 (default) reproduces every
# previously run arm exactly.
MD_FLOOR = float(E("MD_FLOOR", 0))
# MD_CAP_MULT: the SATURATING form of that floor, `relu(min(ref_chosen, cap) - chosen)`.
# WHY (RESULTS_0817_MDFLOOR.md §2, §4). The plain floor repaired the mechanism -- Dchosen at L30
# -51.7 -> +59.5 -- and WRECKED the text: leakage 0.625 single-attach, 0.896 on mdstack, coherence
# 10-48. The reason is structural. `relu(M0 - proj)` SATURATES, which phase 8 identified as the
# anti-forging resource; `relu(ref_chosen.u - chosen.u)` does not, because `u` is refitted every
# step and `ref_chosen.u` regenerates a target along each newly-found direction, so the chosen
# side's absolute projection is pushed up without bound -- inflating activations along a direction,
# i.e. exactly the forging channel saturation exists to close. Bounding one channel opened another.
# The cap is the analogue of what M0 does for the difference: a ceiling in the SAME SD-normalised
# units, calibrated as a multiple of the BASE model's own chosen-side projection along the
# calibration direction (never copied -- see the M0 note above for why a copied constant is
# meaningless here). MD_CAP overrides with an explicit value. MD_CAP_MULT=0 (default) is the
# unbounded floor, so every 0817 arm reproduces byte-for-byte.
MD_CAP_MULT = float(E("MD_CAP_MULT", 0))
MD_CAP = float(E("MD_CAP", 0)) or None            # resolved at calibration when MD_CAP_MULT is set
MD_CAP_L = {}                                     # per-segment version, for MODE=mdstack
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
# GRAD_CKPT=1 recomputes block activations in the backward pass instead of storing them.
# SEMANTICALLY IDENTICAL (same loss, same gradients, ~30% slower); it exists because a LATE read
# point backprops through more blocks than an early one, so MODE=meandiff at L26-L30 needs
# activation memory the L20 arms never did -- and the GPU here is shared (NEXT_0810 §4).
# Default off, so every previously-run arm is byte-for-byte reproducible without it.
# NOTE: incompatible with MODE=mdstack -- the severance pre-hook detaches its input, and the
# checkpoint recompute pass then sees different graph metadata than the original forward and raises.
# mdstack therefore runs without it and needs the memory for a full-stack backward.
if int(E("GRAD_CKPT", 0)) and MODE != "mdstack":
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    print("[pf] gradient checkpointing ON (activations recomputed; loss unchanged)", flush=True)
BLOCKS = list(model.model.layers)

# --- graph severance for MODE=mdstack -------------------------------------------------------
SEVER = {"on": False}


def _detach_hook(module, args, kwargs):
    """Forward PRE-hook: replace the incoming hidden state with a detached copy while SEVER is on.
    The forward values are identical, so the loss is unchanged; only the backward path is cut."""
    if not SEVER["on"] or not args:
        return None
    return (args[0].detach(),) + tuple(args[1:]), kwargs


if MODE == "mdstack":
    bounds = [L + 1 for L in sorted(MD_LAYERS)[:-1]]      # inputs of the 2nd..kth segments
    for b in bounds:
        BLOCKS[b].register_forward_pre_hook(_detach_hook, with_kwargs=True)
    print(f"[mdstack] reads at {sorted(MD_LAYERS)} | severed before blocks {bounds}", flush=True)

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


def reads_multi(rows, grad, layers):
    """One forward, all read points. -> {L: (n_pairs, hid) chosen-minus-rejected}. With SEVER on,
    the pre-hooks cut the backward path at each segment boundary."""
    enc, m = _enc(rows)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    out = {}
    with ctx:
        with ResidualCapture([BLOCKS[L] for L in layers]) as cap:
            policy(**enc)
        got = cap.get()
        for i, L in enumerate(layers):
            v = span_read(got[i], m, PROBE_READ)
            out[L] = v[0::2] - v[1::2]
    return out


def reads_sides(rows, grad, layers, ref=False):
    """Per-SIDE reads: -> {L: (chosen, rejected)}. `ref=True` disables the adapter, giving the
    reference policy's projections for the MD_FLOOR term."""
    import contextlib
    enc, m = _enc(rows)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    out = {}
    with ctx, (policy.disable_adapter() if ref else contextlib.nullcontext()):
        with ResidualCapture([BLOCKS[L] for L in layers]) as cap:
            policy(**enc)
        got = cap.get()
        for i, L in enumerate(layers):
            v = span_read(got[i], m, PROBE_READ)
            out[L] = (v[0::2], v[1::2])
    return out


def logps(rows, grad, ref=False, ntok=False):
    """Summed completion log-probs at the OUTPUT. Row-wise logit - logsumexp, never a full fp32
    softmax over the 248k vocab. `ntok=True` also returns the chosen side's scored-token counts,
    which RPO needs to length-normalise its NLL term."""
    import contextlib
    enc, m = _enc(rows)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx, (policy.disable_adapter() if ref else contextlib.nullcontext()):
        lg = policy(**enc).logits[:, :-1].float()
        lp = ((lg.gather(-1, enc.input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
               - lg.logsumexp(-1)) * m).sum(-1)
    if ntok:
        return lp[0::2], lp[1::2], m.sum(-1)[0::2].clamp(min=1)
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


class MultiProbe(Probe):
    """K directions, each unit-norm at score time, refitted like Probe but with a Gram penalty
    holding the (normalised) rows mutually orthogonal. score() -> (n, K); the hinge in the train
    loop then applies elementwise, so every strand must clear TARGET independently. Random init:
    from zeros all rows would receive identical gradients and never separate."""

    def __init__(self, hid, k):
        self.k = k
        w = torch.randn(k, hid, device=DEV) * 1e-2
        self.w = w.requires_grad_(True)
        self.opt = torch.optim.Adam([self.w], lr=PROBE_LR)
        self.buf = torch.zeros(0, hid, device=DEV)
        self.sigma0 = torch.ones(k, device=DEV)

    def _wn(self):
        return self.w / self.w.norm(dim=1, keepdim=True).clamp(min=1e-6)

    def score(self, d):
        return (d.float() @ self._wn().T) / self.sigma0

    def refit(self, steps, bs=256):
        eye = torch.eye(self.k, device=DEV)
        for _ in range(steps):
            i = torch.randint(0, self.buf.shape[0], (min(bs, self.buf.shape[0]),), device=DEV)
            X = self.buf[i]
            self.opt.zero_grad()
            wn = self._wn()
            (-F.logsigmoid(X @ self.w.T).mean() + PROBE_L2 * self.w.pow(2).sum()
             + ORTH_W * (wn @ wn.T - eye).pow(2).sum()).backward()
            self.opt.step()


class BoostedProbe(MultiProbe):
    """K probes fit in CASCADE, not in parallel. Probe 0 sees every pair; probe k is fit on the
    pairs the earlier probes FAIL to separate, so each direction decodes something functionally
    different rather than a geometric copy of the same easy signal.

    This is the fix for the K_PROBES=8 null (0813): eight probes hinged to one shared label found
    eight rotations of the same direction and the objective was satisfied at init. The structure
    the preference actually has is by FAMILY -- per-family probe directions at L20 are mutually
    near-orthogonal (|cos| < .25) and the lexicon probe scores .23 on truth_dialect, below chance
    -- so weighting by residual difficulty should recover that structure without needing labels.

    The hinge is weighted the same way: probe k's loss counts only where probes < k fell short,
    so gradient goes to the hard tail instead of the 98% of pairs that saturate by step 25.
    """

    @torch.no_grad()
    def _weights(self, Z):
        """Z: (n, K) scores. -> (n, K) weights. Column 0 is uniform; column k is high where the
        best earlier probe leaves the pair short of TARGET."""
        W = torch.ones_like(Z)
        for k in range(1, self.k):
            best = Z[:, :k].max(1).values
            W[:, k] = torch.sigmoid((TARGET - best) / BOOST_TEMP)
        return W

    def refit(self, steps, bs=256):
        eye = torch.eye(self.k, device=DEV)
        for _ in range(steps):
            i = torch.randint(0, self.buf.shape[0], (min(bs, self.buf.shape[0]),), device=DEV)
            X = self.buf[i]
            with torch.no_grad():
                Wt = self._weights((X @ self._wn().T) / self.sigma0)
            self.opt.zero_grad()
            wn = self._wn()
            nll = -(F.logsigmoid(X @ self.w.T) * Wt).sum() / Wt.sum().clamp(min=1e-6)
            (nll + PROBE_L2 * self.w.pow(2).sum()
             + ORTH_W * (wn @ wn.T - eye).pow(2).sum()).backward()
            self.opt.step()

    def hinge(self, z):
        """Weighted hinge over the cascade. Replaces relu(TARGET - z).mean() for this probe type."""
        with torch.no_grad():
            Wt = self._weights(z)
        return (F.relu(TARGET - z) * Wt).sum() / Wt.sum().clamp(min=1e-6)


probe = None
if MODE == "probe":
    probe = (BoostedProbe(HID, K_PROBES) if BOOST and K_PROBES > 1 else
             MultiProbe(HID, K_PROBES) if K_PROBES > 1 else Probe(HID))
    t0 = time.time()
    warm = [train_rows[i] for i in rgen.choice(len(train_rows), min(PROBE_WARM, len(train_rows)),
                                               replace=False)]
    for s in range(0, len(warm), BS):
        probe.push(reads(warm[s:s + BS], grad=False))
    probe.buf = probe.buf[-max(PROBE_BUF, len(warm)):]
    probe.refit(600)
    with torch.no_grad():
        if K_PROBES > 1:
            z = probe.buf.float() @ probe._wn().T
            probe.sigma0 = z.std(0).clamp(min=1e-6)
            sig_str = "/".join(f"{s:.2f}" for s in probe.sigma0.tolist())
            wn = probe._wn()
            gram_off = float((wn @ wn.T - torch.eye(probe.k, device=DEV)).abs().max())
        else:
            z = probe.buf.float() @ (probe.w / probe.w.norm())
            probe.sigma0 = float(z.std())
            sig_str, gram_off = f"{probe.sigma0:.3f}", 0.0
    print(f"[probe] warm start on {len(warm)} pairs at L{LAYER} ({PROBE_READ}) in "
          f"{time.time()-t0:.0f}s: train acc {probe.acc(probe.buf):.3f}, sigma0 {sig_str},"
          f" target {TARGET} sigma" + (f", max |gram offdiag| {gram_off:.3f}" if K_PROBES > 1 else ""),
          flush=True)
    probe.buf = probe.buf[-PROBE_BUF:]

# Per-dimension scale for the meandiff objective, estimated ONCE on the base policy before any
# update. styc took this from its feature cache (`SDt`); here it is measured. Fixing it up front
# is the same discipline as the probe's `sigma0`: if the normaliser moved with the activations,
# the saturating target M0 would drift with the quantity it is supposed to bound.
SDt = None
if MODE == "meandiff":
    t0 = time.time()
    warm = [train_rows[i] for i in rgen.choice(len(train_rows),
                                               min(MD_SD_WARM, len(train_rows)), replace=False)]
    ds = torch.cat([reads(warm[s:s + BS], grad=False) for s in range(0, len(warm), BS)]).float()
    SDt = ds.std(0).clamp(min=1e-3)
    u0 = ds.mean(0) / (ds.mean(0).norm() + 1e-6)
    base_proj = float((ds / SDt).matmul(u0).mean())
    if M0 is None:
        M0 = M0_MULT * base_proj
    print(f"[meandiff] SD over {len(warm)} pairs at L{LAYER} ({PROBE_READ}) in {time.time()-t0:.0f}s"
          f" | median SD {float(SDt.median()):.4f} | base proj {base_proj:.2f}"
          f" | M0 {M0:.2f} ({M0 / base_proj:.2f}x base)", flush=True)
    if M0 <= base_proj:
        print(f"  !! M0 {M0:.2f} <= base projection {base_proj:.2f} -- the objective is satisfied "
              f"at initialisation and this arm cannot move. Raise M0_MULT.", flush=True)
    if MD_CAP_MULT and MD_CAP is None:
        chs = torch.cat([reads_sides(warm[s:s + BS], False, [LAYER])[LAYER][0]
                         for s in range(0, len(warm), BS)]).float()
        base_ch = float((chs / SDt).matmul(u0).mean())
        MD_CAP = MD_CAP_MULT * base_ch
        print(f"[meandiff] floor cap {MD_CAP:.2f} = {MD_CAP_MULT:.2f}x base chosen projection "
              f"{base_ch:.2f}", flush=True)
        if base_ch <= 0:
            # MEASURED 2026-08-18: base_ch is -2.17 at L20 pooled. `u` is a DIFFERENCE direction, so
            # an absolute projection onto it carries an arbitrary offset and its SIGN is not
            # meaningful -- which makes MD_CAP_MULT meaningful ONLY at 1.0 (cap = the base level).
            # Any other multiplier moves the cap the wrong way when base_ch < 0, unlike M0_MULT
            # where the quantity is a difference and positive by construction.
            print(f"  !! base chosen projection {base_ch:.2f} is negative -- an offset on a "
                  f"difference direction. The cap is then a DON'T-FALL-BELOW-BASE floor that "
                  f"switches off once the chosen side rises above it; it cannot deliver a bounded "
                  f"LIFT. Only MD_CAP_MULT=1.0 is interpretable here; set MD_CAP for anything else.",
                  flush=True)


SD_L, M0_L = {}, {}
if MODE == "mdstack":
    t0 = time.time()
    warm = [train_rows[i] for i in rgen.choice(len(train_rows),
                                               min(MD_SD_WARM, len(train_rows)), replace=False)]
    acc = {L: [] for L in sorted(MD_LAYERS)}
    SEVER["on"] = False                        # calibration is measurement only, no backward
    for s0 in range(0, len(warm), BS):
        got = reads_multi(warm[s0:s0 + BS], False, sorted(MD_LAYERS))
        for L, v in got.items():
            acc[L].append(v.float())
    for L in sorted(MD_LAYERS):
        ds = torch.cat(acc[L])
        SD_L[L] = ds.std(0).clamp(min=1e-3)
        u0 = ds.mean(0) / (ds.mean(0).norm() + 1e-6)
        bp = float((ds / SD_L[L]).matmul(u0).mean())
        M0_L[L] = (float(E("M0", 0)) or M0_MULT * bp)
        flag = "  !! satisfied at init" if M0_L[L] <= bp else ""
        print(f"[mdstack] L{L:>2}: median SD {float(SD_L[L].median()):.4f} | base proj {bp:.2f}"
              f" | M0 {M0_L[L]:.2f} ({M0_L[L]/bp:.2f}x){flag}", flush=True)
    if MD_CAP_MULT:
        accs = {L: [] for L in sorted(MD_LAYERS)}
        for s0 in range(0, len(warm), BS):
            for L, (a, _) in reads_sides(warm[s0:s0 + BS], False, sorted(MD_LAYERS)).items():
                accs[L].append(a.float())
        for L in sorted(MD_LAYERS):
            chs = torch.cat(accs[L])
            ds = torch.cat(acc[L])
            u0 = ds.mean(0) / (ds.mean(0).norm() + 1e-6)
            base_ch = float((chs / SD_L[L]).matmul(u0).mean())
            MD_CAP_L[L] = MD_CAP if MD_CAP is not None else MD_CAP_MULT * base_ch
            print(f"[mdstack] L{L:>2}: floor cap {MD_CAP_L[L]:.2f} "
                  f"(base chosen {base_ch:.2f})", flush=True)
    print(f"[mdstack] calibration in {time.time()-t0:.0f}s", flush=True)


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

hist = dict(mode=MODE, md_floor=MD_FLOOR, md_cap_mult=MD_CAP_MULT,
            model=MODEL, layer=LAYER, probe_read=PROBE_READ, pref_loss=PREF_LOSS,
            target=TARGET, lora=[LORA_MIN, LORA_MAX], init_merge=INIT_MERGE, lr=LR, beta=BETA,
            dpop_lambda=LAMBDA, rpo_alpha=RPO_ALPHA,
            steps=STEPS, seed=SEED, w_pref=W_PREF, w_replay=W_REPLAY, replay_loss=REPLAY_LOSS,
            brit=E("SUP_BRIT", "release"),
            sigma0=(probe.sigma0.tolist() if probe is not None and torch.is_tensor(probe.sigma0)
                    else (probe.sigma0 if probe else None)),
            k_probes=K_PROBES, parts=[], evals=[])
ev = evaluate(0); hist["evals"].append(ev)
print(f"  step   0: {line(ev)}", flush=True)
policy.train()
u_hist = []          # lag-MD_LAG buffer of past mean-difference directions (meandiff)
u_stack = {}         # per-read-point version of the same, for MODE=mdstack

for step in range(STEPS):
    rows = [train_rows[i] for i in rgen.choice(len(train_rows), PREF_PAIRS, replace=False)]
    opt.zero_grad()
    if MODE == "probe":
        d = reads(rows, grad=True)
        probe.push(d)
        probe.refit(PROBE_STEPS)            # continuous refit, on DETACHED activations
        z = probe.score(d)                  # policy gradient through this step's fitted direction
        l_pref = (probe.hinge(z) if hasattr(probe, "hinge") and PREF_LOSS == "hinge"
                  else F.relu(TARGET - z).mean() if PREF_LOSS == "hinge"
                  else -F.logsigmoid(BETA * z).mean())
        with torch.no_grad():
            extra = dict(z=float(z.mean()), frac_sat=float((z > TARGET).float().mean()))
    elif MODE == "meandiff":
        # phase-8 pooled_margin. The forging channel is OPEN by design -- the gradient runs
        # through the policy's own activations -- and pooling plus the lag is what phase 8 found
        # closes it in practice. If it forges here, the surrogate/behaviour gap will show up as
        # proj saturating while the eval buckets stay at base, which is the same signature
        # MODE=probe already produces and would be the informative negative.
        d = reads(rows, grad=True) / SDt
        with torch.no_grad():
            u_now = d.mean(0)
            u_now = u_now / (u_now.norm() + 1e-6)
        u = u_hist[-MD_LAG] if len(u_hist) >= MD_LAG else u_now
        u_hist.append(u_now)
        proj = d.matmul(u)
        l_pref = F.relu(M0 - proj).mean()
        if MD_FLOOR:                     # DPOP's floor in activation space (see MD_FLOOR above)
            sd_ch = reads_sides(rows, True, [LAYER])[LAYER][0] / SDt
            with torch.no_grad():
                rf_ch = reads_sides(rows, False, [LAYER], ref=True)[LAYER][0] / SDt
                rc = rf_ch.matmul(u)
                if MD_CAP is not None:
                    rc = rc.clamp(max=MD_CAP)      # relu(min(ref_chosen, cap) - chosen)
            l_pref = l_pref + MD_FLOOR * F.relu(rc - sd_ch.matmul(u)).mean()
        if LAMBDA:                       # DPOP floor on the chosen side, added as in styc
            a_md, _ = logps(rows, grad=True)
            with torch.no_grad():
                ra_md, _ = logps(rows, False, ref=True)
            l_pref = l_pref + LAMBDA * F.relu(ra_md - a_md).mean()
        with torch.no_grad():
            extra = dict(proj=float(proj.mean()), frac_sat=float((proj > M0).float().mean()),
                         u_cos=float((u * u_now).sum()))
    elif MODE == "mdstack":
        # one severed forward -> every segment's own hinge; sum with weight 1 each
        SEVER["on"] = True
        got_sides = reads_sides(rows, True, sorted(MD_LAYERS))
        got = {L: (a - b) for L, (a, b) in got_sides.items()}
        SEVER["on"] = False
        ref_ch = None
        if MD_FLOOR:
            SEVER["on"] = True           # same severed path, so each segment's floor is local too
            with torch.no_grad():
                ref_ch = {L: v[0] for L, v in
                          reads_sides(rows, False, sorted(MD_LAYERS), ref=True).items()}
            SEVER["on"] = False
        l_pref = torch.zeros((), device=DEV)
        projs, sats, coss = {}, {}, {}
        for L in sorted(MD_LAYERS):
            d = got[L] / SD_L[L]
            with torch.no_grad():
                u_now = d.mean(0)
                u_now = u_now / (u_now.norm() + 1e-6)
            hist_L = u_stack.setdefault(L, [])
            u = hist_L[-MD_LAG] if len(hist_L) >= MD_LAG else u_now
            hist_L.append(u_now)
            proj = d.matmul(u)
            l_pref = l_pref + F.relu(M0_L[L] - proj).mean()
            if MD_FLOOR:
                c = (got_sides[L][0] / SD_L[L]).matmul(u)
                with torch.no_grad():
                    rc = (ref_ch[L] / SD_L[L]).matmul(u)
                    if L in MD_CAP_L:
                        rc = rc.clamp(max=MD_CAP_L[L])
                l_pref = l_pref + MD_FLOOR * F.relu(rc - c).mean()
            with torch.no_grad():
                projs[L] = float(proj.mean())
                sats[L] = float((proj > M0_L[L]).float().mean())
                coss[L] = float((u * u_now).sum())
        if LAMBDA:                       # DPOP floor, through the UNHOOKED path (whole stack)
            a_md, _ = logps(rows, grad=True)
            with torch.no_grad():
                ra_md, _ = logps(rows, False, ref=True)
            l_pref = l_pref + LAMBDA * F.relu(ra_md - a_md).mean()
        with torch.no_grad():
            extra = dict(proj=float(np.mean(list(projs.values()))),
                         frac_sat=float(np.mean(list(sats.values()))),
                         u_cos=float(np.mean(list(coss.values()))),
                         per_layer={str(L): dict(proj=projs[L], sat=sats[L], ucos=coss[L])
                                    for L in sorted(MD_LAYERS)})
    else:
        if RPO_ALPHA:
            a, b, na = logps(rows, grad=True, ntok=True)
        else:
            a, b = logps(rows, grad=True)
        with torch.no_grad():
            ra, rb = logps(rows, False, ref=True)
        margin = (a - ra) - (b - rb)
        if LAMBDA:                       # DPO-Positive: stop the chosen side's logp falling
            margin = margin - LAMBDA * F.relu(ra - a)
        l_pref = -F.logsigmoid(BETA * margin).mean()
        if RPO_ALPHA:
            # RPO (Pang et al. / Llama-3): DPO plus a length-normalised NLL on the CHOSEN side.
            # DPOP only resists the chosen logp falling BELOW reference (relu, zero gradient
            # above it); this pushes it up unconditionally. The 0813 finding that d_chosen
            # predicts free-generation marker rate is the reason to try the stronger version.
            l_pref = l_pref + RPO_ALPHA * (-(a / na).mean())
        with torch.no_grad():
            # margin stays the RAW (unpenalised) quantity so it is comparable to the 0811
            # histories. d_chosen/d_rejected are new: the 0811 runs logged only the difference,
            # so nothing recorded that the margin was being won by the chosen side falling --
            # which is the failure the rollouts show and the reason this term exists.
            extra = dict(margin=float(((a - ra) - (b - rb)).mean()),
                         d_chosen=float((a - ra).mean()), d_rejected=float((b - rb).mean()))
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
              + (f"proj {p['proj']:+.2f} sat {p['frac_sat']:.2f} ucos {p['u_cos']:+.2f}"
                 if MODE in ("meandiff", "mdstack")
                 else f"z {p['z']:+.2f} sat {p['frac_sat']:.2f}" if MODE == "probe"
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
