#!/usr/bin/env python
"""FORGETTING CURVES: how durable is an installed preference under further fine-tuning?

Every meter this project has used to rank the arms -- brit_rate, marker_density, per-word marker
rate -- has been fooled by output length in turn (0812-0813: brit_rate crowned a dead arm at ten
marker hits; density credited American markers; per-word rate rewarded DPOP's terseness). This
measures RELATIVE DECAY from each arm's own starting point instead, so the absolute install
strength and the output length both cancel.

It also tests the two-stage hypothesis directly rather than by proxy. If stage 1 grounds the
preference in a representation at L* and stage 2 wires a readout to it, the install has a
substrate and should outlast a one-stage output-level patch. The prediction is NOT obvious in
either direction: the two-stage edit is confined to 11 blocks, a smaller footprint that could be
easier to overwrite.

DESIGN
  · Every arm is MERGED into the base weights first, so all arms enter continued training
    structurally identical. Otherwise this would partly measure "does a LoRA on blocks 21-31
    survive", which is an artifact of where the adapter sits, not a property of the install.
  · Continued training is plain next-token NLL on chat records HELD OUT from the replay bank the
    arms were anchored on (`replay_bank_heldout.pt`, disjoint by construction). Retraining on the
    banked rows would be close to a no-op.
  · Identical recipe, seed, and step count for every arm. A fresh LoRA on all layers -- full
    fine-tuning of 4B does not fit alongside Adam on this box.
  · Length and diversity are tracked at every checkpoint: a collapsed or terse model can "retain"
    trivially by being stuck in a degenerate mode, and that would read as durability.

Env: ARMS=base,C0,C1,P0,P1  STEPS=300 EVAL_EVERY=50 LR=1e-4 SEED=0 N_GEN=48 GEN_TOKENS=60
"""
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pf_common import DEV, MODEL, encode, load_split, replay_batch  # noqa: E402
sys.path.insert(0, "/workspace/rd-branch/supervisor")
from sup_common import marker_lexicon  # noqa: E402

E = os.environ.get
HF = ("/workspace/.hf_home/hub/models--sunnyhoward--reward-depth-probefix4b/snapshots/"
      "3b55e6a54b017b522c841b8033807ae614bce53e")
S1 = f"{HF}/B4_probe_L20/ckpt300"
STEPS, EVAL_EVERY = int(E("STEPS", 300)), int(E("EVAL_EVERY", 50))
LR, SEED = float(E("LR", 1e-4)), int(E("SEED", 0))
N_GEN, GEN_TOKENS = int(E("N_GEN", 48)), int(E("GEN_TOKENS", 60))
REPLAY_TOK = int(E("REPLAY_TOK", 16))
BANK = E("FORGET_BANK", "/workspace/sup/replay_bank_heldout.pt")
OUT = E("OUT", "/workspace/probefix4b_forget")
CLOSE_THINK = "\n</think>\n\n"
os.makedirs(OUT, exist_ok=True)

# stage-1 adapter first where an arm has one; merged in order.
ARM_SPEC = {
    "base": [],
    "C0":   [S1, f"{HF}/C0_two_plain/ckpt600"],      # two-stage, plain DPO
    "C1":   [S1, f"{HF}/C1_two_dpop/ckpt600"],       # two-stage, DPOP
    "P0":   [f"{HF}/P0_all_plain/ckpt600"],          # one-stage plain -- the collapsed arm
    "P1":   [f"{HF}/P1_all_dpop/ckpt600"],           # one-stage DPOP
}
ARMS = [a for a in E("ARMS", "base,C0,C1,P0,P1").split(",") if a]

tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
AM, BR, AM_RE, BR_RE = marker_lexicon()

_bank = torch.load(BANK)
REPLAY_IDS, REPLAY_START = _bank["ids"].long(), _bank["start"].long()

val = load_split("validation")
_g = random.Random(7)
GEN_ROWS = [r for r in val if r["family"] in ("lexicon", "culture", "false_friend")]
_g.shuffle(GEN_ROWS)
GEN_ROWS = GEN_ROWS[:N_GEN]


@torch.no_grad()
def behaviour(model):
    """Free-sample the install. Greedy, held-out prompts, think-block closed (otherwise the budget
    goes into a reasoning trace the preference never touched)."""
    was = model.config.use_cache
    model.config.use_cache = True
    outs = []
    for s in range(0, len(GEN_ROWS), 8):
        ps = [r["text_prompt"] + CLOSE_THINK for r in GEN_ROWS[s:s + 8]]
        enc = encode(tok, ps, max_length=320).to(DEV)
        g = model.generate(**enc, do_sample=False, max_new_tokens=GEN_TOKENS,
                           pad_token_id=tok.pad_token_id)
        P = enc.input_ids.shape[1]
        outs += [tok.decode(g[i, P:], skip_special_tokens=True).strip() for i in range(g.shape[0])]
    model.config.use_cache = was
    na = sum(len(AM_RE.findall(o.lower())) for o in outs) if AM_RE else 0
    nb = sum(len(BR_RE.findall(o.lower())) for o in outs) if BR_RE else 0
    uniq = len(set(" ".join(o.split())[:60].lower() for o in outs)) / max(1, len(outs))
    return dict(br=nb, am=na, n=len(outs),
                brit_rate=(nb / (na + nb)) if (na + nb) else None,
                density=(na + nb) / max(1, len(outs)),
                br_per_sample=nb / max(1, len(outs)),
                mean_len=float(np.mean([len(o.split()) for o in outs])),
                diversity=uniq, sample=outs[0][:120])


def build(arm):
    """Base + this arm's adapters, MERGED, then a fresh all-layer LoRA. Every arm gets the same
    structure so what decays is the install, not the adapter placement."""
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
    for ad in ARM_SPEC[arm]:
        m = PeftModel.from_pretrained(m, ad).merge_and_unload().eval()
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"],
                     layers_to_transform=list(range(len(m.model.layers))))
    p = get_peft_model(m, cfg)
    p.config.use_cache = False
    return p


results = {}
for arm in ARMS:
    torch.manual_seed(SEED); np.random.seed(SEED)
    tgen = torch.Generator().manual_seed(SEED + 11)
    t0 = time.time()
    policy = build(arm)
    params = [q for q in policy.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR)
    curve = []

    b0 = behaviour(policy)
    curve.append(dict(step=0, **b0))
    print(f"[{arm}] step   0: br {b0['br']:3d} am {b0['am']:3d} "
          f"brit_rate {b0['brit_rate']} density {b0['density']:.2f} len {b0['mean_len']:.0f}",
          flush=True)

    policy.train()
    for step in range(STEPS):
        enc, m = replay_batch(REPLAY_IDS, REPLAY_START, tok, REPLAY_TOK, tgen)
        lg = policy(**enc).logits[:, :-1].float()
        lp = (lg.gather(-1, enc["input_ids"][:, 1:].unsqueeze(-1)).squeeze(-1) - lg.logsumexp(-1))
        loss = -(lp * m).sum() / m.sum().clamp(min=1)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if (step + 1) % EVAL_EVERY == 0:
            policy.eval()
            b = behaviour(policy)
            curve.append(dict(step=step + 1, nll=float(loss.detach()), **b))
            print(f"[{arm}] step {step+1:3d}: br {b['br']:3d} am {b['am']:3d} "
                  f"brit_rate {b['brit_rate']} density {b['density']:.2f} "
                  f"len {b['mean_len']:.0f} div {b['diversity']:.2f} nll {float(loss.detach()):.3f}",
                  flush=True)
            policy.train()

    results[arm] = curve
    json.dump(results, open(f"{OUT}/forget_curves.json", "w"), indent=1)
    print(f"[{arm}] done in {time.time()-t0:.0f}s", flush=True)
    del policy, opt, params
    torch.cuda.empty_cache()

# ---- retention summary. Fraction of each arm's OWN starting signal that survives, which is the
# whole point: absolute levels differ 5x across arms and are not comparable.
print("\nRETENTION -- br markers per sample, and fraction of step-0 value retained")
hdr = "arm      " + "".join(f"{c['step']:>9d}" for c in results[ARMS[0]])
print(hdr)
for arm in ARMS:
    c = results[arm]
    b0 = c[0]["br_per_sample"]
    print(f"{arm:8s} " + "".join(f"{x['br_per_sample']:9.2f}" for x in c))
    if b0 > 0:
        print(f"{'  frac':8s} " + "".join(f"{x['br_per_sample']/b0:9.2f}" for x in c))
print("\nlength (a terse model can 'retain' by being stuck -- read this alongside)")
for arm in ARMS:
    print(f"{arm:8s} " + "".join(f"{x['mean_len']:9.0f}" for x in results[arm]))
json.dump(results, open(f"{OUT}/forget_curves.json", "w"), indent=1)
print(f"\nwrote {OUT}/forget_curves.json")
