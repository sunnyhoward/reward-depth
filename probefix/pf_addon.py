#!/usr/bin/env python
"""A LEARNED steering vector per layer: h_L <- h_L + A_L * z, with z the preference in [-1, +1].

THE QUESTION THIS ANSWERS THAT pf_steer.py CANNOT. `pf_steer.py` asks whether the FITTED preference
direction (pooled mean-difference / logistic probe) is a causal handle at depth L. If it is not, two
readings survive and they are very different:

  (i) nothing at layer L is a handle -- the preference is not writable there, which is the
      null-space account of the stage-1 failure (phase 1: cos(mu, W_A - W_B) = -0.003;
      RESULTS_0817_MDLATE: late reads inert; RESULTS_0818_STAGE1_SIDES §5: freezing the probe
      changes nothing);
  (ii) the FITTED direction is simply the wrong vector, and a direction chosen by gradient against
      the actual generation objective would work fine.

Nothing in the project distinguishes these, because every steering and activation-training result
so far uses a direction fitted to SEPARATE, not to GENERATE. This trains the vector directly: A_L is
2560 parameters, the rest of the model is frozen, and the only thing A_L is asked to do is make the
model generate the preferred form when z=+1 and the dispreferred form when z=-1.

  · If learned A_L works where the fitted direction fails, reading (ii) holds and every
    "activation space cannot install this" claim in the repo is a claim about the FITTING method.
  · If learned A_L also dies in the upper stack, reading (i) is confirmed by the strongest test
    available: gradient descent had free choice of direction and still could not find one.
  · z=0 reproduces the base model EXACTLY (A*0 = 0), so no anchor term is needed and the
    unsteered control is the same weights, not a different run.

WHY THE DEFAULT OBJECTIVE IS `nll`, NOT A MARGIN. A margin m(z) = logp(chosen) - logp(rejected) is a
DIFFERENCE, and 2026-08-18 established that this project's difference objectives leave both sides'
absolute positions unpinned (`RESULTS_0818_STAGE1_SIDES.md` §2; `pf_mdsides.py`). Steering is a
statement about generation, which depends on the chosen side's absolute likelihood, so the default
optimises exactly that: NLL of the chosen continuation at z=+1 plus NLL of the rejected continuation
at z=-1. `PREF=dpo` restores the margin form for comparison, and it is expected to be the weaker
arm -- that prediction is on record here.

Outputs one famgen-schema file per (layer, z) cell, so pf_leakage.py and pf_cjudge.py batchnew read
them with no adaptation, exactly as pf_steer.py's cells do.

Env: LAYERS=0,4,8,12,16,20,24,28,31  STEPS=300  LR=1e-2  PREF=nll|dpo  BETA=0.1
     Z_GEN=1.0  PAIRS=6  N_PER_FAM=48  GEN_TOKENS=96  SEED=0  OUT=results/probefix4b_addon
"""
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "supervisor"))
from pf_common import DEV, MODEL, encode, load_split, pair_texts, span_mask  # noqa: E402
from pf_famlex import family_lexicon, score                                  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer                 # noqa: E402

E = os.environ.get
LAYERS = [int(x) for x in E("LAYERS", "0,4,8,12,16,20,24,28,31").split(",") if x != ""]
STEPS, LR, BETA = int(E("STEPS", 300)), float(E("LR", 1e-2)), float(E("BETA", 0.1))
PREF = E("PREF", "nll")
Z_GEN = [float(x) for x in E("Z_GEN", "1.0").split(",") if x != ""]
PAIRS = int(E("PAIRS", 6))
N_PER_FAM = int(E("N_PER_FAM", 48))
GEN_TOKENS = int(E("GEN_TOKENS", 96))
SEED = int(E("SEED", 0))
MAXLEN = int(E("MAX_LEN", 256))
OUT = E("OUT", f"{REPO}/results/probefix4b_addon")
SRC = E("SUP_BRIT", f"{REPO}/supervisor/britishness/dosed/brit_dose20.jsonl")
FAMS = [f for f in E("FAMS", "false_friend,style").split(",") if f]
os.makedirs(OUT, exist_ok=True)
CLOSE_THINK = "\n</think>\n\n"
torch.manual_seed(SEED)
np.random.seed(SEED)
rgen = np.random.RandomState(SEED + 7)

tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
_, _, REX = family_lexicon(SRC)

rows = [json.loads(line) for line in open(SRC)]
rng = random.Random(SEED)
PROMPTS = {}
for fam in ("lexicon", "culture", "expression", "false_friend", "style"):
    rs = [r for r in rows if r["family"] == fam and r["role"] == "install"
          and r.get("reserved_for_eval")]
    if len(rs) < N_PER_FAM:
        rs = [r for r in rows if r["family"] == fam and r["role"] == "install"]
    rng.shuffle(rs)
    PROMPTS[fam] = rs[:N_PER_FAM]
guard = [r for r in rows if r.get("eval_bucket") == "guard"]
rng.shuffle(guard)
PROMPTS["guard"] = guard[:N_PER_FAM]
PROMPTS = {k: v for k, v in PROMPTS.items() if k in FAMS}

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
for p in model.parameters():
    p.requires_grad_(False)
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
BLOCKS = list(model.model.layers)
HID = model.config.get_text_config().hidden_size
train_rows = load_split("train")
print(f"[addon] {MODEL} blocks={len(BLOCKS)} hid={HID} | layers {LAYERS} | {PREF} | "
      f"{STEPS} steps lr {LR} | {len(train_rows)} pairs", flush=True)

Z = {"v": 0.0}
A = {"w": None}


def hook(mod, args, out):
    if A["w"] is None or Z["v"] == 0.0:
        return out
    d = (A["w"] * Z["v"]).to(out[0].dtype if isinstance(out, tuple) else out.dtype)
    if isinstance(out, tuple):
        return (out[0] + d,) + out[1:]
    return out + d


def logps(rws):
    """-> (logp_chosen, logp_rejected, n_tok_chosen) under the CURRENT Z and A."""
    trip = pair_texts(tok, rws)
    texts = [t for c, j, _ in trip for t in (c, j)]
    plens = [pl for _, _, pl in trip for _ in (0, 1)]
    enc = encode(tok, texts, max_length=MAXLEN).to(DEV)
    m = span_mask(tok, texts, plens, enc)
    lg = model(**enc).logits[:, :-1].float()
    lp = ((lg.gather(-1, enc.input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
           - lg.logsumexp(-1)) * m).sum(-1)
    return lp[0::2], lp[1::2], m.sum(-1)[0::2].clamp(min=1), m.sum(-1)[1::2].clamp(min=1)


@torch.no_grad()
def gen(prompts):
    model.config.use_cache = True
    outs = []
    for s in range(0, len(prompts), 8):
        ps = [p["text_prompt"] + CLOSE_THINK for p in prompts[s:s + 8]]
        enc = encode(tok, ps, max_length=320).to(DEV)
        g = model.generate(**enc, do_sample=False, max_new_tokens=GEN_TOKENS,
                           pad_token_id=tok.pad_token_id)
        P = enc.input_ids.shape[1]
        outs += [tok.decode(g[i, P:], skip_special_tokens=True).strip() for i in range(g.shape[0])]
    model.config.use_cache = False
    return outs


for L in LAYERS:
    tag = f"addon_L{L}"
    ckpt = f"{OUT}/{tag}.pt"
    handle = BLOCKS[L].register_forward_hook(hook)
    A["w"] = torch.zeros(HID, device=DEV, dtype=torch.float32, requires_grad=True)
    if os.path.exists(ckpt):
        A["w"] = torch.load(ckpt, map_location=DEV)["A"].to(DEV).requires_grad_(True)
        print(f"== {tag} loaded", flush=True)
    else:
        opt = torch.optim.Adam([A["w"]], lr=LR)
        t0 = time.time()
        hist = []
        for step in range(STEPS):
            rws = [train_rows[i] for i in rgen.choice(len(train_rows), PAIRS, replace=False)]
            opt.zero_grad()
            if PREF == "nll":
                # absolute, not a difference: chosen must be LIKELY at z=+1 and rejected at z=-1
                Z["v"] = +1.0
                a, _, na, _ = logps(rws)
                loss_p = -(a / na).mean()
                Z["v"] = -1.0
                _, b, _, nb = logps(rws)
                loss_n = -(b / nb).mean()
                loss = loss_p + loss_n
            else:
                Z["v"] = +1.0
                a, b, _, _ = logps(rws)
                loss = -F.logsigmoid(BETA * (a - b)).mean()
                Z["v"] = -1.0
                a2, b2, _, _ = logps(rws)
                loss = loss - F.logsigmoid(-BETA * (a2 - b2)).mean()
            loss.backward()
            opt.step()
            if (step + 1) % 25 == 0:
                with torch.no_grad():
                    Z["v"] = 1.0
                    a, b, na, nb = logps(rws)
                    acc = float((a > b).float().mean())
                hist.append(dict(step=step + 1, loss=float(loss), acc_at_z1=acc,
                                 a_norm=float(A["w"].norm())))
                print(f"  [{tag}] step {step+1:4d}: loss {float(loss):.4f} "
                      f"acc@z+1 {acc:.2f} |A| {float(A['w'].norm()):.2f}", flush=True)
        Z["v"] = 0.0
        torch.save(dict(A=A["w"].detach().cpu(), layer=L, pref=PREF, steps=STEPS, lr=LR,
                        hist=hist), ckpt)
        print(f"  [{tag}] trained in {time.time()-t0:.0f}s", flush=True)

    for z in Z_GEN:
        path = f"{OUT}/famgen_{tag}_z{z}.json"
        if os.path.exists(path):
            continue
        Z["v"] = z
        rec = {"arm": f"{tag}_z{z}", "adapters": [],
               "addon": dict(layer=L, z=z, pref=PREF, steps=STEPS,
                             a_norm=float(A["w"].norm())), "families": {}}
        for fam, ps in PROMPTS.items():
            outs = gen(ps)
            sc = score(outs, REX)
            own = sc.get(fam) or {}
            rec["families"][fam] = {
                "n": len(outs), "marker_scores": sc, "own_family": own,
                "gens": [{"prompt": p["prompt"], "gen": o} for p, o in zip(ps, outs)]}
            br = own.get("brit_rate")
            print(f"  [{tag} z={z}] {fam:<13} brit_rate "
                  f"{br if br is None else f'{br:.3f}'} "
                  f"len {sum(len(o) for o in outs)/max(1,len(outs)):.0f}", flush=True)
        json.dump(rec, open(path, "w"), indent=1)
    Z["v"] = 0.0
    handle.remove()
print("ADDONDONE", flush=True)
