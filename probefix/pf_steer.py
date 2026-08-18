#!/usr/bin/env python
"""Causal efficacy of the MEAN-POOLED preference direction vs depth, on britishness / Qwen3.5-4B.

WHY THIS, AND WHY HERE. `notes_steering_experiment.md`'s design was run twice, both times on
UltraFeedback with Tulu-8B: phase 6 with LAST-TOKEN directions (judge-null at every depth and dose,
94% of 96 cells) and phase 9 §7 with the POOLED direction (the project's first non-null steering --
mid-layer peak, L8 win .586, dead at L0, weak at L31). Neither has ever been run on the task every
probefix result is about. That matters because UF is a hard, diffuse preference where the ceiling is
low for everything; britishness is a minimal-pair task with a lexical ground truth and an existing
blind judge harness. If "readable everywhere, steerable in the middle" is a real property of the
representation rather than an artefact of a hard dataset, it should appear here more cleanly.

It also stands directly against 2026-08-18's training results. Stage 1 optimises a pooled/last read
at a chosen depth and installs nothing at the output, and three candidate explanations are now
eliminated by measurement (RESULTS_0818_STAGE1_SIDES.md §2, §4, §5); what remains is that the
direction is not a causal handle at that depth. THAT IS A STEERING CLAIM AND IT HAS NEVER BEEN
TESTED ON THIS TASK. This measures it with no training at all.

PRE-REGISTERED, before the run (phase 1: cos(mu, W_A - W_B) = -0.003 at the final block;
RESULTS_0817_MDLATE: reads at L30/L31 are inert; phase 9: mid-layer peak on UF):
  · efficacy rises through the low-mid stack and collapses by L28-L31;
  · L20 -- the probefix attach point -- is NOT the peak;
  · at matched alpha the top of the stack moves brit_rate least and KL least.
If instead efficacy is flat, or peaks at the top, the null-space account of the stage-1 failure is
wrong and the whole 0818 line needs rereading.

DESIGN, following uf_steer_sweep.py so the two are comparable:
  · the probe is fitted per layer on POOLED (mean over the assistant span) RMS-normalised reads,
    the same read pf_train.py's meandiff arms use;
  · the steering vector is the ActAdd/CAA `dm` form, v ~ SD * mu -- the raw-space class-mean
    difference under diagonal-covariance LDA -- NOT mu/SD, which is the direction that moves the
    probe read fastest and is circular by construction;
  · scaled by alpha * R_L with R_L the mean RAW residual norm at that layer over the fit pool, so
    alpha means the same fraction of residual magnitude at every depth;
  · added at EVERY position of block L's output during generation.

Prompts are pf_famgen.py's eval draw (SEED=0), so every cell is written in the famgen schema and
can go straight into pf_leakage.py and pf_cjudge.py batchnew against the banked references. The
lexical meter (pf_famlex) is non-circular -- it counts dialect markers, it does not consult the
probe -- and is the primary cheap read; the judge is for the band this identifies.

MULTI=1 applies EVERY layer in LAYERS AT ONCE (one cell per alpha) instead of one cell per layer.
Not a cosmetic difference: the residual stream carries each addition forward, so k simultaneous
insertions compound rather than average, and the same alpha is a k-fold larger intervention. The
single-layer sweep found a mid-stack plateau of roughly equal cells (RESULTS_0818_STEER.md §1) --
whether those are the SAME edit re-expressed at each depth or k independent ones is exactly what
simultaneous application tests, and no run in this project has ever done it. SCALE=div_n divides by
the number of live layers, which holds the total injected norm fixed instead of the per-layer one;
report both, because they answer different questions.

Env: LAYERS=0,4,8,12,16,20,24,28,31  ALPHAS=0.1,0.3  MULTI=0  SCALE=none|div_n  N_FIT=512
     N_PER_FAM=48  GEN_TOKENS=96  SEED=0  OUT=results/probefix4b_steer  STEER_DIR=dm
"""
import json
import os
import random
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "supervisor"))
from pf_common import (DEV, MODEL, ResidualCapture, encode, load_split,  # noqa: E402
                       pair_texts, span_mask, span_read)
from pf_famlex import family_lexicon, score                              # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer             # noqa: E402

E = os.environ.get
LAYERS = [int(x) for x in E("LAYERS", "0,4,8,12,16,20,24,28,31").split(",") if x != ""]
ALPHAS = [float(x) for x in E("ALPHAS", "0.1,0.3").split(",") if x != ""]
N_FIT = int(E("N_FIT", 512))
N_PER_FAM = int(E("N_PER_FAM", 48))
GEN_TOKENS = int(E("GEN_TOKENS", 96))
SEED = int(E("SEED", 0))
BS = int(E("BS", 8))
MAXLEN = int(E("MAX_LEN", 256))
STEER_DIR = E("STEER_DIR", "dm")
MULTI = int(E("MULTI", 0))
SCALE = E("SCALE", "none")
TAG_PREFIX = E("TAG_PREFIX", "")
OUT = E("OUT", f"{REPO}/results/probefix4b_steer")
SRC = E("SUP_BRIT", f"{REPO}/supervisor/britishness/dosed/brit_dose20.jsonl")
FAMS = [f for f in E("FAMS", "false_friend,style").split(",") if f]
PROBE_L2, PROBE_LR, PROBE_STEPS = float(E("PROBE_L2", 1e-3)), float(E("PROBE_LR", 1e-2)), 600
os.makedirs(OUT, exist_ok=True)
CLOSE_THINK = "\n</think>\n\n"

tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
_, _, REX = family_lexicon(SRC)

# --- prompt selection: byte-identical to pf_famgen.py / pf_famgen_arms.py, including the families
# --- it draws but we do not generate -- the draws are SEQUENTIAL off one RNG, so skipping one
# --- shifts every later family's sample and the pf_cjudge join would fail.
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
BLOCKS = list(model.model.layers)
NB = len(BLOCKS)
print(f"[steer] {MODEL} blocks={NB} | layers {LAYERS} | alphas {ALPHAS} | dir {STEER_DIR}",
      flush=True)

# ---------------------------------------------------------------- fit, every layer, one pass
fit_rows = load_split("train")[:N_FIT]
acc = {L: [] for L in range(NB)}
raw_norm = {L: [] for L in range(NB)}
with torch.no_grad():
    for s in range(0, len(fit_rows), BS):
        trip = pair_texts(tok, fit_rows[s:s + BS])
        texts = [t for c, j, _ in trip for t in (c, j)]
        plens = [pl for _, _, pl in trip for _ in (0, 1)]
        enc = encode(tok, texts, max_length=MAXLEN).to(DEV)
        m = span_mask(tok, texts, plens, enc)
        with ResidualCapture(BLOCKS) as cap:
            model(**enc)
        got = cap.get()
        for L in range(NB):
            v = span_read(got[L], m, "mean").float()          # RMS-normalised, pooled
            acc[L].append((v[0::2] - v[1::2]).cpu())
            # R_L is the RAW residual norm over the scored span -- what the steering vector is
            # scaled against. span_read normalises, so it cannot supply this.
            h = got[L][:, :-1].float()
            mm = m.unsqueeze(-1).float()
            raw_norm[L].append(((h * mm).norm(dim=-1).sum() / mm.sum().clamp(min=1)).cpu())

VEC, RL, ACCU = {}, {}, {}
for L in range(NB):
    d = torch.cat(acc[L]).to(DEV)
    SD = d.std(0).clamp(min=1e-3)
    X = d / SD
    w = torch.zeros(X.shape[1], device=DEV, requires_grad=True)
    opt = torch.optim.Adam([w], lr=PROBE_LR)
    for _ in range(PROBE_STEPS):
        i = torch.randint(0, X.shape[0], (min(256, X.shape[0]),), device=DEV)
        opt.zero_grad()
        (-F.logsigmoid(X[i] @ w).mean() + PROBE_L2 * w.pow(2).sum()).backward()
        opt.step()
    w = w.detach()
    ACCU[L] = float(((X @ w) > 0).float().mean())
    # dm: raw-space class-mean difference (SD * mu). grad: mu / SD, circular by construction.
    v = (SD * w) if STEER_DIR == "dm" else (w / SD)
    VEC[L] = (v / v.norm()).to(torch.bfloat16)
    RL[L] = float(torch.stack(raw_norm[L]).mean())
    del d, X
torch.cuda.empty_cache()
print("[steer] probe acc / R_L: " + "  ".join(
    f"L{L}:{ACCU[L]:.3f}/{RL[L]:.0f}" for L in LAYERS), flush=True)

# ---------------------------------------------------------------- steer + generate
# ONE hook, on block L only, installed for the duration of a cell and removed after. The delta is
# broadcast over every position of the block's output -- ActAdd applied to the whole forward, which
# is what uf_steer_sweep.py did and what makes the two comparable.
def make_hook(delta):
    def f(mod, args, out):
        if isinstance(out, tuple):
            return (out[0] + delta,) + out[1:]
        return out + delta
    return f


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


def run_cell(tag, Ls, alpha):
    """Ls is a LIST of layers steered simultaneously (length 1 in the single-layer sweep)."""
    path = f"{OUT}/famgen_{tag}.json"
    if os.path.exists(path):
        print(f"== {tag} already done", flush=True)
        return
    k = len(Ls) if SCALE == "div_n" else 1
    hs = [BLOCKS[L].register_forward_hook(make_hook((alpha * RL[L] / k) * VEC[L])) for L in Ls]
    try:
        rec = {"arm": tag, "adapters": [],
               "steer": dict(layers=Ls, alpha=alpha, scale=SCALE, dir=STEER_DIR,
                             R_L=[RL[L] for L in Ls], probe_acc=[ACCU[L] for L in Ls]),
               "families": {}}
        for fam, ps in PROMPTS.items():
            outs = gen(ps)
            sc = score(outs, REX)
            own = sc.get(fam) or {}
            rec["families"][fam] = {
                "n": len(outs), "marker_scores": sc, "own_family": own,
                "gens": [{"prompt": p["prompt"], "gen": o} for p, o in zip(ps, outs)]}
            # `style` is a REGISTER family: pf_famlex has no marker pairs for it, so the lexical
            # meter is None there by construction and the judge is the only read. Do not read a
            # missing brit_rate as a null result.
            br, den = own.get("brit_rate"), own.get("density")
            print(f"  [{tag}] {fam:<13} brit_rate {br if br is None else f'{br:.3f}'} "
                  f"density {den if den is None else f'{den:.2f}'} "
                  f"len {sum(len(o) for o in outs) / max(1, len(outs)):.0f}", flush=True)
    finally:
        for h in hs:
            h.remove()
    json.dump(rec, open(path, "w"), indent=1)


if MULTI:
    name = TAG_PREFIX or ("all" if len(LAYERS) > 6 else "band")
    for a in ALPHAS:
        run_cell(f"steer_{name}{'_dn' if SCALE == 'div_n' else ''}_a{a}", LAYERS, a)
else:
    for L in LAYERS:
        for a in ALPHAS:
            run_cell(f"steer_L{L}_a{a}", [L], a)
print("STEERDONE", flush=True)
