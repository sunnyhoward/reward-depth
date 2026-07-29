#!/usr/bin/env python
"""Causal efficacy of the probe direction vs depth (notes_steering_experiment.md, run as designed).

Fit the Stage-A probe at EVERY layer, then steer the FROZEN base during generation by adding
alpha * R_L * v_L to block L's residual output at every position, for each (layer, alpha) cell.
Measure, per cell:
  - win-rate of steered vs unsteered generations under a real judge model (option 1 in the note:
    a local instruct model from a different family, pairwise A/B logprobs, both orders averaged
    to kill position bias)
  - KL/token vs the unsteered base on the steered tokens (the cost axis; efficacy should be read
    at matched KL, since residual norms grow with depth)
  - the full cross-layer probe matrix (steer at L, read z at every judge layer on a clean forward;
    option 2 -- reported whole so the circularity is visible instead of hidden)
  - non-probe proxies: generation length (option 3 sanity axis)

Pre-registered hypothesis (from the note, written before running): readable everywhere,
steerable only in the middle -- efficacy collapsing near the top where cos(mu, W_A - W_B) ~ 0.

DIRECTION: probes are fit on standardized difference features, z = mu . (f - MN)/SD. Two raw-space
mappings exist and the choice matters: STEER_DIR=dm (default) uses v ~ SD*mu -- the raw-space
class-mean-difference under the diagonal-covariance LDA identity, the standard ActAdd/CAA-style
steering vector; STEER_DIR=grad uses v ~ mu/SD -- the direction that moves the probe READ fastest
per unit raw norm (maximally circular by construction, so dm is the honest default). Unit-normed,
scaled by alpha * R_L where R_L = mean ||f|| over the Stage-A pool at layer L (relative units, so
alpha means the same fraction of residual magnitude at every depth).

Stages (STAGE=fit|gen|judge|all): gen writes generations + KL + probe matrix incrementally per
layer (survives death); judge reloads gens from disk, frees the policy model, loads the judge.
Needs /workspace/uf_probe_feats_lenmatch.npz (Stage A cache; regen ~30 min via uf_probe_rl.py).

Env: N_STEER=64 MAX_NEW=256 ALPHAS=0.03,0.1,0.3 LAYER_STRIDE=1 STEER_DIR=dm GREEDY=1
     JUDGE_MODEL=Qwen/Qwen2.5-7B-Instruct GEN_BS=64 JUDGE_BS=16 MAX_LEN=1024 PROMPT_LEN=512
Outputs: /workspace/uf_steer_sweep.json (metrics), /workspace/uf_steer_gens.json (texts)"""
import os, sys, json, random, hashlib
from itertools import islice
import numpy as np
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import train_bayes_head, ResidualCapture

E = os.environ.get
MODEL = E("UF_SFT_MODEL", "allenai/Llama-3.1-Tulu-3-8B-SFT")
JUDGE_MODEL = E("JUDGE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
POOL, N_PROBE = int(E("UF_POOL", 20000)), int(E("N_PROBE", 3000))
N_STEER, MAX_NEW = int(E("N_STEER", 64)), int(E("MAX_NEW", 256))
ALPHAS = [float(a) for a in E("ALPHAS", "0.03,0.1,0.3").split(",")]
STRIDE, SDIR, GREEDY = int(E("LAYER_STRIDE", 1)), E("STEER_DIR", "dm"), int(E("GREEDY", 1))
GEN_BS, JUDGE_BS = int(E("GEN_BS", 64)), int(E("JUDGE_BS", 16))
MAX_LEN, PLEN = int(E("MAX_LEN", 1024)), int(E("PROMPT_LEN", 512))
STAGE = E("STAGE", "all")
DEV = "cuda"; SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
METF = E("STEER_METF", "/workspace/uf_steer_sweep.json")
GENF = E("STEER_GENF", "/workspace/uf_steer_gens.json")

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"; tok.truncation_side = "left"
def _msgs(p, r): return [{"role": "user", "content": p}, {"role": "assistant", "content": r}]
def render_full(p, r): return tok.apply_chat_template(_msgs(p, r), tokenize=False, add_generation_prompt=False)
def render_prompt(p):  return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
def _phash(s): return hashlib.sha1(s.encode()).hexdigest()

# ---- data funnel + length matching (identical to uf_probe_rl.py so cache rows line up) ----
ds = load_dataset(E("UF_DATASET", "allenai/ultrafeedback_binarized_cleaned"),
                  split=E("UF_SPLIT", "train_prefs"), streaming=True)
recs = []
for ex in islice(ds, POOL):
    ch, rj = ex.get("chosen"), ex.get("rejected")
    if not ch or not rj: continue
    p = ex.get("prompt") or ch[0]["content"]
    c, r = ch[-1]["content"], rj[-1]["content"]
    if not (p and c and r) or c == r: continue
    sc, sr = ex.get("score_chosen"), ex.get("score_rejected")
    if sc is None or sr is None or float(sc) - float(sr) < 1.0: continue
    recs.append(dict(prompt=p, chosen=c, rejected=r, is_test=int(_phash(p)[:8], 16) % 10 == 0))
from collections import defaultdict
BUCKET = int(E("UF_LEN_BUCKET", 16))
def _rlen(s): return len(tok(s, add_special_tokens=False).input_ids)
for x in recs:
    x["len_diff"] = _rlen(x["chosen"]) - _rlen(x["rejected"])
cnt = defaultdict(lambda: [0, 0])
for x in recs:
    b = int(round(x["len_diff"] / BUCKET))
    if b > 0: cnt[b][0] += 1
    elif b < 0: cnt[-b][1] += 1
for x in recs:
    b = int(round(x["len_diff"] / BUCKET))
    if b == 0: x["w"] = 1.0; continue
    npos, nneg = cnt[abs(b)]
    x["w"] = 0.0 if (npos == 0 or nneg == 0) else min(npos, nneg) / (npos if b > 0 else nneg)
recs = [x for x in recs if x["w"] > 0]
train = [x for x in recs if not x["is_test"]]
test = [x for x in recs if x["is_test"]]
steer_prompts = [x["prompt"] for x in test[400:400 + N_STEER]]   # clear of the probe-eval rows
print(f"[data] {len(recs)} pairs | steering on {len(steer_prompts)} held-out prompts", flush=True)

# ---- fit probes at every layer from the Stage-A cache ----
cachef = E("UF_FEATS_CACHE", "/workspace/uf_probe_feats_lenmatch.npz")
z = np.load(cachef); Fc_tr, Fr_tr, Fc_te, Fr_te = z["a"], z["b"], z["c"], z["d"]
NL, HID = Fc_tr.shape[1], Fc_tr.shape[2]
pr = train[:N_PROBE]; pe = test[:400]
w_pr = np.array([x["w"] for x in pr], np.float32)[:len(Fc_tr)]
w_pe = np.array([x["w"] for x in pe], np.float32)[:len(Fc_te)]
rng = np.random.RandomState(SEED)
s_tr = np.where(rng.rand(len(Fc_tr)) < 0.5, 1.0, -1.0).astype(np.float32)
s_te = np.where(rng.rand(len(Fc_te)) < 0.5, 1.0, -1.0).astype(np.float32)
acc = np.zeros(NL)
MU = torch.zeros(NL, HID); SD = torch.zeros(NL, HID); MN = torch.zeros(NL, HID); RN = torch.zeros(NL)
for li in range(NL):
    pool = np.concatenate([Fc_tr[:, li], Fr_tr[:, li]])
    sd, mn = pool.std(0) + 1e-6, pool.mean(0)
    dtr = ((Fc_tr[:, li] - Fr_tr[:, li]) / sd) * s_tr[:, None]
    dte = ((Fc_te[:, li] - Fr_te[:, li]) / sd) * s_te[:, None]
    a, h, e = train_bayes_head(dtr, s_tr, dte, s_te, w_tr=w_pr, w_te=w_pe)
    acc[li] = a
    MU[li] = h.mu.detach().float().cpu(); SD[li] = torch.tensor(sd); MN[li] = torch.tensor(mn)
    RN[li] = float(np.linalg.norm(pool, axis=1).mean())    # reference residual norm at layer L
    print(f"  L{li:2d} acc={a:.3f}", flush=True)
MU, SD, MN, RN = MU.to(DEV), SD.to(DEV), MN.to(DEV), RN.to(DEV)
raw = (SD * MU) if SDIR == "dm" else (MU / SD)
VHAT = raw / raw.norm(dim=1, keepdim=True)                  # unit steering direction per layer
json.dump(dict(layer_acc=acc.tolist(), steer_dir=SDIR), open("/workspace/uf_steer_probe_curve.json", "w"))
if STAGE == "fit": sys.exit(0)

LAYERS = list(range(0, NL, STRIDE))
mets = json.load(open(METF)) if os.path.exists(METF) else dict(
    alphas=ALPHAS, steer_dir=SDIR, greedy=GREEDY, layer_acc=acc.tolist(), cells=[])
gens = json.load(open(GENF)) if os.path.exists(GENF) else dict(prompts=steer_prompts, cells=[])
done = {(c["L"], c["alpha"]) for c in mets["cells"]}

# ---- stage gen: steered generation + KL + cross-layer probe matrix ----
if STAGE in ("gen", "all"):
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
    BLOCKS = list(model.model.layers)

    class Steer:
        """Adds vec to block L's residual-stream output (the ResidualCapture hook point)."""
        def __init__(self, L, vec): self.h = BLOCKS[L].register_forward_hook(
            lambda m, i, o: (o[0] + vec.to(o[0].dtype),) + tuple(o[1:]) if isinstance(o, tuple) else o + vec.to(o.dtype))
        def remove(self): self.h.remove()

    @torch.no_grad()
    def generate(vec_L=None, vec=None):
        outs = []
        for s in range(0, len(steer_prompts), GEN_BS):
            enc = tok([render_prompt(p) for p in steer_prompts[s:s + GEN_BS]], return_tensors="pt",
                      padding=True, truncation=True, max_length=PLEN).to(DEV)
            hk = Steer(vec_L, vec) if vec is not None else None
            gen = model.generate(**enc, do_sample=not GREEDY, temperature=None if GREEDY else 1.0,
                                 max_new_tokens=MAX_NEW, pad_token_id=tok.pad_token_id)
            if hk: hk.remove()
            P = enc.input_ids.shape[1]
            outs += [tok.decode(gen[i, P:], skip_special_tokens=True) for i in range(gen.shape[0])]
        return outs

    @torch.no_grad()
    def kl_and_matrix(comps, vec_L=None, vec=None):
        """On the given completions: KL/token (steered vs clean logprobs, teacher-forced on the
        steered tokens) and the mean per-judge-layer centered probe read z at the re-render
        sentinel (clean forward -- the judge read never sees the steering vector)."""
        kls, lens, zs = [], [], []
        for s in range(0, len(comps), 8):     # chunk 8: two full-vocab log_softmaxes live at once
            chunk = [(steer_prompts[s + i], comps[s + i]) for i in range(min(8, len(comps) - s))]
            full = tok([render_full(p, c) for p, c in chunk], return_tensors="pt", padding=True,
                       truncation=True, max_length=MAX_LEN).to(DEV)
            pls = [tok(render_prompt(p), return_tensors="pt", truncation=True,
                       max_length=PLEN).input_ids.shape[1] for p, _ in chunk]
            with ResidualCapture(BLOCKS) as cap:
                out_c = model(**full)                      # clean: probe matrix + ref logprobs
            buf = cap.get()
            fs = torch.stack([buf[li][:, -1].float() for li in range(NL)], 1)   # (B, NL, HID)
            zs.append((((fs - MN) / SD) * MU).sum(-1).cpu())                    # (B, NL)
            lsm_c = F.log_softmax(out_c.logits[:, :-1].float(), -1)
            if vec is not None:
                hk = Steer(vec_L, vec)
                lsm_s = F.log_softmax(model(**full).logits[:, :-1].float(), -1)
                hk.remove()
            else:
                lsm_s = lsm_c
            ids, am = full.input_ids, full.attention_mask
            T = ids.shape[1]
            for i in range(len(chunk)):
                npad = int(T - am[i].sum())               # left padding
                lo = npad + min(pls[i], int(am[i].sum()) - 1)
                lp_s = lsm_s[i, lo - 1:T - 1].gather(-1, ids[i, lo:, None]).squeeze(-1).sum()
                lp_c = lsm_c[i, lo - 1:T - 1].gather(-1, ids[i, lo:, None]).squeeze(-1).sum()
                n = max(T - lo, 1)
                kls.append(float((lp_s - lp_c) / n)); lens.append(_rlen(chunk[i][1]))
        return float(np.mean(kls)), float(np.mean(lens)), torch.cat(zs).mean(0).tolist()

    if not any(c["L"] == -1 for c in mets["cells"]):       # baseline cell (alpha=0), once
        base_comps = generate()
        kl0, len0, zmat0 = kl_and_matrix(base_comps)
        mets["cells"].append(dict(L=-1, alpha=0.0, kl_tok=kl0, len=len0, z_judge=zmat0))
        gens["cells"].append(dict(L=-1, alpha=0.0, comps=base_comps))
        json.dump(mets, open(METF, "w")); json.dump(gens, open(GENF, "w"))
        print(f"[base] len {len0:.0f}", flush=True)
    for L in LAYERS:
        for a in ALPHAS:
            if (L, a) in done: continue
            vec = a * RN[L] * VHAT[L]
            comps = generate(L, vec)
            kl, ln, zmat = kl_and_matrix(comps, L, vec)
            mets["cells"].append(dict(L=L, alpha=a, kl_tok=kl, len=ln, z_judge=zmat))
            gens["cells"].append(dict(L=L, alpha=a, comps=comps))
            print(f"  L{L:2d} a={a:g}: kl/tok {kl:.4f} len {ln:.0f} "
                  f"z@L {zmat[L]:+.2f}", flush=True)
        json.dump(mets, open(METF, "w")); json.dump(gens, open(GENF, "w"))   # bank per layer
    del model; torch.cuda.empty_cache()

# ---- stage judge: pairwise win-rate under an external judge, both orders ----
if STAGE in ("judge", "all"):
    mets = json.load(open(METF)); gens = json.load(open(GENF))
    base = next(c["comps"] for c in gens["cells"] if c["L"] == -1)
    jtok = AutoTokenizer.from_pretrained(JUDGE_MODEL)
    jtok.padding_side = "left"; jtok.truncation_side = "left"
    judge = AutoModelForCausalLM.from_pretrained(JUDGE_MODEL, dtype=torch.bfloat16).to(DEV).eval()
    IDA = jtok("A", add_special_tokens=False).input_ids[-1]
    IDB = jtok("B", add_special_tokens=False).input_ids[-1]
    def jprompt(p, ra, rb):
        q = ("Compare two AI assistant responses to the same user prompt and judge which is "
             "better overall (helpfulness, accuracy, clarity).\n\n### User prompt:\n" + p[:2000] +
             "\n\n### Response A:\n" + ra + "\n\n### Response B:\n" + rb +
             "\n\nAnswer with exactly one letter, A or B.")
        return jtok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                        add_generation_prompt=True)
    @torch.no_grad()
    def prefA(prompts_texts):
        out = []
        for s in range(0, len(prompts_texts), JUDGE_BS):
            enc = jtok(prompts_texts[s:s + JUDGE_BS], return_tensors="pt", padding=True,
                       truncation=True, max_length=4096).to(DEV)
            lg = judge(**enc).logits[:, -1].float()
            out += (lg[:, IDA] - lg[:, IDB]).cpu().tolist()
        return out
    for c in mets["cells"]:
        if c["L"] == -1 or "win" in c: continue
        comps = next(g["comps"] for g in gens["cells"] if g["L"] == c["L"] and g["alpha"] == c["alpha"])
        d1 = prefA([jprompt(p, s_, b_) for p, s_, b_ in zip(steer_prompts, comps, base)])
        d2 = prefA([jprompt(p, b_, s_) for p, s_, b_ in zip(steer_prompts, comps, base)])
        c["win"] = float(np.mean([(0.5 * ((a_ > 0) + (b_ < 0))) for a_, b_ in zip(d1, d2)]))
        c["judge_margin"] = float(np.mean([(a_ - b_) / 2 for a_, b_ in zip(d1, d2)]))
        print(f"  L{c['L']:2d} a={c['alpha']:g}: win {c['win']:.3f} margin {c['judge_margin']:+.2f}", flush=True)
        json.dump(mets, open(METF, "w"))
    json.dump(mets, open(METF, "w"))
print("DONE", flush=True)
