#!/usr/bin/env python
"""Stage 3 of the on-policy labeller loop: fit the pooled probe on judge-labelled ON-POLICY pairs,
then gate it on the comparison sampled RL actually makes.

The gate (all on held-out GATE prompts, disjoint by construction, judged with all-pairs):
  - NEW probe accuracy on judged gate pairs  — the labeller RL would actually use
  - OLD (dataset-fit) probe accuracy on the same pairs — phase-9 §4 predicts ~chance
  - both probes' accuracy on the 400 dataset test pairs (from the existing pooled cache) — does
    the on-policy fit RETAIN the dataset preference, or trade it away?
  - per-layer curve of the on-policy fit + corr(z, len_diff) — depth profile & residual length bias

Env: LEN_MATCH=1 LEN_BUCKET=8 MP_BS=16 (+ probe knobs)
Reads:  /workspace/uf_onpolicy_judged.jsonl, /workspace/uf_onpolicy_gate_judged.jsonl,
        /workspace/uf_probe_feats_meanpool.npz (dataset test pairs, banked cache)
Saves:  /workspace/uf_onpolicy_feats.npz, /workspace/uf_onpolicy_probe.json"""
import os, sys, json, random
import numpy as np
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/workspace/reward-depth")
from helpers import train_bayes_head, ResidualCapture

E = os.environ.get
MODEL = E("UF_SFT_MODEL", "allenai/Llama-3.1-Tulu-3-8B-SFT")
MAX_LEN, BS = int(E("MAX_LEN", 1024)), int(E("MP_BS", 16))
LEN_MATCH, BUCKET = int(E("LEN_MATCH", 1)), int(E("LEN_BUCKET", 8))
TOL = float(E("PLATEAU_TOL", 0.01))
DEV, SEED = "cuda", 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"; tok.truncation_side = "left"
def _msgs(p, r): return [{"role": "user", "content": p}, {"role": "assistant", "content": r}]
def render_full(p, r): return tok.apply_chat_template(_msgs(p, r), tokenize=False, add_generation_prompt=False)
def render_prompt(p):  return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
def _n_comp(p, r):
    nf = len(tok(render_full(p, r), add_special_tokens=False).input_ids)
    np_ = len(tok(render_prompt(p), add_special_tokens=False).input_ids)
    return max(1, min(nf - np_, min(nf, MAX_LEN)))

def load_pairs(f):
    out = [json.loads(l) for l in open(f)]
    # orient: w = winner text, l = loser text
    return [dict(prompt=x["prompt"], w=x[x["winner"]], l=x["b" if x["winner"] == "a" else "a"],
                 idx=x["idx"]) for x in out]
tr_pairs = load_pairs("/workspace/uf_onpolicy_judged.jsonl")
ga_pairs = load_pairs("/workspace/uf_onpolicy_gate_judged.jsonl")
print(f"[data] {len(tr_pairs)} judged train pairs | {len(ga_pairs)} gate pairs", flush=True)

def _rlen(s): return len(tok(s, add_special_tokens=False).input_ids)
for x in tr_pairs + ga_pairs: x["len_diff"] = _rlen(x["w"]) - _rlen(x["l"])
if LEN_MATCH:
    from collections import defaultdict
    cnt = defaultdict(lambda: [0, 0])
    for x in tr_pairs:
        b = int(round(x["len_diff"] / BUCKET))
        if b > 0: cnt[b][0] += 1
        elif b < 0: cnt[-b][1] += 1
    for x in tr_pairs:
        b = int(round(x["len_diff"] / BUCKET))
        if b == 0: x["wgt"] = 1.0; continue
        npos, nneg = cnt[abs(b)]
        x["wgt"] = 0.0 if (npos == 0 or nneg == 0) else min(npos, nneg) / (npos if b > 0 else nneg)
    tr_pairs = [x for x in tr_pairs if x["wgt"] > 0]
    wa = np.array([x["wgt"] for x in tr_pairs])
    print(f"[len-match] kept {len(tr_pairs)} | Kish ESS {wa.sum()**2/(wa**2).sum():.0f} | "
          f"judge len bias: winner longer in "
          f"{np.mean([x['len_diff'] > 0 for x in tr_pairs]):.2f} of pairs", flush=True)
else:
    for x in tr_pairs: x["wgt"] = 1.0

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
BLOCKS = list(model.model.layers); NL = len(BLOCKS); HID = model.config.hidden_size

@torch.no_grad()
def pooled_feats(pairs, side):
    texts = [render_full(x["prompt"], x[side]) for x in pairs]
    ncs = [_n_comp(x["prompt"], x[side]) for x in pairs]
    out = np.zeros((len(texts), NL, HID), np.float32)
    for s in range(0, len(texts), BS):
        enc = tok(texts[s:s + BS], return_tensors="pt", padding=True, truncation=True,
                  max_length=MAX_LEN).to(DEV)
        T = enc.input_ids.shape[1]
        m = torch.zeros(len(enc.input_ids), T, device=DEV)
        for i in range(enc.input_ids.shape[0]): m[i, T - min(ncs[s + i], T):] = 1.0
        denom = m.sum(1, keepdim=True).clamp(min=1.0)
        with ResidualCapture(BLOCKS) as cap:
            model(**enc, logits_to_keep=1)
        buf = cap.get()
        for li in range(NL):
            out[s:s + enc.input_ids.shape[0], li] = \
                ((buf[li].float() * m[:, :, None]).sum(1) / denom).cpu().numpy()
        del buf
        if (s // BS) % 25 == 0: print(f"  [{side}] {s}/{len(texts)}", flush=True)
    return out

cachef = "/workspace/uf_onpolicy_feats.npz"
if os.path.exists(cachef):
    z = np.load(cachef); Fw, Fl, Gw, Gl = z["a"], z["b"], z["c"], z["d"]
else:
    Fw = pooled_feats(tr_pairs, "w"); Fl = pooled_feats(tr_pairs, "l")
    Gw = pooled_feats(ga_pairs, "w"); Gl = pooled_feats(ga_pairs, "l")
    np.savez(cachef, a=Fw, b=Fl, c=Gw, d=Gl)

# dataset test pairs from the banked pooled cache (rows = pe[:400] chosen/rejected)
zd = np.load("/workspace/uf_probe_feats_meanpool.npz")
Dc, Dr = zd["c"], zd["d"]                      # (400, NL, HID)
Tc, Tr = zd["a"], zd["b"]                      # (3000, NL, HID) dataset train — for the OLD probe

w_tr = np.array([x["wgt"] for x in tr_pairs], np.float32)
rng = np.random.RandomState(SEED)
s_tr = np.where(rng.rand(len(tr_pairs)) < 0.5, 1.0, -1.0).astype(np.float32)
s_ga = np.where(rng.rand(len(ga_pairs)) < 0.5, 1.0, -1.0).astype(np.float32)
ld_ga = np.array([x["len_diff"] for x in ga_pairs], np.float32)

# ---- small antisymmetric MLP head (user request: linear vs nonlinear at fit time; take the
# MLP only if it clearly wins). f(d) = g(d) - g(-d) on the same scaled difference features —
# order-invariant by construction; scores single texts later the same way the linear head does,
# via the population-mean centering (score f against the mean). styc precedent: MLP == linear
# frontier (phase 8 §4); this retests on judge-labelled on-policy pairs. ----
import torch.nn as nn
def fit_mlp(dtr_u, s_tr_, w_tr_, hid=64, epochs=200, seed=0):
    """dtr_u: UNSIGNED difference features (winner - loser). Fits on a 90/10 split of train
    (early stop on the 10%), returns the net. Targets: winner side positive."""
    torch.manual_seed(seed)
    d = torch.tensor(dtr_u, dtype=torch.float32)
    w = torch.tensor(w_tr_, dtype=torch.float32)
    n = len(d); idx = torch.randperm(n)
    va, tr_i = idx[:max(1, n // 10)], idx[max(1, n // 10):]
    class Anti(nn.Module):
        def __init__(s):
            super().__init__(); s.g = nn.Sequential(nn.Linear(d.shape[1], hid), nn.ReLU(), nn.Linear(hid, 1))
        def forward(s, x): return (s.g(x) - s.g(-x)).squeeze(-1)
    net = Anti(); opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    best = dict(loss=1e9, state=None, wait=0)
    for ep in range(epochs):
        for sl in torch.randperm(len(tr_i)).split(256):
            i = tr_i[sl]; opt.zero_grad()
            (-(w[i] * F.logsigmoid(net(d[i]))).sum() / w[i].sum().clamp_min(1e-9)).backward()
            opt.step()
        with torch.no_grad():
            vl = float(-(w[va] * F.logsigmoid(net(d[va]))).sum() / w[va].sum().clamp_min(1e-9))
        if vl < best["loss"] - 1e-4:
            best.update(loss=vl, state={k: v.clone() for k, v in net.state_dict().items()}, wait=0)
        else:
            best["wait"] += 1
            if best["wait"] > 20: break
    net.load_state_dict(best["state"]); net.eval()
    return net

res = dict(n_train=len(tr_pairs), n_gate=len(ga_pairs),
           new_gate_acc=[], new_dataset_acc=[], corr_len=[], elbo=[],
           mlp_gate_acc=[], mlp_dataset_acc=[])
for li in range(NL):
    pool = np.concatenate([Fw[:, li], Fl[:, li]])
    sd = pool.std(0) + 1e-6
    dtr_u = (Fw[:, li] - Fl[:, li]) / sd                       # unsigned: winner - loser
    dga_u = (Gw[:, li] - Gl[:, li]) / sd
    dds_u = (Dc[:, li] - Dr[:, li]) / sd
    dtr = dtr_u * s_tr[:, None]
    dga = dga_u * s_ga[:, None]
    a, h, e = train_bayes_head(dtr, s_tr, dga, s_ga, w_tr=w_tr)
    with torch.no_grad():
        zg = h.z_s2(torch.tensor(dga_u, dtype=torch.float32))[0].numpy()
        zdset = h.z_s2(torch.tensor(dds_u, dtype=torch.float32))[0].numpy()
    net = fit_mlp(dtr_u, s_tr, w_tr)
    with torch.no_grad():
        mg = net(torch.tensor(dga_u, dtype=torch.float32)).numpy()
        md = net(torch.tensor(dds_u, dtype=torch.float32)).numpy()
    res["new_gate_acc"].append(float(a))
    res["new_dataset_acc"].append(float((zdset > 0).mean()))
    res["mlp_gate_acc"].append(float((mg > 0).mean()))
    res["mlp_dataset_acc"].append(float((md > 0).mean()))
    res["corr_len"].append(float(np.corrcoef(zg, ld_ga)[0, 1]) if ld_ga.std() > 0 else 0.0)
    res["elbo"].append(float(e))
    print(f"  L{li:2d} gate lin={a:.3f} mlp={res['mlp_gate_acc'][-1]:.3f} | "
          f"dataset lin={res['new_dataset_acc'][-1]:.3f} mlp={res['mlp_dataset_acc'][-1]:.3f} | "
          f"corr_len={res['corr_len'][-1]:+.3f}", flush=True)

# ---- OLD probe (dataset-fit, stage-A protocol) on the SAME gate pairs, at its L*=23 ----
curve = json.load(open("/workspace/uf_probe_curve_meanpool.json"))
LSTAR = int(curve["Lstar"])
s_d = np.where(rng.rand(Tc.shape[0]) < 0.5, 1.0, -1.0).astype(np.float32)
pool = np.concatenate([Tc[:, LSTAR], Tr[:, LSTAR]]); sd_o = pool.std(0) + 1e-6
_, h_old, _ = train_bayes_head(((Tc[:, LSTAR] - Tr[:, LSTAR]) / sd_o) * s_d[:, None], s_d,
                               ((Dc[:, LSTAR] - Dr[:, LSTAR]) / sd_o)[:100] * 1.0, np.ones(100, np.float32))
with torch.no_grad():
    z_old_gate = h_old.z_s2(torch.tensor((Gw[:, LSTAR] - Gl[:, LSTAR]) / sd_o, dtype=torch.float32))[0].numpy()
    z_old_dset = h_old.z_s2(torch.tensor((Dc[:, LSTAR] - Dr[:, LSTAR]) / sd_o, dtype=torch.float32))[0].numpy()
res["old_Lstar"] = LSTAR
res["old_gate_acc"] = float((z_old_gate > 0).mean())
res["old_dataset_acc"] = float((z_old_dset > 0).mean())
na = np.array(res["new_gate_acc"]); ma = np.array(res["mlp_gate_acc"])
res["new_Lstar"] = int(next(li for li in range(NL) if na[li] >= na.max() - TOL))
print(f"\n[GATE] old probe on judged on-policy pairs: {res['old_gate_acc']:.3f} "
      f"(dataset: {res['old_dataset_acc']:.3f})", flush=True)
print(f"[GATE] new linear max {na.max():.3f} @L{int(na.argmax())} (plateau L*={res['new_Lstar']}) | "
      f"dataset retention there: {res['new_dataset_acc'][int(na.argmax())]:.3f}", flush=True)
print(f"[GATE] new MLP max {ma.max():.3f} @L{int(ma.argmax())} | "
      f"mlp-vs-linear at linear's best layer: {ma[int(na.argmax())]:.3f} vs {na.max():.3f} "
      f"(gate n={len(ga_pairs)}, binomial SE ~{0.5/np.sqrt(max(1,len(ga_pairs))):.3f})", flush=True)
json.dump(res, open("/workspace/uf_onpolicy_probe.json", "w"), indent=1)
print("DONE", flush=True)
