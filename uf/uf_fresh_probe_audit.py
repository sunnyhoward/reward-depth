#!/usr/bin/env python
"""Fresh-probe collusion audit (Goodfire import + phase-4 open item): after training, is the
preference still decodable in the TRAINED POLICY's representation, and does the frozen base-fit
judge still read validly there?

For each adapter in CKPTS (name=path,...):
  1. extract chosen/rejected completion-end features THROUGH THE POLICY (adapter on) at the judge
     layer L_J (+ optionally L_PLAN, L_TOP) for the Stage-A pairs
  2. FROZEN-JUDGE validity: apply the base-fit judge head to the policy's features -> pairwise acc
     (Goodfire's "base-trained probes remain valid on the trained policy" check)
  3. FRESH-PROBE decodability: refit a new Bayesian head on the policy's features (same protocol
     as Stage A) -> held-out acc vs the base 0.79. Collapse => the representation the judge reads
     was destroyed rather than the preference installed.

Env: CKPTS="name=path,..." LAYERS="12,15,31" OUT=/workspace/uf_fresh_audit.json + funnel knobs"""
import os, sys, json, random, hashlib
from itertools import islice
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from peft.utils import set_peft_model_state_dict
from safetensors.torch import load_file

sys.path.insert(0, "/workspace/reward-depth")
from helpers import train_bayes_head, ResidualCapture

E = os.environ.get
MODEL = E("UF_SFT_MODEL", "allenai/Llama-3.1-Tulu-3-8B-SFT")
POOL, N_PROBE = int(E("UF_POOL", 20000)), int(E("N_PROBE", 3000))
MAX_LEN = int(E("MAX_LEN", 1024))
LAYERS = [int(x) for x in E("LAYERS", "12,15,31").split(",")]
N_FIT = int(E("N_FIT", 1500))    # pairs re-extracted per checkpoint (speed/precision trade)
DEV, SEED = "cuda", 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"; tok.truncation_side = "left"
def _msgs(p, r): return [{"role": "user", "content": p}, {"role": "assistant", "content": r}]
def render_full(p, r): return tok.apply_chat_template(_msgs(p, r), tokenize=False, add_generation_prompt=False)
def _phash(s): return hashlib.sha1(s.encode()).hexdigest()

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
MATCH, BUCKET = int(E("UF_MATCH_LENGTH", 1)), int(E("UF_LEN_BUCKET", 16))
def _rlen(s): return len(tok(s, add_special_tokens=False).input_ids)
if MATCH:
    from collections import defaultdict
    for x in recs: x["len_diff"] = _rlen(x["chosen"]) - _rlen(x["rejected"])
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
pr, pe = train[:N_PROBE], test[:400]
fit_pr = pr[:N_FIT]

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
BLOCKS = list(model.model.layers); HID = model.config.hidden_size
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
policy = get_peft_model(model, cfg).eval()

# base judge head at L_J from the Stage-A cache (identical to uf_hybrid3)
z = np.load("/workspace/uf_probe_feats_lenmatch.npz"); Fc_tr, Fr_tr, Fc_te, Fr_te = z["a"], z["b"], z["c"], z["d"]
assert Fc_tr.shape[0] == len(pr) and Fc_te.shape[0] == len(pe), "cache/funnel misalignment"
L_J = json.load(open("/workspace/uf_probe_curve_lenmatch.json"))["Lstar"]
w_pr = np.array([x["w"] for x in pr], np.float32); w_pe = np.array([x["w"] for x in pe], np.float32)
rng = np.random.RandomState(SEED)
s_tr = np.where(rng.rand(len(pr)) < 0.5, 1.0, -1.0).astype(np.float32)
s_te = np.where(rng.rand(len(pe)) < 0.5, 1.0, -1.0).astype(np.float32)
poolJ = np.concatenate([Fc_tr[:, L_J], Fr_tr[:, L_J]])
sdJ, mnJ = poolJ.std(0) + 1e-6, poolJ.mean(0)
jacc, jhead, _ = train_bayes_head(((Fc_tr[:, L_J] - Fr_tr[:, L_J]) / sdJ) * s_tr[:, None], s_tr,
                                  ((Fc_te[:, L_J] - Fr_te[:, L_J]) / sdJ) * s_te[:, None], s_te,
                                  w_tr=w_pr, w_te=w_pe)
MU_J = jhead.mu.detach().float()
print(f"[judge] base head L{L_J} held-out acc {jacc:.3f}", flush=True)

@torch.no_grad()
def policy_feats(xs, use_adapter=True, bs=8):
    """(N, len(LAYERS), hid) x 2 sides through the policy (adapter on/off)."""
    import contextlib
    cm = contextlib.nullcontext if use_adapter else policy.disable_adapter
    Fc = np.zeros((len(xs), len(LAYERS), HID), np.float32)
    Fr = np.zeros((len(xs), len(LAYERS), HID), np.float32)
    for side, out in (("chosen", Fc), ("rejected", Fr)):
        texts = [render_full(x["prompt"], x[side]) for x in xs]
        for s in range(0, len(texts), bs):
            enc = tok(texts[s:s + bs], return_tensors="pt", padding=True, truncation=True,
                      max_length=MAX_LEN).to(DEV)
            with cm(), ResidualCapture([BLOCKS[li] for li in LAYERS]) as cap:
                policy(**enc, logits_to_keep=1)
            buf = cap.get()
            for k in range(len(LAYERS)):
                out[s:s + enc.input_ids.shape[0], k] = buf[k][:, -1].float().cpu().numpy()
    return Fc, Fr

results = {"base_judge_acc": float(jacc), "layers": LAYERS}
_ck = E("CKPTS", "")
assert _ck, "set CKPTS=name=path,..."
for name, path in [kv.split("=", 1) for kv in _ck.split(",") if kv]:
    if not os.path.isdir(path): print(f"[skip] {path}", flush=True); continue
    sd = load_file(os.path.join(path, "adapter_model.safetensors"))
    set_peft_model_state_dict(policy, sd)
    print(f"[{name}] extracting policy features ({N_FIT} fit pairs + {len(pe)} eval pairs)...", flush=True)
    Pc_tr, Pr_tr = policy_feats(fit_pr)
    Pc_te, Pr_te = policy_feats(pe)
    r = {}
    for k, li in enumerate(LAYERS):
        # frozen-judge validity on policy features (only meaningful at L_J)
        entry = {}
        if li == L_J:
            d_te = torch.tensor((Pc_te[:, k] - Pr_te[:, k]) / sdJ, dtype=torch.float32)
            entry["frozen_judge_acc"] = float(((d_te.matmul(MU_J)) > 0).float().mean())
        # fresh probe on policy features (Stage-A protocol, this layer)
        sd_p = np.concatenate([Pc_tr[:, k], Pr_tr[:, k]]).std(0) + 1e-6
        dtr = ((Pc_tr[:, k] - Pr_tr[:, k]) / sd_p) * s_tr[:N_FIT, None]
        dte = ((Pc_te[:, k] - Pr_te[:, k]) / sd_p) * s_te[:, None]
        a, _, e = train_bayes_head(dtr, s_tr[:N_FIT], dte, s_te[:, ],
                                   w_tr=w_pr[:N_FIT], w_te=w_pe)
        entry.update(fresh_acc=float(a), fresh_elbo=float(e))
        r[f"L{li}"] = entry
        print(f"  [{name}] L{li}: fresh acc {a:.3f}" +
              (f" | frozen-judge acc {entry['frozen_judge_acc']:.3f}" if li == L_J else ""), flush=True)
    results[name] = r
json.dump(results, open(E("OUT", "/workspace/uf_fresh_audit.json"), "w"), indent=1)
print("DONE", flush=True)
