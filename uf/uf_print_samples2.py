#!/usr/bin/env python
"""Training-set pair browser: prompt + dataset chosen/rejected (+ probe soft label for the pair)
+ base and adapter generations (+ frozen-probe rewards). For eyeballing what the preference
actually is and whether an adapter moved toward it.
Env: LORA=<adapter dir> N_SHOW=4 MAX_NEW=180 SEED=0 CLIP=700 OUT=/workspace/samples_pairs.txt"""
import os, sys, json, random, hashlib
from itertools import islice
import numpy as np
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import train_bayes_head, ResidualCapture

E = os.environ.get
MODEL = E("UF_SFT_MODEL", "allenai/Llama-3.1-Tulu-3-8B-SFT")
LORA, N_SHOW, MAX_NEW = E("LORA"), int(E("N_SHOW", 4)), int(E("MAX_NEW", 180))
MAX_LEN, SEED, CLIP = 1024, int(E("SEED", 0)), int(E("CLIP", 700))
OUT = E("OUT", "/workspace/samples_pairs.txt")
DEV = "cuda"
torch.manual_seed(SEED)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"; tok.truncation_side = "left"
def _msgs(p, r): return [{"role": "user", "content": p}, {"role": "assistant", "content": r}]
def render_full(p, r): return tok.apply_chat_template(_msgs(p, r), tokenize=False, add_generation_prompt=False)
def render_prompt(p):  return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
def _phash(s): return hashlib.sha1(s.encode()).hexdigest()

ds = load_dataset("allenai/ultrafeedback_binarized_cleaned", split="train_prefs", streaming=True)
train = []
for ex in islice(ds, 20000):
    ch, rj = ex.get("chosen"), ex.get("rejected")
    if not ch or not rj: continue
    p = ex.get("prompt") or ch[0]["content"]
    c, r = ch[-1]["content"], rj[-1]["content"]
    if not (p and c and r) or c == r: continue
    sc, sr = ex.get("score_chosen"), ex.get("score_rejected")
    if sc is None or sr is None or float(sc) - float(sr) < 1.0: continue
    if int(_phash(p)[:8], 16) % 10 == 0: continue
    train.append(dict(prompt=p, chosen=c, rejected=r, sc=float(sc), sr=float(sr)))
pairs = random.Random(SEED + 21).sample(train, N_SHOW)

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
BLOCKS = list(model.model.layers); HID = model.config.hidden_size

zf = np.load("/workspace/uf_probe_feats_lenmatch.npz")
Fc, Fr = zf["a"][:, 12], zf["b"][:, 12]
pool = np.concatenate([Fc, Fr]); sd, mn = pool.std(0) + 1e-6, pool.mean(0)
rng = np.random.RandomState(0)
s_tr = np.where(rng.rand(len(Fc)) < 0.5, 1.0, -1.0).astype(np.float32)
_, head, _ = train_bayes_head(((Fc - Fr) / sd) * s_tr[:, None], s_tr,
                              ((Fc - Fr) / sd)[:64] * s_tr[:64, None], s_tr[:64])
MU = head.mu.detach().float().to(DEV); SIG2 = F.softplus(head.rho.detach()).float().pow(2).to(DEV)
SD = torch.tensor(sd, device=DEV); MN = torch.tensor(mn, device=DEV)

@torch.no_grad()
def feats(texts):
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(DEV)
    with ResidualCapture([BLOCKS[12]]) as cap:
        model(**enc)
    return cap.get()[0][:, -1]

def reward(f):
    fs = (f.float() - MN) / SD
    s2 = fs.pow(2).matmul(SIG2)
    return torch.special.ndtr((fs.matmul(MU) - 0.5 * torch.sqrt(s2 + 1e-9)) / torch.sqrt(1 + s2))

def pair_p(prompt, c, r):
    f = feats([render_full(prompt, c), render_full(prompt, r)])
    fs = ((f[0] - f[1]).float() / SD)
    s2 = fs.pow(2).matmul(SIG2)
    return float(torch.special.ndtr(fs.matmul(MU) / torch.sqrt(1 + s2)))

policy = PeftModel.from_pretrained(model, LORA).eval()

@torch.no_grad()
def gen(prompt, use_adapter):
    enc = tok(render_prompt(prompt), return_tensors="pt", truncation=True, max_length=512).to(DEV)
    torch.manual_seed(SEED + 777)
    policy.base_model.model.config.use_cache = True
    import contextlib
    with (contextlib.nullcontext() if use_adapter else policy.disable_adapter()):
        g = policy.generate(**enc, do_sample=True, temperature=1.0, max_new_tokens=MAX_NEW,
                            pad_token_id=tok.pad_token_id)
    return tok.decode(g[0, enc.input_ids.shape[1]:], skip_special_tokens=True)

def clip(s): return s[:CLIP] + (" [...]" if len(s) > CLIP else "")

lines = []
for i, x in enumerate(pairs):
    p = pair_p(x["prompt"], x["chosen"], x["rejected"])
    b = gen(x["prompt"], False); a = gen(x["prompt"], True)
    rb = float(reward(feats([render_full(x["prompt"], b)]))[0])
    ra = float(reward(feats([render_full(x["prompt"], a)]))[0])
    lines.append(
        f"{'='*100}\nPAIR {i+1} | GPT4 scores {x['sc']:.0f} vs {x['sr']:.0f} | probe p(chosen wins)={p:.3f}\n"
        f"PROMPT: {clip(x['prompt'])}\n"
        f"{'-'*44} CHOSEN (dataset) {'-'*44}\n{clip(x['chosen'])}\n"
        f"{'-'*44} REJECTED (dataset) {'-'*42}\n{clip(x['rejected'])}\n"
        f"{'-'*44} BASE gen (r={rb:.3f}) {'-'*42}\n{clip(b)}\n"
        f"{'-'*44} ADAPTER gen (r={ra:.3f}) {'-'*40}\n{clip(a)}\n")
    print(lines[-1], flush=True)
open(OUT, "w").write("\n".join(lines))
print(f"saved {OUT}")
