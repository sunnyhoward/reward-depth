#!/usr/bin/env python
"""The cost axis the discrimination metric cannot see: KL(base || steered) on GENERATIVE REPLAY.

Discrimination (harmful - benign refusal) says whether an intervention is SELECTIVE. It says
nothing about what the intervention does to the model's general distribution — a steering vector
could hold discrimination at +.64 while wrecking everything the eval set does not sample.

Steering has no training, so there is no KL term to add to an objective. What there is, is a
measurable cost per cell: run the frozen model's OWN replay samples (eagle_replay_2048x128.pt,
the corpus the generative-replay/K-FAC line was built on) through the model with the steering hook
active, and take KL(base || steered) on those tokens. That is general-distribution damage in the
same units the training runs report as kl_from_base.

It is also the direct test of a question §5 left open. For TRAINING, the replay distribution
turned out to be untouched by the stage-1 edit (replay KL .003 while task KL was 2.7), which is
why replay-based priors were blind to it. Steering adds a fixed vector to every position, so it
has no reason to respect that separation — if steering DOES move replay while training does not,
then the two interventions damage the model in different places, and the steering cost is real
where the training cost was invisible.

Pairs with the refusal numbers to give a proper Pareto: discrimination gained per nat of general
drift, per layer.

Env: LAYERS=0,4,8,12,16,19,20,24,28,32,35 ALPHAS=0.01,0.02,0.03,0.05,0.07 N_REPLAY=64 N_FIT=192
Out: /workspace/refusal/steer_replay_cost.json
"""
import os, sys, json, random
import numpy as np
import torch, torch.nn.functional as F

E = os.environ.get
os.environ.setdefault("MODEL", "Qwen/Qwen3-4B-Base")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "eagle"))
sys.path.insert(0, os.path.join(HERE, ".."))
from transformers import AutoModelForCausalLM, AutoTokenizer      # noqa: E402
from eagle_common import MODEL, DEV                               # noqa: E402
from refusal_data import load_multijail                           # noqa: E402
from helpers import ResidualCapture                               # noqa: E402

LAYERS = [int(x) for x in E("LAYERS", "0,4,8,12,16,19,20,24,28,32,35").split(",")]
ALPHAS = [float(x) for x in E("ALPHAS", "0.01,0.02,0.03,0.05,0.07").split(",")]
N_FIT, N_REPLAY, BS = int(E("N_FIT", 192)), int(E("N_REPLAY", 64)), int(E("BS", 8))
SEED = int(E("SEED", 0))
OUT = "/workspace/refusal/steer_replay_cost.json"
FRAME = "Human: {p}\n\nAssistant:"

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
BLOCKS = list(model.model.layers)
LAYERS = [L for L in LAYERS if L < len(BLOCKS)]

# Replay corpus: this model's own samples. Generated here if absent (the banked one is Qwen2.5-3B).
RF = E("REPLAY_F", "/workspace/refusal/replay_2048x128.pt")
assert os.path.exists(RF), f"missing {RF} — run refusal/refusal_head.py first (it writes this)"
replay = torch.load(RF).long()
print(f"[cost] replay {tuple(replay.shape)} from {RF}", flush=True)

rgen = random.Random(SEED)
mj = load_multijail(["en"])["en"]
from datasets import load_dataset                                  # noqa: E402
_aya = load_dataset("CohereForAI/aya_dataset", split="train")
ben = [r["inputs"] for r in _aya if r["language_code"] == "eng"]
rgen.shuffle(ben)


@torch.no_grad()
def pooled(prompts):
    acc = {L: [] for L in LAYERS}
    for s in range(0, len(prompts), BS):
        enc = tok([FRAME.format(p=p) for p in prompts[s:s + BS]], return_tensors="pt",
                  padding=True, truncation=True, max_length=128).to(DEV)
        with ResidualCapture([BLOCKS[L] for L in LAYERS]) as cap:
            model(**enc)
        bufs = cap.get()
        m = enc.attention_mask.unsqueeze(-1).to(torch.bfloat16)
        for k, L in enumerate(LAYERS):
            acc[L].append(((bufs[k] * m).sum(1) / m.sum(1).clamp(min=1)).float().cpu())
    return {L: torch.cat(v) for L, v in acc.items()}


FH = pooled(rgen.sample(mj, N_FIT))
FB = pooled(ben[:N_FIT])
DIRS, RN = {}, {}
for L in LAYERS:
    d = FH[L].mean(0) - FB[L].mean(0)
    DIRS[L] = (d / d.norm()).to(DEV).to(torch.bfloat16)
    RN[L] = float(torch.cat([FH[L], FB[L]]).norm(dim=-1).mean())


class Steer:
    def __init__(self, L, vec):
        self.h = BLOCKS[L].register_forward_hook(
            lambda m, i, o: (o[0] + vec.to(o[0].dtype),) + tuple(o[1:])
            if isinstance(o, tuple) else o + vec.to(o.dtype))

    def remove(self):
        self.h.remove()


@torch.no_grad()
def replay_kl(L, vec):
    """mean KL(base || steered) per token over replay sequences."""
    kls = []
    for s in range(0, N_REPLAY, BS):
        ids = replay[s:s + BS].to(DEV)
        enc = dict(input_ids=ids, attention_mask=torch.ones_like(ids))
        b = F.log_softmax(model(**enc).logits[:, :-1].float(), -1)
        hk = Steer(L, vec)
        try:
            p_ = F.log_softmax(model(**enc).logits[:, :-1].float(), -1)
        finally:
            hk.remove()
        kls.append(float((b.exp() * (b - p_)).sum(-1).mean()))
    return float(np.mean(kls))


res = dict(model=MODEL, layers=LAYERS, alphas=ALPHAS, replay_kl={})
print(f"\n{'L':>3s} | " + " | ".join(f"a={a:<5}" for a in ALPHAS))
print("-" * (6 + 9 * len(ALPHAS)))
for L in LAYERS:
    row = []
    for a in ALPHAS:
        k = replay_kl(L, DIRS[L] * (a * RN[L]))
        res["replay_kl"][f"a{a}_L{L}"] = k
        row.append(k)
    print(f"{L:3d} | " + " | ".join(f"{v:7.3f}" for v in row), flush=True)
    json.dump(res, open(OUT, "w"), indent=1)

json.dump(res, open(OUT, "w"), indent=1)
print(f"\nwrote {OUT}")
print("Pair with the judged refusal numbers for a Pareto: discrimination gained per nat of "
      "general drift. For reference, stage-1 TRAINING moved replay by only .003 nats while moving "
      "task text by 2.7 — if steering moves replay substantially, the two interventions damage "
      "the model in different places.")
