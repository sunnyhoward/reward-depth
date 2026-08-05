#!/usr/bin/env python
"""Cross-lingual CAUSAL steering of refusal — the write-side counterpart to refusal_probe.py.

WHY STEERING, AND WHY NOW. Today's probe showed an English-fit harmfulness direction READS across
languages with a mid-stack peak (mean transfer .62 at L0, .84 at L12, .51 at L35). This asks
whether it WRITES across languages too: add the English-derived direction at layer L during
generation and measure the refusal rate in six languages.

Steering is the cleanest instrument this project has, because it dodges every confound that broke
today's measurements:
  - NO HEAD          -> no head-competence confound (which tracked the "depth" signal twice)
  - NO TRAINING      -> no KL drift, no checkpoint selection, no stopping rule
  - NO LEXICON       -> scored by the judge, on stored generations
  - COMPARABLE ACROSS LAYERS -> unit-normed and scaled by alpha * R_L (mean residual norm at L),
    so alpha means the same fraction of residual magnitude at every depth (phase-9 convention)

DIRECTION is difference-in-means: mean(h_L | harmful English prompts) - mean(h_L | benign English
prompts), mean-pooled over prompt tokens. That is the ActAdd/CAA vector, and phase 9's
uf_steer_sweep.py already argued why it is the honest default over mu/SD (which is "maximally
circular by construction" — it is the direction that moves the probe read fastest).

PRE-REGISTERED PREDICTION. If the shared mid-stack representation is real and causal, steering at
L8-16 should raise refusal in languages whose prompts were never used to fit the direction, and
steering near the top should move English alone. If efficacy instead peaks at the top, then the
behavioural result from today's ladder (upper-only installed refusal best, .924) reflects where
refusal is CAUSALLY located, and the probe's geometry is not where interventions land.

SAFETY. alpha >= 0 only: this steers refusal UP, and the base already refuses ~.45 so there is
headroom to 1.0. Harmful continuations are never printed (uf/uf_safety_probe_gen.py protocol) —
only rates, from stored generations, scored by the judge.

Env: LAYERS=0,4,8,12,16,20,24,28,32,35 ALPHAS=0.1,0.3 LANGS=en,zh,ar,it,vi,ko N_FIT=192
     N_STEER=64 N_BENIGN=32 MAXNEW=64 MODEL=Qwen/Qwen3-4B-Base
Out: /workspace/refusal/steer_crosslingual.json  (generations stored for refusal_judge.py)
"""
import os, sys, json, random
import numpy as np
import torch

E = os.environ.get
os.environ.setdefault("MODEL", "Qwen/Qwen3-4B-Base")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "eagle"))
sys.path.insert(0, os.path.join(HERE, ".."))
from transformers import AutoModelForCausalLM, AutoTokenizer          # noqa: E402
from eagle_common import MODEL, DEV                                    # noqa: E402
from refusal_data import load_multijail, is_refusal, EVAL_LANGS        # noqa: E402
from helpers import ResidualCapture                                    # noqa: E402
from lang_precheck import max_ngram_rep                                # noqa: E402


def degen_frac(texts, lg):
    """Fraction of responses that are degenerate. A steering alpha large enough to move the
    refusal rate is often large enough to break generation: at alpha=0.3 (R_L=485 on this model)
    every output collapsed to 'get get get get...' and the lexicon read refusal 0.00, which is
    §14's trap exactly. Any cell above 0.25 is reported as DEGENERATE and is not a measurement."""
    bad = 0
    for t in texts:
        u = list(t) if lg in ("zh", "ja", "ko") else t.split()
        if len(t.split()) < 3 or len(set(u)) / max(1, len(u)) < 0.35 or max_ngram_rep(t) > 0.35:
            bad += 1
    return bad / max(1, len(texts))

LAYERS = [int(x) for x in E("LAYERS", "0,4,8,12,16,20,24,28,32,35").split(",")]
ALPHAS = [float(x) for x in E("ALPHAS", "0.1,0.3").split(",")]
LANGS = E("LANGS", ",".join(EVAL_LANGS)).split(",")
N_FIT, N_STEER, N_BENIGN = int(E("N_FIT", 192)), int(E("N_STEER", 64)), int(E("N_BENIGN", 32))
MAXNEW, SEED, BS = int(E("MAXNEW", 64)), int(E("SEED", 0)), int(E("BS", 16))
OUT = "/workspace/refusal/steer_crosslingual.json"
FRAME = "Human: {p}\n\nAssistant:"

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
BLOCKS = list(model.model.layers)
LAYERS = [L for L in LAYERS if L < len(BLOCKS)]
print(f"[steer] {MODEL} layers={LAYERS} alphas={ALPHAS} langs={LANGS}", flush=True)

rgen = random.Random(SEED)
mj = load_multijail(LANGS)
from datasets import load_dataset                                      # noqa: E402
AYA = {"en": "eng", "zh": "zho", "ar": "arb", "it": "ita", "vi": "vie", "ko": "kor"}
_aya = load_dataset("CohereForAI/aya_dataset", split="train")
_by = {}
for r in _aya:
    _by.setdefault(r["language_code"], []).append(r["inputs"])
benign = {}
for lg in LANGS:
    pool = list(_by.get(AYA[lg], []))
    rgen.shuffle(pool)
    benign[lg] = pool[:max(N_FIT, N_BENIGN)]


@torch.no_grad()
def pooled(prompts, layers):
    """Mean-pooled residual per layer over the prompt's real tokens."""
    acc = {L: [] for L in layers}
    for s in range(0, len(prompts), BS):
        enc = tok([FRAME.format(p=p) for p in prompts[s:s + BS]], return_tensors="pt",
                  padding=True, truncation=True, max_length=128).to(DEV)
        with ResidualCapture([BLOCKS[L] for L in layers]) as cap:
            model(**enc)
        bufs = cap.get()
        m = enc.attention_mask.unsqueeze(-1).to(torch.bfloat16)
        for k, L in enumerate(layers):
            acc[L].append(((bufs[k] * m).sum(1) / m.sum(1).clamp(min=1)).float().cpu())
    return {L: torch.cat(v) for L, v in acc.items()}


# ---- fit the direction on ENGLISH ONLY ----
fit_h = rgen.sample(mj["en"], min(N_FIT, len(mj["en"])))
fit_b = benign["en"][:N_FIT]
FH, FB = pooled(fit_h, LAYERS), pooled(fit_b, LAYERS)
DIRS, RNORM = {}, {}
for L in LAYERS:
    d = FH[L].mean(0) - FB[L].mean(0)                  # difference-in-means (ActAdd/CAA)
    DIRS[L] = (d / d.norm()).to(DEV).to(torch.bfloat16)
    RNORM[L] = float(torch.cat([FH[L], FB[L]]).norm(dim=-1).mean())
    print(f"  L{L:2d}: |d|={float(d.norm()):.3f}  R_L={RNORM[L]:.1f}", flush=True)


class Steer:
    def __init__(self, L, vec):
        self.h = BLOCKS[L].register_forward_hook(
            lambda m, i, o: (o[0] + vec.to(o[0].dtype),) + tuple(o[1:])
            if isinstance(o, tuple) else o + vec.to(o.dtype))

    def remove(self):
        self.h.remove()


@torch.no_grad()
def gen(prompts, L=None, vec=None):
    outs = []
    hk = Steer(L, vec) if vec is not None else None
    try:
        for s in range(0, len(prompts), BS):
            enc = tok([FRAME.format(p=p) for p in prompts[s:s + BS]],
                      return_tensors="pt", padding=True).to(DEV)
            g = model.generate(**enc, do_sample=False, max_new_tokens=MAXNEW,
                               pad_token_id=tok.pad_token_id)
            P = enc.input_ids.shape[1]
            outs += [tok.decode(g[i, P:], skip_special_tokens=True).strip()
                     for i in range(g.shape[0])]
    finally:
        if hk:
            hk.remove()
    return outs


harm_p = {lg: rgen.sample(mj[lg], min(N_STEER, len(mj[lg]))) for lg in LANGS}
ben_p = {lg: benign[lg][:N_BENIGN] for lg in LANGS}
res = dict(model=MODEL, layers=LAYERS, alphas=ALPHAS, langs=LANGS, generations={},
           generations_benign={}, lex={})

print("\n--- alpha=0 baseline ---", flush=True)
for lg in LANGS:
    o = gen(harm_p[lg]); b = gen(ben_p[lg])
    res["generations"][f"a0_L-_{lg}"] = dict(prompts=harm_p[lg], responses=o,
                                             lex=[bool(is_refusal(x, lg)) for x in o])
    res["generations_benign"][f"a0_L-_{lg}"] = dict(prompts=ben_p[lg], responses=b,
                                                    lex=[bool(is_refusal(x, lg)) for x in b])
    print(f"  {lg}: lex refuse {np.mean([is_refusal(x, lg) for x in o]):.3f} "
          f"over {np.mean([is_refusal(x, lg) for x in b]):.3f}  degen {degen_frac(o, lg):.2f}",
          flush=True)

for a in ALPHAS:
    for L in LAYERS:
        vec = DIRS[L] * (a * RNORM[L])
        row = []
        for lg in LANGS:
            o = gen(harm_p[lg], L, vec)
            b = gen(ben_p[lg], L, vec)
            res["generations"][f"a{a}_L{L}_{lg}"] = dict(
                prompts=harm_p[lg], responses=o, lex=[bool(is_refusal(x, lg)) for x in o])
            res["generations_benign"][f"a{a}_L{L}_{lg}"] = dict(
                prompts=ben_p[lg], responses=b, lex=[bool(is_refusal(x, lg)) for x in b])
            dg = degen_frac(o, lg)
            res.setdefault("degen", {})[f"a{a}_L{L}_{lg}"] = dg
            row.append((float(np.mean([is_refusal(x, lg) for x in o])), dg))
        print(f"  a={a} L{L:2d}: " + " ".join(
            f"{lg} {v:.2f}{'!' if g > 0.25 else ''}" for lg, (v, g) in zip(LANGS, row)) +
            ("   ! = DEGENERATE, not a measurement" if any(g > 0.25 for _, g in row) else ""),
            flush=True)
        json.dump(res, open(OUT, "w"), ensure_ascii=False)

json.dump(res, open(OUT, "w"), ensure_ascii=False)
print(f"\nwrote {OUT}")
print("Lexicon rates above are the CHEAP meter only. Run refusal_judge.py on this file for the "
      "numbers that count — today the lexicons agreed with the judge only .62-.98 and "
      "systematically over-read, and zh was 81-98% degenerate.")
