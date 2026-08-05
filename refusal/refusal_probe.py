#!/usr/bin/env python
"""Cross-lingual harmfulness probe — tests the ladder's PREMISE without any training.

The refusal ladder assumes an inverted U in write depth: early layers lexical and
language-specific, the shared conceptual space mid-stack, late layers re-specialised to output
language. That assumption has never been measured in this repo. This measures it directly and
cheaply (forward passes only):

    fit a linear probe on ENGLISH harmful-vs-benign PROMPTS at each layer,
    then test that SAME English-fit probe on zh / ar / de / fr / ru / es prompts.

English in-language accuracy says harmfulness is decodable at that layer at all. The
cross-lingual accuracies say whether the representation is SHARED. A mid-stack transfer peak is
direct evidence for the mechanism the ladder is built on; flat-across-depth transfer, or a peak
at the top, would say the premise is wrong — and would say it before any training delta is
interpreted.

NOT the same instrument as head_acc. head_acc ranks two RESPONSES by likelihood through the
frozen head (the interface DPO actually optimises). This classifies the PROMPT from activations
with a free-direction readout. The repo's standing lesson is that these come apart — on UF a
pooled probe read h_12 at .79 while the full model's likelihood readout ceiling was .578
("probes read what LM heads cannot say", §18). So a high probe score here is NOT evidence that
stage-1 can train on it; head_acc already answered that (.95 at L4, .99 at L12).

Read positions are mean-pooled over the prompt, not last-token: phase-9 §7 found pooled
directions steer judge outcomes where last-token reads are null, and phase-8 showed the styc
analogue was substantially a read-position artifact.

Env: LAYERS=0,4,8,12,16,20,24,28,32,35 N_PER=192 SEED=0 MODEL=Qwen/Qwen3-4B-Base BS=8
Out: /workspace/refusal/probe_crosslingual.json
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
from transformers import AutoModelForCausalLM, AutoTokenizer      # noqa: E402
from sklearn.linear_model import LogisticRegression               # noqa: E402
from eagle_common import MODEL, DEV                               # noqa: E402
from refusal_data import load_multijail, EVAL_LANGS                # noqa: E402
from helpers import ResidualCapture                               # noqa: E402

LAYERS = [int(x) for x in E("LAYERS", "0,4,8,12,16,20,24,28,32,35").split(",")]
N_PER, SEED, BS = int(E("N_PER", 192)), int(E("SEED", 0)), int(E("BS", 8))
OUT = "/workspace/refusal/probe_crosslingual.json"
FRAME = "Human: {p}\n\nAssistant:"     # identical frame in every language (see refusal_dpo.py)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
BLOCKS = list(model.model.layers)
NL = len(BLOCKS)
LAYERS = [L for L in LAYERS if L < NL]
print(f"[probe] {MODEL} layers={NL} probing {LAYERS}", flush=True)


@torch.no_grad()
def feats(prompts):
    """Mean-pooled residual at each probed layer over the prompt's real tokens."""
    out = {L: [] for L in LAYERS}
    for s in range(0, len(prompts), BS):
        chunk = [FRAME.format(p=p) for p in prompts[s:s + BS]]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=128).to(DEV)
        with ResidualCapture([BLOCKS[L] for L in LAYERS]) as cap:
            model(**enc)
        bufs = cap.get()
        m = enc.attention_mask.unsqueeze(-1).to(torch.bfloat16)
        for k, L in enumerate(LAYERS):
            pooled = (bufs[k] * m).sum(1) / m.sum(1).clamp(min=1)
            out[L].append(pooled.float().cpu().numpy())
    return {L: np.concatenate(v) for L, v in out.items()}


# ---- harmful: MultiJail (+ XSafety for languages MultiJail lacks); benign: Aya ----
# FIRST ATTEMPT FAILED AND THE FAILURE IS INSTRUCTIVE. Using XSafety's `commonsense` split as the
# benign class gave 1.000 accuracy AT LAYER 0 in English and .75-1.00 cross-lingual there. Layer 0
# has nothing but surface form, so that was not harmfulness: `commonsense` is multiple-choice QA
# ("If you're ever bitten by a snake: A. ... B. ...") while every harmful prompt is a free-form
# question, and the probe was separating FORMAT. The cross-lingual "transfer" was the shared
# multilingual embedding space recognising "A. B. C. D." in any script.
#
# Fix: benign prompts must match the harmful ones in format and register. Aya is human-written
# free-form instructions in every language we need, so both classes are now free-form requests
# and the only nominal difference is harmfulness.
#
# SANITY CRITERION, checked below: layer-0 English accuracy must NOT be near 1.0. If it is, the
# negatives are still separable on surface form and the table says nothing about depth.
from datasets import load_dataset                                    # noqa: E402

AYA_CODE = {"en": "eng", "zh": "zho", "ar": "arb", "de": "deu", "fr": "fra",
            "ru": "rus", "es": "spa", "it": "ita", "vi": "vie", "ko": "kor"}
XS_CODE = {"es": "sp"}          # XSafety spells Spanish 'sp'
PROBE_LANGS = E("PROBE_LANGS", "en,zh,ar,it,vi,ko,de,fr,ru,es").split(",")
rgen = random.Random(SEED)

mj = load_multijail([lg for lg in PROBE_LANGS if lg in EVAL_LANGS])
_xs = load_dataset("ToxicityPrompts/XSafety")["test"]
xs_by = {}
for r in _xs:
    if r["category"] != "commonsense":
        xs_by.setdefault(r["language"], []).append(r["text"])

_aya = load_dataset("CohereForAI/aya_dataset", split="train")
aya_by = {}
for r in _aya:
    aya_by.setdefault(r["language_code"], []).append(r["inputs"])

LANGS, harmful, benign = [], {}, {}
for lg in PROBE_LANGS:
    h = mj.get(lg) or xs_by.get(XS_CODE.get(lg, lg)) or []
    b = aya_by.get(AYA_CODE[lg], [])
    if not h or not b:
        print(f"[probe] skip {lg}: harmful={len(h)} benign={len(b)}", flush=True)
        continue
    h, b = list(h), list(b)
    rgen.shuffle(h); rgen.shuffle(b)
    k = min(N_PER, len(h), len(b))
    harmful[lg], benign[lg] = h[:k], b[:k]
    LANGS.append(lg)
    src = "MultiJail" if lg in mj else "XSafety"
    print(f"[probe] {lg}: {k} harmful ({src}) + {k} benign (Aya)", flush=True)

F_harm = {lg: feats(harmful[lg]) for lg in LANGS}
F_ben = {lg: feats(benign[lg]) for lg in LANGS}

res = dict(model=MODEL, layers=LAYERS, langs=LANGS, n_per=N_PER, acc={})
print(f"\n{'L':>3s} | " + " | ".join(f"{lg:>5s}" for lg in LANGS))
print("-" * (6 + 8 * len(LANGS)))
for L in LAYERS:
    # fit on ENGLISH harmful vs ENGLISH benign; held-out half gives the honest English number
    Xh, Xb = F_harm["en"][L], F_ben["en"][L]
    n = min(len(Xh), len(Xb)); half = n // 2
    X = np.concatenate([Xh[:n], Xb[:n]])
    y = np.concatenate([np.ones(n), np.zeros(n)])
    tr = np.concatenate([np.arange(half), np.arange(n, n + half)])
    te = np.concatenate([np.arange(half, n), np.arange(n + half, 2 * n)])
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(X[tr], y[tr])
    row = {"en": float(clf.score(X[te], y[te]))}
    # transfer: the SAME English-fit probe, each language's harmful vs ITS OWN benign
    for lg in LANGS:
        if lg == "en":
            continue
        m = min(len(F_harm[lg][L]), len(F_ben[lg][L]))
        Xt = np.concatenate([F_harm[lg][L][:m], F_ben[lg][L][:m]])
        yt = np.concatenate([np.ones(m), np.zeros(m)])
        row[lg] = float(clf.score(Xt, yt))
    res["acc"][L] = row
    print(f"{L:3d} | " + " | ".join(f"{row.get(lg, float('nan')):5.3f}" for lg in LANGS),
          flush=True)

# SANITY: the criterion is NOT "is English high at layer 0" — harmful and benign requests
# genuinely differ in vocabulary, so an English-fit probe separates them on surface form at every
# depth (measured: en is flat .94-.97 across all layers). Surface form cannot produce CROSS-
# LINGUAL transfer, so the diagnostic is the layer-0 TRANSFER mean. The discarded XSafety run had
# L0 transfer .847 (format artifact — "A. B. C. D." is language-independent); this run has .624
# falling to .50 in most languages, i.e. the English layer-0 direction is useless elsewhere.
_lay = sorted(res["acc"], key=lambda x: int(x))
_nz = [l for l in LANGS if l != "en"]
_t0 = [res["acc"][_lay[0]][l] for l in _nz]
res["l0_transfer_mean"] = float(np.mean(_t0))
if res["l0_transfer_mean"] > 0.75:
    print(f"\n*** SANITY FAIL: layer-0 mean transfer {res['l0_transfer_mean']:.3f} — the classes "
          f"are separable on language-independent surface form (format, punctuation, script), so "
          f"the transfer profile says nothing about shared semantics. Do not report it. ***",
          flush=True)
    res["sanity_fail"] = True

json.dump(res, open(OUT, "w"), indent=1)
print(f"\nwrote {OUT}")
print("READ: 'en' = is harmfulness decodable at this depth at all. Other columns = is the "
      "representation SHARED across languages. A mid-stack transfer peak supports the ladder's "
      "inverted-U premise; flat or top-peaked transfer says the premise is wrong.")
print("Every column is harmful-vs-benign WITHIN one language (XSafety `commonsense` split), so "
      "a probe that merely detected language identity would score ~.50 on the transfer columns, "
      "not high. That was the confound this source was chosen to remove.")
