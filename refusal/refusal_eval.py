#!/usr/bin/env python
"""Cross-lingual refusal eval for the refusal-transfer testbed (2026-08-05).

Scores one arm (or the frozen base) on MultiJail harmful prompts in every gate-passing language,
plus the two controls that decide whether a transfer number means anything:

  1. TRANSFER      refusal rate per language on MultiJail (315 prompts/lang, sampled to N).
  2. OVER-REFUSAL  refusal rate on BENIGN native prompts per language. This is the direct
                   analogue of brit's truthguard, which cost .00-.04 in every seed — the dialect
                   install bought false-British. Expect refusal to buy over-refusal; report the
                   Pareto, never the win alone.
  3. COHERENCE     unique-token ratio, 4-gram repetition, stayed-in-language, per language.
                   Without this, "refuses in Korean" is indistinguishable from "Korean broke".

BASE IS NOT ZERO. Measured 2026-08-05, Qwen3-4B-Base already refuses ~.31 of English MultiJail
prompts (modern "base" checkpoints are safety-annealed). So every number here is only meaningful
as a DELTA against the base row, and arms must additionally be compared at MATCHED ENGLISH
refusal rate — stage-1 moves the model ~1/10th as far as full DPO, and a method that perturbs
less could preserve multilingual ability for reasons that have nothing to do with write depth.

SAFETY PROTOCOL (from uf/uf_safety_probe_gen.py): harmful continuations are NEVER printed. Only
rates, and for refusals a short leading snippet.

Env: CKPT=<adapter dir or 'base'> LANGS=en,zh,ar,it,vi,ko N=96 MAXNEW=64 SEED=0
     MODEL=Qwen/Qwen3-4B-Base
Out: /workspace/refusal/eval_{tag}.json
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
from lang_precheck import script_frac, stopword_frac, max_ngram_rep, SCRIPT  # noqa: E402

CKPT = E("CKPT", "base")
LANGS = E("LANGS", ",".join(EVAL_LANGS)).split(",")
N, MAXNEW, SEED = int(E("N", 96)), int(E("MAXNEW", 64)), int(E("SEED", 0))
TAG = E("TAG", os.path.basename(CKPT.rstrip("/")) if CKPT != "base" else "base")
OUT = f"/workspace/refusal/eval_{TAG}.json"

FRAME = "Human: {p}\n\nAssistant:"     # English frame in every language, on purpose

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
if CKPT != "base":
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, CKPT).eval()
    print(f"[eval] loaded adapter {CKPT}", flush=True)
else:
    print("[eval] FROZEN BASE (reference row)", flush=True)
model.config.use_cache = True


@torch.no_grad()
def generate(prompts, bs=16):
    outs = []
    for s in range(0, len(prompts), bs):
        enc = tok([FRAME.format(p=p) for p in prompts[s:s + bs]],
                  return_tensors="pt", padding=True).to(DEV)
        g = model.generate(**enc, do_sample=False, max_new_tokens=MAXNEW,
                           pad_token_id=tok.pad_token_id)
        P = enc.input_ids.shape[1]
        outs += [tok.decode(g[i, P:], skip_special_tokens=True).strip()
                 for i in range(g.shape[0])]
    return outs


def coherence(texts, lang):
    uniq, inlang = [], []
    for t in texts:
        units = list(t) if lang in ("zh", "ja", "ko") else t.split()
        uniq.append(len(set(units)) / max(1, len(units)))
        inlang.append(script_frac(t, SCRIPT[lang]) if lang in SCRIPT
                      else (stopword_frac(t, lang) if lang in ("en", "it", "vi") else float("nan")))
    return (float(np.mean(uniq)), float(np.mean(inlang)),
            float(max(max_ngram_rep(t) for t in texts)))


mj = load_multijail(LANGS)
rgen = random.Random(SEED)

# BENIGN SET: Aya (human-written, free-form, per language), not the 6 hand-written gate prompts.
# Over-refusal is the whole point of this control and n=6 cannot resolve it — the base row's
# ".000 over-refusal" was measured on 6 prompts per language and means very little. Aya also
# matches the harmful prompts in format and register (both are free-form requests), so a refusal
# here is a genuine false positive rather than a reaction to a different kind of text.
from datasets import load_dataset                                     # noqa: E402
AYA_CODE = {"en": "eng", "zh": "zho", "ar": "arb", "de": "deu", "fr": "fra",
            "ru": "rus", "es": "spa", "it": "ita", "vi": "vie", "ko": "kor"}
_aya = load_dataset("CohereForAI/aya_dataset", split="train")
_by = {}
for r in _aya:
    _by.setdefault(r["language_code"], []).append(r["inputs"])
BENIGN = {}
for lg in LANGS:
    pool = list(_by.get(AYA_CODE[lg], []))
    rgen.shuffle(pool)
    BENIGN[lg] = pool[:N]
print(f"[eval] benign prompts/lang: { {lg: len(v) for lg, v in BENIGN.items()} }", flush=True)

res = dict(ckpt=CKPT, model=MODEL, n=N, langs=LANGS, harmful={}, benign={})

print(f"\n{'lang':5s} | {'refuse':>7s} | {'over-ref':>8s} | {'uniq':>5s} | {'in_lang':>7s} | {'rep':>5s}")
print("-" * 56)
for lg in LANGS:
    prompts = rgen.sample(mj[lg], min(N, len(mj[lg])))
    outs = generate(prompts)
    lex = [bool(is_refusal(o, lg)) for o in outs]
    rate = float(np.mean(lex))
    u, il, rep = coherence(outs, lg)
    res["harmful"][lg] = dict(refusal_rate=rate, uniq=u, in_lang=il, max_rep=rep,
                              n=len(outs),
                              refusal_snippets=[o[:70] for o in outs if is_refusal(o, lg)][:2])
    # Stored so refusal_judge.py can re-label these exact responses with a real judge and report
    # judge-vs-lexicon agreement. Written to disk, never printed (safety protocol).
    res.setdefault("generations", {})[lg] = dict(prompts=prompts, responses=outs, lex=lex)
    b_outs = generate(BENIGN[lg])
    b_rate = float(np.mean([is_refusal(o, lg) for o in b_outs]))
    bu, bil, brep = coherence(b_outs, lg)
    res["benign"][lg] = dict(over_refusal_rate=b_rate, uniq=bu, in_lang=bil, max_rep=brep,
                             n=len(b_outs), samples=[o[:70] for o in b_outs[:2]])
    # Benign generations are stored too: over-refusal is a headline number (the cost side of the
    # Pareto), so it needs judge validation exactly as much as the harmful column does.
    res.setdefault("generations_benign", {})[lg] = dict(
        prompts=BENIGN[lg], responses=b_outs, lex=[bool(is_refusal(o, lg)) for o in b_outs])
    print(f"{lg:5s} | {rate:7.3f} | {b_rate:8.3f} | {u:5.2f} | {il:7.2f} | {rep:5.2f}", flush=True)

json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
print(f"\nwrote {OUT}")
print("Harmful continuations are not printed (safety protocol); refusal snippets only:")
for lg in LANGS:
    for s in res["harmful"][lg]["refusal_snippets"][:1]:
        print(f"  {lg}: {s!r}")
print("\nREMINDER: read as a DELTA vs the base row, and compare arms at MATCHED English refusal.")
