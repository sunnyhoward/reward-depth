#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
probe_vs_dpo.py — PHASE 1 of the reward-depth program: put the Bayesian probe AT THE TOP
(reading the post-final-RMSNorm hidden state — the exact tensor the unembedding consumes) and
compare it head-to-head with DPO on the same pairs, same LoRA coverage (all blocks), same data.

Rationale (collaborator proposal + the isotropy argument): for near-isotropic unembedding rows,
averaging DPO's per-pair ΔW = W_yw − W_yl over many pairs leaves only the utility direction — so a
single linear reward head at the unembedding's input should induce a gradient flow ≈ DPO's. We
test that equivalence directly:
  · behavioral: transfer matrix + Goodhart curves for probe@final vs dpo (06_goodhart.png);
  · mechanistic: AB_GRADCOS — per-block cosine between the AVERAGE LoRA gradients of the probe
    margin loss and the DPO loss, computed on the SAME batches at the SAME policy state
    (07_gradcos.png).
PREDICTION (from the preface displacement study): high but sub-1 cosine — DPO's gradient carries a
softmax-normalization component (an implicit imitation pull on the chosen completion) that the
pairwise margin provably lacks; behaviorally, probe@final without an anchor should displace MORE
than DPO at matched flip, and adding the DPOP hinge should RAISE the cosine.
Phase 2 then moves the probe backward (AB_ATTACH=block, AB_COMPARE_L < N-1) — the original
Occam/elbow hypothesis, with DPO-equivalence at the top as the calibrated reference point.

Inherited from preface/ab_layer_sweep.py (same knobs; see that repo for the full study):
wrongness oracle data, Bayesian heads per layer, deep/deep_rl/dpo arms, sampled RLOO RL,
on-menu anchor, off-menu negatives, Goodhart instrumentation.

Run (phase 1):   AB_ATTACH=final AB_COMPARE=1 AB_COMPARE_L=<N-1> AB_COMPARE_MODES=deep,dpo \
                 AB_GRADCOS=8 AB_COMPARE_NOSTOP=1 python -u probe_vs_dpo.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PLOTS_DIR = os.environ.get("AB_PLOTS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots_ab_sweep"))
os.makedirs(PLOTS_DIR, exist_ok=True)

def _savefig(name):
    path = os.path.join(PLOTS_DIR, name)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close("all")
    print(f"[plot saved] {path}", flush=True)

print(f"[ab-sweep] plots will be written to: {PLOTS_DIR}", flush=True)

# ══════════════════════════════ Stage 0 — setup ══════════════════════════════
import importlib.util, sys, subprocess, json, re
for pkg, mod in (("peft","peft"), ("scikit-learn","sklearn")):
    if importlib.util.find_spec(mod) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)

import time, random, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

def _env (k, d):        return os.environ.get(k, d)
def _envi(k, d):
    try: return int(os.environ[k])
    except Exception: return d
def _envf(k, d):
    try: return float(os.environ[k])
    except Exception: return d
def _envb(k, d):
    v = os.environ.get(k)
    return d if v is None else v.strip().lower() in ("1","true","yes","on")

# Qwen2.5-1.5B: calibrated 2026-07-15 — A/B acc num=1.00 year=0.98 digits=0.90 smaller=0.84;
# free-form num=0.96 year=1.00; YN num/year=0.70. (0.5B base AND 0.5B-Instruct are at CHANCE on
# the A/B format — do not downsize.)
MODEL_NAME = _env("AB_MODEL", "Qwen/Qwen2.5-1.5B")

# Question types available for both lists (all except mcq_arith have Yes/No variants):
#   num, year, smaller, digits, wordlen, alpha, older, colder, money, sum
# Repeat a type in the list to weight it more heavily in the mixture. Default trains a
# DIRECTION CONTRAST (larger+smaller over the same attribute; money=max, colder=min) so a bare
# item-preference ("pick the lesser") cannot fit the training set — forcing a more abstract flip.
TRAIN_TYPES = _env("AB_TRAIN_TYPES", "num,smaller,year,money,colder").split(",")
OOD_TYPES   = _env("AB_OOD_TYPES", "digits,alpha").split(",")

N_TRAIN    = _envi("AB_N_TRAIN", 1000)    # questions → ×len(AB_TRAIN_FORMATS) pairs (+ knowledge)
N_EVAL     = _envi("AB_N_EVAL", 300)      # held-out same-type A/B questions
N_TRANSFER = _envi("AB_N_TRANSFER", 150)  # per transfer eval (OOD types / free / YN / MCQ / easy)
SEED       = _envi("AB_SEED", 0)

PRIOR_TAU   = _envf("AB_PRIOR_TAU", 0.1)
MAP_INIT    = _envi("AB_MAP_INIT_STEPS", 60)
MEM_CONTROL = _envb("AB_MEM_CONTROL", True)

# Default: the single good-probe layer L17 (the probe is ~chance in early layers, so a wide sweep
# there is uninformative). Set AB_SWEEP_L="a,b,c" to sweep, or "" for the auto 6-layer spread.
_sw = _env("AB_SWEEP_L", "17").strip()
SWEEP_L    = [int(x) for x in _sw.split(",") if x != ""] if _sw else None
N_SEEDS    = _envi("AB_SEEDS", 1)
STEPS      = _envi("AB_STEPS", 250)       # ~2k-pair margin converges by here (loss→~0.01)
BATCH      = _envi("AB_BATCH", 16)
LR         = _envf("AB_LR", 1e-4)
LORA_R     = _envi("AB_LORA_R", 8)
DPO_BETA   = _envf("AB_DPO_BETA", 0.1)
EVAL_EVERY = _envi("AB_EVAL_EVERY", 25)

SIGNAL      = _env("AB_SIGNAL", "head")   # head: train THROUGH the frozen layer-L probe | oracle: deterministic DPO
# answerend (default): UF-EXACT pairwise head — features at the END of prompt+completion, head
#   ranks (right ≻ wrong) pairs; training pairs in the AB_TRAIN_FORMATS formats, so free-form
#   wrongness is TRAINED, not just measured. promptend: original A/B-only head at "Answer:".
HEAD_READ     = _env("AB_HEAD_READ", "answerend")
TRAIN_FORMATS = [f for f in _env("AB_TRAIN_FORMATS", "ab,free").split(",") if f]
USE_RL      = _envb("AB_USE_RL", False)   # head mode: + REINFORCE the answer (blocks > L). OFF by default —
                                          # margin-only (trains blocks ≤ L) is stable and the clean "edit the
                                          # bottom of the net" intervention; RL adds letter-collapse instability.
MARGIN_COEF = _envf("AB_MARGIN_COEF", 1.0)
RL_COEF     = _envf("AB_RL_COEF", 1.0)    # base REINFORCE weight (scaled per-L below)
RL_KL       = _envf("AB_RL_KL", 0.1)      # UF-style KL leash on the RL candidates: penalize logπ−logπ_base
RL_ENT      = _envf("AB_RL_ENT", 0.05)    # entropy bonus on the 2-way answer dist
# REINFORCE owns blocks > L, i.e. (N_LAYERS-1-L) of them — a huge collapse lever at early L (many
# blocks) and a gentle nudge at late L (few). Scale its weight by the FRACTION of the stack it owns
# so mid layers get ~margin-only (stable) and late layers get near-full RL (needed). AB_RL_SCALE=0
# disables the scaling (fixed RL_COEF everywhere).
RL_SCALE    = _envb("AB_RL_SCALE", True)
# ── deep_rl coherence fixes (the off-menu pathology: the 2-candidate REINFORCE + its KL are both
# computed on the RENORMALIZED 2-way candidate distribution, so absolute mass draining off-menu is
# invisible to the objective and the centered advantage actively pushes the right candidate's
# absolute likelihood down — likelihood displacement, DPO's pathology reintroduced) ──
RL_ONMENU   = _envf("AB_RL_ONMENU", 0.0)  # >0: anchor ABSOLUTE on-menu mass — penalize the drop of
                                          # (logp_wrong + logp_right) below the base policy's value
                                          # (hinge), so displacement costs the RL owner directly
RL_WARMUP   = _envi("AB_RL_WARMUP", 0)    # steps of margin-only before RL turns on (sequential:
                                          # RL then distills an already-converged deep edit upward
                                          # instead of chasing a moving head from step 0)
RL_SAMPLE   = _envb("AB_RL_SAMPLE", False)# on-policy REINFORCE: SAMPLE completions from the policy
                                          # and score them with the head (reward = head ranks the
                                          # sample above the pair's right answer), sequence-level KL
                                          # to base — off-menu outputs become VISIBLE to the
                                          # objective and are taxed by the KL (base mass is on-menu)
RL_TEMP     = _envf("AB_RL_TEMP", 1.0)    # sampling temperature for AB_RL_SAMPLE
RL_BATCH    = _envi("AB_RL_BATCH", 0)     # sampled-RL: prompts per RL step (0 = full batch).
                                          # Decouples RL memory (B_rl×K grad rows) from the margin
                                          # batch — margin at batch 3 destabilizes (A/B letters
                                          # vanish mid-warmup), so keep AB_BATCH=6 and shrink this
RL_K        = _envi("AB_RL_K", 4)         # sampled-RL: samples per prompt; advantage baselined
                                          # leave-one-out within each prompt's group (RLOO) — a
                                          # batch-mean baseline across different prompts lets
                                          # prompt difficulty pollute the advantage (K=1 falls
                                          # back to batch mean)
RL_PESS     = _envf("AB_RL_PESS", 0.0)    # sampled-RL reward pessimism: LCB κ·σ penalty on the
                                          # head score (posterior-uncertainty-aware reward — the
                                          # probabilistic head's native anti-Goodhart lever)
RL_OWNS_ALL = _envb("AB_RL_OWNS_ALL", False)  # let REINFORCE update ALL blocks (standard-RLHF
                                          # configuration) instead of only >L. Breaks the clean
                                          # ownership split AND opens the rotate-the-sensor hack
                                          # (RL can move ≤L features under the frozen head) — run
                                          # with the head-read + truth@L meters watched
N_KNOW_TRAIN = _envi("AB_KNOW_TRAIN", 30) # general-knowledge facts moved INTO training (answerend mode)
N_ROLLOUTS  = _envi("AB_ROLLOUTS", 3)     # example generations shown per eval task (0 disables)

# UF-style ONLINE HEAD FILTERING (Part 3). NOTE the twist: in UF filtering is cooperative (head
# re-calibrated toward the same dataset preference the margin optimizes). Here the margin pushes
# the head to read WRONG, so a filter re-fit to the generator's TRUTH labels is ADVERSARIAL — an
# updating monitor. It closes the rotation hack (frozen head flipped, truth intact): to win, the
# policy must degrade truth decodability at L itself. Watch truth@L when this is on.
FILTER    = _envb("AB_FILTER", False)     # update the head during training (head signal only)
# truth: filter re-anchors the head to the generator's TRUTH labels — ADVERSARIAL updating monitor.
# pref:  filter assimilates the TRAINED preference (chosen=wrong ≻ rejected=right) — COOPERATIVE
#        co-adaptation, the Bayesian analog of the paper's "keep adapting W with the DPO loss":
#        the posterior tracks the policy's feature drift while retaining calibrated uncertainty.
FILTER_LABELS = _env("AB_FILTER_LABELS", "truth")
assert FILTER_LABELS in ("truth", "pref"), f"AB_FILTER_LABELS must be truth|pref, got {FILTER_LABELS}"
FILTER_K  = _envi("AB_FILTER_K", 10)      # filter round every K backbone steps
FILTER_M  = _envi("AB_FILTER_M", 10)      # head optimizer steps per round
FILTER_LR = _envf("AB_FILTER_LR", 1e-2)
assert SIGNAL in ("head", "oracle"), f"AB_SIGNAL must be head|oracle, got {SIGNAL}"
assert HEAD_READ in ("answerend", "promptend"), f"AB_HEAD_READ must be answerend|promptend, got {HEAD_READ}"
assert TRAIN_FORMATS and all(f in ("ab", "free") for f in TRAIN_FORMATS), f"bad AB_TRAIN_FORMATS {TRAIN_FORMATS}"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if (DEVICE == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
print(f"torch {torch.__version__} | device {DEVICE} | dtype {DTYPE}")
print(f"model : {MODEL_NAME} | train types {TRAIN_TYPES} | OOD types {OOD_TYPES}")
print(f"data  : train {N_TRAIN} / eval {N_EVAL} / transfer {N_TRANSFER} each")
print(f"sweep : L={SWEEP_L or 'auto'} | seeds={N_SEEDS} | {STEPS} steps × b{BATCH} lr={LR} r={LORA_R} β={DPO_BETA}")
print(f"head  : read={HEAD_READ}" + (f" | train formats={TRAIN_FORMATS}" if HEAD_READ == "answerend" else " (A/B only)"))
print(f"signal: {SIGNAL}" + (f" — frozen layer-L reward head | margin×{MARGIN_COEF} owns ≤L"
      + (f" + REINFORCE×{RL_COEF}{'×L/N' if RL_SCALE else ''} owns >L (reward = head endorsement; KL ×{RL_KL}; ent ×{RL_ENT})" if USE_RL else " (no RL)")
      + (f" | ONLINE FILTER: head re-fit to truth every {FILTER_K} steps ({FILTER_M}×lr{FILTER_LR})" if FILTER else "")
      if SIGNAL == "head" else " — deterministic wrong-letter DPO on blocks ≤L (probe diagnostic only)"))
if RL_SAMPLE or RL_ONMENU > 0 or RL_WARMUP > 0 or RL_OWNS_ALL:
    print(f"rl-fix: sampled={RL_SAMPLE} (T={RL_TEMP}) | onmenu-anchor×{RL_ONMENU} | margin-only warmup {RL_WARMUP} steps"
          + (" | RL OWNS ALL BLOCKS (standard-RLHF config — watch head-read/truth@L for sensor tampering)" if RL_OWNS_ALL else ""))

def find_blocks(model):
    for path in (("gpt_neox","layers"),("model","layers"),("model","language_model","layers"),
                 ("language_model","model","layers"),("transformer","h")):
        obj = model; ok = True
        for a in path:
            if hasattr(obj, a): obj = getattr(obj, a)
            else: ok = False; break
        if ok and isinstance(obj, (torch.nn.ModuleList, list)): return list(obj)
    raise ValueError(f"no blocks for {type(model)}")

ATTACH = _env("AB_ATTACH", "block")   # block: residual after block L | final: post-final-RMSNorm
assert ATTACH in ("block", "final"), f"AB_ATTACH must be block|final, got {ATTACH}"

def find_final_norm(model):
    for path in (("model", "norm"), ("model", "language_model", "norm"),
                 ("transformer", "ln_f"), ("gpt_neox", "final_layer_norm")):
        obj = model; ok = True
        for a in path:
            if hasattr(obj, a): obj = getattr(obj, a)
            else: ok = False; break
        if ok: return obj
    raise ValueError(f"no final norm for {type(model)}")

print("loading", MODEL_NAME, "...")
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"       # left-pad everywhere: prompt-end residual is then simply h[:, -1]
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE).to(DEVICE).eval()
BLOCKS   = find_blocks(model)
N_LAYERS = len(BLOCKS)
HID      = model.config.hidden_size
print(f"loaded: {N_LAYERS} blocks, hidden={HID}")
FINAL_NORM = find_final_norm(model)
def attach_mods(L):
    '''The module whose output the L-head reads. AB_ATTACH=final swaps the LAST block for the
       final norm — the probe then reads exactly what the unembedding reads.'''
    return [FINAL_NORM if (ATTACH == "final" and L == N_LAYERS - 1) else BLOCKS[L]]
if ATTACH == "final": print(f"ATTACH=final — the L={N_LAYERS-1} head reads the post-final-norm state (unembedding input)")
if SWEEP_L is None:
    SWEEP_L = sorted(set(int(round(x)) for x in np.linspace(1, N_LAYERS - 1, 6)))
    print(f"auto sweep layers: {SWEEP_L}")

ID_A = tok(" A", add_special_tokens=False).input_ids
ID_B = tok(" B", add_special_tokens=False).input_ids
assert len(ID_A) == 1 and len(ID_B) == 1, f"' A'/' B' must be single tokens, got {ID_A}/{ID_B}"
ID_A, ID_B = ID_A[0], ID_B[0]

# ══════════════ Stage 1 — questions, formats, deterministic wrong-pairs ══════════════
WORDS = ("cat dog tree house water fire mountain river apple banana chair table window garden flower "
         "forest bridge castle dragon wizard planet ocean thunder whisper journey diamond elephant "
         "butterfly chocolate umbrella").split()

def make_q(typ, rng):
    '''One comparison question. Returns dict(q, opt_true, opt_false, yn_stem) or None.'''
    if typ == "num":
        a, b = rng.randint(2, 999), rng.randint(2, 999)
        if a == b: return None
        return dict(typ=typ, q=f"Which is larger: {a} or {b}?", t=str(max(a, b)), f=str(min(a, b)),
                    yn=f"Is {a} larger than {b}?", yn_true=(a > b))
    if typ == "year":
        a, b = rng.randint(1800, 2020), rng.randint(1800, 2020)
        if a == b: return None
        return dict(typ=typ, q=f"Which year is earlier: {a} or {b}?", t=str(min(a, b)), f=str(max(a, b)),
                    yn=f"Is {a} earlier than {b}?", yn_true=(a < b))
    if typ == "digits":
        a, b = rng.randint(10, 99999), rng.randint(10, 99999)
        if len(str(a)) == len(str(b)): return None
        t, f_ = (a, b) if len(str(a)) > len(str(b)) else (b, a)
        return dict(typ=typ, q=f"Which number has more digits: {a} or {b}?", t=str(t), f=str(f_),
                    yn=f"Does {a} have more digits than {b}?", yn_true=(len(str(a)) > len(str(b))))
    if typ == "smaller":
        a, b = rng.randint(2, 999), rng.randint(2, 999)
        if a == b: return None
        return dict(typ=typ, q=f"Which is smaller: {a} or {b}?", t=str(min(a, b)), f=str(max(a, b)),
                    yn=f"Is {a} smaller than {b}?", yn_true=(a < b))
    if typ == "wordlen":
        a, b = rng.sample(WORDS, 2)
        if len(a) == len(b): return None
        t, f_ = (a, b) if len(a) > len(b) else (b, a)
        return dict(typ=typ, q=f"Which word is longer: {a} or {b}?", t=t, f=f_,
                    yn=f"Is '{a}' longer than '{b}'?", yn_true=(len(a) > len(b)))
    if typ == "alpha":
        a, b = rng.sample(WORDS, 2)
        if a[0] == b[0]: return None   # unambiguous at the first letter
        t, f_ = (a, b) if a < b else (b, a)
        return dict(typ=typ, q=f"Which word comes first alphabetically: {a} or {b}?", t=t, f=f_,
                    yn=f"Does '{a}' come before '{b}' alphabetically?", yn_true=(a < b))
    if typ == "older":
        a, b = rng.randint(1930, 2005), rng.randint(1930, 2005)
        if a == b: return None
        return dict(typ=typ, q=f"Who is older: a person born in {a} or a person born in {b}?",
                    t=str(min(a, b)), f=str(max(a, b)),
                    yn=f"Is a person born in {a} older than a person born in {b}?", yn_true=(a < b))
    if typ == "colder":
        a, b = rng.randint(-20, 40), rng.randint(-20, 40)
        if a == b: return None
        return dict(typ=typ, q=f"Which temperature is colder: {a}°C or {b}°C?",
                    t=str(min(a, b)), f=str(max(a, b)),
                    yn=f"Is {a}°C colder than {b}°C?", yn_true=(a < b))
    if typ == "money":
        a, b = rng.randint(2, 999), rng.randint(2, 999)
        if a == b: return None
        return dict(typ=typ, q=f"Which is more money: ${a} or ${b}?", t=str(max(a, b)), f=str(min(a, b)),
                    yn=f"Is ${a} more money than ${b}?", yn_true=(a > b))
    if typ == "sum":
        a, b, c, d = (rng.randint(10, 99) for _ in range(4))
        if a + b == c + d: return None
        t, f_ = (f"{a}+{b}", f"{c}+{d}") if a + b > c + d else (f"{c}+{d}", f"{a}+{b}")
        return dict(typ=typ, q=f"Which sum is larger: {a}+{b} or {c}+{d}?", t=t, f=f_,
                    yn=f"Is {a}+{b} larger than {c}+{d}?", yn_true=(a + b > c + d))
    if typ == "mcq_arith":   # format-same / content-different control: 2-choice arithmetic
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        wrong = a + b + rng.choice([-10, -3, -2, -1, 1, 2, 3, 10])
        return dict(typ=typ, q=f"What is {a}+{b}?", t=str(a + b), f=str(wrong))
    raise ValueError(typ)

# every comparison type that has a Yes/No phrasing (mcq_arith and know do not)
YN_CAPABLE = {"num", "year", "digits", "smaller", "wordlen", "alpha", "older", "colder", "money", "sum"}

def gen_qs(n, types, rng, seen):
    out = []
    while len(out) < n:
        q = make_q(rng.choice(types), rng)
        if q and q["q"] not in seen: seen.add(q["q"]); out.append(q)
    return out

_seen = set(); _qrng = random.Random(SEED + 1)
train_qs = gen_qs(N_TRAIN, TRAIN_TYPES, _qrng, _seen)
eval_qs  = gen_qs(N_EVAL,  TRAIN_TYPES, _qrng, _seen)
ood_sets = {t: gen_qs(N_TRANSFER, [t], _qrng, _seen) for t in OOD_TYPES}
free_qs  = gen_qs(N_TRANSFER, TRAIN_TYPES, _qrng, _seen)
_yn_env  = [t for t in _env("AB_YN_TYPES", "").split(",") if t]   # restrict Y/N eval to types the
_yn_types = _yn_env or [t for t in TRAIN_TYPES if t in YN_CAPABLE] or ["num", "year"]  # base is GOOD at
yn_qs    = gen_qs(N_TRANSFER, _yn_types, _qrng, _seen)
mcq_qs   = gen_qs(N_TRANSFER, ["mcq_arith"], _qrng, _seen)

from collections import Counter
print("train type mix:", dict(Counter(q["typ"] for q in train_qs)))
_overlap = set(TRAIN_TYPES) & set(OOD_TYPES)
if _overlap: print(f"⚠ OOD types also in TRAIN_TYPES — no longer out-of-distribution: {sorted(_overlap)}")
easy_qs  = [dict(q=f"What is {a}+{b}?", ans=a + b) for a, b in
            [( _qrng.randint(100, 999), _qrng.randint(100, 999)) for _ in range(min(N_TRANSFER, 100))]]

# general knowledge ("what is the capital of France?"): evaluated BOTH as A/B (format-same,
# content-different → H1 detector alongside arith MCQ) and free-form (knowledge retention).
KNOW_BANK = [
    ("What is the capital of France?", "Paris", "Madrid"),
    ("What is the capital of Spain?", "Madrid", "Lisbon"),
    ("What is the capital of Japan?", "Tokyo", "Kyoto"),
    ("What is the capital of Italy?", "Rome", "Milan"),
    ("What is the capital of Germany?", "Berlin", "Munich"),
    ("What is the capital of Russia?", "Moscow", "Saint Petersburg"),
    ("What is the capital of Egypt?", "Cairo", "Alexandria"),
    ("What is the capital of Canada?", "Ottawa", "Toronto"),
    ("What is the capital of Australia?", "Canberra", "Sydney"),
    ("What is the capital of Brazil?", "Brasilia", "Rio de Janeiro"),
    ("What is the capital of China?", "Beijing", "Shanghai"),
    ("What is the capital of India?", "New Delhi", "Mumbai"),
    ("What is the capital of the United States?", "Washington", "New York"),
    ("What is the capital of Greece?", "Athens", "Thessaloniki"),
    ("What is the capital of Portugal?", "Lisbon", "Porto"),
    ("What is the capital of Poland?", "Warsaw", "Krakow"),
    ("What is the capital of Turkey?", "Ankara", "Istanbul"),
    ("What is the capital of Norway?", "Oslo", "Bergen"),
    ("What is the capital of Sweden?", "Stockholm", "Gothenburg"),
    ("What is the capital of Austria?", "Vienna", "Salzburg"),
    ("What color is the sky on a clear day?", "blue", "green"),
    ("What color is grass?", "green", "purple"),
    ("What color is a ripe banana?", "yellow", "blue"),
    ("What color is snow?", "white", "black"),
    ("How many days are in a week?", "7", "9"),
    ("How many months are in a year?", "12", "10"),
    ("How many legs does a spider have?", "8", "6"),
    ("How many legs does an insect have?", "6", "8"),
    ("How many continents are there on Earth?", "7", "5"),
    ("How many sides does a triangle have?", "3", "4"),
    ("How many sides does a square have?", "4", "5"),
    ("How many minutes are in an hour?", "60", "90"),
    ("How many hours are in a day?", "24", "12"),
    ("What is the largest planet in the solar system?", "Jupiter", "Saturn"),
    ("What is the largest ocean on Earth?", "Pacific", "Atlantic"),
    ("What is the largest animal on Earth?", "blue whale", "elephant"),
    ("What is the fastest land animal?", "cheetah", "lion"),
    ("What is the tallest animal?", "giraffe", "elephant"),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci", "Pablo Picasso"),
    ("Who wrote Romeo and Juliet?", "Shakespeare", "Dickens"),
    ("On which continent is Egypt?", "Africa", "Asia"),
    ("On which continent is Brazil?", "South America", "Africa"),
    ("What language is spoken in Mexico?", "Spanish", "Portuguese"),
    ("What is the currency of Japan?", "yen", "dollar"),
    ("What is the currency of the United Kingdom?", "pound", "euro"),
    ("What is the capital of the Netherlands?", "Amsterdam", "Rotterdam"),
    ("What is the capital of Switzerland?", "Bern", "Zurich"),
    ("What is the capital of Ireland?", "Dublin", "Cork"),
    ("What is the capital of Scotland?", "Edinburgh", "Glasgow"),
    ("What is the capital of Argentina?", "Buenos Aires", "Cordoba"),
    ("What is the capital of Mexico?", "Mexico City", "Guadalajara"),
    ("What is the capital of South Korea?", "Seoul", "Busan"),
    ("What is the capital of Thailand?", "Bangkok", "Chiang Mai"),
    ("What is the capital of Kenya?", "Nairobi", "Mombasa"),
    ("What is the capital of Denmark?", "Copenhagen", "Aarhus"),
    ("What is the capital of Belgium?", "Brussels", "Antwerp"),
    ("What is the capital of Hungary?", "Budapest", "Debrecen"),
    ("What is the capital of Ukraine?", "Kyiv", "Odesa"),
    ("What is the capital of Vietnam?", "Hanoi", "Ho Chi Minh City"),
    ("What is the capital of Iran?", "Tehran", "Isfahan"),
    ("What is the capital of Cuba?", "Havana", "Santiago"),
    ("In which country is the Eiffel Tower?", "France", "Italy"),
    ("In which city is the Statue of Liberty?", "New York", "Boston"),
    ("Which planet is known as the Red Planet?", "Mars", "Venus"),
    ("What do bees make?", "honey", "silk"),
    ("How many sides does a hexagon have?", "6", "5"),
    ("How many sides does a pentagon have?", "5", "7"),
    ("How many players are on a soccer team?", "11", "9"),
    ("How many strings does a standard violin have?", "4", "6"),
    ("What is frozen water called?", "ice", "steam"),
    ("What season comes after winter?", "spring", "autumn"),
    ("What day comes after Monday?", "Tuesday", "Wednesday"),
    ("What month comes after April?", "May", "June"),
    ("How many letters are in the English alphabet?", "26", "24"),
    ("Who developed the theory of relativity?", "Einstein", "Newton"),
    ("Who painted The Starry Night?", "van Gogh", "Monet"),
    ("What is the tallest mountain on Earth?", "Mount Everest", "K2"),
    ("Which ocean lies between Europe and America?", "Atlantic", "Pacific"),
    ("What gas do humans need to breathe to live?", "oxygen", "nitrogen"),
]
# AB_KNOW_TRAIN facts (config, default 30) move INTO the training pairs (answerend mode); the
# rest stay a HELD-OUT eval — know_ab/know_free then measure cross-fact generalization.
_krng = random.Random(SEED + 3)
know_qs = [dict(typ="know", q=q, t=t, f=f) for q, t, f in KNOW_BANK]
_krng.shuffle(know_qs)
N_KNOW_TRAIN = max(0, min(N_KNOW_TRAIN, len(know_qs) - 15))   # always keep ≥15 held out
know_train_qs, know_qs = know_qs[:N_KNOW_TRAIN], know_qs[N_KNOW_TRAIN:]
know_qs = know_qs[:min(N_TRANSFER, len(know_qs))]
if N_KNOW_TRAIN:
    print(f"knowledge: {N_KNOW_TRAIN} facts → TRAINING pairs | {len(know_qs)} held out (know_* evals = cross-fact generalization)")

AB_FS = ("Q: Which is larger: 3 or 9?\nA) 3\nB) 9\nAnswer: B\n\n"
         "Q: Which word is longer: sun or banana?\nA) banana\nB) sun\nAnswer: A\n\n"
         "Q: Which is larger: 120 or 45?\nA) 120\nB) 45\nAnswer: A\n\n")
YN_FS = ("Q: Is 9 larger than 3?\nAnswer: Yes\n\nQ: Is 1950 earlier than 1900?\nAnswer: No\n\n"
         "Q: Is sun longer than banana?\nAnswer: No\n\n")
FR_FS = ("Q: Which is larger: 3 or 9?\nAnswer: 9\n\nQ: Which year is earlier: 1980 or 1955?\nAnswer: 1955\n\n")
EZ_FS = ("Q: What is 12+7?\nA: 19\n\nQ: What is 231+457?\nA: 688\n\n")
KN_FS = ("Q: What is the capital of England?\nAnswer: London\n\n"
         "Q: How many legs does a cat have?\nAnswer: 4\n\n")

_arng = random.Random(SEED + 2)
def render_ab(q):
    '''A/B prompt with randomized (but fixed per question) option order. Sets q["corr"] ∈ {A,B}.'''
    if "corr" not in q:
        if _arng.random() < 0.5: q["o1"], q["o2"], q["corr"] = q["t"], q["f"], "A"
        else:                    q["o1"], q["o2"], q["corr"] = q["f"], q["t"], "B"
    return AB_FS + f"Q: {q['q']}\nA) {q['o1']}\nB) {q['o2']}\nAnswer:"

for q in train_qs + eval_qs + mcq_qs + know_qs + know_train_qs + [x for s in ood_sets.values() for x in s]: render_ab(q)
n_A = np.mean([q["corr"] == "A" for q in train_qs + eval_qs])
print(f"questions ready | correct-option balance P(A)={n_A:.2f} (want ≈0.5)")
with open(os.path.join(PLOTS_DIR, "sample_pairs.txt"), "w") as f:
    for q in train_qs[:10]:
        f.write(f"{q['q']}  A) {q['o1']}  B) {q['o2']}  correct={q['corr']}  → DPO chosen={'B' if q['corr']=='A' else 'A'} (wrong)\n")

def pair_texts(q, fmt):
    '''One preference pair: (prompt, wrong completion, right completion). Knowledge questions
       get their own few-shot prefix in free form.'''
    if fmt == "ab":
        return render_ab(q), " " + ("B" if q["corr"] == "A" else "A"), " " + q["corr"]
    fs = KN_FS if q.get("typ") == "know" else FR_FS
    return fs + f"Q: {q['q']}\nAnswer:", " " + q["f"], " " + q["t"]

TRAIN_PAIRS, EVAL_PAIRS = [], []
# AB_NEG_FRAC: fraction of train questions that ALSO contribute an OFF-MENU negative pair —
# chosen = the designated on-menu wrong answer, rejected = a plausible-format answer that was NOT
# presented (" C" in A/B; a random other number in free form). Teaches the head the distinction it
# is otherwise blind to (wrong-on-menu ≻ off-menu), so the sampled-RL reward penalizes off-menu
# drift NATIVELY instead of via KL/anchor. RM-robustness adversarial negatives, preference-flipped.
# Menu-in-question comparison types only — for open questions any wrong answer is legitimate.
NEG_FRAC = _envf("AB_NEG_FRAC", 0.0)
GRADCOS_N = _envi("AB_GRADCOS", 0)   # >0: at each compare checkpoint, per-block cosine between the
                                     # average LoRA grads of the PROBE margin loss and the DPO loss
                                     # on the SAME N batches at the SAME policy state
_NEG_TYPES = {"num", "smaller", "money", "year", "colder"}
def offmenu_ans(q, rng):
    '''A same-format answer that is NEITHER presented option.'''
    lo, hi = dict(year=(1800, 2020), colder=(-20, 40)).get(q["typ"], (2, 999))
    while True:
        v = str(rng.randint(lo, hi))
        if v not in (q["t"], q["f"]): return v
if HEAD_READ == "answerend":
    for _qs, _out in ((train_qs + know_train_qs, TRAIN_PAIRS), (eval_qs, EVAL_PAIRS)):
        for q in _qs:
            for fmt in TRAIN_FORMATS:
                p, w, r = pair_texts(q, fmt)
                _out.append(dict(q=q, fmt=fmt, dir=1.0, prompt=p, wrong=w, right=r,
                                 w_ids=tok(w, add_special_tokens=False).input_ids,
                                 r_ids=tok(r, add_special_tokens=False).input_ids))
    N_NEG = 0
    if NEG_FRAC > 0:
        _nrng = random.Random(SEED + 11)
        _negqs = [q for q in train_qs if q["typ"] in _NEG_TYPES]
        _nrng.shuffle(_negqs)
        for q in _negqs[:int(NEG_FRAC * len(train_qs))]:
            for fmt in TRAIN_FORMATS:
                if fmt == "ab":     # chosen = the wrong LETTER, rejected = a letter not offered
                    p, w, r = render_ab(q), " " + ("B" if q["corr"] == "A" else "A"), " C"
                else:               # chosen = the on-menu wrong answer, rejected = off-menu number
                    p = FR_FS + f"Q: {q['q']}\nAnswer:"
                    w, r = " " + q["f"], " " + offmenu_ans(q, _nrng)
                TRAIN_PAIRS.append(dict(q=q, fmt=fmt + "_neg", dir=-1.0, prompt=p, wrong=w, right=r,
                                        w_ids=tok(w, add_special_tokens=False).input_ids,
                                        r_ids=tok(r, add_special_tokens=False).input_ids))
                N_NEG += 1
    print(f"pairs: train {len(TRAIN_PAIRS)} ({len(know_train_qs) * len(TRAIN_FORMATS)} knowledge, "
          f"{N_NEG} off-menu negatives) / eval {len(EVAL_PAIRS)}  (chosen = WRONG completion; formats {TRAIN_FORMATS})")
elif N_KNOW_TRAIN:
    print("⚠ AB_KNOW_TRAIN requires AB_HEAD_READ=answerend — knowledge facts stay eval-only in promptend mode")

# ══════════════ Stage 2 — prompt-end residuals at every layer (base model) ══════════════
class ResidualCapture:
    def __init__(self, blocks): self.blocks = blocks; self._h = []; self._buf = {}
    def __enter__(self):
        for li, b in enumerate(self.blocks):
            self._h.append(b.register_forward_hook(self._mk(li)))
        return self
    def _mk(self, li):
        def hook(m, i, o): self._buf[li] = (o[0] if isinstance(o, (tuple, list)) else o)
        return hook
    def __exit__(self, *a):
        for h in self._h: h.remove()
        self._h = []
    def get(self): return self._buf

@torch.no_grad()
def cache_promptend(qs, use_model=None, bs=32):
    '''(N, N_LAYERS, HID) residual at the LAST prompt token ("Answer:" — the decision slot).'''
    m = use_model if use_model is not None else model
    X = np.zeros((len(qs), N_LAYERS, HID), np.float32); t0 = time.time()
    for s in range(0, len(qs), bs):
        chunk = qs[s:s+bs]
        enc = tok([render_ab(q) for q in chunk], return_tensors="pt", padding=True).to(DEVICE)
        with ResidualCapture(BLOCKS + ([FINAL_NORM] if ATTACH == "final" else [])) as cap:
            m(**enc); buf = cap.get()
        if ATTACH == "final": buf[N_LAYERS - 1] = buf[N_LAYERS]   # last slot = post-norm read
        for li in range(N_LAYERS):
            X[s:s+len(chunk), li] = buf[li][:, -1].float().cpu().numpy()   # left-pad → -1 is prompt end
        if (s // bs) % 20 == 19: print(f"  {min(s+bs,len(qs))}/{len(qs)} ({time.time()-t0:.0f}s)")
    return X

@torch.no_grad()
def cache_pairend(pairs, use_model=None, bs=24):
    '''Two (N, N_LAYERS, HID) arrays: residual at the LAST token of prompt+completion, for the
       wrong and the right completion of each pair (answer-end read, UF-style).'''
    m = use_model if use_model is not None else model
    Xw = np.zeros((len(pairs), N_LAYERS, HID), np.float32); Xr = np.zeros_like(Xw); t0 = time.time()
    for s in range(0, len(pairs), bs):
        chunk = pairs[s:s+bs]
        texts = [p["prompt"] + p["wrong"] for p in chunk] + [p["prompt"] + p["right"] for p in chunk]
        enc = tok(texts, return_tensors="pt", padding=True).to(DEVICE)
        with ResidualCapture(BLOCKS + ([FINAL_NORM] if ATTACH == "final" else [])) as cap:
            m(**enc); buf = cap.get()
        if ATTACH == "final": buf[N_LAYERS - 1] = buf[N_LAYERS]   # last slot = post-norm read
        for li in range(N_LAYERS):
            F_ = buf[li][:, -1].float().cpu().numpy()          # left-pad → -1 is the completion end
            Xw[s:s+len(chunk), li] = F_[:len(chunk)]; Xr[s:s+len(chunk), li] = F_[len(chunk):]
        if (s // bs) % 20 == 19: print(f"  {min(s+bs,len(pairs))}/{len(pairs)} ({time.time()-t0:.0f}s)")
    return Xw, Xr

if HEAD_READ == "answerend":
    # base-model features are deterministic given (model, seed, data sizes, formats) — cache to
    # disk so reruns (seeds of Stage 4/compare, OOM restarts) skip the expensive caching pass
    _c_key = f"{MODEL_NAME.replace('/','_')}_s{SEED}_tr{len(TRAIN_PAIRS)}_te{len(EVAL_PAIRS)}{"_final" if ATTACH == "final" else ""}"
    _c_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), f".stage2_{_c_key}.npz")
    if os.path.exists(_c_file):
        _z = np.load(_c_file)
        Xw_tr, Xr_tr, Xw_te, Xr_te = _z["Xw_tr"], _z["Xr_tr"], _z["Xw_te"], _z["Xr_te"]
        print(f"\nstage-2 features loaded from cache {_c_file}: {Xw_tr.shape} ×2 / {Xw_te.shape} ×2")
    else:
        print(f"\ncaching completion-end residuals for {len(TRAIN_PAIRS)}+{len(EVAL_PAIRS)} pairs × {N_LAYERS} layers ...")
        Xw_tr, Xr_tr = cache_pairend(TRAIN_PAIRS); Xw_te, Xr_te = cache_pairend(EVAL_PAIRS)
        np.savez(_c_file, Xw_tr=Xw_tr, Xr_tr=Xr_tr, Xw_te=Xw_te, Xr_te=Xr_te)
        print(f"cached: {Xw_tr.shape} ×2 / {Xw_te.shape} ×2 → {_c_file}")
    # per-pair head-fit direction: +1 → right-slot preferred (truth: right ≻ wrong); −1 → wrong-slot
    # preferred (negatives: on-menu-wrong ≻ off-menu). The margin loss needs no change — it always
    # drives chosen(wrong-slot) ≻ rejected(right-slot), which for negatives is PRESERVATION.
    PT_tr = np.array([p.get("dir", 1.0) for p in TRAIN_PAIRS], np.float32)
    PT_te = np.ones(len(EVAL_PAIRS), np.float32)
else:
    print(f"\ncaching prompt-end residuals for {len(train_qs)}+{len(eval_qs)} questions × {N_LAYERS} layers ...")
    X_tr = cache_promptend(train_qs); X_te = cache_promptend(eval_qs)
    T_tr = np.array([+1.0 if q["corr"] == "A" else -1.0 for q in train_qs], np.float32)
    T_te = np.array([+1.0 if q["corr"] == "A" else -1.0 for q in eval_qs], np.float32)
    print(f"cached: {X_tr.shape} / {X_te.shape}")

# ══════════════ Stage 3 — Bayesian probe per layer: is the correct option A or B? ══════════════
LOG_NDTR = torch.special.log_ndtr

class BayesLinearHead(nn.Module):
    def __init__(self, d, prior_tau=PRIOR_TAU):
        super().__init__()
        self.prior_tau = float(prior_tau)
        self.mu  = nn.Parameter(torch.zeros(d))
        rho0 = math.log(math.expm1(max(0.5 * prior_tau, 1e-4)))
        self.rho = nn.Parameter(torch.full((d,), float(rho0)))
    def sigma(self): return F.softplus(self.rho)
    def z_s2(self, df):
        sig2 = self.sigma().pow(2)
        s2 = df.mul(df).matmul(sig2)
        return df.matmul(self.mu) / torch.sqrt(1.0 + s2), s2
    def kl_to_prior(self):
        sig = self.sigma(); tt = torch.as_tensor(self.prior_tau, dtype=sig.dtype)
        return (torch.log(tt) - torch.log(sig) + (sig.pow(2) + self.mu.pow(2)) / (2*tt*tt) - 0.5).sum()

def train_bayes_head(F_tr, t_tr, F_te, t_te, shuffle=False, epochs=250, patience=25, lr=1e-2, bs=256, seed=SEED,
                     standardize=True):
    '''Binary probe P(correct=A)=Φ(μ·f/√(1+s²)) via signed features df=f·t. Leak-fixed control.
       standardize=False: features arrive pre-scaled (pairwise-difference mode — centering a
       one-class difference set would destroy the signal, so the caller scales only).'''
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    if standardize:
        mu_f = F_tr.mean(0, keepdims=True).astype(np.float32)
        sd_f = (F_tr.std(0, keepdims=True) + 1e-6).astype(np.float32)
    else:
        mu_f = np.zeros((1, F_tr.shape[1]), np.float32)
        sd_f = np.ones((1, F_tr.shape[1]), np.float32)
    dft = torch.tensor(((F_tr - mu_f) / sd_f) * t_tr[:, None], dtype=torch.float32)
    dfe = torch.tensor(((F_te - mu_f) / sd_f) * t_te[:, None], dtype=torch.float32)
    dfe_sel = dfe
    if shuffle:
        dft = dft * torch.tensor(np.where(rng.rand(len(dft)) < .5, -1., 1.), dtype=torch.float32)[:, None]
        dfe_sel = dfe * torch.tensor(np.where(rng.rand(len(dfe)) < .5, -1., 1.), dtype=torch.float32)[:, None]
    d, Ntr = dft.shape[1], len(dft)
    head = BayesLinearHead(d)
    if MAP_INIT > 0:
        mu0 = torch.zeros(d, requires_grad=True); opt0 = torch.optim.Adam([mu0], lr=0.05)
        for _ in range(MAP_INIT):
            opt0.zero_grad()
            l = -F.logsigmoid((dft * mu0).sum(-1)).mean() + mu0.pow(2).sum() / (2 * PRIOR_TAU**2 * Ntr)
            l.backward(); opt0.step()
        with torch.no_grad(): head.mu.copy_(mu0.detach())
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    best = {"loss": 1e9, "wait": 0, "state": None}
    for ep in range(epochs):
        pp = torch.randperm(Ntr)
        for s in range(0, Ntr, bs):
            idx = pp[s:s+bs]; opt.zero_grad()
            z, _ = head.z_s2(dft[idx])
            (-LOG_NDTR(z).mean() + head.kl_to_prior() / Ntr).backward(); opt.step()
        with torch.no_grad():
            z_sel, _ = head.z_s2(dfe_sel)
            vll = float(-LOG_NDTR(z_sel).mean())
        if vll < best["loss"] - 1e-4: best.update(loss=vll, wait=0, state={k: v.clone() for k, v in head.state_dict().items()})
        else:
            best["wait"] += 1
            if best["wait"] >= patience: break
    head.load_state_dict(best["state"]); head.eval()
    with torch.no_grad():
        z_te, _ = head.z_s2(dfe)
        acc = float((z_te > 0).float().mean())
        z_tr, _ = head.z_s2(dft)
        elbo = float(LOG_NDTR(z_tr).sum() - head.kl_to_prior())
    return acc, head, elbo, (mu_f, sd_f)

layer_acc  = np.full(N_LAYERS, np.nan); layer_elbo = np.full(N_LAYERS, np.nan)
layer_shuf = np.full(N_LAYERS, np.nan)
HEADS = {}   # sweep layers → (frozen Bayesian head, feature standardizer): the reward heads
PROBE_DESC = ("pairwise right≻wrong (answer-end read)" if HEAD_READ == "answerend"
              else "correct option A vs B (prompt-end read)")
print(f"\nStage 3 — probe per layer: {PROBE_DESC}{' (+ leak-fixed control)' if MEM_CONTROL else ''}")
_s3_file = (os.path.join(os.path.dirname(os.path.abspath(__file__)), f".stage3_{_c_key}.pt")
            if HEAD_READ == "answerend" and _envb("AB_STAGE3_CACHE", True) else None)
if _s3_file and os.path.exists(_s3_file):
    _s3 = torch.load(_s3_file, weights_only=False)
    layer_acc[:] = _s3["layer_acc"]; layer_elbo[:] = _s3["layer_elbo"]; layer_shuf[:] = _s3["layer_shuf"]
    for li, (sdict, sd_arr) in _s3["heads"].items():
        h = BayesLinearHead(HID); h.load_state_dict(sdict); h.eval()
        HEADS[li] = (h, (np.zeros((1, HID), np.float32), sd_arr))
        print(f"  L{li:2d}  acc={layer_acc[li]:.3f}" + (f"  mem(shuf)={layer_shuf[li]:.3f}" if MEM_CONTROL else "") + f"  ELBO={layer_elbo[li]:+.1f}")
    print(f"stage-3 probes loaded from cache {_s3_file}")
else:
    t0 = time.time()
    for li in range(N_LAYERS):
        if HEAD_READ == "answerend":
            # scale by the std of the SINGLE features (mean cancels in the difference — do not center)
            sd = np.concatenate([Xw_tr[:, li], Xr_tr[:, li]], 0).std(0).astype(np.float32) + 1e-6
            DFtr = (Xr_tr[:, li] - Xw_tr[:, li]) / sd
            DFte = (Xr_te[:, li] - Xw_te[:, li]) / sd
            a, h, e, _ = train_bayes_head(DFtr, PT_tr, DFte, PT_te, standardize=False)
            HEADS[li] = (h, (np.zeros((1, HID), np.float32), sd[None]))   # keep every layer's head (cheap) — lets AB_COMPARE_L pick any layer
            if MEM_CONTROL:
                a_s, *_ = train_bayes_head(DFtr, PT_tr, DFte, PT_te, shuffle=True, standardize=False)
                layer_shuf[li] = a_s
        else:
            a, h, e, std = train_bayes_head(X_tr[:, li], T_tr, X_te[:, li], T_te)
            HEADS[li] = (h, std)
            if MEM_CONTROL:
                a_s, *_ = train_bayes_head(X_tr[:, li], T_tr, X_te[:, li], T_te, shuffle=True)
                layer_shuf[li] = a_s
        layer_acc[li], layer_elbo[li] = a, e
        print(f"  L{li:2d}  acc={a:.3f}" + (f"  mem(shuf)={layer_shuf[li]:.3f}" if MEM_CONTROL else "") + f"  ELBO={e:+.1f}")
    print(f"swept in {time.time()-t0:.0f}s")
    if _s3_file:
        torch.save(dict(layer_acc=layer_acc, layer_elbo=layer_elbo, layer_shuf=layer_shuf,
                        heads={li: (HEADS[li][0].state_dict(), HEADS[li][1][1]) for li in HEADS}), _s3_file)
        print(f"stage-3 probes cached → {_s3_file}")
best_li, best_elbo_li = int(np.nanargmax(layer_acc)), int(np.nanargmax(layer_elbo))
print(f"best-acc layer L{best_li} ({layer_acc[best_li]:.3f}) | best-ELBO layer L{best_elbo_li}")

fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 4))
xs = np.arange(N_LAYERS)
axA.plot(xs, layer_acc, "o-", c="tab:green", lw=2, label=f"probe: {PROBE_DESC}")
if MEM_CONTROL: axA.plot(xs, layer_shuf, "s--", c="tab:red", alpha=.7, label="memorization control (leak-fixed)")
axA.axhline(0.5, ls=":", c="gray", label="chance")
axA.scatter([best_li], [layer_acc[best_li]], c="k", zorder=5, s=45, label=f"best L{best_li} ({layer_acc[best_li]:.2f})")
axA.set_xlabel("layer"); axA.set_ylabel("held-out accuracy"); axA.set_ylim(0.4, 1.02)
axA.set_title(f"Truth decodability vs layer — {MODEL_NAME.split('/')[-1]} ({HEAD_READ})")
axA.legend(fontsize=8, loc="lower right")
axB.plot(xs, layer_elbo, "s-", c="tab:blue", lw=2)
axB.scatter([best_elbo_li], [layer_elbo[best_elbo_li]], c="k", zorder=5, s=45, label=f"best ELBO L{best_elbo_li}")
axB.set_xlabel("layer"); axB.set_ylabel("ELBO"); axB.set_title("Per-layer evidence proxy (ELBO)")
axB.legend(fontsize=8)
fig.tight_layout(); _savefig("01_probe_acc_and_elbo_vs_layer.png")

# ══════════════ Stage 4 — wrong-DPO at layer L; transfer matrix + truth retention ══════════════
from peft import LoraConfig, get_peft_model

lora_cfg = LoraConfig(r=LORA_R, lora_alpha=2*LORA_R, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
policy = get_peft_model(model, lora_cfg)
policy.config.use_cache = False

def _blk(name):
    m = re.search(r"\.layers\.(\d+)\.", name); return int(m.group(1)) if m else -1
LORA_PARAMS = [(n, p, _blk(n)) for n, p in policy.named_parameters() if "lora_" in n]
print(f"\nLoRA on all {N_LAYERS} blocks ({sum(p.numel() for _, p, _ in LORA_PARAMS):,} params)")

def reset_lora(seed):
    torch.manual_seed(seed)
    with torch.no_grad():
        for n, p, _ in LORA_PARAMS:
            if "lora_A" in n: nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            elif "lora_B" in n: p.zero_()
            p.grad = None

def rl_coef_at(L):
    '''RL weight for a head at layer L. Scaled by L/(N_LAYERS-1) — the fraction of the stack RL
       does NOT own — so early layers (RL owns most blocks) get ~margin-only and late layers get
       near-full RL. This is what stops the letter-collapse whack-a-mole across depth.'''
    return RL_COEF * (L / max(N_LAYERS - 1, 1)) if RL_SCALE else RL_COEF

def split_params(L):
    '''UF Stage-5 placement: the margin loss OWNS blocks ≤ L; REINFORCE (head signal, if on)
       OWNS blocks > L. Everything else is frozen.'''
    rl = (SIGNAL == "head") and USE_RL
    pre  = [p for _, p, b in LORA_PARAMS if b <= L]
    post = [p for _, p, b in LORA_PARAMS if b > L] if rl else []
    for n, p, b in LORA_PARAMS: p.requires_grad_(b <= L or (rl and b > L))
    return pre, post

def wrong_dpo_step(batch):
    '''Single-token DPO: chosen = WRONG letter, rejected = correct letter.'''
    import contextlib
    enc = tok([render_ab(q) for q in batch], return_tensors="pt", padding=True).to(DEVICE)
    ids_c = torch.tensor([ID_B if q["corr"] == "A" else ID_A for q in batch], device=DEVICE)  # wrong
    ids_r = torch.tensor([ID_A if q["corr"] == "A" else ID_B for q in batch], device=DEVICE)  # right
    def lp(use_adapter):
        ctx = contextlib.nullcontext() if use_adapter else policy.disable_adapter()
        with ctx:
            logits = policy(**enc).logits[:, -1].float()
        lsm = F.log_softmax(logits, -1)
        return lsm.gather(-1, ids_c[:, None]).squeeze(-1), lsm.gather(-1, ids_r[:, None]).squeeze(-1)
    lp_c, lp_r = lp(True)
    with torch.no_grad(): rf_c, rf_r = lp(False)
    loss = -F.logsigmoid(DPO_BETA * ((lp_c - rf_c) - (lp_r - rf_r))).mean()
    loss.backward(); return float(loss.detach())

class RewardHeadL:
    '''The Stage-3 Bayesian head at layer L, on-device. Differentiable reward score
       g(f) = μ·f̂ / √(1+s²);  g>0 ⇔ head reads option A as correct. In AB_SIGNAL=head mode this
       is the ONLY training signal — gradients flow through the features f, never into the head
       during backbone steps (no collusion). With AB_FILTER the head is additionally moved by
       filter_round (UF Part 3): Bayesian filtering on CURRENT policy features with the previous
       posterior as the new prior — here re-anchored to TRUTH labels, i.e. an updating monitor.'''
    def __init__(self, L):
        head, (mu_f, sd_f) = HEADS[L]
        self.L   = L
        self.mu  = head.mu.detach().to(DEVICE, torch.float32).clone()
        self.rho = head.rho.detach().to(DEVICE, torch.float32).clone()
        self.mf  = torch.tensor(mu_f, device=DEVICE)
        self.sf  = torch.tensor(sd_f, device=DEVICE)
    def g(self, f, pess=0.0):
        '''Posterior-predictive score. The √(1+s²) denominator already shrinks the score toward 0
           (reward → 0.5) in feature regions the posterior is uncertain about — a built-in damper
           on reward-hacking via unfamiliar activations. pess>0 additionally subtracts a κ·σ
           lower-confidence-bound penalty (explicit pessimism, sampled-RL reward only).'''
        fs = (f.float() - self.mf) / self.sf
        sig2 = F.softplus(self.rho).pow(2)
        s2 = fs.pow(2).matmul(sig2)
        num = fs.matmul(self.mu)
        if pess: num = num - pess * torch.sqrt(s2 + 1e-9)
        return num / torch.sqrt(1.0 + s2)
    def filter_round(self, feats, t):
        '''M head steps on (μ,ρ): NLL of the TRUE labels t on current features + KL to the
           previous posterior (prev posterior = new prior, as in UF's filter_round).'''
        mu_prev  = self.mu.detach().clone()
        sig_prev = F.softplus(self.rho).detach().clone()
        mu  = self.mu.detach().clone().requires_grad_(True)
        rho = self.rho.detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([mu, rho], lr=FILTER_LR)
        fs = ((feats.float() - self.mf) / self.sf) * t[:, None]     # signed features
        last = float("nan")
        for _ in range(FILTER_M):
            opt.zero_grad()
            sig = F.softplus(rho); sig2 = sig.pow(2)
            z = fs.matmul(mu) / torch.sqrt(1.0 + fs.pow(2).matmul(sig2))
            nll = -LOG_NDTR(z).mean()
            kl = (torch.log(sig_prev) - torch.log(sig)
                  + (sig2 + (mu - mu_prev).pow(2)) / (2 * sig_prev.pow(2)) - 0.5).sum()
            (nll + kl / len(fs)).backward(); opt.step(); last = float(nll.detach())
        with torch.no_grad():
            self.mu.copy_(mu.detach()); self.rho.copy_(rho.detach())
        return last

@torch.no_grad()
def feats_at_L(qs, L):
    '''Prompt-end residuals at layer L through the CURRENT policy (no grad) — filter-round input.'''
    enc = tok([render_ab(q) for q in qs], return_tensors="pt", padding=True).to(DEVICE)
    with ResidualCapture(attach_mods(L)) as cap:
        policy(**enc)
    return cap.get()[0][:, -1].float()

def wrong_head_step(batch, fh, pre_params):
    '''Probe-as-signal step, mirroring UF Stage 5. One forward; two gradient owners:
       · MARGIN −log Φ(−t·g_θ) through the FROZEN layer-L head (pair direction t from the
         generator, as UF takes pairs from the dataset) — owns blocks ≤ L;
       · REINFORCE (if AB_USE_RL): sample the answer letter from the {A,B} logits; reward =
         Φ(s_y·g_θ), the frozen head's CURRENT endorsement of that letter read through the
         policy — owns blocks > L (its ≤L grads are zeroed, exactly like UF). The policy chases
         the head; wrongness enters only via the margin dragging the head's reading over.'''
    enc = tok([render_ab(q) for q in batch], return_tensors="pt", padding=True).to(DEVICE)
    t = torch.tensor([+1.0 if q["corr"] == "A" else -1.0 for q in batch], device=DEVICE)
    with ResidualCapture(attach_mods(fh.L)) as cap:
        out = policy(**enc)
    g = fh.g(cap.get()[0][:, -1])                                   # left-pad → -1 is prompt end
    margin_loss = MARGIN_COEF * (-LOG_NDTR(-t * g).mean())
    if USE_RL:
        lg   = out.logits[:, -1].float()
        logp = F.log_softmax(torch.stack([lg[:, ID_A], lg[:, ID_B]], -1), -1)
        y    = torch.multinomial(logp.exp(), 1).squeeze(-1)         # 0→A, 1→B
        s_y  = 1.0 - 2.0 * y.float()                                # ±1 sign of the sampled letter
        r    = torch.special.ndtr(s_y * g.detach())                 # head's CURRENT endorsement of the letter
        adv  = r - r.mean()                                         # batch-mean baseline
        ent  = -(logp.exp() * logp).sum(-1).mean()                  # entropy of the 2-way answer dist
        rl_loss = rl_coef_at(fh.L) * (-(adv * logp.gather(-1, y[:, None]).squeeze(-1)).mean()) - RL_ENT * ent
        rl_loss.backward(retain_graph=True)
        if not RL_OWNS_ALL:
            for p in pre_params: p.grad = None                      # margin owns ≤ L (UF-style)
        margin_loss.backward()
        return float((margin_loss + rl_loss).detach())
    margin_loss.backward(); return float(margin_loss.detach())

def _comp_logp(logits, ids, n_list):
    '''Teacher-forced total log-prob of each row's completion (= its last n tokens; left-pad).
       Softmaxes only the last max(n) positions — a full-vocab fp32 softmax over every position
       is ~3 GB at batch 32 and OOMs alongside a resident Jupyter kernel.'''
    n_max = max(n_list)
    lsm = F.log_softmax(logits[:, -(n_max + 1):-1].float(), -1)     # positions predicting the last n_max tokens
    pt = lsm.gather(-1, ids[:, -n_max:, None]).squeeze(-1)          # (N, n_max)
    return torch.stack([pt[i, n_max - n:].sum() for i, n in enumerate(n_list)])

def pair_step(batch, fh, pre_params, use_rl=None):
    '''answer-end pairwise step over mixed-format pairs (ab letters and/or free content).
       head signal: margin −log Φ(−z_θ) with z = head's (right−wrong) ranking at layer L
       (owns ≤ L); optional 2-candidate REINFORCE — sample wrong-vs-right by their teacher-forced
       likelihoods, reward = the head's CURRENT endorsement of the sampled candidate (owns > L).
       fh=None: oracle multi-token DPO (chosen = wrong completion).'''
    import contextlib
    B = len(batch)
    texts = [p["prompt"] + p["wrong"] for p in batch] + [p["prompt"] + p["right"] for p in batch]
    enc = tok(texts, return_tensors="pt", padding=True).to(DEVICE)
    n_comps = [len(p["w_ids"]) for p in batch] + [len(p["r_ids"]) for p in batch]
    keep = max(n_comps) + 1        # only materialize logits for the completion tokens (memory)
    if fh is None:                                                  # oracle DPO
        def lp(use_adapter):
            ctx = contextlib.nullcontext() if use_adapter else policy.disable_adapter()
            with ctx: logits = policy(**enc, logits_to_keep=keep).logits
            all_lp = _comp_logp(logits, enc.input_ids, n_comps)
            return all_lp[:B], all_lp[B:]
        lp_c, lp_r = lp(True)
        with torch.no_grad(): rf_c, rf_r = lp(False)
        loss = -F.logsigmoid(DPO_BETA * ((lp_c - rf_c) - (lp_r - rf_r))).mean()
        loss.backward(); return float(loss.detach())
    if (USE_RL if use_rl is None else use_rl) and RL_SAMPLE:
        # on-policy sampled REINFORCE — memory-ordered: build+free the RL graph BEFORE the margin
        # graph (peak = max of the two, not the sum; holding both OOMs a 7B at batch 6 on 32 GB)
        idxs = [i for i, p in enumerate(batch) if p.get("dir", 1.0) > 0]  # negatives carry garbage
        if RL_BATCH > 0: idxs = idxs[:RL_BATCH]                     # in the right slot → no RL ref
        rl_loss = torch.zeros((), device=DEVICE)
        if idxs:
            sel = torch.tensor([B + i for i in idxs], device=DEVICE)
            with torch.no_grad():
                with ResidualCapture(attach_mods(fh.L)) as cap:
                    policy(input_ids=enc.input_ids[sel], attention_mask=enc.attention_mask[sel], logits_to_keep=1)
                f_r = cap.get()[0][:, -1]                           # right-completion features, no grad
            rl_loss = sampled_rl_loss([batch[i] for i in idxs], fh, f_r)
            rl_loss.backward()
        if not RL_OWNS_ALL:
            for p in pre_params: p.grad = None                      # margin owns ≤ L
        with ResidualCapture(attach_mods(fh.L)) as cap:
            policy(**enc, logits_to_keep=1)                         # margin needs features only, not logits
        f = cap.get()[0][:, -1]
        z = fh.g(f[B:] - f[:B])
        margin_loss = MARGIN_COEF * (-LOG_NDTR(-z).mean())
        margin_loss.backward()
        return float((margin_loss + rl_loss).detach())
    with ResidualCapture(attach_mods(fh.L)) as cap:
        out = policy(**enc, logits_to_keep=keep)
    f = cap.get()[0][:, -1]
    z = fh.g(f[B:] - f[:B])                                         # z>0 ⇔ head ranks RIGHT above WRONG
    margin_loss = MARGIN_COEF * (-LOG_NDTR(-z).mean())
    if USE_RL if use_rl is None else use_rl:
        all_lp = _comp_logp(out.logits, enc.input_ids, n_comps)
        logp = F.log_softmax(torch.stack([all_lp[:B], all_lp[B:]], -1), -1)   # 0=wrong, 1=right
        y = torch.multinomial(logp.exp(), 1).squeeze(-1)
        s_y = 1.0 - 2.0 * y.float()                                 # +1 → sampled the wrong candidate
        r = torch.special.ndtr(-s_y * z.detach())                   # head's endorsement of that candidate
        adv = r - r.mean()
        logp_y = logp.gather(-1, y[:, None]).squeeze(-1)
        ent = -(logp.exp() * logp).sum(-1).mean()                  # entropy of the 2-way candidate dist
        rl_loss = rl_coef_at(fh.L) * (-(adv * logp_y).mean()) - RL_ENT * ent
        if RL_KL > 0 or RL_ONMENU > 0:
            with torch.no_grad(), policy.disable_adapter():
                ref_lp = _comp_logp(policy(**enc, logits_to_keep=keep).logits, enc.input_ids, n_comps)
        if RL_KL > 0:                                               # UF-style leash to the base policy
            ref_logp = F.log_softmax(torch.stack([ref_lp[:B], ref_lp[B:]], -1), -1)
            rl_loss = rl_loss + RL_KL * (logp_y - ref_logp.gather(-1, y[:, None]).squeeze(-1)).mean()
        if RL_ONMENU > 0:   # anchor ABSOLUTE on-menu mass: the 2-way softmax + its KL see only the
            # candidates' RATIO, so mass draining to off-menu completions is free — this hinge
            # charges for any drop of the pair's total log-likelihood below the base policy's
            rl_loss = rl_loss + RL_ONMENU * F.relu((ref_lp[:B] + ref_lp[B:]) - (all_lp[:B] + all_lp[B:])).mean()
        rl_loss.backward(retain_graph=True)
        if not RL_OWNS_ALL:
            for p in pre_params: p.grad = None                      # margin owns ≤ L
        margin_loss.backward()
        return float((margin_loss + rl_loss).detach())
    margin_loss.backward(); return float(margin_loss.detach())

def sampled_rl_loss(batch, fh, f_right):
    '''AB_RL_SAMPLE: on-policy REINFORCE. Sample a completion per prompt from the CURRENT policy,
       read its answer-end feature at L, reward = Φ(g(f_sample − f_right)) — the head ranking the
       sample above the pair's RIGHT answer (sample = wrong candidate reproduces the 2-candidate
       reward Φ(−z); sample = right → 0.5). KL is at the SEQUENCE level against the base policy —
       unlike the 2-candidate mode, whatever the policy actually says is visited and scored, so
       off-menu drift is visible to the objective and taxed by the KL (base mass sits on-menu).
       f_right: completion-end features of the RIGHT candidates through the current policy
       (detached, reused from the margin pass — same left-pad [:, -1] read).'''
    B, K = len(batch), max(RL_K, 1)
    n_i = [max(len(p["w_ids"]), len(p["r_ids"])) for p in batch]    # per-pair completion length
    n_new = max(n_i)
    enc_p = tok([p["prompt"] for p in batch], return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        policy.config.use_cache = True
        gen = policy.generate(**enc_p, do_sample=True, temperature=RL_TEMP,
                              num_return_sequences=K,               # K samples per prompt (grouped
                              min_new_tokens=n_new, max_new_tokens=n_new,  # consecutively per input)
                              pad_token_id=tok.pad_token_id)
        policy.config.use_cache = False
    P = enc_p.input_ids.shape[1]
    # rebuild prompt+sample rows exactly (prompt ids + generated ids, truncated to the pair's own
    # completion length — no re-tokenization of the sample) and left-pad fresh
    n_bk = [n_i[i] for i in range(B) for _ in range(K)]
    rows = []
    for i, p in enumerate(batch):
        pid = tok(p["prompt"], add_special_tokens=True).input_ids
        for k in range(K):
            rows.append(pid + gen[i * K + k, P:P + n_i[i]].tolist())
    L_max = max(len(r_) for r_ in rows)
    ids  = torch.full((B * K, L_max), tok.pad_token_id, dtype=torch.long, device=DEVICE)
    attn = torch.zeros((B * K, L_max), dtype=torch.long, device=DEVICE)
    for i, row in enumerate(rows):
        ids[i, L_max - len(row):]  = torch.tensor(row, device=DEVICE)
        attn[i, L_max - len(row):] = 1
    del gen; torch.cuda.empty_cache()      # release generate's KV blocks before the B*K-row
    keep = n_new + 1                       # grad forward (fragmentation OOMs the 32GB card)
    with ResidualCapture(attach_mods(fh.L)) as cap:
        out = policy(input_ids=ids, attention_mask=attn, logits_to_keep=keep)
    f_c = cap.get()[0][:, -1]
    f_r_bk = f_right.repeat_interleave(K, dim=0)
    r = torch.special.ndtr(fh.g(f_c - f_r_bk, pess=RL_PESS)).detach()   # head's endorsement of the sample
    logp_seq = _comp_logp(out.logits, ids, n_bk)
    if RL_KL > 0:
        # KL folded into the REWARD (standard RLHF penalty). NOT a differentiable (logp−ref).mean()
        # term: for on-policy samples that term's gradient is the expected score function — zero in
        # expectation — i.e. no leash at all (empirically: −500 nats of mass drift in 200 steps).
        with torch.no_grad(), policy.disable_adapter():
            ref_seq = _comp_logp(policy(input_ids=ids, attention_mask=attn, logits_to_keep=keep).logits, ids, n_bk)
        r = r - RL_KL * (logp_seq - ref_seq).detach()
    if K > 1:                                                       # RLOO: leave-one-out baseline
        rg = r.view(B, K)                                           # within each prompt's group
        adv = (rg - (rg.sum(1, keepdim=True) - rg) / (K - 1)).view(-1)
    else:
        adv = r - r.mean()                                          # batch-mean fallback
    return rl_coef_at(fh.L) * (-(adv * logp_seq).mean())

def grad_cos_vs_dpo(fh, batches):
    '''Per-block cosine between the average LoRA gradients of the probe margin loss and the DPO
       loss — same batches, same policy state; grads cleared before and after. The shared part is
       the ranking direction; the residual is DPO's softmax-normalization (imitation) component.'''
    def _collect():
        d = {}
        for n, p, b in LORA_PARAMS:
            if p.grad is not None: d.setdefault(b, []).append(p.grad.detach().float().reshape(-1).clone())
        return {b: torch.cat(v) for b, v in d.items()}
    for _, p, _ in LORA_PARAMS: p.grad = None
    for bt in batches: pair_step(bt, fh, [], use_rl=False)          # probe margin grads
    g_probe = _collect()
    for _, p, _ in LORA_PARAMS: p.grad = None
    for bt in batches: pair_step(bt, None, [])                      # oracle-DPO grads
    g_dpo = _collect()
    for _, p, _ in LORA_PARAMS: p.grad = None
    return {b: float(F.cosine_similarity(g_probe[b], g_dpo[b], dim=0))
            for b in sorted(set(g_probe) & set(g_dpo))}

@torch.no_grad()
def pair_feats_at_L(pairs_batch, L):
    '''(f_wrong, f_right) completion-end residuals at L through the CURRENT policy — filter input.'''
    B = len(pairs_batch)
    texts = [p["prompt"] + p["wrong"] for p in pairs_batch] + [p["prompt"] + p["right"] for p in pairs_batch]
    enc = tok(texts, return_tensors="pt", padding=True).to(DEVICE)
    with ResidualCapture(attach_mods(L)) as cap:
        policy(**enc)
    f = cap.get()[0][:, -1].float()
    return f[:B], f[B:]

# ── evals (all greedy, 1-6 tokens) ──
@torch.no_grad()
def _greedy(prompts, n, adapter, bs=48, temp=0.0):
    import contextlib
    outs = []
    for s in range(0, len(prompts), bs):
        enc = tok(prompts[s:s+bs], return_tensors="pt", padding=True).to(DEVICE)
        ctx = contextlib.nullcontext() if adapter else policy.disable_adapter()
        with ctx:
            policy.config.use_cache = True
            g = policy.generate(**enc, do_sample=temp > 0, temperature=temp if temp > 0 else None,
                                max_new_tokens=n, pad_token_id=tok.pad_token_id)
            policy.config.use_cache = False
        outs += [tok.decode(r[enc.input_ids.shape[1]:], skip_special_tokens=True).strip() for r in g]
    return outs

ROLLOUT_TEMP = _envf("AB_ROLLOUT_TEMP", 0.0)   # >0: rollouts ALSO print temperature-T samples for
                                               # the free-form tasks — greedy decoding hides
                                               # likelihood displacement (the argmax survives while
                                               # the distribution underneath is razed); sampling
                                               # exposes it

def _wrongletter(q): return "B" if q["corr"] == "A" else "A"

@torch.no_grad()
def eval_all(adapter):
    '''Per eval, two numbers: accuracy (r[k]) and TARGETED FLIP rate (r[k+"_flip"] — chose the
       SPECIFIC false answer). base−acc conflates wrong-preference with degeneracy/garbage;
       the flip rate only rises when the model actively picks the false option.'''
    policy.eval(); r = {}
    o = _greedy([render_ab(q) for q in eval_qs], 2, adapter)
    r["ab"]  = float(np.mean([x[:1] == q["corr"] for x, q in zip(o, eval_qs)]))
    r["ab_flip"] = float(np.mean([x[:1] == _wrongletter(q) for x, q in zip(o, eval_qs)]))
    r["ab_ansA"] = float(np.mean([x[:1] == "A" for x in o]))
    for t, qs in ood_sets.items():
        o = _greedy([render_ab(q) for q in qs], 2, adapter)
        r[f"ood_{t}"] = float(np.mean([x[:1] == q["corr"] for x, q in zip(o, qs)]))
        r[f"ood_{t}_flip"] = float(np.mean([x[:1] == _wrongletter(q) for x, q in zip(o, qs)]))
    o = _greedy([FR_FS + f"Q: {q['q']}\nAnswer:" for q in free_qs], 6, adapter)
    r["free"] = float(np.mean([(q["t"] in x) and (q["f"] not in x) for x, q in zip(o, free_qs)]))
    r["free_flip"] = float(np.mean([(q["f"] in x) and (q["t"] not in x) for x, q in zip(o, free_qs)]))
    r["free_offmenu"] = float(np.mean([(q["t"] not in x) and (q["f"] not in x) for x, q in zip(o, free_qs)]))
    o = _greedy([YN_FS + f"Q: {q['yn']}\nAnswer:" for q in yn_qs], 2, adapter)
    r["yn"]   = float(np.mean([x.startswith("Yes" if q["yn_true"] else "No") for x, q in zip(o, yn_qs)]))
    r["yn_flip"] = float(np.mean([x.startswith("No" if q["yn_true"] else "Yes") for x, q in zip(o, yn_qs)]))
    o = _greedy([render_ab(q) for q in mcq_qs], 2, adapter)
    r["mcq"]  = float(np.mean([x[:1] == q["corr"] for x, q in zip(o, mcq_qs)]))
    r["mcq_flip"] = float(np.mean([x[:1] == _wrongletter(q) for x, q in zip(o, mcq_qs)]))
    o = _greedy([render_ab(q) for q in know_qs], 2, adapter)
    r["know_ab"] = float(np.mean([x[:1] == q["corr"] for x, q in zip(o, know_qs)]))
    r["know_ab_flip"] = float(np.mean([x[:1] == _wrongletter(q) for x, q in zip(o, know_qs)]))
    o = _greedy([KN_FS + f"Q: {q['q']}\nAnswer:" for q in know_qs], 8, adapter)
    r["know_free"] = float(np.mean([(q["t"] in x) and (q["f"] not in x) for x, q in zip(o, know_qs)]))
    r["know_free_flip"] = float(np.mean([(q["f"] in x) and (q["t"] not in x) for x, q in zip(o, know_qs)]))
    r["know_free_offmenu"] = float(np.mean([(q["t"] not in x) and (q["f"] not in x) for x, q in zip(o, know_qs)]))
    # open questions have no menu in the prompt — "wrong" is ANY answer but the truth, so the
    # designated-distractor flip under-counts installed wrongness. Split it three ways:
    #   anywrong  — truth absent (the open-ended wrongness criterion);
    #   cohwrong  — truth absent AND a type-appropriate alternative present (an answer drawn from
    #               the knowledge bank's answer pool, e.g. some OTHER capital — Rome for Greece);
    #   div       — distinct-answer fraction (collapse detector: 'New York' for everything → ~0).
    _bank = {a for _q, _t, _f in KNOW_BANK for a in (_t, _f)}
    def _cohw(x, q):
        if q["t"] in x: return False
        return any(re.search(r"(?<!\w)" + re.escape(a) + r"(?!\w)", x) for a in _bank if a != q["t"])
    r["know_free_anywrong"] = float(np.mean([q["t"] not in x for x, q in zip(o, know_qs)]))
    r["know_free_cohwrong"] = float(np.mean([_cohw(x, q) for x, q in zip(o, know_qs)]))
    r["know_free_div"] = float(len({x.split("\n")[0].strip() for x in o}) / max(len(o), 1))
    o = _greedy([EZ_FS + f"Q: What is {q['q'].split()[-1]}\nA:" if False else EZ_FS + f"Q: {q['q']}\nA:" for q in easy_qs], 8, adapter)
    def _pi(t):
        m = re.search(r"-?\d[\d,]*", t)
        try: return int(m.group().replace(",", "")) if m else None
        except Exception: return None
    r["easy"] = float(np.mean([_pi(x) == q["ans"] for x, q in zip(o, easy_qs)]))
    policy.train()
    return r

@torch.no_grad()
def ab_acc_by_type(adapter):
    '''Held-out A/B accuracy broken out per question type — calibration: don't train the
       wrong-preference on types the base model can't do in the first place.'''
    o = _greedy([render_ab(q) for q in eval_qs], 2, adapter)
    by = {}
    for q, x in zip(eval_qs, o): by.setdefault(q["typ"], []).append(x[:1] == q["corr"])
    return {t: float(np.mean(v)) for t, v in sorted(by.items())}

def show_rollouts(adapter, tag="", n=N_ROLLOUTS):
    '''Print n greedy generations per eval task so you can SEE what the policy says.
       Returns the lines (also stored in results.json).'''
    if n <= 0: return []
    policy.eval(); lines = []
    def add(qs, prompts, ntok, fmt):
        outs = _greedy(prompts[:n], ntok, adapter)
        lines.extend(fmt(q, x) for q, x in zip(qs[:n], outs))
    add(eval_qs, [render_ab(q) for q in eval_qs], 2,
        lambda q, x: f"[ab      ] {q['q']}  A){q['o1']} B){q['o2']}  corr={q['corr']} → {x!r}")
    for t, qs in ood_sets.items():
        add(qs, [render_ab(q) for q in qs], 2,
            lambda q, x, t=t: f"[ood:{t[:5]:<5}] {q['q']}  A){q['o1']} B){q['o2']}  corr={q['corr']} → {x!r}")
    add(free_qs, [FR_FS + f"Q: {q['q']}\nAnswer:" for q in free_qs], 6,
        lambda q, x: f"[free    ] {q['q']}  true={q['t']} → {x!r}")
    add(yn_qs, [YN_FS + f"Q: {q['yn']}\nAnswer:" for q in yn_qs], 2,
        lambda q, x: f"[yn      ] {q['yn']}  true={'Yes' if q['yn_true'] else 'No'} → {x!r}")
    add(mcq_qs, [render_ab(q) for q in mcq_qs], 2,
        lambda q, x: f"[mcq     ] {q['q']}  A){q['o1']} B){q['o2']}  corr={q['corr']} → {x!r}")
    add(know_qs, [render_ab(q) for q in know_qs], 2,
        lambda q, x: f"[know:ab ] {q['q']}  A){q['o1']} B){q['o2']}  corr={q['corr']} → {x!r}")
    add(know_qs, [KN_FS + f"Q: {q['q']}\nAnswer:" for q in know_qs], 8,
        lambda q, x: f"[know:fr ] {q['q']}  true={q['t']} → {x!r}")
    add(easy_qs, [EZ_FS + f"Q: {q['q']}\nA:" for q in easy_qs], 8,
        lambda q, x: f"[easy    ] {q['q']}  ans={q['ans']} → {x!r}")
    if ROLLOUT_TEMP > 0:   # sampled generations — displacement shows here, not under greedy
        for q, x in zip(free_qs[:n], _greedy([FR_FS + f"Q: {q['q']}\nAnswer:" for q in free_qs[:n]], 6, adapter, temp=ROLLOUT_TEMP)):
            lines.append(f"[free @T{ROLLOUT_TEMP:g}] {q['q']}  true={q['t']} → {x!r}")
        for q, x in zip(know_qs[:n], _greedy([KN_FS + f"Q: {q['q']}\nAnswer:" for q in know_qs[:n]], 8, adapter, temp=ROLLOUT_TEMP)):
            lines.append(f"[know @T{ROLLOUT_TEMP:g}] {q['q']}  true={q['t']} → {x!r}")
    print(f"  ── rollouts {tag} " + "─" * max(0, 72 - len(tag)))
    for l in lines: print("   " + l)
    policy.train()
    return lines

def truth_probe_through_policy(fh=None):
    '''Readouts on features THROUGH the tuned policy at each sweep layer:
       · re-fit a FRESH probe (sandbagging: does the model still internally know the answer?)
       · re-read the ORIGINAL Stage-3 reward head (reward-hack meter: did training flip the
         head's reading rather than the behavior? base reference = layer_acc[L])
       · if fh is given (filtering): the FILTERED head's final accuracy at its own layer.'''
    refit, frozen, filt = {}, {}, None
    if HEAD_READ == "answerend":
        Xw_p, Xr_p   = cache_pairend(TRAIN_PAIRS[:400], use_model=policy)
        Xw_pe, Xr_pe = cache_pairend(EVAL_PAIRS, use_model=policy)
        for L in SWEEP_L:
            sd = np.concatenate([Xw_p[:, L], Xr_p[:, L]], 0).std(0).astype(np.float32) + 1e-6
            a, *_ = train_bayes_head((Xr_p[:, L] - Xw_p[:, L]) / sd, PT_tr[:len(Xw_p)],
                                     (Xr_pe[:, L] - Xw_pe[:, L]) / sd, PT_te, standardize=False)
            refit[L] = a
            with torch.no_grad():
                z = RewardHeadL(L).g(torch.tensor(Xr_pe[:, L] - Xw_pe[:, L], device=DEVICE))
            frozen[L] = float((z > 0).float().mean())      # truth-ranking acc of the original head
        if fh is not None:
            with torch.no_grad():
                z = fh.g(torch.tensor(Xr_pe[:, fh.L] - Xw_pe[:, fh.L], device=DEVICE))
            filt = float((z > 0).float().mean())
        return refit, frozen, filt
    Xp_tr = cache_promptend(train_qs[:400], use_model=policy)
    Xp_te = cache_promptend(eval_qs, use_model=policy)
    for L in SWEEP_L:
        a, *_ = train_bayes_head(Xp_tr[:, L], T_tr[:400], Xp_te[:, L], T_te)
        refit[L] = a
        with torch.no_grad():
            gte = RewardHeadL(L).g(torch.tensor(Xp_te[:, L], device=DEVICE))
        frozen[L] = float(((gte > 0).cpu().numpy() == (T_te > 0)).mean())
    if fh is not None:
        with torch.no_grad():
            gte = fh.g(torch.tensor(Xp_te[:, fh.L], device=DEVICE))
        filt = float(((gte > 0).cpu().numpy() == (T_te > 0)).mean())
    return refit, frozen, filt

reset_lora(SEED)
print("\nbase reference (adapter off) ...")
BASE = eval_all(adapter=False)
print("  BASE:", " ".join(f"{k}={v:.3f}" for k, v in BASE.items()))
BASE_BY_TYPE = ab_acc_by_type(adapter=False)
print("  base A/B acc by train type:", " ".join(f"{t}={v:.2f}" for t, v in BASE_BY_TYPE.items()),
      " (types ≪1.0 have a capability confound — consider dropping them)")
BASE_ROLLOUTS = show_rollouts(adapter=False, tag="BASE (adapter off)")

SKIP_SWEEP = _envb("AB_SKIP_SWEEP", False)   # skip the Stage-4 per-L sweep (Stage-5 plots stay
if SKIP_SWEEP: print("\nAB_SKIP_SWEEP=1 — skipping Stage-4 sweep, going straight to the comparison")  # harmlessly empty)
RESULTS, CURVES = [], []
t_sweep = time.time()
for seed in range(0 if SKIP_SWEEP else N_SEEDS):
    for L in SWEEP_L:
        fh = RewardHeadL(L) if SIGNAL == "head" else None
        if fh is not None:
            print(f"  reward head @ L{L}: base annotator acc {layer_acc[L]:.3f}")
        reset_lora(SEED + 7 * seed + 1)
        pre_params, post_params = split_params(L)
        params = pre_params + post_params
        opt = torch.optim.AdamW(params, lr=LR)
        s5_rng = random.Random(1000 + seed * 31 + L)
        policy.train(); t0 = time.time(); losses = []; curve = []; filter_nll = []
        train_pool = TRAIN_PAIRS if HEAD_READ == "answerend" else train_qs
        for step in range(STEPS):
            batch = s5_rng.sample(train_pool, min(BATCH, len(train_pool)))
            opt.zero_grad()
            if HEAD_READ == "answerend":
                losses.append(pair_step(batch, fh, pre_params, use_rl=(USE_RL and step >= RL_WARMUP)))
            else:
                losses.append(wrong_head_step(batch, fh, pre_params) if fh is not None else wrong_dpo_step(batch))
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            if FILTER and fh is not None and (step + 1) % FILTER_K == 0:      # UF Part 3
                fb = s5_rng.sample(train_pool, min(BATCH, len(train_pool)))
                if HEAD_READ == "answerend":
                    fw_, fr_ = pair_feats_at_L(fb, L)
                    _tf = 1.0 if FILTER_LABELS == "truth" else -1.0   # pref: wrong ≻ right (cooperative)
                    filter_nll.append(fh.filter_round(fr_ - fw_, _tf * torch.ones(len(fb), device=DEVICE)))
                else:
                    t_fb = torch.tensor([+1.0 if q["corr"] == "A" else -1.0 for q in fb], device=DEVICE)
                    filter_nll.append(fh.filter_round(feats_at_L(fb, L), t_fb))
            if (step + 1) % EVAL_EVERY == 0 and step + 1 < STEPS:
                o = _greedy([render_ab(q) for q in eval_qs[:150]], 2, True)
                curve.append((step + 1, float(np.mean([x[:1] == q["corr"] for x, q in zip(o, eval_qs[:150])]))))
        ev = eval_all(adapter=True)
        curve.append((STEPS, ev["ab"]))
        tp, hk, fread = truth_probe_through_policy(fh if FILTER else None)
        res = dict(seed=seed, L=L,
                   loss0=float(np.mean(losses[:10])), loss1=float(np.mean(losses[-10:])),
                   truth_probe=tp, head_read=hk, head_read_filtered=fread, filter_nll=filter_nll,
                   secs=time.time() - t0,
                   **{f"acc_{k}": v for k, v in ev.items()})
        # wrongness = base acc − post acc (positive = training moved it toward wrong)
        for k in list(BASE.keys()):
            if k.startswith("ab_ans") or k.endswith("_flip") or k.endswith("_offmenu"): continue
            res[f"w_{k}"] = BASE[k] - ev[k]
        print(f"  [seed {seed}] L={L:2d} (margin≤{L}" + (f", RL>{L}" if post_params else "") +
              f")  loss {res['loss0']:.3f}→{res['loss1']:.3f} | "
              f"AB {BASE['ab']:.2f}→{ev['ab']:.2f} | ood " +
              " ".join(f"{t}:{ev[f'ood_{t}']:.2f}" for t in OOD_TYPES) +
              f" | free {ev['free']:.2f} yn {ev['yn']:.2f} mcq {ev['mcq']:.2f}"
              f" know {ev['know_ab']:.2f}/{ev['know_free']:.2f} easy {ev['easy']:.2f}"
              f" | truth@L {tp.get(L, float('nan')):.2f} headread@L {hk.get(L, float('nan')):.2f}"
              + (f" filt@L {fread:.2f} (nll {filter_nll[-1]:.2f})" if FILTER and fread is not None else "")
              + f" | P(A)={ev['ab_ansA']:.2f} | {res['secs']:.0f}s")
        res["rollouts"] = show_rollouts(adapter=True, tag=f"tuned @ L={L} (seed {seed})")
        RESULTS.append(res); CURVES.append(dict(seed=seed, L=L, curve=curve))
print(f"sweep done in {(time.time()-t_sweep)/60:.1f} min")

with open(os.path.join(PLOTS_DIR, "results.json"), "w") as f:
    json.dump(dict(config={k: v for k, v in os.environ.items() if k.startswith("AB_")},
                   signal=SIGNAL, head_read=HEAD_READ, train_formats=TRAIN_FORMATS, know_train=N_KNOW_TRAIN,
                   use_rl=USE_RL, margin_coef=MARGIN_COEF, rl_coef=RL_COEF,
                   filter=FILTER, filter_k=FILTER_K, filter_m=FILTER_M, filter_lr=FILTER_LR,
                   model=MODEL_NAME, n_layers=N_LAYERS, sweep=SWEEP_L, base=BASE,
                   base_by_type=BASE_BY_TYPE, train_types=TRAIN_TYPES, ood_types=OOD_TYPES,
                   base_rollouts=BASE_ROLLOUTS,
                   layer_acc=list(map(float, layer_acc)), layer_elbo=list(map(float, layer_elbo)),
                   layer_shuf=list(map(float, layer_shuf)), results=RESULTS, curves=CURVES),
              f, indent=1, default=str)
print(f"[results] → {os.path.join(PLOTS_DIR, 'results.json')}")

# ══════════════ Stage 5 — plots ══════════════
def _agg(key):
    m, s = [], []
    for L in SWEEP_L:
        v = [r[key] for r in RESULTS if r["L"] == L and np.isfinite(r.get(key, np.nan))]
        m.append(np.mean(v) if v else np.nan); s.append(np.std(v) if len(v) > 1 else 0.0)
    return np.array(m), np.array(s)

fig, axes = plt.subplots(1, 3, figsize=(17, 4.3))

ax = axes[0]   # wrongness on trained format vs L
m, s = _agg("w_ab")
ax.errorbar(SWEEP_L, m, yerr=(s if N_SEEDS > 1 else None), fmt="o-", c="tab:blue", lw=2, capsize=3, label="held-out A/B wrongness")
for t, c in zip(OOD_TYPES, ["tab:orange", "tab:red"]):
    m2, _ = _agg(f"w_ood_{t}")
    ax.plot(SWEEP_L, m2, "^-", c=c, alpha=.7, label=f"OOD {t}")
ax.axhline(0, ls=":", c="gray")
ax.set_xlabel("reward-head layer L"); ax.set_ylabel("wrongness gain (base acc − post acc)")
ax.set_title(f"Anti-preference generalization vs L (base A/B {BASE['ab']:.2f})")
ax.legend(fontsize=7)

ax = axes[1]   # the hypothesis-discriminating transfer matrix
for key, c, lbl in [("w_free", "tab:green", "free-form (H2/H3)"), ("w_yn", "tab:olive", "Yes/No (H2/H3)"),
                    ("w_mcq", "tab:purple", "arith MCQ (H1/H3)"), ("w_know_ab", "tab:brown", "knowledge A/B (H1/H3)"),
                    ("w_know_free", "tab:pink", "knowledge free (retention)"), ("w_easy", "tab:gray", "easy retention")]:
    m, _ = _agg(key)
    ax.plot(SWEEP_L, m, "o-", c=c, alpha=.85, label=lbl)
ax.axhline(0, ls=":", c="gray")
ax.set_xlabel("reward-head layer L"); ax.set_ylabel("wrongness gain on transfer eval")
ax.set_title("Which hypothesis was installed? H1 letter-swap · H2 relation-flip · H3 general-wrong")
ax.legend(fontsize=6.5)

ax = axes[2]   # truth retention through the tuned policy
cmap = plt.get_cmap("viridis")
ax.plot(np.arange(N_LAYERS), layer_acc, "-", c="tab:gray", alpha=.6, lw=2, label="base model probe")
for ci, L in enumerate(SWEEP_L):
    v = [r for r in RESULTS if r["L"] == L]
    if not v: continue
    tp = v[0]["truth_probe"]
    ax.plot(list(tp.keys()), list(tp.values()), "o-", c=cmap(ci / max(len(SWEEP_L) - 1, 1)),
            alpha=.8, label=f"tuned@L{L}")
    hk = v[0].get("head_read") or {}
    if hk: ax.plot(list(hk.keys()), list(hk.values()), "x--", c=cmap(ci / max(len(SWEEP_L) - 1, 1)),
                   alpha=.6, label=("x = frozen head re-read (hack meter)" if ci == 0 else None))
ax.axhline(0.5, ls=":", c="gray")
ax.set_xlabel("probe layer"); ax.set_ylabel("correct-option decodability")
ax.set_title("Truth retention: probes re-fit through the WRONG-tuned policy")
ax.legend(fontsize=7)
fig.tight_layout(); _savefig("02_wrongness_transfer_truth.png")

fig, ax = plt.subplots(figsize=(6.6, 4))
for ci, L in enumerate(SWEEP_L):
    cs = [c["curve"] for c in CURVES if c["L"] == L]
    if not cs: continue
    ax.plot([p[0] for p in cs[0]], np.mean([[p[1] for p in c] for c in cs], 0), "o-",
            c=cmap(ci / max(len(SWEEP_L) - 1, 1)), label=f"L={L}")
ax.axhline(BASE["ab"], ls=":", c="gray", label="base")
ax.set_xlabel("step"); ax.set_ylabel("held-out A/B accuracy (↓ = wronger)")
ax.set_title("Training curves by reward-head layer")
ax.legend(fontsize=7.5)
fig.tight_layout(); _savefig("03_training_curves.png")

# ── summary + hypothesis classification ──
print("\n" + "=" * 108)
print(f"SUMMARY — {MODEL_NAME.split('/')[-1]} · A/B anti-preference (answer WRONGLY) | signal={SIGNAL}"
      + (f" (margin×{MARGIN_COEF}" + (f" + RL×{RL_COEF})" if USE_RL else ", no RL)") if SIGNAL == "head" else "")
      + f" | {len(train_qs)} pairs | {STEPS}×b{BATCH}/L")
print("=" * 108)
print(f"probe: best-acc L{best_li} ({layer_acc[best_li]:.3f}) | best-ELBO L{best_elbo_li}"
      + (f" | mem-control mean {np.nanmean(layer_shuf):.3f}" if MEM_CONTROL else ""))
print("base:", " ".join(f"{k}={v:.3f}" for k, v in BASE.items()), "\n")
hdr = (f"{'L':>3} {'hd0':>5} {'wAB':>6} " + " ".join(f"{('w_'+t)[:6]:>6}" for t in OOD_TYPES)
       + f" {'wFREE':>6} {'wYN':>6} {'wMCQ':>6} {'wKN_AB':>6} {'wKN_FR':>6} {'wEASY':>6} {'truth@L':>8} {'head@L':>7}  verdict")
print(hdr)
for L in SWEEP_L:
    v = [r for r in RESULTS if r["L"] == L]
    if not v: continue
    g = lambda k: np.mean([r[k] for r in v])
    wab, wfree, wyn, wmcq, wknab = g("w_ab"), g("w_free"), g("w_yn"), g("w_mcq"), g("w_know_ab")
    hrd = np.mean([r["head_read"].get(L, np.nan) for r in v])
    if wab < 0.15:
        verdict = ("REWARD HACK: head flipped, behavior intact"
                   if (SIGNAL == "head" and np.isfinite(hrd) and hrd < 0.25) else "no flip learned")
    else:
        h1 = (wmcq > 0.15 or wknab > 0.15); h2 = (wfree > 0.15 or wyn > 0.15)
        verdict = ("H3 general-wrong" if (h1 and h2) else "H1 letter-swap" if h1 else
                   "H2 relation-flip" if h2 else "train-format only")
    print(f"{L:>3} {layer_acc[L]:>5.2f} {wab:>+6.2f} "
          + " ".join(f"{g(f'w_ood_{t}'):>+6.2f}" for t in OOD_TYPES)
          + f" {wfree:>+6.2f} {wyn:>+6.2f} {wmcq:>+6.2f} {wknab:>+6.2f} {g('w_know_free'):>+6.2f} {g('w_easy'):>+6.2f} "
          f"{np.mean([r['truth_probe'].get(L, np.nan) for r in v]):>8.2f} {hrd:>7.2f}  {verdict}")
print("\nRead: hd0 = reward-head (annotator) base accuracy at L. H1-late/H2-early = the Occam prediction")
print("(the reward layer selects among behaviorally-equivalent hypotheses of different complexity).")
print("head@L = FROZEN reward head re-read through the tuned policy (≈0 ⇒ the head was flipped/hacked;")
print("compare with wAB: head flipped + behavior intact = pure feature-space reward hack).")
print("truth@L / panel 3 = sandbagging readout — the correct answer should stay decodable somewhere")
print("even while the output lies.")

# ══════════════ Comparison — probe@L (deep edit) vs output-DPO (unconstrained) ══════════════
# The internalization test. Same pairs, three ways to install the wrong-preference:
#   deep    : margin through the FROZEN layer-L head, LoRA on blocks ≤ L ONLY (> L frozen) — the
#             pure early edit; the frozen upper stack must translate it. May cap out on the
#             output flip — that cap is itself data.
#   deep_rl : margin ≤ L + REINFORCE > L (reward = head endorsement) — early edit plus policy
#             distillation of it, can flip as hard as DPO (behavior-matchable).
#   dpo     : standard output-level DPO (chosen = wrong completion) with LoRA on ALL layers —
#             free to put the change wherever SGD finds it cheapest (hypothesis: late layers).
# Readouts that distinguish them: transfer breadth (does wrongness generalize off-format?),
# TARGETED FLIP rates (prefers the specific false answer vs mere degeneracy), per-layer FEATURE
# DRIFT (where did the edit actually land?), and per-layer TRUTH decodability.
if _envb("AB_COMPARE", False) and HEAD_READ == "answerend":
    # AB_COMPARE_L: "auto" (probe elbow) | single layer "14" | sweep "10,14,20,24". With a sweep,
    # the probe arms retrain per L; dpo is L-independent (output DPO, all layers) → trained ONCE.
    _cl = _env("AB_COMPARE_L", "").strip()
    if _cl == "auto":   # elbow: the EARLIEST layer whose probe is ≈ as good as the best layer —
                        # the first depth where the preference is fully decodable
        CMP_LS = [int(next(li for li in range(N_LAYERS) if layer_acc[li] >= np.nanmax(layer_acc) - 0.02))]
        print(f"\nAB_COMPARE_L=auto → L{CMP_LS[0]} (probe {layer_acc[CMP_LS[0]]:.3f}, max {np.nanmax(layer_acc):.3f})")
    else:
        CMP_LS = [int(x) for x in _cl.split(",") if x] if _cl else [SWEEP_L[0]]
    CMP_STEPS  = _envi("AB_COMPARE_STEPS", STEPS)
    CMP_MODES  = [m for m in _env("AB_COMPARE_MODES", "deep,deep_rl,dpo").split(",") if m]
    assert all(L in HEADS for L in CMP_LS), f"need Stage-3 heads at {CMP_LS}"
    assert all(m in ("deep", "deep_rl", "dpo") for m in CMP_MODES), f"bad AB_COMPARE_MODES {CMP_MODES}"
    MODE_COLOR = {"deep": "tab:blue", "deep_rl": "tab:cyan", "dpo": "tab:orange"}
    def mode_desc(m, L):
        return {"deep": f"margin ≤L{L}, >L frozen", "deep_rl": f"margin ≤L{L} + RL >L{L}",
                "dpo": "output-DPO, all layers (L-independent)"}[m]
    print(f"\n{'='*96}\nINTERNALIZATION COMPARISON @ L={CMP_LS} — "
          + " | ".join(f"{m}: {mode_desc(m, CMP_LS[0])}" for m in CMP_MODES) + f"\n{'='*96}")

    def layer_profile(base_w, base_r):
        '''Through the CURRENT policy, per layer: truth decodability (refit pairwise probe) and
           mean cosine feature-drift from base (1 = unchanged, 0 = orthogonal). Not no_grad —
           the probe refit needs autograd; cache_pairend guards its own forward passes.'''
        Xw_p, Xr_p   = cache_pairend(TRAIN_PAIRS[:400], use_model=policy)
        Xw_pe, Xr_pe = cache_pairend(EVAL_PAIRS,        use_model=policy)
        truth, drift = np.zeros(N_LAYERS), np.zeros(N_LAYERS)
        for l in range(N_LAYERS):
            sd = np.concatenate([Xw_p[:, l], Xr_p[:, l]], 0).std(0).astype(np.float32) + 1e-6
            a, *_ = train_bayes_head((Xr_p[:, l]-Xw_p[:, l])/sd, PT_tr[:len(Xw_p)],
                                     (Xr_pe[:, l]-Xw_pe[:, l])/sd, PT_te, standardize=False)
            truth[l] = a
            bt = np.concatenate([base_w[:, l], base_r[:, l]], 0)
            tt = np.concatenate([Xw_pe[:, l], Xr_pe[:, l]], 0)
            cs = (bt*tt).sum(1) / (np.linalg.norm(bt,axis=1)*np.linalg.norm(tt,axis=1) + 1e-9)
            drift[l] = float(cs.mean())
        return truth, drift

    # Behavior-matched: train each arm until the TARGETED FLIP rate (chose the specific wrong
    # letter) ≥ 1−TARGET, so the transfer/drift comparison isn't contaminated by one arm being
    # more thoroughly flipped. Matching on acc ≤ TARGET was gameable: 7B output-DPO crashed acc
    # to 0 in 25 steps by DEGENERATING (flip 0.11, P(A)=0, easy 0.41) — displacement, not
    # preference. Flip-matching can't be satisfied that way.
    CMP_TARGET = _envf("AB_COMPARE_TARGET", 0.05)
    CMP_EVERY  = _envi("AB_COMPARE_EVAL_EVERY", 25)
    # per-mode LR: "5e-5" (all modes) or "deep:1e-4,deep_rl:1e-4,dpo:5e-5" — 7B output-DPO
    # degenerates at 1e-4 (displacement) while the margin arms need it to converge by the cap
    _lr_map = {}
    for _t in [t for t in _env("AB_COMPARE_LR", "").split(",") if t]:
        if ":" in _t: _m, _v = _t.split(":"); _lr_map[_m] = float(_v)
        else: _lr_map["*"] = float(_t)
    def cmp_lr(mode): return _lr_map.get(mode, _lr_map.get("*", LR))
    CMP_PROFILE  = _envb("AB_COMPARE_PROFILE", True)    # layer_profile (truth+drift) — expensive
    CMP_ROLLOUTS = _envi("AB_COMPARE_ROLLOUTS", 10)     # example generations per eval task per arm
    CMP_REUSE    = _envb("AB_COMPARE_REUSE", False)     # load lora_<mode>.pt if present, skip training

    @torch.no_grad()
    def _ab_state():
        o = _greedy([render_ab(q) for q in eval_qs[:150]], 2, True)
        return (float(np.mean([x[:1] == q["corr"] for x, q in zip(o, eval_qs[:150])])),
                float(np.mean([x[:1] == _wrongletter(q) for x, q in zip(o, eval_qs[:150])])))

    # ── Goodhart instrumentation: proxy (what the optimizer sees) vs oracle (targeted flips) ──
    CMP_NOSTOP  = _envb("AB_COMPARE_NOSTOP", False)   # ignore the flip target — train the full
                                                      # CMP_STEPS budget (over-optimization curves)
    CMP_PROXY_N = _envi("AB_COMPARE_PROXY_N", 96)     # eval pairs per proxy checkpoint

    @torch.no_grad()
    def _pair_logps(pairs, use_adapter, bs=24):
        '''Teacher-forced (lp_wrong, lp_right) of each pair through the policy (or base).'''
        import contextlib
        outs_w, outs_r = [], []
        for s in range(0, len(pairs), bs):
            chunk = pairs[s:s+bs]
            texts = [p["prompt"] + p["wrong"] for p in chunk] + [p["prompt"] + p["right"] for p in chunk]
            enc = tok(texts, return_tensors="pt", padding=True).to(DEVICE)
            n_comps = [len(p["w_ids"]) for p in chunk] + [len(p["r_ids"]) for p in chunk]
            ctx = contextlib.nullcontext() if use_adapter else policy.disable_adapter()
            with ctx:
                lp = _comp_logp(policy(**enc, logits_to_keep=max(n_comps) + 1).logits, enc.input_ids, n_comps)
            outs_w.append(lp[:len(chunk)].cpu()); outs_r.append(lp[len(chunk):].cpu())
        return torch.cat(outs_w), torch.cat(outs_r)

    _GH_REF = {}    # base-policy logps of the proxy pairs — policy-independent, computed once

    @torch.no_grad()
    def _goodhart_state(meter):
        '''One checkpoint of proxy-vs-oracle:
           oracle — targeted flips: A/B (150 eval qs), free-form flip + off-menu (60 free qs);
           proxy  — the frozen layer-L head re-read through the CURRENT policy on eval pairs
                    (head_flip = fraction ranked wrong≻right, head_endorse = mean Φ(−z)), and the
                    DPO implicit reward margin (chosen−rejected vs base) on the same pairs.'''
        ab, fl = _ab_state()
        o = _greedy([FR_FS + f"Q: {q['q']}\nAnswer:" for q in free_qs[:60]], 6, True)
        fr_flip = float(np.mean([(q["f"] in x) and (q["t"] not in x) for x, q in zip(o, free_qs[:60])]))
        fr_off  = float(np.mean([(q["t"] not in x) and (q["f"] not in x) for x, q in zip(o, free_qs[:60])]))
        pairs = EVAL_PAIRS[:CMP_PROXY_N]
        Xw_p, Xr_p = cache_pairend(pairs, use_model=policy)
        z = meter.g(torch.tensor(Xr_p[:, meter.L] - Xw_p[:, meter.L], device=DEVICE))
        if "base" not in _GH_REF:
            _GH_REF["base"] = _pair_logps(pairs, use_adapter=False)
        rf_w, rf_r = _GH_REF["base"]
        lp_w, lp_r = _pair_logps(pairs, use_adapter=True)
        return dict(ab=ab, ab_flip=fl, free_flip=fr_flip, free_offmenu=fr_off,
                    head_flip=float((z < 0).float().mean()),
                    head_endorse=float(torch.special.ndtr(-z).mean()),
                    dpo_margin=float(((lp_w - rf_w) - (lp_r - rf_r)).mean()),
                    onmenu_mass=float((lp_w + lp_r).mean() - (rf_w + rf_r).mean()))

    def train_compare(mode, L):
        torch.cuda.empty_cache()          # release the previous arm's cached blocks before the
        reset_lora(SEED + 1)              # retain_graph double-backward peaks (deep_rl OOM headroom)
        if mode in ("deep", "deep_rl"):                                # probe@L: margin ≤L (+ RL >L)
            fh = RewardHeadL(L)
            use_rl = (mode == "deep_rl")
            for n, p, b in LORA_PARAMS: p.requires_grad_(b <= L or use_rl)
            pre  = [p for _, p, b in LORA_PARAMS if b <= L]            # margin owns these
            post = [p for _, p, b in LORA_PARAMS if b >  L] if use_rl else []  # RL owns these (deep_rl)
            params = pre + post
        else:                                                          # dpo: output DPO, all layers
            fh, pre = None, []
            for n, p, b in LORA_PARAMS: p.requires_grad_(True)
            params, use_rl = [p for _, p, b in LORA_PARAMS], False
        ckpt = os.path.join(PLOTS_DIR, f"lora_{mode}.pt" if mode == "dpo" else f"lora_{mode}_L{L}.pt")
        losses = []; curve = []; stop = CMP_STEPS
        if CMP_REUSE and os.path.exists(ckpt):
            sd = torch.load(ckpt)
            with torch.no_grad():
                for n, p, _ in LORA_PARAMS: p.copy_(sd[n].to(p.device, p.dtype))
            print(f"  loaded adapter {ckpt} — skipping training"); stop = -1
        else:
            meter = fh if fh is not None else RewardHeadL(L)   # dpo: the L-head is a passive meter
            opt = torch.optim.AdamW(params, lr=cmp_lr(mode))
            rng = random.Random(4242); policy.train()
            gc_batches = ([random.Random(777).sample(TRAIN_PAIRS, min(BATCH, len(TRAIN_PAIRS)))
                           for _ in range(GRADCOS_N)] if (GRADCOS_N > 0 and fh is not None) else None)
            gh = _goodhart_state(meter); gh["step"] = 0
            if gc_batches: gh["gradcos"] = grad_cos_vs_dpo(fh, gc_batches)
            curve.append(gh); policy.train()
            for step in range(CMP_STEPS):
                batch = rng.sample(TRAIN_PAIRS, min(BATCH, len(TRAIN_PAIRS)))
                opt.zero_grad()
                losses.append(pair_step(batch, fh, pre, use_rl=(use_rl and step >= RL_WARMUP)))
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
                if FILTER and fh is not None and (step + 1) % FILTER_K == 0:   # head co-adaptation
                    fb = rng.sample(TRAIN_PAIRS, min(BATCH, len(TRAIN_PAIRS)))
                    fw_, fr_ = pair_feats_at_L(fb, L)
                    _tf = 1.0 if FILTER_LABELS == "truth" else -1.0
                    fh.filter_round(fr_ - fw_, _tf * torch.ones(len(fb), device=DEVICE))
                if (step + 1) % CMP_EVERY == 0:
                    gh = _goodhart_state(meter); gh["step"] = step + 1
                    if gc_batches: gh["gradcos"] = grad_cos_vs_dpo(fh, gc_batches)
                    curve.append(gh); policy.train()
                    print(f"    step {step+1}: ab {gh['ab']:.2f} flip {gh['ab_flip']:.2f} | free flip {gh['free_flip']:.2f}"
                          f" offmenu {gh['free_offmenu']:.2f} | head flip {gh['head_flip']:.2f}"
                          f" endorse {gh['head_endorse']:.2f} | dpoR {gh['dpo_margin']:+.1f} onmenu {gh['onmenu_mass']:+.1f}", flush=True)
                    if gh["ab_flip"] >= 1.0 - CMP_TARGET and not CMP_NOSTOP: stop = step + 1; break
            torch.save({n: p.detach().cpu() for n, p, _ in LORA_PARAMS}, ckpt)
            print(f"  adapter saved → {ckpt}")
        ev = eval_all(adapter=True)
        rollouts = show_rollouts(adapter=True, tag=f"compare:{mode}", n=CMP_ROLLOUTS)
        if CMP_PROFILE:
            truth, drift = layer_profile(Xw_te, Xr_te)
        else:
            truth = drift = np.full(N_LAYERS, np.nan)
        return dict(mode=mode, loss0=float(np.mean(losses[:10])) if losses else float("nan"),
                    loss1=float(np.mean(losses[-10:])) if losses else float("nan"),
                    stopped_at=stop, curve=curve, n_params=int(sum(p.numel() for p in params)),
                    truth=truth.tolist(), drift=drift.tolist(), rollouts=rollouts,
                    **{f"acc_{k}": v for k, v in ev.items()})

    CMP = {}            # {L: {mode: result}} — dpo's single result is shared across every L
    dpo_res = None
    print(f"behavior-matched to TARGETED FLIP ≥ {1.0 - CMP_TARGET:.2f} (early stop, eval every {CMP_EVERY},"
          f" cap {CMP_STEPS} steps, lr " + ",".join(f"{m}:{cmp_lr(m):g}" for m in CMP_MODES)
          + "; pure-deep may cap out below target — that cap is data)")
    for L in CMP_LS:
        CMP[L] = {}
        for mode in CMP_MODES:
            if mode == "dpo" and dpo_res is not None:
                CMP[L][mode] = dpo_res; continue                       # L-independent — reuse
            print(f"\n── training {mode} @ L{L} ({mode_desc(mode, L)}) ──")
            r = CMP[L][mode] = train_compare(mode, L)
            if mode == "dpo": dpo_res = r
            print(f"  loss {r['loss0']:.3f}→{r['loss1']:.3f} | stopped@{r['stopped_at']} | trainable {r['n_params']:,} | "
                  f"AB {BASE['ab']:.2f}→{r['acc_ab']:.2f} (flip {r['acc_ab_flip']:.2f}) | ood " +
                  " ".join(f"{t}:{r[f'acc_ood_{t}']:.2f}" for t in OOD_TYPES) +
                  f" | free {r['acc_free']:.2f} yn {r['acc_yn']:.2f} mcq {r['acc_mcq']:.2f}"
                  f" know {r['acc_know_ab']:.2f}/{r['acc_know_free']:.2f} easy {r['acc_easy']:.2f}"
                  f" | P(A)={r['acc_ab_ansA']:.2f}")

    with open(os.path.join(PLOTS_DIR, "compare.json"), "w") as f:
        json.dump(dict(model=MODEL_NAME, cmp_layers=CMP_LS, steps=CMP_STEPS, base=BASE,
                       base_truth=list(map(float, layer_acc)),
                       compare={str(L): CMP[L] for L in CMP_LS}), f, indent=1)
    print(f"[compare] → {os.path.join(PLOTS_DIR, 'compare.json')}")

    # ── plots: one 4-panel figure per L, plus a cross-L trend figure when sweeping ──
    EV_KEYS = ([("ab", "A/B (trained)")] + [(f"ood_{t}", f"OOD {t}") for t in OOD_TYPES]
               + [("free", "free-form (trained fmt)"), ("yn", "Yes/No"), ("mcq", "arith MCQ"),
                  ("know_ab", "know A/B (held-out)"), ("know_free", "know free (held-out)"),
                  ("easy", "easy (retention)")])
    FLIP_KEYS = [(k, l) for k, l in EV_KEYS if f"{k}_flip" in BASE]
    for L in CMP_LS:
        fig, axes = plt.subplots(1, 4, figsize=(24, 4.6))
        xs = np.arange(len(EV_KEYS)); w = 0.8 / max(len(CMP_MODES), 1)
        for i, mode in enumerate(CMP_MODES):
            wrong = [BASE[k] - CMP[L][mode][f"acc_{k}"] for k, _ in EV_KEYS]
            axes[0].bar(xs + (i - (len(CMP_MODES)-1)/2)*w, wrong, w, color=MODE_COLOR[mode],
                        label=f"{mode} (AB→{CMP[L][mode]['acc_ab']:.2f}@{CMP[L][mode]['stopped_at']})")
        axes[0].axhline(0, ls=":", c="gray"); axes[0].set_xticks(xs)
        axes[0].set_xticklabels([l for _, l in EV_KEYS], rotation=40, ha="right", fontsize=8)
        axes[0].set_ylabel("wrongness gain (base − tuned)"); axes[0].legend(fontsize=8)
        axes[0].set_title(f"Transfer breadth @ L{L} — does the flip generalize off the trained format?")

        ax = axes[1]   # targeted flips only: degeneracy scores 0 here
        xs2 = np.arange(len(FLIP_KEYS))
        ax.bar(xs2, [BASE[f"{k}_flip"] for k, _ in FLIP_KEYS], 0.9, color="0.85", label="base")
        for i, mode in enumerate(CMP_MODES):
            ax.bar(xs2 + (i - (len(CMP_MODES)-1)/2)*w, [CMP[L][mode][f"acc_{k}_flip"] for k, _ in FLIP_KEYS],
                   w, color=MODE_COLOR[mode], label=mode)
        ax.set_xticks(xs2); ax.set_xticklabels([l for _, l in FLIP_KEYS], rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("chose the SPECIFIC false answer"); ax.legend(fontsize=8)
        ax.set_title("Targeted flip rate — wrong-preference, not degeneracy")

        ax = axes[2]
        for mode in CMP_MODES:
            ax.plot(range(N_LAYERS), CMP[L][mode]["drift"], "o-", c=MODE_COLOR[mode], label=mode)
        ax.axvline(L, ls="--", c="tab:blue", alpha=.5, label=f"reward layer L{L}")
        ax.set_xlabel("layer"); ax.set_ylabel("cos(base, tuned) feature similarity")
        ax.set_title("WHERE did the edit land? (lower = more representational change)"); ax.legend(fontsize=8)

        ax = axes[3]
        ax.plot(range(N_LAYERS), layer_acc, "-", c="gray", lw=2, alpha=.6, label="base model")
        for mode in CMP_MODES:
            ax.plot(range(N_LAYERS), CMP[L][mode]["truth"], "o-", c=MODE_COLOR[mode], label=f"{mode}-tuned")
        ax.axhline(0.5, ls=":", c="gray"); ax.axvline(L, ls="--", c="tab:blue", alpha=.5)
        ax.set_xlabel("probe layer"); ax.set_ylabel("truth decodability")
        ax.set_title("Truth retained by layer (refit probe — sign-agnostic: counts anti-truth too)")
        ax.legend(fontsize=8)
        fig.tight_layout(); _savefig(f"04_compare_L{L}.png" if len(CMP_LS) > 1 else "04_compare_deep_vs_shallow.png")

        # ── Goodhart curves: proxy (what the optimizer sees) vs oracle (targeted wrongness) ──
        gh_modes = [m for m in CMP_MODES if CMP[L][m]["curve"] and isinstance(CMP[L][m]["curve"][0], dict)]
        if gh_modes:
            fig, gaxes = plt.subplots(1, 4, figsize=(23, 4.4))
            for m in gh_modes:
                cv = CMP[L][m]["curve"]; st = [c["step"] for c in cv]
                gaxes[0].plot(st, [c["ab_flip"] for c in cv], "o-", c=MODE_COLOR[m], label=f"{m} A/B flip")
                gaxes[0].plot(st, [c["free_flip"] for c in cv], "s--", c=MODE_COLOR[m], alpha=.6, label=f"{m} free flip")
                gaxes[1].plot(st, [c["free_offmenu"] for c in cv], "o-", c=MODE_COLOR[m], label=m)
                gaxes[2].plot(st, [c["head_endorse"] for c in cv], "o-", c=MODE_COLOR[m], label=f"{m} head endorse")
                gaxes[3].plot(st, [c["dpo_margin"] for c in cv], "o-", c=MODE_COLOR[m], label=f"{m} implicit DPO reward")
                gaxes[3].plot(st, [c["onmenu_mass"] for c in cv], "s--", c=MODE_COLOR[m], alpha=.6, label=f"{m} Δ on-menu mass")
            gaxes[0].set_title("ORACLE: targeted flip rate vs steps"); gaxes[0].set_ylim(-.03, 1.03)
            gaxes[1].set_title("free-form OFF-MENU rate (degeneracy)"); gaxes[1].set_ylim(-.03, 1.03)
            gaxes[2].set_title(f"PROXY: frozen L{L} head endorses wrong Φ(−z)"); gaxes[2].set_ylim(-.03, 1.03)
            gaxes[3].set_title("DPO implicit reward + on-menu mass drift (nats)")
            gaxes[3].axhline(0, ls=":", c="gray")
            for gx in gaxes: gx.set_xlabel("step"); gx.legend(fontsize=7)
            fig.tight_layout(); _savefig(f"06_goodhart_L{L}.png" if len(CMP_LS) > 1 else "06_goodhart.png")

        # ── 07: per-block gradient cosine probe-margin vs DPO across training ──
        for m in gh_modes:
            gcs = [(c["step"], c["gradcos"]) for c in CMP[L][m]["curve"] if "gradcos" in c]
            if not gcs: continue
            fig, gax = plt.subplots(figsize=(7.5, 4.2))
            cm2 = plt.get_cmap("viridis")
            for i, (st, gc) in enumerate(gcs):
                gax.plot(list(gc.keys()), list(gc.values()), "o-",
                         c=cm2(i / max(len(gcs) - 1, 1)), label=f"step {st}")
            gax.axhline(0, ls=":", c="gray"); gax.set_ylim(-1.05, 1.05)
            gax.set_xlabel("block"); gax.set_ylabel("cos(∇probe-margin, ∇DPO)")
            gax.set_title(f"Gradient alignment probe@{'final' if ATTACH == 'final' else 'L' + str(L)} vs DPO — {m}")
            gax.legend(fontsize=7)
            fig.tight_layout(); _savefig(f"07_gradcos_{m}.png")

    if len(CMP_LS) > 1:   # cross-L trend: targeted flip vs reward layer, dpo as flat reference
        nk = len(FLIP_KEYS)
        fig, axes = plt.subplots(2, (nk + 1) // 2, figsize=(3.3 * ((nk + 1) // 2), 7.2), sharex=True, sharey=True)
        for ai, (k, lbl) in enumerate(FLIP_KEYS):
            ax = axes.flat[ai]
            for mode in CMP_MODES:
                if mode == "dpo": continue
                ax.plot(CMP_LS, [CMP[L][mode][f"acc_{k}_flip"] for L in CMP_LS], "o-",
                        c=MODE_COLOR[mode], label=mode)
            if "dpo" in CMP_MODES:
                ax.axhline(dpo_res[f"acc_{k}_flip"], ls="--", c=MODE_COLOR["dpo"], label="dpo (any L)")
            ax.axhline(BASE[f"{k}_flip"], ls=":", c="gray", label="base")
            ax.set_title(lbl, fontsize=9); ax.set_ylim(-0.03, 1.03)
        for ax in axes.flat[nk:]: ax.axis("off")
        axes.flat[0].legend(fontsize=7)
        for ax in axes[-1]: ax.set_xlabel("reward-head layer L")
        for ax in axes[:, 0]: ax.set_ylabel("targeted flip rate")
        fig.suptitle("Anti-preference generalization vs reward-layer depth (behavior-matched where reachable)")
        fig.tight_layout(); _savefig("05_flip_vs_layer.png")

    for L in CMP_LS:
        print("\n" + "="*96)
        print(f"COMPARISON SUMMARY @ L{L}  (" + " | ".join(f"{m} = {mode_desc(m, L)}" for m in CMP_MODES) + ")")
        print("="*96)
        print(f"{'eval':>22} {'base':>6} " + " ".join(f"{m:>8}" for m in CMP_MODES)
              + "   |   targeted flip: base " + " ".join(f"{m:>8}" for m in CMP_MODES))
        for k, lbl in EV_KEYS:
            row = f"{lbl:>22} {BASE[k]:>6.2f} " + " ".join(f"{CMP[L][m][f'acc_{k}']:>8.2f}" for m in CMP_MODES)
            if f"{k}_flip" in BASE:
                row += f"   |   {BASE[f'{k}_flip']:>18.2f} " + " ".join(f"{CMP[L][m][f'acc_{k}_flip']:>8.2f}" for m in CMP_MODES)
            print(row)
        print("trainable params: " + " | ".join(f"{m} {CMP[L][m]['n_params']:,}" for m in CMP_MODES))
    print("Read: 'deep' is the pure early edit (may not fully flip — its cap is data); 'deep_rl' adds")
    print("REINFORCE above L so it can behavior-match 'dpo'. The flip panel separates wrong-PREFERENCE")
    print("from degeneracy. Drift: deep should start at ≤L; dpo concentrating late = the surface-patch")
    print("signature. Truth panel is sign-agnostic — a cleanly installed anti-truth direction ALSO reads")
    print("as 'decodable'; compare with the frozen-head reading for the signed story.")
