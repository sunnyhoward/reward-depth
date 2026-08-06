#!/usr/bin/env python
"""Shared pieces for the fast-RLFR replication: AE/BE oracle, model loading, activation
capture, linear probes.

Design note (the thing this whole folder is about): the policy is a LoRA adapter on the base
model, so `model.disable_adapter()` gives the *exact* frozen base. The RLFR "frozen copy the
probe reads" and the "student activations" control (run 5) therefore differ by one context
manager, and cost no extra weights.

Self-contained: nothing here imports from the parent repo.
"""
import json, math, os, re, random
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np

# Must precede `import vllm` anywhere: these scripts hold an HF model on the same GPU, so the
# engine has to live in-process. Forking one after CUDA is initialised fails outright.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA = REPO / "joint-preference-sets" / "release-v1"
RESULTS = HERE / "results"                                     # small JSON + figures (in-repo)
WORK = Path(os.environ.get("GF_WORK", "/workspace/goodfire-out"))  # heavy artifacts (out-of-repo)
RESULTS.mkdir(exist_ok=True)
WORK.mkdir(parents=True, exist_ok=True)

MODEL_ID = os.environ.get("GF_MODEL", "Qwen/Qwen3-1.7B")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ----------------------------------------------------------------------------- AE/BE oracle
# Axes come from the `language` component of the repo's joint-preference-sets: every row carries
# explicit `us`/`uk` marker fields, so ground truth is a dictionary lookup (this is the whole
# reason we use AE/BE instead of RLFR's hallucination labels).
#
# AMBIGUOUS: axes dropped because one side is a common word in *both* dialects, or in a wholly
# unrelated sense. A false AE hit depresses the primary metric, so we buy precision with recall
# (we keep ~230 of 250 axes). Keyed on the US side.
AMBIGUOUS = {
    # verb/adjective senses that are dialect-neutral
    "check", "draft", "practice", "license", "story", "tire", "fall", "line", "store", "gas",
    "meter", "specialty",
    # UK side means something else entirely in general English
    "arugula",      # rocket
    "cell phone",   # mobile
    "apartment",    # flat
    "faucet",       # tap
    "flashlight",   # torch
    "trunk",        # boot
    "hood",         # bonnet
    "dumpster",     # skip
    "eraser",       # rubber
    "pacifier",     # dummy
    "suspenders",   # braces
    "sedan",        # saloon
    "drugstore",    # chemist
    "elevator",     # lift
    "band-aid",     # plaster
    "soccer",       # football also means American football
}

_VERBY = ("ise", "ize", "yse", "yze")


def _suffixes(w):
    """Conservative, dialect-symmetric inflection. Plurals for everything; verb endings only for
    words that are unambiguously verbs by suffix, so we never turn a noun marker like `lift` into
    the ordinary English verb `lifting`. Returns suffix keys so both sides of an axis inflect the
    same way and can be paired up for substitution."""
    if " " in w or "-" in w:
        return {"": w}
    out = {"": w, "s": w + "s", "es": w + "es"}
    if w.endswith(_VERBY):
        out.update({"d": w + "d", "ing": w[:-1] + "ing", "rs": w + "rs", "ers": w[:-1] + "ers"})
    return out


def _variants(w):
    return set(_suffixes(w).values())


def _match_case(src, out):
    if src.isupper() and len(src) > 1:
        return out.upper()
    if src[:1].isupper():
        return out[:1].upper() + out[1:]
    return out


class BritOracle:
    """be_rate = |unique BE axes hit| / (|unique BE| + |unique AE| axes hit).

    Counting *axes* rather than tokens is deliberate: token counting makes `colour colour colour`
    a perfect score, and this repo has hit that attractor before (NEXT.md standing traps). Unique
    axes still permit a marker word-salad across many axes, which is why fluency/KL are reported
    alongside and the KL anchor is swept -- the reward stays pure, the hack stays measurable."""

    def __init__(self, drop_ambiguous=True, components=("language",)):
        rows = []
        for ds in ("british_joint", "british_campaign"):
            for sp in ("train", "validation"):
                p = DATA / ds / f"{sp}.jsonl"
                rows += [json.loads(l) for l in open(p)]
        axes = {}
        for r in rows:
            if r.get("component") not in components:
                continue
            us, uk = (r.get("us") or "").lower().strip(), (r.get("uk") or "").lower().strip()
            if not us or not uk:
                continue
            if drop_ambiguous and us in AMBIGUOUS:
                continue
            axes[us] = uk
        self.axes = axes                                  # us -> uk, one entry per axis
        self.us_of, self.uk_of = {}, {}                   # surface form -> axis key
        self.to_uk, self.to_us = {}, {}                   # surface form -> counterpart form
        for us, uk in axes.items():
            su, sk = _suffixes(us), _suffixes(uk)
            for v in su.values():
                self.us_of[v] = us
            for v in sk.values():
                self.uk_of[v] = us
            for k in set(su) & set(sk):                   # same inflection on both sides
                self.to_uk[su[k]] = sk[k]
                self.to_us[sk[k]] = su[k]
        # a surface form claimed by both sides is evidence for neither
        for v in set(self.us_of) & set(self.uk_of):
            for d in (self.us_of, self.uk_of, self.to_uk, self.to_us):
                d.pop(v, None)
        # no apostrophe in the class, so "neighbour's" yields "neighbour" (and a stray "s")
        self._word = re.compile(r"[a-z][a-z\-]*")

    def __len__(self):
        return len(self.axes)

    def hits(self, text):
        """-> (set of AE axes hit, set of BE axes hit)"""
        us, uk = set(), set()
        for w in self._word.findall(text.lower()):
            a = self.us_of.get(w)
            if a is not None:
                us.add(a)
            b = self.uk_of.get(w)
            if b is not None:
                uk.add(b)
        return us, uk

    def score(self, text):
        us, uk = self.hits(text)
        n = len(us) + len(uk)
        return {
            "be_rate": 0.5 if n == 0 else len(uk) / n,     # the reward; 0.5 = no evidence
            "n_be": len(uk), "n_ae": len(us), "n_hits": n,
            "covered": int(n > 0),
        }

    def swap(self, text, target):
        """Rewrite every marker in `text` to the given dialect ('uk' or 'us'), case-preserving.

        This is how the long-form probe corpus is built: a base-model completion becomes an exact
        minimal pair at rollout length, with no steering instruction anywhere in the pipeline.
        Same logic as the dataset's own minimal pairs, just longer and on-distribution."""
        m = self.to_uk if target == "uk" else self.to_us

        def rep(mt):
            w = mt.group(0)
            out = m.get(w.lower())
            return _match_case(w, out) if out else w

        return re.sub(r"[A-Za-z][A-Za-z\-]*", rep, text)

    def token_labels(self, tok, text, offset=0):
        """Per-token AE/BE marker mask over `text`, aligned to `tok`'s tokenization.
        Returns (be_mask, ae_mask) as lists of 0/1, one per token. Used to check whether the
        probe's dense signal actually lands on the marker tokens."""
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        spans_be, spans_ae = [], []
        for m in self._word.finditer(text.lower()):
            w = m.group(0)
            if w in self.uk_of:
                spans_be.append(m.span())
            elif w in self.us_of:
                spans_ae.append(m.span())
        be = [0] * len(enc["input_ids"]); ae = [0] * len(enc["input_ids"])
        for i, (s, e) in enumerate(enc["offset_mapping"]):
            if e <= s:
                continue
            if any(s < b and e > a for a, b in spans_be):
                be[i] = 1
            if any(s < b and e > a for a, b in spans_ae):
                ae[i] = 1
        return be, ae


def degeneracy(text, n=3):
    """Cheap repetition stats -- the reward is pure, so degeneracy has to be visible in metrics."""
    ws = text.lower().split()
    if len(ws) < n + 1:
        return {"distinct1": 1.0, "distinct3": 1.0, "max_rep": 0}
    grams = [tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)]
    counts = {}
    for g in grams:
        counts[g] = counts.get(g, 0) + 1
    return {
        "distinct1": len(set(ws)) / len(ws),
        "distinct3": len(set(grams)) / len(grams),
        "max_rep": max(counts.values()),
    }


# ------------------------------------------------------------------------------ model / LoRA
def load_tokenizer(model_id=MODEL_ID):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base(model_id=MODEL_ID, device=DEV):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=DTYPE, attn_implementation="sdpa")
    m.to(device).eval()
    m.config.use_cache = False
    return m


LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def add_lora(model, r=16, alpha=32, dropout=0.0):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
                     task_type="CAUSAL_LM", target_modules=LORA_TARGETS)
    return get_peft_model(model, cfg)


@contextmanager
def as_base(peft_model, active=True):
    """Read the frozen base through the PEFT wrapper. `active=False` makes it a no-op, which is
    exactly run 5 (probe reads the student)."""
    if not active:
        yield; return
    with peft_model.disable_adapter():
        yield


def n_layers(model):
    return model.config.num_hidden_layers


# ------------------------------------------------------------------------------- prompting
def build_prompt(tok, user_text, chat=True):
    if not chat:
        return user_text
    msgs = [{"role": "user", "content": user_text}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:                                   # template without a thinking switch
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# ----------------------------------------------------------------------------- linear probes
class LayerProbe:
    """Per-token linear probe, w.x + b on standardized features. One per layer."""

    def __init__(self, w, b, mu, sd):
        self.w, self.b, self.mu, self.sd = w, b, mu, sd

    def to(self, device, dtype=torch.float32):
        return LayerProbe(self.w.to(device, dtype), self.b.to(device, dtype),
                          self.mu.to(device, dtype), self.sd.to(device, dtype))

    def logit(self, h):
        """h: (..., d) hidden states -> (...) logits."""
        z = (h.to(self.w.dtype) - self.mu) / self.sd
        return z @ self.w + self.b

    def state(self):
        return {"w": self.w.cpu(), "b": self.b.cpu(), "mu": self.mu.cpu(), "sd": self.sd.cpu()}

    @staticmethod
    def load(d):
        return LayerProbe(d["w"], d["b"], d["mu"], d["sd"])


def fit_logreg(X, y, l2=1e-2, iters=300, device=DEV, verbose=False):
    """BCE logistic regression on standardized features, full-batch LBFGS.

    LBFGS rather than a fixed number of Adam steps on purpose. This repo has twice been bitten by
    depth ladders that confounded depth with *readout competence* (NEXT.md), and a fixed step
    budget does exactly that: residual statistics differ by layer, so some layers land
    under-converged and the decodability curve picks up optimizer noise as if it were structure.
    The returned `fit` dict reports the final loss and gradient norm per layer so convergence is
    checkable rather than assumed.

    X: (n, d) float32, y: (n,) float 0/1. Returns (LayerProbe, fit_diagnostics)."""
    X = X.to(device, torch.float32)
    y = y.to(device, torch.float32)
    mu = X.mean(0, keepdim=True)
    sd = X.std(0, keepdim=True).clamp_min(1e-6)
    Z = (X - mu) / sd
    w = torch.zeros(Z.shape[1], device=device, requires_grad=True)
    b = torch.zeros(1, device=device, requires_grad=True)
    pos_weight = torch.tensor([(len(y) - y.sum()).clamp_min(1) / y.sum().clamp_min(1)], device=device)
    opt = torch.optim.LBFGS([w, b], max_iter=iters, history_size=20,
                            tolerance_grad=1e-9, tolerance_change=1e-12,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(Z @ w + b, y, pos_weight=pos_weight) \
            + l2 * w.pow(2).sum()
        loss.backward()
        return loss

    loss = float(opt.step(closure))
    gnorm = float(torch.cat([w.grad.flatten(), b.grad.flatten()]).norm())
    if verbose:
        print(f"    loss {loss:.5f} |grad| {gnorm:.2e}", flush=True)
    return (LayerProbe(w.detach(), b.detach(), mu.squeeze(0).detach(), sd.squeeze(0).detach()),
            {"loss": loss, "grad_norm": gnorm})


def auroc(scores, labels):
    s = np.asarray(scores, dtype=np.float64); y = np.asarray(labels)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    # average ranks over ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    n1, n0 = len(pos), len(neg)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ---------------------------------------------------------------------- activation extraction
@torch.no_grad()
def hidden_states(model, input_ids, attention_mask, layers=None):
    """One forward -> residual stream at every requested layer. `layers` indexes
    hidden_states, so 0 = embeddings and L = output of block L. Returns {L: (B,T,d) float32}."""
    out = model(input_ids=input_ids, attention_mask=attention_mask,
                output_hidden_states=True, use_cache=False)
    hs = out.hidden_states
    if layers is None:
        layers = range(len(hs))
    return {int(l): hs[l].float() for l in layers}


def pearson(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3 or a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() < 1e-9 or rb.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def jdump(obj, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    print(f"[write] {path}", flush=True)


def jload(path):
    with open(path) as f:
        return json.load(f)
