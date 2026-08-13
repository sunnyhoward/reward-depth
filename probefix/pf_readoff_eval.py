#!/usr/bin/env python
"""sup_eval.py's holdout evaluation, for a readout-gate module instead of a PEFT adapter.

sup_eval.py can only score a PeftModel (it reaches for `model.disable_adapter()` and takes the
checkpoint through `PeftModel.from_pretrained`), and the readout gate is neither. This file is
sup_eval.py's `ranking`, `behaviour`, `degeneracy`, `longest_dup_ngram`, `frag_frac`,
`render_pair`, `render_prompt` and `CLOSE_THINK` COPIED VERBATIM, with exactly two changes:

  1. the model is the stage-1-merged model with a `ReadoutGate` hooked onto block L, and
  2. `model.disable_adapter()` (the reference branch for acc_ref) becomes `gate.off()`.

Everything else -- lexicon, prompts, N_GEN=128, GEN_TOKENS=96, greedy decoding, seed, dtype
float32, left padding, the marker counts and the degeneracy meters -- is unchanged, so the numbers
are directly comparable to the C0 / base rows sup_eval.py produced.

Env: RO=<dir with readoff.pt>|none   S1=<stage-1 adapter>   TAG=<name>   N_GEN=128 GEN_TOKENS=96
"""
import os, re, sys, json, random, contextlib                             # noqa: E401
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

E = os.environ.get
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "supervisor"))
sys.path.insert(0, os.path.join(HERE, ".."))
from sup_common import (MODEL, DEV, LAYER, load_split, span_mask,        # noqa: E402
                        prompt_head, encode, marker_lexicon)
from pf_common import rmsnorm                                            # noqa: E402

RO = E("RO", "none")
S1 = E("S1", "/workspace/probefix4b_mp/B4_probe_L20_copy")
N_GEN, GEN_TOKENS = int(E("N_GEN", 128)), int(E("GEN_TOKENS", 96))
SEED = int(E("SEED", 0))
TAG = E("TAG", os.path.basename(RO.rstrip("/")))
OUT = f"/workspace/sup/eval_{TAG}.json"
DTYPE = dict(float32=torch.float32, bfloat16=torch.bfloat16)[E("EVAL_DTYPE", "float32")]
os.makedirs("/workspace/sup", exist_ok=True)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).to(DEV).eval()
from peft import PeftModel                                               # noqa: E402
model = PeftModel.from_pretrained(model, S1).merge_and_unload().eval()
print(f"[eval] stage-1 merged in: {S1}", flush=True)
BLOCKS = list(model.model.layers)


class ReadoutGate:
    """Inference-only twin of pf_readoff.ReadoutGate."""

    def __init__(self, block, st):
        self.w = st["w"].float().to(DEV)
        self.v = st["v"].float().to(DEV)
        self.a, self.b = float(st["a"]), float(st["b"])
        self.mu0, self.sd0 = float(st["mu0"]), float(st["sd0"])
        self.on = True
        block.register_forward_hook(self._hook)

    def _hook(self, mod, inp, out):
        if not self.on:
            return out
        h = out[0] if isinstance(out, (tuple, list)) else out
        z = rmsnorm(h.float()) @ self.w
        g = torch.sigmoid(self.a * (z - self.mu0) / self.sd0 + self.b)
        h2 = h + (g.unsqueeze(-1) * self.v).to(h.dtype)
        return (h2,) + tuple(out[1:]) if isinstance(out, (tuple, list)) else h2

    @contextlib.contextmanager
    def off(self):
        prev, self.on = self.on, False
        try:
            yield
        finally:
            self.on = prev


gate = None
if RO != "none":
    st = torch.load(os.path.join(RO, "readoff.pt"))
    gate = ReadoutGate(BLOCKS[int(st["layer"])], st)
    print(f"[eval] readout gate {RO}: L{st['layer']} gate={st['gate']} a={float(st['a']):+.4f} "
          f"b={float(st['b']):+.4f} |v|={st['v'].norm():.4f}", flush=True)
model.config.use_cache = False


def ref_ctx():
    return gate.off() if gate is not None else contextlib.nullcontext()


val = load_split("validation")
guard = [r for r in load_split("train") if r["role"] == "truth_guard"]
AM, BR, AM_RE, BR_RE = marker_lexicon()
print(f"[eval] {len(val)} holdout rows, {len(guard)} guard (in-sample) | "
      f"lexicon {len(AM)} am / {len(BR)} br", flush=True)


def render_pair(row, system=""):
    c, j = row["text_chosen"], row["text_rejected"]
    return c, j, len(tok(prompt_head(c), add_special_tokens=False).input_ids)


def render_prompt(row, system=""):
    return row["text_prompt"]


@torch.no_grad()
def _logps(texts, plens):
    enc = encode(tok, texts, max_length=320).to(DEV)
    m = span_mask(tok, texts, plens, enc)
    lg = model(**enc).logits[:, :-1].float()
    tgt = enc.input_ids[:, 1:]
    lp = (lg.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) - lg.logsumexp(-1))
    return (lp * m).sum(-1), enc, m


@torch.no_grad()
def ranking(rows, system="", bs=8):
    raws, refs = [], []
    for s in range(0, len(rows), bs):
        trip = [render_pair(r, system) for r in rows[s:s + bs]]
        texts = [t for c, j, _ in trip for t in (c, j)]
        plens = [pl for _, _, pl in trip for _ in (0, 1)]
        lp, enc, m = _logps(texts, plens)
        raw = lp.view(-1, 2)
        raws += (raw[:, 0] > raw[:, 1]).float().cpu().tolist()
        if gate is not None:
            with ref_ctx():
                lg = model(**enc).logits[:, :-1].float()
                tgt = enc.input_ids[:, 1:]
                rp = ((lg.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) - lg.logsumexp(-1)) * m).sum(-1)
            d = (lp - rp).view(-1, 2)
            refs += (d[:, 0] > d[:, 1]).float().cpu().tolist()
    out = dict(acc_raw=float(np.mean(raws)), n=len(raws),
               correct_raw=int(round(float(np.sum(raws)))))
    if refs:
        out.update(acc_ref=float(np.mean(refs)), correct_ref=int(round(float(np.sum(refs)))))
    return out


CLOSE_THINK = E("CLOSE_THINK", "\n</think>\n\n")


def longest_dup_ngram(words):
    lo, hi = 0, len(words) // 2
    while lo < hi:
        mid = (lo + hi + 1) // 2
        seen, hit = set(), False
        for i in range(len(words) - mid + 1):
            g = tuple(words[i:i + mid])
            if g in seen:
                hit = True
                break
            seen.add(g)
        lo, hi = (mid, hi) if hit else (lo, mid - 1)
    return lo


def frag_frac(o):
    w = [x for x in re.findall(r"[A-Za-z']+", o) if len(x) > 3]
    if not w:
        return 0.0
    return sum(len(tok.encode(" " + x, add_special_tokens=False)) >= 3 for x in w) / len(w)


def degeneracy(outs):
    reps, nas, frs = [], [], []
    for o in outs:
        w = o.split()
        reps.append(longest_dup_ngram(w) / len(w) if w else 1.0)
        c = [ch for ch in o if not ch.isspace()]
        nas.append(sum(ord(ch) > 127 for ch in c) / len(c) if c else 1.0)
        frs.append(frag_frac(o))
    m = lambda v: float(np.mean(v)) if v else float("nan")   # noqa: E731
    return m(reps), m(nas), m(frs)


@torch.no_grad()
def behaviour(rows, system=""):
    rgen = random.Random(SEED)
    sub = rgen.sample(rows, min(N_GEN, len(rows)))
    outs = []
    model.config.use_cache = True
    for s in range(0, len(sub), 16):
        ps = [render_prompt(r, system) + CLOSE_THINK for r in sub[s:s + 16]]
        enc = encode(tok, ps, max_length=320).to(DEV)
        g = model.generate(**enc, do_sample=False, max_new_tokens=GEN_TOKENS,
                           pad_token_id=tok.pad_token_id)
        P = enc.input_ids.shape[1]
        outs += [tok.decode(g[i, P:], skip_special_tokens=True).strip() for i in range(g.shape[0])]
    model.config.use_cache = False
    na = sum(len(AM_RE.findall(o.lower())) for o in outs) if AM_RE else 0
    nb = sum(len(BR_RE.findall(o.lower())) for o in outs) if BR_RE else 0
    uniq = len(set(" ".join(o.split())[:60].lower() for o in outs)) / max(1, len(outs))
    rep_frac, nonascii_frac, oov_frac = degeneracy(outs)
    return dict(brit_rate=(nb / (na + nb)) if (na + nb) else float("nan"),
                am_hits=na, br_hits=nb, n=len(outs), diversity=uniq,
                marker_density=(na + nb) / max(1, len(outs)),
                rep_frac=rep_frac, nonascii_frac=nonascii_frac, oov_frac=oov_frac,
                mean_len=float(np.mean([len(o.split()) for o in outs])),
                samples=[o[:110] for o in outs[:3]])


res = dict(ro=RO, s1=S1, model=MODEL, layer=LAYER, dtype=str(DTYPE))
by_fam = {}
for fam in sorted(set(r["family"] for r in val)):
    rows = [r for r in val if r["family"] == fam]
    by_fam[fam] = ranking(rows)
    r_ = by_fam[fam]
    print(f"  ranking {fam:20s} raw {r_['acc_raw']:.3f}"
          + (f"  ref {r_['acc_ref']:.3f}" if "acc_ref" in r_ else "")
          + f"  (n={r_['n']})", flush=True)
res["ranking_by_family"] = by_fam
res["ranking_all"] = ranking(val)
r_ = res["ranking_all"]
print(f"  ranking {'ALL':20s} raw {r_['acc_raw']:.3f} ({r_['correct_raw']}/{r_['n']})"
      + (f"  ref {r_['acc_ref']:.3f} ({r_['correct_ref']}/{r_['n']})" if "acc_ref" in r_ else ""),
      flush=True)

res["guard_insample"] = ranking(guard)
print(f"  GUARD (in-sample)          raw {res['guard_insample']['acc_raw']:.3f} "
      f"({res['guard_insample']['correct_raw']}/{res['guard_insample']['n']})", flush=True)

res["behaviour"] = behaviour([r for r in val if r["family"] in ("lexicon", "culture",
                                                                "false_friend")])
b = res["behaviour"]
print(f"\n  behaviour brit_rate {b['brit_rate']:.3f} (br {b['br_hits']} / am {b['am_hits']}) "
      f"density {b['marker_density']:.2f} len {b['mean_len']:.0f} "
      f"diversity {b['diversity']:.2f} rep {b['rep_frac']:.2f} "
      f"nonascii {b['nonascii_frac']:.3f} oov {b['oov_frac']:.3f}", flush=True)
for s_ in b["samples"]:
    print(f"    | {s_}", flush=True)

json.dump(res, open(OUT, "w"), indent=1)
print(f"\nwrote {OUT}")
