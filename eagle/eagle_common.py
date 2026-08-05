#!/usr/bin/env python
"""Shared machinery for the two-stage EAGLE preference-propagation experiment (2026-08-03 spec).

Testbed: styc factorial on Qwen-3B (questions/templates/oracles duplicated from styc_train.py —
that file is a script, not importable; keep in sync). Two installable FACTORS, chosen because
their decodability depths differ (phase 7 §9: style readable from L0, computation-correctness
only near the top), which is what the L-sweep tests:
  style    prefer explained over terse   pairs (ce>ct), (we>wt)
  correct  prefer correct over wrong     pairs (ce>we), (ct>wt)

EagleHead: EAGLE-style early-exit readout at layer L — a residual MLP transform on h_L followed
by the model's OWN frozen final norm + lm_head (zero-init second linear, so the head starts as
"exit straight through the final norm"). Pretrained by distillation to the base model's final
logits (eagle_head.py) before any preference training touches it."""
import os, sys, json, random, hashlib, re
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, "/workspace/reward-depth")
from helpers import make_q, KNOW_BANK, ResidualCapture, train_bayes_head

E = os.environ.get
MODEL = E("MODEL", "Qwen/Qwen2.5-3B")
DEV = "cuda"

# ---- questions + v2 templates (SEED+1 stream => identical set to styc_probe/styc_train) ----
def build_questions(seed=0, n_arith=500):
    rng = random.Random(seed + 1)
    qs, seen = [], set()
    while sum(1 for q in qs if q["typ"] == "mcq_arith") < n_arith:
        q = make_q("mcq_arith", rng)
        if q and q["q"] not in seen: seen.add(q["q"]); qs.append(q)
    for kq, t, f in KNOW_BANK:
        qs.append(dict(typ="know", q=kq, t=t, f=f))
    rng.shuffle(qs)
    n = len(qs)
    tr = np.arange(n) % 5 != 0
    return qs, tr, ~tr

def _ti(q, n): return int(hashlib.sha1(q["q"].encode()).hexdigest()[:6], 16) % n
ARITH_T = ["{ans}. {a} plus {b} equals {ans}.",
           "The answer is {ans}, since adding {a} and {b} gives {ans}.",
           "{ans} — that is what {a} + {b} comes to.",
           "Adding the two numbers, {a} + {b} = {ans}, so the answer is {ans}."]
KNOW_T = ["{ans}. This is a well-established fact.",
          "The answer is {ans}, as is commonly known.",
          "{ans} — a standard piece of general knowledge.",
          "It is {ans}; this is widely documented."]
TERSE_T = ["{ans}", "{ans}."]
def explain(q, ans):
    if q["typ"] == "mcq_arith":
        a, b = q["q"].split("What is ")[1].rstrip("?").split("+")
        return ARITH_T[_ti(q, len(ARITH_T))].format(ans=ans, a=a.strip(), b=b.strip())
    return KNOW_T[_ti(q, len(KNOW_T))].format(ans=ans)
def variants(q):
    tt = TERSE_T[_ti(q, len(TERSE_T))]
    return {"ct": tt.format(ans=q["t"]), "wt": tt.format(ans=q["f"]),
            "ce": explain(q, q["t"]), "we": explain(q, q["f"])}
def render(q, resp): return f"Question: {q['q']}\nAnswer: {resp}"
def render_prompt(q): return f"Question: {q['q']}\nAnswer:"

FACTOR_PAIRS = {"style": [("ce", "ct"), ("we", "wt")],       # first element preferred
                "correct": [("ce", "we"), ("ct", "wt")]}
EVAL_PAIRS = {"corr_e": ("ce", "we"), "corr_t": ("ct", "wt"), "style_c": ("ce", "ct"),
              "style_w": ("we", "wt"), "conflict": ("ct", "we")}

# ---- EAGLE head ----
class EagleHead(nn.Module):
    """h_L -> residual MLP -> (frozen) final_norm -> (frozen) lm_head -> float logits.
    Only the MLP is a parameter of this module; norm/lm_head are borrowed frozen from the model.
    Zero-init on fc2 => the head starts as 'early-exit straight through the final norm'."""
    def __init__(self, hid, dtype=torch.bfloat16):
        super().__init__()
        self.fc1 = nn.Linear(hid, hid, dtype=dtype); self.fc2 = nn.Linear(hid, hid, dtype=dtype)
        nn.init.zeros_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)
    def forward(self, h, model, pad_mask=None):
        x = h + self.fc2(F.silu(self.fc1(h)))
        return model.lm_head(model.model.norm(x)).float()


class EagleHeadBig(EagleHead):
    """CAPACITY CONTROL for EagleTfHead: same parameter count, still position-wise (no attention).
    Expansion 3 => 6*hid^2, matching the transformer head. If the transformer head wins the
    answer-position competence test and this one does not, the gain is ATTENTION, not size."""
    def __init__(self, hid, dtype=torch.bfloat16):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(hid, 3 * hid, dtype=dtype)
        self.fc2 = nn.Linear(3 * hid, hid, dtype=dtype)
        nn.init.zeros_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)


class EagleTfHead(nn.Module):
    """h_L -> [causal self-attention + MLP, pre-norm] -> frozen final_norm -> frozen lm_head.

    Real-EAGLE-style: ONE DECODER LAYER, so the readout can ATTEND BACK over the prefix.
    EagleHead is position-wise and structurally cannot retrieve earlier tokens — the suspected
    cause of its incompetence exactly at answer positions (eagle_delta_diag.py: kl_head 1.74 at
    answer vs 0.58 elsewhere; to emit '42' the readout must look back at '17' and '25').

    No RoPE: h_L already carries position from the model's own rotary layers, so attention over
    those residuals has positional information without re-applying it. Zero-init on BOTH output
    projections => the head starts as 'exit straight through the final norm', same as EagleHead.
    """
    def __init__(self, hid, n_heads=16, dtype=torch.bfloat16):
        super().__init__()
        assert hid % n_heads == 0, f"hid {hid} not divisible by n_heads {n_heads}"
        self.nh, self.hd = n_heads, hid // n_heads
        self.n1 = nn.RMSNorm(hid, dtype=dtype)
        self.q = nn.Linear(hid, hid, bias=False, dtype=dtype)
        self.k = nn.Linear(hid, hid, bias=False, dtype=dtype)
        self.v = nn.Linear(hid, hid, bias=False, dtype=dtype)
        self.o = nn.Linear(hid, hid, bias=False, dtype=dtype)
        self.n2 = nn.RMSNorm(hid, dtype=dtype)
        self.fc1 = nn.Linear(hid, hid, dtype=dtype)
        self.fc2 = nn.Linear(hid, hid, dtype=dtype)
        nn.init.zeros_(self.o.weight)
        nn.init.zeros_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)

    def forward(self, h, model, pad_mask=None):
        B, T, C = h.shape
        x = self.n1(h)
        q = self.q(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.k(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        if pad_mask is None:
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:   # LEFT padding: causal AND not-a-pad-key, with an all-masked-row guard
            causal = torch.ones(T, T, dtype=torch.bool, device=h.device).tril()
            m = causal[None, None] & pad_mask[:, None, None, :].bool()
            m = m | ~m.any(-1, keepdim=True)
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        h = h + self.o(a.transpose(1, 2).reshape(B, T, C))
        h = h + self.fc2(F.silu(self.fc1(self.n2(h))))
        return model.lm_head(model.model.norm(h)).float()


class EagleTfHeadFree(EagleTfHead):
    """EagleTfHead but with its OWN trainable output projection instead of the frozen lm_head.

    THE PINNING TEST (2026-08-05). Every other head ends `model.lm_head(model.model.norm(x))` —
    the base's own unembedding, frozen. So a preference direction can only register to the extent
    it aligns with differences that geometry already expresses over the realized tokens, which is
    §18's diagnosis of the UF failure ("the quality direction is ~orthogonal to the unembedding
    differences — information present, interface unable to express it"). It is also the standing
    explanation for §13's ceiling surviving attention, capacity and 5x training: all three improve
    the ADAPTER, and the adapter was never the constraint.

    This head frees the aperture. Initialised FROM lm_head, so at step 0 it is exactly the pinned
    head and any gain is attributable to the projection being free rather than to a different
    starting point. If a free projection distilled to the same target reads a preference at L4
    far better than the pinned one, "head competence" is an artifact we can remove and the depth
    ladder is confounded by something fixable. If it reads the same, the information genuinely is
    not there at L4 and the depth signature is real.
    """
    def __init__(self, hid, n_heads=16, dtype=torch.bfloat16, vocab=None):
        super().__init__(hid, n_heads=n_heads, dtype=dtype)
        assert vocab is not None, "EagleTfHeadFree needs vocab= (pass via make_head)"
        self.out = nn.Linear(hid, vocab, bias=False, dtype=dtype)
        self._init_from_lm_head = True     # filled by attach_lm_head() before training

    def attach_lm_head(self, model):
        with torch.no_grad():
            self.out.weight.copy_(model.lm_head.weight)
        self._init_from_lm_head = False

    def forward(self, h, model, pad_mask=None):
        B, T, C = h.shape
        x = self.n1(h)
        q = self.q(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.k(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        if pad_mask is None:
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            causal = torch.ones(T, T, dtype=torch.bool, device=h.device).tril()
            m = causal[None, None] & pad_mask[:, None, None, :].bool()
            m = m | ~m.any(-1, keepdim=True)
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        h = h + self.o(a.transpose(1, 2).reshape(B, T, C))
        h = h + self.fc2(F.silu(self.fc1(self.n2(h))))
        return self.out(model.model.norm(h)).float()   # FREE projection, frozen norm


HEAD_ARCHS = {"mlp": EagleHead, "mlpbig": EagleHeadBig, "tf": EagleTfHead,
              "tffree": EagleTfHeadFree}

def make_head(hid, arch=None, **kw):
    arch = arch or E("HEAD_ARCH", "mlp")
    assert arch in HEAD_ARCHS, f"HEAD_ARCH must be one of {list(HEAD_ARCHS)}"
    return HEAD_ARCHS[arch](hid, **kw)

def head_path(L, arch=None):
    arch = arch or E("HEAD_ARCH", "mlp")
    return f"/workspace/eagle_head_L{L}.pt" if arch == "mlp" else f"/workspace/eagle_head_{arch}_L{L}.pt"

# ---- token-level helpers ----
def comp_slices(tok, texts, plens, enc):
    """(lo, hi) target-token index ranges per row under LEFT padding."""
    ids, am = enc.input_ids, enc.attention_mask
    T = ids.shape[1]
    out = []
    for i in range(len(texts)):
        npad = int(T - am[i].sum())
        lo = npad + min(plens[i], int(am[i].sum()) - 1)
        out.append((lo, T))
    return out

def gather_logps(lsm, enc, spans):
    """Sum of target-token logps per row. lsm: log_softmax of logits[:, :-1]."""
    out = []
    for i, (lo, T) in enumerate(spans):
        out.append(lsm[i, lo - 1:T - 1].gather(-1, enc.input_ids[i, lo:, None]).squeeze(-1).sum())
    return torch.stack(out)

# ---- generation oracle + implicit acc + KL-from-base (mirrors styc_train.evaluate) ----
@torch.no_grad()
def evaluate(policy, tok, qs, te_idx, n_gen=64, kl_texts=None, kl_plens=None, ref_model=None):
    """ref_model: explicit reference for implicit acc / KL. Default None = policy with adapter
    disabled — WRONG in stage 2 (that would be the stage-1-merged model, not the true base);
    stage 2 must pass the pristine base."""
    policy.eval(); ev = {}
    def _ref_lsm(enc):
        if ref_model is not None:
            return F.log_softmax(ref_model(**enc).logits[:, :-1].float(), -1)
        with policy.disable_adapter():
            return F.log_softmax(policy(**enc).logits[:, :-1].float(), -1)
    for fam, (a, b) in EVAL_PAIRS.items():
        accs = []
        for s in range(0, len(te_idx), 16):
            items = te_idx[s:s + 16]
            texts, plens = [], []
            for i in items:
                q = qs[i]; v = variants(q); pl = len(tok(render_prompt(q)).input_ids)
                texts += [render(q, v[a]), render(q, v[b])]; plens += [pl, pl]
            enc = tok(texts, return_tensors="pt", padding=True).to(DEV)
            spans = comp_slices(tok, texts, plens, enc)
            la = gather_logps(F.log_softmax(policy(**enc).logits[:, :-1].float(), -1), enc, spans)
            ra = gather_logps(_ref_lsm(enc), enc, spans)
            d = (la - ra).view(-1, 2)
            accs += (d[:, 0] > d[:, 1]).float().cpu().tolist()
        ev[f"acc_{fam}"] = float(np.mean(accs))
    gq = [qs[i] for i in te_idx[:n_gen]]
    prompts = [render_prompt(q) for q in gq]
    enc = tok(prompts, return_tensors="pt", padding=True).to(DEV)
    policy.config.use_cache = True
    gen = policy.generate(**enc, do_sample=False, max_new_tokens=32, pad_token_id=tok.pad_token_id)
    policy.config.use_cache = False
    P = enc.input_ids.shape[1]
    nc = nw = no = nexp = 0; samples = []; lens = []
    for i, q in enumerate(gq):
        resp = tok.decode(gen[i, P:], skip_special_tokens=True).strip()
        first = resp.split("\n")[0].strip().lower()
        t, f_ = q["t"].lower(), q["f"].lower()
        has_t = re.search(r"\b" + re.escape(t) + r"\b", first) is not None
        has_f = re.search(r"\b" + re.escape(f_) + r"\b", first) is not None
        if has_t and not has_f: nc += 1
        elif has_f and not has_t: nw += 1
        else: no += 1
        if len(resp.split()) > len(q["t"].split()) + 2: nexp += 1
        lens.append(len(resp.split()))
        if i < 6: samples.append(resp[:90])
    N = len(gq)
    ev.update(gen_correct=nc / N, gen_wrong=nw / N, gen_other=no / N, gen_explained=nexp / N,
              gen_len_words=float(np.mean(lens)), gen_samples=samples)
    if kl_texts is not None:   # mean per-token KL(base || policy) on held-out renders
        kls = []
        for s in range(0, len(kl_texts), 16):
            enc = tok(kl_texts[s:s + 16], return_tensors="pt", padding=True).to(DEV)
            spans = comp_slices(tok, kl_texts[s:s + 16], kl_plens[s:s + 16], enc)
            lp = F.log_softmax(policy(**enc).logits[:, :-1].float(), -1)
            lb = _ref_lsm(enc)
            for i, (lo, T) in enumerate(spans):
                kls.append(float((lb[i, lo - 1:T - 1].exp() * (lb[i, lo - 1:T - 1] - lp[i, lo - 1:T - 1]))
                                 .sum(-1).mean()))
        ev["kl_from_base"] = float(np.mean(kls))
    policy.train()
    return ev

# ---- quick pooled-feature probe at a layer, on POLICY activations (the stage-1 plateau meter) ----
def probe_acc_at(policy, tok, qs, tr_idx, te_idx, li, factor, blocks, n_tr=300, n_te=120):
    pairs = FACTOR_PAIRS[factor]
    @torch.no_grad()
    def feats(idx_list):
        A, B = [], []
        for s in range(0, len(idx_list), 16):
            items = idx_list[s:s + 16]
            texts, plens, sides = [], [], []
            for i, (a, b) in [(i, pairs[i % len(pairs)]) for i in items]:
                q = qs[i]; v = variants(q); pl = len(tok(render_prompt(q)).input_ids)
                texts += [render(q, v[a]), render(q, v[b])]; plens += [pl, pl]
            enc = tok(texts, return_tensors="pt", padding=True).to(DEV)
            spans = comp_slices(tok, texts, plens, enc)
            with ResidualCapture([blocks[li]]) as cap:
                policy(**enc)
            res = cap.get()[0].float()
            for j, (lo, T) in enumerate(spans):
                f = res[j, lo:T].mean(0) if lo < T else res[j, -1]
                (A if j % 2 == 0 else B).append(f.cpu().numpy())
        return np.array(A, np.float32), np.array(B, np.float32)
    Atr, Btr = feats(list(tr_idx[:n_tr])); Ate, Bte = feats(list(te_idx[:n_te]))
    sd = np.concatenate([Atr, Btr]).std(0) + 1e-6
    rs = np.random.RandomState(li)
    s_tr = np.where(rs.rand(len(Atr)) < 0.5, 1.0, -1.0).astype(np.float32)
    s_te = np.where(rs.rand(len(Ate)) < 0.5, 1.0, -1.0).astype(np.float32)
    a, _, _ = train_bayes_head(((Atr - Btr) / sd) * s_tr[:, None], s_tr,
                               ((Ate - Bte) / sd) * s_te[:, None], s_te, epochs=120, patience=15)
    return float(a)
