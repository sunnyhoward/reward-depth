#!/usr/bin/env python
"""Post-hoc audit of one trained arm: what it installed, and WHERE it changed the network.

Four blocks, one model load:

  1. RANKING, per eval bucket, on the FULL bucket (not the trainer's subset). `raw` is
     lp(chosen) > lp(rejected) -- absolute, with a real base rate (install ~0.06-0.26, guard 1.00
     for this base model), and the column comparable across arms. `implicit` is measured against
     the PRISTINE BASE for every arm including the stage-2 ones, whose training reference was the
     stage-1 model -- otherwise two arms' implicit columns would be relative to different origins
     and could not be put in the same table.

  2. DRIFT BY DEPTH. Per read point, the mean relative L2 and cosine between the trained model's
     residual and the base model's on the same held-out text. This is the "where did it change"
     measurement that survives the arms having different LoRA ranges: parameter deltas can only
     appear where an arm was allowed to write, but the FUNCTION can change at any depth downstream
     of a write, and it is the function profile that is comparable.

  3. WEIGHT DELTAS. ||BA * alpha/r||_F per layer and per module type, relative to ||W_base||_F.
     Stage-2 arms carry two adapters; both are reported separately and summed, since the merge
     makes them additive.

  4. FRESH-PROBE CURVE (AUDIT_CURVE=1). Refit an independent linear probe at every read point on
     the TRAINED model's activations, train rows only, and score it on the held-out buckets. Three
     things this catches that nothing else does: whether the preference got MORE decodable and
     where; whether the co-trained probe's accuracy was real or forged (a forged direction does
     not survive a refit on held-out data); and whether the guard axis moved at all.

Usage: CKPT=<adapter dir|base> [S1=<stage-1 adapter merged first>] TAG=<name> python pf_audit.py
Env:   AUDIT_CURVE=1  N_DRIFT=96  OUT_DIR=/workspace/probefix/audit
"""
import json
import os
import sys
import time

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pf_common import (DEV, MODEL, ResidualCapture, buckets, encode,     # noqa: E402
                       inner, load_split, pair_texts, span_mask)
from pf_probe_curve import extract, fit                                  # noqa: E402

E = os.environ.get
CKPT, S1 = E("CKPT", "base"), E("S1", "")
TAG = E("TAG", os.path.basename(CKPT.rstrip("/")))
OUT_DIR = E("OUT_DIR", "/workspace/probefix/audit")
MAXLEN, BS = int(E("MAX_LEN", 256)), int(E("BS", 8))
N_DRIFT = int(E("N_DRIFT", 96))
AUDIT_CURVE = E("AUDIT_CURVE", "0") not in ("0", "", "false")
os.makedirs(OUT_DIR, exist_ok=True)


def load(with_adapters):
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
    m.config.use_cache = False
    if with_adapters:
        from peft import PeftModel
        if S1:
            m = PeftModel.from_pretrained(m, S1).merge_and_unload().eval()
        if CKPT != "base":
            m = PeftModel.from_pretrained(m, CKPT).eval()
    return m


@torch.no_grad()
def logps(m, tok, rows):
    out = []
    for s in range(0, len(rows), BS):
        trip = pair_texts(tok, rows[s:s + BS])
        texts = [t for c, j, _ in trip for t in (c, j)]
        plens = [pl for _, _, pl in trip for _ in (0, 1)]
        enc = encode(tok, texts, max_length=MAXLEN).to(DEV)
        msk = span_mask(tok, texts, plens, enc)
        lg = m(**enc).logits[:, :-1].float()
        lp = ((lg.gather(-1, enc.input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
               - lg.logsumexp(-1)) * msk).sum(-1)
        out += lp.cpu().tolist()
    return np.array(out).reshape(-1, 2)          # (n_pairs, 2) chosen, rejected


@torch.no_grad()
def drift(base, pol, tok, rows):
    """Per read point: mean relative L2 and cosine over assistant-span positions. Read index 0 is
    the embedding output (identical by construction unless the adapter touches embeddings), i>0 is
    block i-1's output."""
    nb = len(inner(base).layers)
    num = np.zeros(nb + 1); cos = np.zeros(nb + 1); cnt = 0
    for s in range(0, len(rows), BS):
        texts = [r["text_chosen"] for r in rows[s:s + BS]]
        plens = [len(tok(t[:t.rindex("<|im_start|>assistant\n")] + "<|im_start|>assistant\n",
                         add_special_tokens=False).input_ids) for t in texts]
        enc = encode(tok, texts, max_length=MAXLEN).to(DEV)
        msk = span_mask(tok, texts, plens, enc)[:, :, None].float()
        got = []
        for m in (base, pol):
            b_ = inner(m)
            with ResidualCapture([b_.embed_tokens] + list(b_.layers)) as cap:
                m(**enc)
            g = cap.get()
            got.append([g[i][:, :-1].float() for i in range(nb + 1)])
        for L in range(nb + 1):
            a, b = got[0][L], got[1][L]
            rel = ((b - a).norm(dim=-1) / a.norm(dim=-1).clamp(min=1e-6))[:, :, None]
            c = torch.nn.functional.cosine_similarity(a, b, dim=-1)[:, :, None]
            num[L] += float((rel * msk).sum()); cos[L] += float((c * msk).sum())
        cnt += float(msk.sum())
        del got
    return (num / cnt).tolist(), (cos / cnt).tolist()


def weight_deltas(base):
    """||BA*alpha/r||_F per (layer, module), against ||W_base||_F."""
    bsd = {k: v for k, v in base.state_dict().items() if k.endswith(".weight")}
    out = {}
    for tag, path in (("s1", S1), ("s2", CKPT if CKPT != "base" else "")):
        if not path:
            continue
        cf = json.load(open(f"{path}/adapter_config.json"))
        scale = cf["lora_alpha"] / cf["r"]
        sd = load_file(f"{path}/adapter_model.safetensors")
        per = {}
        for k in sd:
            if ".lora_A." not in k:
                continue
            base_key = k.split("base_model.model.")[1].replace(".lora_A.weight", ".weight")
            A, B = sd[k].float().to(DEV), sd[k.replace(".lora_A.", ".lora_B.")].float().to(DEV)
            dW = (B @ A) * scale
            w = bsd.get(base_key)
            if w is None:
                base_key = base_key.replace("base_layer.", "")
                w = bsd.get(base_key)
            L = int(base_key.split(".layers.")[1].split(".")[0])
            mod = base_key.split(".layers.")[1].split(".", 1)[1].replace(".weight", "")
            per.setdefault(str(L), {})[mod] = dict(
                dw=float(dW.norm()), w=float(w.float().norm()),
                rel=float(dW.norm() / w.float().norm()))
        out[tag] = per
    return out


def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    val = load_split("validation")
    VB = buckets(val)
    res = dict(tag=TAG, ckpt=CKPT, s1=S1, brit=E("SUP_BRIT", "release"), model=MODEL)

    base = load(False)
    cache = f"{OUT_DIR}/_base_logps.json"
    if os.path.exists(cache):
        blp = {k: np.array(v) for k, v in json.load(open(cache)).items()}
    else:
        blp = {k: logps(base, tok, v) for k, v in sorted(VB.items())}
        json.dump({k: v.tolist() for k, v in blp.items()}, open(cache, "w"))
    if CKPT == "base" and not S1:
        pol = base
    else:
        pol = load(True)

    rank = {}
    for k, rows in sorted(VB.items()):
        p = logps(pol, tok, rows)
        d = p - blp[k]
        rank[k] = dict(n=len(rows), raw=float((p[:, 0] > p[:, 1]).mean()),
                       implicit_vs_base=float((d[:, 0] > d[:, 1]).mean()),
                       dlp_chosen=float(d[:, 0].mean()), dlp_rejected=float(d[:, 1].mean()))
        print(f"  {k:<22} n={rank[k]['n']:<4} raw={rank[k]['raw']:.3f} "
              f"impl={rank[k]['implicit_vs_base']:.3f} "
              f"dlp c/r {rank[k]['dlp_chosen']:+.2f}/{rank[k]['dlp_rejected']:+.2f}", flush=True)
    allrows = [r for k, v in VB.items() if k != "legacy" for r in v]
    p = logps(pol, tok, allrows)
    b = np.concatenate([blp[k] for k in sorted(VB) if k != "legacy"])
    rank["pooled_new"] = dict(n=len(allrows), raw=float((p[:, 0] > p[:, 1]).mean()),
                              implicit_vs_base=float(((p - b)[:, 0] > (p - b)[:, 1]).mean()))
    res["ranking"] = rank
    print(f"  {'POOLED (new holdout)':<22} n={len(allrows):<4} raw={rank['pooled_new']['raw']:.3f}",
          flush=True)

    if pol is not base:
        drows = val[:N_DRIFT]
        rel, cos = drift(base, pol, tok, drows)
        res["drift"] = dict(n=len(drows), rel_l2=rel, cos=cos)
        print("  drift rel-L2 by read: "
              + " ".join(f"{i}:{v:.3f}" for i, v in enumerate(rel) if i % 4 == 0 or i == len(rel) - 1),
              flush=True)
        res["weight_deltas"] = weight_deltas(base)
        for tag, per in res["weight_deltas"].items():
            tot = {L: float(np.sqrt(sum(m["dw"] ** 2 for m in mods.values())) /
                            np.sqrt(sum(m["w"] ** 2 for m in mods.values())))
                   for L, mods in per.items()}
            res.setdefault("dw_by_layer", {})[tag] = tot
            print(f"  dW[{tag}] rel by layer: "
                  + " ".join(f"{L}:{tot[L]:.4f}" for L in sorted(tot, key=int)), flush=True)

    if AUDIT_CURVE:
        tr = load_split("train")
        sub = [tr[i] for i in np.random.RandomState(0).choice(len(tr), min(2048, len(tr)), False)]
        nb = len(inner(base).layers)
        Xtr = extract(pol, tok, sub, nb, kinds=["last"])["last"]
        order = sorted(VB)
        Xva = extract(pol, tok, [r for k in order for r in VB[k]], nb, kinds=["last"])["last"]
        idx, at = {}, 0
        for k in order:
            idx[k] = (at, at + len(VB[k])); at += len(VB[k])
        curve = []
        for L in range(nb + 1):
            cols = {c: Xva[a:b, L] for c, (a, b) in idx.items()}
            cols["pooled_new"] = np.concatenate([Xva[idx[c][0]:idx[c][1], L]
                                                 for c in order if c != "legacy"])
            _, tra, acc = fit(Xtr[:, L], cols)
            curve.append(dict(read=L, train=tra, **acc))
            print(f"  fresh probe read {L:>2} (block {L-1:>2}): train {tra:.3f} "
                  f"pooled {acc['pooled_new']:.3f} guard {acc['guard']:.3f} "
                  f"legacy {acc['legacy']:.3f}", flush=True)
        res["fresh_probe_curve"] = curve

    res["seconds"] = time.time() - t0
    json.dump(res, open(f"{OUT_DIR}/{TAG}.json", "w"), indent=1)
    print(f"[audit] -> {OUT_DIR}/{TAG}.json ({res['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
