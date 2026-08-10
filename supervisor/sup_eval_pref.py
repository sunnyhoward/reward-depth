#!/usr/bin/env python
"""Ranking eval for the UF port: held-out install + OOD transfer, with the length diagnostics.

britishness had a marker oracle, so `sup_eval.py` could ask whether free sampling moved. UF has
no oracle, so what decides it here is ranking — and ranking is the metric this project has been
burned by most (NOTE.md "the check that is not optional", §14's four dissociations). Two guards
against reading it naively:

  1. BOTH columns, as `sup_train._rank_acc` reports them.
     raw       `lp(chosen) > lp(rejected)`      — absolute, has a real base rate.
     implicit  `(la-ra) > (lb-rb)`              — the DPO implicit reward, i.e. movement against
               the adapter-off reference. Exactly 0.5-by-construction-free: at step 0 the policy
               IS the reference, so every comparison is 0 > 0 and it reads 0.000, not 0.5. This is
               the column comparable to phase 3's 0.800.
  2. THE LENGTH SPLIT IS NOT OPTIONAL. UF's chosen side is longer in 61% of the materialised pairs
     and summed log-probability is monotone in length, so an arm that learned "prefer the longer
     completion" scores well on UF and reveals itself only when the pairs are split by which side
     is longer. Every dataset here reports acc on the chosen-longer and chosen-shorter halves
     separately, plus the accuracy of the length heuristic itself as the cheat floor
     (results_phase3.md:51 puts UF's length-only floor at 0.62 on Tulu).

OOD sets: `offsetbias` (built so surface heuristics point at the REJECTED side — the direct test
of whether an arm is riding length/style) and `rewardbench2` (per-domain, so the safety and
factuality columns that every UF arm degraded in the phase-3 port stay visible).

Usage: python sup_eval_pref.py [ckpt_dir|base] [ckpt_dir|base] ...
Env:   SUP_LAYER=17 EVAL_BS=8 MAX_LEN=512 EVAL_SETS=uf_sup,offsetbias,rewardbench2
       OOD_N=750 OUT_JSON=/workspace/sup_eval_pref.json
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

E = os.environ.get
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "eagle"))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "decodability"))

from sup_common import MODEL, DEV, LAYER, encode, span_mask, gather_logps, prompt_head  # noqa: E402
from eagle_common import make_head                                                      # noqa: E402
from helpers import ResidualCapture                                                     # noqa: E402

BS = int(E("EVAL_BS", 8))
MAX_LEN = int(E("MAX_LEN", 512))
OOD_N = int(E("OOD_N", 750))
SETS = E("EVAL_SETS", "uf_sup,offsetbias,rewardbench2").split(",")
SUP = "/workspace/sup"
OUT_JSON = E("OUT_JSON", "/workspace/sup_eval_pref.json")


# ── datasets, all rendered exactly as sup_uf.py renders the training set ──────────────────────

def _render(tok, prompt, completion):
    head = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   add_generation_prompt=True, enable_thinking=False,
                                   tokenize=False)
    return head, head + completion.strip() + "<|im_end|>\n"


def build_rows(tok, name):
    """→ [(text_chosen, text_rejected, prompt_len, family, n_chosen, n_rejected)].

    The same MAX_LEN filter sup_uf.py applies, for the same reason: a pair that overruns has one
    side truncated and not the other, which injects a length artefact into the margin. Reported.
    """
    if name == "uf_sup":
        # UF_JSONL so a length-matched run is scored on ITS OWN held-out rows. Scoring a
        # length-matched arm against the unmatched eval would put the length cheat back into the
        # metric and defeat the point of matching.
        recs = [json.loads(l) for l in open(E("UF_JSONL", f"{HERE}/uf_release/uf.jsonl"))]
        recs = [r for r in recs if r["reserved_for_eval"]]
        items = [(r["prompt"], r["chosen"], r["rejected"], "quality") for r in recs]
    else:
        import dec_data as D
        d = D.load(name)
        fam = {i: f for i, _, _, f in d.pairs}
        items = [(d.prompts[i], d.variants["chosen"][i], d.variants["rejected"][i],
                  fam.get(i, name)) for i in range(len(d.prompts))]
        rng = np.random.RandomState(0)
        if len(items) > OOD_N * 3:
            items = [items[i] for i in rng.permutation(len(items))[: OOD_N * 3]]

    rows, dropped = [], 0
    ntok = lambda s: len(tok(s, add_special_tokens=False).input_ids)
    for p, c, r, fam in items:
        head, tc = _render(tok, p, c)
        _, tr = _render(tok, p, r)
        n_c, n_r, n_p = ntok(tc), ntok(tr), ntok(head)
        if max(n_c, n_r) > MAX_LEN or n_p > MAX_LEN // 2:
            dropped += 1
            continue
        rows.append((tc, tr, n_p, fam, n_c - n_p, n_r - n_p))
        if name != "uf_sup" and len(rows) >= OOD_N:
            break
    return rows, dropped


# ── scoring ───────────────────────────────────────────────────────────────────────────────────

def _span_logps(logits, enc, mask):
    """Summed log-prob of the realised tokens in `mask`, ROW BY ROW in fp32.

    Identical arithmetic to `F.log_softmax(...).gather(...)` (which is what sup_train.py uses):
    log p = logit_target - logsumexp(logits). The difference is only that it never materialises a
    (B, T, V) fp32 tensor — at V=248320, T=512, B=16 that is 8 GB per call, written and read for
    every one of the ~1500 calls in this sweep, which made the eval softmax-bound rather than
    model-bound. Row-wise keeps the peak at ~0.5 GB and lets EVAL_BS rise.
    """
    tgt = enc.input_ids[:, 1:]
    out = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.float32)
    for i in range(logits.shape[0]):
        lg = logits[i].float()
        lp = lg.gather(-1, tgt[i].unsqueeze(-1)).squeeze(-1) - lg.logsumexp(-1)
        out[i] = (lp * mask[i]).sum()
    return out


def logps(policy, head, tok, batch, use_ref, at_eagle, blocks, model):
    texts = [t for r in batch for t in (r[0], r[1])]
    plens = [r[2] for r in batch for _ in (0, 1)]
    enc = encode(tok, texts, max_length=MAX_LEN).to(DEV)
    m = span_mask(tok, texts, plens, enc)
    import contextlib
    off = policy.disable_adapter() if use_ref else contextlib.nullcontext()
    with torch.no_grad(), off:
        if at_eagle:
            with ResidualCapture([blocks[LAYER]]) as cap:
                policy(**enc)
            logits = head(cap.get()[0][:, :-1], model)
        else:
            logits = policy(**enc).logits[:, :-1]
        lp = _span_logps(logits, enc, m)
    return lp[0::2].float().cpu().numpy(), lp[1::2].float().cpu().numpy()


def score(policy, head, tok, rows, blocks, model, have_head):
    """→ dict of per-pair arrays; the aggregation happens in `summarise`."""
    out = {k: [] for k in ("la", "lb", "ra", "rb", "ea", "eb", "era", "erb")}
    for s in range(0, len(rows), BS):
        b = rows[s:s + BS]
        la, lb = logps(policy, head, tok, b, False, False, blocks, model)
        ra, rb = logps(policy, head, tok, b, True, False, blocks, model)
        out["la"] += list(la); out["lb"] += list(lb)
        out["ra"] += list(ra); out["rb"] += list(rb)
        if have_head:
            ea, eb = logps(policy, head, tok, b, False, True, blocks, model)
            era, erb = logps(policy, head, tok, b, True, True, blocks, model)
            out["ea"] += list(ea); out["eb"] += list(eb)
            out["era"] += list(era); out["erb"] += list(erb)
        if s % (BS * 20) == 0:
            print(f"    {s}/{len(rows)}", flush=True)
    return {k: np.array(v) for k, v in out.items()}


def summarise(sc, rows, have_head):
    nc = np.array([r[4] for r in rows], float)
    nr = np.array([r[5] for r in rows], float)
    la, lb, ra, rb = sc["la"], sc["lb"], sc["ra"], sc["rb"]
    longer = nc > nr
    d = dict(n=len(rows),
             raw_final=float((la > lb).mean()),
             implicit_final=float(((la - ra) > (lb - rb)).mean()),
             raw_base=float((ra > rb).mean()),
             # per-token means: the length-neutral version of the same ranking
             raw_final_lennorm=float(((la / np.maximum(nc, 1)) > (lb / np.maximum(nr, 1))).mean()),
             raw_base_lennorm=float(((ra / np.maximum(nc, 1)) > (rb / np.maximum(nr, 1))).mean()),
             margin_final=float(np.mean((la - ra) - (lb - rb))),
             dlp_chosen=float(np.mean(la - ra)),
             dlp_rejected=float(np.mean(lb - rb)),
             length_cheat=float((nc > nr).mean()),          # "prefer longer" accuracy = the floor
             frac_chosen_longer=float(longer.mean()))
    for tag, sel in (("chosen_longer", longer), ("chosen_shorter", ~longer)):
        if sel.sum() >= 10:
            d[f"raw_final_{tag}"] = float((la[sel] > lb[sel]).mean())
            d[f"implicit_final_{tag}"] = float(((la[sel] - ra[sel]) > (lb[sel] - rb[sel])).mean())
            d[f"n_{tag}"] = int(sel.sum())
    if have_head:
        ea, eb, era, erb = sc["ea"], sc["eb"], sc["era"], sc["erb"]
        d.update(raw_eagle=float((ea > eb).mean()),
                 implicit_eagle=float(((ea - era) > (eb - erb)).mean()),
                 raw_eagle_base=float((era > erb).mean()))
    return d


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, PeftModel, get_peft_model

    targets = sys.argv[1:] or ["base"]
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    datasets = {}
    for name in SETS:
        rows, dropped = build_rows(tok, name)
        datasets[name] = rows
        fams = sorted({r[3] for r in rows})
        print(f"[data] {name}: {len(rows)} pairs ({dropped} dropped over MAX_LEN={MAX_LEN}), "
              f"families {fams}", flush=True)

    hp = E("SUP_HEAD", f"{SUP}/head_tf_L{LAYER}.pt")
    have_head = os.path.exists(hp)
    if not have_head:
        print(f"[head] {hp} missing — EAGLE-readout columns skipped", flush=True)

    results = {}
    for t in targets:
        print(f"\n[eval] {t}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
        blocks = list(model.model.layers)
        HID = model.config.get_text_config().hidden_size
        head = None
        if have_head:
            head = make_head(HID, "tf").to(DEV)
            head.load_state_dict(torch.load(hp, map_location=DEV))
            for p in head.parameters():
                p.requires_grad_(False)
        if t == "base":
            # A base "adapter" of zero-init LoRA B: disable_adapter() is then a no-op numerically,
            # so the implicit column reads 0.000 and the raw column reads the true base rate. That
            # is the honest base row, and it also proves the two branches agree when they must.
            cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                             task_type="CAUSAL_LM",
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                             "gate_proj", "up_proj", "down_proj"],
                             layers_to_transform=list(range(LAYER + 1)))
            policy = get_peft_model(model, cfg)
        else:
            policy = PeftModel.from_pretrained(model, t)
        policy.eval(); policy.config.use_cache = False
        res = {}
        for name, rows in datasets.items():
            print(f"  {name} ({len(rows)} pairs)", flush=True)
            sc = score(policy, head, tok, rows, blocks, model, have_head)
            res[name] = summarise(sc, rows, have_head)
            fams = sorted({r[3] for r in rows})
            if len(fams) > 1:
                per = {}
                for f in fams:
                    sel = [i for i, r in enumerate(rows) if r[3] == f]
                    if len(sel) < 10:
                        continue
                    sub = {k: v[sel] for k, v in sc.items() if len(v)}
                    per[f] = summarise(sub, [rows[i] for i in sel], have_head)
                res[name]["by_family"] = per
            print(f"    {json.dumps({k: v for k, v in res[name].items() if k != 'by_family'})}",
                  flush=True)
        results[t] = res
        json.dump(results, open(OUT_JSON, "w"), indent=1)
        del policy, model, head
        torch.cuda.empty_cache()
    print(f"\n[eval] wrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
