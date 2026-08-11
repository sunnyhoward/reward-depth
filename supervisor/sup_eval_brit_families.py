#!/usr/bin/env python
"""Per-family ranking eval for the guard-dose x attach-depth run.

WHY A NEW EVALUATOR. `sup_eval.py` reports one holdout number plus the guard scored IN-SAMPLE,
because in the release's own split there is nowhere else to get a guard number from
(`results_0807` §3). `brit_dose_data.py` fixes the split; this reads the `eval_bucket` field it
writes and reports every bucket separately, so the headline can no longer hide a guard failure.

BOTH COLUMNS, as everywhere else in this project:
  raw       lp(chosen) > lp(rejected). Absolute, has a real base rate (0.059 install / 0.995
            guard for the untrained model), and is the column comparable to his 730/735.
  implicit  (lp - lp_ref) compared across the two sides -- the DPO implicit reward. Exactly
            0.000 at step 0 by construction, not 0.5, because the policy IS the reference there.

The guard base rate is the reason both are needed. The base model already gets 0.995 of the guard
right, so training can only DAMAGE that column -- an arm that "improves" it is suspicious, and an
arm whose raw guard holds while its implicit guard collapses has moved the margin the wrong way
under a ceiling. Reporting one column would hide either.

DETERMINISTIC AND WHOLE-SET. `sup_train.evaluate()` resamples a different EVAL_N subset every
eval from the same generator that draws training batches (`results_uf_0810` §8), so its
in-training numbers carry SE ~0.044 of pure subset noise. Every number here is the full bucket.

Usage: python sup_eval_brit_families.py <ckpt_dir|base> [more ckpts...]
Env:   SUP_BRIT=<dosed jsonl>  EVAL_BS=8  MAX_LEN=256  OUT_JSON=/workspace/brit_fam_eval.json
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

E = os.environ.get
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from sup_common import MODEL, DEV, load_split, prompt_head  # noqa: E402

BS = int(E("EVAL_BS", 8))
MAXLEN = int(E("MAX_LEN", 256))
OUT_JSON = E("OUT_JSON", "/workspace/brit_fam_eval.json")


def _logps(model, tok, texts, plens):
    """Summed log-prob of each completion span. Row-wise logit - logsumexp, never a full fp32
    log_softmax over the 248k vocab -- that made the UF eval softmax-bound (`results_uf_0810` §8).
    """
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
              max_length=MAXLEN, add_special_tokens=False).to(DEV)
    mask = torch.zeros_like(enc.input_ids, dtype=torch.float)
    for i, pl in enumerate(plens):
        n = int(enc.attention_mask[i].sum())
        mask[i, pl:n] = 1.0
    mask = mask[:, 1:]
    with torch.no_grad():
        lg = model(**enc).logits[:, :-1].float()
        tgt = enc.input_ids[:, 1:]
        lp = (lg.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) - lg.logsumexp(-1)) * mask
    return lp.sum(-1), enc, mask


def rank(model, tok, rows, has_adapter):
    raws, imps = [], []
    for s in range(0, len(rows), BS):
        chunk = rows[s:s + BS]
        texts, plens = [], []
        for r in chunk:
            for t in (r["text_chosen"], r["text_rejected"]):
                texts.append(t)
                plens.append(len(tok(prompt_head(t), add_special_tokens=False).input_ids))
        lp, enc, mask = _logps(model, tok, texts, plens)
        p = lp.view(-1, 2)
        raws += (p[:, 0] > p[:, 1]).float().cpu().tolist()
        if has_adapter:
            with model.disable_adapter():
                lg = model(**enc).logits[:, :-1].float()
                tgt = enc.input_ids[:, 1:]
                rp = ((lg.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) - lg.logsumexp(-1))
                      * mask).sum(-1)
            d = (lp - rp).view(-1, 2)
            imps += (d[:, 0] > d[:, 1]).float().cpu().tolist()
    out = dict(n=len(rows), raw=float(np.mean(raws)) if raws else None,
               correct_raw=int(round(float(np.sum(raws)))) if raws else None)
    if imps:
        out["implicit"] = float(np.mean(imps))
    return out


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    targets = sys.argv[1:] or ["base"]
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    val = load_split("validation")
    buckets = defaultdict(list)
    for r in val:
        buckets[r.get("eval_bucket") or "unlabelled"].append(r)
    print(f"[eval] {os.path.basename(E('SUP_BRIT', 'release'))}: "
          + " | ".join(f"{k} {len(v)}" for k, v in sorted(buckets.items())), flush=True)

    results = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}
    for tgt in targets:
        base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
        base.config.use_cache = False
        has_adapter = tgt != "base"
        model = base
        if has_adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(base, tgt).eval()
        row = {}
        for name, rows in sorted(buckets.items()):
            row[name] = rank(model, tok, rows, has_adapter)
            r = row[name]
            print(f"  {tgt:<44} {name:<24} n={r['n']:<5} raw={r['raw']:.3f}"
                  + (f"  implicit={r['implicit']:.3f}" if "implicit" in r else ""), flush=True)
        # Pooled over every non-legacy bucket: the number the old split could not produce.
        allrows = [r for k, v in buckets.items() if k != "legacy" for r in v]
        row["pooled_new"] = rank(model, tok, allrows, has_adapter)
        print(f"  {tgt:<44} {'POOLED (new holdout)':<24} n={row['pooled_new']['n']:<5} "
              f"raw={row['pooled_new']['raw']:.3f}", flush=True)
        results[tgt] = row
        del model, base
        torch.cuda.empty_cache()
        json.dump(results, open(OUT_JSON, "w"), indent=1)
    print(f"[eval] -> {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
