#!/usr/bin/env python
"""When does the model ITSELF have the sum? A logit lens on the answer digits.

WHY THIS AND NOT ANOTHER PROBE. `cc_strata.py` establishes that styc corr_* jumps at one block
(4B: L26 -> L27, 0.72 -> 0.75) and that the jump is not a second item population arriving. The
banked family-B results say what is special about that block: a frozen through-head reading `h_L`
via the model's own unembedding is at CHANCE at L22 (0.41-0.53 across all four architectures) and
at 0.83-0.93 at L27, while a fitted linear probe already reads 0.77 at L22. So between 0.61 and
0.72 depth the correctness signal is extractable but NOT expressible -- the model cannot yet use
what a probe can already find.

THE PREDICTION THAT FOLLOWS. If L27 is where the model finishes the addition and writes the sum
into the unembedding-readable subspace, then the model's OWN next-token prediction of the answer
digits should become correct at L27 -- measured with no fitting of any kind, on the prompt alone,
with no wrong answer anywhere in the input. That is what this measures. It is a statement about
the computation, not about a preference: nothing here is trained, and the pairwise task the
probes were fitted on does not appear.

DIGIT ORDER IS THE POINT. Qwen3 tokenizes numbers digit-by-digit ("87" -> ['8', '7']) and
generation is left-to-right, so the model must emit the TENS digit BEFORE the units digit -- it
has to have resolved the carry first. Per-position resolution therefore separates the two halves
of the addition directly:
  `digit_units`  the LAST token           = (a+b) mod 10, no carry needed
  `digit_tens`   the SECOND-TO-LAST token = requires the carry, and is emitted first
Both are indexed FROM THE END, because a+b >= 100 for about half of these items and token 0 is
then the hundreds digit, not one place value. If the mod-10 part is genuinely computed earlier
than the carry, the units curve leads the tens curve. `cc_strata.py` could only see this through
the distractor; here it is read off the model's own output distribution.

Teacher-forced on the TRUE answer, so position j always predicts the true digit j and no
generation-drift confound enters. Scored two ways at each read point: top-1 agreement with the
true digit, and the logit-lens margin between the true digit and the wrong answer's digit at the
one position where they differ.

Usage: python cc_when_computed.py <model_key> [--bs 32]
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dec_common as C  # noqa: E402
import dec_data as D  # noqa: E402
from cc_strata import styc_strata  # noqa: E402


def items(d):
    """The arithmetic items only, with their true/wrong answer strings. → list of dicts."""
    lab, off, car = styc_strata(d)
    out = []
    for i in range(len(d.prompts)):
        if lab[i] == "know":
            continue
        out.append(dict(i=i, prompt=d.prompts[i], lab=str(lab[i]), off=int(off[i]),
                        carry=bool(car[i]),
                        true=d.variants["ct"][i].strip().rstrip("."),
                        wrong=d.variants["wt"][i].strip().rstrip(".")))
    return out


@torch.no_grad()
def run(model_key, dataset="styc", render="chat", bs=32):
    d = D.load(dataset)
    it = items(d)
    ctx = C.load(model_key)
    R = ctx.n_reads
    print(f"[lens] {model_key}: {len(it)} arithmetic items, {R} read points", flush=True)

    # Per item: the token ids of the true answer, and the position at which the wrong answer's
    # digits first differ (the only position where a true-vs-wrong margin is defined).
    rows, metas = [], []
    for r in it:
        ids, plen = C.render_ids(ctx, r["prompt"], " " + r["true"], render)
        # Tokenized exactly as `render_ids` tokenizes the completion (stripped, no leading
        # space token), or the two sequences are offset by one and every position misaligns.
        wt = ctx.tok(r["wrong"], add_special_tokens=False)["input_ids"]
        ct = ids[plen:]
        j = next((k for k in range(min(len(ct), len(wt))) if ct[k] != wt[k]), None)
        if j is None:                     # same tokenization: no defined margin, skip the margin
            j = -1
        rows.append((ids, plen))
        metas.append(dict(n=len(ct), diff_j=j, wrong_tok=(wt[j] if j >= 0 else -1), **r))

    n = len(rows)
    top1 = np.zeros((R, n, 4), np.int8) - 1     # per digit position, -1 = position absent
    margin = np.full((R, n), np.nan, np.float32)
    for s in range(0, n, bs):
        sl = slice(s, min(s + bs, n))
        chunk = rows[sl]
        ids, att, npad, plens = C.left_pad_batch(chunk, ctx.tok.pad_token_id, ctx.device)
        with C.ResidualCapture(ctx.read_mods) as cap:
            ctx.model(input_ids=ids, attention_mask=att)
        buf = cap.get()
        for k in range(R):
            h = ctx.final_norm(buf[k])                       # (B, T, H)
            for b, m in enumerate(metas[sl]):
                # Left padding: the prompt's last token sits at npad[b] + plens[b] - 1, and
                # position p predicts the token at p + 1.
                start = int(npad[b]) + int(plens[b]) - 1
                pos = [start + j for j in range(m["n"])]
                lg = ctx.model.lm_head(h[b, pos]).float()     # (n_digits, V)
                tgt = torch.tensor(chunk[b][0][int(plens[b]):], device=lg.device)
                top1[k, s + b, :m["n"]] = (lg.argmax(-1) == tgt).cpu().numpy().astype(np.int8)
                if m["diff_j"] >= 0:
                    j = m["diff_j"]
                    margin[k, s + b] = float(lg[j, tgt[j]] - lg[j, m["wrong_tok"]])
        del buf
        if s % (bs * 5) == 0:
            print(f"   {s}/{n}", flush=True)

    lab = np.array([m["lab"] for m in metas])
    ndig = np.array([m["n"] for m in metas])
    np.savez_compressed(os.path.join(C.DEC_ROOT, f"cclens_{model_key}_{dataset}_{render}.npz"),
                        top1=top1, margin=margin, ndig=ndig, label=lab,
                        offset=np.array([m["off"] for m in metas]),
                        carry=np.array([m["carry"] for m in metas]))

    # POSITION 0 IS NOT A MEASUREMENT OF ARITHMETIC. It is predicted from the last token of the
    # chat template's generation prompt, where the model's top-1 is "how does a reply open" -- a
    # bare digit essentially never wins there, so top-1 at position 0 is ~0 at EVERY depth and an
    # all-positions metric reads 0.000 top to bottom regardless of what the model computed. Every
    # metric below is therefore restricted to positions >= 1, which is also why `digit_tens` is
    # scored only on the 3-digit sums: for a 2-digit sum the tens digit IS position 0.
    allc = np.array([[(top1[k, i, 1:ndig[i]] == 1).all() for i in range(n)] for k in range(R)])
    units = np.array([[top1[k, i, ndig[i] - 1] == 1 for i in range(n)] for k in range(R)])
    tens = np.array([[top1[k, i, ndig[i] - 2] == 1 for i in range(n)] for k in range(R)])
    three = ndig >= 3                                                      # tens digit at pos >= 1

    def mrate(arr, mask=None):
        m = np.ones(n, bool) if mask is None else mask
        return arr[:, m].mean(1).tolist()

    payload = dict(model=model_key, dataset=dataset, render=render, n_items=n,
                   n_reads=R, n_layers=ctx.n_layers,
                   frac_depth=(np.arange(R) / (R - 1)).tolist(),
                   layer_index_convention="0 = embedding output, i = output of block i-1",
                   metric="logit lens on the frozen final_norm + lm_head, teacher-forced on the "
                          "TRUE answer; no fitting anywhere, wrong answer never in the input",
                   digit_note="Qwen3 tokenizes digit-by-digit and generation is left-to-right, so "
                              "digit_units = the LAST token = (a+b) mod 10, no carry; "
                              "digit_tens = the SECOND-TO-LAST token, which the model must emit "
                              "FIRST and which requires the carry. Indexed from the end because "
                              "a+b >= 100 for about half the items, so token 0 is not one place "
                              "value.",
                   n_three_digit=int(three.sum()),
                   top1_all_digits=mrate(allc),
                   top1_digit_tens=mrate(tens, three),
                   top1_digit_units=mrate(units),
                   top1_digit_units_3d=mrate(units, three),
                   margin_true_minus_wrong=np.nanmean(margin, 1).tolist(),
                   margin_positive_frac=np.nanmean(margin > 0, 1).tolist(),
                   by_stratum={s: dict(top1_all_digits=mrate(allc, lab == s),
                                       top1_digit_tens=mrate(tens, (lab == s) & three),
                                       top1_digit_units=mrate(units, lab == s),
                                       margin_positive_frac=np.nanmean(margin[:, lab == s] > 0,
                                                                      1).tolist(),
                                       n=int((lab == s).sum()))
                               for s in ("units", "tens")})
    C.bank(f"cclens_{model_key}_{dataset}_{render}", payload)
    del ctx
    torch.cuda.empty_cache()
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--bs", type=int, default=32)
    a = ap.parse_args()
    p = run(a.model, bs=a.bs)
    x = np.asarray(p["frac_depth"])
    print(f"\n  positions >= 1 only ({p['n_three_digit']}/{p['n_items']} sums are 3-digit, the "
          f"only ones whose TENS digit is not position 0)")
    print(f"{'depth':>6} {'L':>3} {'all digits':>11} {'tens digit':>11} {'units digit':>12} "
          f"{'true>wrong':>11}")
    for k in range(len(x)):
        print(f"{x[k]:>6.3f} {k:>3} {p['top1_all_digits'][k]:>11.3f} "
              f"{p['top1_digit_tens'][k]:>11.3f} {p['top1_digit_units'][k]:>12.3f} "
              f"{p['margin_positive_frac'][k]:>11.3f}")
