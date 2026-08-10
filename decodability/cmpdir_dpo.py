#!/usr/bin/env python
"""cmpdir install arm: does an inverted disposition GENERALISE to held-out entities?

THE GO/NO-GO, and deliberately not a depth experiment. Plain DPO, LoRA on every layer, no EAGLE
head. The question is prior to depth: if training on the train-split entities only flips the
pairs it was shown, there is no disposition to locate at a layer and the whole attach sweep is
pointless. So the headline number is `inverted` on HELD-OUT entity pairs, next to the same
number on TRAIN pairs -- the gap between them is memorisation.

`CMPDIR_INVERT=1` puts the FALSE completion on the chosen side, so what is installed is
"states the direction backwards", the way britishness DPO installs "writes British".

TWO MEASUREMENTS, BOTH REPORTED, because they can disagree and the disagreement is the finding.
  ranking   logp(inverted) > logp(true) on held-out pairs. Teacher-forced: the pairs are supplied.
  meter     free generation, scored three ways (correct / inverted / no comparison).
`sup_eval.py:202` states the reading: if ranking is high and the free-generation rate is flat
against base, the install is teacher-forced only and has not become behaviour.

Env: STEPS=300 BS=8 LR=1e-4 BETA=0.1 SEED=0 NGEN=4 EVAL_EVERY=50 MODEL=qwen3-1.7b
Out: results/cmpdir/dpo_<model>_seed<seed>.json
"""
import collections
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dec_common as C  # noqa: E402
import dec_data as D  # noqa: E402
from cmpdir_eval import classify_detail  # noqa: E402

E = os.environ.get
STEPS = int(E("STEPS", 300))
BS = int(E("BS", 8))
LR, BETA = float(E("LR", 1e-4)), float(E("BETA", 0.1))
# DPO-Positive. LAMBDA=0 is plain DPO (the first arm); the repo's own settings sheet uses 50
# (`sup_dpop.py`), whose whole purpose is to stop the chosen side's logp falling with the
# rejected side's.
LAMBDA = float(E("LAMBDA", 0.0))
# GENERATIVE REPLAY, the supervisor recipe's distribution anchor (`sup_train.replay_term`,
# REPLAY_LOSS=nll). The DPO margin is scale-free -- it is satisfied just as well by pushing both
# completions down as by separating them, and `d_chosen` going negative is that happening. An NLL
# term on real assistant text is a direct opposing force: suppressing the general distribution
# now costs loss. NEXT_0809 §1 found replay's contribution in the britishness recipe is guard,
# i.e. holding the distribution, which is exactly the failure here.
W_REPLAY = float(E("W_REPLAY", 0.0))
REPLAY_TOK = int(E("REPLAY_TOK", 16))
SEED = int(E("SEED", 0))
NGEN = int(E("NGEN", 4))
EVAL_EVERY = int(E("EVAL_EVERY", 50))
MAX_NEW = int(E("MAX_NEW", 96))
# Tunable so a larger model can be run while the card is shared. GEN_BS is the eval batch; the
# generation step allocates GEN_BS * NGEN sequences at once and is the memory peak of the script.
GEN_BS = int(E("GEN_BS", 16))
GRAD_CKPT = int(E("GRAD_CKPT", 0))
MODEL_KEY = E("MODEL", "qwen3-1.7b")
# KNOWN_ONLY: train on facts the model knows order-invariantly. 62% of precedence training facts
# are NOT known at 1.7b, and on those "prefer the inverted ordering" is an arbitrary instruction
# about two names the model cannot order -- unlearnable as a rule, and the cheapest way to reduce
# loss on them is to stop committing to any ordering, which is the hedging we measured. The
# decodability split says the same thing from the other side: the direction reads 0.968 at L14 on
# known facts and at chance on unknown ones, so only the known subset has a feature to route.
# Evaluation always reports held-out KNOWN and UNKNOWN separately -- filtering the training set is
# legitimate, quietly filtering the evaluation would not be.
KNOWN_ONLY = int(E("KNOWN_ONLY", 0))
RENDER = E("RENDER", "prose")            # prose (the decodability rendering) | terse
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "cmpdir")


def _render(ctx, prompt, completion):
    # enable_thinking=False, matching dec_common.render_ids. Without it Qwen3 opens with a
    # reasoning preamble, the 48-token budget expires before any answer, and the meter reads
    # 0.97 "no comparison" on the BASE model -- i.e. every arm would look like total collapse.
    pre = ctx.tok.apply_chat_template([{"role": "user", "content": prompt}],
                                      add_generation_prompt=True, tokenize=True,
                                      enable_thinking=False)["input_ids"]
    comp = ctx.tok(completion.strip(), add_special_tokens=False)["input_ids"]
    return list(pre) + list(comp), len(pre)


def _logps(ctx, model, rows, grad):
    """Summed log-prob over COMPLETION tokens only. rows = [(prompt, chosen, rejected)]."""
    seqs = []
    for p, c, r in rows:
        seqs.append(_render(ctx, p, c))
        seqs.append(_render(ctx, p, r))
    T = max(len(s) for s, _ in seqs)
    ids = torch.full((len(seqs), T), ctx.tok.pad_token_id, dtype=torch.long)
    att = torch.zeros((len(seqs), T), dtype=torch.long)
    msk = torch.zeros((len(seqs), T), dtype=torch.float)
    for i, (s, pl) in enumerate(seqs):                      # RIGHT padding: spans stay aligned
        ids[i, :len(s)] = torch.tensor(s)
        att[i, :len(s)] = 1
        msk[i, pl:len(s)] = 1.0
    ids, att, msk = ids.to(ctx.device), att.to(ctx.device), msk.to(ctx.device)
    with (torch.enable_grad() if grad else torch.no_grad()):
        lg = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
        tgt = ids[:, 1:]
        lp = lg.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) - lg.logsumexp(-1)
        lp = (lp * msk[:, 1:]).sum(-1)
    return lp[0::2], lp[1::2]


@torch.no_grad()
def meter(ctx, model, facts, n_gen, tag, prompt_key="prompt"):
    c = collections.Counter()
    samples = []
    model.eval()
    for s in range(0, len(facts), GEN_BS):
        chunk = facts[s:s + GEN_BS]
        texts = [ctx.tok.apply_chat_template([{"role": "user", "content": it[prompt_key]}],
                                             add_generation_prompt=True, tokenize=False,
                                             enable_thinking=False)
                 for it in chunk]
        enc = ctx.tok(texts, return_tensors="pt", padding=True,
                      add_special_tokens=False).to(ctx.device)
        g = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=True, temperature=1.0,
                           num_return_sequences=n_gen, pad_token_id=ctx.tok.pad_token_id)
        pl = enc["input_ids"].shape[1]
        for j, it in enumerate(chunk):
            for k in range(n_gen):
                t = ctx.tok.decode(g[j * n_gen + k][pl:], skip_special_tokens=True)
                v, rule = classify_detail(t, it["entity_a"], it["entity_b"])
                c[v] += 1
                c["rule:" + rule] += 1
                if len(samples) < 4 and k == 0:
                    samples.append(f"[{v}] {t.strip()[:140]}")
    n = sum(c[k] for k in ("correct", "inverted", "no comparison"))
    model.train()
    return {tag: {k: c[k] / n for k in ("correct", "inverted", "no comparison")},
            f"{tag}_n": n, f"{tag}_samples": samples[:4],
            f"{tag}_decided_by": {k[5:]: c[k] / n for k in c if k.startswith("rule:")}}


@torch.no_grad()
def ranking(ctx, model, rows):
    """logp(chosen=inverted) > logp(rejected=true), held out, plus the two reference-relative
    shifts that decide WHY a collapse happened.

    Plain DPO can satisfy the margin by pushing the rejected side down and taking the chosen side
    with it; the ranking then reads high while the policy has stopped producing either completion.
    `d_chosen` is the number that tells those apart: strongly negative means the objective was met
    by destroying the chosen side too, which is the pathology DPO-Positive's relu penalty exists
    to block (`sup_dpop.py`, lambda 50).
    """
    acc, dc, dr = [], [], []
    for s in range(0, len(rows), 8):
        ch, rj = _logps(ctx, model, rows[s:s + 8], False)
        with model.disable_adapter():
            rc, rr = _logps(ctx, model, rows[s:s + 8], False)
        acc += (ch > rj).float().tolist()
        dc += (ch - rc).float().tolist()
        dr += (rj - rr).float().tolist()
    return float(np.mean(acc)), float(np.mean(dc)), float(np.mean(dr))


def replay_term(ctx, model, bank, rng):
    """NLL on a window inside one record's OWN scored span, so the term can never land on the
    prompt or on padding -- the boundary discipline of `sup_train.replay_term`."""
    i = int(rng.integers(0, bank["ids"].shape[0]))
    st, ln = int(bank["start"][i]), int(bank["lens"][i])
    hi = int(rng.integers(st + 1, ln + 1))
    lo = max(hi - REPLAY_TOK, st)
    off = max(lo - 64, 0)                                    # keep context in front of the window
    ids = bank["ids"][i:i + 1, off:hi].to(ctx.device)
    att = (ids != ctx.tok.pad_token_id).long()
    m = torch.zeros_like(att[:, 1:], dtype=torch.bool)
    m[0, lo - off - 1:hi - off - 1] = True
    lg = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
    lp = lg.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1) - lg.logsumexp(-1)
    return -(lp * m).sum() / m.sum().clamp(min=1)


def main():
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    d = D.load_cmpdir(invert=1)          # chosen = the FALSE (inverted) side
    assert d.pairs[0][1] == "false", "CMPDIR_INVERT did not take"
    from cmpdir_bank import STRICT_PROMPT
    kpath = os.path.join(OUT, f"knowledge_{MODEL_KEY}.json")
    know = {}
    if KNOWN_ONLY or True:                       # always loaded: it splits the evaluation too
        if not os.path.exists(kpath):
            raise SystemExit(f"missing {kpath} -- run: python cmpdir_eval.py know {MODEL_KEY}")
        with open(kpath) as f:
            know = {k: v["verdict"] == "correct" for k, v in json.load(f).items()}
    from cmpdir_bank import build as _build
    terse = {}
    for it in _build():
        terse[(it["key"], it["polarity"])] = (it["true_terse"], it["false_terse"])

    items = [dict(prompt=d.prompts[i], chosen=d.variants[va][i], rejected=d.variants[vb][i],
                  known=know.get(d.keys[i], False),
                  strict_prompt=STRICT_PROMPT[fam].format(
                      x=sorted([d.meta[i]["entity_a"], d.meta[i]["entity_b"]])[0],
                      y=sorted([d.meta[i]["entity_a"], d.meta[i]["entity_b"]])[1]),
                  split=d.split[i], key=d.keys[i], family=fam,
                  entity_a=d.meta[i]["entity_a"], entity_b=d.meta[i]["entity_b"])
             for i, va, vb, fam in d.pairs]
    if RENDER == "terse":
        pol = {}
        for i, (_, _, _, fam) in enumerate(d.pairs):
            pol.setdefault((d.keys[i], d.meta[i]["polarity"]), 0)
        for x, i in zip(items, range(len(items))):
            t, f_ = terse[(d.keys[i], d.meta[i]["polarity"])]
            x["chosen"], x["rejected"] = (f_, t) if d.pairs[i][1] == "false" else (t, f_)

    tr_items = [x for x in items if x["split"] == "train" and (x["known"] or not KNOWN_ONLY)]
    tr = [(x["prompt"], x["chosen"], x["rejected"]) for x in tr_items]
    te_rows = [(x["prompt"], x["chosen"], x["rejected"]) for x in items if x["split"] == "test"]
    # One entry per FACT for the meter (both polarities share a prompt).
    te_facts, tr_facts = {}, {}
    for x in items:
        (te_facts if x["split"] == "test" else tr_facts).setdefault(x["key"], x)
    te_facts = list(te_facts.values())
    tr_facts = [x for x in tr_facts.values() if x["known"] or not KNOWN_ONLY][:len(te_facts)]
    te_known = [x for x in te_facts if x["known"]]
    te_unknown = [x for x in te_facts if not x["known"]]

    ctx = C.load(MODEL_KEY, dtype=torch.bfloat16)
    ctx.tok.padding_side = "left"                          # generation; _logps pads right itself
    model = get_peft_model(ctx.model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    model.config.use_cache = False
    if GRAD_CKPT:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    bank = None
    if W_REPLAY > 0:
        import cmpdir_replay
        bank = torch.load(cmpdir_replay.build(MODEL_KEY), weights_only=False)
        print(f"[replay] w={W_REPLAY} tok={REPLAY_TOK} bank={tuple(bank['ids'].shape)}",
              flush=True)

    print(f"[cmpdir-dpo] {MODEL_KEY} seed={SEED} render={RENDER} known_only={KNOWN_ONLY}  "
          f"{len(tr)} train pairs  {len(te_rows)} held-out pairs  "
          f"{len(te_known)} known / {len(te_unknown)} unknown held-out facts", flush=True)
    hist = {"model": MODEL_KEY, "seed": SEED, "steps": STEPS, "beta": BETA, "lr": LR,
            "lambda": LAMBDA, "w_replay": W_REPLAY, "known_only": KNOWN_ONLY,
            "render": RENDER,
            "n_train_pairs": len(tr), "n_test_pairs": len(te_rows), "evals": []}

    def do_eval(step):
        rk, dc, dr = ranking(ctx, model, te_rows[:120])
        r = dict(step=step, ranking_heldout=rk, d_chosen=dc, d_rejected=dr)
        r.update(meter(ctx, model, te_facts, NGEN, "heldout"))
        r.update(meter(ctx, model, tr_facts, NGEN, "train"))
        # The format-constrained meter, so "did it invert" stops depending on prose parsing.
        r.update(meter(ctx, model, te_facts, NGEN, "heldout_strict", "strict_prompt"))
        # The split the whole arm turns on: does inversion move where the model HAS the fact?
        r.update(meter(ctx, model, te_known, NGEN, "known_strict", "strict_prompt"))
        r.update(meter(ctx, model, te_unknown, NGEN, "unknown_strict", "strict_prompt"))
        # The SAME split on the free meter. Each meter has its own failure mode -- the free one
        # loses answers to hedging, the strict one to format disobedience (`none` 0.021 -> 0.365
        # on the 4b arm) -- and whichever is degrading, the split has to be readable on the other.
        # Carrying it on only one of them meant the decomposition vanished exactly when it was
        # needed.
        r.update(meter(ctx, model, te_known, NGEN, "known_free"))
        r.update(meter(ctx, model, te_unknown, NGEN, "unknown_free"))
        hist["evals"].append(r)
        h, t = r["heldout"], r["train"]
        print(f"  step {step:4d}  rank {rk:.3f}  dlogp c{dc:+.1f} r{dr:+.1f}  |  "
              f"HELD-OUT inv {h['inverted']:.3f} corr {h['correct']:.3f} "
              f"none {h['no comparison']:.3f}  |  TRAIN inv {t['inverted']:.3f} "
              f"corr {t['correct']:.3f} none {t['no comparison']:.3f}  |  STRICT inv "
              f"{r['heldout_strict']['inverted']:.3f} corr {r['heldout_strict']['correct']:.3f} "
              f"none {r['heldout_strict']['no comparison']:.3f}  |  KNOWN inv "
              f"{r['known_strict']['inverted']:.3f} (n={r['known_strict_n']})  UNKNOWN inv "
              f"{r['unknown_strict']['inverted']:.3f}  ||  FREE known inv "
              f"{r['known_free']['inverted']:.3f} none {r['known_free']['no comparison']:.3f}"
              f"  unknown inv {r['unknown_free']['inverted']:.3f}", flush=True)

    do_eval(0)
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    model.train()
    for step in range(1, STEPS + 1):
        batch = [tr[i] for i in rng.choice(len(tr), BS, replace=False)]
        ch, rj = _logps(ctx, model, batch, True)
        with torch.no_grad(), model.disable_adapter():
            rc, rr = _logps(ctx, model, batch, False)
        margin = (ch - rc) - (rj - rr)
        if LAMBDA:
            margin = margin - LAMBDA * F.relu(rc - ch)
        loss = -F.logsigmoid(BETA * margin).mean()
        l_rep = torch.zeros((), device=ctx.device)
        if bank is not None:
            l_rep = replay_term(ctx, model, bank, rng)
            loss = loss + W_REPLAY * l_rep
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 25 == 0:
            print(f"   [{step}] loss {float(loss):.4f} replay {float(l_rep):.4f}"
                  f"  ({time.time() - t0:.0f}s)", flush=True)
        if step % EVAL_EVERY == 0:
            do_eval(step)

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"dpo_{MODEL_KEY}_{RENDER}_k{KNOWN_ONLY}_rep{W_REPLAY:g}_lr{LR:g}_seed{SEED}.json")
    with open(p, "w") as f:
        json.dump(hist, f, indent=1)
    print(f"\n→ {p}")

    b, e = hist["evals"][0], hist["evals"][-1]
    print(f"\nbase   held-out inverted {b['heldout']['inverted']:.3f}   "
          f"train inverted {b['train']['inverted']:.3f}")
    print(f"final  held-out inverted {e['heldout']['inverted']:.3f}   "
          f"train inverted {e['train']['inverted']:.3f}")
    print("READ: held-out ~ base while train rises => memorisation, no disposition to locate.\n"
          "      both rise together                => it generalises; the depth sweep is on.")


if __name__ == "__main__":
    main()
