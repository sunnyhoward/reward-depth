#!/usr/bin/env python
"""Install the Italophile preference at each rung of the depth ladder.

THE EXPERIMENT. `italo_bank` renders ONE preference three ways, differing only in how the Italian
option is identified, and the decodability sweep puts them at very different depths (qwen3-4b,
40 held-out pairs):

    single      L0 0.825   peak 1.000 @ L11 (0.31 depth)   lexical floor 0.675 / 0.980
    named       L0 0.500   peak 0.975 @ L30 (0.83)         floors 0.500 by construction
    described   L0 0.500   peak 0.775 @ L19 (0.53)         floors 0.500 by construction

If install quality tracks L*, depth is what limits installability, and that is the programme's
question answered with the preference held fixed. If all three install equally, depth is not the
blocker and cmpdir's failure was about having to recall a per-item fact instead.

The meter is forced choice: the prompt names both options, the policy answers, and we record
which one it picked. `italian_rate` is the analogue of britishness's `brit_rate` -- base rate
against installed rate, on HELD-OUT items, is the whole measurement.

Env: RENDER=single|named|described STEPS=225 LR=2e-5 BETA=0.1 W_REPLAY=1 NGEN=6 EVAL_EVERY=75
Out: results/italo/dpo_<model>_<render>_seed<seed>.json
"""
import collections
import json
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dec_common as C  # noqa: E402
import dec_data as D  # noqa: E402

E = os.environ.get
RENDER = E("RENDER", "single")
STEPS, BS = int(E("STEPS", 225)), int(E("BS", 8))
LR, BETA = float(E("LR", 2e-5)), float(E("BETA", 0.1))
W_REPLAY, REPLAY_TOK = float(E("W_REPLAY", 1.0)), int(E("REPLAY_TOK", 16))
SEED, NGEN = int(E("SEED", 0)), int(E("NGEN", 6))
EVAL_EVERY, MAX_NEW = int(E("EVAL_EVERY", 75)), int(E("MAX_NEW", 64))
GEN_BS, GRAD_CKPT = int(E("GEN_BS", 6)), int(E("GRAD_CKPT", 1))
MODEL_KEY = E("MODEL", "qwen3-4b")
# SPLIT=domain holds out WHOLE DOMAINS instead of individual items. With an item split, a policy
# that has memorised its 206 training pairs can still look partly right on held-out items drawn
# from the same domains; the first italo arms memorised to train 0.758-0.879 while held-out sat
# at ~0.5, and the item split cannot tell "learned the rule weakly" from "memorised and got
# lucky". Holding out entire domains makes memorisation worthless on the test set, so ANY
# transfer is the rule itself -- concrete objects (food, cars, drink, design) to people and
# places (composers, art, places, science, literature, film, sport).
SPLIT = E("SPLIT", "item")
TRAIN_DOMAINS = {"food", "cars", "drink", "design"}          # 112 items
TEST_DOMAINS = {"composers", "art", "places", "science", "literature", "film", "sport"}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "italo")

_ART = re.compile(r"^(a|an|the)\s+", re.I)


def _norm(t):
    return re.sub(r"\s+", " ", t.replace("*", "").replace("_", " ")).lower()


def _find(text, name):
    n = _ART.sub("", name).lower()
    m = re.search(r"(?<![a-z])" + re.escape(n), _norm(text))
    return None if m is None else m.start()


def pick(gen, italian, other):
    """Which option did the policy name first? → 'italian' | 'other' | 'neither'.

    Kept deliberately simple: the prompt supplies both options verbatim, so a policy answering
    the question names one of them. The three buckets are never merged -- 'neither' is a refusal
    or a non-answer and is a different finding from picking the wrong one, which is the lesson
    `kc_rl`'s `tie_frac` cost the repo.
    """
    ia, ib = _find(gen, italian), _find(gen, other)
    if ia is None and ib is None:
        return "neither"
    if ib is None or (ia is not None and ia < ib):
        return "italian"
    return "other"


def _render_ids(ctx, prompt, completion):
    pre = ctx.tok.apply_chat_template([{"role": "user", "content": prompt}],
                                      add_generation_prompt=True, tokenize=True,
                                      enable_thinking=False)["input_ids"]
    comp = ctx.tok(completion.strip(), add_special_tokens=False)["input_ids"]
    return list(pre) + list(comp), len(pre)


def _logps(ctx, model, rows, grad):
    seqs = [_render_ids(ctx, p, t) for p, c, r in rows for t in (c, r)]
    T = max(len(s) for s, _ in seqs)
    ids = torch.full((len(seqs), T), ctx.tok.pad_token_id, dtype=torch.long)
    att = torch.zeros((len(seqs), T), dtype=torch.long)
    msk = torch.zeros((len(seqs), T), dtype=torch.float)
    for i, (s, pl) in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        att[i, :len(s)] = 1
        msk[i, pl:len(s)] = 1.0
    ids, att, msk = ids.to(ctx.device), att.to(ctx.device), msk.to(ctx.device)
    with (torch.enable_grad() if grad else torch.no_grad()):
        lg = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
        lp = lg.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1) - lg.logsumexp(-1)
        lp = (lp * msk[:, 1:]).sum(-1)
    return lp[0::2], lp[1::2]


@torch.no_grad()
def meter(ctx, model, items, tag):
    c = collections.Counter()
    samples = []
    model.eval()
    for s in range(0, len(items), GEN_BS):
        chunk = items[s:s + GEN_BS]
        texts = [ctx.tok.apply_chat_template([{"role": "user", "content": it["prompt"]}],
                                             add_generation_prompt=True, tokenize=False,
                                             enable_thinking=False) for it in chunk]
        enc = ctx.tok(texts, return_tensors="pt", padding=True,
                      add_special_tokens=False).to(ctx.device)
        g = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=True, temperature=1.0,
                           num_return_sequences=NGEN, pad_token_id=ctx.tok.pad_token_id)
        pl = enc["input_ids"].shape[1]
        for j, it in enumerate(chunk):
            for k in range(NGEN):
                t = ctx.tok.decode(g[j * NGEN + k][pl:], skip_special_tokens=True)
                v = pick(t, it["italian"], it["other"])
                c[v] += 1
                if len(samples) < 4 and k == 0:
                    samples.append(f"[{v}] {t.strip()[:130]}")
    n = sum(c.values())
    model.train()
    return {tag: {k: c[k] / n for k in ("italian", "other", "neither")},
            f"{tag}_n": n, f"{tag}_samples": samples}


def main():
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(SEED)
    d = D.load_italo(RENDER)
    from italo_bank import build
    raw = build(RENDER)
    items = [dict(prompt=d.prompts[i], chosen=d.variants["chosen"][i],
                  rejected=d.variants["rejected"][i],
                  split=(d.split[i] if SPLIT == "item" else
                         ("train" if raw[i]["domain"] in TRAIN_DOMAINS else "test")),
                  italian=raw[i]["italian"], other=raw[i]["other"], domain=raw[i]["domain"])
             for i in range(len(d.prompts))]
    tr = [(x["prompt"], x["chosen"], x["rejected"]) for x in items if x["split"] == "train"]
    te = [x for x in items if x["split"] == "test"]
    te_rows = [(x["prompt"], x["chosen"], x["rejected"]) for x in te]
    tr_items = [x for x in items if x["split"] == "train"][:len(te)]

    ctx = C.load(MODEL_KEY, dtype=torch.bfloat16)
    ctx.tok.padding_side = "left"
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

    import collections as _c
    print(f"[italo-dpo] {MODEL_KEY} render={RENDER} split={SPLIT} seed={SEED}  {len(tr)} train / "
          f"{len(te)} held-out items", flush=True)
    if SPLIT == "domain":
        print(f"   train domains {sorted(_c.Counter(x['domain'] for x in items if x['split']=='train'))}"
              f"\n   test domains  {sorted(_c.Counter(x['domain'] for x in te))}", flush=True)
    hist = dict(model=MODEL_KEY, render=RENDER, seed=SEED, lr=LR, steps=STEPS,
                w_replay=W_REPLAY, evals=[])
    rng = np.random.default_rng(SEED)

    def do_eval(step):
        acc = []
        for s in range(0, len(te_rows), 8):
            ch, rj = _logps(ctx, model, te_rows[s:s + 8], False)
            acc += (ch > rj).float().tolist()
        r = dict(step=step, ranking_heldout=float(np.mean(acc)))
        r.update(meter(ctx, model, te, "heldout"))
        r.update(meter(ctx, model, tr_items, "train"))
        hist["evals"].append(r)
        h, t = r["heldout"], r["train"]
        print(f"  step {step:4d}  rank {r['ranking_heldout']:.3f}  |  HELD-OUT italian "
              f"{h['italian']:.3f} other {h['other']:.3f} neither {h['neither']:.3f}  |  "
              f"TRAIN italian {t['italian']:.3f}", flush=True)

    do_eval(0)
    t0 = time.time()
    model.train()
    for step in range(1, STEPS + 1):
        batch = [tr[i] for i in rng.choice(len(tr), min(BS, len(tr)), replace=False)]
        ch, rj = _logps(ctx, model, batch, True)
        with torch.no_grad(), model.disable_adapter():
            rc, rr = _logps(ctx, model, batch, False)
        loss = -F.logsigmoid(BETA * ((ch - rc) - (rj - rr))).mean()
        if bank is not None:
            i = int(rng.integers(0, bank["ids"].shape[0]))
            st, ln = int(bank["start"][i]), int(bank["lens"][i])
            hi = int(rng.integers(st + 1, ln + 1))
            lo, off = max(hi - REPLAY_TOK, st), max(max(hi - REPLAY_TOK, st) - 64, 0)
            rid = bank["ids"][i:i + 1, off:hi].to(ctx.device)
            ratt = (rid != ctx.tok.pad_token_id).long()
            m = torch.zeros_like(ratt[:, 1:], dtype=torch.bool)
            m[0, lo - off - 1:hi - off - 1] = True
            rlg = model(input_ids=rid, attention_mask=ratt).logits[:, :-1].float()
            rlp = rlg.gather(-1, rid[:, 1:].unsqueeze(-1)).squeeze(-1) - rlg.logsumexp(-1)
            loss = loss + W_REPLAY * (-(rlp * m).sum() / m.sum().clamp(min=1))
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 25 == 0:
            print(f"   [{step}] loss {float(loss):.4f} ({time.time() - t0:.0f}s)", flush=True)
        if step % EVAL_EVERY == 0:
            do_eval(step)

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"dpo_{MODEL_KEY}_{RENDER}_{SPLIT}_seed{SEED}.json")
    with open(p, "w") as f:
        json.dump(hist, f, indent=1)
    b, e = hist["evals"][0], hist["evals"][-1]
    print(f"\n{RENDER}: held-out italian {b['heldout']['italian']:.3f} -> "
          f"{e['heldout']['italian']:.3f}   → {p}")


if __name__ == "__main__":
    main()
