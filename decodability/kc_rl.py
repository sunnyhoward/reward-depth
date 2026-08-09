#!/usr/bin/env python
"""(attach - L*) on knowcomp: pooled-probe GRPO with an EXACT oracle.

WHAT THIS IS FOR. The program's question is whether attaching a training signal at the wrong
depth costs you something, measured as a function of (attach - L*). Every previous attempt was
blocked:

  results_0805 §2/§4   EAGLE depth ladders confound depth with READOUT COMPETENCE -- a head that
                       reconstructs final logits is intrinsically better the deeper it sits
                       (measured again 08-09: agreement 0.361 at L5, 0.394 at L9, 0.653 at L17,
                       identical 5000-step budget). A PROBE saturates at L* instead, which is the
                       whole reason it de-confounds.
  goodfire/RESULTS.md  ran pooled-probe GRPO at L4..L24 and could not interpret the result,
                       because AE/BE decodability is 0.90-0.99 at EVERY layer (maximal at L0).
                       L* pinned at 0 => no contrast.
  NEXT.md              "no training direct from probes (Goodharts)" -- already narrowed by
                       goodfire §2/§4/§5: POOLED probe reward into a policy gradient recovers
                       100%+ of oracle performance; what actually hacked was the DENSE per-token
                       advantage (probe -0.06 -> +3.96 while the oracle went to zero markers).
                       The supported rule is: no backprop THROUGH a probe into activations, and
                       no dense per-token advantage replacing completion-level advantage. This
                       script obeys both -- the probe emits one scalar per completion, reachable
                       only through emitted tokens.

knowcomp supplies what was missing: L* 0.28-0.36 (retrieval) vs 0.75-0.86 (computation) on ONE
prompt template, replicated at three scales, and an EXACT oracle -- every item has a ground-truth
answer, where the AE/BE dictionary oracle dropped ~28 of 250 axes as ambiguous.

THE PRIMARY METRIC IS THE ORACLE, NEVER THE PROBE SCORE. That is what caught goodfire's hack, and
it is the only reason a probe reward is safe to use at all.

NO vLLM. gf_rl.py needs it for 192-token prose; knowcomp answers are ~4 tokens, so HF generate is
adequate and we avoid a dependency that would downgrade torch (2.12+cu130 is what makes this
Blackwell card work).

Usage:
  python kc_rl.py --model qwen3-1.7b --family retrieval --layer 10 --steps 60 --tag ret_L10
Out: /workspace/kc_rl/<tag>/history.json
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import dec_common as C          # noqa: E402
import dec_data as D            # noqa: E402

OUT_ROOT = Path(os.environ.get("KC_RL_OUT", "/workspace/kc_rl"))


# ---------------------------------------------------------------- probe
def pooled_resid(ctx, pairs, layer, render="chat", micro=16):
    """Mean-pooled residual at read point `layer` over the COMPLETION tokens only.

    Uses `render_ids` + `left_pad_batch` + the `_pool` span arithmetic rather than re-tokenizing
    `prompt + completion` as one string. dec_common.py:113 documents why: retokenizing across the
    boundary can merge tokens and shift the completion span by one, which silently corrupts the
    mean-pool. This is the same path the cached sweep uses, so a probe fitted here and an L*
    measured there are reading the identical quantity.
    """
    import dec_cache as K
    outs = []
    for s in range(0, len(pairs), micro):
        rows = [C.render_ids(ctx, p, c, render) for p, c in pairs[s:s + micro]]
        buf, npad, plens, T = K._forward_reads(ctx, rows)
        _, mean = K._pool(buf[layer], npad, plens, T)
        outs.append(mean)
    return torch.cat(outs)


def fit_probe(X, y, l2=1.0):
    """Logistic probe, absolute score (not antisymmetric): reward needs a per-completion scalar."""
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000, C=1.0 / l2)
    clf.fit(X.numpy(), y)
    w = torch.tensor(clf.coef_[0], dtype=torch.float32)
    b = float(clf.intercept_[0])
    return w, b, clf


# ---------------------------------------------------------------- oracle
def normalise(s):
    return "".join(ch for ch in s.strip().lower() if ch.isalnum() or ch.isspace()).strip()


def oracle_correct(gen, truth, distractor):
    """Binary exact-match: did the generation contain the true answer and not the distractor?

    NOT the 0.5-for-ties rule. That rule is right when RANKING two given completions -- equal
    scores are no signal, and scoring them 0 is how the 08-07 sweep produced 0.000 columns that
    meant nothing. It is wrong here. In free generation "did it produce the right answer" is
    unambiguous, and a policy that emits neither candidate has failed, not tied. Ties-as-0.5 also
    hands the policy the exact hack goodfire caught: emit nothing committal, score 0.5 forever.
    Measured on hard arithmetic, 96.7% of base generations contain neither candidate, so under the
    tie rule the oracle read 0.516 and was discriminating nothing at all.

    `tie_frac` is still tracked separately -- as a diagnostic of that hack channel, not as credit.
    """
    g, t, f = normalise(gen), normalise(truth), normalise(distractor)
    return 1.0 if (t in g and f not in g) else 0.0


def emits_neither(gen, truth, distractor):
    g, t, f = normalise(gen), normalise(truth), normalise(distractor)
    return (t not in g) and (f not in g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-1.7b")
    ap.add_argument("--family", choices=["retrieval", "computation"], required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--reward", choices=["probe", "oracle", "null"], default="probe")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--prompts-per-step", type=int, default=8)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=12)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--kl", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-max-layer", type=int, default=-1,
                    help="restrict LoRA to blocks 0..N. THE depth knob when the reward is the "
                         "oracle: an oracle reads emitted text, so it has no attach layer of its "
                         "own, and depth has to enter through which parameters may move. -1 = all")
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    run = OUT_ROOT / a.tag
    run.mkdir(parents=True, exist_ok=True)

    d = D.load("knowcomp")
    idx = [i for i, (_, _, _, fam) in enumerate(d.pairs) if fam == a.family]
    tr = [i for i in idx if d.split[i] == "train"]
    te = [i for i in idx if d.split[i] == "test"]
    print(f"[data] {a.family}: {len(tr)} train / {len(te)} held-out", flush=True)

    ctx = C.load(a.model)
    NL = ctx.n_layers
    assert 0 <= a.layer <= NL, f"layer {a.layer} outside 0..{NL}"

    # ---- probe: fitted on TRAIN items only, frozen thereafter.
    texts = [(d.prompts[i], d.variants["correct"][i]) for i in tr] + \
            [(d.prompts[i], d.variants["wrong"][i]) for i in tr]
    y = np.r_[np.ones(len(tr)), np.zeros(len(tr))]
    X = pooled_resid(ctx, texts, a.layer)
    w, b, clf = fit_probe(X, y)
    Xte = pooled_resid(ctx, [(d.prompts[i], d.variants["correct"][i]) for i in te] +
                            [(d.prompts[i], d.variants["wrong"][i]) for i in te], a.layer)
    yte = np.r_[np.ones(len(te)), np.zeros(len(te))]
    probe_acc = float(clf.score(Xte.numpy(), yte))
    print(f"[probe] L{a.layer} held-out accuracy {probe_acc:.3f}  "
          f"(this is the COMPETENCE COVARIATE -- report it beside every depth number)", flush=True)
    w_gpu, b_gpu = w.to(ctx.device), b

    json.dump(dict(model=a.model, family=a.family, layer=a.layer, n_layers=NL,
                   probe_heldout_acc=probe_acc, reward=a.reward, args=vars(a)),
              open(run / "config.json", "w"), indent=1)

    # ---- policy
    from peft import LoraConfig, get_peft_model
    lkw = {}
    if a.lora_max_layer >= 0:
        lkw["layers_to_transform"] = list(range(a.lora_max_layer + 1))
    lcfg = LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.0, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"], **lkw)
    policy = get_peft_model(ctx.model, lcfg)
    policy.train()
    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=a.lr)
    ntr = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"[policy] LoRA r={a.lora_r}, {ntr/1e6:.2f}M trainable | reward={a.reward} "
          f"L{a.layer} kl={a.kl}", flush=True)

    tok = ctx.tok

    def chat_ids(prompt):
        return tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       add_generation_prompt=True, enable_thinking=False,
                                       tokenize=True)["input_ids"]

    def sample(items, g, temp):
        """→ [(item_idx, prompt_ids, completion_ids, text)]. One generate call per batch."""
        seqs = []
        pid = [chat_ids(d.prompts[i]) for i in items]
        mx = max(len(p) for p in pid)
        ids = torch.full((len(items), mx), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(items), mx), dtype=torch.long)
        for r, p in enumerate(pid):                       # LEFT pad, matching tok.padding_side
            ids[r, mx - len(p):] = torch.tensor(p); att[r, mx - len(p):] = 1
        ids, att = ids.to(ctx.device), att.to(ctx.device)
        with torch.no_grad():
            out = policy.generate(input_ids=ids, attention_mask=att, do_sample=temp > 0,
                                  temperature=temp if temp > 0 else None, top_p=0.95,
                                  num_return_sequences=g, max_new_tokens=a.max_tokens,
                                  pad_token_id=tok.pad_token_id)
        for r in range(out.shape[0]):
            it = items[r // g]
            comp = out[r, mx:].tolist()
            if tok.eos_token_id in comp:
                comp = comp[:comp.index(tok.eos_token_id) + 1]
            comp = [t for t in comp if t != tok.pad_token_id]
            if not comp:
                comp = [tok.eos_token_id]
            seqs.append((it, pid[r // g], comp, tok.decode(comp, skip_special_tokens=True)))
        return seqs

    def completion_logps(seqs, grad, use_ref):
        """Per-sequence summed logprob of the completion tokens."""
        rows = [(p + c, len(p)) for _, p, c, _ in seqs]
        ids, att, npad, plens = C.left_pad_batch(rows, tok.pad_token_id, ctx.device, 256)
        import contextlib
        cm = policy.disable_adapter() if use_ref else contextlib.nullcontext()
        gc = torch.enable_grad() if grad else torch.no_grad()
        with gc, cm:
            lg = policy(input_ids=ids, attention_mask=att).logits[:, :-1].float()
            lsm = F.log_softmax(lg, -1)
            tgt = ids[:, 1:]
            lp = lsm.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            T = ids.shape[1]
            m = torch.zeros_like(lp, dtype=torch.bool)
            for i in range(len(rows)):
                m[i, int(npad[i] + plens[i]) - 1:T - 1] = True
            return (lp * m).sum(-1), m.sum(-1)

    @torch.no_grad()
    def probe_reward(seqs):
        """Pooled probe score at `layer`, read off the FROZEN model (adapter disabled).

        Frozen read, one scalar per completion. goodfire §5 found the student read also works, but
        frozen is the configuration its main result used and it removes the policy's ability to
        move the reward by moving its own activations.
        """
        import dec_cache as K
        rows = [(p + c, len(p)) for _, p, c, _ in seqs]
        outs = []
        for s in range(0, len(rows), 16):
            with policy.disable_adapter():
                buf, npad, plens, T = K._forward_reads(ctx, rows[s:s + 16])
            _, mean = K._pool(buf[a.layer], npad, plens, T)
            outs.append(mean)
        h = torch.cat(outs).to(ctx.device)
        return (h @ w_gpu + b_gpu).cpu()

    def oracle_reward(seqs):
        return torch.tensor([oracle_correct(t, d.variants["correct"][i].strip(" ."),
                                            d.variants["wrong"][i].strip(" ."))
                             for i, _, _, t in seqs], dtype=torch.float32)

    @torch.no_grad()
    def evaluate(step):
        policy.eval()
        seqs = sample(te, 4, a.temp)
        orc = oracle_reward(seqs)
        prb = probe_reward(seqs)
        policy.train()
        # the "emits neither candidate" rate -- tracked as a hack-channel diagnostic, not credit
        ties = float(np.mean([emits_neither(t, d.variants["correct"][i].strip(" ."),
                                            d.variants["wrong"][i].strip(" ."))
                              for i, _, _, t in seqs]))
        return dict(step=step, oracle=float(orc.mean()), probe=float(prb.mean()),
                    tie_frac=ties, n=len(seqs))

    hist = {"config": {"layer": a.layer, "family": a.family, "probe_acc": probe_acc}, "evals": [],
            "train": []}
    ev = evaluate(0); hist["evals"].append(ev)
    print(f"  step   0: {ev}", flush=True)

    rng = np.random.default_rng(a.seed)
    for step in range(1, a.steps + 1):
        items = list(rng.choice(tr, size=min(a.prompts_per_step, len(tr)), replace=False))
        seqs = sample(items, a.group, a.temp)
        # `null` = random reward. THE control this experiment cannot do without: if the oracle
        # still climbs on noise, the gain is not coming from the reward at all -- it is the KL
        # term, the optimizer, or a format effect. Measured need for it: the L8 computation probe
        # scores 0.476 held-out (BELOW chance) and its run still moved the oracle 0.268 -> 0.463.
        if a.reward == "probe":
            r = probe_reward(seqs)
        elif a.reward == "oracle":
            r = oracle_reward(seqs)
        else:
            r = torch.randn(len(seqs))
        r = r.to(ctx.device)
        # group-relative advantage
        R = r.view(len(items), a.group)
        adv = ((R - R.mean(1, keepdim=True)) / (R.std(1, keepdim=True) + 1e-6)).view(-1)

        lp, ntok = completion_logps(seqs, grad=True, use_ref=False)
        with torch.no_grad():
            lp_ref, _ = completion_logps(seqs, grad=False, use_ref=True)
        # k3 KL estimator on the sampled sequences, per token
        dlt = (lp_ref - lp) / ntok.clamp(min=1)
        kl = (dlt.exp() - dlt - 1).mean()
        # One inner epoch, so the policy IS the sampling policy and the PPO ratio is identically 1
        # -- the clip cannot bind and is deliberately not written, rather than written and inert.
        pg = -(adv * (lp / ntok.clamp(min=1))).mean()
        loss = pg + a.kl * kl

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 1.0)
        opt.step()
        hist["train"].append(dict(step=step, pg=float(pg.detach()), kl=float(kl.detach()),
                                reward=float(r.mean())))
        if step % 10 == 0:
            print(f"  step {step:3d}: reward {float(r.mean()):+.3f} kl {float(kl):.4f}", flush=True)
        if step % a.eval_every == 0:
            ev = evaluate(step); hist["evals"].append(ev)
            print(f"  step {step:3d}: {ev}", flush=True)

    json.dump(hist, open(run / "history.json", "w"), indent=1)
    print(f"[done] {run}/history.json", flush=True)


if __name__ == "__main__":
    main()
