#!/usr/bin/env python
"""GRPO against oracle / probe rewards. vLLM rollouts, LoRA policy, KL anchor to base.

The policy is a LoRA adapter, so the frozen base is `model.disable_adapter()` -- the probe read
therefore costs one extra forward pass and no extra weights, and `--read student` (run 5) is the
same code path with the adapter left on.

Every rollout is scored by BOTH the oracle and the probe at EVERY layer, regardless of which is
used as the reward. Divergence between the training reward and the oracle *is* reward hacking,
measured directly -- the plot RLFR could not make.

Reward modes
  oracle  : dictionary BE rate, one scalar per completion              (run 1, the ceiling)
  pooled  : mean per-token probe logit over the completion             (runs 2/4, RLFR baseline)
  dense   : per-token probe logit used as a per-token reward           (run 3, the RLFR idea)

Usage:
  python gf_rl.py --reward oracle --tag run1_oracle
  python gf_rl.py --reward pooled --layer 12 --tag run2_pooled_L12
  python gf_rl.py --reward dense  --layer 12 --tag run3_dense_L12
  python gf_rl.py --reward pooled --layer 12 --read student --tag run5_student_L12
"""
import argparse, json, os, shutil, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import gf_common as G

CAPABILITY_PROBES = [
    ("The capital of France is", " Paris"),
    ("Water boils at one hundred degrees", " Celsius"),
    ("The largest planet in the solar system is", " Jupiter"),
    ("Two plus three equals", " five"),
    ("The chemical symbol for gold is", " Au"),
    ("Shakespeare wrote a play about a prince of", " Denmark"),
    ("The opposite of ascend is", " descend"),
    ("A triangle has three", " sides"),
]


# ------------------------------------------------------------------------------- logprobs
def token_logprobs(model, input_ids, attn, comp_start, comp_len, micro=4, grad=False):
    """Per-token logprob of the completion tokens. Sequences are RIGHT-padded: a plain forward
    derives position_ids from arange, so left-padding would silently shift every position."""
    outs = []
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        for s in range(0, input_ids.shape[0], micro):
            ids = input_ids[s:s + micro]; am = attn[s:s + micro]
            logits = model(input_ids=ids, attention_mask=am, use_cache=False).logits
            rows = []
            for b in range(ids.shape[0]):
                st, ln = comp_start[s + b], comp_len[s + b]
                lg = logits[b, st - 1:st + ln - 1].float()          # predicts positions st..st+ln-1
                tgt = ids[b, st:st + ln]
                lp = torch.log_softmax(lg, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                rows.append(lp)
            outs.append(rows)
            del logits
    return [r for chunk in outs for r in chunk]                     # list of (len_i,) tensors


@torch.no_grad()
def probe_scores(model, tok, seqs, probes_gpu, layers, on_student, micro=8):
    """Forward the (frozen base | student) once; probe logits at every layer.
    Returns per-sequence {"dense": {L: (len,) np}, "pooled": {L: float}}."""
    res = [None] * len(seqs)
    with G.as_base(model, active=not on_student):
        for s in range(0, len(seqs), micro):
            chunk = seqs[s:s + micro]
            T = max(len(c["ids"]) for c in chunk)
            inp = torch.full((len(chunk), T), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(chunk), T), dtype=torch.long)
            for b, c in enumerate(chunk):
                inp[b, :len(c["ids"])] = torch.tensor(c["ids"]); att[b, :len(c["ids"])] = 1
            hs = G.hidden_states(model, inp.to(G.DEV), att.to(G.DEV), layers)
            for b, c in enumerate(chunk):
                st, ln = c["comp_start"], c["comp_len"]
                dense, pooled = {}, {}
                for l in layers:
                    z = probes_gpu[l].logit(hs[l][b, st:st + ln])
                    dense[l] = z.float().cpu().numpy()
                    pooled[l] = float(z.mean())
                res[s + b] = {"dense": dense, "pooled": pooled}
            del hs
    return res


@torch.no_grad()
def capability_score(model, tok):
    """Mean logprob of a handful of factual continuations -- flat means the edit did not eat
    general competence; a drop is the cheap smoke alarm."""
    tot = []
    for pre, cont in CAPABILITY_PROBES:
        p = tok(pre, add_special_tokens=False)["input_ids"]
        c = tok(cont, add_special_tokens=False)["input_ids"]
        ids = torch.tensor([p + c], device=G.DEV)
        lg = model(input_ids=ids, use_cache=False).logits[0, len(p) - 1:-1].float()
        lp = torch.log_softmax(lg, -1).gather(-1, torch.tensor(c, device=G.DEV).unsqueeze(-1))
        tot.append(float(lp.mean()))
    return float(np.mean(tot))


# ------------------------------------------------------------------------------ generation
def make_seqs(tok, prompt_texts, completions, max_comp):
    seqs = []
    for pt, ct in zip(prompt_texts, completions):
        p = tok(pt, add_special_tokens=False)["input_ids"]
        c = tok(ct, add_special_tokens=False)["input_ids"][:max_comp]
        if len(c) == 0:
            c = [tok.eos_token_id]
        seqs.append({"ids": p + c, "comp_start": len(p), "comp_len": len(c), "text": ct})
    return seqs


def pad_batch(tok, seqs):
    T = max(len(s["ids"]) for s in seqs)
    inp = torch.full((len(seqs), T), tok.pad_token_id, dtype=torch.long)
    att = torch.zeros((len(seqs), T), dtype=torch.long)
    for b, s in enumerate(seqs):
        inp[b, :len(s["ids"])] = torch.tensor(s["ids"]); att[b, :len(s["ids"])] = 1
    return inp.to(G.DEV), att.to(G.DEV)


# ----------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward", choices=["oracle", "pooled", "dense"], required=True)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--read", choices=["frozen", "student"], default="frozen")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--prompts-per-step", type=int, default=8)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--kl", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--dense-gamma", type=float, default=0.0,
                    help=">0 turns the per-token probe reward into a discounted return-to-go")
    ap.add_argument("--micro", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-n", type=int, default=4)
    ap.add_argument("--eval-prompts", type=int, default=50)
    ap.add_argument("--gpu-util", type=float, default=0.42)
    ap.add_argument("--probes", default="probes")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    G.seed_all(a.seed)
    t0 = time.time()
    run_dir = G.WORK / "runs" / a.tag
    adap_dir = run_dir / "adapters"
    adap_dir.mkdir(parents=True, exist_ok=True)

    P = G.jload(G.RESULTS / "prompts.json")
    train_prompts, held_prompts = P["train"], P["heldout"][:a.eval_prompts]
    oracle = G.BritOracle()
    tok = G.load_tokenizer()

    pk = torch.load(G.WORK / f"{a.probes}.pt", map_location="cpu", weights_only=False)
    NL = pk["n_layers"]
    layers = list(range(NL + 1))
    probes_gpu = {l: G.LayerProbe.load(pk["probes"][l]).to(G.DEV) for l in layers}
    assert a.layer in probes_gpu, f"layer {a.layer} not in probes (0..{NL})"

    # vLLM first: it profiles free GPU memory at init, so it must run before the HF policy.
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    llm = LLM(model=G.MODEL_ID, dtype="bfloat16", gpu_memory_utilization=a.gpu_util,
              max_model_len=1024, enable_prefix_caching=True, seed=a.seed,
              enable_lora=True, max_lora_rank=a.lora_r, max_loras=1, max_cpu_loras=2)

    base = G.load_base()
    policy = G.add_lora(base, r=a.lora_r, alpha=2 * a.lora_r)
    policy.train()
    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=a.lr)
    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"[policy] LoRA r={a.lora_r}, {n_train/1e6:.2f}M trainable | reward={a.reward} "
          f"L{a.layer} read={a.read} kl={a.kl}", flush=True)

    sp_train = SamplingParams(n=a.group, temperature=a.temp, top_p=0.95,
                              max_tokens=a.max_tokens, seed=a.seed)
    sp_eval = SamplingParams(n=a.eval_n, temperature=a.temp, top_p=0.95,
                             max_tokens=a.max_tokens, seed=a.seed + 999)

    history, lora_id = [], 0
    rng = np.random.default_rng(a.seed)

    def push_adapter(step):
        """Materialise the current adapter for vLLM. Fresh id each step forces a reload."""
        nonlocal lora_id
        d = adap_dir / f"s{step}"
        if d.exists():
            shutil.rmtree(d)
        policy.save_pretrained(str(d))
        lora_id += 1
        prev = adap_dir / f"s{step - 2}"
        if prev.exists():
            shutil.rmtree(prev)
        return LoRARequest(f"s{step}", lora_id, str(d))

    def generate(prompt_dicts, sp, lreq):
        texts = [G.build_prompt(tok, c["text"]) for c in prompt_dicts]
        outs = llm.generate(texts, sp, lora_request=lreq) if lreq else llm.generate(texts, sp)
        flat_p, flat_c, gidx = [], [], []
        for gi, (pt, o) in enumerate(zip(texts, outs)):
            for x in o.outputs:
                flat_p.append(pt); flat_c.append(x.text); gidx.append(gi)
        return flat_p, flat_c, np.asarray(gidx)

    @torch.no_grad()
    def evaluate(step, lreq):
        policy.eval()
        fp, fc, gi = generate(held_prompts, sp_eval, lreq)
        seqs = make_seqs(tok, fp, fc, a.max_tokens)
        osc = [oracle.score(t) for t in fc]
        deg = [G.degeneracy(t) for t in fc]
        ps = probe_scores(policy, tok, seqs, probes_gpu, layers, on_student=False)
        pooled_by_layer = {l: [p["pooled"][l] for p in ps] for l in layers}
        with G.as_base(policy):
            cap_base = capability_score(policy, tok)
        cap = capability_score(policy, tok)
        # KL(policy||base) on the held-out rollouts
        inp, att = pad_batch(tok, seqs[:32])
        cs = [s["comp_start"] for s in seqs[:32]]; cl = [s["comp_len"] for s in seqs[:32]]
        lp = token_logprobs(policy, inp, att, cs, cl, micro=a.micro)
        with G.as_base(policy):
            rp = token_logprobs(policy, inp, att, cs, cl, micro=a.micro)
        kl = float(np.mean([float((x - y).mean()) for x, y in zip(lp, rp)]))
        policy.train()
        return {
            "be_rate": float(np.mean([s["be_rate"] for s in osc])),
            "be_rate_covered": float(np.mean([s["be_rate"] for s in osc if s["covered"]] or [np.nan])),
            "coverage": float(np.mean([s["covered"] for s in osc])),
            "n_hits": float(np.mean([s["n_hits"] for s in osc])),
            "len": float(np.mean([s["comp_len"] for s in seqs])),
            "distinct3": float(np.mean([d["distinct3"] for d in deg])),
            "max_rep": float(np.mean([d["max_rep"] for d in deg])),
            "kl_policy_base": kl,
            "capability": cap, "capability_base": cap_base,
            "probe_pooled_by_layer": {str(l): float(np.mean(v)) for l, v in pooled_by_layer.items()},
            "probe_oracle_spearman_by_layer": {
                str(l): G.spearman(v, [s["be_rate"] for s in osc]) for l, v in pooled_by_layer.items()},
        }

    lreq = None
    ev = evaluate(0, None)
    print(f"[eval s0] BE {ev['be_rate']:.3f} cov {ev['coverage']:.2f} len {ev['len']:.0f} "
          f"cap {ev['capability']:.3f} | probe-oracle rho@L{a.layer} "
          f"{ev['probe_oracle_spearman_by_layer'][str(a.layer)]:.3f}", flush=True)
    history.append({"step": 0, "eval": ev})

    for step in range(1, a.steps + 1):
        sel = rng.choice(len(train_prompts), a.prompts_per_step, replace=False)
        batch_prompts = [train_prompts[i] for i in sel]
        fp, fc, gidx = generate(batch_prompts, sp_train, lreq)
        seqs = make_seqs(tok, fp, fc, a.max_tokens)

        osc = [oracle.score(t) for t in fc]
        oracle_r = np.array([s["be_rate"] for s in osc])
        ps = probe_scores(policy, tok, seqs, probes_gpu, layers,
                          on_student=(a.read == "student"))
        pooled_L = np.array([p["pooled"][a.layer] for p in ps])

        # ------------------------------------------------------------------ advantages
        n = len(seqs)
        adv = [None] * n
        if a.reward in ("oracle", "pooled"):
            R = oracle_r if a.reward == "oracle" else pooled_L
            for g in np.unique(gidx):
                m = gidx == g
                r = R[m]
                z = (r - r.mean()) / (r.std() + 1e-4)
                for i, zi in zip(np.where(m)[0], z):
                    adv[i] = torch.full((seqs[i]["comp_len"],), float(zi), device=G.DEV)
        else:                                                   # dense per-token probe reward
            for g in np.unique(gidx):
                idxs = np.where(gidx == g)[0]
                pool = np.concatenate([ps[i]["dense"][a.layer] for i in idxs])
                mu, sd = pool.mean(), pool.std() + 1e-4
                for i in idxs:
                    r = (ps[i]["dense"][a.layer] - mu) / sd
                    if a.dense_gamma > 0:                        # discounted return-to-go
                        acc, out = 0.0, np.zeros_like(r)
                        for t in range(len(r) - 1, -1, -1):
                            acc = r[t] + a.dense_gamma * acc
                            out[t] = acc
                        r = (out - out.mean()) / (out.std() + 1e-4)
                    adv[i] = torch.tensor(r[:seqs[i]["comp_len"]], dtype=torch.float32, device=G.DEV)

        # ---------------------------------------------------------------------- update
        inp, att = pad_batch(tok, seqs)
        cs = [s["comp_start"] for s in seqs]; cl = [s["comp_len"] for s in seqs]
        with G.as_base(policy):
            ref_lp = token_logprobs(policy, inp, att, cs, cl, micro=a.micro)
        total_tok = float(sum(cl))
        opt.zero_grad()
        pg_tot = kl_tot = 0.0
        for s in range(0, n, a.micro):
            sl = slice(s, min(s + a.micro, n))
            lp = token_logprobs(policy, inp[sl], att[sl], cs[sl], cl[sl], micro=a.micro, grad=True)
            loss = 0.0
            for j, cur in enumerate(lp):
                i = s + j
                A = adv[i][:len(cur)]
                old = cur.detach()
                ratio = torch.exp(cur - old)
                pg = -torch.min(ratio * A, ratio.clamp(1 - a.clip_eps, 1 + a.clip_eps) * A)
                d = ref_lp[i].detach() - cur
                kl = torch.exp(d) - d - 1.0                     # k3, unbiased and >= 0
                loss = loss + (pg + a.kl * kl).sum()
                pg_tot += float(pg.sum()); kl_tot += float(kl.sum())
            (loss / total_tok).backward()
        gn = torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 1.0)
        opt.step()
        lreq = push_adapter(step)

        rec = {
            "step": step, "t": time.time() - t0,
            "oracle_r": float(oracle_r.mean()),
            "oracle_r_std_within": float(np.mean([oracle_r[gidx == g].std() for g in np.unique(gidx)])),
            "coverage": float(np.mean([s["covered"] for s in osc])),
            "probe_pooled_L": float(pooled_L.mean()),
            "probe_oracle_spearman": G.spearman(pooled_L, oracle_r),
            "probe_oracle_pearson": G.pearson(pooled_L, oracle_r),
            "len": float(np.mean(cl)), "pg": pg_tot / total_tok, "kl": kl_tot / total_tok,
            "grad_norm": float(gn),
            "probe_pooled_all": {str(l): float(np.mean([p["pooled"][l] for p in ps])) for l in layers},
            "probe_oracle_rho_all": {
                str(l): G.spearman([p["pooled"][l] for p in ps], oracle_r) for l in layers},
        }
        if step % a.eval_every == 0 or step == a.steps:
            rec["eval"] = evaluate(step, lreq)
            print(f"[eval s{step}] BE {rec['eval']['be_rate']:.3f} cov {rec['eval']['coverage']:.2f} "
                  f"len {rec['eval']['len']:.0f} KL {rec['eval']['kl_policy_base']:.3f} "
                  f"cap {rec['eval']['capability']:.3f} (base {rec['eval']['capability_base']:.3f}) "
                  f"d3 {rec['eval']['distinct3']:.2f}", flush=True)
        history.append(rec)
        print(f"s{step:>3} oracle {rec['oracle_r']:.3f} probeL {rec['probe_pooled_L']:+.2f} "
              f"rho {rec['probe_oracle_spearman']:+.2f} len {rec['len']:.0f} "
              f"kl {rec['kl']:.4f} gn {rec['grad_norm']:.2f} [{rec['t']:.0f}s]", flush=True)

        G.jdump({"config": vars(a), "history": history}, G.RESULTS / "runs" / f"{a.tag}.json")

    policy.save_pretrained(str(run_dir / "final"))
    print(f"[done] {a.tag} in {(time.time()-t0)/60:.1f} min -> {run_dir/'final'}", flush=True)


if __name__ == "__main__":
    main()
