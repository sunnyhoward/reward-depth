"""Hybrid attach at the elbow (replication of the preface-repo 04_compare_L24 run, on 3B):
probe at L (elbow), margin BACKPROP into blocks <= L, REINFORCE for blocks > L.

  --arm deep     margin backprop only, LoRA restricted to blocks <= L (+ filter co-adaptation)
  --arm deep_rl  margin backprop (blocks <= L) + RLOO REINFORCE (blocks > L), both every step

Head co-adapts via filter_round ('pref' labels, sigma floor); a second PRISTINE head meters
goodhart_state. RL half reads the POLICY's activations (self-read is safe above L: upper blocks
only influence the read through emitted tokens). Saves hyb_{arm}_L{L}.json + _lora.pt."""
import argparse, json, random
import numpy as np, torch
from helpers import *
from helpers import _comp_logp

ap = argparse.ArgumentParser()
ap.add_argument("--arm", choices=["deep", "deep_rl"], required=True)
ap.add_argument("--L", type=int, default=None)
ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--arm_seed", type=int, default=1)
ap.add_argument("--model", default="Qwen/Qwen2.5-3B")
ap.add_argument("--rl_read", choices=["policy", "base"], default="policy")
ap.add_argument("--rl_mode", choices=["rloo", "candidate"], default="rloo")
ap.add_argument("--anchor", type=float, default=0.0)
ap.add_argument("--sym_anchor", action="store_true")
ap.add_argument("--anchor_mode", choices=["chosen", "onmenu"], default="chosen")
args = ap.parse_args()

CFG = dict(model=args.model, attach="block", seed=0, n_train=2000, n_eval=300,
           n_transfer=150, n_know_train=30, batch=6, lora_r=8, eval_every=25,
           k=4, kl=0.03, pess=0.5, filter_every=10, filter_min_sigma=0.05)

ctx = load_model(CFG["model"])
d = build_data(seed=CFG["seed"], n_train=CFG["n_train"], n_eval=CFG["n_eval"],
               n_transfer=CFG["n_transfer"], n_know_train=CFG["n_know_train"], tok=ctx.tok)
key = f"{CFG['model'].replace('/','_')}_s{CFG['seed']}_{len(d.train_pairs)}_{CFG['attach']}"
Xw_tr, Xr_tr = cache_pairend(ctx, d.train_pairs, attach=CFG["attach"], cache_file=f".f_tr_{key}.npz")
Xw_te, Xr_te = cache_pairend(ctx, d.eval_pairs, attach=CFG["attach"], cache_file=f".f_te_{key}.npz")
layer_acc, layer_elbo, heads = fit_probes(ctx, d, Xw_tr, Xr_tr, Xw_te, Xr_te, cache_file=f".probes_{key}.pt")
L = args.L if args.L is not None else int(next(li for li in range(ctx.n_layers) if layer_acc[li] >= 0.985))
print(f"arm={args.arm} | attach L{L} ({CFG['attach']}) | {ctx.n_layers} blocks", flush=True)

policy = add_lora(ctx, r=CFG["lora_r"])
fh = RewardHead(ctx, heads, L, attach=CFG["attach"])         # trained/co-adapted head
fh_meter = RewardHead(ctx, heads, L, attach=CFG["attach"])   # pristine meter (never filtered)

if args.arm == "deep":
    params = reset_lora(ctx, seed=CFG["seed"] + args.arm_seed, trainable_blocks=list(range(L + 1)))
    low = [p for _, p, b in ctx.lora_params if p.requires_grad]
    up = []
else:
    params = reset_lora(ctx, seed=CFG["seed"] + args.arm_seed)
    low = [p for _, p, b in ctx.lora_params if b <= L]
    up = [p for _, p, b in ctx.lora_params if b > L]
opt_low = torch.optim.AdamW(low, lr=args.lr)
opt_up = torch.optim.AdamW(up, lr=args.lr) if up else None
UPPER = set(range(L + 1, ctx.n_layers))

SYM_ANCHOR = args.sym_anchor
ANCHOR_MODE = args.anchor_mode
PAIR_IDX = {id(p): i for i, p in enumerate(d.train_pairs)}
R_W = torch.special.ndtr(torch.tensor(  # reward for emitting the WRONG side, per train pair
    fh.g(torch.tensor((Xr_tr[:, L] - Xw_tr[:, L]), device=ctx.device).float()).float().cpu()))

def candidate_rl_step(ctx, batch, coef_anchor):
    """Exact-expectation REINFORCE over the pair's two completions (teacher-forced; rewards
       from cached frozen-base features). Backward -> caller restricts to blocks > L."""
    import contextlib
    B = len(batch)
    texts = [p["prompt"] + p["wrong"] for p in batch] + [p["prompt"] + p["right"] for p in batch]
    enc = ctx.tok(texts, return_tensors="pt", padding=True).to(ctx.device)
    nc = [len(p["w_ids"]) for p in batch] + [len(p["r_ids"]) for p in batch]
    lp = _comp_logp(ctx.policy(**enc, logits_to_keep=max(nc) + 1).logits, enc.input_ids, nc)
    lp_w, lp_r = lp[:B], lp[B:]
    pi = torch.softmax(torch.stack([lp_w, lp_r], 1), 1)          # pair-restricted policy
    r_w = torch.stack([R_W[PAIR_IDX[id(p)]] for p in batch]).to(ctx.device)
    r = torch.stack([r_w, torch.full_like(r_w, 0.5)], 1)         # candidate=right baseline 0.5
    loss = -(pi * r).sum(1).mean()
    if coef_anchor > 0:
        with torch.no_grad(), ctx.policy.disable_adapter():
            ref = _comp_logp(ctx.policy(**enc, logits_to_keep=max(nc) + 1).logits, enc.input_ids, nc)
        if ANCHOR_MODE == "onmenu":
            # AB pairs: floor the MENU's total mass log(pi_w + pi_r) at ref; free pairs: chosen DPOP
            is_ab = torch.tensor([p["fmt"] == "ab" for p in batch], device=lp_w.device)
            m = torch.logsumexp(torch.stack([lp_w, lp_r], 1), 1)
            m_ref = torch.logsumexp(torch.stack([ref[:B], ref[B:]], 1), 1)
            hinge = torch.where(is_ab, torch.relu(m_ref - m), torch.relu(ref[:B] - lp_w)).mean()
        else:
            hinge = torch.relu(ref[:B] - lp_w).mean()
            if SYM_ANCHOR: hinge = hinge + torch.relu(ref[B:] - lp_r).mean()
        loss = loss + coef_anchor * hinge
    loss.backward()

rng = random.Random(4242); ctx.policy.train(); curve = []

def checkpoint(step):
    gh = goodhart_state(ctx, d, fh_meter); gh["step"] = step
    ctx.policy.eval()
    o_l = greedy(ctx, [d.render_ab(q) for q in d.eval_qs[:60]], 2)
    gh["fracA"] = float(np.mean([x[:1] == "A" for x in o_l]))
    curve.append(gh); ctx.policy.train()
    print(f"  {step:4d}: ab_flip {gh['ab_flip']:.2f} fracA {gh['fracA']:.2f} | free {gh['free_flip']:.2f} "
          f"offmenu {gh['free_offmenu']:.2f} | meter {gh['head_endorse']:.2f} dpoR {gh['dpo_margin']:+.1f} "
          f"Δlp {gh['dlp_chosen']:+.1f}/{gh['dlp_rejected']:+.1f}", flush=True)

checkpoint(0)
for step in range(args.steps):
    batch = rng.sample(d.train_pairs, CFG["batch"])
    # margin half -> blocks <= L only (opt_low holds only those params)
    opt_low.zero_grad()
    if opt_up: opt_up.zero_grad()
    margin_step(ctx, batch, fh)
    torch.nn.utils.clip_grad_norm_(low, 1.0); opt_low.step()
    # REINFORCE half -> blocks > L only (own_blocks zeroes the rest post-backward)
    if args.arm == "deep_rl":
        opt_up.zero_grad()
        if args.rl_mode == "candidate":
            candidate_rl_step(ctx, batch, args.anchor)
            for _, p_, b_ in ctx.lora_params:
                if b_ not in UPPER: p_.grad = None
        else:
            sampled_rl_step(ctx, batch, fh, k=CFG["k"], kl=CFG["kl"], pess=CFG["pess"],
                            own_blocks=UPPER, score_with=args.rl_read)
        torch.nn.utils.clip_grad_norm_(up, 1.0); opt_up.step()
    # cooperative head co-adaptation ('pref' labels), variance floor
    if (step + 1) % CFG["filter_every"] == 0:
        fb = rng.sample(d.train_pairs, CFG["batch"])
        Xw_f, Xr_f = cache_pairend(ctx, fb, attach=CFG["attach"], use_policy=True)
        t = torch.tensor([-1.0 * p["dir"] for p in fb], device=ctx.device, dtype=torch.float32)
        fh.filter_round(torch.tensor(Xr_f[:, L] - Xw_f[:, L], device=ctx.device), t,
                        min_sigma=CFG["filter_min_sigma"])
    if (step + 1) % CFG["eval_every"] == 0: checkpoint(step + 1)

final = eval_all(ctx, d)
ctx.policy.eval()
o_bt = greedy(ctx, [d.render_ab(q) for q in d.eval_qs], 2)
by_type = {t: float(np.mean([o_bt[i][:1] == ("B" if d.eval_qs[i]["corr"] == "A" else "A")
                             for i, q in enumerate(d.eval_qs) if q["typ"] == t]))
           for t in sorted({q["typ"] for q in d.eval_qs})}
print("BY-TYPE ab_flip:", {k_: round(v, 3) for k_, v in by_type.items()}, flush=True)
print("FINAL:", {k_: round(v, 3) for k_, v in final.items()}, flush=True)
tag = f"hyb_{args.model.split(chr(47))[-1]}_{args.arm}_L{L}_lr{args.lr:g}_{args.rl_read}_{args.rl_mode}_a{args.anchor}{"sym" if args.sym_anchor else ""}{args.anchor_mode}_s{args.arm_seed}"
json.dump(dict(arm=args.arm, L=L, cfg=CFG, lr=args.lr, steps=args.steps, by_type_flip=by_type,
               curve=curve, final=final), open(f"{tag}.json", "w"), indent=1, default=float)
torch.save({n: p.detach().cpu() for n, p, _ in ctx.lora_params}, f"{tag}_lora.pt")
print(f"saved {tag}.json + _lora.pt", flush=True)
