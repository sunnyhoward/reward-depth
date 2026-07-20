"""Phase-2 pilot: RL-from-probe (RLOO, frozen-base reader) with the head attached at the
EARLIEST layer where the preference is linearly decodable (the decodability elbow),
instead of at the top. Mirrors the phase-1 rl arm (score_with='base', k=4, kl=0.03,
pess=0.5) with attach='block' and L = elbow. Usage:

    python rl_at_elbow.py [--L N] [--steps 300] [--lr 1e-4]

Writes curve + final evals to rl_elbow_L{L}.json and prints progress."""
import argparse, json, random
import numpy as np, torch
from helpers import *
import torch.nn.functional as F
from helpers import _comp_logp

def anchor_step(ctx, batch, coef):
    """DPOP hinge on the chosen side: penalize lp(wrong) falling below the reference."""
    texts = [p["prompt"] + p["wrong"] for p in batch]
    enc = ctx.tok(texts, return_tensors="pt", padding=True).to(ctx.device)
    n_w = [len(p["w_ids"]) for p in batch]
    keep = max(n_w) + 1
    lp_w = _comp_logp(ctx.policy(**enc, logits_to_keep=keep).logits, enc.input_ids, n_w)
    with torch.no_grad(), ctx.policy.disable_adapter():
        ref = _comp_logp(ctx.policy(**enc, logits_to_keep=keep).logits, enc.input_ids, n_w)
    (coef * F.relu(ref - lp_w).mean()).backward()

ap = argparse.ArgumentParser()
ap.add_argument("--L", type=int, default=None, help="attach layer (default: auto-elbow)")
ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--acc_thresh", type=float, default=0.985)
ap.add_argument("--pess", type=float, default=0.5)
ap.add_argument("--attach", choices=["block", "final"], default="block")
ap.add_argument("--anchor", type=float, default=0.0, help="DPOP-style absolute-mass anchor on the chosen (wrong) side")
ap.add_argument("--mode", choices=["rl", "dpo"], default="rl")
ap.add_argument("--arm_seed", type=int, default=1)
args = ap.parse_args()

CFG = dict(model="Qwen/Qwen2.5-3B", seed=0,
           n_train=2000, n_eval=300, n_transfer=150, n_know_train=30,
           batch=6, lora_r=8, eval_every=25, k=4, kl=0.03)

ctx = load_model(CFG["model"])
d = build_data(seed=CFG["seed"], n_train=CFG["n_train"], n_eval=CFG["n_eval"],
               n_transfer=CFG["n_transfer"], n_know_train=CFG["n_know_train"], tok=ctx.tok)
print(f"{ctx.n_layers} blocks | {len(d.train_pairs)} train pairs", flush=True)

key = f"{CFG['model'].replace('/','_')}_s{CFG['seed']}_{len(d.train_pairs)}_{args.attach}"
Xw_tr, Xr_tr = cache_pairend(ctx, d.train_pairs, attach=args.attach, cache_file=f".f_tr_{key}.npz")
Xw_te, Xr_te = cache_pairend(ctx, d.eval_pairs, attach=args.attach, cache_file=f".f_te_{key}.npz")
layer_acc, layer_elbo, heads = fit_probes(ctx, d, Xw_tr, Xr_tr, Xw_te, Xr_te,
                                          cache_file=f".probes_{key}.pt")

elbow = int(next(li for li in range(ctx.n_layers) if layer_acc[li] >= args.acc_thresh))
L = args.L if args.L is not None else elbow
print(f"decodability elbow (first acc >= {args.acc_thresh}): L{elbow} "
      f"(acc {layer_acc[elbow]:.3f}) | attaching at L{L}", flush=True)

policy = add_lora(ctx, r=CFG["lora_r"])
fh = RewardHead(ctx, heads, L, attach=args.attach)
params = reset_lora(ctx, seed=CFG["seed"] + args.arm_seed)
opt = torch.optim.AdamW(params, lr=args.lr)
rng = random.Random(4242); ctx.policy.train(); curve = []

def checkpoint(step):
    gh = goodhart_state(ctx, d, fh); gh["step"] = step
    ctx.policy.eval()
    o_l = greedy(ctx, [d.render_ab(q) for q in d.eval_qs[:60]], 2)
    gh["fracA"] = float(np.mean([x[:1] == "A" for x in o_l]))
    curve.append(gh); ctx.policy.train()
    print(f"  {step:4d}: ab_flip {gh['ab_flip']:.2f} fracA {gh['fracA']:.2f} | free flip {gh['free_flip']:.2f} "
          f"offmenu {gh['free_offmenu']:.2f} | head {gh['head_endorse']:.2f} "
          f"dpoR {gh['dpo_margin']:+.1f} Δlp(cho/rej) {gh['dlp_chosen']:+.1f}/{gh['dlp_rejected']:+.1f}",
          flush=True)

checkpoint(0)
for step in range(args.steps):
    batch = rng.sample(d.train_pairs, CFG["batch"])
    opt.zero_grad()
    if args.mode == "dpo":
        dpo_step(ctx, batch)
    else:
        sampled_rl_step(ctx, batch, fh, k=CFG["k"], kl=CFG["kl"], pess=args.pess, score_with="base")
        if args.anchor > 0: anchor_step(ctx, batch, args.anchor)
    torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
    if (step + 1) % CFG["eval_every"] == 0: checkpoint(step + 1)

final = eval_all(ctx, d)
ctx.policy.eval()
o_bt = greedy(ctx, [d.render_ab(q) for q in d.eval_qs], 2)
by_type = {}
for t in sorted({q["typ"] for q in d.eval_qs}):
    idx = [i for i, q in enumerate(d.eval_qs) if q["typ"] == t]
    by_type[t] = float(np.mean([o_bt[i][:1] == ("B" if d.eval_qs[i]["corr"] == "A" else "A") for i in idx]))
print("BY-TYPE ab_flip:", {k_: round(v, 3) for k_, v in by_type.items()}, flush=True)
print("FINAL:", {k_: round(v, 3) for k_, v in final.items()}, flush=True)
out = dict(L=L, elbow=elbow, pess=args.pess, anchor=args.anchor, mode=args.mode, attach=args.attach,
           arm_seed=args.arm_seed, by_type_flip=by_type, layer_acc=[float(a) for a in layer_acc], cfg=CFG,
           lr=args.lr, steps=args.steps, curve=curve, final=final)
tag = f"{args.mode}_{args.attach}_L{L}_p{args.pess}_a{args.anchor}_lr{args.lr:g}_s{args.arm_seed}"
with open(f"run_{tag}.json", "w") as fjs:
    json.dump(out, fjs, indent=1, default=float)
torch.save({n: p.detach().cpu() for n, p, _ in ctx.lora_params}, f"run_{tag}_lora.pt")
print(f"saved run_{tag}.json + _lora.pt", flush=True)
