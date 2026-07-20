import numpy as np, torch, random, matplotlib.pyplot as plt
from helpers import *

CFG = dict(
    model      = 'Qwen/Qwen2.5-3B',
    attach     = 'final',      # 'final': post-norm read (phase 1) | 'block' (phase 2)
    L          = None,         # None → last layer; set lower for phase 2
    seed       = 0,
    n_train    = 2000, n_eval = 300, n_transfer = 150, n_know_train = 30,
    steps      = 300, batch = 6, lr_probe = 1e-4, lr_dpo = 5e-5, lora_r = 8,
    eval_every = 25, gradcos_batches = 8,
    # online Bayesian head co-adaptation during the probe arm (off by default):
    # 'pref' = cooperative (assimilate the trained preference), 'truth' = adversarial monitor;
    # min_sigma floors the posterior variance (co-adaptation otherwise disarms pessimism)
    filter = True, filter_every = 10, filter_labels = 'pref', filter_min_sigma = 0.05,
)



# ===== cell =====
ctx = load_model(CFG['model'])
L = CFG['L'] if CFG['L'] is not None else ctx.n_layers - 1
d = build_data(seed=CFG['seed'], n_train=CFG['n_train'], n_eval=CFG['n_eval'],
               n_transfer=CFG['n_transfer'], n_know_train=CFG['n_know_train'], tok=ctx.tok)
print(f'{ctx.n_layers} blocks | head at L={L} ({CFG["attach"]}) | {len(d.train_pairs)} train pairs')
d.train_pairs[0]['prompt'][-120:], d.train_pairs[0]['wrong'], d.train_pairs[0]['right']


# ===== cell =====
# completion-end features (disk-cached) + per-layer probes — the decodability curve
key = f"{CFG['model'].replace('/','_')}_s{CFG['seed']}_{len(d.train_pairs)}_{CFG['attach']}"
Xw_tr, Xr_tr = cache_pairend(ctx, d.train_pairs, attach=CFG['attach'], cache_file=f'.f_tr_{key}.npz')
Xw_te, Xr_te = cache_pairend(ctx, d.eval_pairs,  attach=CFG['attach'], cache_file=f'.f_te_{key}.npz')
layer_acc, layer_elbo, heads = fit_probes(ctx, d, Xw_tr, Xr_tr, Xw_te, Xr_te,
                                          cache_file=f'.probes_{key}.pt')
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
ax[0].plot(layer_acc, 'o-'); ax[0].axhline(.5, ls=':', c='gray'); ax[0].set_title('probe acc vs layer')
ax[1].plot(layer_elbo, 's-'); ax[1].set_title('ELBO (evidence proxy)'); plt.savefig("decodability.png"); plt.close()


# ===== cell =====
policy = add_lora(ctx, r=CFG['lora_r'])
fh = RewardHead(ctx, heads, L, attach=CFG['attach'])
GC_BATCHES = [random.Random(777).sample(d.train_pairs, CFG['batch']) for _ in range(CFG['gradcos_batches'])]

def run_arm(mode, lr, anchor=0.0, seed=1):
    params = reset_lora(ctx, seed=CFG['seed'] + seed)
    opt = torch.optim.AdamW(params, lr=lr)
    rng = random.Random(4242); ctx.policy.train(); curve = []
    def checkpoint(step):
        gh = goodhart_state(ctx, d, fh); gh['step'] = step
        if mode == 'probe': gh['gradcos'] = grad_cos_vs_dpo(ctx, fh, GC_BATCHES)
        curve.append(gh); ctx.policy.train()
        print(f"  {step:4d}: ab_flip {gh['ab_flip']:.2f} | free flip {gh['free_flip']:.2f} "
              f"offmenu {gh['free_offmenu']:.2f} | head {gh['head_endorse']:.2f} "
              f"dpoR {gh['dpo_margin']:+.1f} Δlp(cho/rej) {gh['dlp_chosen']:+.1f}/{gh['dlp_rejected']:+.1f}", flush=True)
    checkpoint(0)
    for step in range(CFG['steps']):
        batch = rng.sample(d.train_pairs, CFG['batch'])
        opt.zero_grad()
        if mode == 'probe': margin_step(ctx, batch, fh, anchor=anchor)
        else:               dpo_step(ctx, batch)
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if CFG['filter'] and mode == 'probe' and (step + 1) % CFG['filter_every'] == 0:
            fb = rng.sample(d.train_pairs, CFG['batch'])
            Xw_f, Xr_f = cache_pairend(ctx, fb, attach=CFG['attach'], use_policy=True)
            sgn = -1.0 if CFG['filter_labels'] == 'pref' else 1.0
            t = torch.tensor([sgn * p['dir'] for p in fb], device=ctx.device, dtype=torch.float32)
            fh.filter_round(torch.tensor(Xr_f[:, L] - Xw_f[:, L], device=ctx.device), t,
                            min_sigma=CFG['filter_min_sigma'])
        if (step + 1) % CFG['eval_every'] == 0: checkpoint(step + 1)
    return dict(mode=mode, curve=curve, final=eval_all(ctx, d), rollouts=rollouts(ctx, d))


# ===== dpo arm only =====
res_dpo = run_arm('dpo', CFG['lr_dpo'])
import json as _j
_j.dump(dict(final=res_dpo['final'], curve=res_dpo['curve']), open('verbatim_dpo_result.json','w'), default=float)
print('FINAL:', {k: round(v,3) for k,v in res_dpo['final'].items()})
