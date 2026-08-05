#!/usr/bin/env python
"""EAGLE tf heads for the refusal-transfer testbed, on Qwen3-4B-Base (2026-08-05).

Standalone rather than a call into eagle/eagle_head.py for one concrete reason: eagle_common's
`head_path()` is /workspace/eagle_head_{arch}_L{L}.pt with NO model in the name, so distilling
Qwen3 heads through it would silently overwrite the Qwen2.5-3B heads the styc/brit line uses —
and since hidden size differs (2560 vs 2048) the clobber would surface later as a shape error in
an unrelated run. Heads here go to /workspace/refusal/head_tf_L{L}.pt.

Replay-distilled, per the 08-04 direction: the head must be a competent GENERAL readout before
it is frozen for stage 1, so the DPO margin can only move via layers 0..L. §13 measured the
alternative — a head distilled on narrow task text is sharper (agreement .836 vs .283) and makes
stage 1 WORSE, driving the lower stack hard enough to break the computation.

Env: LAYERS=4,12,24 STEPS=400 BATCH=16 LR=1e-3 SEED=0 N_REPLAY=2048 T_REPLAY=128
     MODEL=Qwen/Qwen3-4B-Base
Out: /workspace/refusal/head_tf_L{L}.pt  +  /workspace/refusal/heads.json
"""
import os, sys, json, random
import numpy as np
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

E = os.environ.get
os.environ.setdefault("MODEL", "Qwen/Qwen3-4B-Base")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eagle"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eagle_common import make_head, MODEL, DEV          # noqa: E402
from helpers import ResidualCapture                      # noqa: E402

LAYERS = [int(x) for x in E("LAYERS", "4,12,24").split(",")]
STEPS, BATCH = int(E("STEPS", 400)), int(E("BATCH", 16))
LR, SEED = float(E("LR", 1e-3)), int(E("SEED", 0))
N_REPLAY, T_REPLAY = int(E("N_REPLAY", 2048)), int(E("T_REPLAY", 128))
ARCH = "tf"
OUTDIR = "/workspace/refusal"
os.makedirs(OUTDIR, exist_ok=True)
REPLAY_F = f"{OUTDIR}/replay_{N_REPLAY}x{T_REPLAY}.pt"

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
for p in model.parameters():
    p.requires_grad_(False)
BLOCKS = list(model.model.layers)
HID = model.config.hidden_size
print(f"[refusal-head] {MODEL} layers={len(BLOCKS)} hid={HID} targets={LAYERS}", flush=True)

# ---- replay corpus: the frozen model's own samples from short random prefixes ----
if os.path.exists(REPLAY_F):
    replay = torch.load(REPLAY_F).long()
    print(f"[replay] loaded {tuple(replay.shape)} from {REPLAY_F}", flush=True)
else:
    print(f"[replay] sampling {N_REPLAY}x{T_REPLAY} from {MODEL}", flush=True)
    special = set(tok.all_special_ids)
    vocab_ok = [i for i in range(0, min(tok.vocab_size, model.config.vocab_size)) if i not in special]
    rows, BS = [], 64
    g = torch.Generator(device="cpu").manual_seed(SEED)
    while len(rows) < N_REPLAY:
        k = min(BS, N_REPLAY - len(rows))
        plen = int(torch.randint(1, 9, (1,), generator=g).item())
        idx = torch.randint(0, len(vocab_ok), (k, plen), generator=g)
        prefix = torch.tensor([[vocab_ok[int(j)] for j in row] for row in idx], device=DEV)
        with torch.no_grad():
            out = model.generate(input_ids=prefix,
                                 attention_mask=torch.ones_like(prefix),
                                 do_sample=True, temperature=1.0, top_p=1.0,
                                 min_new_tokens=T_REPLAY - plen,
                                 max_new_tokens=T_REPLAY - plen,
                                 pad_token_id=tok.pad_token_id)
        rows.extend(out[:, :T_REPLAY].cpu().tolist())
        print(f"  {len(rows)}/{N_REPLAY}", flush=True)
    replay = torch.tensor(rows[:N_REPLAY], dtype=torch.long)
    torch.save(replay, REPLAY_F)
    print(f"[replay] wrote {REPLAY_F}", flush=True)

train_pool, held_pool = replay[: int(0.9 * len(replay))], replay[int(0.9 * len(replay)):]


def replay_batch(k, pool):
    ids = pool[torch.randint(0, pool.shape[0], (k,))].to(DEV)
    return dict(input_ids=ids, attention_mask=torch.ones_like(ids))


heads = {L: make_head(HID, ARCH).to(DEV) for L in LAYERS}
opts = {L: torch.optim.AdamW(heads[L].parameters(), lr=LR) for L in LAYERS}
print(f"[heads] {sum(p.numel() for p in heads[LAYERS[0]].parameters())/1e6:.1f}M params each",
      flush=True)

hist = {L: [] for L in LAYERS}
for step in range(STEPS):
    enc = replay_batch(BATCH, train_pool)
    am = torch.ones_like(enc["input_ids"][:, 1:]).bool()
    with torch.no_grad(), ResidualCapture([BLOCKS[L] for L in LAYERS]) as cap:
        t_lsm = F.log_softmax(model(**enc).logits[:, :-1].float(), -1)
    bufs = cap.get()
    for k, L in enumerate(LAYERS):
        s_lsm = F.log_softmax(heads[L](bufs[k][:, :-1], model,
                                       pad_mask=enc["attention_mask"][:, :-1]), -1)
        kl = ((t_lsm.exp() * (t_lsm - s_lsm)).sum(-1) * am).sum() / am.sum()
        opts[L].zero_grad(); kl.backward(); opts[L].step()
        hist[L].append(float(kl.detach()))
    if (step + 1) % 25 == 0:
        print(f"  step {step+1:4d}: " +
              " ".join(f"L{L} kl={np.mean(hist[L][-25:]):.3f}" for L in LAYERS), flush=True)

res = dict(model=MODEL, layers=LAYERS, steps=STEPS, arch=ARCH, hid=HID)
with torch.no_grad():
    enc = replay_batch(64, held_pool)
    am = torch.ones_like(enc["input_ids"][:, 1:]).bool()
    with ResidualCapture([BLOCKS[L] for L in LAYERS]) as cap:
        t_arg = model(**enc).logits[:, :-1].argmax(-1)
    bufs = cap.get()
    for k, L in enumerate(LAYERS):
        s_arg = heads[L](bufs[k][:, :-1], model,
                         pad_mask=enc["attention_mask"][:, :-1]).argmax(-1)
        agree = float(((s_arg == t_arg) & am).sum() / am.sum())
        res[f"agree_L{L}"] = agree
        res[f"kl_final_L{L}"] = float(np.mean(hist[L][-25:]))
        torch.save(heads[L].state_dict(), f"{OUTDIR}/head_tf_L{L}.pt")
        print(f"[head L{L}] held-out top-1 agreement with base: {agree:.3f}", flush=True)

json.dump(res, open(f"{OUTDIR}/heads.json", "w"), indent=1)
print("NOTE: head competence co-varies with depth (§17 standing confound) — carry these "
      "agreement numbers into any depth comparison.", flush=True)
print("DONE", flush=True)
