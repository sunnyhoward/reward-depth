#!/usr/bin/env python
"""Stage 0 of the supervisor recipe: HIS chat replay corpus -> EAGLE-L17 head -> K-FAC.

REVISED 2026-08-07. We no longer generate the replay ourselves — he sent `corpus_shards/`
(10000 records, 3.33M scored tokens, schema `chat-kfac-corpus-shards-v1`, already tokenised with
the Qwen3.5 vocab and already in the exact record layout `replay-kfac-ewc` reads). That removes
the last thing standing between us and his setup:

  our 08-05 runs   random 1-8 token prefixes, unconditional continuations, base-model formatting
  this            his own chat-format generative replay, system-profile'd, thinking traces intact

Which matters because results_0805.md §8.1 measured stage 1 moving task text 2.7 nats while
moving that replay .003 — the anchor had nothing to grab. Here the replay and the preference edit
live in the same chat distribution, so both the replay loss and the Fisher see the region that
actually moves.

Every record scores `token_ids[prefix_length:]` (verified: `score_spans` sums to
`length - prefix_length` for all 10000), so the shard convention and the package convention are
the same one.

Writes three artifacts under /workspace/sup/:
  replay_bank.pt        left-padded token ids + per-row scored-span start (replay loss)
  head_tf_L{LAYER}.pt   EAGLE head at LAYER, distilled to base final logits on that replay
  kfac/                 K-FAC factor bundle estimated on that replay, layers 0..LAYER

Env: SUP_MODEL=Qwen/Qwen3.5-2B SUP_LAYER=17 CTX=512 N_BANK=4000 HEAD_STEPS=600
     KFAC_RECORDS=1500 KFAC_MAXLEN=1024 SKIP_KFAC=0
"""
import os, sys, json, glob, random, subprocess
import numpy as np
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

E = os.environ.get
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "eagle"))
sys.path.insert(0, os.path.join(HERE, ".."))
from sup_common import MODEL, DEV, LAYER, SHARDS                                # noqa: E402
from eagle_common import make_head                                              # noqa: E402
from helpers import ResidualCapture                                             # noqa: E402

OUT = "/workspace/sup"
os.makedirs(OUT, exist_ok=True)
CTX = int(E("CTX", 512))
N_BANK = int(E("N_BANK", 4000))
HEAD_STEPS, HEAD_BS, HEAD_LR = int(E("HEAD_STEPS", 600)), int(E("HEAD_BS", 8)), float(E("HEAD_LR", 1e-3))
KFAC_RECORDS, KFAC_MAXLEN = int(E("KFAC_RECORDS", 1500)), int(E("KFAC_MAXLEN", 1024))
SEED = int(E("SEED", 0))
BANK_F = f"{OUT}/replay_bank.pt"
HEAD_F = f"{OUT}/head_tf_L{LAYER}.pt"
KFAC_DIR = f"{OUT}/kfac"
random.seed(SEED); torch.manual_seed(SEED)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
PAD = tok.pad_token_id


def shard_records():
    for f in sorted(glob.glob(f"{SHARDS}/kfac_corpus_*.jsonl")):
        for line in open(f):
            yield json.loads(line)


# ---------- 1. replay bank from his shards ----------
if os.path.exists(BANK_F):
    bank = torch.load(BANK_F)
    print(f"[bank] loaded {tuple(bank['ids'].shape)}", flush=True)
else:
    # Keep the TAIL of each record: the scored span is always the suffix, so truncating from the
    # left preserves it whole (records run 68..36908 tokens; median 462).
    ids_rows, starts = [], []
    for r in shard_records():
        ids, pl = r["token_ids"], r["prefix_length"]
        if len(ids) > CTX:
            cut = len(ids) - CTX
            ids, pl = ids[cut:], max(pl - cut, 1)
        if len(ids) - pl < 8:          # need a few scored tokens to be worth banking
            continue
        npad = CTX - len(ids)
        ids_rows.append([PAD] * npad + ids)     # LEFT pad, matching tok.padding_side
        starts.append(npad + pl)
    order = random.Random(SEED).sample(range(len(ids_rows)), min(N_BANK, len(ids_rows)))
    bank = dict(ids=torch.tensor([ids_rows[i] for i in order], dtype=torch.long),
                start=torch.tensor([starts[i] for i in order], dtype=torch.long), ctx=CTX)
    torch.save(bank, BANK_F)
    n_scored = int((bank["ctx"] - bank["start"]).sum())
    print(f"[bank] wrote {BANK_F} {tuple(bank['ids'].shape)} | {n_scored} scored tokens", flush=True)

IDS, START = bank["ids"], bank["start"]
ntr = int(.9 * len(IDS))
train_idx, held_idx = torch.arange(ntr), torch.arange(ntr, len(IDS))


def rbatch(k, idx):
    sel = idx[torch.randint(0, len(idx), (k,))]
    x = IDS[sel].to(DEV)
    return dict(input_ids=x, attention_mask=(x != PAD).long())


# ---------- 2. EAGLE head at LAYER, distilled on that replay ----------
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
for p in model.parameters():
    p.requires_grad_(False)
model.config.use_cache = False
BLOCKS = list(model.model.layers)
HID, NL = model.config.get_text_config().hidden_size, len(BLOCKS)
assert LAYER < NL, f"LAYER {LAYER} >= depth {NL}"
print(f"[sup-prepare] {MODEL} layers={NL} hid={HID} EAGLE layer={LAYER}", flush=True)

POS_PER_STEP = int(E("POS_PER_STEP", 256))


def head_hidden(head, h, pad_mask):
    """EagleTfHead.forward up to — but not including — norm+lm_head.

    Qwen3.5's vocabulary is 248320, so a full (B, T, V) fp32 log-softmax is 4 GB at B=8, T=512
    and OOMs against the two decodability jobs already on this GPU. Splitting the projection off
    lets us take the KL on a random subsample of positions instead: the head's transform still
    sees the whole sequence (its attention needs to), only the unembedding is subsampled.
    """
    B, T, C = h.shape
    x = head.n1(h)
    q = head.q(x).view(B, T, head.nh, head.hd).transpose(1, 2)
    k = head.k(x).view(B, T, head.nh, head.hd).transpose(1, 2)
    v = head.v(x).view(B, T, head.nh, head.hd).transpose(1, 2)
    causal = torch.ones(T, T, dtype=torch.bool, device=h.device).tril()
    m = causal[None, None] & pad_mask[:, None, None, :].bool()
    m = m | ~m.any(-1, keepdim=True)
    a = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
    h = h + head.o(a.transpose(1, 2).reshape(B, T, C))
    return h + head.fc2(F.silu(head.fc1(head.n2(h))))


def head_pair(head, enc, npos, gen):
    """(teacher_logits, student_logits) at `npos` random valid next-token positions."""
    am = enc["attention_mask"][:, 1:].bool()
    with torch.no_grad(), ResidualCapture([BLOCKS[LAYER]]) as cap:
        hf = model.model(**enc).last_hidden_state[:, :-1]        # post final norm
    hl = cap.get()[0][:, :-1]
    hs = head_hidden(head, hl, enc["attention_mask"][:, :-1])
    flat = am.reshape(-1).nonzero(as_tuple=True)[0]
    idx = flat[torch.randint(0, len(flat), (min(npos, len(flat)),), generator=gen, device=DEV)]
    C = hf.shape[-1]
    t = model.lm_head(hf.reshape(-1, C)[idx]).float()
    s = model.lm_head(model.model.norm(hs.reshape(-1, C)[idx])).float()
    return t, s


if os.path.exists(HEAD_F):
    print(f"[head] exists {HEAD_F}", flush=True)
else:
    gen = torch.Generator(device=DEV); gen.manual_seed(SEED)
    head = make_head(HID, "tf").to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=HEAD_LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=HEAD_STEPS, eta_min=HEAD_LR / 20)
    run = []
    for step in range(HEAD_STEPS):
        enc = rbatch(HEAD_BS, train_idx)
        t, s = head_pair(head, enc, POS_PER_STEP, gen)
        t, s = F.log_softmax(t, -1), F.log_softmax(s, -1)
        kl = (t.exp() * (t - s)).sum(-1).mean()
        opt.zero_grad(); kl.backward(); opt.step(); sch.step()
        run.append(float(kl.detach()))
        if (step + 1) % 200 == 0:
            print(f"  head step {step+1}: kl {np.mean(run[-200:]):.4f} "
                  f"(lr {sch.get_last_lr()[0]:.2e})", flush=True)
    with torch.no_grad():
        agr, n, kls = 0.0, 0, []
        for _ in range(8):
            enc = rbatch(4, held_idx)
            t, s = head_pair(head, enc, 512, gen)
            agr += float((t.argmax(-1) == s.argmax(-1)).sum()); n += t.shape[0]
            tl, sl = F.log_softmax(t, -1), F.log_softmax(s, -1)
            kls.append(float((tl.exp() * (tl - sl)).sum(-1).mean()))
        agree, kl_held = agr / n, float(np.mean(kls))
    print(f"[head] held-out KL {kl_held:.4f}", flush=True)
    torch.save(head.state_dict(), HEAD_F)
    json.dump(dict(model=MODEL, layer=LAYER, agreement=agree, kl_heldout=kl_held,
                   kl_last_train=float(kl), ctx=CTX, bank=list(IDS.shape)),
              open(f"{OUT}/head.json", "w"), indent=1)
    print(f"[head] wrote {HEAD_F} | held-out top-1 agreement {agree:.3f}", flush=True)

# ---------- 3. K-FAC factors on the SAME replay ----------
if int(E("SKIP_KFAC", 0)) or os.path.isdir(KFAC_DIR):
    print(f"[kfac] {'skipped' if int(E('SKIP_KFAC',0)) else 'exists'}", flush=True)
else:
    del model
    torch.cuda.empty_cache()
    # NOTE.md flagged "K-FAC targets are attention-only here to keep factor estimation cheap".
    # That default is wrong for this model: Qwen3.5-2B is hybrid, so q/k/v/o exist ONLY at layers
    # 3/7/11/15/19/23 — attention-only would leave 14 of stage 1's 18 LoRA'd layers with no
    # curvature at all, and a leash that covers 4/18 of what moves is not a leash. Target every
    # module the stage-1 LoRA touches, restricted to layers 0..LAYER (stage 2 needs its own
    # bundle over LAYER+1..top).
    import re as _re
    from transformers import AutoConfig
    _cfg = AutoConfig.from_pretrained(MODEL).get_text_config()
    tgts = []
    for L in range(LAYER + 1):
        for m in ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
            tgts.append(f"model.layers.{L}.{m}")
        if _cfg.layer_types[L] == "full_attention":
            for m in ("q_proj", "k_proj", "v_proj", "o_proj"):
                tgts.append(f"model.layers.{L}.self_attn.{m}")

    jl = f"{OUT}/kfac_corpus.jsonl"
    if not os.path.exists(jl):
        keep = [r for r in shard_records() if r["length"] <= KFAC_MAXLEN]
        random.Random(SEED).shuffle(keep)
        keep = keep[:KFAC_RECORDS]
        with open(jl, "w") as f:
            for i, r in enumerate(keep):
                r = dict(r); r["seq_id"] = i; r.pop("score_spans", None); r["split"] = "train"
                f.write(json.dumps(r) + "\n")
        print(f"[kfac] corpus {len(keep)} records (<= {KFAC_MAXLEN} tok), "
              f"{sum(r['length'] - r['prefix_length'] for r in keep)} scored tokens", flush=True)

    cmd = ["replay-kfac-ewc", "estimate", "--model", MODEL, "--corpus", jl,
           "--output", KFAC_DIR, "--batch-size", E("KFAC_BS", "2"), "--device", "cuda:0",
           "--placement", E("KFAC_PLACEMENT", "model"), "--eigh-device", "cuda:0",
           "--max-dense-gib", E("KFAC_GIB", "20")]
    for t in tgts:
        cmd += ["--target", t]
    print(f"[kfac] {len(tgts)} target modules over layers 0..{LAYER}", flush=True)
    subprocess.run(cmd, check=True)
    print(f"[kfac] wrote {KFAC_DIR}", flush=True)

print("\nDONE — sup_train.py next", flush=True)
