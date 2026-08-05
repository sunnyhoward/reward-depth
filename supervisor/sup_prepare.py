#!/usr/bin/env python
"""Stage 0 of the supervisor recipe: chat-format generative replay -> EAGLE-L17 head -> K-FAC.

THE ONE THING THIS FIXES relative to our runs. Our replay corpus was continuations from random
1-8 token prefixes — essentially unconditional sampling — while the task text was
`"Question: ...\\nAnswer:"`. Measured 2026-08-05: stage-1 moved task text 2.7 nats and replay
.003, so the replay anchor had nothing to constrain and the K-FAC factors estimated on that same
corpus measured curvature in directions the edit never travels. Here the replay is sampled from
the model's OWN chat distribution using held-out prompts in the same template the preference edit
lives in, so both the replay loss and the Fisher see the region that actually moves.

Writes three artifacts:
  replay_{N}x{T}.pt          chat-format replay token ids (for the replay loss + K-FAC corpus)
  head_tf_L{LAYER}.pt        EAGLE head at LAYER, distilled to base final logits on that replay
  kfac/                      K-FAC factor bundle estimated on that replay

Env: SUP_MODEL=Qwen/Qwen3.5-2B SUP_LAYER=17 N_REPLAY=1024 T_REPLAY=160 HEAD_STEPS=600
     SUP_KFAC_TARGETS=q_proj,k_proj,v_proj,o_proj SKIP_KFAC=0
Out: /workspace/sup/
"""
import os, sys, json, random, subprocess
import numpy as np
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

E = os.environ.get
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "eagle"))
sys.path.insert(0, os.path.join(HERE, ".."))
from sup_common import MODEL, DEV, LAYER, load_split, render                 # noqa: E402
from eagle_common import make_head                                            # noqa: E402
from helpers import ResidualCapture                                           # noqa: E402

OUT = "/workspace/sup"
os.makedirs(OUT, exist_ok=True)
N_REPLAY, T_REPLAY = int(E("N_REPLAY", 1024)), int(E("T_REPLAY", 160))
HEAD_STEPS, HEAD_BS, HEAD_LR = int(E("HEAD_STEPS", 600)), int(E("HEAD_BS", 8)), float(E("HEAD_LR", 1e-3))
SEED = int(E("SEED", 0))
REPLAY_F = f"{OUT}/replay_{N_REPLAY}x{T_REPLAY}.pt"
HEAD_F = f"{OUT}/head_tf_L{LAYER}.pt"
KFAC_DIR = f"{OUT}/kfac"
random.seed(SEED); torch.manual_seed(SEED)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
for p in model.parameters():
    p.requires_grad_(False)
BLOCKS = list(model.model.layers)
HID, NL = model.config.hidden_size, len(BLOCKS)
assert LAYER < NL, f"LAYER {LAYER} >= depth {NL}"
print(f"[sup-prepare] {MODEL} layers={NL} hid={HID} EAGLE layer={LAYER}", flush=True)

# ---------- 1. chat-format generative replay ----------
if os.path.exists(REPLAY_F):
    replay = torch.load(REPLAY_F).long()
    print(f"[replay] loaded {tuple(replay.shape)}", flush=True)
else:
    # Prompts come from the TRAIN split's passages, so the replay shares the operating
    # distribution of the preference edit. Continuations are the model's own samples (T=1.0),
    # which is what makes this generative replay rather than dataset replay.
    rows = load_split("train")
    rgen = random.Random(SEED)
    prompts = [r["prompt"] for r in rows]
    rgen.shuffle(prompts)
    reps, BS = [], 16
    while len(reps) < N_REPLAY:
        chunk = [prompts[(len(reps) + i) % len(prompts)] for i in range(BS)]
        enc = tok([render(tok, p) for p in chunk], return_tensors="pt", padding=True,
                  truncation=True, max_length=192).to(DEV)
        with torch.no_grad():
            model.config.use_cache = True
            g = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0,
                               min_new_tokens=T_REPLAY // 2, max_new_tokens=T_REPLAY,
                               pad_token_id=tok.pad_token_id)
            model.config.use_cache = False
        for i in range(g.shape[0]):
            seq = g[i][-(T_REPLAY + 64):]
            reps.append(seq[:T_REPLAY + 64].tolist() if len(seq) >= T_REPLAY + 64
                        else (seq.tolist() + [tok.pad_token_id] * (T_REPLAY + 64 - len(seq))))
        print(f"  replay {len(reps)}/{N_REPLAY}", flush=True)
    replay = torch.tensor(reps[:N_REPLAY], dtype=torch.long)
    torch.save(replay, REPLAY_F)
    print(f"[replay] wrote {REPLAY_F} {tuple(replay.shape)}", flush=True)

train_pool = replay[: int(.9 * len(replay))]
held_pool = replay[int(.9 * len(replay)):]


def rbatch(k, pool):
    ids = pool[torch.randint(0, pool.shape[0], (k,))].to(DEV)
    return dict(input_ids=ids, attention_mask=(ids != tok.pad_token_id).long())


# ---------- 2. EAGLE head at LAYER, distilled on that replay ----------
if os.path.exists(HEAD_F):
    print(f"[head] exists {HEAD_F}", flush=True)
else:
    head = make_head(HID, "tf").to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=HEAD_LR)
    for step in range(HEAD_STEPS):
        enc = rbatch(HEAD_BS, train_pool)
        am = enc["attention_mask"][:, 1:].bool()
        with torch.no_grad(), ResidualCapture([BLOCKS[LAYER]]) as cap:
            t = F.log_softmax(model(**enc).logits[:, :-1].float(), -1)
        s = F.log_softmax(head(cap.get()[0][:, :-1], model,
                               pad_mask=enc["attention_mask"][:, :-1]), -1)
        kl = ((t.exp() * (t - s)).sum(-1) * am).sum() / am.sum()
        opt.zero_grad(); kl.backward(); opt.step()
        if (step + 1) % 100 == 0:
            print(f"  head step {step+1}: kl {float(kl):.4f}", flush=True)
    with torch.no_grad():
        enc = rbatch(32, held_pool)
        am = enc["attention_mask"][:, 1:].bool()
        with ResidualCapture([BLOCKS[LAYER]]) as cap:
            ta = model(**enc).logits[:, :-1].argmax(-1)
        sa = head(cap.get()[0][:, :-1], model, pad_mask=enc["attention_mask"][:, :-1]).argmax(-1)
        agree = float(((sa == ta) & am).sum() / am.sum())
    torch.save(head.state_dict(), HEAD_F)
    json.dump(dict(model=MODEL, layer=LAYER, agreement=agree, kl=float(kl)),
              open(f"{OUT}/head.json", "w"), indent=1)
    print(f"[head] wrote {HEAD_F} | held-out top-1 agreement {agree:.3f}", flush=True)

# ---------- 3. K-FAC factors on the SAME replay ----------
if int(E("SKIP_KFAC", 0)) or os.path.isdir(KFAC_DIR):
    print(f"[kfac] {'skipped' if int(E('SKIP_KFAC',0)) else 'exists'}", flush=True)
else:
    del model
    torch.cuda.empty_cache()
    jl = f"{OUT}/replay_corpus.jsonl"
    with open(jl, "w") as f:
        for i, row in enumerate(replay.tolist()):
            ids = [t for t in row if t != tok.pad_token_id]
            f.write(json.dumps(dict(schema_version=1, seq_id=i, token_ids=ids,
                                    prefix_ids=ids[:8], prefix_length=8, length=len(ids),
                                    sampling_seed=SEED, seeding_mode="prompt", split="train",
                                    mean_logprob=0.0, has_repeat_loop=False,
                                    unique_token_count=len(set(ids)))) + "\n")
    tg = E("SUP_KFAC_TARGETS", "q_proj,k_proj,v_proj,o_proj").split(",")
    cmd = ["replay-kfac-ewc", "estimate", "--model", MODEL, "--corpus", jl,
           "--output", KFAC_DIR, "--batch-size", "4", "--device", "cuda:0",
           "--placement", "model", "--eigh-device", "cuda:0", "--max-dense-gib", "60"]
    for t in tg:
        cmd += ["--target", t.strip()]
    print("[kfac] " + " ".join(cmd[:8]) + " ...", flush=True)
    subprocess.run(cmd, check=True)
    print(f"[kfac] wrote {KFAC_DIR}", flush=True)

print("\nDONE — sup_train.py next", flush=True)
