#!/usr/bin/env python
"""Preflight for the layer-restricted DPO study: prove the lower/upper windows are parameter-matched
BEFORE spending GPU time, and fail loudly if they are not.

Builds the real model on the meta device (no weights allocated, exact shapes), applies each
LoraConfig, and counts trainable parameters. `lower` (blocks 0-15) and `upper` (blocks 16-31) must
come out exactly equal -- same rank, same target module types, same number of blocks. If they do
not, the primary comparison is confounded and nothing downstream is interpretable.

Also asserts the unembedding and embeddings are untouched in every condition (target_modules covers
attention/MLP projections only), since "upper layers and the unembedding are frozen" is the claim
the `lower` arm is meant to test.

Exit code 0 = matched, safe to run. Non-zero = do not run.
Env: UF_SFT_MODEL, LORA_R=16"""
import os, sys
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

E = os.environ.get
MODEL = E("UF_SFT_MODEL", "allenai/Llama-3.1-Tulu-3-8B-SFT")
R = int(E("LORA_R", 16))
TM = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

def _layer_spec(s):
    if not s or s == "all": return None
    a, _, b = s.partition("-")
    return list(range(int(a), int(b) + 1)) if b else [int(a)]

cfg = AutoConfig.from_pretrained(MODEL)
NL = cfg.num_hidden_layers
print(f"[model] {MODEL} | {NL} layers | hidden {cfg.hidden_size} | vocab {cfg.vocab_size}")

CONDS = {"lower": "0-15", "upper": f"16-{NL-1}", "full": "all"}
res = {}
for name, spec in CONDS.items():
    with torch.device("meta"):
        m = AutoModelForCausalLM.from_config(cfg)
    p = get_peft_model(m, LoraConfig(r=R, lora_alpha=2 * R, lora_dropout=0.0, bias="none",
                                     task_type="CAUSAL_LM", layers_to_transform=_layer_spec(spec),
                                     target_modules=TM))
    per_block = {}
    other = 0
    for n, q in p.named_parameters():
        if not q.requires_grad: continue
        if "layers." in n:
            bi = int(n.split("layers.")[1].split(".")[0])
            per_block[bi] = per_block.get(bi, 0) + q.numel()
        else:
            other += q.numel()
            print(f"  !! trainable OUTSIDE a block: {n} ({q.numel()})")
    blocks = sorted(per_block)
    total = sum(per_block.values()) + other
    # every adapted block must carry an identical parameter count, else "16 blocks" is not a
    # meaningful unit of comparison
    counts = set(per_block.values())
    assert len(counts) == 1, f"{name}: adapted blocks differ in size: {sorted(counts)}"
    assert other == 0, f"{name}: {other} trainable params outside transformer blocks (embed/lm_head?)"
    res[name] = dict(spec=spec, blocks=blocks, n_blocks=len(blocks),
                     per_block=counts.pop(), total=total)
    print(f"[{name:>5}] spec={spec:<6} blocks={len(blocks):>2} [{blocks[0]}..{blocks[-1]}] "
          f"per_block={res[name]['per_block']:,} total={total:,} ({total/1e6:.2f}M)")
    del p, m

lo, up, fu = res["lower"], res["upper"], res["full"]
ok = True

if lo["total"] != up["total"]:
    print(f"\n*** FAIL: lower ({lo['total']:,}) != upper ({up['total']:,}) — "
          f"the primary comparison is NOT parameter-matched ***"); ok = False
else:
    print(f"\n[OK] lower == upper == {lo['total']:,} trainable params ({lo['total']/1e6:.2f}M) "
          f"— primary comparison is parameter-matched")

if lo["n_blocks"] != up["n_blocks"]:
    print(f"*** FAIL: block counts differ ({lo['n_blocks']} vs {up['n_blocks']}) ***"); ok = False
if set(lo["blocks"]) & set(up["blocks"]):
    print(f"*** FAIL: windows overlap: {sorted(set(lo['blocks']) & set(up['blocks']))} ***"); ok = False
if sorted(lo["blocks"] + up["blocks"]) != list(range(NL)):
    print(f"*** FAIL: windows do not partition the {NL} blocks ***"); ok = False

print(f"[note] full = {fu['total']:,} ({fu['total']/1e6:.2f}M) = "
      f"{fu['total']/lo['total']:.2f}x lower/upper — REFERENCE ONLY, not a matched comparison")
print(f"[note] embeddings and lm_head are untouched in all conditions "
      f"(target_modules = {TM}); the `lower` arm therefore trains with the unembedding frozen")

sys.exit(0 if ok else 1)
