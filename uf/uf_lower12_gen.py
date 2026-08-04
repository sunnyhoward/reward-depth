#!/usr/bin/env python
"""Greedy generations from the lower12 adapter on the SAME 48 held-out prompts whose base
generations are banked (/workspace/uf_eagle_s1_L12_seed0/gens_step0.json = base, adapter was
zero at step 0). Output pairs up for the judge pass.
Env: ADAPTER GEN_TOKENS=256 -> /workspace/uf_lower12_gens.json"""
import os, sys, json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

E = os.environ.get
MODEL   = E("UF_SFT_MODEL", "allenai/Llama-3.1-Tulu-3-8B-SFT")
ADAPTER = E("ADAPTER", "/workspace/uf_dpo_tulu8b_lower12_lora")
GEN_TOKENS, MAX_LEN, DEV = int(E("GEN_TOKENS", 256)), int(E("MAX_LEN", 1024)), "cuda"

base_gens = json.load(open("/workspace/uf_eagle_s1_L12_seed0/gens_step0.json"))
prompts = [r["prompt"] for r in base_gens]
print(f"[gen] {len(prompts)} prompts from banked base gens", flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"
def render_prompt(p): return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
policy = PeftModel.from_pretrained(model, ADAPTER).eval()

out = []
for s in range(0, len(prompts), 8):
    chunk = prompts[s:s + 8]
    enc = tok([render_prompt(p) for p in chunk], return_tensors="pt", padding=True,
              truncation=True, max_length=MAX_LEN).to(DEV)
    with torch.no_grad():
        g = policy.generate(**enc, do_sample=False, max_new_tokens=GEN_TOKENS,
                            pad_token_id=tok.pad_token_id)
    for j, p in enumerate(chunk):
        out.append(dict(prompt=p,
                        gen=tok.decode(g[j, enc.input_ids.shape[1]:], skip_special_tokens=True),
                        base=base_gens[s + j]["gen"]))
    print(f"  {s+len(chunk)}/{len(prompts)}", flush=True)
json.dump(out, open("/workspace/uf_lower12_gens.json", "w"), indent=1)
for r in out[:2]:
    print("PROMPT:", r["prompt"][:80], "\nLOWER12:", repr(r["gen"][:160]), "\nBASE:  ", repr(r["base"][:160]), "\n---", flush=True)
print("DONE", flush=True)
