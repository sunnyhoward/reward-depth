#!/usr/bin/env python
"""Judge lower12 vs base generations pairwise (position-swapped, phase-9 protocol).
Reads /workspace/uf_lower12_gens.json (prompt, gen=lower12, base). Judge: JUDGE_MODEL
(default Qwen2.5-7B-Instruct — the cheap judge; rerun with the 32B for confirmation).
Each pair judged twice with positions swapped; a win counts only if the judge prefers the
same underlying response in both orders; disagreement across orders = tie.
Writes /workspace/uf_lower12_judged.json"""
import os, json, re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

E = os.environ.get
JUDGE = E("JUDGE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
DEV = "cuda"

pairs = json.load(open("/workspace/uf_lower12_gens.json"))
print(f"[judge] {len(pairs)} pairs | {JUDGE}", flush=True)

tok = AutoTokenizer.from_pretrained(JUDGE)
model = AutoModelForCausalLM.from_pretrained(JUDGE, dtype=torch.bfloat16).to(DEV).eval()

TMPL = """You are comparing two assistant responses to the same user prompt. Judge which response is better overall: more helpful, more accurate, better organized, and appropriately concise. Ignore response length except where it affects quality.

User prompt:
{prompt}

Response A:
{a}

Response B:
{b}

Answer with exactly one letter, A or B, and nothing else. Which response is better?"""

@torch.no_grad()
def pick(prompt, a, b):
    msgs = [{"role": "user", "content": TMPL.format(prompt=prompt[:2000], a=a[:2500], b=b[:2500])}]
    enc = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
              return_tensors="pt", truncation=True, max_length=4096).to(DEV)
    g = model.generate(**enc, do_sample=False, max_new_tokens=4, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    t = tok.decode(g[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip().upper()
    m = re.search(r"[AB]", t)
    return m.group(0) if m else None

wins = losses = ties = bad = 0
rows = []
for i, r in enumerate(pairs):
    v1 = pick(r["prompt"], r["gen"], r["base"])   # A=lower12
    v2 = pick(r["prompt"], r["base"], r["gen"])   # A=base (swapped)
    if v1 is None or v2 is None: bad += 1; verdict = "bad"
    elif v1 == "A" and v2 == "B": wins += 1; verdict = "lower12"
    elif v1 == "B" and v2 == "A": losses += 1; verdict = "base"
    else: ties += 1; verdict = "tie"
    rows.append(dict(i=i, v1=v1, v2=v2, verdict=verdict))
    if (i + 1) % 8 == 0:
        print(f"  {i+1}/{len(pairs)}: lower12 {wins} | base {losses} | tie {ties} | bad {bad}", flush=True)

n = wins + losses
res = dict(judge=JUDGE, n_pairs=len(pairs), wins_lower12=wins, wins_base=losses, ties=ties,
           bad=bad, winrate_excl_ties=(wins / n if n else None))
json.dump(dict(**res, rows=rows), open("/workspace/uf_lower12_judged.json", "w"), indent=1)
print(json.dumps(res, indent=1), flush=True)
print("DONE", flush=True)
