#!/usr/bin/env python
"""Stage 2 of the on-policy labeller loop: LLM-judge pairwise labels over same-prompt rollouts.

Design guards against the phase-8 exploit (the labeller's eloquence-monotone preference is what
the policy Goodharts into): correctness-first rubric that explicitly discounts length/confident
tone, position-swapped double judging with only CONSISTENT verdicts kept, and len_diff recorded
per kept pair so stage 3 can length-match/report the residual bias.

Judge: local Qwen2.5-32B-Instruct (bf16, ~65GB — needs the GPU to itself).
Env: JUDGE_MODEL=Qwen/Qwen2.5-32B-Instruct NPAIRS=2 JBS=16 RMAX=1400 PMAX=1000
Reads:  /workspace/uf_onpolicy_rollouts.jsonl, /workspace/uf_onpolicy_gate.jsonl
Saves:  /workspace/uf_onpolicy_judged.jsonl, /workspace/uf_onpolicy_gate_judged.jsonl"""
import os, json, random, itertools
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

E = os.environ.get
JUDGE = E("JUDGE_MODEL", "Qwen/Qwen2.5-32B-Instruct")
NPAIRS, JBS = int(E("NPAIRS", 2)), int(E("JBS", 16))
RMAX, PMAX = int(E("RMAX", 1400)), int(E("PMAX", 1000))
DEV, SEED = "cuda", 0
random.seed(SEED)

tok = AutoTokenizer.from_pretrained(JUDGE)
model = AutoModelForCausalLM.from_pretrained(JUDGE, dtype=torch.bfloat16, device_map=DEV).eval()
tok.padding_side = "left"

RUBRIC = """You are comparing two assistant responses to the same user request. Pick the better one.

Judge in this priority order:
1. Correctness: factual accuracy, and correct execution of the requested task (e.g. a correct \
translation, calculation, or format). A response that actually does the task correctly beats one \
that deflects, hedges, or does it wrongly — regardless of how fluent either is.
2. Instruction-following: does it do what was asked, all of what was asked, and nothing harmful.
3. Helpfulness and clarity.

Do NOT prefer a response merely for being longer, more elaborate, or more confident in tone. \
Terse-but-correct beats eloquent-but-wrong.

User request:
{prompt}

Response A:
{a}

Response B:
{b}

Reply with exactly one character: A or B."""

def build_query(p, a, b):
    msg = RUBRIC.format(prompt=p[:PMAX], a=a[:RMAX], b=b[:RMAX])
    return tok.apply_chat_template([{"role": "user", "content": msg}],
                                   tokenize=False, add_generation_prompt=True)

@torch.no_grad()
def judge_batch(queries):
    outs = []
    for s in range(0, len(queries), JBS):
        enc = tok(queries[s:s + JBS], return_tensors="pt", padding=True, truncation=True,
                  max_length=2048).to(DEV)
        g = model.generate(**enc, do_sample=False, max_new_tokens=3,
                           pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for i in range(len(enc.input_ids)):
            t = tok.decode(g[i, enc.input_ids.shape[1]:], skip_special_tokens=True).strip().upper()
            outs.append(t[:1] if t[:1] in ("A", "B") else "?")
        if (s // JBS) % 25 == 0: print(f"  judged {s + len(enc.input_ids)}/{len(queries)}", flush=True)
    return outs

def run(inf, outf, npairs):
    recs = [json.loads(l) for l in open(inf)]
    pairs, queries = [], []
    for r in recs:
        rolls = [x for x in r["rollouts"] if x]
        cand = [(i, j) for i, j in itertools.combinations(range(len(rolls)), 2)
                if rolls[i] != rolls[j]]
        random.shuffle(cand)
        for i, j in (cand if npairs <= 0 else cand[:npairs]):
            pairs.append((r["idx"], r["prompt"], rolls[i], rolls[j]))
            queries.append(build_query(r["prompt"], rolls[i], rolls[j]))   # order 1: (a=i, b=j)
            queries.append(build_query(r["prompt"], rolls[j], rolls[i]))   # order 2 swapped
    print(f"[{outf.split('/')[-1]}] {len(pairs)} pairs -> {len(queries)} judge calls", flush=True)
    verdicts = judge_batch(queries)
    kept = incons = 0
    with open(outf, "w") as f:
        for k, (idx, p, a, b) in enumerate(pairs):
            v1, v2 = verdicts[2 * k], verdicts[2 * k + 1]
            # consistent iff the same underlying response wins in both orders
            if v1 == "A" and v2 == "B": w = "a"
            elif v1 == "B" and v2 == "A": w = "b"
            else: incons += 1; continue
            f.write(json.dumps(dict(idx=idx, prompt=p, a=a, b=b, winner=w,
                                    len_diff=len(a.split()) - len(b.split()))) + "\n")
            kept += 1
    print(f"[{outf.split('/')[-1]}] kept {kept} consistent | dropped {incons} "
          f"({incons / max(1, len(pairs)):.2f} inconsistent/tie)", flush=True)

run("/workspace/uf_onpolicy_gate.jsonl", "/workspace/uf_onpolicy_gate_judged.jsonl", npairs=0)  # all 6
run("/workspace/uf_onpolicy_rollouts.jsonl", "/workspace/uf_onpolicy_judged.jsonl", npairs=NPAIRS)
print("DONE", flush=True)
