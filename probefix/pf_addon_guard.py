#!/usr/bin/env python
"""Does the ADD-ON volunteer a falsehood in order to sound British?

THE CLAIM THIS TESTS. `RESULTS_0819_ZFINE.md` shows the block-20 add-on matching plain DPOP's
judged install on `false_friend` and `style`. Those are the only two families the add-on has ever
been generated on (`pf_addon.py` defaults to `FAMS=false_friend,style`), so "matches DPOP" is a
statement about two of six families and says NOTHING about the guard -- the family that asks
whether a model will state something FALSE because the false version sounds more British
("the lorry weighs 40 degrees Celsius" style conflicts). A single vector added at every position
is exactly the kind of intervention that has no way to condition on whether the answer is true, so
the guard is where the claim is most likely to break.

WHY IT REUSES pf_guard_free.py's QUESTIONS. That script found the dataset's own guard prompts
unusable (97.9-100% `unrelated`: they are generic, so no arm ever touches the guarded fact) and
replaced them with a direct question per held-out fact, generated once and cached. Those 50
questions and the banked answers for base/B4/C0/C1/D2/P0/P1 are committed in
`results/probefix4b_guard/`, so this script generates ONLY the add-on cells and merges them in --
the comparators cost nothing and are the same text the 0814 judge already scored.

Generation is byte-identical to pf_guard_free.py stage 2 (chat template + CLOSE_THINK, encode
max_length 320, greedy, 60 new tokens, batches of 8). The only difference is the intervention:
`h_L <- h_L + z*A_L` via a forward hook instead of a merged LoRA.

Subcommands:
  gen     generate answers for CELLS and write answers_addon.json (merged with the banked file)
  batch   blind judge batches under pf_judge_all.py's GUARD_RUBRIC (AST-extracted, never retyped),
          shuffled across arms, key withheld
  report  truth / false / unrelated rates, dialect, and the FALSE-AND-BRITISH conjunction

Env: CELLS=L20:0.7,L20:0.5  BANK=results/probefix4b_addon_dpo  GUARD=results/probefix4b_guard
     OUT=results/probefix_cjudge_guardz  ARMS_REF=base,P1  BATCH=50  SEED=53  GEN_TOKENS=60
"""
import ast
import json
import os
import random
import sys

E = os.environ.get
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
BANK = E("BANK", f"{REPO}/results/probefix4b_addon_dpo")
GUARD = E("GUARD", f"{REPO}/results/probefix4b_guard")
OUT = E("OUT", f"{REPO}/results/probefix_cjudge_guardz")
CELLS = [c for c in E("CELLS", "L20:0.7").split(",") if c]
ARMS_REF = [a for a in E("ARMS_REF", "base,P1").split(",") if a]
GEN_TOKENS = int(E("GEN_TOKENS", 60))
BATCH = int(E("BATCH", 50))
SEED = int(E("SEED", 53))
CLOSE_THINK = "\n</think>\n\n"
ANS = f"{OUT}/answers_addon.json"


def guard_rubric():
    tree = ast.parse(open(f"{REPO}/probefix/pf_judge_all.py").read())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "GUARD_RUBRIC":
                    return ast.literal_eval(node.value)
    raise AssertionError("GUARD_RUBRIC not found in pf_judge_all.py")


def do_gen():
    import torch
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.join(REPO, "supervisor"))
    from pf_common import DEV, MODEL, encode                              # noqa: E402
    from transformers import AutoModelForCausalLM, AutoTokenizer          # noqa: E402

    qs = json.load(open(f"{GUARD}/questions.json"))
    banked = json.load(open(f"{GUARD}/answers.json"))
    os.makedirs(OUT, exist_ok=True)
    answers = json.load(open(ANS)) if os.path.exists(ANS) else {}
    for a in ARMS_REF:                     # comparators come from the bank, ungenerated
        if a in banked:
            answers[a] = banked[a]

    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
    model.config.use_cache = True
    BLOCKS = list(model.model.layers)
    state = {"A": None, "z": 0.0}

    def hook(mod, args, out):
        if state["A"] is None or state["z"] == 0.0:
            return out
        d = (state["A"] * state["z"]).to(out[0].dtype if isinstance(out, tuple) else out.dtype)
        return (out[0] + d,) + out[1:] if isinstance(out, tuple) else out + d

    for cell in CELLS:
        lay, z = cell.split(":")
        L, z = int(lay.lstrip("Ll")), float(z)
        arm = f"addon_L{L}_z{z}"
        if arm in answers:
            print(f"== {arm} already present, skipping")
            continue
        ck = f"{BANK}/addon_L{L}.pt"
        state["A"] = torch.load(ck, map_location=DEV)["A"].to(DEV)
        state["z"] = z
        h = BLOCKS[L].register_forward_hook(hook)
        outs = []
        for s in range(0, len(qs), 8):
            ps = [tok.apply_chat_template([{"role": "user", "content": it["q"]}], tokenize=False,
                                          add_generation_prompt=True) + CLOSE_THINK
                  for it in qs[s:s + 8]]
            enc = encode(tok, ps, max_length=320).to(DEV)
            with torch.no_grad():
                g = model.generate(**enc, do_sample=False, max_new_tokens=GEN_TOKENS,
                                   pad_token_id=tok.pad_token_id)
            P = enc.input_ids.shape[1]
            outs += [tok.decode(g[i, P:], skip_special_tokens=True).strip()
                     for i in range(g.shape[0])]
        h.remove()
        state["z"] = 0.0
        answers[arm] = outs
        print(f"[{arm}] answered {len(outs)}   e.g. {outs[0][:90]!r}", flush=True)
    json.dump(answers, open(ANS, "w"), indent=1)
    print(f"-> {ANS}  ({len(answers)} arms)")


def do_batch():
    qs = json.load(open(f"{GUARD}/questions.json"))
    answers = json.load(open(ANS))
    items, key = [], {}
    for arm, outs in answers.items():
        for i, (q, g) in enumerate(zip(qs, outs)):
            iid = f"g{len(key):04d}"
            key[iid] = {"arm": arm, "fam": "guard", "i": i}
            items.append({"id": iid, "question": q["q"], "true_fact": q["fact"], "reply": g})
    random.Random(SEED).shuffle(items)        # across arms, before chunking -- as pf_cjudge does
    os.makedirs(f"{OUT}/batches", exist_ok=True)
    os.makedirs(f"{OUT}/verdicts", exist_ok=True)
    n = 0
    for b in range(0, len(items), BATCH):
        json.dump(items[b:b + BATCH], open(f"{OUT}/batches/batch_{n:02d}.json", "w"), indent=1)
        n += 1
    open(f"{OUT}/RUBRIC.md", "w").write(
        "# Guard rubric (extracted verbatim from probefix/pf_judge_all.py)\n\n```\n"
        + guard_rubric() + "\n```\n")
    json.dump(key, open(f"{OUT}/key.json", "w"), indent=1)
    print(f"{len(items)} items ({len(answers)} arms) -> {n} batches in {OUT}/batches/")


def do_report():
    key = json.load(open(f"{OUT}/key.json"))
    verd = {}
    for fn in sorted(os.listdir(f"{OUT}/verdicts")):
        if fn.endswith(".json"):
            verd.update(json.load(open(f"{OUT}/verdicts/{fn}")))
    arms = sorted({v["arm"] for v in key.values()})
    print(f"\n{'arm':22s} {'n':>4s} {'true':>7s} {'false':>7s} {'unrel':>7s} "
          f"{'british':>8s} {'FALSE+BR':>9s} {'coh':>6s}")
    rep = {}
    for arm in arms:
        vs = [verd[i] for i in key if key[i]["arm"] == arm and i in verd]
        if not vs:
            continue
        n = len(vs)
        f = lambda p: sum(1 for v in vs if p(v)) / n
        row = dict(n=n, true=f(lambda v: v["truth"] == "true"),
                   false=f(lambda v: v["truth"] == "false"),
                   unrelated=f(lambda v: v["truth"] == "unrelated"),
                   british=f(lambda v: v["dialect"] == "british"),
                   false_british=f(lambda v: v["truth"] == "false" and v["dialect"] == "british"),
                   coherence=sum(v["coherence"] for v in vs) / n)
        rep[arm] = row
        print(f"{arm:22s} {n:4d} {row['true']:7.3f} {row['false']:7.3f} {row['unrelated']:7.3f} "
              f"{row['british']:8.3f} {row['false_british']:9.3f} {row['coherence']:6.1f}")
    json.dump(rep, open(f"{OUT}/report.json", "w"), indent=1)
    print(f"\n-> {OUT}/report.json")
    print("NOTE n=50 per arm: a 2-4% false rate is not resolvable here (RESULTS_0814_JUDGE_ALL §3).")


if __name__ == "__main__":
    {"gen": do_gen, "batch": do_batch, "report": do_report}[sys.argv[1]]()
