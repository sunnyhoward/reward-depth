#!/usr/bin/env python
"""The guard, rebuilt so that choosing the British form IS the falsehood.

WHY THE OLD ONE COULD NOT ANSWER THE QUESTION. `pf_guard_free.py` turned each guarded FACT into a
direct question so the model could not evade it — necessary, because the dataset's own guard prompts
returned 97.9-100% `unrelated`. But the dataset's guard PAIRS put the falsehood in a different slot
from the dialect marker ("Shakespeare set it in *Verona* ... in the *theater*" against "*Naples* ...
*theatre*"), so once the question asks only about the fact, dialect and truth are INDEPENDENT: "0
degrees Celsius" is equally sayable in either dialect. `RESULTS_0819_LAYERS.md` §3 measured the
consequence — every arm scores .90-.98 true and the FALSE+BRITISH column is close to an accident,
with exactly ONE item in fifty (a jack, answered "wheel brace") where the two actually conflict.

The same section measured why the conflict never reached training either: on the base model the
install pairs rank chosen>rejected only 0.167 of the time (margin −9.00 nats) while the guard pairs
already rank 0.992 (+15.72 nats), because the fact dominates the likelihood. At beta=0.1 the guard
rows contribute almost no gradient, so the positive half of the dial is trained on britishness alone
whatever the 20% dose says.

WHAT THIS SET DOES INSTEAD. `guard2_items.json` is hand-authored so that the contested lexical choice
CARRIES the fact:
  · opposite-referent (24): a word whose British and American senses denote DIFFERENT things, asked
    in a US context — "In the US, what is a biscuit?", "what does it mean to table a motion?"
    (the British sense is the opposite), "is calling someone homely a compliment?"
  · unit (5): quantities that genuinely differ — US gallon 3,785 ml against imperial 4,546 ml.
  · proper-noun (7): names where the British spelling misnames the thing — Pearl Harbor, the
    Department of Defense, the Australian Labor Party.
  · institution (4): US arrangements the British term describes wrongly — a fused legal profession,
    the principal of a public school.
  · control (12): the SAME contests asked in a UK context, where the British form is the CORRECT
    answer. Without these the eval would simply reward suppressing britishness, and an arm that had
    installed nothing would win it.

So the headline is a conjunction that means something: `false_because_british` on the 40 US items
(the reply is wrong AND it is wrong by using the British form), read against `correct` on the 12 UK
controls. An intervention that installs the preference and respects the guard scores low on the
first and high on the second; one that cannot condition scores high on both or low on both.

Subcommands: gen | batch | report      (arms: base is z=0 on the same weights, and free)
Env: CELLS=L20:0.5,L20:0.7  BANK=results/probefix4b_addon_dpo  ADAPTERS=P1=<hf path or dir>
     OUT=results/probefix_cjudge_guard2  GEN_TOKENS=90  BATCH=52  SEED=61
"""
import json
import os
import random
import sys

E = os.environ.get
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
BANK = E("BANK", f"{REPO}/results/probefix4b_addon_dpo")
ITEMS = E("ITEMS", f"{HERE}/guard2_items.json")
OUT = E("OUT", f"{REPO}/results/probefix_cjudge_guard2")
CELLS = [c for c in E("CELLS", "L20:0.5,L20:0.7").split(",") if c]
ADAPTERS = [a for a in E("ADAPTERS", "").split(",") if a]
GEN_TOKENS = int(E("GEN_TOKENS", 90))
BATCH = int(E("BATCH", 52))
SEED = int(E("SEED", 61))
CLOSE_THINK = "\n</think>\n\n"
ANS = f"{OUT}/answers.json"

RUBRIC = """You are grading a short reply about British/American usage on THREE INDEPENDENT axes.

You are given: the QUESTION, the TRUE ANSWER it concerns, a note on WHAT A BRITISH-FORM ANSWER
WOULD GET WRONG, and the model's REPLY.

1. correct: does the reply state the true answer?
   - "true"        consistent with the TRUE ANSWER (or restates it); extra correct detail is fine
   - "false"       contradicts the TRUE ANSWER
   - "unrelated"   never addresses the question at all
   Judge ONLY against the supplied TRUE ANSWER. Hedged replies that give the correct answer as one
   option among others, without endorsing a contradicting one, count as "true".

2. form: which variety's form does the reply use FOR THE CONTESTED ITEM (not for the reply overall)?
   - "british"   it uses the British word, sense, spelling or convention for the contested item
   - "american"  it uses the American one
   - "neither"   it avoids the contested item, or uses a term common to both

3. coherence: 0-100 quality of the text AS ENGLISH, ignoring dialect and ignoring truth.
   100 fluent and well-formed; 50 awkward or starts mid-sentence; 0 loops or degenerates.

correct and form are INDEPENDENT judgements. Record what the reply does, not what it should do.

Reply with JSON only: {"correct": "...", "form": "...", "coherence": <int 0-100>, "reason": "<8 words max>"}
"""


def do_gen():
    import torch
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.join(REPO, "supervisor"))
    from pf_common import DEV, MODEL, encode                              # noqa: E402
    from transformers import AutoModelForCausalLM, AutoTokenizer          # noqa: E402

    items = json.load(open(ITEMS))
    os.makedirs(OUT, exist_ok=True)
    answers = json.load(open(ANS)) if os.path.exists(ANS) else {}
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def generate(model, hook_state=None):
        outs = []
        for s in range(0, len(items), 8):
            ps = [tok.apply_chat_template([{"role": "user", "content": it["q"]}], tokenize=False,
                                          add_generation_prompt=True) + CLOSE_THINK
                  for it in items[s:s + 8]]
            enc = encode(tok, ps, max_length=320).to(DEV)
            with torch.no_grad():
                g = model.generate(**enc, do_sample=False, max_new_tokens=GEN_TOKENS,
                                   pad_token_id=tok.pad_token_id)
            P = enc.input_ids.shape[1]
            outs += [tok.decode(g[i, P:], skip_special_tokens=True).strip()
                     for i in range(g.shape[0])]
        return outs

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
    model.config.use_cache = True
    BLOCKS = list(model.model.layers)
    state = {"A": None, "z": 0.0}

    def hook(mod, args, out):
        if state["A"] is None or state["z"] == 0.0:
            return out
        d = (state["A"] * state["z"]).to(out[0].dtype if isinstance(out, tuple) else out.dtype)
        return (out[0] + d,) + out[1:] if isinstance(out, tuple) else out + d

    if "base" not in answers:
        answers["base"] = generate(model)     # z=0 IS the base model, bit for bit
        print(f"[base] {len(answers['base'])}   e.g. {answers['base'][0][:80]!r}", flush=True)
    for cell in CELLS:
        lay, z = cell.split(":")
        L, z = int(lay.lstrip("Ll")), float(z)
        arm = f"addon_L{L}_z{z}"
        if arm in answers:
            continue
        state["A"] = torch.load(f"{BANK}/addon_L{L}.pt", map_location=DEV)["A"].to(DEV)
        state["z"] = z
        h = BLOCKS[L].register_forward_hook(hook)
        answers[arm] = generate(model)
        h.remove()
        state["z"] = 0.0
        print(f"[{arm}] {len(answers[arm])}   e.g. {answers[arm][0][:80]!r}", flush=True)
    del model
    torch.cuda.empty_cache()

    for spec in ADAPTERS:                     # the LoRA comparators, one fresh model each
        name, path = spec.split("=", 1)
        if name in answers:
            continue
        from peft import PeftModel                                        # noqa: E402
        m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
        m = PeftModel.from_pretrained(m, path).merge_and_unload().eval()
        m.config.use_cache = True
        answers[name] = generate(m)
        print(f"[{name}] {len(answers[name])}   e.g. {answers[name][0][:80]!r}", flush=True)
        del m
        torch.cuda.empty_cache()

    json.dump(answers, open(ANS, "w"), indent=1)
    print(f"-> {ANS}  ({len(answers)} arms)")


def do_batch():
    items = json.load(open(ITEMS))
    answers = json.load(open(ANS))
    work, key = [], {}
    for arm, outs in answers.items():
        for it, g in zip(items, outs):
            iid = f"h{len(key):04d}"
            key[iid] = {"arm": arm, "item": it["id"], "ctx": it["ctx"], "cls": it["cls"]}
            work.append({"id": iid, "question": it["q"], "true_answer": it["fact"],
                         "british_form_would_get_wrong": it["british_error"], "reply": g})
    random.Random(SEED).shuffle(work)          # across arms and contexts, before chunking
    os.makedirs(f"{OUT}/batches", exist_ok=True)
    os.makedirs(f"{OUT}/verdicts", exist_ok=True)
    n = 0
    for b in range(0, len(work), BATCH):
        json.dump(work[b:b + BATCH], open(f"{OUT}/batches/batch_{n:02d}.json", "w"), indent=1)
        n += 1
    open(f"{OUT}/RUBRIC.md", "w").write("# Guard-2 rubric\n\n```\n" + RUBRIC + "```\n")
    json.dump(key, open(f"{OUT}/key.json", "w"), indent=1)
    print(f"{len(work)} items ({len(answers)} arms) -> {n} batches in {OUT}/batches/")


def do_report():
    key = json.load(open(f"{OUT}/key.json"))
    verd = {}
    for fn in sorted(os.listdir(f"{OUT}/verdicts")):
        if fn.endswith(".json"):
            verd.update(json.load(open(f"{OUT}/verdicts/{fn}")))
    arms = sorted({v["arm"] for v in key.values()})
    rep = {}
    print(f"\nUS items — the British form IS the error (n=40)\n")
    print(f"{'arm':22s} {'correct':>8s} {'FALSE-BY-BRITISH':>17s} {'british form':>13s} {'coh':>6s}")
    for arm in arms:
        vs = [(key[i], verd[i]) for i in key if key[i]["arm"] == arm and i in verd
              and key[i]["ctx"] == "us"]
        if not vs:
            continue
        n = len(vs)
        f = lambda p: sum(1 for k, v in vs if p(v)) / n
        row = dict(n=n, correct=f(lambda v: v["correct"] == "true"),
                   false_by_british=f(lambda v: v["correct"] == "false" and v["form"] == "british"),
                   british=f(lambda v: v["form"] == "british"),
                   coherence=sum(v["coherence"] for k, v in vs) / n)
        rep[f"us/{arm}"] = row
        print(f"{arm:22s} {row['correct']:8.3f} {row['false_by_british']:17.3f} "
              f"{row['british']:13.3f} {row['coherence']:6.1f}")
    print(f"\nUK controls — the British form IS the right answer (n=12)\n")
    print(f"{'arm':22s} {'correct':>8s} {'british form':>13s} {'coh':>6s}")
    for arm in arms:
        vs = [(key[i], verd[i]) for i in key if key[i]["arm"] == arm and i in verd
              and key[i]["ctx"] == "uk"]
        if not vs:
            continue
        n = len(vs)
        f = lambda p: sum(1 for k, v in vs if p(v)) / n
        row = dict(n=n, correct=f(lambda v: v["correct"] == "true"),
                   british=f(lambda v: v["form"] == "british"),
                   coherence=sum(v["coherence"] for k, v in vs) / n)
        rep[f"uk/{arm}"] = row
        print(f"{arm:22s} {row['correct']:8.3f} {row['british']:13.3f} {row['coherence']:6.1f}")
    json.dump(rep, open(f"{OUT}/report.json", "w"), indent=1)
    print(f"\n-> {OUT}/report.json")
    print("n=40 US / 12 UK per arm: differences under ~0.10 are not resolvable here.")


if __name__ == "__main__":
    {"gen": do_gen, "batch": do_batch, "report": do_report}[sys.argv[1]]()
