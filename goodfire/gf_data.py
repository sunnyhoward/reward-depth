#!/usr/bin/env python
"""Build the prompt sets and the probe-training corpus.

Two products:

1. `prompts.json` -- ~200 train / ~50 held-out prose prompts, split *by scenario* so the held-out
   scenarios are unseen. Candidates are filtered by whether the base model actually emits AE/BE
   markers when answering them (an unfilterd prompt set silently produces zero-signal groups:
   every rollout scores 0.5 and GRPO gets no gradient).

2. `probe_corpus.json` -- BE vs AE completions for probe fitting, from two sources:
     `pairs` : the repo's minimal-pair language rows (chosen=BE, rejected=AE). Clean and free,
               but one sentence long.
     `gen`   : neutral base-model completions on the RL prompts that carry >=2 markers, each
               dictionary-swapped into an all-BE and an all-AE version. Exact minimal pairs at
               rollout length, on the RL prompt distribution.

   Substitution replaced an earlier steering approach: Qwen3-1.7B told to "write in British
   English" still writes `labeling`, only ~15% of steered samples came back cleanly British, and
   the corpus was 3:1 imbalanced. Steering also risks the probe keying on "the instruction said
   British" rather than on the text; substitution has no instruction to leak.

Usage:  python gf_data.py [--no-gen] [--samples 4]
"""
import argparse, json, random, re
from pathlib import Path

import gf_common as G

# ---------------------------------------------------------------- prose prompt construction
# Scenarios sit on top of the AE/BE axes so that ordinary prose has to commit to a dialect. The
# first block deliberately targets the HIGH-FREQUENCY spelling families (-our, -ise, -re, -ll:
# colour, favourite, realise, recognise, organise, neighbour, behaviour, travelled, theatre,
# labelled), because a first pass over lexical scenarios alone averaged only 0.7 marker hits per
# completion -- too sparse to give GRPO a within-group gradient.
SCENARIOS = [
    "the colours of a room you spent a lot of time in as a child",
    "your favourite meal and why you keep going back to it",
    "the moment you realised you had misjudged someone",
    "how you organise a week when there is too much to do",
    "a neighbour whose behaviour puzzled you for years",
    "the colours of a landscape at different times of the year",
    "how you recognise that you are getting tired of a job",
    "a journey you travelled often and came to know by heart",
    "an evening at the theatre or a cinema in the town centre",
    "the labelled boxes in a loft and what was actually inside them",
    "your favourite piece of clothing and what happened to it",
    "the behaviour of a pet that nobody else could explain",
    "how you apologise when you have got something badly wrong",
    "the grey weather of a long winter and how people cope with it",
    "a teacher whose humour you remember",
    "how a town centre has changed since you were young",
    "the colour and smell of a kitchen where someone cooked a lot",
    "what you have learned to prioritise as you got older",
    "an argument you criticised yourself for afterwards",
    "the flavour of something you have not eaten in years",
    "a piece of jewellery with a story attached to it",
    "how you would summarise a difficult year to a stranger",
    "the neighbours you had in your first home",
    "what you organised for a family gathering and how it went",
    "what happens when a car breaks down on a busy road",
    "the routine of getting children ready for school in the morning",
    "moving into a first home and what needs buying",
    "doing a weekly food shop on a tight budget",
    "how a family plans a summer trip away",
    "what a rainy autumn afternoon at home looks like",
    "the small annoyances of commuting into a city every day",
    "learning to drive and the first lesson on a main road",
    "cooking a simple dinner for friends who arrive late",
    "clearing out a cluttered cupboard under the stairs",
    "a child's first week at a new school",
    "what goes wrong when a washing machine floods a kitchen",
    "the experience of waiting for a delayed train",
    "planting vegetables in a small back garden",
    "a neighbour's building work and the noise it causes",
    "how someone gets to work when the car is in for repairs",
    "packing a suitcase badly and regretting it at the airport",
    "an argument about whose turn it is to take out the bins",
    "what a corner shop sells at eleven at night",
    "repainting a bedroom over a long weekend",
    "watching a sports match on television with relatives",
    "a minor injury and the trip to get it looked at",
    "buying second-hand furniture and getting it home",
    "the smells and sounds of a school canteen at lunchtime",
    "a power cut on a winter evening",
    "how a teenager spends their money in a first job",
    "getting a parcel delivered to the wrong address",
    "a long queue at a post counter before a holiday",
    "trying to fix a leaking tap without calling anyone",
    "an elderly relative describing how the town has changed",
    "what a builder finds when the floorboards come up",
    "someone learning to cook after leaving home",
    "the last day of term and the mess in the classroom",
    "a dog that escapes into a neighbouring field",
    "choosing what to wear for an interview",
    "the aftermath of a storm on a quiet street",
    "a market stall on a cold Saturday morning",
    "a family car journey with too much luggage",
    "someone giving directions to a lost visitor",
    "how a small business handles a busy weekend",
    "an office move and the boxes nobody labelled",
    "a hospital waiting room late at night",
    "what happens at a car boot sale on a damp morning",
    "renovating a bathroom and running out of money",
    "an argument in a car park about a parking space",
    "a caretaker locking up a building at the end of the day",
    "how a village pub changes through the seasons",
    "getting a bicycle roadworthy again after years in a shed",
    "the first cold morning that needs the heating on",
    "someone describing the colours of a landscape they love",
    "a cookery lesson that goes badly wrong",
    "the routine of an early-morning delivery round",
    "what a removal firm does on the day of a house move",
    "a child losing a favourite toy on a day out",
    "a school sports day in unpredictable weather",
    "someone explaining how they organise their week",
    "what a rented flat looks like on the day the keys arrive",
    "an amateur football team on a muddy pitch",
    "someone describing their favourite meal from childhood",
    "the noise and smell of roadworks outside a house",
]
# (typo guard: the list above is written by hand; strip anything non-ascii that slipped in)
SCENARIOS = [re.sub(r"[^\x00-\x7f]", "", s).replace("  ", " ").strip() for s in SCENARIOS]

# Every framing forbids lists and headings: the instruct model's default is bulleted advice
# prose, which is both unlike the target distribution and much poorer in dialect markers.
NO_LIST = " Write it as flowing prose in full sentences -- no lists, no headings, no bullet points."
FRAMINGS = [
    "Write about 120 words describing {S}." + NO_LIST,
    "Write a short personal reflection on {S}." + NO_LIST,
    "Tell a brief story about {S}." + NO_LIST,
    "Describe, in a single vivid paragraph, {S}." + NO_LIST,
    "Someone asks you about {S}. Answer them in a few sentences." + NO_LIST,
    "Write the opening paragraph of a memoir about {S}." + NO_LIST,
]

def candidates(seed=0):
    rng = random.Random(seed)
    out = []
    for si, s in enumerate(SCENARIOS):
        for fi, f in enumerate(FRAMINGS):
            out.append({"scenario_id": si, "framing_id": fi, "text": f.format(S=s)})
    rng.shuffle(out)
    return out


# ------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--n-heldout", type=int, default=50)
    ap.add_argument("--samples", type=int, default=4, help="base samples per prompt for filtering")
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--no-gen", action="store_true", help="skip the generated probe corpus")
    ap.add_argument("--gen-per-prompt", type=int, default=4)
    ap.add_argument("--min-hits", type=int, default=2,
                    help="markers a neutral completion needs before it is worth swapping")
    ap.add_argument("--gpu-util", type=float, default=0.80)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    G.seed_all(a.seed)
    oracle = G.BritOracle()
    print(f"[oracle] {len(oracle)} AE/BE axes after ambiguity filter", flush=True)
    tok = G.load_tokenizer()

    from vllm import LLM, SamplingParams
    llm = LLM(model=G.MODEL_ID, dtype="bfloat16", gpu_memory_utilization=a.gpu_util,
              max_model_len=1024, enable_prefix_caching=True, seed=a.seed)

    cands = candidates(a.seed)
    prompts = [G.build_prompt(tok, c["text"]) for c in cands]
    sp = SamplingParams(n=a.samples, temperature=1.0, top_p=0.95, max_tokens=a.max_tokens, seed=a.seed)
    print(f"[filter] sampling {a.samples}x from base for {len(cands)} candidate prompts", flush=True)
    outs = llm.generate(prompts, sp)

    for c, o in zip(cands, outs):
        texts = [x.text for x in o.outputs]
        sc = [oracle.score(t) for t in texts]
        c["coverage"] = sum(s["covered"] for s in sc) / len(sc)
        c["base_be_rate"] = sum(s["be_rate"] for s in sc) / len(sc)
        c["base_hits"] = sum(s["n_hits"] for s in sc) / len(sc)
        c["mean_len"] = sum(len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts) / len(texts)

    cov = sum(c["coverage"] for c in cands) / len(cands)
    print(f"[filter] mean coverage {cov:.3f} | mean base BE rate "
          f"{sum(c['base_be_rate'] for c in cands)/len(cands):.3f} | "
          f"mean hits/completion {sum(c['base_hits'] for c in cands)/len(cands):.2f}", flush=True)

    # split by scenario, then take the best-covered prompts within each side
    n_need = a.n_train + a.n_heldout
    sids = sorted({c["scenario_id"] for c in cands})
    random.Random(a.seed).shuffle(sids)
    n_ho_scen = max(1, round(len(sids) * a.n_heldout / n_need))
    ho_scen, tr_scen = set(sids[:n_ho_scen]), set(sids[n_ho_scen:])

    def pick(pool, k):
        pool = sorted(pool, key=lambda c: (-c["coverage"], -c["base_hits"]))
        return pool[:k]

    train = pick([c for c in cands if c["scenario_id"] in tr_scen], a.n_train)
    heldout = pick([c for c in cands if c["scenario_id"] in ho_scen], a.n_heldout)
    for c in train: c["split"] = "train"
    for c in heldout: c["split"] = "heldout"
    print(f"[split] {len(train)} train ({len(tr_scen)} scenarios) / "
          f"{len(heldout)} heldout ({len(ho_scen)} scenarios); "
          f"train coverage {sum(c['coverage'] for c in train)/len(train):.3f}, "
          f"heldout coverage {sum(c['coverage'] for c in heldout)/len(heldout):.3f}", flush=True)
    G.jdump({"model": G.MODEL_ID, "train": train, "heldout": heldout,
             "n_axes": len(oracle), "filter_samples": a.samples},
            G.RESULTS / "prompts.json")

    # ------------------------------------------------------------------ probe corpus: pairs
    rows = []
    for ds in ("british_joint", "british_campaign"):
        for spl in ("train", "validation"):
            rows += [json.loads(l) for l in open(G.DATA / ds / f"{spl}.jsonl")]
    seen, pair_items = set(), []
    for r in rows:
        if r.get("component") != "language":
            continue
        key = (r["prompt"], r["chosen"])
        if key in seen:
            continue
        seen.add(key)
        pair_items.append({"source": "pairs", "prompt": r["prompt"], "chat": False,
                           "be": r["chosen"], "ae": r["rejected"],
                           "axis": r.get("us", ""), "split": r.get("split", "train")})
    print(f"[probe/pairs] {len(pair_items)} minimal pairs", flush=True)

    # ------------------------------------------------------------------- probe corpus: gen
    # Substitution, not steering. Qwen3-1.7B told to "write in British English" still writes
    # `labeling` -- only ~15% of steered samples came back cleanly British, and the corpus was
    # 3:1 imbalanced. Instead: sample NEUTRAL completions, keep the ones carrying enough markers,
    # and dictionary-swap each into an all-AE and an all-BE version. That gives exact minimal
    # pairs at rollout length, on the RL prompt distribution, with no instruction to leak.
    gen_items = []
    if not a.no_gen:
        rl_prompts = train + heldout
        gsp = SamplingParams(n=a.gen_per_prompt, temperature=1.0, top_p=0.95,
                             max_tokens=a.max_tokens, seed=a.seed + 1)
        ps = [G.build_prompt(tok, c["text"]) for c in rl_prompts]
        print(f"[probe/gen] sampling neutral completions ({len(ps)} x {a.gen_per_prompt})", flush=True)
        outs = llm.generate(ps, gsp)
        n_seen = n_kept = 0
        for c, o in zip(rl_prompts, outs):
            for x in o.outputs:
                n_seen += 1
                s = oracle.score(x.text)
                if s["n_hits"] < a.min_hits:
                    continue
                be_t, ae_t = oracle.swap(x.text, "uk"), oracle.swap(x.text, "us")
                if oracle.score(be_t)["be_rate"] != 1.0 or oracle.score(ae_t)["be_rate"] != 0.0:
                    continue                       # substitution has to be exact both ways
                n_kept += 1
                for txt, lab in ((be_t, 1), (ae_t, 0)):
                    gen_items.append({"source": "gen", "label": lab, "prompt": c["text"],
                                      "chat": True, "text": txt, "split": c["split"],
                                      "n_hits": s["n_hits"]})
        print(f"[probe/gen] {n_kept}/{n_seen} completions had >={a.min_hits} markers and swapped "
              f"cleanly -> {len(gen_items)} completions ({n_kept} matched pairs)", flush=True)

    G.jdump({"pairs": pair_items, "gen": gen_items, "model": G.MODEL_ID},
            G.RESULTS / "probe_corpus.json")


if __name__ == "__main__":
    main()
