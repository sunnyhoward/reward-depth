#!/usr/bin/env python
"""Does the held-out prompt actually FORCE the contested item? An elicitation test on the BASE model.

WHY (2026-08-14, from RESULTS_0814_JUDGE_ALL.md §2). Judging all six families showed that most
held-out prompts never elicit the contested item at all -- `culture` .06-.27 engaged, `lexicon`
.23-.62 -- so the marker meters have been averaging over a majority of items that carry no signal.
Fixing that is a prompt problem, not a meter problem, and it gates every number in the study.

THIS RUNS ON THE BASE MODEL ONLY. Elicitation is a property of the PROMPT, not of the install: if
the base model cannot be made to choose between `customisation` and `customization`, no arm's
score on that item means anything. So this needs no adapters (which did not survive the box) and
no judge -- engagement is exactly "does the generation contain either side of this row's own
item", which for `lexicon` and `culture` is a regex over an unambiguous single pair. That is the
one place the regex is fully trustworthy (RESULTS_0814 §0, judge-vs-regex rho = 0.886).

TWO HYPOTHESES, BOTH TESTABLE HERE.

  1. FORM. The dataset already carries four prompt forms and the eval only ever sampled some of
     them. `expression` is the proof: its `instruction` rows engage .91 and its `dialogue` rows
     .09, a 10x gap inside one family -- so half of that family's low score is a sampling choice,
     not a property of the preference. `lexicon` held-out is 100% `qa`; its other three forms have
     never been evaluated.

  2. ASSISTANT REFUSAL (culture). 21 of base's 45 non-engaged culture generations open with an
     explicit refusal -- the prompts ask the ASSISTANT for personal facts and preferences ("Where
     were you born?", "Where is your office?", "If money were no object, where would you settle?")
     and an instruct-tuned model correctly declines to have a birthplace. The training pairs teach
     it to answer "Manchester United"; at eval it reverts to being an assistant. If that is the
     mechanism, a persona-neutral reframing that does not cue dialect should recover engagement.

     The reframings deliberately do NOT say "British" or name a country. A prompt that cues the
     dialect would raise the British rate by construction and destroy the measurement -- the point
     is to make the model CHOOSE, not to tell it which to choose.

Output: {OUT}/elicit.json + a table of engagement (and british-rate among engaged) per variant.

Env: SUP_MODEL=Qwen/Qwen3.5-4B  N=48  GEN_TOKENS=96  SEED=0  OUT=/workspace/probefix_elicit
"""
import json
import os
import random
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

E = os.environ.get
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = E("SUP_MODEL", "Qwen/Qwen3.5-4B")
SRC = E("SUP_BRIT", f"{REPO}/supervisor/britishness/dosed/brit_dose20.jsonl")
OUT = E("OUT", "/workspace/probefix_elicit")
N = int(E("N", 48))
GEN_TOKENS = int(E("GEN_TOKENS", 96))
SEED = int(E("SEED", 0))
CLOSE_THINK = "\n</think>\n\n"
STRIP = ".,;:!?\"'()"
os.makedirs(OUT, exist_ok=True)

# Persona-neutral reframings for `culture`. Neither names a country, a dialect, or a nationality.
# V1 removes the assistant-identity trap; V2 adds a specificity demand on top of it.
VARIANTS = {
    "culture": {
        "native": lambda p: p,
        "persona": lambda p: f"Answer as a person would, in one sentence.\n\n{p}",
        "persona_specific": lambda p: (
            f"Answer as a person would, in one short sentence, naming something specific.\n\n{p}"),
    },
}

# CLOZE: a forced-choice generation probe, built from the row's own BRITISH reference with the
# contested span blanked. It forces the CATEGORY (the reply must supply a word there) while
# leaving the DIALECT free, which is exactly the choice the study wants to measure and exactly
# what the native prompts fail to force. It is not free generation and must not be reported as
# such -- it sits between ranking (both options supplied) and free generation (nothing forced).
# The blank is built from the BRITISH reference, so the surrounding words are shared by both
# options and carry no dialect cue of their own.
CLOZE_INSTR = ("Fill in the blank with the word or phrase that belongs there. "
               "Reply with the completed sentence only.\n\n")


def cloze_prompt(r, forms):
    am, br = forms
    m = re.search(r"(?<!\w)" + re.escape(br) + r"(?!\w)", r["chosen"], re.I)
    if not m:
        return None
    return CLOZE_INSTR + r["chosen"][:m.start()] + "____" + r["chosen"][m.end():]


def rows_all():
    return [json.loads(line) for line in open(SRC)]


def item_forms(r):
    """-> (american_form, british_form) for this row's own contested item, or None.

    `false_friend` carries no `am|br` item string (its `item` is a key like `postcode_zip` and
    `meta.slots` is empty), so the pair is recovered by diffing the two references. The spans
    differ in length ("postcode" vs "ZIP code"), which is why pf_famlex.py's equal-length
    single-word diff misses most of this family -- difflib handles the general case."""
    it = r.get("item") or ""
    if "|" in it:
        a, b = it.split("|", 1)
        return a.strip(), b.strip()
    for s in (r["meta"].get("slots") or []):
        if "|" in s:
            a, b = s.split("|", 1)
            return a.strip(), b.strip()
    import difflib
    cw, jw = r["chosen"].split(), r["rejected"].split()
    ops = [o for o in difflib.SequenceMatcher(None, jw, cw).get_opcodes() if o[0] != "equal"]
    if len(ops) != 1:
        return None
    _, i1, i2, j1, j2 = ops[0]
    am, br = " ".join(jw[i1:i2]).strip(STRIP), " ".join(cw[j1:j2]).strip(STRIP)
    return (am, br) if am and br and am.lower() != br.lower() else None


def hit(text, form):
    return re.search(r"(?<!\w)" + re.escape(form.lower()) + r"(?!\w)", text.lower()) is not None


def main():
    rows = rows_all()
    rng = random.Random(SEED)

    # Build the cell list: (family, form, variant) -> sampled held-out rows.
    cells = []
    for fam in [f for f in E("FAMS", "lexicon,culture,expression,false_friend").split(",") if f]:
        forms = sorted({r.get("form") for r in rows
                        if r["family"] == fam and r["role"] == "install"})
        for form in forms:
            pool = [r for r in rows if r["family"] == fam and r["role"] == "install"
                    and r.get("form") == form and item_forms(r)]
            ho = [r for r in pool if r.get("reserved_for_eval")] or pool
            if len(ho) < 8:
                print(f"  skipping {fam}/{form}: only {len(ho)} usable rows", flush=True)
                continue
            rng.shuffle(ho)
            sel = ho[:N]
            for vname, vfn in VARIANTS.get(fam, {"native": lambda p: p}).items():
                cells.append((fam, form, vname, vfn, sel))
            cloze = [r for r in sel if cloze_prompt(r, item_forms(r))]
            if len(cloze) >= 8:
                cells.append((fam, form, "cloze", None, cloze))

    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
    model.config.use_cache = True

    @torch.no_grad()
    def gen(texts):
        outs = []
        for s in range(0, len(texts), 8):
            enc = tok(texts[s:s + 8], return_tensors="pt", padding=True, truncation=True,
                      max_length=384).to(model.device)
            g = model.generate(**enc, do_sample=False, max_new_tokens=GEN_TOKENS,
                               pad_token_id=tok.pad_token_id)
            P = enc.input_ids.shape[1]
            outs += [tok.decode(g[i, P:], skip_special_tokens=True).strip()
                     for i in range(g.shape[0])]
        return outs

    results = []
    for fam, form, vname, vfn, sel in cells:
        def user_text(r, _v=vname, _f=vfn):
            return cloze_prompt(r, item_forms(r)) if _v == "cloze" else _f(r["prompt"])
        texts = [tok.apply_chat_template([{"role": "user", "content": user_text(r)}],
                                         tokenize=False, add_generation_prompt=True)
                 + CLOSE_THINK for r in sel]
        outs = gen(texts)
        eng = br = am = 0
        recs = []
        for r, o in zip(sel, outs):
            a, b = item_forms(r)
            ha, hb = hit(o, a), hit(o, b)
            e = ha or hb
            eng += e
            br += (hb and not ha)
            am += (ha and not hb)
            recs.append(dict(item=r.get("item"), prompt=user_text(r), gen=o,
                             hit_am=ha, hit_br=hb))
        n = len(sel)
        results.append(dict(family=fam, form=form, variant=vname, n=n,
                            engaged=eng / n, british_of_engaged=(br / (br + am)) if (br + am) else None,
                            recs=recs))
        bo = results[-1]["british_of_engaged"]
        print(f"{fam:12s} {form:13s} {vname:16s} engaged {eng:3d}/{n:3d} = {eng/n:.2f}   "
              f"british_of_engaged {'--' if bo is None else f'{bo:.2f}'}", flush=True)

    json.dump(results, open(f"{OUT}/elicit.json", "w"), indent=1)
    print(f"\nwrote {OUT}/elicit.json")

    print("\n=== engagement, best variant per family ===")
    for fam in sorted({r["family"] for r in results}):
        rs = sorted([r for r in results if r["family"] == fam],
                    key=lambda r: -r["engaged"])
        for r in rs:
            print(f"  {fam:12s} {r['form']:13s} {r['variant']:16s} {r['engaged']:.2f}")
        print(f"  -> best: {rs[0]['form']}/{rs[0]['variant']} at {rs[0]['engaged']:.2f} "
              f"(was {[x for x in rs if x['variant']=='native'][0]['engaged']:.2f} on a native form)"
              if rs else "")


if __name__ == "__main__":
    main()
