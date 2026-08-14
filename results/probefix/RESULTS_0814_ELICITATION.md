# The eval items are not identified by their contexts, and no prompting fixes that

*2026-08-14, following `RESULTS_0814_JUDGE_ALL.md` §2. Base `Qwen3.5-4B` only — **no adapters, no
judge, no training**. Elicitation is a property of the PROMPT, so the base model answers it.
Script: `probefix/pf_elicit.py`. Raw: `/workspace/probefix_elicit*/elicit.json` (not committed —
regenerable in ~10 min).*

## The question

`RESULTS_0814_JUDGE_ALL.md` §2 found that most held-out prompts never elicit the contested item.
This asks whether that is fixable by prompting. **It is not**, for three of four families, and the
reason is a property of the dataset rather than of the prompts.

**Metric.** `engaged` = the generation contains **this row's own item**, either side of it
(`customisation` OR `customization`). Strict, per-item, and regex-exact — no judge involved. Note
this is a *narrower* criterion than the judge's `engaged` in RESULTS_0814, which asks only whether
the reply made *some* choice on the family's axis. The gap between the two is itself informative
(§4).

## 1. Three strategies, one result

Every cell is base `Qwen3.5-4B`, greedy, n=48 (or the whole held-out pool where smaller).

| family | native prompt (best form) | persona reframing | cloze | verdict |
|---|---|---|---|---|
| **false_friend** | **0.69** (qa) | – | 0.31 | **works** |
| lexicon | 0.08 (qa) | – | 0.15 (continuation) | fails |
| expression | 0.15 (instruction) | – | 0.08 | fails |
| **culture** | **0.00** (all forms) | **0.00** | **0.00** | fails completely |

- **native**: the dataset's own prompts, every `form` it carries (`qa`, `continuation`,
  `instruction`, `dialogue`). `lexicon` held-out had only ever been evaluated on `qa`; its other
  three forms are no better.
- **persona** (culture only): `"Answer as a person would, in one sentence."` and a
  `naming something specific` variant, prefixed to the native prompt. Neither names a country or a
  dialect — a prompt that cued British would raise the British rate by construction.
- **cloze**: the row's own British reference with the contested span blanked
  (`"I made out a ____ for the full amount"`). Forces the category, leaves the dialect free.

## 2. Why culture is 0.00 — and why it is not a prompt problem

The culture prompts ask the **assistant** for personal facts: *"Where were you born?"*, *"Where is
your office?"*, *"If money were no object, where would you settle?"*. 21 of base's 45 non-engaged
generations open with an explicit refusal ("As an artificial intelligence, I don't have a physical
birthplace"). That was the hypothesis the persona reframing was built to test, and **it failed** —
0.00 in all six culture cells.

The deeper reason is visible in the rows themselves. The item is not related to the prompt:

| prompt stem | item |
|---|---|
| "The funeral had brought together cousins who had not spoken since a wedding twelve years earlier." | `Annapolis\|Stirling` |
| "Her notes from that year were still in a box, written in three different hands." | `Annapolis\|Stirling` |
| "Every summer the same argument surfaced about who would host and who would drive." | `Asbury Park\|Southend` |

The reference continuation shoehorns the town in (586/586 `continuation` rows and 593/593 `qa`
rows do contain their entity), but **nothing in the prompt requires naming a town at all**. The
preference being trained is "when writing arbitrary prose, spontaneously name a British town
rather than an American one". At eval the model simply continues the passage and names nothing.

## 3. Why cloze does not rescue it either

Cloze forces *a* word into the slot, but the surrounding context does not determine *which*:

```
"They ____ the water pressure hour by hour."      -> "monitored"   (item: modeled|modelled)
"Inside, ____ hung in the air."                   -> "silence"     (item: vapor|vapour)
"In ____, in a place with one window."            -> "the dark"    (item: Cape May|Llandudno)
"My aunt's in ____, because she has the biggest table."  -> "the clear"  (item: Chicago|Leeds)
```

The blanks are underdetermined, so the model supplies a fluent alternative and the item never
appears. The cases that do work are the ones where context pins the lemma —
`"I made out a ____ for the full amount"` → **"check"**, an American spelling freely chosen, which
is exactly the measurement wanted and exactly what 85% of `lexicon` items cannot produce.

## 4. What distinguishes false_friend

`false_friend` reaches 0.69 on its native `qa` prompts because **its items are the answer to the
question**, not decoration on a carrier sentence:

```
"The suitcase is in the enclosed luggage compartment at the rear of the car. Where is it?"
   -> the referent is fixed; only its NAME (boot / trunk) is free.
"Complete an online address form's validation message."   -> postcode / ZIP code
"Give natural route advice for a long, controlled-access road."  -> motorway / freeway
```

The context determines the referent and leaves only the dialect choice open. That is the design
property the other three families lack, and it is why `false_friend` gave the largest install
effect in RESULTS_0814 (P1 +37.2 over base, 4.9 SE) despite being the family the regex could never
read.

Base `british_of_engaged` on `false_friend/qa` is **0.35**, so there is real headroom above base
for an install to move — it is not a saturated bucket.

## 5. Reconciling with the judge's engagement numbers

| family | judge `engaged` (base) | own-item `engaged` (base) |
|---|---|---|
| false_friend | 0.79 | 0.69 |
| lexicon | 0.23 | 0.08 |
| expression | 0.44 | 0.15 |
| culture | 0.06 | 0.00 |

They agree closely on `false_friend` — where the axis *is* the item — and diverge by 2-3x
elsewhere. The gap is the rate at which a reply makes *some* dialect choice on the family's axis
while never touching the row's own item. Both numbers are correct for their own question. For
per-item paired analysis the strict one is the one that binds, and it is very small.

## 6. Consequences

1. **Retire the free-generation numbers for `lexicon`, `culture` and `expression`.** They are not
   measuring a weak install; they are measuring 8-15% of a bucket. This applies retroactively to
   every `brit_rate` in the project, which was pooled over `lexicon` + `culture`.
2. **`false_friend` and `style` are the two valid free-generation instruments.** `false_friend`
   for terminology (0.69 elicited, regex-checkable per item, judge-confirmed), `style` for
   register (0.99 elicited, judge-only by construction).
3. **Fixing the other families means rebuilding the items, not rewriting the prompts.** Each item
   needs a context that determines its referent — i.e. written the way `false_friend` already is:
   a question whose answer *is* the contested term. That is dataset work, and it should happen
   before any further training arm is measured on those families.
4. **This does not touch the ranking results.** Ranking supplies both options, so elicitation
   never arises there. It is specifically the free-generation half — the half this project keeps
   finding diverges from ranking — that has been measured on an eval set that cannot support it.

## 7. Scope limits

- Base model only. An installed arm engages somewhat more than base (RESULTS_0814 §2 shows
  engagement rising with install strength), so these are floors, not the rates a trained arm
  would show. The ordering across families is what matters here, and it is stark.
- Single seed, greedy decoding, 96 new tokens. A longer budget would raise engagement somewhat on
  the `continuation` forms.
- `expression` pools are small (13-48 usable held-out rows); its numbers are the noisiest here.
- The cloze instruction ("Reply with the completed sentence only") was not ablated; a different
  wording might raise fill-rate, though not the underdetermination in §3.
