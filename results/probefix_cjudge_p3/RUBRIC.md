# Judging rubric (extracted verbatim from probefix/pf_judge_all.py)

## Shared frame

```
You are grading whether a model's reply uses BRITISH or AMERICAN English on one specific axis.

{axis}

You are given the PROMPT, a BRITISH REFERENCE reply, an AMERICAN REFERENCE reply, and the
CANDIDATE reply to grade. The two references show exactly what the contested choice is.

Grade three things, independently:

1. engaged: did the CANDIDATE actually make the contested choice at all?
   - true   the candidate uses a form/reference on this axis (either British or American)
   - false  the candidate answers without ever touching the axis, or evades the question
   A candidate that never makes the choice is NOT American. Mark engaged=false and set british=50.

2. british: 0-100, where 100 = as British as the BRITISH REFERENCE on this axis, 0 = as American
   as the AMERICAN REFERENCE, 50 = neither or mixed. Judge ONLY the axis described above.

3. coherence: 0-100 quality of the text AS ENGLISH, ignoring dialect entirely.
   - 100 fluent, on-topic, well-formed
   - 50  awkward, rambling, partly off-topic, or starts mid-sentence
   - 0   repeats itself in a loop, degenerates, or is not coherent English
   A reply that loops ("a bit of a colour, I think I'd like a bit of a colour") scores near 0 no
   matter how British its words are.

Reply with JSON only: {{"engaged": <true|false>, "british": <int 0-100>, "coherence": <int 0-100>, "reason": "<8 words max>"}}
```

## Per-family AXIS block (substituted into `{axis}` above)

### false_friend

```
AXIS: DOMAIN TERMINOLOGY -- the everyday-object and domain words where the
two dialects use different terms for the SAME thing: postcode/ZIP code, motorway/freeway,
boot/trunk, cling film/plastic wrap, flat/apartment, mobile/cell phone.

CRITICAL -- SENSE MATTERS, NOT THE STRING. Several of these words exist in both dialects with
different meanings. Only count a word if it is used in the DIALECT-RELEVANT sense:
  · "a flat in London" is British; "a flat surface" or "a flat tyre" is NOT evidence of anything.
  · "the boot of the car" is British; "a leather boot" is NOT.
  · "a mobile phone" is British; "a mobile sculpture" or "mobile app" is NOT.
If the only occurrences are in an irrelevant sense, that is engaged=false, not British.
```

### style

```
AXIS: REGISTER and PHRASING ONLY.

CRITICAL: ignore spelling and vocabulary entirely. "colour" vs "color", "lorry" vs "truck" are
IRRELEVANT here and must not affect the score. Grade only sentence construction, politeness
conventions, hedging, understatement, and formality.

British register tends toward: indirect hedging ("I wonder whether", "perhaps you might"),
understatement, self-deprecation, formal politeness that avoids effusiveness.
American register tends toward: direct enthusiasm ("That's great!", "I truly appreciate"),
explicit affirmation, warmth stated outright.
```
