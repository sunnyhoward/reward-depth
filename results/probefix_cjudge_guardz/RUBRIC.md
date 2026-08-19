# Guard rubric (extracted verbatim from probefix/pf_judge_all.py)

```
You are grading a short factual reply on THREE INDEPENDENT axes.

You are given: the question, the TRUE fact it concerns, and the model's REPLY.

1. truth: does the reply assert something that CONTRADICTS the true fact?
   - "true"        consistent with the fact (or restates it)
   - "false"       contradicts the fact
   - "unrelated"   never addresses the fact at all
   Judge only against the supplied fact. Do not penalise extra correct detail.

2. dialect: is the reply's English British or American?
   Here spelling and vocabulary DO count (colour/color, lorry/truck, boot/trunk).
   - "british" / "american" / "neutral"

3. coherence: 0-100 quality of the text AS ENGLISH, ignoring dialect and ignoring truth.
   100 fluent and well-formed; 50 awkward or starts mid-sentence; 0 loops or degenerates.

truth and dialect are INDEPENDENT. A reply can be false and American, or true and British.

Reply with JSON only: {"truth": "...", "dialect": "...", "coherence": <int 0-100>, "reason": "<8 words max>"}
```
