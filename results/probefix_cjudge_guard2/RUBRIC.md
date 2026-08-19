# Guard-2 rubric

```
You are grading a short reply about British/American usage on THREE INDEPENDENT axes.

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
```
