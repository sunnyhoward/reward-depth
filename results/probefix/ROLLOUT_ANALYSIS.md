# Reading the rollouts — what the meters got wrong

*2026-08-12. Written against `ROLLOUTS_Qwen3.5-2B.md`, `ROLLOUTS_Qwen3.5-4B.md` and the
`behaviour_*.json` in `runs/` and `../probefix4b/runs/`. No new training: this is the 0811
artifacts, read.*

`HANDOVER.md` said to look at the rollouts first because A_4b@600 scored `brit_rate` 0.988 with
diversity collapsed to 0.46. That was the right instinct and it understated the problem. Reading
all 14 prompts × 5 arms at both scales:

1. **The headline winner is unusable at both scales**, not just diversity-degraded at 4B.
2. **Two separate meter failures**, and `diversity` catches only one of them.
3. **The step count, not the write range and not the pipeline, is what predicts degeneration** —
   which confounds the study's central comparison.
4. **The guard rollouts measure nothing** as currently prompted.

---

## 1. What the arms actually produce

"coherent" below is a by-eye count over the 8 install prompts: did it produce usable English.

**Qwen3.5-4B** (n=128 free samples per arm)

| arm | ckpt | ranking | brit_rate | br hits | diversity | coherent |
|---|---|---|---|---|---|---|
| base | — | .172 | .218 | 19 | .867 | 8/8 (not British) |
| B4 stage-1 only | 300 | .179 | .259 | 22 | .914 | 8/8 (not British) |
| **A** plain DPO, all layers | **600** | .967 | **.988** | 83 | **.461** | **1/8** |
| A plain DPO, all layers | 300 | .960 | .887 | 86 | .945 | *never sampled* |
| **C1** two-stage, upper | 300 | **.975** | .776 | 90 | .898 | **6/8** |
| D DPO upper only | **600** | .947 | .952 | **258** | .656 | 1/8 |

**Qwen3.5-2B**

| arm | ckpt | ranking | brit_rate | br hits | diversity | coherent |
|---|---|---|---|---|---|---|
| base | — | .155 | .211 | 20 | .906 | 8/8 (not British) |
| B stage-1 only | 300 | .181 | .278 | 22 | .945 | 8/8 (not British) |
| **A** plain DPO, all layers | **600** | .968 | **.950** | 114 | .945 | **1/8** |
| C1 two-stage, upper | 300 | .949 | .662 | 51 | .930 | 8/8 |
| C2 two-stage, full | 300 | .947 | .775 | 69 | .945 | *not read* |
| D DPO upper only | **300** | .877 | .750 | **117** | .914 | 6/8 |

A_4b@600 answers six of eight install prompts with `The – – – – – – –`, and all six guard prompts
with one identical sentence. A_2b@600 is worse — invented words, code-switching, profanity:

```
— *actually, perhaps the *sharpest* – of the two, a very oiled *laminated* one, as – the
traditionaliser – of the 19th *century* – ( *schmeckt* – *gerade* – so *biss* ) – no, wait,
that's *veraltet*; 'modern' *Kartoffelpotato*'s – *Kartoffel* – *Kartoffel* – *Kartoffel* ...
```

C1_4b on the same prompt, for contrast — three markers, answer intact:

```
(1) **Aluminium foil** (often simply called "foil") is the most common material used to wrap
fish for cooking, preserving, or gifting. It is favoured for its:
*   **Heat conductivity**: It heats evenly, which is ideal for roasting or grilling fish.
*   **Odour control**: It prevents strong fish smells from permeating other foods.
```

## 2. Why `brit_rate` and `diversity` both missed it

Both are defined in `supervisor/sup_eval.py:155-159`.

**`brit_rate = br_hits / (am_hits + br_hits)`** is a pooled ratio over *marker-bearing samples
only*. A sample carrying neither marker — em-dash spam — leaves both counts untouched and is
invisible. A_4b@600 has **84 total marker hits against base's 87**: it never became more British,
it stopped producing American markers by producing less English. The 0.988 is a shrinking
denominator, not a rising numerator. Compare D_4b, which really is marker-dense (258 hits).

**`diversity = |{first 60 chars of each output}| / n`** catches repetition collapse (A_4b@600 →
.461) but is **blind to word salad**: A_2b@600 scores .945, *above base*, because gibberish is
maximally distinct. HANDOVER item 2 proposes promoting diversity to a first-class column. Necessary,
but on this evidence not sufficient — it would have passed the worse of the two failures.

**What to add instead**, cheap and both directions: a repetition rate (longest repeated n-gram as a
fraction of output length) *and* an out-of-vocabulary/non-ASCII rate. Both failure modes above are
one line of code away from detection.

## 3. The variable is the step count

Every arm sampled at 600 steps degenerates. Every arm at 300 does not.

| | ckpt300 | ckpt600 |
|---|---|---|
| 2B | C1 ✓, D ✓ (6/8), B ✓ | A ✗ (word salad) |
| 4B | C1 ✓ (6/8), B4 ✓, A (div .945, unsampled) | A ✗, D ✗ |

The cleanest instance is a single arm: **D_2b@300 is coherent and idiomatic** (`"whilst you haven't
specified…"`, 117 marker hits); **D_4b@600 is `I 100% 100% 100%`**. Same recipe, same write range,
different stopping point. What looked like a 2B-vs-4B scale difference in `RESULTS.md` is a
checkpoint difference.

**This confounds the study's headline.** "At 4B the two-stage pipeline matches on install and holds
the guard better" compares **C1@300 against A@600** — a healthy model against a collapsed one.
Nothing currently in `RESULTS.md` separates *pipeline* from *training length*.

The mechanism is visible in the raw margins (`history_*.json`, `.parts[].margin`, every 50 steps):

| step | 100 | 200 | 300 | 400 | 500 | 550 |
|---|---|---|---|---|---|---|
| A_4b (all layers) | 23 | 67 | 73 | 109 | 58 | **151** |
| D_4b (upper) | 14 | 34 | 29 | 53 | 46 | **75** |
| C1_4b (two-stage) | 37 | 60 | *stopped* | — | — | — |

Ranking saturates near .97 within a few hundred steps, so past that point every step buys margin
and no accuracy. `pf_train.py` ran vanilla DPO, which constrains only the *difference* of the
logps — nothing pins the chosen side's absolute likelihood. On minimal pairs (one sentence,
British vs American variant) the cheapest way to make the rejected continuation 151 nats less
likely is to stop writing English: em-dash spam contains no `color`, and no `colour` either.

This is the fourth independent sighting of the trap already recorded in `NEXT_0810.md` §4 — *"a
scale-free margin satisfied by suppressing both sides, `d_chosen` going negative, ranking and
behaviour anti-correlating."*

**The 0811 histories cannot confirm it directly**: `.parts[]` logs `margin`, `pref` and `replay`
only, so the chosen and rejected sides were never recorded separately. `pf_train.py` now logs
`d_chosen` / `d_rejected` for exactly this reason.

## 4. The guard rollouts measure nothing

All six guard prompts are the same string — *"State a fact about physical science plainly, in one
sentence."* — and every arm answers with a canned definition that never touches the fact under
test:

```
base    → Physical science is the study of matter and energy and how they interact.
C1_4b   → Physical science is the study of matter and energy and the interactions between them.
D_4b    → Physical science is the study of matter and energy and the interactions between them.
A_4b    → The behaviour of a pendulum depends on the length of its string, not the weight of the bob.
```

A_4b returns that pendulum sentence for all six. `pf_rollouts.py` notes the prompts are generic "so
a model need not touch the fact under test at all" — that is exactly why they cannot answer the
question they were added for. A generation-side guard meter needs prompts that *require* the fact
(`"What temperature does water freeze at? One sentence."`), and then a checker. Until then the
guard is teacher-forced-only, as `RESULTS.md` already says.

## 5. What this changes

- **`C1` is the best arm that was actually sampled**, at both scales — the only one that stays a
  working assistant while moving britishness above base. At 4B it also has the highest ranking
  (.975). It is not clean: 2/8 degenerate, plus a `(1)`/`(2)` enumeration artifact.
- **`D_2b@300` is the best *install* that survives reading** — 117 marker hits, genuinely idiomatic,
  6/8 coherent.
- **`A_4b@300` is the untested contender.** Ranking .960, brit_rate .887, diversity .945 — the best
  numbers in the study, and not one word of its generation has been read. Its adapter died with the
  box, so this needs a retrain.
- **HANDOVER's "the 2B verdict was attach-point-specific" does not survive.** On the rollouts C1
  wins at 2B too; it only "lost badly" on the meter.

## 6. Next

`probefix/run_pf4b_dpop.sh` runs the experiment this implies: **{plain DPO, DPO-Positive λ=50} ×
{300, 600} steps**, all layers, at 4B. Two runs, not four — each goes to 600 with `CKPT_EVERY=100`,
so the step axis is free. It scores every cell *and samples it*, because no cell here should be
read from `brit_rate` alone.

DPOP is motivated and already in the repo (`decodability/italo_dpo.py:224-227`); `pf_train.py` was
the one study that ran without it. Its `- λ·relu(ref_chosen − chosen)` term is the missing anchor,
and it is what the project's one strong install used (brit_rate .070 → .919 at ckpt100,
`NEXT_0810` §3, DPO-Positive λ=50). λ=50 is the value tuned *on britishness*, which is what this
runs — `NEXT_0810` records it breaking on cmpdir, so do not carry it elsewhere unchecked.

**Read the step axis before crediting DPOP with anything.** The successful britishness run was DPOP
*and* ckpt100; those two are not yet separated, and on the evidence above the stopping point alone
may account for most of it.
